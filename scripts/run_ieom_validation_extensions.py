"""Generate IEOM-targeted validation artifacts from existing experiment outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_thesis_quantification import (  # noqa: E402
    contains_explicit_accuracy_first,
    infer_deployable,
    iter_groups,
    load_method_tables,
    sort_accuracy_first,
)

OUTPUT_TABLE_DIR = REPO_ROOT / "outputs" / "tables"
IEOM_TABLE_DIR = REPO_ROOT / "paper_submission_ieom" / "tables"
IEOM_FIGURE_DIR = REPO_ROOT / "paper_submission_ieom" / "figures"

DEFAULT_LAMBDA_VOLATILITY = 0.10
DEFAULT_LAMBDA_EXECUTION = 0.10
DEFAULT_LAMBDA_SWITCH = 0.05


def main() -> None:
    """Build all IEOM validation outputs."""
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    IEOM_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    IEOM_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    statistical_summary = build_statistical_validation()
    write_csv_and_tex(
        statistical_summary,
        OUTPUT_TABLE_DIR / "statistical_validation_summary.csv",
        IEOM_TABLE_DIR / "statistical_validation_summary_table.tex",
        caption="Descriptive validation of the planning-execution gap.",
        label="tab:statistical-validation",
        column_spec="@{}p{0.30\\linewidth}p{0.17\\linewidth}p{0.22\\linewidth}p{0.21\\linewidth}@{}",
    )

    sensitivity_detail, sensitivity_summary = build_objective_component_sensitivity()
    sensitivity_detail.to_csv(OUTPUT_TABLE_DIR / "objective_component_sensitivity.csv", index=False)
    write_csv_and_tex(
        sensitivity_summary,
        OUTPUT_TABLE_DIR / "objective_component_sensitivity_summary.csv",
        IEOM_TABLE_DIR / "objective_component_sensitivity_table.tex",
        caption="Objective-component sensitivity analysis using deployable strategy selections.",
        label="tab:objective-sensitivity",
        column_spec="@{}p{0.24\\linewidth}p{0.20\\linewidth}p{0.12\\linewidth}p{0.12\\linewidth}p{0.13\\linewidth}p{0.09\\linewidth}@{}",
    )
    plot_objective_component_sensitivity(sensitivity_detail)
    plot_favorita_accuracy_execution_frontier()

    regime_map = build_operating_regime_map()
    regime_map.to_csv(OUTPUT_TABLE_DIR / "operating_regime_strategy_map.csv", index=False)
    plot_operating_regime_map(regime_map)


def build_statistical_validation() -> pd.DataFrame:
    """Compute bootstrap and sign-test validation for deployable evaluation groups."""
    method_tables = load_method_tables(OUTPUT_TABLE_DIR)
    records: List[Dict[str, object]] = []
    for source_table, table in method_tables:
        for group_key, group in iter_groups(table):
            deployable = group[group["deployable_inferred"]].copy()
            if len(deployable) < 2 or not contains_explicit_accuracy_first(deployable):
                continue
            accuracy_first = sort_accuracy_first(deployable).iloc[0]
            operational_best = deployable.sort_values(
                ["normalized_total_loss", "method_name", "strategy"]
            ).iloc[0]
            records.append(
                {
                    "source_table": source_table,
                    "dataset_name": str(group["dataset_name"].dropna().iloc[0])
                    if "dataset_name" in group.columns and group["dataset_name"].notna().any()
                    else "unknown",
                    "group_key": group_key,
                    "accuracy_first_strategy": accuracy_first["method_name"],
                    "operational_best_strategy": operational_best["method_name"],
                    "best_deployable_strategy_family": strategy_family(operational_best),
                    "mismatch": str(accuracy_first["strategy"]) != str(operational_best["strategy"]),
                    "accuracy_first_peg_rate": float(accuracy_first["execution_violation_rate"]),
                    "operational_best_peg_rate": float(operational_best["execution_violation_rate"]),
                    "peg_rate_difference": float(
                        accuracy_first["execution_violation_rate"]
                        - operational_best["execution_violation_rate"]
                    ),
                    "normalized_loss_difference": float(
                        accuracy_first["normalized_total_loss"]
                        - operational_best["normalized_total_loss"]
                    ),
                }
            )
    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError("No deployable evaluation groups were available for descriptive validation.")

    rng = np.random.default_rng(20260702)
    mismatch = detail["mismatch"].astype(float).to_numpy()
    peg_diff = detail["peg_rate_difference"].to_numpy(dtype=float)

    sign_test = exact_sign_test(peg_diff)
    rows = [
        {
            "Metric": "Evaluation groups",
            "Estimate": str(len(detail)),
            "95% CI": "not applicable",
            "Validation note": "Deployable strategies only; oracle references excluded",
        },
        {
            "Metric": "Accuracy-first mismatch rate",
            "Estimate": format_percent(float(mismatch.mean())),
            "95% CI": format_ci(bootstrap_ci(mismatch, np.mean, rng)),
            "Validation note": "WAPE-best differs from operational-loss-best",
        },
        {
            "Metric": "Mean PEG-rate reduction",
            "Estimate": format_percentage_points(float(np.mean(peg_diff))),
            "95% CI": format_pp_ci(bootstrap_ci(peg_diff, np.mean, rng)),
            "Validation note": "Accuracy-first PEG minus operational-loss-best PEG",
        },
        {
            "Metric": "Median PEG-rate reduction",
            "Estimate": format_percentage_points(float(np.median(peg_diff))),
            "95% CI": format_pp_ci(bootstrap_ci(peg_diff, np.median, rng)),
            "Validation note": "Robust central tendency across groups",
        },
        {
            "Metric": "Sign test for PEG reduction",
            "Estimate": "{} positive / {} nonzero".format(sign_test["positive"], sign_test["nonzero"]),
            "95% CI": format_p_value(sign_test["p_value"]),
            "Validation note": "Two-sided exact sign test",
        },
    ]
    return pd.DataFrame(rows)


def build_objective_component_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select deployable methods under objective variants and summarize outcomes."""
    frames = load_component_tables()
    detail_records: List[Dict[str, object]] = []
    objective_variants = [
        ("accuracy_only", "WAPE"),
        ("inventory_only", "inventory_score"),
        ("inventory_volatility", "inventory_volatility_score"),
        ("inventory_execution", "inventory_execution_score"),
        ("inventory_switching", "inventory_switching_score"),
        ("full_feasibility", "full_feasibility_score"),
    ]
    for source_name, table in frames:
        group_columns = comparison_group_columns(table)
        for group_key_values, group in table.groupby(group_columns, dropna=False, sort=False):
            if not isinstance(group_key_values, tuple):
                group_key_values = (group_key_values,)
            deployable = group[group["deployable_inferred"]].copy()
            if len(deployable) < 2:
                continue
            deployable = add_objective_scores(deployable)
            deployable["_full_rank"] = deployable["full_feasibility_score"].rank(
                method="min", ascending=True
            )
            group_context = dict(zip(group_columns, group_key_values))
            full_winner = deployable.sort_values(["full_feasibility_score", "method_name"]).iloc[0]
            for objective_name, score_column in objective_variants:
                selected = deployable.sort_values([score_column, "method_name", "strategy"]).iloc[0]
                detail_records.append(
                    {
                        "source_table": source_name,
                        **group_context,
                        "objective_variant": objective_name,
                        "selected_strategy": selected["method_name"],
                        "selected_strategy_family": strategy_family(selected),
                        "full_feasibility_strategy": full_winner["method_name"],
                        "WAPE": float(selected["WAPE"]),
                        "PEG_rate": float(selected["execution_violation_rate"]),
                        "execution_penalty": float(selected.get("execution_penalty", np.nan)),
                        "planning_volatility": float(selected.get("planning_volatility", np.nan)),
                        "model_switch_count": float(selected.get("model_switch_count", np.nan)),
                        "normalized_planning_loss_under_full_objective": float(
                            selected["full_feasibility_score"]
                        ),
                        "rank_difference_vs_full_feasibility_selection": float(selected["_full_rank"] - 1.0),
                    }
                )
    detail = pd.DataFrame(detail_records)
    if detail.empty:
        raise RuntimeError("No component-level tables were available for objective sensitivity analysis.")

    summary_records = []
    for objective, group in detail.groupby("objective_variant", sort=False):
        summary_records.append(
            {
                "Objective": display_objective(objective),
                "Most frequent selected family": most_common(group["selected_strategy_family"]),
                "Mean WAPE": format_percent(float(group["WAPE"].mean())),
                "Mean PEG": format_percent(float(group["PEG_rate"].mean())),
                "Mean full loss": "{:.3f}".format(
                    float(group["normalized_planning_loss_under_full_objective"].mean())
                ),
                "Rank gap": "{:.2f}".format(
                    float(group["rank_difference_vs_full_feasibility_selection"].mean())
                ),
            }
        )
    return detail, pd.DataFrame(summary_records)


def build_operating_regime_map() -> pd.DataFrame:
    """Recompute Favorita baseline strategy winners over feasibility-weight regimes."""
    path = OUTPUT_TABLE_DIR / "favorita_improved_feasibility_methods.csv"
    table = pd.read_csv(path)
    table = table[table["scenario_name"].astype(str) == "baseline"].copy()
    table["deployable_inferred"] = infer_deployable(table)
    deployable = table[table["deployable_inferred"]].copy()
    deployable = add_unweighted_component_estimates(deployable)

    lambda_execution_grid = [0.00, 0.025, 0.05, 0.10, 0.20, 0.50, 1.00]
    lambda_volatility_grid = [0.00, 0.025, 0.05, 0.10, 0.20, 0.50, 1.00]
    rows: List[Dict[str, object]] = []
    for lambda_execution in lambda_execution_grid:
        for lambda_volatility in lambda_volatility_grid:
            scored = deployable.copy()
            scored["regime_loss"] = (
                scored["unweighted_inventory_component"]
                + lambda_volatility * scored["unweighted_volatility_component"]
                + lambda_execution * scored["unweighted_execution_component"]
                + DEFAULT_LAMBDA_SWITCH * scored["unweighted_switch_component"]
            )
            ranked = scored.sort_values(["regime_loss", "method_name", "strategy"])
            winner = ranked.iloc[0]
            runner_up = ranked.iloc[1] if len(ranked) > 1 else winner
            rows.append(
                {
                    "lambda_execution": lambda_execution,
                    "lambda_volatility": lambda_volatility,
                    "lambda_switch": DEFAULT_LAMBDA_SWITCH,
                    "winning_strategy_family": strategy_family(winner),
                    "winning_strategy": winner["method_name"],
                    "second_best_strategy": runner_up["method_name"],
                    "winner_normalized_loss": float(winner["regime_loss"]),
                    "runner_up_normalized_loss": float(runner_up["regime_loss"]),
                    "winner_runner_up_loss_gap": float(runner_up["regime_loss"] - winner["regime_loss"]),
                }
            )
    return pd.DataFrame(rows)


def load_component_tables() -> List[tuple[str, pd.DataFrame]]:
    """Load current component-level paper-facing tables."""
    paths = [
        OUTPUT_TABLE_DIR / "favorita_improved_feasibility_methods.csv",
        OUTPUT_TABLE_DIR / "m5_robustness_summary.csv",
        OUTPUT_TABLE_DIR / "walmart_robustness_summary.csv",
    ]
    frames: List[tuple[str, pd.DataFrame]] = []
    required = {
        "WAPE",
        "execution_violation_rate",
        "normalized_inventory_component",
        "normalized_volatility_component",
        "normalized_execution_component",
        "normalized_switch_component",
        "normalized_total_loss",
    }
    for path in paths:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if required.difference(table.columns):
            continue
        if "dataset_name" not in table.columns:
            table["dataset_name"] = infer_dataset_from_filename(path.stem)
        if "method_name" not in table.columns and "strategy" in table.columns:
            table["method_name"] = table["strategy"].astype(str)
        if "strategy" not in table.columns:
            table["strategy"] = table["method_name"].astype(str)
        table["source_table"] = path.stem
        table["deployable_inferred"] = infer_deployable(table)
        for column in required | {"execution_penalty", "planning_volatility", "model_switch_count"}:
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce")
        frames.append((path.stem, table))
    return frames


def comparison_group_columns(table: pd.DataFrame) -> List[str]:
    """Return available grouping columns for comparable objective selections."""
    candidates = [
        "dataset_name",
        "run_mode",
        "scenario_name",
        "grain_level",
        "intermittency_bucket",
        "feature_set",
        "experiment_name",
        "window_type",
        "module_name",
    ]
    return [column for column in candidates if column in table.columns]


def add_objective_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Add objective-variant scores using current normalized component contributions."""
    result = table.copy()
    result["inventory_score"] = result["normalized_inventory_component"]
    result["inventory_volatility_score"] = (
        result["normalized_inventory_component"] + result["normalized_volatility_component"]
    )
    result["inventory_execution_score"] = (
        result["normalized_inventory_component"] + result["normalized_execution_component"]
    )
    result["inventory_switching_score"] = (
        result["normalized_inventory_component"] + result["normalized_switch_component"]
    )
    result["full_feasibility_score"] = result["normalized_total_loss"]
    return result


def add_unweighted_component_estimates(table: pd.DataFrame) -> pd.DataFrame:
    """Estimate unweighted normalized components from stored component contributions."""
    result = table.copy()
    base_execution = pd.to_numeric(result.get("lambda_execution", DEFAULT_LAMBDA_EXECUTION), errors="coerce")
    base_execution = base_execution.replace(0, DEFAULT_LAMBDA_EXECUTION).fillna(DEFAULT_LAMBDA_EXECUTION)
    result["unweighted_inventory_component"] = result["normalized_inventory_component"]
    result["unweighted_volatility_component"] = (
        result["normalized_volatility_component"] / DEFAULT_LAMBDA_VOLATILITY
    )
    result["unweighted_execution_component"] = result["normalized_execution_component"] / base_execution
    result["unweighted_switch_component"] = result["normalized_switch_component"] / DEFAULT_LAMBDA_SWITCH
    return result


def plot_objective_component_sensitivity(detail: pd.DataFrame) -> None:
    """Plot full-objective loss of strategies selected by each objective component."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = [
        "accuracy_only",
        "inventory_only",
        "inventory_volatility",
        "inventory_execution",
        "inventory_switching",
        "full_feasibility",
    ]
    labels = [display_objective(value) for value in order]
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_data = detail.copy()
    plot_data["objective_variant"] = pd.Categorical(plot_data["objective_variant"], order, ordered=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    sns.pointplot(
        data=plot_data,
        x="objective_variant",
        y="normalized_planning_loss_under_full_objective",
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color="#2C7FB8",
        markers="o",
        errwidth=1.1,
        capsize=0.12,
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_xlabel("Selection Objective")
    ax.set_ylabel("Full Feasibility Loss")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(IEOM_FIGURE_DIR / "objective_component_sensitivity.pdf")
    fig.savefig(IEOM_FIGURE_DIR / "objective_component_sensitivity.png", dpi=300)
    plt.close(fig)


def plot_operating_regime_map(regime_map: pd.DataFrame) -> None:
    """Plot a categorical strategy-family map across execution and volatility weights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    family_order = [
        "Inventory-first",
        "Ensemble",
        "Smoothing",
        "Stability-first",
        "Accuracy-first",
        "Rule-based feasibility",
    ]
    colors = {
        "Inventory-first": "#86DBD4",
        "Ensemble": "#95C9E1",
        "Smoothing": "#F4E0E1",
        "Stability-first": "#F6A9BD",
        "Accuracy-first": "#E15759",
        "Rule-based feasibility": "#79706E",
    }
    data = regime_map.copy()
    data["family_code"] = data["winning_strategy_family"].map(
        {family: idx for idx, family in enumerate(family_order)}
    )
    pivot = data.pivot(index="lambda_execution", columns="lambda_volatility", values="family_code")
    pivot = pivot.sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(4.9, 2.55))
    cmap = ListedColormap([colors[family] for family in family_order])
    ax.imshow(pivot.values, cmap=cmap, vmin=-0.5, vmax=len(family_order) - 0.5, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["{:.3g}".format(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["{:.3g}".format(value) for value in pivot.index])
    ax.set_xlabel(r"$\lambda$ volatility")
    ax.set_ylabel(r"$\lambda$ execution")
    ax.tick_params(axis="both", labelsize=8.2)
    ax.xaxis.label.set_size(9.2)
    ax.yaxis.label.set_size(9.2)
    for spine in ax.spines.values():
        spine.set_color("#9A9A9A")
        spine.set_linewidth(0.6)
    ax.tick_params(axis="both", color="#9A9A9A", width=0.6)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    used_families = [family for family in family_order if family in set(data["winning_strategy_family"])]
    legend_handles = [Patch(facecolor=colors[family], label=family) for family in used_families]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9.4,
        handlelength=1.05,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    fig.tight_layout()
    fig.savefig(IEOM_FIGURE_DIR / "operating_regime_strategy_map.pdf", bbox_inches="tight")
    fig.savefig(IEOM_FIGURE_DIR / "operating_regime_strategy_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_favorita_accuracy_execution_frontier() -> None:
    """Save the IEOM-facing Favorita accuracy-execution frontier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    source_path = OUTPUT_TABLE_DIR / "favorita_method_family_summary.csv"
    if not source_path.exists():
        source_path = REPO_ROOT / "paper" / "tables" / "favorita_method_family_summary_table.csv"
    table = pd.read_csv(source_path)
    baseline = table[table["scenario_name"].astype(str) == "baseline"].copy()
    baseline["wape_pct"] = pd.to_numeric(baseline["WAPE"], errors="coerce") * 100.0
    baseline["execution_penalty_thousands"] = (
        pd.to_numeric(baseline["execution_penalty"], errors="coerce") / 1000.0
    )

    point_styles = {
        "Oracle": {"color": "#7A7A7A", "marker": "D", "size": 66},
        "Global Best": {"color": "#E69F00", "marker": "o", "size": 70},
        "Family Best": {"color": "#E69F00", "marker": "s", "size": 60},
        "Feasibility-Aware": {"color": "#333333", "marker": "h", "size": 58},
        "Operational Ensemble": {"color": "#0072B2", "marker": "v", "size": 70},
        "Simple Ensemble": {"color": "#009E73", "marker": "^", "size": 70},
        "Smoothed Alpha 0.25": {"color": "#6A5ACD", "marker": "P", "size": 68},
        "Utility Alpha": {"color": "#4C78A8", "marker": "*", "size": 92},
        "Best Stability": {"color": "#CC79A7", "marker": "x", "size": 74},
    }
    label_offsets = {
        "Oracle": (0.82, 615.0),
        "Global Best": (12.55, 565.0),
        "Family Best": (13.05, 500.0),
        "Feasibility-Aware": (13.55, 380.0),
        "Operational Ensemble": (13.80, 138.0),
        "Simple Ensemble": (13.30, 48.0),
        "Smoothed Alpha 0.25": (3.15, 64.0),
        "Utility Alpha": (6.10, 18.0),
        "Best Stability": (16.85, 82.0),
    }
    label_text = {
        "Smoothed Alpha 0.25": "Smoothed 0.25",
    }

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.grid(True, color="#E5E5E5", linewidth=0.75)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#B5B5B5")

    for method_name, style in point_styles.items():
        row = baseline[baseline["method_name"].astype(str) == method_name]
        if row.empty:
            continue
        record = row.iloc[0]
        x = float(record["wape_pct"])
        y = float(record["execution_penalty_thousands"])
        scatter_kwargs = {
            "s": style["size"],
            "marker": style["marker"],
            "color": style["color"],
            "linewidth": 0.7,
            "zorder": 3,
        }
        if style["marker"] != "x":
            scatter_kwargs["edgecolor"] = "white"
        ax.scatter(x, y, **scatter_kwargs)
        text_x, text_y = label_offsets[method_name]
        if method_name == "Oracle":
            label = "Perfect-demand ref.\nnon-deployable diagnostic reference"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(text_x, text_y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=9.4,
                fontstyle="italic",
                color="#4A4A4A",
                arrowprops=dict(arrowstyle="-", color="#8A8A8A", lw=0.75, shrinkA=3, shrinkB=4),
            )
            continue
        ax.annotate(
            label_text.get(method_name, method_name),
            xy=(x, y),
            xytext=(text_x, text_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=9.4,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=3, shrinkB=4),
        )

    ax.set_xlabel("WAPE (%)", fontsize=11)
    ax.set_ylabel("Execution Penalty (thousand units)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.set_xlim(-0.5, 20.6)
    ax.set_ylim(-35, 750)
    fig.tight_layout()
    fig.savefig(IEOM_FIGURE_DIR / "favorita_accuracy_execution_frontier.pdf", bbox_inches="tight")
    fig.savefig(
        IEOM_FIGURE_DIR / "favorita_accuracy_execution_frontier.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def strategy_family(row: Mapping[str, object]) -> str:
    """Map a strategy row to a paper-facing strategy family."""
    strategy = str(row.get("strategy", ""))
    method = str(row.get("method_name", ""))
    method_family = str(row.get("method_family", ""))
    text = "{} {} {}".format(strategy, method, method_family).lower()
    if "ensemble" in text:
        return "Ensemble"
    if "smoothed" in text or "alpha" in text:
        return "Smoothing"
    if strategy in {"global_best_model", "best_accuracy", "family_best_model"}:
        return "Accuracy-first"
    if method in {"Global Best", "Family Best"}:
        return "Accuracy-first"
    if "inventory" in text:
        return "Inventory-first"
    if "stability" in text:
        return "Stability-first"
    if "dp_feasibility" in text:
        return "DP"
    if "feasibility" in text:
        return "Rule-based feasibility"
    return method_family if method_family and method_family != "nan" else "Other"


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    rng: np.random.Generator,
    n_bootstrap: int = 5000,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return (np.nan, np.nan)
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.choice(clean, size=len(clean), replace=True)
        estimates[index] = statistic(sample)
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def exact_sign_test(values: Sequence[float]) -> Dict[str, float]:
    """Compute a two-sided exact sign test for positive differences."""
    clean = [float(value) for value in values if np.isfinite(value) and abs(float(value)) > 1e-12]
    positive = sum(value > 0 for value in clean)
    nonzero = len(clean)
    if nonzero == 0:
        return {"positive": 0, "nonzero": 0, "p_value": np.nan}
    lower_tail = binomial_tail(nonzero, min(positive, nonzero - positive), lower=True)
    upper_tail = binomial_tail(nonzero, max(positive, nonzero - positive), lower=False)
    return {"positive": positive, "nonzero": nonzero, "p_value": min(1.0, 2.0 * min(lower_tail, upper_tail))}


def binomial_tail(n: int, k: int, lower: bool) -> float:
    """Return a binomial tail under p=0.5."""
    if lower:
        indices: Iterable[int] = range(0, k + 1)
    else:
        indices = range(k, n + 1)
    return float(sum(combination_count(n, i) for i in indices) / (2 ** n))


def combination_count(n: int, k: int) -> int:
    """Return n choose k with Python 3.7 compatibility."""
    if hasattr(math, "comb"):
        return math.comb(n, k)
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    numerator = 1
    denominator = 1
    for value in range(1, k + 1):
        numerator *= n - (k - value)
        denominator *= value
    return numerator // denominator


def write_csv_and_tex(
    table: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
    column_spec: str,
) -> None:
    """Write a CSV and compact IEOM-style LaTeX table."""
    table.to_csv(csv_path, index=False)
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[H]\n")
        handle.write("\\centering\n")
        handle.write("\\caption{" + latex_escape(caption) + "}\n")
        handle.write("\\label{" + label + "}\n")
        handle.write("\\scriptsize\n")
        handle.write("\\setlength{\\tabcolsep}{2pt}\n")
        handle.write("\\renewcommand{\\arraystretch}{0.86}\n")
        handle.write("\\begin{tabular}{" + column_spec + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join("\\textbf{" + latex_escape(col) + "}" for col in table.columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for _, row in table.iterrows():
            handle.write(" & ".join(latex_escape(row[col]) for col in table.columns) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")


def latex_escape(value: object) -> str:
    """Escape text for a LaTeX table."""
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("<", "$<$").replace(">", "$>$")
    return text


def infer_dataset_from_filename(stem: str) -> str:
    """Infer dataset name from a table filename."""
    if stem.startswith("favorita"):
        return "favorita"
    if stem.startswith("m5"):
        return "m5"
    if stem.startswith("walmart"):
        return "walmart"
    return "unknown"


def format_percent(value: float) -> str:
    """Format a fraction as a percentage."""
    return "{:.1f}%".format(100.0 * value)


def format_percentage_points(value: float) -> str:
    """Format a fraction as percentage points."""
    return "{:.1f} pp".format(100.0 * value)


def format_ci(ci: tuple[float, float]) -> str:
    """Format a fraction CI as percentages."""
    return "[{:.1f}%, {:.1f}%]".format(100.0 * ci[0], 100.0 * ci[1])


def format_pp_ci(ci: tuple[float, float]) -> str:
    """Format a fraction CI as percentage points."""
    return "[{:.1f}, {:.1f}] pp".format(100.0 * ci[0], 100.0 * ci[1])


def format_numeric_ci(ci: tuple[float, float]) -> str:
    """Format a numeric confidence interval."""
    return "[{:.3f}, {:.3f}]".format(ci[0], ci[1])


def format_p_value(value: float) -> str:
    """Format a p-value compactly for an IEOM table."""
    if not np.isfinite(value):
        return "not applicable"
    if value < 0.001:
        return "p < 0.001"
    return "p = {:.3f}".format(value)


def display_objective(objective_name: str) -> str:
    """Return a compact paper-facing objective label."""
    labels = {
        "accuracy_only": "Accuracy only",
        "inventory_only": "Inventory only",
        "inventory_volatility": "Inventory + volatility",
        "inventory_execution": "Inventory + execution",
        "inventory_switching": "Inventory + switching",
        "full_feasibility": "Full feasibility",
    }
    return labels.get(objective_name, objective_name.replace("_", " ").title())


def most_common(values: pd.Series) -> str:
    """Return the most common string value."""
    counts = values.astype(str).value_counts()
    return counts.index[0] if not counts.empty else ""


if __name__ == "__main__":
    main()
