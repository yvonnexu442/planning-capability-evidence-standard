"""Build the public paper figures from canonical IEOM result tables.

Full experiment runs write detailed artifacts to ``results/working``.
The compact tables under ``results/final`` are tracked so a public clone can
rebuild every manuscript figure without the large unit-period intermediates.
"""

from pathlib import Path
import gzip
import hashlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "working"
RESULTS = ROOT / "results" / "final"
FIGURES = ROOT / "figures" / "final"
SUPPLEMENTARY_FIGURES = ROOT / "figures" / "supplementary"

BLUE = "#73A9C2"
TEAL = "#69B8B0"
PINK = "#E7A6B4"
DARK = "#3F4A50"
GRID = "#D9E1E5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#7A858A",
    "axes.linewidth": 0.6,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


COPY_FILES = [
    "dynamic_operational_summary.csv",
    "cross_dataset_frozen_transfer.csv",
    "cross_dataset_frozen_configurations.csv",
    "rate_limiter_mechanism_summary.csv",
    "switch_budget_mechanism_summary.csv",
    "switch_budget_mechanism_audit.csv",
    "nonbinding_governance_independent_summary.csv",
    "nonbinding_governance_disagreements.csv",
    "governance_independent_branch_checksums.csv",
    "mechanism_decomposition_summary.csv",
    "static_dynamic_ranking_diagnostics.csv",
    "time_block_bootstrap.csv",
    "walmart_stress_period_results.csv",
]

RAW_BRANCH_FILES = [
    "governance_unconstrained_dp_summary_raw.csv",
    "governance_unconstrained_dp_period_raw.csv",
    "governance_budgeted_dp_summary_raw.csv",
    "governance_budgeted_dp_period_raw.csv",
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_raw_governance_branches():
    """Publish deterministic compressed copies of both independent branch outputs."""
    raw_directory = RESULTS / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name in RAW_BRANCH_FILES:
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError("Missing independent branch output: {}".format(source))
        destination = raw_directory / (name + ".gz")
        with source.open("rb") as source_handle, destination.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                shutil.copyfileobj(source_handle, compressed)
        records.append({
            "branch": "budgeted_dp" if "budgeted_dp" in name else "unconstrained_dp",
            "level": "period" if "period" in name else "summary",
            "path": str(destination.relative_to(ROOT)),
            "rows": sum(1 for _ in source.open("rb")) - 1,
            "uncompressed_sha256": _sha256(source),
            "compressed_sha256": _sha256(destination),
        })
    pd.DataFrame(records).to_csv(
        RESULTS / "governance_independent_branch_checksums.csv", index=False
    )


def rebuild_dynamic_policy_summary():
    """Rebuild the compact strategy summary from the current independent run."""
    detailed = pd.read_csv(SOURCE / "dynamic_policy_stress_results.csv")
    compact = detailed.groupby(
        ["dataset", "scenario_id", "strategy"], as_index=False
    )["total_dynamic_cost"].mean()
    compact.to_csv(RESULTS / "dynamic_policy_stress_summary.csv", index=False)


def save_figure(fig, name, *, tight_rect=None, pad=1.08):
    """Save matched vector and raster copies with identical plot bounds."""
    fig.tight_layout(rect=tight_rect, pad=pad)
    fig.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIGURES / (name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def screening_figure():
    """Show ranking agreement alongside dataset-specific upper-tail regret."""
    data = pd.read_csv(RESULTS / "static_dynamic_ranking_diagnostics.csv")
    quantile_rows = []
    for dataset, group in data.groupby("dataset", sort=True):
        regret = group.dynamic_regret_of_static_winner.astype(float)
        quantile_rows.append({
            "dataset": dataset,
            "units": len(group),
            "p50_raw_regret": float(regret.quantile(0.5)),
            "p90_raw_regret": float(regret.quantile(0.9)),
            "maximum_raw_regret": float(regret.max()),
            "aggregation": "within_dataset",
        })
    pooled = data.dynamic_regret_of_static_winner.astype(float)
    quantile_rows.append({
        "dataset": "Pooled",
        "units": len(data),
        "p50_raw_regret": float(pooled.quantile(0.5)),
        "p90_raw_regret": float(pooled.quantile(0.9)),
        "maximum_raw_regret": float(pooled.max()),
        "aggregation": "descriptive_raw_pool",
    })
    quantiles = pd.DataFrame(quantile_rows)
    quantiles.to_csv(RESULTS / "static_dynamic_regret_quantiles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.15), sharey=True)
    colors = {"M5": BLUE, "Walmart": PINK}
    for dataset, group in data.groupby("dataset"):
        values = np.sort(group.dynamic_regret_of_static_winner.to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axes[0].step(values, cumulative, where="post", color=colors[dataset], linewidth=1.8, label=dataset)
        tail = np.sort(values[values > 1e-6])
        tail_cumulative = np.arange(1, len(tail) + 1) / len(tail)
        axes[1].step(tail, tail_cumulative, where="post", color=colors[dataset], linewidth=1.8, label=dataset)
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xscale("log")
    axes[0].set_title("All units", fontsize=9.5)
    axes[1].set_title("Non-equivalent choices ($R>10^{-6}$)", fontsize=9.5)
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Dynamic regret", fontsize=9)
        ax.tick_params(axis="both", labelsize=7.8)
    axes[0].set_ylabel("Cumulative share", fontsize=9)
    p90_by_dataset = dict(zip(quantiles.dataset, quantiles.p90_raw_regret))
    median_rank_correlation = float(data.strategy_rank_correlation.median())
    winner_agreement = float(data.winner_agreement.mean())
    equivalence_rate = float(data.dynamic_regret_of_static_winner.le(1e-6).mean())
    for dataset in ("M5", "Walmart"):
        axes[1].axvline(
            p90_by_dataset[dataset], color=colors[dataset], linestyle="--", linewidth=1.0
        )
    axes[1].text(
        0.97,
        0.08,
        "P90 raw regret\nM5 = {:,.1f}\nWalmart = {:,.0f}".format(
            p90_by_dataset["M5"], p90_by_dataset["Walmart"]
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color=DARK,
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": GRID, "pad": 2.0},
    )
    axes[0].text(
        0.03,
        0.08,
        "Rank correlation = {:.3f}\nWinner agreement = {:.1f}%\nMedian regret = 0\nCost-equivalent = {:.1f}%".format(
            median_rank_correlation,
            100.0 * winner_agreement,
            100.0 * equivalence_rate,
        ),
        transform=axes[0].transAxes,
        fontsize=8.5,
        color=DARK,
        bbox={"facecolor": "white", "edgecolor": GRID, "pad": 2.0},
    )
    axes[1].legend(frameon=False, loc="upper left", fontsize=8.5, handlelength=1.6)
    fig.suptitle("Rank agreement can coexist with upper-tail regret", y=0.945, fontsize=10)
    save_figure(fig, "static_dynamic_screening", tight_rect=(0, 0, 1, 0.90), pad=0.28)


def mechanism_comparison_audit():
    """Aggregate existing period records into public mechanism diagnostics.

    This is a deterministic summary of completed rollouts, not a new
    experiment.  It preserves separate notions of exact cost tie and executed
    order-path agreement so an inactive safeguard is not credited with value.
    """
    period_path = SOURCE / "cross_dataset_period_results.csv"
    output = RESULTS / "mechanism_comparison_audit.csv"
    if not period_path.exists():
        if not output.exists():
            raise FileNotFoundError("Missing mechanism audit and its completed period records")
        return

    periods = pd.read_csv(period_path, low_memory=False)
    decomposition = pd.read_csv(RESULTS / "mechanism_decomposition_summary.csv")
    effects = decomposition.groupby("mechanism").total_dynamic_cost_difference.mean()
    unit_keys = ["dataset", "grain", "transfer_regime", "planning_unit"]
    specifications = [
        ("Equal ensemble versus accuracy-first", "ensemble_diversification", "simple_ensemble", "accuracy_first", None),
        ("Rate limit on accuracy-first", "rate_limit_on_accuracy", "accuracy_rate_limited", "accuracy_first", "accuracy_first"),
        ("Rate limit on ensemble", "rate_limit_on_ensemble", "simple_ensemble_rate_limited", "simple_ensemble", "simple_ensemble"),
        ("Ensemble stabilization versus accuracy-first", "combined_vs_accuracy", "simple_ensemble_rate_limited", "accuracy_first", "simple_ensemble"),
    ]
    records = []
    for label, mechanism, treatment, comparator, activation_comparator in specifications:
        selected = periods[periods.strategy.isin([treatment, comparator])]
        agreements, ties, activations = [], [], []
        for key, group in selected.groupby(unit_keys):
            treated = group[group.strategy.eq(treatment)].sort_values("date")
            compared = group[group.strategy.eq(comparator)].sort_values("date")
            if len(treated) != len(compared):
                continue
            cost_difference = float(treated.total_dynamic_cost.sum() - compared.total_dynamic_cost.sum())
            agreement = bool(np.allclose(treated.order_quantity, compared.order_quantity, rtol=0.0, atol=1e-9))
            agreements.append(agreement)
            ties.append(bool(agreement and abs(cost_difference) <= 1e-9))
            if activation_comparator is not None:
                plain = periods[
                    periods.dataset.eq(key[0])
                    & periods.grain.eq(key[1])
                    & periods.transfer_regime.eq(key[2])
                    & periods.planning_unit.eq(key[3])
                    & periods.strategy.eq(activation_comparator)
                ].sort_values("date")
                activations.append(
                    bool(np.any(np.abs(treated.order_up_to_target.to_numpy(float) - plain.order_up_to_target.to_numpy(float)) > 1e-9))
                )
        records.append({
            "comparison": label,
            "mean_dynamic_cost_difference": float(effects[mechanism]),
            "activation_rate": np.nan if not activations else float(np.mean(activations)),
            "tie_rate": float(np.mean(ties)),
            "executed_plan_agreement": float(np.mean(agreements)),
            "n_units": len(agreements),
        })
    pd.DataFrame(records).to_csv(output, index=False)


def regime_figure():
    """Build the regime map from the tracked strategy-level summary.

    Relative values use ensemble stabilization as the denominator. Negative
    cells therefore indicate lower realized rollout cost than that baseline.
    """
    summary = pd.read_csv(RESULTS / "dynamic_policy_stress_summary.csv")
    pivot = summary.pivot_table(index=["dataset", "scenario_id"], columns="strategy", values="total_dynamic_cost")
    relative = 100.0 * (pivot.divide(pivot["simple_ensemble_rate_limited"], axis=0) - 1.0)
    display = relative[[column for column in ["greedy", "dp", "budgeted_dp"] if column in relative]].copy()
    scenario_order = [
        "frozen_horizon", "long_lead", "persistent_backlog", "high_switch_cost",
        "strict_switch_budget", "tight_capacity", "governance_fixed_K2",
        "governance_validation_q25",
    ]
    scenario_labels = [
        "Frozen horizon", "Long lead", "Persistent backlog", "High switch cost",
        "Strict budget", "Tight capacity", "Fixed governance cap",
        "Validation-quartile cap",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.9), sharey=True)
    cost_cmap = LinearSegmentedColormap.from_list("cost", [TEAL, "#F7F7F4", PINK])
    image = None
    for ax, dataset in zip(axes, ["M5", "Walmart"]):
        panel = display.xs(dataset, level="dataset").reindex(scenario_order)
        values = panel.to_numpy()
        image = ax.imshow(values, aspect="auto", cmap=cost_cmap, vmin=-35.0, vmax=35.0)
        ax.set_xticks(range(len(panel.columns)))
        ax.set_xticklabels(["Greedy", "Approx. DP", "Budgeted DP"][: len(panel.columns)], rotation=24, ha="right")
        ax.set_yticks(range(len(panel)))
        ax.set_yticklabels(scenario_labels)
        ax.set_title(dataset, pad=3)
        ax.set_xticks(np.arange(-0.5, len(panel.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(panel), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    "{:+.1f}%".format(value),
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color="white" if abs(value) >= 20 else DARK,
                )
    fig.suptitle("Dynamic cost relative to ensemble stabilization", y=0.99, fontsize=9)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
    colorbar.set_label("Dynamic-cost change (%)\nNegative = lower cost")
    fig.subplots_adjust(left=0.25, right=0.89, bottom=0.20, top=0.84, wspace=0.08)
    SUPPLEMENTARY_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUPPLEMENTARY_FIGURES / "regime_policy_map.pdf", bbox_inches="tight")
    fig.savefig(SUPPLEMENTARY_FIGURES / "regime_policy_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        source = SOURCE / name
        destination = RESULTS / name
        if source.exists():
            shutil.copy2(source, destination)
        elif not destination.exists():
            raise FileNotFoundError(
                "Missing both detailed and canonical result table: {}".format(name)
            )
    preserve_raw_governance_branches()
    rebuild_dynamic_policy_summary()
    mechanism_comparison_audit()
    screening_figure()
    regime_figure()
    print("Built canonical IEOM tables and figures.")


if __name__ == "__main__":
    main()
