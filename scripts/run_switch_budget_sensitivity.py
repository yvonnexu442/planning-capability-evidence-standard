"""Run switch-budget sensitivity analysis for Budgeted DP selectors."""

import argparse
import copy
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from data_loaders.m5_loader import load_m5_modeling_table, validate_m5_files
from data_loaders.walmart_loader import load_walmart_modeling_table, validate_walmart_files
from evaluation.planning_utility import planning_loss_weight
from reporting.latex_export import export_summary_table
from utils.config import load_config
from utils.logging_utils import setup_logger
from visualization.plots import apply_paper_style, format_axis, place_legend, save_paper_figure, strategy_color, strategy_linestyle, strategy_marker

from run_m5_robustness_pipeline import (
    build_candidate_forecasts as build_m5_candidate_forecasts,
    build_decision_outputs as build_m5_decision_outputs,
    assign_splits as assign_m5_splits,
    normalize_run_mode,
    resolve_m5_max_series,
    short_strategy_label,
    summarize_decisions_for_scenario as summarize_m5_decisions_for_scenario,
    summarize_forecast_metrics as summarize_m5_forecast_metrics,
)
from run_walmart_robustness_pipeline import (
    build_walmart_candidate_forecasts,
    build_walmart_decision_outputs,
    assign_walmart_splits,
    resolve_walmart_max_series,
    summarize_forecast_metrics as summarize_walmart_forecast_metrics,
    summarize_walmart_decisions,
)


PLOT_STRATEGIES = [
    "global_best_model",
    "greedy_feasibility_selector",
    "dp_feasibility_selector",
    "budgeted_dp_feasibility_selector",
    "oracle_dp_feasibility_selector",
    "oracle_realized_demand",
]

INTERNAL_ONLY_STRATEGIES = {"full_outcome_oracle_dp_feasibility_selector"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run switch-budget K sensitivity analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["m5", "walmart"], default="walmart")
    parser.add_argument("--run-mode", choices=["quick", "medium", "full", "quick_mode", "medium_mode", "full_mode"], default="quick")
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--feature-set", choices=["history_only", "history_plus_context"], default="history_plus_context")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--paper-table-dir", default="paper/tables")
    parser.add_argument("--paper-figure-dir", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    """Run switch-budget K sensitivity and export paper-ready assets."""
    args = parse_args()
    config = load_config(args.config)
    run_mode = normalize_run_mode(args.run_mode)
    output_dir = Path(args.output_dir or config.get("project", {}).get("output_dir", "outputs"))
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    internal_dir = output_dir / "internal"
    paper_table_dir = Path(args.paper_table_dir)
    paper_figure_dir = Path(args.paper_figure_dir)
    for path in [table_dir, figure_dir, log_dir, internal_dir, paper_table_dir, paper_figure_dir]:
        path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("switch_budget_sensitivity", log_file=log_dir / "switch_budget_sensitivity.log")
    logger.info("Running %s switch-budget sensitivity for K values: %s.", args.dataset, args.k_values)

    if args.dataset == "walmart":
        result = run_walmart_switch_budget_sensitivity(args, config, run_mode, logger)
    else:
        result = run_m5_switch_budget_sensitivity(args, config, run_mode, logger)

    stem = "{}_switch_budget_sensitivity".format(args.dataset)
    paper_result = filter_internal_diagnostics(result)
    result.to_csv(internal_dir / "{}_with_internal_diagnostics.csv".format(stem), index=False)
    paper_result.to_csv(table_dir / "{}.csv".format(stem), index=False)
    export_summary_table(
        data=paper_result,
        table_name="{}_table".format(stem),
        output_dir=paper_table_dir,
        caption="{} switch-budget K sensitivity analysis.".format(args.dataset.upper()),
        label="tab:{}".format(stem.replace("_", "-")),
        numeric_precision=3,
        resize_to_textwidth=True,
    )
    save_k_sensitivity_figure(
        paper_result,
        metric="normalized_total_loss",
        y_label="Normalized Total Loss",
        png_path=figure_dir / "{}_normalized_loss.png".format(stem),
        pdf_path=paper_figure_dir / "{}_normalized_loss.pdf".format(stem),
    )
    save_k_sensitivity_figure(
        paper_result,
        metric="model_switch_count",
        y_label="Model Switch Count",
        png_path=figure_dir / "{}_switch_count.png".format(stem),
        pdf_path=paper_figure_dir / "{}_switch_count.pdf".format(stem),
    )
    logger.info("Switch-budget sensitivity completed successfully.")


def filter_internal_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Remove internal-only hindsight diagnostics from paper-facing exports."""
    if "strategy" not in data.columns:
        return data.copy()
    return data[~data["strategy"].astype(str).isin(INTERNAL_ONLY_STRATEGIES)].copy()


def run_walmart_switch_budget_sensitivity(
    args: argparse.Namespace,
    config: Mapping[str, object],
    run_mode: str,
    logger,
) -> pd.DataFrame:
    """Run Walmart K sensitivity using the current robustness pipeline."""
    raw_data_dir = Path(args.raw_data_dir or config.get("walmart_pipeline", {}).get("raw_data_dir", "data/raw/walmart"))
    validate_walmart_files(raw_data_dir)
    max_series = args.max_series if args.max_series is not None else resolve_walmart_max_series(run_mode, config)
    pipeline_config = config.get("walmart_pipeline", {})
    selection_holdout = int(pipeline_config.get("validation_horizon", 13)) + int(
        pipeline_config.get("test_horizon", 13)
    )
    modeling_table, _ = load_walmart_modeling_table(
        raw_data_dir=raw_data_dir,
        run_mode=run_mode,
        max_series=max_series,
        min_history_length=int(config.get("walmart_pipeline", {}).get("min_history_length", 80)),
        min_nonzero_observations=int(config.get("walmart_pipeline", {}).get("min_nonzero_observations", 20)),
        random_seed=int(config.get("project", {}).get("random_seed", 42)),
        selection_holdout_periods=selection_holdout,
    )
    modeling_table = assign_walmart_splits(modeling_table, config)
    forecasts = build_walmart_candidate_forecasts(modeling_table, args.feature_set, config)
    forecast_metrics = summarize_walmart_forecast_metrics(forecasts)

    rows = []
    for k_value in sorted(set(int(value) for value in args.k_values)):
        scenario_config = switch_budget_config(config, k_value)
        logger.info("Evaluating Walmart switch budget K=%s.", k_value)
        decisions = build_walmart_decision_outputs(forecasts, modeling_table, forecast_metrics, scenario_config)
        summary = summarize_walmart_decisions(
            decisions=decisions,
            config=scenario_config,
            run_mode=run_mode,
            feature_set=args.feature_set,
            experiment_name="switch_budget_sensitivity",
            window_type="all",
            scenario_name="K_{}".format(k_value),
            lambda_execution=planning_loss_weight(scenario_config.get("planning_loss_weights", {}), "lambda_execution"),
        )
        summary["max_switches"] = k_value
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_m5_switch_budget_sensitivity(
    args: argparse.Namespace,
    config: Mapping[str, object],
    run_mode: str,
    logger,
) -> pd.DataFrame:
    """Run M5 K sensitivity using the current robustness pipeline."""
    raw_data_dir = Path(args.raw_data_dir or config.get("m5_pipeline", {}).get("raw_data_dir", "data/raw/m5"))
    validate_m5_files(raw_data_dir)
    max_series = args.max_series if args.max_series is not None else resolve_m5_max_series(run_mode, config)
    pipeline_config = config.get("m5_pipeline", {})
    selection_holdout = int(pipeline_config.get("validation_horizon", 28)) + int(
        pipeline_config.get("test_horizon", 28)
    )
    modeling_table, _ = load_m5_modeling_table(
        raw_data_dir=raw_data_dir,
        run_mode=run_mode,
        max_series=max_series,
        min_history_length=int(config.get("m5_pipeline", {}).get("min_history_length", 365)),
        min_nonzero_observations=int(config.get("m5_pipeline", {}).get("min_nonzero_observations", 10)),
        selection_holdout_periods=selection_holdout,
    )
    modeling_table = assign_m5_splits(modeling_table, config)
    forecasts = build_m5_candidate_forecasts(modeling_table)
    forecast_metrics = summarize_m5_forecast_metrics(forecasts)

    rows = []
    for k_value in sorted(set(int(value) for value in args.k_values)):
        scenario_config = switch_budget_config(config, k_value)
        logger.info("Evaluating M5 switch budget K=%s.", k_value)
        decisions = build_m5_decision_outputs(forecasts, modeling_table, forecast_metrics, scenario_config)
        summary = summarize_m5_decisions_for_scenario(
            decisions=decisions,
            config=scenario_config,
            run_mode=run_mode,
            grain_level="item_store",
            intermittency_bucket="all",
            scenario_name="K_{}".format(k_value),
            lambda_execution=planning_loss_weight(scenario_config.get("planning_loss_weights", {}), "lambda_execution"),
        )
        summary["max_switches"] = k_value
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def switch_budget_config(config: Mapping[str, object], max_switches: int) -> Mapping[str, object]:
    """Return a deep-copied config with a different Budgeted-DP switch budget."""
    updated = copy.deepcopy(dict(config))
    updated.setdefault("feasibility_analysis", {}).setdefault("dp_selector", {})["max_switches"] = int(max_switches)
    return updated


def save_k_sensitivity_figure(data: pd.DataFrame, metric: str, y_label: str, png_path: Path, pdf_path: Path) -> None:
    """Save a switch-budget sensitivity line figure."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.35, 4.45))
    plot_data = data[data["strategy"].isin(PLOT_STRATEGIES)].copy()
    if plot_data.empty:
        plot_data = data.copy()
    for index, strategy in enumerate(PLOT_STRATEGIES):
        strategy_data = plot_data[plot_data["strategy"] == strategy].sort_values("max_switches")
        if strategy_data.empty:
            continue
        ax.plot(
            strategy_data["max_switches"],
            strategy_data[metric],
            marker=strategy_marker(strategy),
            color=strategy_color(strategy, index),
            linestyle=strategy_linestyle(strategy),
            linewidth=1.7,
            label=short_strategy_label(strategy),
        )
    format_axis(ax, x_label="Maximum Switches K", y_label=y_label)
    place_legend(ax, columns=3)
    save_paper_figure(fig, png_path, pdf_path)


if __name__ == "__main__":
    main()
