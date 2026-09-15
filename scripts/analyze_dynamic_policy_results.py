"""Analyze held-out dynamic results with paired planning-unit uncertainty."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "working"
PRIMARY_SCENARIO = "L2_B5_moderate"
SEED = 42


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    treatment: str,
    comparator: str,
    metrics: List[str],
    repetitions: int = 2000,
) -> pd.DataFrame:
    subset = frame[frame["strategy"].isin([treatment, comparator])].copy()
    wide = subset.pivot(index="planning_unit", columns="strategy", values=metrics)
    units = wide.index.to_numpy()
    rng = np.random.RandomState(SEED)
    records = []
    for metric in metrics:
        differences = wide[metric][treatment] - wide[metric][comparator]
        differences = differences.dropna()
        observed = float(differences.mean())
        draws = np.empty(repetitions, dtype=float)
        values = differences.to_numpy(dtype=float)
        for index in range(repetitions):
            draws[index] = float(rng.choice(values, size=len(values), replace=True).mean())
        records.append(
            {
                "scenario_id": PRIMARY_SCENARIO,
                "treatment": treatment,
                "comparator": comparator,
                "metric": metric,
                "planning_units": len(differences),
                "mean_paired_difference": observed,
                "ci_lower_95": float(np.quantile(draws, 0.025)),
                "ci_upper_95": float(np.quantile(draws, 0.975)),
                "bootstrap_repetitions": repetitions,
            }
        )
    return pd.DataFrame(records)


def dynamic_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "WAPE",
        "MASE",
        "total_dynamic_cost",
        "fill_rate",
        "planning_execution_gap_rate",
        "plan_volatility",
        "switch_count",
    ]
    return (
        results.groupby(["scenario_id", "strategy"], as_index=False)[metrics]
        .mean()
        .sort_values(["scenario_id", "total_dynamic_cost", "strategy"])
    )


def weight_uncertainty(results: pd.DataFrame, draws: int = 500) -> pd.DataFrame:
    """Re-rank strategies over admissible operational-component weights.

    This is a sensitivity analysis over saved test outcomes, not a deployment
    tuning step. It cannot feed parameters back into the held-out policies.
    """

    components = [
        "total_holding_cost",
        "total_shortage_cost",
        "execution_violation_units",
        "plan_volatility",
        "switch_count",
    ]
    aggregate = results.groupby("strategy", as_index=False)[components].mean().set_index("strategy")
    reference = aggregate.loc["accuracy_first"].abs().replace(0.0, np.nan)
    fallback = aggregate.abs().replace(0.0, np.nan).median().fillna(1.0)
    reference = reference.fillna(fallback).fillna(1.0)
    normalized = aggregate.divide(reference, axis=1)
    rng = np.random.RandomState(SEED)
    weights = rng.dirichlet(np.ones(len(components)), size=draws)
    records = []
    for draw_id, weight in enumerate(weights):
        losses = normalized.to_numpy().dot(weight)
        best = float(losses.min())
        for strategy, loss in zip(normalized.index, losses):
            records.append(
                {
                    "draw_id": draw_id,
                    "strategy": strategy,
                    "is_optimal": int(np.isclose(loss, best)),
                    "weighted_loss": float(loss),
                    "regret": float(loss - best),
                    **{"weight_{}".format(component): float(value) for component, value in zip(components, weight)},
                }
            )
    return pd.DataFrame(records)



def main() -> None:
    results = pd.read_csv(RESULT_DIR / "dynamic_inventory_results.csv")
    summary = dynamic_summary(results)
    summary.to_csv(RESULT_DIR / "dynamic_operational_summary.csv", index=False)
    primary = results[results["scenario_id"] == PRIMARY_SCENARIO]
    metrics = [
        "total_dynamic_cost",
        "planning_execution_gap_rate",
        "fill_rate",
        "plan_volatility",
        "switch_count",
        "WAPE",
    ]
    bootstrap = pd.concat(
        [
            paired_cluster_bootstrap(primary, "feasibility_aware", "accuracy_first", metrics),
            paired_cluster_bootstrap(primary, "feasibility_aware", "accuracy_rate_limited", metrics),
            paired_cluster_bootstrap(primary, "feasibility_aware", "simple_ensemble_rate_limited", metrics),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(RESULT_DIR / "clustered_bootstrap.csv", index=False)
    robustness = weight_uncertainty(results, draws=500)
    robustness.to_csv(RESULT_DIR / "weight_robustness.csv", index=False)
    robustness.groupby("strategy").agg(
        probability_optimal=("is_optimal", "mean"),
        mean_regret=("regret", "mean"),
        median_regret=("regret", "median"),
        regret_90th=("regret", lambda x: float(np.quantile(x, 0.90))),
        worst_case_regret=("regret", "max"),
    ).reset_index().to_csv(RESULT_DIR / "weight_robustness_summary.csv", index=False)
    metric_summary = results.groupby("strategy", as_index=False)[["WAPE", "MASE"]].mean()
    metric_summary["rank_by_WAPE"] = metric_summary["WAPE"].rank(method="min")
    metric_summary["rank_by_MASE"] = metric_summary["MASE"].rank(method="min")
    metric_summary.to_csv(RESULT_DIR / "forecast_metric_sensitivity.csv", index=False)
    print("Wrote dynamic-policy summaries, uncertainty, and robustness tables.")


if __name__ == "__main__":
    main()
