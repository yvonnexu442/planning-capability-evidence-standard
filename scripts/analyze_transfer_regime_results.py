"""Compute block-bootstrap, ranking, and stress-period summaries."""

import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
OUT = ROOT / "results" / "working"
SEED = 42

from planning_environment.dynamic_inventory_simulator import DynamicInventoryConfig, simulate_dynamic_inventory  # noqa: E402
from decision_layer.practitioner_baselines import rate_limit_signal  # noqa: E402
from run_dynamic_policy_experiments import base_signals, feasibility_signal_lookup, load_inputs, safety_stock_lookup  # noqa: E402


def favorita_primary_periods() -> pd.DataFrame:
    forecasts, decisions = load_inputs(100)
    validation, test = forecasts[forecasts.split == "validation"], forecasts[forecasts.split == "test"]
    signals = base_signals(test, "global_lightgbm", safety_stock_lookup(validation))
    feasible = feasibility_signal_lookup(decisions)
    rows = []
    for unit, values in signals.items():
        mean = float(validation[validation.series_id.astype(str) == unit].drop_duplicates("date").actual.mean())
        capacity = 1.25 * max(mean, 1.0)
        strategies = {
            "accuracy_first": (values["accuracy_target"], np.zeros(len(values["actual"]))),
            "simple_ensemble_rate_limited": (rate_limit_signal(values["ensemble_target"], 0.4), np.zeros(len(values["actual"]))),
        }
        if unit in feasible:
            strategies["feasibility_aware"] = (feasible[unit]["target"], feasible[unit]["switch"])
        for strategy, (signal, switches) in strategies.items():
            target = np.asarray(signal) * 3.0
            config = DynamicInventoryConfig(lead_time=2, holding_cost_rate=1.0, shortage_cost_rate=5.0, order_change_cost_rate=0.05, expedite_cost_rate=0.10, replenishment_capacity=capacity, max_order_change_rate=0.5, initial_on_hand=float(target[0]), initial_order_quantity=mean)
            sim = simulate_dynamic_inventory(values["actual"], target, config, model_switch=switches, switching_cost_rate=0.05)
            rows.append(sim.assign(dataset="Favorita", planning_unit=unit, strategy=strategy, date=values["date"], scenario_id="L2_B5_moderate"))
    return pd.concat(rows, ignore_index=True)


def _paired_series(periods: pd.DataFrame, treatment: str, comparator: str, metric: str) -> pd.DataFrame:
    subset = periods[periods.strategy.isin([treatment, comparator])].copy()
    values = subset.pivot_table(index=["planning_unit", "date"], columns="strategy", values=metric, aggfunc="sum").dropna()
    return (values[treatment] - values[comparator]).rename("difference").reset_index()


def block_bootstrap(periods: pd.DataFrame, dataset: str, treatment: str, comparator: str, blocks: Iterable[int], comparison: str, scenario: str = None, repetitions: int = 1000) -> pd.DataFrame:
    frame = periods[periods.dataset == dataset].copy()
    if scenario is not None:
        frame = frame[frame.scenario_id == scenario]
    paired = _paired_series(frame, treatment, comparator, "total_dynamic_cost")
    rng = np.random.RandomState(SEED)
    records = []
    unit_values = {unit: group.sort_values("date").difference.to_numpy(float) for unit, group in paired.groupby("planning_unit")}
    observed = float(np.mean([values.sum() for values in unit_values.values()]))
    for block in blocks:
        draws = []
        for _ in range(repetitions):
            sampled_units = rng.choice(list(unit_values), size=len(unit_values), replace=True)
            totals = []
            for unit in sampled_units:
                values = unit_values[unit]
                needed, pieces = len(values), []
                while sum(map(len, pieces)) < needed:
                    start = rng.randint(0, len(values))
                    pieces.append(np.take(values, np.arange(start, start + block) % len(values)))
                totals.append(np.concatenate(pieces)[:needed].sum())
            draws.append(np.mean(totals))
        records.append({"dataset": dataset, "comparison": comparison, "scenario_id": scenario or "primary_transfer", "treatment": treatment, "comparator": comparator, "block_length": block, "observed_mean_paired_cost_difference": observed, "ci_lower_95": float(np.quantile(draws, .025)), "ci_upper_95": float(np.quantile(draws, .975)), "repetitions": repetitions, "planning_units": len(unit_values)})
    return pd.DataFrame(records)


def ranking_diagnostics(periods: pd.DataFrame, grouping: List[str]) -> pd.DataFrame:
    frame = periods.copy()
    frame["static_proxy"] = (
        (frame["order_up_to_target"] / 3.0 - frame["demand"]).abs()
        + 0.05 * frame["order_change"]
        + frame["execution_violation_units"]
    )
    aggregate = frame.groupby(grouping + ["strategy"], as_index=False).agg(static_objective=("static_proxy", "sum"), dynamic_objective=("total_dynamic_cost", "sum"))
    records = []
    for keys, group in aggregate.groupby(grouping):
        static_rank = group.static_objective.rank(method="average")
        dynamic_rank = group.dynamic_objective.rank(method="average")
        static_winner = group.loc[group.static_objective.idxmin(), "strategy"]
        dynamic_winner = group.loc[group.dynamic_objective.idxmin(), "strategy"]
        static_dynamic_cost = float(group.loc[group.strategy == static_winner, "dynamic_objective"].iloc[0])
        best_dynamic = float(group.dynamic_objective.min())
        key_values = keys if isinstance(keys, tuple) else (keys,)
        records.append({**dict(zip(grouping, key_values)), "strategy_rank_correlation": float(static_rank.corr(dynamic_rank, method="spearman")), "winner_agreement": int(static_winner == dynamic_winner), "static_winner": static_winner, "dynamic_winner": dynamic_winner, "dynamic_regret_of_static_winner": static_dynamic_cost - best_dynamic})
    return pd.DataFrame(records)



def main() -> None:
    cross_period = pd.read_csv(OUT / "cross_dataset_period_results.csv", parse_dates=["date"])
    dp_period = pd.read_csv(OUT / "dynamic_policy_period_results.csv", parse_dates=["date"])
    fav = favorita_primary_periods()
    fav.to_csv(OUT / "favorita_primary_period_results.csv", index=False)
    primary_cross = cross_period[(cross_period.transfer_regime == "strict_frozen") & (((cross_period.dataset == "M5") & (cross_period.grain == "item_store")) | (cross_period.dataset == "Walmart"))]
    boot = [
        block_bootstrap(fav, "Favorita", "simple_ensemble_rate_limited", "accuracy_first", [7, 14, 28], "ensemble_rl_vs_accuracy"),
        block_bootstrap(fav, "Favorita", "simple_ensemble_rate_limited", "feasibility_aware", [7, 14, 28], "ensemble_rl_vs_feasibility"),
        block_bootstrap(primary_cross, "M5", "simple_ensemble_rate_limited", "accuracy_first", [7, 14, 28], "ensemble_rl_vs_accuracy"),
        block_bootstrap(primary_cross, "Walmart", "simple_ensemble_rate_limited", "accuracy_first", [4, 8, 13], "ensemble_rl_vs_accuracy"),
    ]
    for dataset, blocks in (("M5", [7, 14, 28]), ("Walmart", [4, 8, 13])):
        for scenario in sorted(dp_period.scenario_id.unique()):
            boot.append(block_bootstrap(dp_period, dataset, "dp", "simple_ensemble_rate_limited", blocks, "dp_vs_ensemble_rl", scenario))
            boot.append(block_bootstrap(dp_period, dataset, "budgeted_dp", "dp", blocks, "budgeted_dp_vs_dp", scenario))
    bootstrap = pd.concat(boot, ignore_index=True)
    bootstrap.to_csv(OUT / "time_block_bootstrap.csv", index=False)
    ranking = ranking_diagnostics(primary_cross, ["dataset", "grain", "planning_unit", "transfer_regime"])
    ranking.to_csv(OUT / "static_dynamic_ranking_diagnostics.csv", index=False)
    stress = (
        cross_period[cross_period.dataset == "Walmart"]
        .groupby(["transfer_regime", "strategy", "stress_period"], dropna=False)
        .total_dynamic_cost.mean()
        .reset_index()
    )
    stress.to_csv(OUT / "walmart_stress_period_results.csv", index=False)
    print("Wrote time-block, ranking, and Walmart stress-period tables.")


if __name__ == "__main__":
    main()
