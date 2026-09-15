"""Run frozen transfer and dynamic-policy operating-regime experiments.

The runner stores period-level outcomes for paired block bootstrap. Realized
test demand is used only for forward simulation and ex-post diagnostics; policy
parameters and governance caps are fixed from validation information. Any
reported minimum across test-policy outcomes is a retrospective protocol
envelope, not a validation-selected deployable policy.
"""

import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from decision_layer.dynamic_inventory_policies import DynamicPolicyConfig, select_dynamic_policy  # noqa: E402
from decision_layer.practitioner_baselines import rate_limit_signal  # noqa: E402
from planning_environment.dynamic_inventory_simulator import (  # noqa: E402
    DynamicInventoryConfig,
    simulate_dynamic_inventory,
    summarize_dynamic_inventory,
)
from utils.config import load_config  # noqa: E402


OUT = ROOT / "results" / "working"
SEED = 42
STRICT_RATE = 0.40


def wape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    denominator = float(np.abs(actual).sum())
    return 0.0 if denominator == 0 else float(np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / denominator)


def _aggregate_candidates(frame: pd.DataFrame, keys: Sequence[str], label: str) -> pd.DataFrame:
    group = ["date", "split", "model_name", *keys]
    output = frame.groupby(group, as_index=False).agg(actual=("actual", "sum"), forecast=("forecast", "sum"))
    output["series_id"] = output[list(keys)].astype(str).agg("__".join, axis=1)
    output["grain"] = label
    return output


def m5_panels(max_series: int, config: Mapping[str, object]) -> Dict[str, pd.DataFrame]:
    # Heavy dataset/plotting modules are imported only for a full raw-data run;
    # the live ensemble formulas remain independently regression-testable.
    from data_loaders.m5_loader import load_m5_modeling_table
    from run_m5_robustness_pipeline import assign_splits, build_candidate_forecasts

    pipeline_config = config.get("m5_pipeline", {})
    selection_holdout = int(pipeline_config.get("validation_horizon", 28)) + int(
        pipeline_config.get("test_horizon", 28)
    )
    table, _ = load_m5_modeling_table(
        ROOT / "data/raw/m5", "quick", max_series=max_series,
        min_history_length=365, min_nonzero_observations=10,
        selection_holdout_periods=selection_holdout,
    )
    table = assign_splits(table, config)
    forecasts = build_candidate_forecasts(table)
    item = forecasts.copy()
    item["grain"] = "item_store"
    return {
        "item_store": item,
        "department_store": _aggregate_candidates(forecasts, ["dept_id", "store_id"], "department_store"),
        "category_store": _aggregate_candidates(forecasts, ["cat_id", "store_id"], "category_store"),
    }


def walmart_panel(max_series: int, config: Mapping[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from data_loaders.walmart_loader import load_walmart_modeling_table
    from run_walmart_robustness_pipeline import assign_walmart_splits, build_walmart_candidate_forecasts

    pipeline_config = config.get("walmart_pipeline", {})
    selection_holdout = int(pipeline_config.get("validation_horizon", 13)) + int(
        pipeline_config.get("test_horizon", 13)
    )
    table, _ = load_walmart_modeling_table(
        ROOT / "data/raw/walmart", "quick", max_series=max_series, min_history_length=80,
        min_nonzero_observations=20, random_seed=SEED,
        selection_holdout_periods=selection_holdout,
    )
    table = assign_walmart_splits(table, config)
    forecasts = build_walmart_candidate_forecasts(table, "history_only", config)
    context = table[["series_id", "date", "split", "is_holiday", "markdown_total"]].copy()
    historical_markdowns = pd.to_numeric(
        context.loc[context["split"] != "test", "markdown_total"], errors="coerce"
    ).fillna(0.0)
    positive_historical_markdowns = historical_markdowns[historical_markdowns > 0.0]
    markdown_stress_threshold = float(positive_historical_markdowns.quantile(0.75)) if len(positive_historical_markdowns) else float("inf")
    markdown = pd.to_numeric(context["markdown_total"], errors="coerce").fillna(0.0)
    context["stress_period"] = np.where(
        (pd.to_numeric(context["is_holiday"], errors="coerce").fillna(0).astype(int) == 1)
        | ((markdown > 0.0) & (markdown >= markdown_stress_threshold)),
        "holiday_or_markdown_stress", "ordinary",
    )
    forecasts["grain"] = "department_store"
    return forecasts, context


def _validation_weights(frame: pd.DataFrame) -> Dict[str, float]:
    validation = frame[frame["split"] == "validation"]
    errors = validation.groupby("model_name").apply(lambda x: wape(x["actual"].to_numpy(), x["forecast"].to_numpy()))
    inverse = 1.0 / np.maximum(errors.to_numpy(dtype=float), 1e-6)
    inverse /= inverse.sum()
    return dict(zip(errors.index.astype(str), inverse))


def equal_weight_ensemble(pivot: pd.DataFrame) -> np.ndarray:
    """Return the live equal-weight ensemble used in reported transfer results."""
    return pivot.mean(axis=1).to_numpy(float)


def validation_weighted_ensemble(pivot: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    """Return the live frozen validation-weighted ensemble."""
    return sum(pivot[name].to_numpy(float) * weights.get(str(name), 0.0) for name in pivot.columns)


def _tune_rate(frame: pd.DataFrame, weights: Mapping[str, float]) -> float:
    candidates = [0.10, 0.20, 0.40, 0.75]
    costs = []
    for rate in candidates:
        unit_costs = []
        for _, group in frame[frame["split"] == "validation"].groupby("series_id"):
            pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
            actual = group.drop_duplicates("date").sort_values("date")["actual"].to_numpy(float)
            signal = sum(pivot[name].to_numpy(float) * weights.get(str(name), 0.0) for name in pivot.columns)
            target = rate_limit_signal(signal * 3.0, rate)
            sim = simulate_dynamic_inventory(actual, target, DynamicInventoryConfig(lead_time=2, shortage_cost_rate=5.0, initial_on_hand=target[0], initial_order_quantity=float(np.mean(actual))))
            unit_costs.append(sim["total_dynamic_cost"].sum())
        costs.append((float(np.mean(unit_costs)), rate))
    return min(costs)[1]


def _mechanism_signals(group: pd.DataFrame, regime: str, rate: float, weights: Mapping[str, float]) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
    validation_best = group.attrs.get("best_model")
    if validation_best not in pivot.columns:
        validation_best = sorted(pivot.columns)[0]
    best = pivot[validation_best].to_numpy(float)
    ensemble = equal_weight_ensemble(pivot)
    weighted = validation_weighted_ensemble(pivot, weights)
    yield "accuracy_first", best, best
    yield "accuracy_rate_limited", best, rate_limit_signal(best, rate)
    yield "simple_ensemble", ensemble, ensemble
    yield "simple_ensemble_rate_limited", ensemble, rate_limit_signal(ensemble, rate)
    yield "best_single_model_rate_limited", best, rate_limit_signal(best, rate)
    yield "operationally_weighted_ensemble", weighted, weighted


def _run_transfer(dataset: str, panel: pd.DataFrame, context: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    validation = panel[panel["split"] == "validation"]
    test = panel[panel["split"] == "test"]
    best_model = min(
        ((wape(g["actual"].to_numpy(), g["forecast"].to_numpy()), str(m)) for m, g in validation.groupby("model_name")),
    )[1]
    equal = {str(model): 1.0 / validation["model_name"].nunique() for model in validation["model_name"].unique()}
    calibrated = _validation_weights(panel)
    regimes = {
        "strict_frozen": (STRICT_RATE, equal),
        "family_frozen_parameter_recalibrated": (_tune_rate(panel, equal), equal),
        "fully_in_domain_tuned": (_tune_rate(panel, calibrated), calibrated),
    }
    summaries, periods, configs = [], [], []
    intermittency = validation.drop_duplicates(["series_id", "date"]).groupby("series_id")["actual"].apply(lambda x: float((x == 0).mean()))
    for regime, (rate, weights) in regimes.items():
        configs.append({"dataset": dataset, "transfer_regime": regime, "rate_limit": rate, "weights": json.dumps(weights, sort_keys=True), "best_model": best_model, "test_tuned": False})
        for series_id, group in test.groupby("series_id"):
            group = group.copy()
            group.attrs["best_model"] = best_model
            actual_rows = group.drop_duplicates("date").sort_values("date")
            actual = actual_rows["actual"].to_numpy(float)
            dates = actual_rows["date"].to_numpy()
            scale = float(validation[validation["series_id"] == series_id].drop_duplicates("date")["actual"].std() or 0.0)
            capacity = 1.25 * max(float(validation[validation["series_id"] == series_id].drop_duplicates("date")["actual"].mean()), 1.0)
            for strategy, forecast, base in _mechanism_signals(group, regime, rate, weights):
                target = np.maximum(base * 3.0 + 0.5 * scale, 0.0)
                config = DynamicInventoryConfig(lead_time=2, holding_cost_rate=1.0, shortage_cost_rate=5.0, order_change_cost_rate=0.05, expedite_cost_rate=0.0, replenishment_capacity=capacity, max_order_change_rate=0.5, initial_on_hand=float(target[0]), initial_order_quantity=capacity / 1.25)
                sim = simulate_dynamic_inventory(actual, target, config)
                summary = summarize_dynamic_inventory(sim)
                zero_rate = float(intermittency.get(series_id, 0.0))
                bucket = "high" if zero_rate >= 0.5 else ("medium" if zero_rate >= 0.2 else "low")
                summaries.append({"dataset": dataset, "grain": group["grain"].iloc[0], "planning_unit": series_id, "transfer_regime": regime, "strategy": strategy, "WAPE": wape(actual, forecast), "intermittency": bucket, **summary})
                sim = sim.assign(dataset=dataset, grain=group["grain"].iloc[0], planning_unit=series_id, transfer_regime=regime, strategy=strategy, date=dates)
                if context is not None:
                    lookup = context[(context["series_id"] == series_id) & (context["split"] == "test")].set_index("date")["stress_period"]
                    sim["stress_period"] = pd.to_datetime(sim["date"]).map(lookup)
                periods.append(sim)
    return pd.DataFrame(summaries), pd.concat(periods, ignore_index=True), configs


def _run_dynamic_policy_stress(dataset: str, panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validation, test = panel[panel["split"] == "validation"], panel[panel["split"] == "test"]
    scenarios = [
        ("long_lead", 7, 0.25, 0.5, "validation_q25", 1, 1.0),
        ("high_switch_cost", 2, 2.0, 0.5, "validation_q25", 1, 1.0),
        ("strict_switch_budget", 2, 0.5, 0.5, "fixed_K2", 1, 1.0),
        ("frozen_horizon", 2, 0.5, 0.5, "validation_q25", 4, 1.0),
        ("persistent_backlog", 4, 0.5, 0.3, "validation_q25", 1, 1.0),
        ("tight_capacity", 4, 0.5, 0.15, "validation_q25", 2, 1.0),
        ("governance_fixed_K2", 4, 0.5, 0.5, "fixed_K2", 1, 1.0),
        ("governance_validation_q25", 4, 0.5, 0.5, "validation_q25", 1, 1.0),
    ]
    validation_switches = {}
    for name, lead, switch_cost, rate, budget_rule, freeze, persistence in scenarios:
        counts = []
        for series_id, group in validation.groupby("series_id"):
            pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
            actual = group.drop_duplicates("date").sort_values("date")["actual"]
            mean, safety = max(float(actual.mean()), 1.0), 0.5 * float(actual.std() or 0.0)
            capacity = mean * (0.75 if name in {"tight_capacity", "persistent_backlog"} else 1.25)
            cfg = DynamicPolicyConfig(
                lead_time=lead, shortage_cost_rate=5.0, switching_cost_rate=switch_cost,
                capacity=capacity, max_order_change_rate=rate, frozen_horizon=freeze,
                backlog_persistence=persistence, state_rounding=max(mean * 0.1, 0.25), beam_width=96,
            )
            output = select_dynamic_policy(
                {str(column): pivot[column].to_numpy(float) for column in pivot.columns},
                safety, (lead + 1) * mean, mean, cfg, "dp",
            )
            counts.append(int(output["switch"].sum()))
        validation_switches[name] = counts
    selected_budgets = {
        name: (2 if rule == "fixed_K2" else int(np.floor(np.quantile(validation_switches[name], 0.25))))
        for name, _, _, _, rule, _, _ in scenarios
    }
    summaries, periods = [], []
    for series_id, group in test.groupby("series_id"):
        pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
        actual_rows = group.drop_duplicates("date").sort_values("date")
        actual, dates = actual_rows["actual"].to_numpy(float), actual_rows["date"].to_numpy()
        val_actual = validation[validation["series_id"] == series_id].drop_duplicates("date")["actual"]
        mean, safety = max(float(val_actual.mean()), 1.0), 0.5 * float(val_actual.std() or 0.0)
        ensemble = pivot.mean(axis=1).to_numpy(float)
        for name, lead, switch_cost, rate, budget_rule, freeze, persistence in scenarios:
            budget = selected_budgets[name]
            capacity = mean * (0.75 if name in {"tight_capacity", "persistent_backlog"} else 1.25)
            policy_config = DynamicPolicyConfig(lead_time=lead, shortage_cost_rate=5.0, switching_cost_rate=switch_cost, capacity=capacity, max_order_change_rate=rate, frozen_horizon=freeze, backlog_persistence=persistence, state_rounding=max(mean * 0.1, 0.25), beam_width=96)
            strategies = {"simple_ensemble_rate_limited": {"forecast": ensemble, "plan": rate_limit_signal(ensemble * (lead + 1) + safety, STRICT_RATE), "switch": np.zeros(len(actual)), "model": np.repeat("simple_ensemble", len(actual))}}
            candidate_map = {str(column): pivot[column].to_numpy(float) for column in pivot.columns}
            for method in ("greedy", "dp", "budgeted_dp"):
                cfg = replace(policy_config, switch_budget=(budget if method == "budgeted_dp" else None))
                started = time.perf_counter()
                output = select_dynamic_policy(candidate_map, safety, (lead + 1) * mean, mean, cfg, method)
                output["runtime_seconds"] = time.perf_counter() - started
                output["policy_branch_execution"] = "independent"
                strategies[method] = output
            for strategy, output in strategies.items():
                config = DynamicInventoryConfig(lead_time=lead, holding_cost_rate=1.0, shortage_cost_rate=5.0, order_change_cost_rate=0.05, expedite_cost_rate=0.0, replenishment_capacity=capacity, max_order_change_rate=rate, initial_on_hand=(lead + 1) * mean, initial_order_quantity=mean)
                sim = simulate_dynamic_inventory(actual, output["plan"], config, model_switch=output["switch"], switching_cost_rate=switch_cost)
                summaries.append({"dataset": dataset, "planning_unit": series_id, "scenario_id": name, "strategy": strategy, "runtime_seconds": output.get("runtime_seconds", np.nan), "policy_branch_execution": output.get("policy_branch_execution", "not_applicable"), "selected_budget_K": budget if strategy == "budgeted_dp" else np.nan, "budget_rule": budget_rule if strategy == "budgeted_dp" else "not_applicable", "validation_unconstrained_switch_q25": float(np.quantile(validation_switches[name], 0.25)), "validation_unconstrained_switch_median": float(np.median(validation_switches[name])), "WAPE": wape(actual, output["forecast"]), **summarize_dynamic_inventory(sim)})
                periods.append(sim.assign(dataset=dataset, planning_unit=series_id, scenario_id=name, strategy=strategy, selected_model=output["model"], selected_budget_K=budget if strategy == "budgeted_dp" else np.nan, budget_rule=budget_rule if strategy == "budgeted_dp" else "not_applicable", date=dates))
    return pd.DataFrame(summaries), pd.concat(periods, ignore_index=True)


def _sha256(path: Path) -> str:
    """Return the SHA-256 checksum of one preserved branch artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_independent_policy_branches(summary: pd.DataFrame, periods: pd.DataFrame) -> None:
    """Write immutable-by-convention raw outputs for both independently run DP branches."""
    records = []
    for strategy, branch in (("dp", "unconstrained_dp"), ("budgeted_dp", "budgeted_dp")):
        branch_summary = summary[summary.strategy.eq(strategy)].sort_values(
            ["dataset", "scenario_id", "planning_unit"]
        )
        branch_periods = periods[periods.strategy.eq(strategy)].sort_values(
            ["dataset", "scenario_id", "planning_unit", "date"]
        )
        for level, frame in (("summary", branch_summary), ("period", branch_periods)):
            path = OUT / f"governance_{branch}_{level}_raw.csv"
            frame.to_csv(path, index=False)
            records.append({
                "branch": branch,
                "level": level,
                "path": str(path.relative_to(ROOT)),
                "rows": len(frame),
                "sha256": _sha256(path),
            })
    pd.DataFrame(records).to_csv(OUT / "governance_independent_branch_checksums.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/default.yaml")
    all_summaries, all_periods, frozen = [], [], []
    panels = m5_panels(100, config)
    for grain, panel in panels.items():
        summary, period, configs = _run_transfer("M5", panel)
        all_summaries.append(summary); all_periods.append(period); frozen.extend(configs)
    walmart, context = walmart_panel(100, config)
    summary, period, configs = _run_transfer("Walmart", walmart, context)
    all_summaries.append(summary); all_periods.append(period); frozen.extend(configs)
    dp_summary_m5, dp_period_m5 = _run_dynamic_policy_stress("M5", panels["item_store"])
    dp_summary_wm, dp_period_wm = _run_dynamic_policy_stress("Walmart", walmart)
    pd.concat(all_summaries, ignore_index=True).to_csv(OUT / "cross_dataset_frozen_transfer.csv", index=False)
    pd.concat(all_periods, ignore_index=True).to_csv(OUT / "cross_dataset_period_results.csv", index=False)
    dynamic_summary = pd.concat([dp_summary_m5, dp_summary_wm], ignore_index=True)
    dynamic_periods = pd.concat([dp_period_m5, dp_period_wm], ignore_index=True)
    dynamic_summary.to_csv(OUT / "dynamic_policy_stress_results.csv", index=False)
    dynamic_periods.to_csv(OUT / "dynamic_policy_period_results.csv", index=False)
    preserve_independent_policy_branches(dynamic_summary, dynamic_periods)
    pd.DataFrame(frozen).to_csv(OUT / "cross_dataset_frozen_configurations.csv", index=False)
    print("Wrote transfer and operating-regime results to", OUT)


if __name__ == "__main__":
    main()
