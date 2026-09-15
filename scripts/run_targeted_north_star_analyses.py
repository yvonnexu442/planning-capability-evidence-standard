"""Run the frozen targeted analyses required by the North Star decision gate.

The script consumes retained unit-period rollouts where possible and runs only
the predeclared three-level lead-time and switching-cost sweeps. Forecast models
are not retrained and no test outcome is used to tune a policy parameter.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from decision_layer.dynamic_inventory_policies import DynamicPolicyConfig, select_dynamic_policy  # noqa: E402
from decision_layer.practitioner_baselines import rate_limit_signal  # noqa: E402
from planning_environment.dynamic_inventory_simulator import (  # noqa: E402
    DynamicInventoryConfig,
    simulate_dynamic_inventory,
)
from run_transfer_regime_experiments import m5_panels, walmart_panel  # noqa: E402
from utils.config import load_config  # noqa: E402


WORKING = ROOT / "results" / "working"
FINAL = ROOT / "results" / "final"
FIGURES = ROOT / "figures" / "final"
SUPPLEMENTARY_FIGURES = ROOT / "figures" / "supplementary"
REPORTS = ROOT / "reports"
TOL = 1e-6
SEED = 42
COLORS = {"M5": "#73A9C2", "Walmart": "#E7A6B4"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 8.0,
    "mathtext.fontset": "stix",
})


def save_figure(fig: plt.Figure, name: str, *, manuscript: bool = False) -> None:
    destination = FIGURES if manuscript else SUPPLEMENTARY_FIGURES
    destination.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(destination / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def unit_bootstrap_ci(values: pd.Series, repetitions: int = 2000) -> tuple[float, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return np.nan, np.nan
    rng = np.random.RandomState(SEED)
    draws = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def coupling_labelled_regret() -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(WORKING / "static_dynamic_ranking_diagnostics.csv")
    periods = pd.read_csv(WORKING / "cross_dataset_period_results.csv", low_memory=False)
    rate = pd.read_csv(WORKING / "rate_limiter_mechanism_audit.csv")
    key = ["dataset", "grain", "planning_unit", "transfer_regime"]
    costs = periods.groupby(key + ["strategy"], as_index=False).total_dynamic_cost.sum()
    best = costs.groupby(key).total_dynamic_cost.min().rename("best_dynamic_policy_cost").reset_index()
    simple = costs.groupby(key).total_dynamic_cost.min().rename("best_simple_policy_cost").reset_index()
    joined = diagnostics.merge(best, on=key).merge(simple, on=key).merge(
        rate[key + ["rate_limit_binding", "executed_plan_agreement"]], on=key, how="left"
    )
    joined["material_regret_threshold"] = np.maximum(TOL, 0.01 * joined.best_dynamic_policy_cost)
    joined["within_material_tolerance"] = (
        joined.dynamic_regret_of_static_winner <= joined.material_regret_threshold
    )
    joined["mechanism_status"] = np.where(joined.rate_limit_binding, "active", "inactive")
    output = pd.DataFrame({
        "dataset": joined.dataset,
        "unit_id": joined.planning_unit,
        "regime_id": joined.transfer_regime,
        "predeclared_mechanism": "ensemble_rate_cap",
        "mechanism_status": joined.mechanism_status,
        "activation_measure": np.where(joined.rate_limit_binding, "target_changed", "target_unchanged"),
        "static_selected_policy": joined.static_winner,
        "dynamic_selected_policy": joined.dynamic_winner,
        "winner_agreement": joined.winner_agreement,
        "dynamic_regret": joined.dynamic_regret_of_static_winner,
        "within_material_tolerance": joined.within_material_tolerance,
        "execution_agreement": joined.executed_plan_agreement,
        "best_simple_policy_cost": joined.best_simple_policy_cost,
        "best_advanced_policy_cost": np.nan,
        "attainable_improvement": np.nan,
        "simple_policy_improvement_recovered": np.nan,
        "counterexample_flag": (~joined.rate_limit_binding) & (~joined.within_material_tolerance),
        "evidence_source": "results/working/static_dynamic_ranking_diagnostics.csv;results/working/rate_limiter_mechanism_audit.csv",
    })
    output.to_csv(FINAL / "coupling_labelled_regret_audit.csv", index=False)
    summary = output.groupby(["dataset", "mechanism_status"], as_index=False).agg(
        units=("unit_id", "size"),
        within_material_tolerance_rate=("within_material_tolerance", "mean"),
        median_regret=("dynamic_regret", "median"),
        p90_regret=("dynamic_regret", lambda x: float(x.quantile(0.9))),
        maximum_regret=("dynamic_regret", "max"),
        execution_agreement=("execution_agreement", "mean"),
        counterexamples=("counterexample_flag", "sum"),
    )
    return output, summary


def _describe_effect(group: pd.DataFrame, effect: str) -> dict[str, object]:
    values = group[effect].dropna()
    low, high = unit_bootstrap_ci(values)
    return {
        "units": len(group),
        "mean_effect": float(values.mean()) if len(values) else np.nan,
        "median_effect": float(values.median()) if len(values) else np.nan,
        "q25_effect": float(values.quantile(0.25)) if len(values) else np.nan,
        "q75_effect": float(values.quantile(0.75)) if len(values) else np.nan,
        "p90_effect": float(values.quantile(0.90)) if len(values) else np.nan,
        "ci_lower_95": low,
        "ci_upper_95": high,
        "execution_path_difference_rate": float((~group.execution_agreement).mean()),
        "materially_favorable_rate": float(group.materially_favorable.mean()),
        "materially_adverse_rate": float(group.materially_adverse.mean()),
    }


def activation_to_gain() -> tuple[pd.DataFrame, pd.DataFrame]:
    rate = pd.read_csv(WORKING / "rate_limiter_mechanism_audit.csv")
    rate_rows = pd.DataFrame({
        "dataset": rate.dataset,
        "unit_id": rate.planning_unit,
        "regime_id": rate.transfer_regime,
        "capability": "ensemble_rate_limiting",
        "mechanism": "rate_cap_activation",
        "mechanism_status": np.where(rate.rate_limit_binding, "active", "inactive"),
        "activation_measure": rate.activation_frequency,
        "reference_cost": rate.dynamic_cost_plain,
        "capability_cost": rate.dynamic_cost_limited,
        "paired_cost_effect": rate.dynamic_cost_difference,
        "switch_reduction": np.nan,
        "execution_agreement": rate.executed_plan_agreement.astype(bool),
        "evidence_source": "results/working/rate_limiter_mechanism_audit.csv",
    })
    governance = pd.read_csv(WORKING / "switch_budget_mechanism_audit.csv")
    stress = pd.read_csv(WORKING / "dynamic_policy_stress_results.csv")
    stress_cost = stress.pivot_table(
        index=["dataset", "scenario_id", "planning_unit"], columns="strategy", values="total_dynamic_cost"
    ).reset_index()
    gov_rows = governance.merge(stress_cost, on=["dataset", "scenario_id", "planning_unit"])
    gov_rows = pd.DataFrame({
        "dataset": gov_rows.dataset,
        "unit_id": gov_rows.planning_unit,
        "regime_id": gov_rows.scenario_id,
        "capability": "governance_aware_dp",
        "mechanism": "switch_budget_binding",
        "mechanism_status": np.where(gov_rows.switch_budget_binding, "active", "inactive"),
        "activation_measure": gov_rows.unconstrained_dp_switch_count - gov_rows.selected_budget_K,
        "reference_cost": gov_rows.dp,
        "capability_cost": gov_rows.budgeted_dp,
        "paired_cost_effect": gov_rows.dynamic_cost_difference,
        "switch_reduction": gov_rows.switch_reduction,
        "execution_agreement": gov_rows.executed_plan_agreement.astype(bool),
        "evidence_source": "results/working/switch_budget_mechanism_audit.csv",
    })
    periods = pd.read_csv(WORKING / "dynamic_policy_period_results.csv", low_memory=False)
    cost = stress_cost.copy()
    horizon_records = []
    mechanism_by_scenario = {
        "long_lead": "pipeline_carryover",
        "persistent_backlog": "backlog_persistence",
        "frozen_horizon": "frozen_action_restriction",
        "high_switch_cost": "switching_sequence_dependence",
    }
    for (dataset, scenario, unit), group in periods[
        periods.scenario_id.isin(mechanism_by_scenario)
    ].groupby(["dataset", "scenario_id", "planning_unit"]):
        dp = group[group.strategy.eq("dp")].sort_values("date")
        simple_path = group[group.strategy.eq("simple_ensemble_rate_limited")].sort_values("date")
        row = cost[(cost.dataset == dataset) & (cost.scenario_id == scenario) & (cost.planning_unit == unit)].iloc[0]
        mechanism = mechanism_by_scenario[scenario]
        if mechanism == "pipeline_carryover":
            activation = float((dp.on_order_end > TOL).mean())
            measurable = True
        elif mechanism == "backlog_persistence":
            activation = float(((dp.backlog_end.shift(1).fillna(0) > TOL) & (dp.backlog_end > TOL)).mean())
            measurable = True
        elif mechanism == "switching_sequence_dependence":
            activation = float((dp.model_switch > TOL).mean())
            measurable = True
        else:
            activation = np.nan
            measurable = False
        agreement = bool(np.allclose(dp.order_quantity, simple_path.order_quantity, rtol=0, atol=TOL))
        horizon_records.append({
            "dataset": dataset,
            "unit_id": unit,
            "regime_id": scenario,
            "capability": "horizon_aware_dp",
            "mechanism": mechanism,
            "mechanism_status": "active" if measurable and activation > 0 else ("inactive" if measurable else "unmeasured"),
            "activation_measure": activation,
            "reference_cost": row.simple_ensemble_rate_limited,
            "capability_cost": row.dp,
            "paired_cost_effect": row.dp - row.simple_ensemble_rate_limited,
            "switch_reduction": np.nan,
            "execution_agreement": agreement,
            "evidence_source": "results/working/dynamic_policy_period_results.csv",
        })
    unit = pd.concat([rate_rows, gov_rows, pd.DataFrame(horizon_records)], ignore_index=True)
    threshold = np.maximum(TOL, 0.01 * unit.reference_cost.abs())
    unit["materially_favorable"] = unit.paired_cost_effect < -threshold
    unit["materially_adverse"] = unit.paired_cost_effect > threshold
    unit["counterexample_flag"] = (
        unit.mechanism_status.eq("active") & ~unit.materially_favorable
    ) | (unit.mechanism_status.eq("inactive") & (unit.paired_cost_effect.abs() > threshold))
    unit.to_csv(FINAL / "activation_to_gain_unit_results.csv", index=False)
    records = []
    for keys, group in unit.groupby(["dataset", "capability", "mechanism", "mechanism_status"]):
        records.append({
            "dataset": keys[0], "capability": keys[1], "mechanism": keys[2], "mechanism_status": keys[3],
            **_describe_effect(group, "paired_cost_effect"),
            "mean_switch_reduction": float(group.switch_reduction.mean()) if group.switch_reduction.notna().any() else np.nan,
            "counterexamples": int(group.counterexample_flag.sum()),
        })
    summary = pd.DataFrame(records)
    contrasts = []
    for keys, group in unit.groupby(["dataset", "capability", "mechanism"]):
        means = group.groupby("mechanism_status").paired_cost_effect.mean()
        if "active" in means and "inactive" in means:
            contrasts.append({"dataset": keys[0], "capability": keys[1], "mechanism": keys[2], "active_minus_inactive_effect": float(means.active - means.inactive)})
    contrast = pd.DataFrame(contrasts)
    summary = summary.merge(contrast, on=["dataset", "capability", "mechanism"], how="left")
    summary.to_csv(FINAL / "activation_to_gain_summary.csv", index=False)
    return unit, summary


def _panel_inputs() -> dict[str, pd.DataFrame]:
    config = load_config(ROOT / "configs" / "default.yaml")
    m5 = m5_panels(100, config)["item_store"]
    walmart, _ = walmart_panel(100, config)
    return {"M5": m5, "Walmart": walmart}


def _summarize_sweep(unit: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in unit.groupby(["dataset", "mechanism", "level", "strength"]):
        low, high = unit_bootstrap_ci(group.paired_difference)
        records.append({
            "dataset": keys[0], "mechanism": keys[1], "level": keys[2], "strength": keys[3],
            "units": len(group), "simple_reference_cost": float(group.simple_reference_cost.mean()),
            "approximate_dp_cost": float(group.approximate_dp_cost.mean()),
            "mean_paired_difference": float(group.paired_difference.mean()),
            "median_paired_difference": float(group.paired_difference.median()),
            "mean_percentage_difference": float(group.percentage_difference.mean()),
            "execution_path_difference_rate": float(group.execution_path_difference.mean()),
            "ci_lower_95": low, "ci_upper_95": high, "adverse_units": int(group.adverse_unit.sum()),
        })
    summary = pd.DataFrame(records)
    correlation_records = []
    for keys, group in unit.groupby(["dataset", "mechanism", "unit_id"]):
        strength_rank = group.strength.rank(method="average").to_numpy(float)
        gain_rank = (-group.paired_difference).rank(method="average").to_numpy(float)
        correlation = float(np.corrcoef(strength_rank, gain_rank)[0, 1]) if np.std(gain_rank) > 0 else np.nan
        correlation_records.append({"dataset": keys[0], "mechanism": keys[1], "unit_id": keys[2], "unit_rank_correlation": correlation})
    correlations = pd.DataFrame(correlation_records)
    corr_summary = correlations.groupby(["dataset", "mechanism"], as_index=False).unit_rank_correlation.mean()
    summary = summary.merge(corr_summary, on=["dataset", "mechanism"], how="left")
    summary.to_csv(FINAL / "coupling_strength_sweep_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.45), sharey=False)
    for ax, mechanism in zip(axes, ["lead_time", "switching_cost"]):
        for dataset, group in summary[summary.mechanism.eq(mechanism)].groupby("dataset"):
            group = group.sort_values("strength")
            y = -group.mean_percentage_difference.to_numpy(float)
            ax.plot(group.strength, y, marker="o", label=dataset, color=COLORS[dataset])
        ax.axhline(0, color="#555555", linewidth=0.7)
        title = "Lead Time\nM5 graded; Walmart non-monotonic" if mechanism == "lead_time" else "Switching Cost\nNon-monotonic in both datasets"
        ax.set_title(title, fontsize=8.5, pad=1.5)
        ax.set_xlabel("Lead time" if mechanism == "lead_time" else "Switching-cost coefficient")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=7.8)
        ax.grid(color="#D9E1E5", linewidth=0.5)
    axes[0].legend(frameon=False)
    fig.text(0.5, 0.015, "Approximate-DP cost reduction vs. simple reference (%)", ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(bottom=0.23, top=0.86, wspace=0.28)
    save_figure(fig, "coupling_strength_vs_policy_value", manuscript=True)
    return summary


def coupling_strength_sweep() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple, dict]]:
    checkpoint = FINAL / "coupling_strength_sweep_unit.csv"
    if checkpoint.exists():
        unit = pd.read_csv(checkpoint)
        return unit, _summarize_sweep(unit), {}
    retained = list(WORKING.glob("*coupling*strength*.csv"))
    if retained:
        raise RuntimeError("Unexpected retained coupling-strength output; review before reuse: {}".format(retained))
    unit_rows = []
    trace_inputs: dict[tuple, dict] = {}
    sweep = {
        "lead_time": [("low", 0.0), ("medium", 2.0), ("high", 7.0)],
        "switching_cost": [("low", 0.0), ("medium", 0.05), ("high", 0.20)],
    }
    for dataset, panel in _panel_inputs().items():
        validation, test = panel[panel.split.eq("validation")], panel[panel.split.eq("test")]
        for unit, group in test.groupby("series_id"):
            pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
            actual_rows = group.drop_duplicates("date").sort_values("date")
            actual = actual_rows.actual.to_numpy(float)
            dates = actual_rows.date.to_numpy()
            val = validation[validation.series_id.eq(unit)].drop_duplicates("date").actual
            mean = max(float(val.mean()), 1.0)
            safety = 0.5 * float(val.std() or 0.0)
            candidates = {str(column): pivot[column].to_numpy(float) for column in pivot.columns}
            ensemble = pivot.mean(axis=1).to_numpy(float)
            for mechanism, levels in sweep.items():
                for level, strength in levels:
                    lead = int(strength) if mechanism == "lead_time" else 2
                    switch_cost = float(strength) if mechanism == "switching_cost" else 0.05
                    capacity = 1.25 * mean
                    policy_config = DynamicPolicyConfig(
                        lead_time=lead, shortage_cost_rate=5.0, switching_cost_rate=switch_cost,
                        capacity=capacity, max_order_change_rate=0.5, frozen_horizon=1,
                        backlog_persistence=1.0, state_rounding=max(mean * 0.1, 0.25), beam_width=96,
                    )
                    initial_inventory = (lead + 1) * mean
                    dp = select_dynamic_policy(candidates, safety, initial_inventory, mean, policy_config, "dp")
                    simple_plan = rate_limit_signal(ensemble * (lead + 1) + safety, 0.40)
                    sim_config = DynamicInventoryConfig(
                        lead_time=lead, holding_cost_rate=1.0, shortage_cost_rate=5.0, order_change_cost_rate=0.05, expedite_cost_rate=0.0,
                        replenishment_capacity=capacity, max_order_change_rate=0.5,
                        initial_on_hand=initial_inventory, initial_order_quantity=mean,
                    )
                    simple_sim = simulate_dynamic_inventory(actual, simple_plan, sim_config)
                    dp_sim = simulate_dynamic_inventory(actual, dp["plan"], sim_config, model_switch=dp["switch"], switching_cost_rate=switch_cost)
                    simple_cost = float(simple_sim.total_dynamic_cost.sum())
                    dp_cost = float(dp_sim.total_dynamic_cost.sum())
                    diff = dp_cost - simple_cost
                    path_diff = bool(not np.allclose(simple_sim.order_quantity, dp_sim.order_quantity, rtol=0, atol=TOL))
                    unit_rows.append({
                        "dataset": dataset, "unit_id": unit, "mechanism": mechanism, "level": level,
                        "strength": strength, "simple_reference_cost": simple_cost, "approximate_dp_cost": dp_cost,
                        "paired_difference": diff, "percentage_difference": np.nan if simple_cost <= TOL else 100 * diff / simple_cost,
                        "execution_path_difference": path_diff,
                        "adverse_unit": diff > max(TOL, 0.01 * simple_cost),
                    })
                    trace_inputs[(dataset, str(unit), mechanism, level)] = {
                        "dates": dates, "actual": actual, "candidates": candidates, "ensemble": ensemble,
                        "simple_plan": simple_plan, "simple_sim": simple_sim, "dp": dp, "dp_sim": dp_sim,
                        "lead": lead, "switch_cost": switch_cost, "capacity": capacity,
                    }
    unit = pd.DataFrame(unit_rows)
    unit.to_csv(FINAL / "coupling_strength_sweep_unit.csv", index=False)
    summary = _summarize_sweep(unit)
    return unit, summary, trace_inputs


def rebuild_trace_info(row: pd.Series) -> dict:
    panel = _panel_inputs()[row.dataset]
    validation, test = panel[panel.split.eq("validation")], panel[panel.split.eq("test")]
    group = test[test.series_id.astype(str).eq(str(row.unit_id))]
    pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
    actual_rows = group.drop_duplicates("date").sort_values("date")
    actual, dates = actual_rows.actual.to_numpy(float), actual_rows.date.to_numpy()
    val = validation[validation.series_id.astype(str).eq(str(row.unit_id))].drop_duplicates("date").actual
    mean, safety = max(float(val.mean()), 1.0), 0.5 * float(val.std() or 0.0)
    candidates = {str(column): pivot[column].to_numpy(float) for column in pivot.columns}
    ensemble = pivot.mean(axis=1).to_numpy(float)
    lead = int(row.strength) if row.mechanism == "lead_time" else 2
    switch_cost = float(row.strength) if row.mechanism == "switching_cost" else 0.05
    capacity = 1.25 * mean
    policy_config = DynamicPolicyConfig(
        lead_time=lead, shortage_cost_rate=5.0, switching_cost_rate=switch_cost,
        capacity=capacity, max_order_change_rate=0.5, frozen_horizon=1,
        backlog_persistence=1.0, state_rounding=max(mean * 0.1, 0.25), beam_width=96,
    )
    initial_inventory = (lead + 1) * mean
    dp = select_dynamic_policy(candidates, safety, initial_inventory, mean, policy_config, "dp")
    simple_plan = rate_limit_signal(ensemble * (lead + 1) + safety, 0.40)
    sim_config = DynamicInventoryConfig(
        lead_time=lead, holding_cost_rate=1.0, shortage_cost_rate=5.0, order_change_cost_rate=0.05, expedite_cost_rate=0.0,
        replenishment_capacity=capacity, max_order_change_rate=0.5,
        initial_on_hand=initial_inventory, initial_order_quantity=mean,
    )
    simple_sim = simulate_dynamic_inventory(actual, simple_plan, sim_config)
    dp_sim = simulate_dynamic_inventory(actual, dp["plan"], sim_config, model_switch=dp["switch"], switching_cost_rate=switch_cost)
    return {"dates": dates, "actual": actual, "candidates": candidates, "ensemble": ensemble, "simple_plan": simple_plan, "simple_sim": simple_sim, "dp": dp, "dp_sim": dp_sim, "lead": lead, "switch_cost": switch_cost, "capacity": capacity}


def static_gate_outputs(regret: pd.DataFrame) -> pd.DataFrame:
    predictions = regret[["dataset", "unit_id", "dynamic_regret"]].copy()
    predictions["eligible_validation_only_features"] = False
    predictions["frozen_threshold"] = np.nan
    predictions["gate_prediction"] = "not_issued"
    predictions["reason"] = "Available static proxy uses held-out demand and execution outcomes"
    predictions.to_csv(FINAL / "static_gate_predictions.csv", index=False)
    metrics = []
    for dataset, group in predictions.groupby("dataset"):
        metrics.append({
            "dataset": dataset, "status": "not_estimable_without_leakage", "frozen_threshold": np.nan,
            "material_regret_threshold": "1% of best dynamic cost and >1e-6",
            "false_simplification_rate": np.nan, "false_escalation_rate": np.nan,
            "precision": np.nan, "recall": np.nan, "negative_predictive_value": np.nan,
            "simplified_units": 0, "simplified_regret_p90": np.nan, "simplified_regret_max": np.nan,
        })
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(FINAL / "static_gate_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    data = [predictions[predictions.dataset.eq(d)].dynamic_regret.to_numpy(float) for d in ["M5", "Walmart"]]
    ax.boxplot(data, labels=["M5", "Walmart"], showfliers=True)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Held-out dynamic regret (label only)")
    ax.set_title("No leakage-safe static gate can be estimated from retained features")
    ax.grid(axis="y", color="#D9E1E5", linewidth=0.5)
    save_figure(fig, "static_gate_tradeoff")
    return metrics


def _trace_frame(info: dict, policy: str) -> pd.DataFrame:
    sim = info["simple_sim"] if policy == "simple" else info["dp_sim"]
    selected = np.repeat("simple_ensemble", len(sim)) if policy == "simple" else info["dp"]["model"]
    forecasts = info["ensemble"] if policy == "simple" else info["dp"]["forecast"]
    candidate_json = [json.dumps({name: float(values[i]) for name, values in info["candidates"].items()}, sort_keys=True) for i in range(len(sim))]
    frame = pd.DataFrame({
        "period": sim.period_index,
        "date": info["dates"],
        "actual_demand": info["actual"],
        "candidate_forecasts": candidate_json,
        "selected_forecast_model": selected,
        "selected_forecast": forecasts,
        "desired_order": sim.desired_order,
        "executed_order": sim.order_quantity,
        "inventory": sim.on_hand_end,
        "backlog": sim.backlog_end,
        "on_order_pipeline": sim.on_order_end,
        "capacity": info["capacity"],
        "switch_event": sim.model_switch,
        "rate_limit_activation": np.nan,
        "governance_binding": False,
        "period_cost": sim.total_dynamic_cost,
        "cumulative_cost": sim.total_dynamic_cost.cumsum(),
        "policy": policy,
    })
    return frame


def representative_traces(unit_sweep: pd.DataFrame, trace_inputs: dict[tuple, dict]) -> pd.DataFrame:
    weak_candidates = unit_sweep[(unit_sweep.mechanism.eq("switching_cost")) & (unit_sweep.level.eq("low"))].copy()
    weak_candidates["abs_effect"] = weak_candidates.paired_difference.abs()
    weak_row = weak_candidates.sort_values(["execution_path_difference", "abs_effect"]).iloc[0]
    active_candidates = unit_sweep[(unit_sweep.mechanism.eq("lead_time")) & (unit_sweep.level.eq("high")) & (~unit_sweep.adverse_unit)].copy()
    active_candidates = active_candidates[active_candidates.execution_path_difference]
    active_row = active_candidates.sort_values("paired_difference").iloc[0]
    selections = [("weak", weak_row), ("active", active_row)]
    manifest = []
    for case, row in selections:
        info = trace_inputs.get((row.dataset, str(row.unit_id), row.mechanism, row.level))
        if info is None:
            info = rebuild_trace_info(row)
        frames = [_trace_frame(info, "simple"), _trace_frame(info, "approximate_dp")]
        trace = pd.concat(frames, ignore_index=True)
        trace.to_csv(FINAL / f"trace_case_{case}.csv", index=False)
        manifest.append({
            "case": case, "dataset": row.dataset, "unit_id": row.unit_id, "mechanism": row.mechanism,
            "level": row.level, "strength": row.strength, "simple_cost": row.simple_reference_cost,
            "advanced_cost": row.approximate_dp_cost, "paired_difference": row.paired_difference,
            "execution_path_difference": row.execution_path_difference,
            "source": "scripts/run_targeted_north_star_analyses.py minimal frozen sweep",
        })
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.7), sharex=True)
        for policy, group in trace.groupby("policy"):
            axes[0].plot(group.period, group.executed_order, label=policy)
            axes[1].plot(group.period, group.cumulative_cost, label=policy)
        axes[0].plot(frames[0].period, frames[0].actual_demand, color="#555555", linestyle="--", label="actual demand")
        axes[0].set_ylabel("Units")
        axes[0].set_title(f"{case.title()} trace: {row.dataset} {row.unit_id}")
        axes[1].set_ylabel("Cumulative comparative cost")
        axes[1].set_xlabel("Period")
        for ax in axes:
            ax.grid(color="#D9E1E5", linewidth=0.5)
            ax.legend(frameon=False, ncol=3)
        save_figure(fig, f"trace_case_{case}")
    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(FINAL / "trace_case_manifest.csv", index=False)
    return manifest_frame


def write_reports(regret_summary: pd.DataFrame, activation_summary: pd.DataFrame, sweep_summary: pd.DataFrame, gate: pd.DataFrame, traces: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        def format_value(value: object) -> str:
            if pd.isna(value):
                return "NA"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4f}"
            return str(value).replace("|", "\\|")
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        lines.extend("| " + " | ".join(format_value(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
        return "\n".join(lines)

    inactive = regret_summary[regret_summary.mechanism_status.eq("inactive")]
    (REPORTS / "coupling_labelled_regret_audit.md").write_text(
        "# Coupling-Labelled Regret Audit\n\n"
        "The only implementation-grounded activation label shared with the 200-unit static diagnostic is ensemble rate-cap activation. "
        "The diagnostic contains no advanced-policy cost, so attainable-value recovery is not estimable and no percentage-of-attainable-value claim is made.\n\n"
        + markdown_table(regret_summary)
        + "\n\nThe predeclared weak/inactive criterion is evaluated separately by dataset. Even if the 80% tolerance condition holds, this result is mechanism-specific and cannot establish general weak-coupling sufficiency. Tail regret and all counterexamples remain in the canonical CSV.\n",
        encoding="utf-8",
    )
    (REPORTS / "activation_to_gain_audit.md").write_text(
        "# Activation-to-Gain Audit\n\n"
        "This audit uses observed target changes, unconstrained switch-budget violations, positive pipeline/backlog carryover, and observed switching. Frozen-horizon activation remains unmeasured because the retained rollout lacks the unrestricted within-window counterfactual. Negative cost effects favor the capability.\n\n"
        + markdown_table(activation_summary)
        + "\n\nAn active mechanism predicts that a capability can change execution, not that it must reduce operating cost. Governance is therefore reported as switch enforcement with a cost trade-off. Counterexamples are flagged at unit level.\n",
        encoding="utf-8",
    )
    relationship = []
    for keys, group in sweep_summary.groupby(["dataset", "mechanism"]):
        ordered = group.sort_values("strength").mean_percentage_difference.to_numpy(float)
        if np.all(np.diff(ordered) <= 0):
            label = "monotonic"
        elif abs(ordered[0]) <= 1.0 and np.any(ordered[1:] < -1.0):
            label = "threshold-like"
        elif np.any(np.diff(ordered) > 0) and np.any(np.diff(ordered) < 0):
            label = "non-monotonic"
        else:
            label = "no reliable relationship"
        relationship.append({"dataset": keys[0], "mechanism": keys[1], "classification": label})
    relationship = pd.DataFrame(relationship)
    (REPORTS / "coupling_strength_trend_audit.md").write_text(
        "# Coupling-Strength Trend Audit\n\n"
        "Forecasts and policy settings were frozen; only lead time or switching cost changed across the three predeclared levels. Negative paired differences favor approximate DP.\n\n"
        + markdown_table(sweep_summary)
        + "\n\n## Required classification\n\n" + markdown_table(relationship)
        + "\n\nA descriptive monotonic pattern does not authorize `governs` unless uncertainty is separated from zero under the protocol's dataset-specific rule.\n",
        encoding="utf-8",
    )
    (REPORTS / "static_gate_validation.md").write_text(
        "# Static Complexity-Gate Validation\n\n"
        "**Verdict: FAIL as an automated gate.** The retained static proxy is computed with held-out demand and execution violations. It is a valid ex-post diagnostic but an ineligible deployment-time feature. No threshold or prediction was issued, so false-simplification, false-escalation, precision, recall, and negative predictive value are not estimable without leakage.\n\n"
        + markdown_table(gate)
        + "\n\nAuthorized wording: *static screening can support shortlist formation but cannot safely automate escalation.*\n",
        encoding="utf-8",
    )
    (REPORTS / "representative_trace_analysis.md").write_text(
        "# Representative Trace Analysis\n\n"
        "The two cases are deterministic exports from the frozen minimal sweep, not field deployments or realized business savings. Candidate forecasts, selected model, desired and executed orders, state, activation fields, and comparative costs are stored period by period.\n\n"
        + markdown_table(traces)
        + "\n\nThe weak case is selected before narration as the smallest absolute effect at zero switching cost, prioritizing execution agreement. The active case is the largest favorable high-lead-time effect among units whose executed paths differ. These are representative mechanism reconstructions, not claims of population prevalence.\n",
        encoding="utf-8",
    )


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    regret, regret_summary = coupling_labelled_regret()
    _, activation_summary = activation_to_gain()
    unit_sweep, sweep_summary, trace_inputs = coupling_strength_sweep()
    gate = static_gate_outputs(regret)
    traces = representative_traces(unit_sweep, trace_inputs)
    write_reports(regret_summary, activation_summary, sweep_summary, gate, traces)
    print("Wrote targeted North Star analyses and reports.")


if __name__ == "__main__":
    main()
