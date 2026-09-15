"""Audit structural zeros and binding mechanisms in IEOM dynamic results."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "working"
TOL = 1e-9


METRICS = {
    "dynamic_cost": "total_dynamic_cost",
    "PEG": "planning_execution_gap_rate",
    "volatility": "plan_volatility",
    "fill_rate": "fill_rate",
    "backlog": "ending_backlog",
}


def _unit_metric(period: pd.DataFrame, metric: str) -> float:
    if metric == "planning_execution_gap_rate":
        return float(period["execution_violation"].mean())
    if metric == "fill_rate":
        demand = float(period["demand"].sum())
        return 1.0 if demand == 0 else float(period["current_demand_fulfilled"].sum() / demand)
    if metric == "ending_backlog":
        return float(period.sort_values("date")["backlog_end"].iloc[-1])
    if metric == "plan_volatility":
        return float(period["order_change"].sum())
    return float(period[metric].sum())


def rate_limiter_audit(periods: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "grain", "transfer_regime", "planning_unit"]
    subset = periods[periods.strategy.isin(["simple_ensemble", "simple_ensemble_rate_limited"])].copy()
    records = []
    for values, group in subset.groupby(keys):
        plain = group[group.strategy == "simple_ensemble"].sort_values("date")
        limited = group[group.strategy == "simple_ensemble_rate_limited"].sort_values("date")
        if len(plain) != len(limited):
            continue
        target_diff = limited.order_up_to_target.to_numpy(float) - plain.order_up_to_target.to_numpy(float)
        order_diff = limited.order_quantity.to_numpy(float) - plain.order_quantity.to_numpy(float)
        cost_difference = float(limited.total_dynamic_cost.sum() - plain.total_dynamic_cost.sum())
        binding = bool(np.any(np.abs(target_diff) > TOL))
        executed_agreement = bool(np.all(np.abs(order_diff) <= TOL))
        record = dict(zip(keys, values))
        record.update({
            "constraint_activated": binding,
            "rate_limit_binding": binding,
            "switch_budget_binding": False,
            "path_agreement": True,
            "executed_plan_agreement": executed_agreement,
            "unrounded_cost_difference": cost_difference,
            "structural_equivalence": bool(executed_agreement and abs(cost_difference) <= TOL),
            "tie_flag": bool(executed_agreement and abs(cost_difference) <= TOL),
            "activation_frequency": float(np.mean(np.abs(target_diff) > TOL)),
        })
        for label, metric in METRICS.items():
            record[f"{label}_plain"] = _unit_metric(plain, metric)
            record[f"{label}_limited"] = _unit_metric(limited, metric)
            record[f"{label}_difference"] = record[f"{label}_limited"] - record[f"{label}_plain"]
        records.append(record)
    audit = pd.DataFrame(records)
    group_keys = ["dataset", "grain", "transfer_regime"]
    counts = audit.groupby(group_keys).rate_limit_binding.agg(n_binding_units="sum", n_total_units="size").reset_index()
    counts["activation_rate"] = counts.n_binding_units / counts.n_total_units
    binding_effect = audit[audit.rate_limit_binding].groupby(group_keys).unrounded_cost_difference.mean().rename("mean_effect_when_binding").reset_index()
    audit = audit.merge(counts, on=group_keys).merge(binding_effect, on=group_keys, how="left")
    summaries = []
    for keys, group in audit.groupby(group_keys):
        for slice_name, selected in (
            ("all_units", group),
            ("binding_units", group[group.rate_limit_binding]),
            ("nonbinding_units", group[~group.rate_limit_binding]),
        ):
            if selected.empty:
                continue
            row = dict(zip(group_keys, keys)); row["unit_slice"] = slice_name
            row.update({
                "constraint_activated": bool(selected.rate_limit_binding.any()),
                "rate_limit_binding": bool(selected.rate_limit_binding.any()),
                "switch_budget_binding": False,
                "path_agreement": bool(selected.path_agreement.all()),
                "executed_plan_agreement": float(selected.executed_plan_agreement.mean()),
                "n_binding_units": int(group.rate_limit_binding.sum()),
                "n_total_units": int(len(group)),
                "activation_rate": float(group.rate_limit_binding.mean()),
                "mean_effect_when_binding": float(group.loc[group.rate_limit_binding, "unrounded_cost_difference"].mean()) if group.rate_limit_binding.any() else np.nan,
                "unrounded_cost_difference": float(selected.unrounded_cost_difference.mean()),
                "structural_equivalence": bool(selected.structural_equivalence.all()),
                "tie_flag": bool(selected.tie_flag.all()),
                "activation_frequency": float(selected.activation_frequency.mean()),
            })
            for label in METRICS:
                row[f"{label}_plain"] = float(selected[f"{label}_plain"].mean())
                row[f"{label}_limited"] = float(selected[f"{label}_limited"].mean())
                row[f"{label}_difference"] = float(selected[f"{label}_difference"].mean())
            summaries.append(row)
    return audit, pd.DataFrame(summaries)


def switch_budget_audit(summary: pd.DataFrame, periods: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare independently executed unconstrained and budgeted DP branches."""
    keys = ["dataset", "scenario_id", "planning_unit"]
    outcome_columns = [
        "total_holding_cost", "total_shortage_cost", "total_order_change_cost",
        "total_expedite_cost", "total_switching_cost", "fill_rate",
        "average_on_hand_inventory", "ending_backlog", "total_lost_sales",
        "execution_violation_units", "planning_execution_gap_rate", "plan_volatility",
    ]
    records = []
    for values, group in periods[periods.strategy.isin(["dp", "budgeted_dp"])].groupby(keys):
        dp = group[group.strategy == "dp"].sort_values("date")
        budgeted = group[group.strategy == "budgeted_dp"].sort_values("date")
        if len(dp) != len(budgeted):
            continue
        dp_row = summary[(summary.dataset == values[0]) & (summary.scenario_id == values[1]) & (summary.planning_unit == values[2]) & (summary.strategy == "dp")].iloc[0]
        budget_row = summary[(summary.dataset == values[0]) & (summary.scenario_id == values[1]) & (summary.planning_unit == values[2]) & (summary.strategy == "budgeted_dp")].iloc[0]
        k = int(budget_row.selected_budget_K)
        path_agreement = bool((dp.selected_model.to_numpy(str) == budgeted.selected_model.to_numpy(str)).all())
        executed_agreement = bool(np.allclose(dp.order_quantity, budgeted.order_quantity, rtol=0.0, atol=TOL))
        cost_difference = float(budgeted.total_dynamic_cost.sum() - dp.total_dynamic_cost.sum())
        expected_nonbinding = bool(dp.model_switch.sum() <= k)
        expected_binding = not expected_nonbinding
        switch_count_agreement = bool(abs(dp.model_switch.sum() - budgeted.model_switch.sum()) <= TOL)
        cost_agreement = bool(abs(cost_difference) <= TOL)
        outcome_differences = {
            column: float(budget_row[column] - dp_row[column]) for column in outcome_columns
        }
        outcomes_agree = bool(all(abs(value) <= TOL for value in outcome_differences.values()))
        exact_cost_tie = bool(abs(cost_difference) <= TOL)
        record = dict(zip(keys, values))
        record.update({
            "selected_budget_K": k,
            "budget_rule": budget_row.budget_rule,
            "unconstrained_dp_switch_count": float(dp.model_switch.sum()),
            "budgeted_dp_switch_count": float(budgeted.model_switch.sum()),
            "switch_reduction": float(dp.model_switch.sum() - budgeted.model_switch.sum()),
            "expected_nonbinding": expected_nonbinding,
            "expected_binding": expected_binding,
            "unconstrained_path_exceedance": expected_binding,
            "classification_basis": "unconstrained_switch_count > K",
            "constraint_activated": expected_binding,
            "rate_limit_binding": False,
            "switch_budget_binding": expected_binding,
            "path_agreement": path_agreement,
            "model_path_agreement": path_agreement,
            "executed_plan_agreement": executed_agreement,
            "switch_count_agreement": switch_count_agreement,
            "unrounded_cost_agreement": cost_agreement,
            "operational_outcomes_agreement": outcomes_agree,
            "dynamic_cost_difference": cost_difference,
            "governance_cost_difference": float(budgeted.switching_cost.sum() - dp.switching_cost.sum()),
            "unrounded_cost_difference": cost_difference,
            "structural_equivalence": bool(path_agreement and abs(cost_difference) <= TOL),
            "tie_flag": exact_cost_tie,
            "beam_width": 96,
            "state_rounding_rule": "0.10*validation_mean; floor=0.25",
            "candidate_ordering": "sorted model names",
            "tie_breaking": "deterministic accumulated cost then model path",
            "implementation_branch_difference": "switch-budget feasibility filter only",
            "numerical_tolerance": TOL,
        })
        record.update({f"{column}_difference": value for column, value in outcome_differences.items()})
        if path_agreement and executed_agreement and switch_count_agreement and cost_agreement and outcomes_agree:
            record["disagreement_diagnostic"] = "none"
        elif not path_agreement and cost_agreement:
            record["disagreement_diagnostic"] = (
                "exact-cost alternative path; candidate ordering is identical and deterministic "
                "tie-breaking acts on branch-specific retained beams"
            )
        elif not path_agreement:
            record["disagreement_diagnostic"] = (
                "branch-specific beam-pruning path: the budget feasibility filter changes the "
                "retained approximate state set despite a feasible unconstrained terminal path"
            )
        elif not executed_agreement:
            record["disagreement_diagnostic"] = "execution divergence despite model-path agreement; inspect branch inputs and tolerance"
        elif not switch_count_agreement:
            record["disagreement_diagnostic"] = "switch-count derivation mismatch"
        elif not cost_agreement or not outcomes_agree:
            record["disagreement_diagnostic"] = "numerical or operational-accounting difference"
        records.append(record)
    audit = pd.DataFrame(records)
    group_keys = ["dataset", "scenario_id"]
    counts = audit.groupby(group_keys).expected_binding.agg(n_expected_binding_units="sum", n_total_units="size").reset_index()
    counts["expected_binding_rate"] = counts.n_expected_binding_units / counts.n_total_units
    effects = audit[audit.expected_binding].groupby(group_keys).unrounded_cost_difference.mean().rename("mean_effect_when_expected_binding").reset_index()
    audit = audit.merge(counts, on=group_keys).merge(effects, on=group_keys, how="left")
    # Backward-compatible aliases remain explicit derivatives of the theoretical
    # classification; manuscript claims use the expected-binding names above.
    audit["n_binding_units"] = audit["n_expected_binding_units"]
    audit["activation_rate"] = audit["expected_binding_rate"]
    audit["mean_effect_when_binding"] = audit["mean_effect_when_expected_binding"]
    summaries = audit.groupby(group_keys, as_index=False).agg(
        selected_budget_K=("selected_budget_K", "first"),
        budget_rule=("budget_rule", "first"),
        unconstrained_switch_min=("unconstrained_dp_switch_count", "min"),
        unconstrained_switch_q25=("unconstrained_dp_switch_count", lambda x: float(np.quantile(x, .25))),
        unconstrained_switch_median=("unconstrained_dp_switch_count", "median"),
        unconstrained_switch_q75=("unconstrained_dp_switch_count", lambda x: float(np.quantile(x, .75))),
        unconstrained_switch_max=("unconstrained_dp_switch_count", "max"),
        constraint_activated=("constraint_activated", "any"),
        rate_limit_binding=("rate_limit_binding", "any"),
        switch_budget_binding=("switch_budget_binding", "any"),
        expected_nonbinding_count=("expected_nonbinding", "sum"),
        expected_nonbinding_rate=("expected_nonbinding", "mean"),
        expected_binding_count=("expected_binding", "sum"),
        expected_binding_rate=("expected_binding", "mean"),
        unconstrained_path_exceedance_rate=("unconstrained_path_exceedance", "mean"),
        path_agreement=("path_agreement", "mean"),
        model_path_agreement=("model_path_agreement", "mean"),
        executed_plan_agreement=("executed_plan_agreement", "mean"),
        switch_count_agreement=("switch_count_agreement", "mean"),
        unrounded_cost_agreement=("unrounded_cost_agreement", "mean"),
        operational_outcomes_agreement=("operational_outcomes_agreement", "mean"),
        n_expected_binding_units=("expected_binding", "sum"),
        n_binding_units=("expected_binding", "sum"),
        n_total_units=("switch_budget_binding", "size"),
        activation_rate=("expected_binding", "mean"),
        mean_effect_when_expected_binding=("mean_effect_when_expected_binding", "first"),
        mean_effect_when_binding=("mean_effect_when_expected_binding", "first"),
        switch_reduction=("switch_reduction", "mean"),
        dynamic_cost_difference=("dynamic_cost_difference", "mean"),
        governance_cost_difference=("governance_cost_difference", "mean"),
        unrounded_cost_difference=("unrounded_cost_difference", "mean"),
        structural_equivalence=("structural_equivalence", "all"),
        tie_flag=("tie_flag", "all"),
    )
    return audit, summaries


def nonbinding_governance_report(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize independent agreement only within theoretically nonbinding cases."""
    selected = audit[audit.expected_nonbinding].copy()
    agreement_columns = [
        "model_path_agreement", "executed_plan_agreement", "switch_count_agreement",
        "unrounded_cost_agreement", "operational_outcomes_agreement",
    ]
    groupings = [([], "all"), (["dataset"], "dataset"), (["dataset", "scenario_id", "budget_rule"], "regime_cap")]
    rows = []
    for keys, level in groupings:
        grouped = [((), selected)] if not keys else selected.groupby(keys, dropna=False)
        for values, group in grouped:
            values = values if isinstance(values, tuple) else (values,)
            row = {"summary_level": level, "expected_nonbinding_comparisons": len(group)}
            row.update(dict(zip(keys, values)))
            for column in agreement_columns:
                row[f"independent_{column}_rate"] = float(group[column].mean()) if len(group) else np.nan
            rows.append(row)
    disagreement_mask = ~selected[agreement_columns].all(axis=1)
    disagreements = selected[disagreement_mask].copy()
    return pd.DataFrame(rows), disagreements


def update_relevant_tables(rate_audit: pd.DataFrame, switch_audit: pd.DataFrame) -> None:
    audit_columns = ["constraint_activated", "rate_limit_binding", "switch_budget_binding", "path_agreement", "executed_plan_agreement", "n_binding_units", "n_total_units", "activation_rate", "mean_effect_when_binding", "unrounded_cost_difference", "structural_equivalence", "tie_flag"]
    transfer_path = OUT / "cross_dataset_frozen_transfer.csv"
    transfer = pd.read_csv(transfer_path)
    transfer = transfer.drop(columns=[column for column in audit_columns if column in transfer.columns])
    rate_fields = rate_audit[["dataset", "grain", "transfer_regime", "planning_unit", "constraint_activated", "rate_limit_binding", "switch_budget_binding", "path_agreement", "executed_plan_agreement", "n_binding_units", "n_total_units", "activation_rate", "mean_effect_when_binding", "unrounded_cost_difference", "structural_equivalence", "tie_flag"]]
    transfer = transfer.merge(rate_fields, on=["dataset", "grain", "transfer_regime", "planning_unit"], how="left")
    transfer.to_csv(transfer_path, index=False)
    cross_period_path = OUT / "cross_dataset_period_results.csv"
    cross_period = pd.read_csv(cross_period_path, low_memory=False)
    cross_period = cross_period.drop(columns=[column for column in audit_columns if column in cross_period.columns])
    cross_period = cross_period.merge(rate_fields, on=["dataset", "grain", "transfer_regime", "planning_unit"], how="left")
    cross_period.to_csv(cross_period_path, index=False)

    dynamic_path = OUT / "dynamic_policy_stress_results.csv"
    dynamic = pd.read_csv(dynamic_path)
    dynamic = dynamic.drop(columns=[column for column in audit_columns if column in dynamic.columns])
    switch_fields = switch_audit[["dataset", "scenario_id", "planning_unit", "constraint_activated", "rate_limit_binding", "switch_budget_binding", "path_agreement", "executed_plan_agreement", "n_binding_units", "n_total_units", "activation_rate", "mean_effect_when_binding", "unrounded_cost_difference", "structural_equivalence", "tie_flag"]]
    dynamic = dynamic.merge(switch_fields, on=["dataset", "scenario_id", "planning_unit"], how="left")
    dynamic.loc[dynamic.strategy == "simple_ensemble_rate_limited", "runtime_seconds"] = np.nan
    dynamic.to_csv(dynamic_path, index=False)
    dynamic_period_path = OUT / "dynamic_policy_period_results.csv"
    dynamic_period = pd.read_csv(dynamic_period_path, low_memory=False)
    dynamic_period = dynamic_period.drop(columns=[column for column in audit_columns if column in dynamic_period.columns])
    dynamic_period = dynamic_period.merge(switch_fields, on=["dataset", "scenario_id", "planning_unit"], how="left")
    dynamic_period.to_csv(dynamic_period_path, index=False)


def mechanism_decomposition(transfer: pd.DataFrame) -> pd.DataFrame:
    metrics = ["total_dynamic_cost", "planning_execution_gap_rate", "plan_volatility", "fill_rate", "ending_backlog"]
    keys = ["dataset", "grain", "transfer_regime"]
    means = transfer.groupby(keys + ["strategy"], as_index=False)[metrics].mean()
    records = []
    comparisons = [
        ("ensemble_diversification", "simple_ensemble", "accuracy_first"),
        ("rate_limit_on_accuracy", "accuracy_rate_limited", "accuracy_first"),
        ("rate_limit_on_ensemble", "simple_ensemble_rate_limited", "simple_ensemble"),
        ("combined_vs_accuracy", "simple_ensemble_rate_limited", "accuracy_first"),
        ("weighted_ensemble_vs_accuracy", "operationally_weighted_ensemble", "accuracy_first"),
    ]
    for key_values, group in means.groupby(keys):
        indexed = group.set_index("strategy")
        for mechanism, treatment, comparator in comparisons:
            if treatment not in indexed.index or comparator not in indexed.index:
                continue
            row = dict(zip(keys, key_values)); row.update({"mechanism": mechanism, "treatment": treatment, "comparator": comparator})
            for metric in metrics:
                row[f"{metric}_difference"] = float(indexed.loc[treatment, metric] - indexed.loc[comparator, metric])
            records.append(row)
    return pd.DataFrame(records)



def main() -> None:
    cross = pd.read_csv(OUT / "cross_dataset_period_results.csv", parse_dates=["date"], low_memory=False)
    dynamic = pd.read_csv(OUT / "dynamic_policy_stress_results.csv")
    dynamic_period = pd.read_csv(OUT / "dynamic_policy_period_results.csv", parse_dates=["date"], low_memory=False)
    rate_audit, rate_summary = rate_limiter_audit(cross)
    switch_audit, switch_summary = switch_budget_audit(dynamic, dynamic_period)
    nonbinding_summary, nonbinding_disagreements = nonbinding_governance_report(switch_audit)
    decomposition = mechanism_decomposition(pd.read_csv(OUT / "cross_dataset_frozen_transfer.csv"))
    rate_audit.to_csv(OUT / "rate_limiter_mechanism_audit.csv", index=False)
    rate_summary.to_csv(OUT / "rate_limiter_mechanism_summary.csv", index=False)
    switch_audit.to_csv(OUT / "switch_budget_mechanism_audit.csv", index=False)
    switch_summary.to_csv(OUT / "switch_budget_mechanism_summary.csv", index=False)
    nonbinding_summary.to_csv(OUT / "nonbinding_governance_independent_summary.csv", index=False)
    nonbinding_disagreements.to_csv(OUT / "nonbinding_governance_disagreements.csv", index=False)
    decomposition.to_csv(OUT / "mechanism_decomposition_summary.csv", index=False)
    update_relevant_tables(rate_audit, switch_audit)
    print("Wrote mechanism audit and decomposition tables.")


if __name__ == "__main__":
    main()
