"""Planning utility functions that combine operational planning terms."""

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.stability_metrics import compute_absolute_plan_change


DEFAULT_PLANNING_LOSS_WEIGHTS = {
    "alpha_forecast": 0.0,
    "beta_inventory": 1.0,
    "lambda_volatility": 0.10,
    "lambda_execution": 0.10,
    "lambda_switch": 0.05,
}

ORACLE_GAP_GROUP_COLUMNS = [
    "dataset_name",
    "run_mode",
    "scenario_name",
    "grain_level",
    "intermittency_bucket",
    "evaluation_mode",
    "capacity_scenario",
    "scenario_id",
    "split_name",
    "weight_setting",
    "loss_weight_setting",
    "alpha_forecast",
    "beta_inventory",
    "lambda_volatility",
    "lambda_execution",
    "lambda_switch",
]


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric sequence into a one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional sequence.".format(name))
    return array


def _component_sum(values: Sequence[float], name: str) -> float:
    """Return the finite sum of a planning-loss component."""
    array = _as_float_array(values, name)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains non-finite values.".format(name))
    return float(np.sum(array))


def compute_execution_penalty(
    planning_signal: Sequence[float],
    execution_capacity: Sequence[float],
) -> np.ndarray:
    """Return the execution adaptation penalty for each period.

    The penalty is positive when the absolute plan change exceeds the execution
    capacity for that period. It directly measures the planning-infrastructure
    gap: the plan changes faster than the operation can absorb.

    This term makes execution constraints visible in the planning objective
    rather than treating them as an informal planner concern.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    capacity = _as_float_array(execution_capacity, "execution_capacity")
    if signal.shape != capacity.shape:
        raise ValueError("planning_signal and execution_capacity must have the same shape.")
    plan_change = compute_absolute_plan_change(signal)
    return np.maximum(plan_change - np.maximum(capacity, 0.0), 0.0)


def compute_total_planning_loss(
    forecast_error: Sequence[float],
    inventory_cost: Sequence[float],
    planning_signal_volatility: Sequence[float],
    model_switching_cost: Sequence[float],
    execution_adaptation_penalty: Sequence[float],
    weights: Mapping[str, float],
) -> float:
    """Return a weighted scalar planning loss.

    The default scalar objective combines inventory cost, plan volatility, model
    switching, and execution adaptation. Forecast error is accepted as an input
    for explicit sensitivity settings, but its default weight is zero because
    WAPE is reported separately as a diagnostic metric.

    The paper should not rely on this scalar alone. The scalar is a useful
    optimization device, while the individual components explain the tradeoffs
    created by the planning-infrastructure gap.
    """
    return float(
        planning_loss_weight(weights, "alpha_forecast") * _component_sum(forecast_error, "forecast_error")
        + planning_loss_weight(weights, "beta_inventory") * _component_sum(inventory_cost, "inventory_cost")
        + planning_loss_weight(weights, "lambda_volatility")
        * _component_sum(planning_signal_volatility, "planning_signal_volatility")
        + planning_loss_weight(weights, "lambda_switch") * _component_sum(model_switching_cost, "model_switching_cost")
        + planning_loss_weight(weights, "lambda_execution")
        * _component_sum(execution_adaptation_penalty, "execution_adaptation_penalty")
    )


def planning_loss_weight(weights: Mapping[str, float], name: str) -> float:
    """Return a planning-loss weight with repository-wide operational defaults."""
    return float((weights or {}).get(name, DEFAULT_PLANNING_LOSS_WEIGHTS[name]))


def compute_multi_objective_summary(
    forecast_error: Sequence[float],
    inventory_cost: Sequence[float],
    planning_signal_volatility: Sequence[float],
    model_switching_cost: Sequence[float],
    execution_adaptation_penalty: Sequence[float],
) -> Dict[str, float]:
    """Return Pareto-style objective components for a planning strategy.

    The summary reports each objective separately so the paper can compare
    strategies without hiding tradeoffs inside one scalar score.

    This is important for the planning-infrastructure gap because a model can
    improve forecast error while worsening stability or execution adaptation.
    Separate objective columns make that failure mode visible.
    """
    components = {
        "forecast_error_total": _component_sum(forecast_error, "forecast_error"),
        "inventory_cost_total": _component_sum(inventory_cost, "inventory_cost"),
        "planning_signal_volatility_total": _component_sum(
            planning_signal_volatility, "planning_signal_volatility"
        ),
        "model_switching_cost_total": _component_sum(model_switching_cost, "model_switching_cost"),
        "execution_adaptation_penalty_total": _component_sum(
            execution_adaptation_penalty, "execution_adaptation_penalty"
        ),
    }
    components["forecast_error_mean"] = float(np.mean(_as_float_array(forecast_error, "forecast_error")))
    components["inventory_cost_mean"] = float(np.mean(_as_float_array(inventory_cost, "inventory_cost")))
    components["execution_penalty_period_share"] = float(
        np.mean(_as_float_array(execution_adaptation_penalty, "execution_adaptation_penalty") > 0.0)
    )
    return components


def add_normalized_planning_loss(
    summary: pd.DataFrame,
    weights: Mapping[str, float],
    reference_strategy: str = "global_best_model",
    dataset_name: str = "unknown_dataset",
    run_mode: str = "unknown_run_mode",
    split_name: str = "test",
    epsilon: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add normalized planning-loss components to a strategy summary.

    The normalized objective is anchored to the accuracy-first reference
    strategy within the same dataset, run mode, and split. This prevents the
    inventory-cost scale from mechanically dominating execution feasibility
    terms while still keeping all raw operational metrics available.
    """
    frame = summary.copy()
    required_columns = [
        "strategy",
        "total_inventory_cost",
        "planning_signal_volatility_total",
        "execution_adaptation_penalty_total",
        "model_switching_cost_total",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError("Normalized planning loss is missing required columns: {}".format(missing))

    component_specs = [
        ("inventory_cost", "total_inventory_cost", "inventory_cost_ref", "beta_inventory", "normalized_inventory_component"),
        (
            "planning_volatility",
            "planning_signal_volatility_total",
            "volatility_ref",
            "lambda_volatility",
            "normalized_volatility_component",
        ),
        (
            "execution_penalty",
            "execution_adaptation_penalty_total",
            "execution_penalty_ref",
            "lambda_execution",
            "normalized_execution_component",
        ),
        (
            "model_switch_count",
            "model_switching_cost_total",
            "switch_count_ref",
            "lambda_switch",
            "normalized_switch_component",
        ),
    ]

    reference_values: Dict[str, float] = {}
    fallback_flags: Dict[str, bool] = {}
    fallback_reasons: Dict[str, str] = {}
    audit_records = []
    split_mask = (frame["split_name"] == split_name) if "split_name" in frame.columns else pd.Series(True, index=frame.index)
    reference_rows = frame[split_mask & (frame["strategy"] == reference_strategy)]

    for component_name, source_column, reference_column, weight_name, output_column in component_specs:
        raw_values = pd.to_numeric(frame[source_column], errors="coerce")
        reference_value = np.nan
        fallback_used = False
        fallback_reason = ""
        if not reference_rows.empty:
            reference_value = float(pd.to_numeric(reference_rows.iloc[0][source_column], errors="coerce"))
        if not np.isfinite(reference_value) or abs(reference_value) <= epsilon:
            nonzero_values = raw_values[np.isfinite(raw_values) & (raw_values.abs() > epsilon)]
            if not nonzero_values.empty:
                reference_value = float(nonzero_values.median())
                fallback_reason = "Reference strategy value was zero or unavailable; used median nonzero strategy value."
            else:
                reference_value = 1.0
                fallback_reason = "Reference strategy value and all strategy values were zero or unavailable; used 1.0."
            fallback_used = True

        weight = planning_loss_weight(weights, weight_name)
        frame[output_column] = weight * raw_values / max(abs(reference_value), epsilon)
        reference_values[reference_column] = float(reference_value)
        fallback_flags[reference_column] = fallback_used
        fallback_reasons[reference_column] = fallback_reason
        audit_records.append(
            {
                "dataset_name": dataset_name,
                "run_mode": run_mode,
                "split_name": split_name,
                "component_name": component_name,
                "raw_min": float(raw_values.min()),
                "raw_median": float(raw_values.median()),
                "raw_max": float(raw_values.max()),
                "reference_strategy": reference_strategy,
                "reference_value": float(reference_value),
                "component_weight": weight,
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "normalized_median": float(frame[output_column].median()),
                "normalized_max": float(frame[output_column].max()),
            }
        )

    frame["normalized_total_loss"] = (
        frame["normalized_inventory_component"]
        + frame["normalized_volatility_component"]
        + frame["normalized_execution_component"]
        + frame["normalized_switch_component"]
    )
    frame["raw_total_planning_loss"] = frame.get("total_planning_loss", np.nan)

    fallback_used_any = any(fallback_flags.values())
    reference_table = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "run_mode": run_mode,
                "split_name": split_name,
                "reference_strategy": reference_strategy,
                "inventory_cost_ref": reference_values["inventory_cost_ref"],
                "volatility_ref": reference_values["volatility_ref"],
                "execution_penalty_ref": reference_values["execution_penalty_ref"],
                "switch_count_ref": reference_values["switch_count_ref"],
                "fallback_used": bool(fallback_used_any),
                "fallback_reason": "; ".join(
                    reason for reason in fallback_reasons.values() if reason
                ),
            }
        ]
    )
    audit_table = pd.DataFrame(audit_records)
    return frame, reference_table, audit_table


def add_oracle_gap_columns(
    summary: pd.DataFrame,
    group_columns: Sequence[str] = ORACLE_GAP_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Add DP-oracle and perfect-oracle normalized-loss gaps.

    ``gap_to_dp_oracle`` compares each strategy to the non-deployable
    Realized-Inventory Oracle DP over the same candidate forecasts.
    ``gap_to_perfect_oracle`` compares each strategy to the realized-demand
    benchmark. Gaps are computed within the grouping columns that are present in
    the table, so scenarios, grains, and evaluation modes are not mixed.
    """
    frame = summary.copy()
    required_columns = {"strategy", "normalized_total_loss"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError("Oracle gap computation is missing required columns: {}".format(sorted(missing)))

    frame["gap_to_dp_oracle"] = np.nan
    frame["gap_to_perfect_oracle"] = np.nan
    present_group_columns = [column for column in group_columns if column in frame.columns]
    grouped = [((), frame)] if not present_group_columns else frame.groupby(present_group_columns, dropna=False, sort=False)

    for _, group in grouped:
        dp_oracle_value = _oracle_normalized_loss(group, "oracle_dp_feasibility_selector")
        perfect_oracle_value = _oracle_normalized_loss(group, "oracle_realized_demand")
        normalized_loss = pd.to_numeric(group["normalized_total_loss"], errors="coerce")
        if np.isfinite(dp_oracle_value):
            frame.loc[group.index, "gap_to_dp_oracle"] = normalized_loss - dp_oracle_value
        if np.isfinite(perfect_oracle_value):
            frame.loc[group.index, "gap_to_perfect_oracle"] = normalized_loss - perfect_oracle_value
    return frame


def _oracle_normalized_loss(group: pd.DataFrame, strategy_name: str) -> float:
    """Return the oracle normalized loss for one already-scoped group."""
    oracle_rows = group[group["strategy"] == strategy_name]
    if oracle_rows.empty:
        return np.nan
    values = pd.to_numeric(oracle_rows["normalized_total_loss"], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])
