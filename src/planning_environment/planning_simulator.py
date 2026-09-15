"""A simple simulator for evaluating forecast-driven planning strategies."""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.forecast_metrics import mean_absolute_error, weighted_absolute_percentage_error
from evaluation.inventory_metrics import (
    compute_service_level,
    compute_total_inventory_cost,
)
from evaluation.planning_utility import compute_execution_penalty, compute_multi_objective_summary
from evaluation.stability_metrics import compute_absolute_plan_change, compute_percentage_plan_change
from planning_environment.execution_capacity import compute_execution_capacity
from planning_environment.planning_actions import forecast_to_inventory_target


def simulate_planning_outcomes(
    actual_demand: Sequence[float],
    forecast: Sequence[float],
    safety_stock: Sequence[float],
    holding_cost_rate: float,
    shortage_cost_rate: float,
    max_plan_change_rate: Optional[float] = None,
    absolute_execution_capacity: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate planning outcomes for one demand series.

    The simulator converts forecasts into inventory targets and then evaluates
    cost, service, stability, and execution penalties. It is intentionally simple
    in this first skeleton so the research logic remains transparent.

    The output table is the bridge between forecast metrics and paper claims
    about the planning-infrastructure gap.
    """
    demand = np.asarray(actual_demand, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    safety_stock_array = np.asarray(safety_stock, dtype=float)
    if not (demand.shape == forecast_array.shape == safety_stock_array.shape):
        raise ValueError("actual_demand, forecast, and safety_stock must have the same shape.")

    planning_signal = forecast_to_inventory_target(forecast_array, safety_stock_array)
    inventory_cost = compute_total_inventory_cost(
        planning_signal,
        demand,
        holding_cost_rate=holding_cost_rate,
        shortage_cost_rate=shortage_cost_rate,
    )
    execution_capacity = compute_execution_capacity(
        planning_signal,
        max_plan_change_rate=max_plan_change_rate,
        absolute_capacity=absolute_execution_capacity,
    )
    execution_penalty = compute_execution_penalty(planning_signal, execution_capacity)

    return pd.DataFrame(
        {
            "period_index": np.arange(demand.size),
            "actual_demand": demand,
            "forecast": forecast_array,
            "safety_stock": safety_stock_array,
            "planning_signal": planning_signal,
            "inventory_cost": inventory_cost,
            "absolute_plan_change": compute_absolute_plan_change(planning_signal),
            "percentage_plan_change": compute_percentage_plan_change(planning_signal),
            "execution_capacity": execution_capacity,
            "execution_penalty": execution_penalty,
        }
    )


def evaluate_planning_strategy(
    simulation_frame: pd.DataFrame,
    model_switching_cost: Optional[Sequence[float]] = None,
    loss_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Return scalar and multi-objective summaries for a simulated strategy.

    This function gives paper authors a compact strategy-level summary while
    preserving the separate objective components needed for Pareto analysis.

    It matters for the planning-infrastructure gap because it reports forecast
    error and execution penalties side by side.
    """
    required_columns = {
        "actual_demand",
        "forecast",
        "planning_signal",
        "inventory_cost",
        "percentage_plan_change",
        "execution_penalty",
    }
    missing = required_columns.difference(set(simulation_frame.columns))
    if missing:
        raise ValueError("simulation_frame is missing required columns: {}".format(sorted(missing)))

    demand = simulation_frame["actual_demand"].to_numpy(dtype=float)
    forecast = simulation_frame["forecast"].to_numpy(dtype=float)
    switching = (
        np.zeros(simulation_frame.shape[0], dtype=float)
        if model_switching_cost is None
        else np.asarray(model_switching_cost, dtype=float)
    )
    if switching.shape[0] != simulation_frame.shape[0]:
        raise ValueError("model_switching_cost must match the number of simulation rows.")

    forecast_error = np.abs(demand - forecast)
    summary = compute_multi_objective_summary(
        forecast_error=forecast_error,
        inventory_cost=simulation_frame["inventory_cost"].to_numpy(dtype=float),
        planning_signal_volatility=simulation_frame["percentage_plan_change"].to_numpy(dtype=float),
        model_switching_cost=switching,
        execution_adaptation_penalty=simulation_frame["execution_penalty"].to_numpy(dtype=float),
    )
    summary["mean_absolute_error"] = mean_absolute_error(demand, forecast)
    summary["weighted_absolute_percentage_error"] = weighted_absolute_percentage_error(demand, forecast)
    summary["service_level"] = compute_service_level(simulation_frame["planning_signal"].to_numpy(dtype=float), demand)

    if loss_weights is not None:
        from evaluation.planning_utility import compute_total_planning_loss

        summary["total_planning_loss"] = compute_total_planning_loss(
            forecast_error=forecast_error,
            inventory_cost=simulation_frame["inventory_cost"].to_numpy(dtype=float),
            planning_signal_volatility=simulation_frame["percentage_plan_change"].to_numpy(dtype=float),
            model_switching_cost=switching,
            execution_adaptation_penalty=simulation_frame["execution_penalty"].to_numpy(dtype=float),
            weights=loss_weights,
        )
    return summary
