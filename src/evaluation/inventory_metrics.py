"""Inventory and service metrics for executable planning signals."""

from typing import Sequence

import numpy as np


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric sequence into a one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional sequence.".format(name))
    return array


def _validate_pair(left: Sequence[float], right: Sequence[float], left_name: str, right_name: str) -> tuple:
    """Validate two arrays used in inventory calculations."""
    left_array = _as_float_array(left, left_name)
    right_array = _as_float_array(right, right_name)
    if left_array.shape != right_array.shape:
        raise ValueError("{} and {} must have the same shape.".format(left_name, right_name))
    if left_array.size == 0:
        raise ValueError("{} and {} must not be empty.".format(left_name, right_name))
    return left_array, right_array


def compute_inventory_target(forecast: Sequence[float], safety_stock: Sequence[float]) -> np.ndarray:
    """Convert a forecast and safety stock into an inventory target.

    The inventory target is the executable planning signal used by downstream
    operations. It matters for the paper because the operation does not execute
    raw forecasts; it executes a target such as inventory, staffing, or capacity.

    This function keeps the transformation explicit so the planning-
    infrastructure gap can be measured on the final target rather than only on
    the forecast.
    """
    forecast_array, safety_stock_array = _validate_pair(forecast, safety_stock, "forecast", "safety_stock")
    return np.maximum(forecast_array + safety_stock_array, 0.0)


def compute_holding_cost(
    inventory_target: Sequence[float],
    actual_demand: Sequence[float],
    holding_cost_rate: float,
) -> np.ndarray:
    """Return period-level holding cost for excess planning signal.

    Holding cost measures surplus inventory or over-allocation. It matters for
    the paper because a stability-aware policy may reduce plan churn while
    creating extra inventory exposure.

    Reporting this term separately helps distinguish useful stability from
    stability that merely hides demand change behind excess stock.
    """
    target_array, demand_array = _validate_pair(inventory_target, actual_demand, "inventory_target", "actual_demand")
    return float(holding_cost_rate) * np.maximum(target_array - demand_array, 0.0)


def compute_shortage_cost(
    inventory_target: Sequence[float],
    actual_demand: Sequence[float],
    shortage_cost_rate: float,
) -> np.ndarray:
    """Return period-level shortage cost for unmet demand.

    Shortage cost measures the operational consequence of planning below demand.
    It matters because overly stable planning signals may fail to adapt when
    demand changes.

    The planning-infrastructure gap is not solved by freezing every plan; this
    metric keeps demand responsiveness visible.
    """
    target_array, demand_array = _validate_pair(inventory_target, actual_demand, "inventory_target", "actual_demand")
    return float(shortage_cost_rate) * np.maximum(demand_array - target_array, 0.0)


def compute_total_inventory_cost(
    inventory_target: Sequence[float],
    actual_demand: Sequence[float],
    holding_cost_rate: float,
    shortage_cost_rate: float,
) -> np.ndarray:
    """Return total inventory cost as holding cost plus shortage cost.

    This is the inventory component of planning utility. It matters because the
    paper evaluates planning outcomes, not only forecast error.

    Inventory cost can reveal whether a stability-aware decision layer improves
    execution quality by making the plan smoother without creating unacceptable
    service or stock exposure.
    """
    holding_cost = compute_holding_cost(inventory_target, actual_demand, holding_cost_rate)
    shortage_cost = compute_shortage_cost(inventory_target, actual_demand, shortage_cost_rate)
    return holding_cost + shortage_cost


def compute_service_level(
    inventory_target: Sequence[float],
    actual_demand: Sequence[float],
    epsilon: float = 1e-8,
) -> float:
    """Return an aggregate fill-rate style service level.

    Service level is computed as one minus total unmet demand divided by total
    demand. It matters because a stable plan is only useful if it still serves
    demand acceptably.

    This metric connects to the planning-infrastructure gap by showing whether
    smoothing or execution-aware constraints preserve operational service while
    reducing instability.
    """
    target_array, demand_array = _validate_pair(inventory_target, actual_demand, "inventory_target", "actual_demand")
    shortage_units = np.maximum(demand_array - target_array, 0.0)
    denominator = max(float(np.sum(np.maximum(demand_array, 0.0))), float(epsilon))
    return float(1.0 - np.sum(shortage_units) / denominator)
