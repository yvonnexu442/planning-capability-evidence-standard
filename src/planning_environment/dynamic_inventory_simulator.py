"""Rolling inventory simulation for held-out forecast-to-plan evaluation.

This module is intentionally separate from the paper's original static planning
metrics. Policies provide forecast-driven order-up-to targets; realized demand
is consumed only by the forward state transition used for ex-post evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicInventoryConfig:
    """Parameters for a deterministic rolling inventory evaluation."""

    lead_time: int = 0
    holding_cost_rate: float = 1.0
    shortage_cost_rate: float = 5.0
    order_change_cost_rate: float = 0.0
    expedite_cost_rate: float = 0.0
    replenishment_capacity: Optional[float] = None
    max_order_change_rate: Optional[float] = None
    backlog_mode: str = "backorder"
    initial_on_hand: float = 0.0
    initial_backlog: float = 0.0
    initial_order_quantity: Optional[float] = None

    def validate(self) -> None:
        if self.lead_time < 0:
            raise ValueError("lead_time must be nonnegative.")
        if self.backlog_mode not in {"backorder", "lost_sales"}:
            raise ValueError("backlog_mode must be 'backorder' or 'lost_sales'.")
        for name in (
            "holding_cost_rate",
            "shortage_cost_rate",
            "order_change_cost_rate",
            "expedite_cost_rate",
            "initial_on_hand",
            "initial_backlog",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("{} must be nonnegative.".format(name))
        if self.replenishment_capacity is not None and self.replenishment_capacity < 0.0:
            raise ValueError("replenishment_capacity must be nonnegative.")
        if self.max_order_change_rate is not None and self.max_order_change_rate < 0.0:
            raise ValueError("max_order_change_rate must be nonnegative.")
        if self.initial_order_quantity is not None and self.initial_order_quantity < 0.0:
            raise ValueError("initial_order_quantity must be nonnegative.")


def _one_dimensional(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be one-dimensional.".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains non-finite values.".format(name))
    return array


def _clip_order_change(
    desired_order: float,
    previous_order: Optional[float],
    max_order_change_rate: Optional[float],
) -> float:
    if max_order_change_rate is None or previous_order is None:
        return desired_order
    base = max(abs(previous_order), 1.0)
    limit = float(max_order_change_rate) * base
    return float(np.clip(desired_order, max(0.0, previous_order - limit), previous_order + limit))


def simulate_dynamic_inventory(
    actual_demand: Sequence[float],
    order_up_to_target: Sequence[float],
    config: DynamicInventoryConfig,
    model_switch: Optional[Sequence[float]] = None,
    switching_cost_rate: float = 0.0,
) -> pd.DataFrame:
    """Roll inventory, backlog, on-order stock, and constrained orders forward.

    ``order_up_to_target`` must be fixed before this function sees held-out
    ``actual_demand``. The simulator never uses future demand when calculating
    an order; demand is read only after that period's constrained order has
    been determined.
    """

    config.validate()
    demand = _one_dimensional(actual_demand, "actual_demand")
    target = _one_dimensional(order_up_to_target, "order_up_to_target")
    if demand.shape != target.shape:
        raise ValueError("actual_demand and order_up_to_target must have the same shape.")
    if np.any(demand < 0.0) or np.any(target < 0.0):
        raise ValueError("demand and order-up-to targets must be nonnegative.")

    switches = np.zeros(demand.size, dtype=float) if model_switch is None else _one_dimensional(model_switch, "model_switch")
    if switches.shape != demand.shape:
        raise ValueError("model_switch must match the evaluation horizon.")
    if switching_cost_rate < 0.0:
        raise ValueError("switching_cost_rate must be nonnegative.")

    on_hand = float(config.initial_on_hand)
    backlog = float(config.initial_backlog if config.backlog_mode == "backorder" else 0.0)
    arrival_pipeline: Dict[int, float] = {}
    previous_order: Optional[float] = config.initial_order_quantity
    records = []

    for period in range(demand.size):
        arrivals = float(arrival_pipeline.pop(period, 0.0))
        on_hand += arrivals
        on_order_before = float(sum(arrival_pipeline.values()))
        inventory_position = on_hand + on_order_before - backlog
        desired_order = max(0.0, float(target[period]) - inventory_position)

        capacity_order = desired_order
        if config.replenishment_capacity is not None:
            capacity_order = min(capacity_order, float(config.replenishment_capacity))
        constrained_order = _clip_order_change(capacity_order, previous_order, config.max_order_change_rate)
        constrained_order = max(0.0, constrained_order)
        execution_violation = max(0.0, desired_order - constrained_order)
        expedite_units = max(0.0, desired_order - capacity_order)

        if config.lead_time == 0:
            on_hand += constrained_order
            same_period_arrival = constrained_order
        else:
            arrival_pipeline[period + config.lead_time] = arrival_pipeline.get(period + config.lead_time, 0.0) + constrained_order
            same_period_arrival = 0.0

        prior_backlog = backlog
        backlog_fulfilled = min(on_hand, prior_backlog) if config.backlog_mode == "backorder" else 0.0
        on_hand -= backlog_fulfilled
        remaining_backlog = prior_backlog - backlog_fulfilled
        current_fulfilled = min(on_hand, float(demand[period]))
        on_hand -= current_fulfilled
        unmet_current = float(demand[period]) - current_fulfilled

        if config.backlog_mode == "backorder":
            backlog = remaining_backlog + unmet_current
            shortage_units = backlog
            lost_sales = 0.0
        else:
            backlog = 0.0
            shortage_units = unmet_current
            lost_sales = unmet_current

        holding_cost = config.holding_cost_rate * on_hand
        shortage_cost = config.shortage_cost_rate * shortage_units
        order_change = 0.0 if previous_order is None else abs(constrained_order - previous_order)
        order_change_cost = config.order_change_cost_rate * order_change
        expedite_cost = config.expedite_cost_rate * expedite_units
        switching_cost = switching_cost_rate * max(float(switches[period]), 0.0)
        total_cost = holding_cost + shortage_cost + order_change_cost + expedite_cost + switching_cost

        records.append(
            {
                "period_index": period,
                "demand": float(demand[period]),
                "order_up_to_target": float(target[period]),
                "arrivals": arrivals + same_period_arrival,
                "inventory_position_before_order": inventory_position,
                "desired_order": desired_order,
                "order_quantity": constrained_order,
                "on_order_end": float(sum(arrival_pipeline.values())),
                "on_hand_end": on_hand,
                "backlog_end": backlog,
                "lost_sales": lost_sales,
                "current_demand_fulfilled": current_fulfilled,
                "execution_violation_units": execution_violation,
                "execution_violation": float(execution_violation > 0.0),
                "order_change": order_change,
                "model_switch": max(float(switches[period]), 0.0),
                "holding_cost": holding_cost,
                "shortage_cost": shortage_cost,
                "order_change_cost": order_change_cost,
                "expedite_cost": expedite_cost,
                "switching_cost": switching_cost,
                "total_dynamic_cost": total_cost,
            }
        )
        previous_order = constrained_order

    return pd.DataFrame.from_records(records)


def summarize_dynamic_inventory(simulation: pd.DataFrame) -> Dict[str, float]:
    """Return strategy-level operational metrics from a dynamic rollout."""

    required = {
        "demand",
        "current_demand_fulfilled",
        "on_hand_end",
        "backlog_end",
        "lost_sales",
        "execution_violation_units",
        "execution_violation",
        "order_change",
        "model_switch",
        "holding_cost",
        "shortage_cost",
        "order_change_cost",
        "expedite_cost",
        "switching_cost",
        "total_dynamic_cost",
    }
    missing = required.difference(simulation.columns)
    if missing:
        raise ValueError("simulation is missing columns: {}".format(sorted(missing)))
    demand_total = float(simulation["demand"].sum())
    return {
        "total_holding_cost": float(simulation["holding_cost"].sum()),
        "total_shortage_cost": float(simulation["shortage_cost"].sum()),
        "total_order_change_cost": float(simulation["order_change_cost"].sum()),
        "total_expedite_cost": float(simulation["expedite_cost"].sum()),
        "total_switching_cost": float(simulation["switching_cost"].sum()),
        "total_dynamic_cost": float(simulation["total_dynamic_cost"].sum()),
        "fill_rate": 1.0 if demand_total <= 0.0 else float(simulation["current_demand_fulfilled"].sum() / demand_total),
        "average_on_hand_inventory": float(simulation["on_hand_end"].mean()),
        "ending_backlog": float(simulation["backlog_end"].iloc[-1]) if len(simulation) else 0.0,
        "total_lost_sales": float(simulation["lost_sales"].sum()),
        "execution_violation_units": float(simulation["execution_violation_units"].sum()),
        "planning_execution_gap_rate": float(simulation["execution_violation"].mean()) if len(simulation) else 0.0,
        "plan_volatility": float(simulation["order_change"].sum()),
        "switch_count": float(simulation["model_switch"].sum()),
    }
