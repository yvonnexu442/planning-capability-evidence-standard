"""Deployable greedy and finite-horizon policies over dynamic inventory state.

Candidate forecasts are fixed before held-out evaluation.  Policy search advances
an expected inventory state using candidate demand, never realized test demand.
The DP is a declared finite-state approximation: continuous state is rounded and
only the lowest-cost path per rounded state is retained, with a deterministic
beam cap for computational control.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from decision_layer.no_leakage import require_no_future_outcomes


@dataclass(frozen=True)
class DynamicPolicyConfig:
    lead_time: int = 2
    holding_cost_rate: float = 1.0
    shortage_cost_rate: float = 5.0
    order_change_cost_rate: float = 0.05
    switching_cost_rate: float = 0.05
    capacity: Optional[float] = None
    max_order_change_rate: Optional[float] = None
    switch_budget: Optional[int] = None
    frozen_horizon: int = 1
    backlog_persistence: float = 1.0
    state_rounding: float = 1.0
    beam_width: int = 128


@dataclass(frozen=True)
class _State:
    cost: float
    on_hand: float
    backlog: float
    pipeline: Tuple[float, ...]
    prior_model: str
    prior_order: float
    prior_plan: float
    switches: int
    path: Tuple[str, ...]


def _clip_order(desired: float, prior: float, config: DynamicPolicyConfig) -> float:
    order = min(desired, config.capacity) if config.capacity is not None else desired
    if config.max_order_change_rate is not None:
        limit = config.max_order_change_rate * max(abs(prior), 1.0)
        order = float(np.clip(order, max(0.0, prior - limit), prior + limit))
    return max(0.0, float(order))


def _advance(state: _State, model: str, forecast: float, plan: float, config: DynamicPolicyConfig) -> _State:
    pipeline = list(state.pipeline)
    arrival = pipeline.pop(0) if pipeline else 0.0
    on_hand = state.on_hand + arrival
    inventory_position = on_hand + sum(pipeline) - state.backlog
    desired = max(0.0, plan - inventory_position)
    order = _clip_order(desired, state.prior_order, config)
    if config.lead_time == 0:
        on_hand += order
    else:
        while len(pipeline) < config.lead_time:
            pipeline.append(0.0)
        pipeline[-1] += order
    backlog = state.backlog * config.backlog_persistence
    served_backlog = min(on_hand, backlog)
    on_hand -= served_backlog
    backlog -= served_backlog
    served = min(on_hand, max(forecast, 0.0))
    on_hand -= served
    backlog += max(forecast, 0.0) - served
    switched = int(bool(state.prior_model) and state.prior_model != model)
    cost = (
        config.holding_cost_rate * on_hand
        + config.shortage_cost_rate * backlog
        + config.order_change_cost_rate * abs(order - state.prior_order)
        + config.switching_cost_rate * switched
        + max(0.0, desired - order)
    )
    return _State(
        cost=state.cost + cost,
        on_hand=on_hand,
        backlog=backlog,
        pipeline=tuple(pipeline),
        prior_model=model,
        prior_order=order,
        prior_plan=plan,
        switches=state.switches + switched,
        path=state.path + (model,),
    )


def _key(state: _State, config: DynamicPolicyConfig) -> Tuple[object, ...]:
    scale = max(config.state_rounding, 1e-8)
    rounded = lambda value: int(round(value / scale))
    return (
        state.prior_model,
        # Retain switch count for both unconstrained and budgeted DP so the
        # state approximation is identical; the budget must be the only
        # difference between the two searches.
        state.switches,
        rounded(state.on_hand),
        rounded(state.backlog),
        tuple(rounded(value) for value in state.pipeline),
        rounded(state.prior_order),
        rounded(state.prior_plan),
    )


def select_dynamic_policy(
    candidate_forecasts: Mapping[str, Sequence[float]],
    safety_stock: float,
    initial_inventory: float,
    initial_order: float,
    config: DynamicPolicyConfig,
    method: str,
) -> Dict[str, np.ndarray]:
    """Return a deployable model path, forecast, plan, and switch indicators."""
    require_no_future_outcomes(candidate_forecasts, "select_dynamic_policy")
    models = sorted(candidate_forecasts)
    if not models:
        raise ValueError("candidate_forecasts cannot be empty")
    arrays = {name: np.asarray(values, dtype=float) for name, values in candidate_forecasts.items()}
    horizon = len(arrays[models[0]])
    if any(len(values) != horizon for values in arrays.values()):
        raise ValueError("all candidate horizons must match")
    initial = _State(0.0, initial_inventory, 0.0, (0.0,) * config.lead_time, "", initial_order, initial_inventory, 0, ())
    states = [initial]
    for period in range(horizon):
        next_states = []
        for state in states:
            allowed = models
            if config.frozen_horizon > 1 and state.prior_model and period % config.frozen_horizon:
                allowed = [state.prior_model]
            for model in allowed:
                switches = state.switches + int(bool(state.prior_model) and state.prior_model != model)
                if config.switch_budget is not None and switches > config.switch_budget:
                    continue
                forecast = max(float(arrays[model][period]), 0.0)
                plan = (config.lead_time + 1) * forecast + safety_stock
                next_states.append(_advance(state, model, forecast, plan, config))
        if method == "greedy":
            states = [min(next_states, key=lambda value: (value.cost, value.path))]
        else:
            retained: Dict[Tuple[object, ...], _State] = {}
            for state in next_states:
                key = _key(state, config)
                incumbent = retained.get(key)
                if incumbent is None or (state.cost, state.path) < (incumbent.cost, incumbent.path):
                    retained[key] = state
            states = sorted(retained.values(), key=lambda value: (value.cost, value.path))[: config.beam_width]
        if not states:
            raise RuntimeError("no feasible dynamic-policy state remains")
    best = min(states, key=lambda value: (value.cost, value.path))
    path = np.asarray(best.path, dtype=object)
    forecast = np.asarray([arrays[model][period] for period, model in enumerate(path)], dtype=float)
    plan = (config.lead_time + 1) * forecast + float(safety_stock)
    switches = np.r_[0.0, (path[1:] != path[:-1]).astype(float)] if horizon else np.array([], dtype=float)
    return {"model": path, "forecast": forecast, "plan": plan, "switch": switches}
