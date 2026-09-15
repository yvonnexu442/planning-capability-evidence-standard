"""Execution capacity functions for measuring absorbable plan changes."""

from typing import Optional, Sequence

import numpy as np

from evaluation.stability_metrics import compute_absolute_plan_change


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric sequence into a one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional sequence.".format(name))
    return array


def compute_execution_capacity(
    planning_signal: Sequence[float],
    max_plan_change_rate: Optional[float] = None,
    absolute_capacity: Optional[float] = None,
    minimum_capacity: float = 0.0,
) -> np.ndarray:
    """Return the maximum plan change the operation can absorb each period.

    Execution capacity represents constraints in procurement, staffing,
    transportation, production, software systems, and planner attention. It
    matters because a forecast can adapt faster than these systems.

    Capacity can be specified as a percentage of the previous plan, an absolute
    value, or the maximum of both.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    if signal.size == 0:
        raise ValueError("planning_signal must not be empty.")

    capacity = np.repeat(float(minimum_capacity), signal.size)
    if max_plan_change_rate is not None:
        prior_plan = np.zeros_like(signal)
        prior_plan[1:] = np.abs(signal[:-1])
        capacity = np.maximum(capacity, prior_plan * float(max_plan_change_rate))
    if absolute_capacity is not None:
        capacity = np.maximum(capacity, float(absolute_capacity))
    return capacity


def check_plan_change_constraint(
    planning_signal: Sequence[float],
    execution_capacity: Sequence[float],
) -> np.ndarray:
    """Return a boolean array indicating whether each plan change is absorbable.

    A value of `True` means the period-to-period plan change is within execution
    capacity. This matters for the paper because it turns execution feasibility
    into an explicit measurable property.

    Constraint failures identify moments where the planning-infrastructure gap
    becomes operationally visible.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    capacity = _as_float_array(execution_capacity, "execution_capacity")
    if signal.shape != capacity.shape:
        raise ValueError("planning_signal and execution_capacity must have the same shape.")
    return compute_absolute_plan_change(signal) <= np.maximum(capacity, 0.0)


def compute_execution_violation(
    planning_signal: Sequence[float],
    execution_capacity: Sequence[float],
) -> np.ndarray:
    """Return the amount by which each plan change exceeds capacity.

    This is the unweighted execution adaptation penalty. It matters because it
    quantifies how much the plan asks the operation to do beyond its absorbable
    change limit.

    Positive values are direct evidence that forecast or planning updates are
    outpacing execution infrastructure.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    capacity = _as_float_array(execution_capacity, "execution_capacity")
    if signal.shape != capacity.shape:
        raise ValueError("planning_signal and execution_capacity must have the same shape.")
    return np.maximum(compute_absolute_plan_change(signal) - np.maximum(capacity, 0.0), 0.0)
