"""Metrics for planning signal stability and model switching burden."""

from typing import Sequence

import numpy as np


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric sequence into a one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional sequence.".format(name))
    return array


def compute_absolute_plan_change(planning_signal: Sequence[float]) -> np.ndarray:
    """Return absolute period-to-period changes in a planning signal.

    This metric measures how much the executable plan moves between periods. It
    matters because planners, suppliers, stores, and systems must absorb these
    changes, even when forecast error metrics look favorable.

    The first period has no prior plan in the input sequence, so its change is
    reported as zero.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    if signal.size == 0:
        raise ValueError("planning_signal must not be empty.")
    changes = np.zeros_like(signal)
    changes[1:] = np.abs(np.diff(signal))
    return changes


def compute_percentage_plan_change(planning_signal: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    """Return absolute percentage changes in a planning signal.

    Percentage change normalizes plan movement by the prior plan level. It
    matters because a 100-unit change has different operational meaning for a
    small planning unit than for a large one.

    Large percentage changes are one way the planning-infrastructure gap becomes
    visible: the forecast layer may generate jumps that downstream execution
    cannot absorb.
    """
    signal = _as_float_array(planning_signal, "planning_signal")
    if signal.size == 0:
        raise ValueError("planning_signal must not be empty.")
    changes = np.zeros_like(signal)
    prior = np.maximum(np.abs(signal[:-1]), float(epsilon))
    changes[1:] = np.abs(np.diff(signal)) / prior
    return changes


def compute_large_jump_count(
    planning_signal: Sequence[float],
    jump_threshold: float,
    use_percentage_change: bool = True,
) -> int:
    """Count planning signal changes above a jump threshold.

    Large jumps are operationally important because they can trigger planner
    intervention, supplier renegotiation, system reconfiguration, or inventory
    target overrides.

    Counting large jumps helps the paper show when a numerically accurate model
    creates unstable signals that execution infrastructure cannot absorb.
    """
    if use_percentage_change:
        changes = compute_percentage_plan_change(planning_signal)
    else:
        changes = compute_absolute_plan_change(planning_signal)
    return int(np.sum(changes > float(jump_threshold)))


def compute_model_switch_count(selected_models: Sequence[object]) -> int:
    """Count hard model-selection changes across periods.

    Model switching matters because each selected model can imply different
    validation, deployment, monitoring, and explanation requirements.

    A high switch count can be evidence of the planning-infrastructure gap: the
    forecasting layer is changing model logic faster than governance and
    execution systems can comfortably absorb.
    """
    models = list(selected_models)
    if not models:
        return 0
    return int(sum(1 for previous, current in zip(models[:-1], models[1:]) if previous != current))


def compute_model_switch_rate(selected_models: Sequence[object]) -> float:
    """Return the share of transitions where the selected model changes.

    The switch rate normalizes switching by the number of available transitions.
    It matters when comparing planning units or test windows with different
    lengths.

    A high switch rate can indicate that forecast selection is too volatile for
    stable operational execution.
    """
    models = list(selected_models)
    if len(models) <= 1:
        return 0.0
    return float(compute_model_switch_count(models) / float(len(models) - 1))


def compute_weight_change(model_weights: Sequence[Sequence[float]]) -> np.ndarray:
    """Return total ensemble weight movement between periods.

    For soft ensembles, instability can occur even without hard model switches.
    This metric sums absolute weight changes across models at each transition.

    Weight movement matters for the paper because an ensemble can appear smooth
    in forecast error while still shifting responsibility across models in ways
    that create monitoring and governance burden.
    """
    weights = np.asarray(model_weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("model_weights must be a two-dimensional array of shape time by model.")
    if weights.shape[0] == 0:
        raise ValueError("model_weights must not be empty.")
    changes = np.zeros(weights.shape[0], dtype=float)
    changes[1:] = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    return changes
