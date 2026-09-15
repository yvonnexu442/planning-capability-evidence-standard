"""Transparent deployable controls used as strong practitioner baselines."""

from typing import Dict, Sequence

import numpy as np


BASELINE_METADATA: Dict[str, Dict[str, object]] = {
    "accuracy_first": {
        "future_information_used": False,
        "validation_tuning_required": True,
        "model_switching_required": False,
        "execution_capacity_required": False,
    },
    "accuracy_rate_limited": {
        "future_information_used": False,
        "validation_tuning_required": True,
        "model_switching_required": False,
        "execution_capacity_required": True,
    },
    "accuracy_smoothed": {
        "future_information_used": False,
        "validation_tuning_required": True,
        "model_switching_required": False,
        "execution_capacity_required": False,
    },
    "accuracy_frozen_horizon": {
        "future_information_used": False,
        "validation_tuning_required": True,
        "model_switching_required": False,
        "execution_capacity_required": False,
    },
    "accuracy_order_up_to": {
        "future_information_used": False,
        "validation_tuning_required": True,
        "model_switching_required": False,
        "execution_capacity_required": False,
    },
    "simple_ensemble": {
        "future_information_used": False,
        "validation_tuning_required": False,
        "model_switching_required": False,
        "execution_capacity_required": False,
    },
    "simple_ensemble_rate_limited": {
        "future_information_used": False,
        "validation_tuning_required": True,
        "model_switching_required": False,
        "execution_capacity_required": True,
    },
}


def _array(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite one-dimensional sequence.".format(name))
    return result


def rate_limit_signal(signal: Sequence[float], max_change_rate: float) -> np.ndarray:
    """Clip period-to-period changes relative to the previous executed signal."""

    values = _array(signal, "signal")
    if max_change_rate < 0.0:
        raise ValueError("max_change_rate must be nonnegative.")
    if not len(values):
        return values.copy()
    output = np.empty_like(values)
    output[0] = max(values[0], 0.0)
    for index in range(1, len(values)):
        limit = max(abs(output[index - 1]), 1.0) * max_change_rate
        output[index] = np.clip(values[index], max(0.0, output[index - 1] - limit), output[index - 1] + limit)
    return output


def exponential_smooth_signal(signal: Sequence[float], alpha: float) -> np.ndarray:
    """Apply causal exponential smoothing to a fixed forecast-driven signal."""

    values = _array(signal, "signal")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if not len(values):
        return values.copy()
    output = np.empty_like(values)
    output[0] = max(values[0], 0.0)
    for index in range(1, len(values)):
        output[index] = alpha * max(values[index], 0.0) + (1.0 - alpha) * output[index - 1]
    return output


def frozen_horizon_signal(signal: Sequence[float], freeze_periods: int) -> np.ndarray:
    """Freeze the executed signal within non-overlapping execution windows."""

    values = _array(signal, "signal")
    if freeze_periods < 1:
        raise ValueError("freeze_periods must be at least one.")
    output = values.copy()
    for start in range(0, len(values), freeze_periods):
        output[start : start + freeze_periods] = max(values[start], 0.0)
    return output


def simple_ensemble_signal(candidate_signals: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the row-wise mean of aligned candidate planning signals."""

    matrix = np.asarray(candidate_signals, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("candidate_signals must be a finite two-dimensional array.")
    if matrix.shape[0] < 1:
        raise ValueError("at least one candidate signal is required.")
    return np.maximum(matrix.mean(axis=0), 0.0)
