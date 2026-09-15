"""Forecast error metrics used as one component of planning utility."""

from typing import Optional, Sequence

import numpy as np


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a numeric sequence into a one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional sequence.".format(name))
    return array


def _validate_pair(actual: Sequence[float], predicted: Sequence[float]) -> tuple:
    """Validate actual and predicted arrays for elementwise forecast metrics."""
    actual_array = _as_float_array(actual, "actual")
    predicted_array = _as_float_array(predicted, "predicted")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if actual_array.size == 0:
        raise ValueError("actual and predicted must not be empty.")
    return actual_array, predicted_array


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the mean absolute forecast error.

    This metric measures the average absolute distance between actual demand and
    a forecast. It matters for the paper because it represents the conventional
    forecast accuracy objective that many demand planning systems optimize.

    The planning-infrastructure gap appears when this value improves while
    planning signal volatility, model switching, or execution penalties worsen.
    """
    actual_array, predicted_array = _validate_pair(actual, predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the root mean squared forecast error.

    RMSE penalizes large forecast misses more strongly than MAE. It is useful
    when large demand misses are operationally important, but it still evaluates
    only prediction error rather than whether the resulting plan is stable or
    executable.

    In this repository, RMSE is reported beside planning metrics so the paper
    can show when a lower-error model creates a less absorbable planning signal.
    """
    actual_array, predicted_array = _validate_pair(actual, predicted)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def root_mean_squared_log_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the root mean squared logarithmic forecast error.

    RMSLE measures proportional error for non-negative demand. It can be useful
    when the research question spans planning units with very different demand
    scales. The metric is included as a forecast-side baseline, not as a complete
    measure of operational planning quality.

    The function requires non-negative actuals and predictions because negative
    demand or inventory signals do not have a valid logarithmic interpretation.
    """
    actual_array, predicted_array = _validate_pair(actual, predicted)
    if np.any(actual_array < 0) or np.any(predicted_array < 0):
        raise ValueError("RMSLE requires non-negative actual and predicted values.")
    return float(np.sqrt(np.mean((np.log1p(actual_array) - np.log1p(predicted_array)) ** 2)))


def weighted_absolute_percentage_error(
    actual: Sequence[float],
    predicted: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    epsilon: float = 1e-8,
) -> float:
    """Return weighted absolute percentage error.

    WAPE divides absolute forecast error by total absolute demand. It is often
    more stable than period-level percentage errors when demand can be zero.

    This metric matters because it summarizes forecast accuracy at the portfolio
    level. The paper should compare WAPE with stability and execution metrics to
    identify cases where aggregate accuracy improves but executable planning
    quality declines.
    """
    actual_array, predicted_array = _validate_pair(actual, predicted)
    if weights is None:
        weight_array = np.ones_like(actual_array)
    else:
        weight_array = _as_float_array(weights, "weights")
        if weight_array.shape != actual_array.shape:
            raise ValueError("weights must have the same shape as actual.")

    numerator = np.sum(weight_array * np.abs(actual_array - predicted_array))
    denominator = max(float(np.sum(weight_array * np.abs(actual_array))), float(epsilon))
    return float(numerator / denominator)
