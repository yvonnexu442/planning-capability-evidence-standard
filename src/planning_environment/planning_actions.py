"""Planning actions that convert forecasts into executable targets."""

from typing import Sequence, Union

import numpy as np


NumericOrSequence = Union[float, Sequence[float]]


def _as_float_array(values: NumericOrSequence, reference_length: int, name: str) -> np.ndarray:
    """Convert a scalar or sequence into an array aligned to a reference length."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.repeat(float(array), reference_length)
    if array.ndim != 1:
        raise ValueError("{} must be a scalar or one-dimensional sequence.".format(name))
    if array.size != reference_length:
        raise ValueError("{} must have length {}.".format(name, reference_length))
    return array


def apply_safety_stock(
    forecast: Sequence[float],
    safety_stock: NumericOrSequence = 0.0,
    method: str = "constant",
) -> np.ndarray:
    """Return safety stock units aligned to the forecast horizon.

    Safety stock bridges forecast uncertainty and operational service goals. It
    matters for the paper because planning signals are forecasts plus policy
    buffers, not raw model outputs.

    The initial implementation supports constant and percentage methods. Future
    work can add service-level formulas that depend on forecast uncertainty and
    lead time.
    """
    forecast_array = np.asarray(forecast, dtype=float)
    if forecast_array.ndim != 1:
        raise ValueError("forecast must be a one-dimensional sequence.")

    safety_array = _as_float_array(safety_stock, forecast_array.size, "safety_stock")
    if method == "constant":
        return np.maximum(safety_array, 0.0)
    if method == "percentage":
        return np.maximum(forecast_array * safety_array, 0.0)
    raise ValueError("Unsupported safety stock method: {}".format(method))


def forecast_to_inventory_target(
    forecast: Sequence[float],
    safety_stock: NumericOrSequence = 0.0,
    safety_stock_method: str = "constant",
) -> np.ndarray:
    """Convert a forecast into an executable inventory target.

    This function is the first step from prediction to operation. It matters
    because the research question concerns planning signals, not only forecast
    values.

    The returned target is clipped at zero because negative inventory targets do
    not represent executable operational plans.
    """
    forecast_array = np.asarray(forecast, dtype=float)
    if forecast_array.ndim != 1:
        raise ValueError("forecast must be a one-dimensional sequence.")
    stock = apply_safety_stock(forecast_array, safety_stock, method=safety_stock_method)
    return np.maximum(forecast_array + stock, 0.0)
