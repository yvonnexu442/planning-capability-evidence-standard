"""Generic selector utilities for hard and soft forecast selection."""

from typing import Dict, Mapping

import numpy as np


def select_lowest_loss_model(model_losses: Mapping[str, float]) -> str:
    """Return the model name with the lowest loss.

    Hard selection is a simple decision-layer baseline. It matters because
    switching the selected model can create governance and execution burden even
    when the selected forecast is numerically accurate.
    """
    if not model_losses:
        raise ValueError("model_losses must not be empty.")
    return min(model_losses, key=lambda model_name: float(model_losses[model_name]))


def normalize_weights(raw_weights: Mapping[str, float], epsilon: float = 1e-8) -> Dict[str, float]:
    """Normalize non-negative model weights so they sum to one.

    Soft composition can reduce hard model switching, but changing weights can
    still create monitoring and governance instability. Normalized weights make
    that movement measurable.
    """
    if not raw_weights:
        raise ValueError("raw_weights must not be empty.")
    clipped = {name: max(float(weight), 0.0) for name, weight in raw_weights.items()}
    total = max(float(sum(clipped.values())), float(epsilon))
    return {name: weight / total for name, weight in clipped.items()}


def combine_forecasts(candidate_forecasts: Mapping[str, np.ndarray], model_weights: Mapping[str, float]) -> np.ndarray:
    """Return a weighted forecast composition.

    This function implements the soft composition view from the mathematical
    model. It keeps ensemble construction separate from metric evaluation so the
    paper can compare hard selection and soft composition under the same
    planning utility framework.
    """
    weights = normalize_weights(model_weights)
    missing = set(weights).difference(set(candidate_forecasts))
    if missing:
        raise ValueError("Missing forecasts for weighted models: {}".format(sorted(missing)))

    combined = None
    for model_name, weight in weights.items():
        forecast = np.asarray(candidate_forecasts[model_name], dtype=float)
        combined = forecast * weight if combined is None else combined + forecast * weight
    return combined
