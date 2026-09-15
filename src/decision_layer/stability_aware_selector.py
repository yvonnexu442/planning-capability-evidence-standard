"""Stability-aware hard model selection logic."""

from typing import Mapping, Optional

from decision_layer.selectors import select_lowest_loss_model


def select_stability_aware_model(
    model_losses: Mapping[str, float],
    previous_model: Optional[str] = None,
    switch_penalty: float = 0.0,
) -> str:
    """Select a model using an input loss plus an optional switching penalty.

    This initial selector captures the simplest version of the paper's decision
    layer: a model should not switch merely because it is slightly better on a
    diagnostic or validation loss if switching creates operational burden.

    The selector makes the planning-infrastructure gap explicit by adding a
    cost for changing model logic between planning periods.
    """
    if not model_losses:
        raise ValueError("model_losses must not be empty.")
    adjusted_losses = {}
    for model_name, loss in model_losses.items():
        penalty = 0.0 if previous_model is None or model_name == previous_model else float(switch_penalty)
        adjusted_losses[model_name] = float(loss) + penalty
    return select_lowest_loss_model(adjusted_losses)
