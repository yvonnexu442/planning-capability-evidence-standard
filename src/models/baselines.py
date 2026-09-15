"""Transparent baseline forecasting functions."""

from typing import Sequence

import numpy as np


def naive_last_value_forecast(history: Sequence[float], horizon: int) -> np.ndarray:
    """Return a last-observation-carried-forward forecast.

    This baseline is intentionally simple. It gives the paper a transparent
    reference point before introducing model selection or ensemble logic.

    A simple baseline is also useful for showing that the project is not a
    leaderboard exercise; the research contribution is the planning decision
    layer and evaluation framework.
    """
    history_array = np.asarray(history, dtype=float)
    if history_array.size == 0:
        raise ValueError("history must not be empty.")
    return np.repeat(history_array[-1], int(horizon))


def rolling_mean_forecast(history: Sequence[float], horizon: int, window_size: int) -> np.ndarray:
    """Return a rolling-mean forecast repeated over the requested horizon.

    Rolling means provide a stable and interpretable baseline. They are useful
    for comparing how smoother forecasts affect inventory cost and execution
    penalties.

    The function uses the available history when the requested window is longer
    than the observed series.
    """
    history_array = np.asarray(history, dtype=float)
    if history_array.size == 0:
        raise ValueError("history must not be empty.")
    window = min(int(window_size), history_array.size)
    return np.repeat(float(np.mean(history_array[-window:])), int(horizon))


def seasonal_naive_forecast(history: Sequence[float], horizon: int, season_length: int) -> np.ndarray:
    """Return a seasonal naive forecast.

    Seasonal naive forecasts repeat the most recent seasonal pattern. They are
    transparent enough for research baselines and can create realistic planning
    signal movement when demand has weekly or seasonal structure.
    """
    history_array = np.asarray(history, dtype=float)
    if history_array.size == 0:
        raise ValueError("history must not be empty.")
    season = history_array[-min(int(season_length), history_array.size) :]
    repeats = int(np.ceil(float(horizon) / float(season.size)))
    return np.tile(season, repeats)[: int(horizon)]
