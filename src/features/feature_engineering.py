"""Basic feature engineering utilities for chronological demand tables."""

from typing import Iterable

import pandas as pd


def add_calendar_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Return a copy of the data with transparent calendar features.

    Calendar features are simple and interpretable, which is important for a
    research repository focused on planning logic rather than leaderboard model
    complexity.

    The function avoids modifying input data in place so experiments remain
    reproducible and easy to audit.
    """
    frame = data.copy()
    dates = pd.to_datetime(frame[date_column])
    frame["day_of_week"] = dates.dt.dayofweek
    frame["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    frame["month"] = dates.dt.month
    frame["year"] = dates.dt.year
    return frame


def add_lag_features(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    target_column: str,
    lags: Iterable[int],
) -> pd.DataFrame:
    """Return a copy of the data with demand lag features.

    Lag features are common transparent baselines for demand forecasting. They
    are included because they help build understandable candidate forecasts
    before adding more complex models.

    The chronological sort protects against lookahead leakage.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    for lag in lags:
        frame["{}_lag_{}".format(target_column, int(lag))] = frame.groupby(id_columns)[target_column].shift(int(lag))
    return frame


def add_rolling_demand_features(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    target_column: str,
    windows: Iterable[int],
) -> pd.DataFrame:
    """Return a copy with leakage-aware rolling demand means.

    Rolling demand features are shifted by one period before aggregation. This
    ensures that the feature available at date `t` uses only demand observed
    before date `t`.

    These features are intentionally interpretable because the project studies
    planning behavior rather than leaderboard forecasting accuracy.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    shifted = frame.groupby(id_columns)[target_column].shift(1)
    for window in windows:
        feature_name = "{}_rolling_mean_{}".format(target_column, int(window))
        frame[feature_name] = (
            shifted.groupby([frame[column] for column in id_columns])
            .rolling(int(window), min_periods=1)
            .mean()
            .reset_index(level=list(range(len(id_columns))), drop=True)
        )
    return frame


def add_recent_demand_volatility(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    target_column: str,
    windows: Iterable[int],
) -> pd.DataFrame:
    """Return a copy with recent demand volatility features.

    Recent volatility is measured as a shifted rolling standard deviation. It
    helps the pipeline create safety stock and planning-risk features without
    using future demand.

    This is useful for the paper because unstable demand can create unstable
    planning signals even when forecast models are simple.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    shifted = frame.groupby(id_columns)[target_column].shift(1)
    for window in windows:
        feature_name = "{}_rolling_std_{}".format(target_column, int(window))
        frame[feature_name] = (
            shifted.groupby([frame[column] for column in id_columns])
            .rolling(int(window), min_periods=2)
            .std()
            .reset_index(level=list(range(len(id_columns))), drop=True)
            .fillna(0.0)
        )
    return frame


def add_exponential_smoothing_features(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    target_column: str,
    alphas: Iterable[float],
) -> pd.DataFrame:
    """Return a copy with leakage-aware exponential smoothing features.

    The smoothed value at date `t` is computed after shifting demand by one
    period, so it uses only demand observed before date `t`. This gives the
    minimal Favorita pipeline an interpretable exponential-smoothing candidate
    without relying on a heavier time-series package.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    shifted = frame.groupby(id_columns)[target_column].shift(1)
    group_keys = [frame[column] for column in id_columns]
    for alpha in alphas:
        alpha_value = float(alpha)
        alpha_label = str(alpha_value).replace(".", "_")
        feature_name = "{}_ewm_alpha_{}".format(target_column, alpha_label)
        frame[feature_name] = (
            shifted.groupby(group_keys)
            .transform(lambda values: values.ewm(alpha=alpha_value, adjust=False, min_periods=1).mean())
            .fillna(0.0)
        )
    return frame


def add_zero_demand_features(
    data: pd.DataFrame,
    series_id_columns: Iterable[str],
    date_column: str,
    target_column: str,
    windows: Iterable[int],
) -> pd.DataFrame:
    """Return a copy with zero-demand and intermittent-demand indicators.

    Zero-demand behavior matters for operational planning because sparse demand
    can make forecast accuracy and planning stability tradeoffs more visible.

    The rolling zero rates are shifted by one period so they can be used as
    planning-time features.
    """
    id_columns = list(series_id_columns)
    frame = data.copy().sort_values(id_columns + [date_column])
    frame["is_zero_demand"] = (frame[target_column] == 0).astype(int)
    shifted_zero = frame.groupby(id_columns)["is_zero_demand"].shift(1)
    for window in windows:
        feature_name = "zero_demand_rate_{}".format(int(window))
        frame[feature_name] = (
            shifted_zero.groupby([frame[column] for column in id_columns])
            .rolling(int(window), min_periods=1)
            .mean()
            .reset_index(level=list(range(len(id_columns))), drop=True)
            .fillna(0.0)
        )
    return frame


def add_context_availability_flags(data: pd.DataFrame, context_columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with flags showing whether context signals are available.

    Context availability matters because real planning systems may not know all
    context variables at planning time. Explicit flags make the assumption
    visible and auditable.

    Promotion and holiday features are generally treated as known-in-advance,
    while oil and transaction features should use lagged or rolling values.
    """
    frame = data.copy()
    for column in context_columns:
        if column in frame.columns:
            frame["{}_available".format(column)] = frame[column].notna().astype(int)
    availability_columns = [column for column in frame.columns if column.endswith("_available")]
    if availability_columns:
        frame["known_context_available"] = frame[availability_columns].min(axis=1)
    else:
        frame["known_context_available"] = 0
    return frame
