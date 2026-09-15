"""Loader and validation utilities for the Favorita Store Sales dataset."""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from features.feature_engineering import (
    add_calendar_features,
    add_context_availability_flags,
    add_exponential_smoothing_features,
    add_lag_features,
    add_recent_demand_volatility,
    add_rolling_demand_features,
    add_zero_demand_features,
)


PathLike = Union[str, Path]

REQUIRED_FILES = {
    "train": "train.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}

OPTIONAL_FILES = {"test": "test.csv"}

REQUIRED_COLUMNS = {
    "train.csv": {"date", "store_nbr", "family", "sales", "onpromotion"},
    "stores.csv": {"store_nbr", "city", "state", "type", "cluster"},
    "oil.csv": {"date", "dcoilwtico"},
    "holidays_events.csv": {"date", "type", "locale", "locale_name", "description", "transferred"},
    "transactions.csv": {"date", "store_nbr", "transactions"},
}


def load_favorita_modeling_table(
    raw_data_dir: PathLike = "data/raw/favorita",
    max_series: Optional[int] = None,
    min_history_length: int = 90,
    min_nonzero_observations: int = 30,
    selection_holdout_periods: int = 0,
    output_table_dir: Optional[PathLike] = None,
    output_log_dir: Optional[PathLike] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Favorita data and return a modeling table plus a quality report.

    The modeling table uses store-family-date as the planning grain and creates
    a stable `series_id` for each store-family pair. It includes demand,
    promotion, store metadata, holiday/event context, oil context, transaction
    context, and leakage-aware lagged demand features.

    Transactions and oil are treated as operational context that may not be
    known at planning time. The loader keeps raw values for profiling but builds
    lagged and rolling versions for modeling. Promotions and holidays are
    treated as known-in-advance context because retailers often plan around
    promotion calendars and public holiday calendars.
    """
    raw_path = Path(raw_data_dir)
    validate_favorita_files(raw_path)

    train = pd.read_csv(raw_path / REQUIRED_FILES["train"], parse_dates=["date"])
    stores = pd.read_csv(raw_path / REQUIRED_FILES["stores"])
    oil = pd.read_csv(raw_path / REQUIRED_FILES["oil"], parse_dates=["date"])
    holidays = pd.read_csv(raw_path / REQUIRED_FILES["holidays"], parse_dates=["date"])
    transactions = pd.read_csv(raw_path / REQUIRED_FILES["transactions"], parse_dates=["date"])

    selection_source = _selection_history(
        train,
        selection_holdout_periods=selection_holdout_periods,
    )
    train = _filter_eligible_series(
        train=train,
        selection_source=selection_source,
        max_series=max_series,
        min_history_length=min_history_length,
        min_nonzero_observations=min_nonzero_observations,
    )
    train["series_id"] = make_series_id(train["store_nbr"], train["family"])
    train = train.rename(columns={"sales": "demand", "onpromotion": "promotion_count"})
    train["is_promotion"] = (train["promotion_count"].fillna(0) > 0).astype(int)

    modeling = train.merge(stores, on="store_nbr", how="left")
    modeling = modeling.rename(columns={"type": "store_type", "cluster": "store_cluster"})
    modeling = _merge_holiday_context(modeling, holidays)
    modeling = _merge_oil_context(modeling, oil)
    modeling = _merge_transaction_context(modeling, transactions)
    modeling = _add_favorita_features(modeling)
    modeling = modeling.sort_values(["series_id", "date"]).reset_index(drop=True)

    quality_report = build_favorita_quality_report(modeling)
    if output_table_dir is not None:
        Path(output_table_dir).mkdir(parents=True, exist_ok=True)
        quality_report.to_csv(Path(output_table_dir) / "favorita_data_quality_report.csv", index=False)
    if output_log_dir is not None:
        Path(output_log_dir).mkdir(parents=True, exist_ok=True)
        quality_report.to_csv(Path(output_log_dir) / "favorita_data_quality_report.csv", index=False)
    return modeling, quality_report


def validate_favorita_files(raw_data_dir: PathLike = "data/raw/favorita") -> None:
    """Validate required Favorita files and required columns.

    The function raises clear English errors when local data is missing. It does
    not attempt to download Kaggle data automatically.
    """
    raw_path = Path(raw_data_dir)
    missing_files = [file_name for file_name in REQUIRED_FILES.values() if not (raw_path / file_name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required Favorita files: {}. Place the manually downloaded CSV files under {}.".format(
                ", ".join(sorted(missing_files)),
                raw_path,
            )
        )

    for file_name, required_columns in REQUIRED_COLUMNS.items():
        available_columns = set(pd.read_csv(raw_path / file_name, nrows=0).columns)
        missing_columns = required_columns.difference(available_columns)
        if missing_columns:
            raise ValueError(
                "Favorita file {} is missing required columns: {}.".format(
                    file_name,
                    ", ".join(sorted(missing_columns)),
                )
            )


def make_series_id(store_nbr: Sequence[object], family: Sequence[object]) -> pd.Series:
    """Create stable store-family series identifiers."""
    return pd.Series(store_nbr).astype(str).str.zfill(3) + "__" + pd.Series(family).astype(str).str.lower().str.replace(
        r"[^a-z0-9]+",
        "_",
        regex=True,
    ).str.strip("_")


def build_favorita_quality_report(modeling_table: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row quality report for the loaded Favorita modeling table."""
    frame = modeling_table
    available_context_features = [
        column
        for column in [
            "promotion_count",
            "is_promotion",
            "is_holiday_event",
            "national_holiday_count",
            "regional_holiday_count",
            "local_holiday_count",
            "dcoilwtico",
            "oil_lag_1",
            "oil_rolling_mean_7",
            "transactions",
            "transactions_lag_1",
            "transactions_rolling_mean_7",
            "city",
            "state",
            "store_type",
            "store_cluster",
        ]
        if column in frame.columns
    ]
    report = {
        "row_count": len(frame),
        "series_count": frame["series_id"].nunique(),
        "start_date": str(frame["date"].min().date()),
        "end_date": str(frame["date"].max().date()),
        "missing_value_count": int(frame.isna().sum().sum()),
        "missing_value_rate": float(frame.isna().sum().sum() / max(frame.size, 1)),
        "zero_sales_rate": float((frame["demand"] == 0).mean()),
        "store_count": frame["store_nbr"].nunique(),
        "family_count": frame["family"].nunique(),
        "promotion_coverage": float((frame["promotion_count"].fillna(0) > 0).mean()),
        "holiday_event_coverage": float((frame["is_holiday_event"].fillna(0) > 0).mean()),
        "oil_observation_coverage": float(frame["dcoilwtico"].notna().mean()) if "dcoilwtico" in frame.columns else 0.0,
        "transaction_observation_coverage": float(frame["transactions"].notna().mean())
        if "transactions" in frame.columns
        else 0.0,
        "available_context_features": ", ".join(available_context_features),
    }
    return pd.DataFrame([report])


def _filter_eligible_series(
    train: pd.DataFrame,
    selection_source: Optional[pd.DataFrame],
    max_series: Optional[int],
    min_history_length: int,
    min_nonzero_observations: int,
) -> pd.DataFrame:
    """Filter the training table to eligible store-family series."""
    profile_frame = train if selection_source is None else selection_source
    grouped = (
        profile_frame.groupby(["store_nbr", "family"])
        .agg(history_length=("date", "count"), nonzero_observations=("sales", lambda values: int((values > 0).sum())), total_sales=("sales", "sum"))
        .reset_index()
    )
    eligible = grouped[
        (grouped["history_length"] >= int(min_history_length))
        & (grouped["nonzero_observations"] >= int(min_nonzero_observations))
    ].sort_values("total_sales", ascending=False)
    if max_series is not None:
        eligible = eligible.head(int(max_series))
    return train.merge(eligible[["store_nbr", "family"]], on=["store_nbr", "family"], how="inner")


def _selection_history(train: pd.DataFrame, selection_holdout_periods: int) -> pd.DataFrame:
    """Return the history available when panel membership is frozen."""
    holdout = int(selection_holdout_periods)
    if holdout < 0:
        raise ValueError("selection_holdout_periods must be nonnegative")
    if holdout == 0:
        return train
    dates = pd.Index(pd.to_datetime(train["date"]).dropna().unique()).sort_values()
    if holdout >= len(dates):
        raise ValueError("selection_holdout_periods leaves no history for panel selection")
    last_selection_date = dates[-holdout - 1]
    return train[pd.to_datetime(train["date"]) <= last_selection_date].copy()


def _merge_holiday_context(modeling: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Merge known-in-advance holiday and event context into the modeling table."""
    holidays = holidays.copy()
    holidays["transferred"] = holidays["transferred"].astype(bool)
    active_holidays = holidays[~holidays["transferred"]].copy()

    national = _holiday_counts(active_holidays[active_holidays["locale"] == "National"], ["date"], "national_holiday_count")
    regional = _holiday_counts(
        active_holidays[active_holidays["locale"] == "Regional"],
        ["date", "locale_name"],
        "regional_holiday_count",
    ).rename(columns={"locale_name": "state"})
    local = _holiday_counts(
        active_holidays[active_holidays["locale"] == "Local"],
        ["date", "locale_name"],
        "local_holiday_count",
    ).rename(columns={"locale_name": "city"})

    frame = modeling.merge(national, on="date", how="left")
    frame = frame.merge(regional, on=["date", "state"], how="left")
    frame = frame.merge(local, on=["date", "city"], how="left")
    for column in ["national_holiday_count", "regional_holiday_count", "local_holiday_count"]:
        frame[column] = frame[column].fillna(0).astype(int)
    frame["holiday_event_count"] = (
        frame["national_holiday_count"] + frame["regional_holiday_count"] + frame["local_holiday_count"]
    )
    frame["is_holiday_event"] = (frame["holiday_event_count"] > 0).astype(int)
    return frame


def _holiday_counts(holidays: pd.DataFrame, group_columns: Iterable[str], output_column: str) -> pd.DataFrame:
    """Aggregate holiday rows into event counts."""
    if holidays.empty:
        return pd.DataFrame(columns=list(group_columns) + [output_column])
    return holidays.groupby(list(group_columns)).size().reset_index(name=output_column)


def _merge_oil_context(modeling: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    """Merge oil context and leakage-aware lagged oil features."""
    oil_frame = oil.sort_values("date").copy()
    full_dates = pd.date_range(oil_frame["date"].min(), oil_frame["date"].max(), freq="D")
    oil_frame = oil_frame.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
    oil_frame["dcoilwtico"] = oil_frame["dcoilwtico"].ffill()
    oil_frame["oil_lag_1"] = oil_frame["dcoilwtico"].shift(1)
    oil_frame["oil_lag_7"] = oil_frame["dcoilwtico"].shift(7)
    oil_frame["oil_rolling_mean_7"] = oil_frame["dcoilwtico"].shift(1).rolling(7, min_periods=1).mean()
    return modeling.merge(oil_frame, on="date", how="left")


def _merge_transaction_context(modeling: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Merge transaction context and leakage-aware transaction features."""
    tx = transactions.sort_values(["store_nbr", "date"]).copy()
    tx["transactions_lag_1"] = tx.groupby("store_nbr")["transactions"].shift(1)
    tx["transactions_lag_7"] = tx.groupby("store_nbr")["transactions"].shift(7)
    tx["transactions_rolling_mean_7"] = (
        tx.groupby("store_nbr")["transactions"].shift(1).groupby(tx["store_nbr"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    return modeling.merge(tx, on=["date", "store_nbr"], how="left")


def _add_favorita_features(modeling: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable forecasting and context features to Favorita data."""
    frame = add_calendar_features(modeling, "date")
    frame = add_lag_features(frame, ["series_id"], "date", "demand", lags=[1, 7, 14, 28])
    frame = add_rolling_demand_features(frame, ["series_id"], "date", "demand", windows=[7, 14, 28, 56])
    frame = add_recent_demand_volatility(frame, ["series_id"], "date", "demand", windows=[7, 28, 56])
    frame = add_exponential_smoothing_features(frame, ["series_id"], "date", "demand", alphas=[0.3])
    frame = add_zero_demand_features(frame, ["series_id"], "date", "demand", windows=[7, 28])
    frame = add_context_availability_flags(
        frame,
        context_columns=[
            "promotion_count",
            "is_holiday_event",
            "oil_lag_1",
            "oil_rolling_mean_7",
            "transactions_lag_1",
            "transactions_rolling_mean_7",
            "city",
            "state",
            "store_type",
            "store_cluster",
        ],
    )
    return frame
