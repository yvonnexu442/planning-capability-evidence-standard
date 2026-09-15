"""Loader and profiling utilities for Walmart weekly retail sales data."""

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
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
    "features": "features.csv",
    "stores": "stores.csv",
}

REQUIRED_COLUMNS = {
    "train.csv": {"Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"},
    "features.csv": {"Store", "Date", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"},
    "stores.csv": {"Store", "Type", "Size"},
}

MARKDOWN_COLUMNS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
CONTEXT_COLUMNS = [
    "temperature",
    "fuel_price",
    "markdown_1",
    "markdown_2",
    "markdown_3",
    "markdown_4",
    "markdown_5",
    "cpi",
    "unemployment",
    "is_holiday",
    "store_type",
    "store_size",
]


def validate_walmart_files(raw_data_dir: PathLike) -> Dict[str, Path]:
    """Validate that the Walmart raw-data directory contains required files."""
    base = Path(raw_data_dir)
    if not base.exists():
        raise FileNotFoundError("Walmart raw data directory does not exist: {}".format(base))

    paths = {name: base / filename for name, filename in REQUIRED_FILES.items()}
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing Walmart raw data files: {}".format(", ".join(missing_files)))

    for filename, required_columns in REQUIRED_COLUMNS.items():
        path = base / filename
        available = set(pd.read_csv(path, nrows=0).columns)
        missing_columns = sorted(required_columns.difference(available))
        if missing_columns:
            raise ValueError("{} is missing required columns: {}".format(path, ", ".join(missing_columns)))
    return paths


def load_walmart_dataset(raw_data_dir: PathLike) -> pd.DataFrame:
    """Load the merged Walmart data without sampling or split assignment."""
    paths = validate_walmart_files(raw_data_dir)
    return _load_merged_walmart_data(paths)


def load_walmart_modeling_table(
    raw_data_dir: PathLike = "data/raw/walmart",
    run_mode: str = "quick",
    max_series: Optional[int] = None,
    min_history_length: int = 80,
    min_nonzero_observations: int = 20,
    random_seed: int = 42,
    selection_holdout_periods: int = 0,
    output_table_dir: Optional[PathLike] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return a sampled weekly modeling table and a data quality report.

    Walmart is used as a business-context robustness check. The table keeps the
    weekly cadence of the source data and exposes interpretable history and
    context features for context-aware versus context-free forecasting.
    """
    paths = validate_walmart_files(raw_data_dir)
    merged = _load_merged_walmart_data(paths)
    selection_source = _selection_history(
        merged,
        selection_holdout_periods=selection_holdout_periods,
    )
    selected_series = select_walmart_series(
        selection_source,
        max_series=max_series,
        min_history_length=min_history_length,
        min_nonzero_observations=min_nonzero_observations,
        random_seed=random_seed,
    )
    if selected_series.empty:
        raise ValueError("No Walmart series met the configured history and nonzero-demand requirements.")

    modeling = merged[merged["series_id"].isin(selected_series["series_id"])].copy()
    modeling = add_walmart_features(modeling)
    modeling["run_mode"] = str(run_mode)
    modeling = modeling.sort_values(["series_id", "date"]).reset_index(drop=True)

    quality_report = build_walmart_quality_report(modeling, selected_series)
    if output_table_dir is not None:
        output_path = Path(output_table_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        quality_report.to_csv(output_path / "walmart_data_quality_report.csv", index=False)
    return modeling, quality_report


def select_walmart_series(
    data: pd.DataFrame,
    max_series: Optional[int],
    min_history_length: int,
    min_nonzero_observations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Return eligible store-department series for the configured sample size."""
    profile = (
        data.groupby(["series_id", "store", "dept"], dropna=False)
        .agg(
            history_length=("date", "nunique"),
            nonzero_observations=("weekly_sales", lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0.0) > 0.0).sum())),
            total_sales=("weekly_sales", "sum"),
            mean_sales=("weekly_sales", "mean"),
        )
        .reset_index()
    )
    eligible = profile[
        (profile["history_length"] >= int(min_history_length))
        & (profile["nonzero_observations"] >= int(min_nonzero_observations))
    ].copy()
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(["total_sales", "history_length", "series_id"], ascending=[False, False, True])
    if max_series is not None and len(eligible) > int(max_series):
        sample_size = int(max_series)
        high_volume_size = int(np.ceil(sample_size * 0.70))
        high_volume = eligible.head(high_volume_size)
        remainder = eligible.iloc[high_volume_size:]
        random_size = max(sample_size - len(high_volume), 0)
        if random_size > 0 and not remainder.empty:
            sampled = remainder.sample(n=min(random_size, len(remainder)), random_state=int(random_seed))
            eligible = pd.concat([high_volume, sampled], ignore_index=True)
        else:
            eligible = high_volume
    return eligible.sort_values("series_id").reset_index(drop=True)


def _selection_history(data: pd.DataFrame, selection_holdout_periods: int) -> pd.DataFrame:
    """Return the history available when panel membership is frozen."""
    holdout = int(selection_holdout_periods)
    if holdout < 0:
        raise ValueError("selection_holdout_periods must be nonnegative")
    if holdout == 0:
        return data
    dates = pd.Index(pd.to_datetime(data["date"]).dropna().unique()).sort_values()
    if holdout >= len(dates):
        raise ValueError("selection_holdout_periods leaves no history for panel selection")
    last_selection_date = dates[-holdout - 1]
    return data[pd.to_datetime(data["date"]) <= last_selection_date].copy()


def make_walmart_series_id(store: object, dept: object) -> str:
    """Return a stable store-department series identifier."""
    return "store_{}__dept_{}".format(_format_numeric_id(store, 3), _format_numeric_id(dept, 3))


def add_walmart_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-aware weekly history, volatility, and context features."""
    frame = data.copy().sort_values(["series_id", "date"])
    frame = add_calendar_features(frame, "date")
    frame = add_lag_features(frame, ["series_id"], "date", "demand", [1, 4, 13, 26])
    frame = add_rolling_demand_features(frame, ["series_id"], "date", "demand", [4, 13])
    frame = add_recent_demand_volatility(frame, ["series_id"], "date", "demand", [4, 13])
    frame = add_exponential_smoothing_features(frame, ["series_id"], "date", "demand", [0.3])
    frame = add_zero_demand_features(frame, ["series_id"], "date", "demand", [13])
    frame = add_context_availability_flags(frame, CONTEXT_COLUMNS)
    frame["markdown_total"] = frame[["markdown_{}".format(index) for index in range(1, 6)]].sum(axis=1)
    frame["demand_lag_1"] = frame["demand_lag_1"].fillna(frame.groupby("series_id")["demand"].transform("median")).fillna(0.0)
    for column in [
        "demand_lag_4",
        "demand_lag_13",
        "demand_lag_26",
        "demand_rolling_mean_4",
        "demand_rolling_mean_13",
        "demand_rolling_std_4",
        "demand_rolling_std_13",
        "demand_ewm_alpha_0_3",
        "zero_demand_rate_13",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(frame["demand_lag_1"]).fillna(0.0)
    frame["demand_rolling_std_28"] = frame["demand_rolling_std_13"]
    frame["zero_demand_rate_28"] = frame["zero_demand_rate_13"]
    frame["item_id"] = frame["dept_id"]
    frame["dept_id"] = frame["dept_id"]
    frame["cat_id"] = frame["store_type"].fillna("unknown_type").astype(str)
    frame["store_id"] = frame["store_id"].astype(str)
    frame["state_id"] = frame["store_type"].fillna("unknown_type").astype(str)
    return frame


def build_walmart_quality_report(modeling_table: pd.DataFrame, selected_series: pd.DataFrame) -> pd.DataFrame:
    """Return a compact English data-profile report for Walmart."""
    markdown_total = pd.to_numeric(modeling_table.get("markdown_total", 0.0), errors="coerce").fillna(0.0)
    records = [
        {"metric": "row_count", "value": int(len(modeling_table))},
        {"metric": "series_count", "value": int(modeling_table["series_id"].nunique())},
        {"metric": "store_count", "value": int(modeling_table["store"].nunique())},
        {"metric": "department_count", "value": int(modeling_table["dept"].nunique())},
        {"metric": "date_count", "value": int(modeling_table["date"].nunique())},
        {"metric": "start_date", "value": str(pd.to_datetime(modeling_table["date"]).min().date())},
        {"metric": "end_date", "value": str(pd.to_datetime(modeling_table["date"]).max().date())},
        {"metric": "selected_series_min_history", "value": int(selected_series["history_length"].min())},
        {"metric": "selected_series_median_history", "value": float(selected_series["history_length"].median())},
        {"metric": "holiday_row_share", "value": float(pd.to_numeric(modeling_table["is_holiday"], errors="coerce").fillna(0.0).mean())},
        {"metric": "markdown_positive_row_share", "value": float((markdown_total > 0.0).mean())},
        {"metric": "context_available_share", "value": float(pd.to_numeric(modeling_table["known_context_available"], errors="coerce").fillna(0.0).mean())},
    ]
    return pd.DataFrame(records)


def _load_merged_walmart_data(paths: Mapping[str, Path]) -> pd.DataFrame:
    train = pd.read_csv(paths["train"], parse_dates=["Date"])
    features = pd.read_csv(paths["features"], parse_dates=["Date"])
    stores = pd.read_csv(paths["stores"])

    for column in MARKDOWN_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0

    features_no_holiday = features.drop(columns=["IsHoliday"], errors="ignore")
    merged = train.merge(features_no_holiday, on=["Store", "Date"], how="left")
    merged = merged.merge(stores, on="Store", how="left")

    rename_map = {
        "Store": "store",
        "Dept": "dept",
        "Date": "date",
        "Weekly_Sales": "weekly_sales",
        "IsHoliday": "is_holiday",
        "Temperature": "temperature",
        "Fuel_Price": "fuel_price",
        "MarkDown1": "markdown_1",
        "MarkDown2": "markdown_2",
        "MarkDown3": "markdown_3",
        "MarkDown4": "markdown_4",
        "MarkDown5": "markdown_5",
        "CPI": "cpi",
        "Unemployment": "unemployment",
        "Type": "store_type",
        "Size": "store_size",
    }
    merged = merged.rename(columns=rename_map)
    merged["date"] = pd.to_datetime(merged["date"])
    merged["weekly_sales"] = pd.to_numeric(merged["weekly_sales"], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["demand"] = merged["weekly_sales"]
    merged["is_holiday"] = merged["is_holiday"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    merged["store"] = pd.to_numeric(merged["store"], errors="coerce").astype("Int64")
    merged["dept"] = pd.to_numeric(merged["dept"], errors="coerce").astype("Int64")
    merged["store_id"] = merged["store"].map(lambda value: "store_{}".format(_format_numeric_id(value, 3)))
    merged["dept_id"] = merged["dept"].map(lambda value: "dept_{}".format(_format_numeric_id(value, 3)))
    merged["series_id"] = [make_walmart_series_id(store, dept) for store, dept in zip(merged["store"], merged["dept"])]

    for column in ["temperature", "fuel_price", "cpi", "unemployment", "store_size"] + ["markdown_{}".format(index) for index in range(1, 6)]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    merged = merged.sort_values(["store", "dept", "date"]).reset_index(drop=True)
    context_fill_columns = ["temperature", "fuel_price", "cpi", "unemployment", "store_size"]
    for column in context_fill_columns:
        merged[column] = merged.groupby("store")[column].transform(lambda values: values.ffill().bfill())
        merged[column] = merged[column].fillna(merged[column].median())
    for column in ["markdown_{}".format(index) for index in range(1, 6)]:
        merged[column] = merged[column].fillna(0.0).clip(lower=0.0)
    merged["store_type"] = merged["store_type"].fillna("unknown_type").astype(str)

    keep_columns = [
        "date",
        "series_id",
        "store",
        "dept",
        "store_id",
        "dept_id",
        "weekly_sales",
        "demand",
        "is_holiday",
        "store_type",
        "store_size",
        "temperature",
        "fuel_price",
        "markdown_1",
        "markdown_2",
        "markdown_3",
        "markdown_4",
        "markdown_5",
        "cpi",
        "unemployment",
    ]
    return merged[keep_columns].sort_values(["series_id", "date"]).reset_index(drop=True)


def _format_numeric_id(value: object, width: int) -> str:
    if pd.isna(value):
        return "unknown"
    try:
        return str(int(value)).zfill(int(width))
    except (TypeError, ValueError):
        return str(value)
