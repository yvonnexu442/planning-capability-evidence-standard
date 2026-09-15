"""Loader and validation utilities for the M5 retail demand dataset."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

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
    "calendar": "calendar.csv",
    "sell_prices": "sell_prices.csv",
    "sales": "sales_train_validation.csv",
}

OPTIONAL_FILES = {
    "sales_evaluation": "sales_train_evaluation.csv",
    "sample_submission": "sample_submission.csv",
}

REQUIRED_COLUMNS = {
    "calendar.csv": {
        "date",
        "wm_yr_wk",
        "d",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    },
    "sell_prices.csv": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    "sales_train_validation.csv": {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"},
}


def validate_m5_files(raw_data_dir: PathLike = "data/raw/m5") -> None:
    """Validate required M5 files and columns with clear English errors."""
    raw_path = Path(raw_data_dir)
    missing_files = [file_name for file_name in REQUIRED_FILES.values() if not (raw_path / file_name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required M5 files: {}. Place the manually downloaded M5 CSV files under {}.".format(
                ", ".join(sorted(missing_files)),
                raw_path,
            )
        )

    for file_name, required_columns in REQUIRED_COLUMNS.items():
        available_columns = set(pd.read_csv(raw_path / file_name, nrows=0).columns)
        if file_name == "sales_train_validation.csv":
            has_day_columns = any(column.startswith("d_") for column in available_columns)
            if not has_day_columns:
                raise ValueError("M5 sales_train_validation.csv must contain daily columns named d_1, d_2, and so on.")
        missing_columns = required_columns.difference(available_columns)
        if missing_columns:
            raise ValueError(
                "M5 file {} is missing required columns: {}.".format(
                    file_name,
                    ", ".join(sorted(missing_columns)),
                )
            )


def load_m5_modeling_table(
    raw_data_dir: PathLike = "data/raw/m5",
    run_mode: str = "quick",
    max_series: Optional[int] = None,
    min_history_length: int = 365,
    min_nonzero_observations: int = 10,
    selection_holdout_periods: int = 0,
    output_table_dir: Optional[PathLike] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 data and return a normalized item-store daily modeling table.

    The loader converts the wide M5 daily sales format into a long panel at the
    item-store planning grain, then merges calendar, SNAP, event, and sell-price
    context. It intentionally samples eligible series for quick and medium
    robustness experiments so M5 supports the paper without becoming a separate
    leaderboard project.
    """
    raw_path = Path(raw_data_dir)
    validate_m5_files(raw_path)

    sales = pd.read_csv(raw_path / REQUIRED_FILES["sales"])
    day_columns = sorted(
        (column for column in sales.columns if column.startswith("d_")),
        key=lambda column: int(column.split("_", 1)[1]),
    )
    selection_day_columns = _selection_day_columns(day_columns, selection_holdout_periods)
    selected_sales = select_m5_series(
        sales=sales,
        day_columns=day_columns,
        profile_day_columns=selection_day_columns,
        max_series=max_series,
        min_history_length=min_history_length,
        min_nonzero_observations=min_nonzero_observations,
    )
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long_sales = selected_sales.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="demand",
    )
    long_sales["series_id"] = make_m5_series_id(long_sales["item_id"], long_sales["store_id"])
    long_sales["demand"] = pd.to_numeric(long_sales["demand"], errors="coerce").fillna(0.0).astype(float)

    calendar = pd.read_csv(raw_path / REQUIRED_FILES["calendar"], parse_dates=["date"])
    calendar = prepare_m5_calendar(calendar)
    modeling = long_sales.merge(calendar, on="d", how="left")
    modeling = merge_m5_sell_prices(modeling, raw_path / REQUIRED_FILES["sell_prices"])
    modeling = add_m5_features(modeling)
    modeling = modeling.sort_values(["series_id", "date"]).reset_index(drop=True)

    quality_report = build_m5_quality_report(
        modeling_table=modeling,
        run_mode=run_mode,
        selected_series_count=selected_sales.shape[0],
        available_series_count=sales.shape[0],
    )
    if output_table_dir is not None:
        output_path = Path(output_table_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        quality_report.to_csv(output_path / "m5_data_quality_report.csv", index=False)
    return modeling, quality_report


def load_m5_dataset(raw_data_dir: PathLike) -> pd.DataFrame:
    """Backward-compatible loader returning the normalized M5 modeling table."""
    modeling, _ = load_m5_modeling_table(raw_data_dir=raw_data_dir)
    return modeling


def select_m5_series(
    sales: pd.DataFrame,
    day_columns: Sequence[str],
    max_series: Optional[int],
    min_history_length: int,
    min_nonzero_observations: int,
    profile_day_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Select eligible item-store series before melting the wide table."""
    profile_columns = list(day_columns if profile_day_columns is None else profile_day_columns)
    if len(profile_columns) < int(min_history_length):
        raise ValueError(
            "M5 sales history has {} daily columns, below the required minimum history length {}.".format(
                len(profile_columns),
                int(min_history_length),
            )
        )
    demand = sales[profile_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    profile = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    profile["total_demand"] = demand.sum(axis=1)
    profile["nonzero_observations"] = (demand > 0).sum(axis=1)
    eligible_ids = profile[profile["nonzero_observations"] >= int(min_nonzero_observations)].sort_values(
        ["total_demand", "nonzero_observations", "id"],
        ascending=[False, False, True],
    )
    if max_series is not None:
        eligible_ids = eligible_ids.head(int(max_series))
    return sales.merge(eligible_ids[["id"]], on="id", how="inner")


def _selection_day_columns(day_columns: Sequence[str], selection_holdout_periods: int) -> Sequence[str]:
    """Return M5 day columns available when panel membership is frozen."""
    holdout = int(selection_holdout_periods)
    if holdout < 0:
        raise ValueError("selection_holdout_periods must be nonnegative")
    if holdout == 0:
        return list(day_columns)
    if holdout >= len(day_columns):
        raise ValueError("selection_holdout_periods leaves no history for panel selection")
    return list(day_columns[:-holdout])


def prepare_m5_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Prepare calendar, event, and SNAP context features."""
    frame = calendar.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        frame[column] = frame[column].fillna("none").astype(str)
    frame["event_count"] = (frame["event_name_1"] != "none").astype(int) + (frame["event_name_2"] != "none").astype(int)
    frame["has_event"] = (frame["event_count"] > 0).astype(int)
    for column in ["snap_CA", "snap_TX", "snap_WI"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    return frame


def merge_m5_sell_prices(modeling: pd.DataFrame, sell_prices_path: PathLike) -> pd.DataFrame:
    """Merge sell-price context for selected item-store-week combinations."""
    frame = modeling.copy()
    pairs = frame[["store_id", "item_id"]].drop_duplicates()
    prices = pd.read_csv(sell_prices_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    prices = prices.merge(pairs, on=["store_id", "item_id"], how="inner")
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    frame = frame.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    frame = frame.sort_values(["series_id", "date"])
    frame["sell_price"] = frame.groupby("series_id")["sell_price"].transform(lambda values: values.ffill().bfill()).fillna(0.0)
    frame["price_available"] = (frame["sell_price"] > 0).astype(int)
    frame["snap_active"] = np.select(
        [
            frame["state_id"].eq("CA"),
            frame["state_id"].eq("TX"),
            frame["state_id"].eq("WI"),
        ],
        [
            frame["snap_CA"],
            frame["snap_TX"],
            frame["snap_WI"],
        ],
        default=0,
    ).astype(int)
    return frame


def add_m5_features(modeling: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-aware demand, calendar, price, and context features."""
    frame = modeling.copy()
    frame = add_calendar_features(frame, date_column="date")
    frame = add_lag_features(frame, ["series_id"], "date", "demand", lags=[1, 7, 28])
    frame = add_rolling_demand_features(frame, ["series_id"], "date", "demand", windows=[7, 28])
    frame = add_recent_demand_volatility(frame, ["series_id"], "date", "demand", windows=[7, 28])
    frame = add_exponential_smoothing_features(frame, ["series_id"], "date", "demand", alphas=[0.3])
    frame = add_zero_demand_features(frame, ["series_id"], "date", "demand", windows=[28])
    frame["sell_price_lag_1"] = frame.groupby("series_id")["sell_price"].shift(1)
    frame["sell_price_rolling_mean_7"] = (
        frame.groupby("series_id")["sell_price"]
        .shift(1)
        .groupby(frame["series_id"])
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame = add_context_availability_flags(
        frame,
        context_columns=["sell_price", "has_event", "snap_active", "event_count"],
    )
    numeric_feature_columns = [
        "demand_lag_1",
        "demand_lag_7",
        "demand_lag_28",
        "demand_rolling_mean_7",
        "demand_rolling_mean_28",
        "demand_rolling_std_7",
        "demand_rolling_std_28",
        "demand_ewm_alpha_0_3",
        "zero_demand_rate_28",
        "sell_price_lag_1",
        "sell_price_rolling_mean_7",
    ]
    for column in numeric_feature_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def make_m5_series_id(item_id: Sequence[object], store_id: Sequence[object]) -> pd.Series:
    """Create stable item-store series identifiers."""
    return pd.Series(item_id).astype(str) + "__" + pd.Series(store_id).astype(str)


def build_m5_quality_report(
    modeling_table: pd.DataFrame,
    run_mode: str,
    selected_series_count: int,
    available_series_count: int,
) -> pd.DataFrame:
    """Build a one-row M5 quality and scope report."""
    frame = modeling_table
    return pd.DataFrame(
        [
            {
                "dataset_name": "m5",
                "run_mode": run_mode,
                "row_count": len(frame),
                "selected_series_count": int(selected_series_count),
                "available_series_count": int(available_series_count),
                "start_date": str(frame["date"].min().date()),
                "end_date": str(frame["date"].max().date()),
                "item_count": frame["item_id"].nunique(),
                "store_count": frame["store_id"].nunique(),
                "department_count": frame["dept_id"].nunique(),
                "category_count": frame["cat_id"].nunique(),
                "zero_demand_rate": float((frame["demand"] == 0).mean()),
                "event_day_rate": float(frame["has_event"].mean()),
                "snap_active_rate": float(frame["snap_active"].mean()),
                "price_available_rate": float(frame["price_available"].mean()),
            }
        ]
    )
