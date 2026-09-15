"""Loader and profiling utilities for the DataCo supply chain dataset."""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd


PathLike = Union[str, Path]

ORDERS_FILE = "DataCoSupplyChainDataset.csv"
DESCRIPTION_FILE = "DescriptionDataCoSupplyChain.csv"
ACCESS_LOG_FILE = "tokenized_access_logs.csv"

ORDER_DATE_COLUMN = "order_date_dateorders"
SHIPPING_DATE_COLUMN = "shipping_date_dateorders"

CUSTOMER_IDENTITY_COLUMNS = {
    "customer_email",
    "customer_fname",
    "customer_lname",
    "customer_password",
    "customer_street",
    "customer_zipcode",
}


def load_dataco_orders(
    raw_data_dir: PathLike = "data/raw/dataco",
    nrows: Optional[int] = None,
    normalize_columns: bool = True,
    drop_customer_identity_columns: bool = True,
) -> pd.DataFrame:
    """Load the main DataCo order-item table.

    The main table is useful for shipment delay, delivery risk, fulfillment, and
    demand aggregation analysis. Each row represents an order item with product,
    customer, geography, order, shipping, and financial fields.

    The loader parses order and shipping dates and adds explicit delay fields so
    later research code can evaluate delivery risk and execution burden without
    repeatedly reconstructing these basics.
    """
    path = _resolve_dataco_file(raw_data_dir, ORDERS_FILE)
    frame = _read_csv_with_fallback_encoding(path, nrows=nrows)
    if normalize_columns:
        frame = normalize_column_names(frame)

    frame = _add_order_date_features(frame)
    frame = _add_shipping_delay_features(frame)

    if drop_customer_identity_columns:
        frame = frame.drop(columns=[col for col in CUSTOMER_IDENTITY_COLUMNS if col in frame.columns])
    return frame


def load_dataco_description(raw_data_dir: PathLike = "data/raw/dataco") -> pd.DataFrame:
    """Load the DataCo field description table.

    The description table is small metadata. It helps paper authors understand
    field meanings and decide which columns are appropriate for forecasting,
    fulfillment, risk, and planning-stability analyses.
    """
    path = _resolve_dataco_file(raw_data_dir, DESCRIPTION_FILE)
    frame = _read_csv_with_fallback_encoding(path)
    return normalize_column_names(frame)


def load_dataco_access_logs(
    raw_data_dir: PathLike = "data/raw/dataco",
    nrows: Optional[int] = None,
    normalize_columns: bool = True,
) -> pd.DataFrame:
    """Load tokenized web access logs associated with DataCo products.

    Access logs are not direct fulfillment records. They can be used as context
    signals or robustness features because they describe product interest over
    time. Their date range is shorter than the order table, so they should be
    used carefully in experiments.
    """
    path = _resolve_dataco_file(raw_data_dir, ACCESS_LOG_FILE)
    frame = _read_csv_with_fallback_encoding(path, nrows=nrows)
    if normalize_columns:
        frame = normalize_column_names(frame)
    if "date" in frame.columns:
        frame["access_datetime"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["access_date"] = frame["access_datetime"].dt.floor("D")
    return frame


def build_dataco_daily_product_panel(
    orders: pd.DataFrame,
    group_columns: Sequence[str] = ("product_card_id", "product_name", "category_id", "category_name"),
) -> pd.DataFrame:
    """Aggregate order items into a daily product-level planning panel.

    This panel is useful for exploratory demand and planning stability
    simulations. It is not a perfect inventory planning dataset because DataCo
    does not include on-hand inventory, replenishment decisions, or warehouse
    capacity.

    The resulting table can support context-aware planning experiments by using
    daily quantity, sales, late delivery rate, and shipment delay summaries.
    """
    required_columns = {"order_date_day", "order_item_quantity", "sales", "late_delivery_risk"}
    missing = required_columns.difference(set(orders.columns))
    if missing:
        raise ValueError("Orders table is missing required columns: {}".format(sorted(missing)))

    available_group_columns = [column for column in group_columns if column in orders.columns]
    grouped = (
        orders.groupby(["order_date_day"] + available_group_columns, dropna=False)
        .agg(
            demand_units=("order_item_quantity", "sum"),
            sales_total=("sales", "sum"),
            order_item_count=("order_item_id", "count"),
            late_delivery_rate=("late_delivery_risk", "mean"),
            average_shipment_delay_days=("shipment_delay_days", "mean"),
        )
        .reset_index()
        .rename(columns={"order_date_day": "date"})
    )
    return grouped.sort_values(["date"] + available_group_columns).reset_index(drop=True)


def profile_dataco_dataset(
    raw_data_dir: PathLike = "data/raw/dataco",
    output_dir: Optional[PathLike] = None,
    nrows: Optional[int] = None,
    include_access_logs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Profile DataCo and assess which research use cases it supports.

    The returned dictionary contains English-only tables that can be saved to
    `outputs/tables/` and later exported into the LaTeX manuscript. The profiling
    focuses on the research question: whether DataCo is best used for shipment
    risk, fulfillment planning, inventory/distribution planning, planning
    stability simulation, or supply-chain context robustness.
    """
    orders = load_dataco_orders(raw_data_dir=raw_data_dir, nrows=nrows, drop_customer_identity_columns=False)
    column_profile = build_column_profile(orders)
    dataset_summary = build_dataco_dataset_summary(orders)
    categorical_profile = build_dataco_categorical_profile(orders)
    daily_panel_summary = build_daily_panel_summary(orders)
    research_fit = assess_dataco_research_fit(dataset_summary, daily_panel_summary)

    results: Dict[str, pd.DataFrame] = {
        "dataco_dataset_summary": dataset_summary,
        "dataco_column_profile": column_profile,
        "dataco_categorical_profile": categorical_profile,
        "dataco_daily_panel_summary": daily_panel_summary,
        "dataco_research_fit": research_fit,
    }

    if include_access_logs:
        access_logs = load_dataco_access_logs(raw_data_dir=raw_data_dir, nrows=nrows)
        results["dataco_access_log_summary"] = build_access_log_summary(access_logs)

    if output_dir is not None:
        save_profile_tables(results, output_dir)
    return results


def build_column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missingness, cardinality, and example values for each column."""
    row_count = max(len(frame), 1)
    rows = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        example_value = "" if non_null.empty else str(non_null.iloc[0])
        rows.append(
            {
                "column_name": column,
                "data_type": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().sum() / row_count),
                "unique_count": int(series.nunique(dropna=True)),
                "example_value": example_value[:120],
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "column_name"], ascending=[False, True])


def build_dataco_dataset_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Return high-level DataCo order table profile statistics."""
    order_dates = orders[ORDER_DATE_COLUMN] if ORDER_DATE_COLUMN in orders.columns else pd.Series(dtype="datetime64[ns]")
    shipping_dates = (
        orders[SHIPPING_DATE_COLUMN] if SHIPPING_DATE_COLUMN in orders.columns else pd.Series(dtype="datetime64[ns]")
    )
    summary = {
        "row_count": len(orders),
        "column_count": len(orders.columns),
        "order_id_count": _nunique(orders, "order_id"),
        "order_item_id_count": _nunique(orders, "order_item_id"),
        "product_count": _nunique(orders, "product_card_id"),
        "product_name_count": _nunique(orders, "product_name"),
        "category_count": _nunique(orders, "category_id"),
        "department_count": _nunique(orders, "department_name"),
        "market_count": _nunique(orders, "market"),
        "order_region_count": _nunique(orders, "order_region"),
        "order_country_count": _nunique(orders, "order_country"),
        "order_start_date": _min_timestamp(order_dates),
        "order_end_date": _max_timestamp(order_dates),
        "shipping_start_date": _min_timestamp(shipping_dates),
        "shipping_end_date": _max_timestamp(shipping_dates),
        "late_delivery_rate": _mean(orders, "late_delivery_risk"),
        "shipping_canceled_rate": _value_share(orders, "delivery_status", "Shipping canceled"),
        "average_order_item_quantity": _mean(orders, "order_item_quantity"),
        "average_sales": _mean(orders, "sales"),
        "average_shipment_delay_days": _mean(orders, "shipment_delay_days"),
    }
    return pd.DataFrame([summary])


def build_dataco_categorical_profile(orders: pd.DataFrame) -> pd.DataFrame:
    """Return value counts for important categorical columns."""
    categorical_columns = [
        "delivery_status",
        "late_delivery_risk",
        "shipping_mode",
        "order_status",
        "market",
        "order_region",
        "department_name",
    ]
    frames: List[pd.DataFrame] = []
    for column in categorical_columns:
        if column not in orders.columns:
            continue
        counts = orders[column].value_counts(dropna=False).rename_axis("value").reset_index(name="count")
        counts["column_name"] = column
        counts["share"] = counts["count"] / max(len(orders), 1)
        frames.append(counts[["column_name", "value", "count", "share"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["column_name", "value", "count", "share"])


def build_daily_panel_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Return row counts for possible daily planning panel granularities."""
    if "order_date_day" not in orders.columns:
        return pd.DataFrame()

    panel_specs = {
        "daily_product": ["order_date_day", "product_card_id"],
        "daily_category": ["order_date_day", "category_id"],
        "daily_country_category": ["order_date_day", "order_country", "category_id"],
        "daily_region_category": ["order_date_day", "order_region", "category_id"],
        "daily_market_category": ["order_date_day", "market", "category_id"],
    }
    rows = []
    for panel_name, columns in panel_specs.items():
        available_columns = [column for column in columns if column in orders.columns]
        if len(available_columns) != len(columns):
            continue
        row_count = orders.groupby(available_columns, dropna=False).size().shape[0]
        rows.append(
            {
                "panel_name": panel_name,
                "grouping_columns": ", ".join(columns),
                "row_count": int(row_count),
                "planning_signal_candidate": "demand_units",
                "notes": "Aggregates order item quantity by date and planning unit.",
            }
        )
    return pd.DataFrame(rows)


def build_access_log_summary(access_logs: pd.DataFrame) -> pd.DataFrame:
    """Return high-level profile statistics for the tokenized access logs."""
    date_series = access_logs["access_datetime"] if "access_datetime" in access_logs.columns else pd.Series(dtype="datetime64[ns]")
    summary = {
        "row_count": len(access_logs),
        "column_count": len(access_logs.columns),
        "product_count": _nunique(access_logs, "product"),
        "category_count": _nunique(access_logs, "category"),
        "department_count": _nunique(access_logs, "department"),
        "access_start_datetime": _min_timestamp(date_series),
        "access_end_datetime": _max_timestamp(date_series),
        "recommended_role": "Context signal or robustness feature, not a primary fulfillment label.",
    }
    return pd.DataFrame([summary])


def assess_dataco_research_fit(
    dataset_summary: pd.DataFrame,
    daily_panel_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assess DataCo suitability for candidate research use cases."""
    summary = dataset_summary.iloc[0].to_dict()
    row_count = int(summary.get("row_count", 0))
    late_rate = float(summary.get("late_delivery_rate", 0.0))
    product_count = int(summary.get("product_count", 0))
    country_count = int(summary.get("order_country_count", 0))
    daily_product_rows = _panel_row_count(daily_panel_summary, "daily_product")

    rows = [
        {
            "use_case": "Shipment delay and delivery risk",
            "suitability": "High",
            "evidence": (
                "The table contains actual shipping days, scheduled shipment days, delivery status, "
                "shipping mode, and a late_delivery_risk label."
            ),
            "profile_signal": "{} rows with late delivery rate {:.3f}.".format(row_count, late_rate),
            "recommended_role": "Primary supervised learning and operational risk analysis use case.",
        },
        {
            "use_case": "Order fulfillment planning",
            "suitability": "Medium-High",
            "evidence": (
                "The table contains order status, shipping mode, market, country, product, quantity, "
                "sales, profit, and shipment timing."
            ),
            "profile_signal": "Orders span {} countries and {} products.".format(country_count, product_count),
            "recommended_role": "Useful for fulfillment risk, prioritization, and service policy experiments.",
        },
        {
            "use_case": "Inventory and distribution planning",
            "suitability": "Medium",
            "evidence": (
                "Order item quantity can be aggregated by product, category, region, country, or market, "
                "but the dataset lacks inventory positions, replenishment orders, warehouse capacity, and stockouts."
            ),
            "profile_signal": "Daily product panel contains {} rows.".format(daily_product_rows),
            "recommended_role": "Use as a demand and distribution context dataset, not as a complete inventory-control dataset.",
        },
        {
            "use_case": "Planning stability simulation",
            "suitability": "Medium",
            "evidence": (
                "The dataset can generate daily planning signals from aggregated quantity or sales, but it does not "
                "include historical planner targets or execution capacity."
            ),
            "profile_signal": "Synthetic stability constraints can be applied to daily product or category panels.",
            "recommended_role": "Good for simulation after defining artificial execution capacity and stability policies.",
        },
        {
            "use_case": "Supply-chain context robustness",
            "suitability": "High",
            "evidence": (
                "The table includes diverse markets, regions, countries, departments, products, statuses, and shipment modes. "
                "Access logs provide a separate context signal over part of the order period."
            ),
            "profile_signal": "Rich context fields support stress tests across geography, product, and fulfillment mode.",
            "recommended_role": "Use as a robustness and context-shift dataset alongside cleaner demand benchmarks.",
        },
    ]
    return pd.DataFrame(rows)


def save_profile_tables(profile_tables: Mapping[str, pd.DataFrame], output_dir: PathLike) -> Dict[str, Path]:
    """Save profile tables as CSV files and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}
    for table_name, frame in profile_tables.items():
        path = output_path / "{}.csv".format(table_name)
        frame.to_csv(path, index=False)
        saved_paths[table_name] = path
    return saved_paths


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case English column names."""
    renamed = frame.copy()
    renamed.columns = [to_snake_case(column) for column in renamed.columns]
    return renamed


def to_snake_case(value: str) -> str:
    """Convert a raw column name into a stable snake_case name."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _resolve_dataco_file(raw_data_dir: PathLike, filename: str) -> Path:
    """Return a DataCo CSV path, allowing one nested extraction directory."""
    root = Path(raw_data_dir)
    direct_path = root / filename
    if direct_path.exists():
        return direct_path
    matches = sorted(root.rglob(filename)) if root.exists() else []
    if matches:
        return matches[0]
    return direct_path


def _read_csv_with_fallback_encoding(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV file using UTF-8 first and Latin-1 as fallback."""
    if not path.exists():
        raise FileNotFoundError("DataCo file does not exist: {}".format(path))
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError("Unable to read CSV file with supported encodings: {}".format(path)) from last_error


def _add_order_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add parsed order and shipping date columns when available."""
    updated = frame.copy()
    if ORDER_DATE_COLUMN in updated.columns:
        updated[ORDER_DATE_COLUMN] = pd.to_datetime(updated[ORDER_DATE_COLUMN], errors="coerce")
        updated["order_date_day"] = updated[ORDER_DATE_COLUMN].dt.floor("D")
    if SHIPPING_DATE_COLUMN in updated.columns:
        updated[SHIPPING_DATE_COLUMN] = pd.to_datetime(updated[SHIPPING_DATE_COLUMN], errors="coerce")
        updated["shipping_date_day"] = updated[SHIPPING_DATE_COLUMN].dt.floor("D")
    return updated


def _add_shipping_delay_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add shipment delay fields from actual and scheduled shipping days."""
    updated = frame.copy()
    if {"days_for_shipping_real", "days_for_shipment_scheduled"}.issubset(set(updated.columns)):
        updated["shipment_delay_days"] = (
            updated["days_for_shipping_real"] - updated["days_for_shipment_scheduled"]
        )
        updated["is_delayed_beyond_schedule"] = updated["shipment_delay_days"] > 0
    return updated


def _nunique(frame: pd.DataFrame, column: str) -> int:
    """Return distinct count for a column if present."""
    return int(frame[column].nunique(dropna=True)) if column in frame.columns else 0


def _mean(frame: pd.DataFrame, column: str) -> float:
    """Return numeric mean for a column if present."""
    return float(frame[column].mean()) if column in frame.columns else 0.0


def _value_share(frame: pd.DataFrame, column: str, value: object) -> float:
    """Return the share of rows equal to a value if the column exists."""
    if column not in frame.columns or len(frame) == 0:
        return 0.0
    return float((frame[column] == value).mean())


def _min_timestamp(series: pd.Series) -> str:
    """Return the minimum timestamp as an ISO date string."""
    if series.empty:
        return ""
    value = series.min()
    return "" if pd.isna(value) else str(value)


def _max_timestamp(series: pd.Series) -> str:
    """Return the maximum timestamp as an ISO date string."""
    if series.empty:
        return ""
    value = series.max()
    return "" if pd.isna(value) else str(value)


def _panel_row_count(daily_panel_summary: pd.DataFrame, panel_name: str) -> int:
    """Return a row count for a named panel summary."""
    if daily_panel_summary.empty:
        return 0
    matches = daily_panel_summary[daily_panel_summary["panel_name"] == panel_name]
    if matches.empty:
        return 0
    return int(matches.iloc[0]["row_count"])
