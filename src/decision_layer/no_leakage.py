"""Utilities that separate deployable selection inputs from evaluation outcomes."""

from typing import Iterable, Mapping, Sequence, Union

import pandas as pd


DEFAULT_OUTCOME_COLUMNS = (
    "actual",
    "actual_demand",
    "demand",
    "future_demand",
    "future_outcomes",
    "realized_demand",
    "sales",
    "unit_sales",
)


def drop_future_outcomes(
    frame: pd.DataFrame,
    outcome_columns: Iterable[str] = DEFAULT_OUTCOME_COLUMNS,
) -> pd.DataFrame:
    """Return a copy without future outcome columns for deployable selectors."""
    columns_to_drop = [column for column in outcome_columns if column in frame.columns]
    return frame.drop(columns=columns_to_drop).copy()


def require_no_future_outcomes(
    frame: Union[pd.DataFrame, Mapping[str, object]],
    context: str,
    outcome_columns: Iterable[str] = DEFAULT_OUTCOME_COLUMNS,
) -> None:
    """Raise if a deployable input exposes a realized-outcome field.

    The guard accepts either a tabular selector input or the named candidate
    mapping used by the inventory-state beam search.  This keeps the same
    runtime information-boundary check at both deployable selector interfaces.
    """
    available_names = frame.columns if isinstance(frame, pd.DataFrame) else frame.keys()
    leaked_columns = [column for column in outcome_columns if column in available_names]
    if leaked_columns:
        raise ValueError(
            "{} received future outcome columns: {}".format(
                context,
                ", ".join(sorted(leaked_columns)),
            )
        )


def attach_actuals_for_evaluation(
    selected_decisions: pd.DataFrame,
    outcome_frame: pd.DataFrame,
    key_columns: Sequence[str] = ("date", "series_id"),
    actual_column: str = "actual",
) -> pd.DataFrame:
    """Attach realized demand only after deployable decisions have been made."""
    require_no_future_outcomes(
        selected_decisions,
        context="attach_actuals_for_evaluation selected_decisions",
        outcome_columns=(actual_column,),
    )
    lookup_columns = list(key_columns) + [actual_column]
    missing_columns = [column for column in lookup_columns if column not in outcome_frame.columns]
    if missing_columns:
        raise ValueError("Outcome frame is missing columns: {}".format(", ".join(missing_columns)))

    outcome_lookup = outcome_frame[lookup_columns].copy()
    conflicting_actuals = outcome_lookup.groupby(list(key_columns), dropna=False)[actual_column].nunique(dropna=False)
    if (conflicting_actuals > 1).any():
        raise ValueError("Outcome frame contains conflicting actuals for the same evaluation key.")
    outcome_lookup = outcome_lookup.drop_duplicates(list(key_columns))

    merged = selected_decisions.merge(outcome_lookup, on=list(key_columns), how="left")
    if merged[actual_column].isna().any():
        missing = merged.loc[merged[actual_column].isna(), list(key_columns)].drop_duplicates().head(5)
        raise ValueError("Missing evaluation actuals for selected decisions: {}".format(missing.to_dict("records")))
    return merged
