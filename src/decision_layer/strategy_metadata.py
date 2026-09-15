"""Shared metadata for decision-layer strategy audit columns."""

from typing import Dict, List

import pandas as pd


REALIZED_INVENTORY_ORACLE = "oracle_dp_feasibility_selector"
FULL_OUTCOME_ORACLE = "full_outcome_oracle_dp_feasibility_selector"
REALIZED_DEMAND_ORACLE = "oracle_realized_demand"
NON_DEPLOYABLE_STRATEGIES = {
    REALIZED_INVENTORY_ORACLE,
    FULL_OUTCOME_ORACLE,
    REALIZED_DEMAND_ORACLE,
}


def strategy_deployable(strategy: str) -> bool:
    """Return whether a strategy is intended to be deployable."""
    return str(strategy) not in NON_DEPLOYABLE_STRATEGIES


def strategy_oracle_type(strategy: str) -> str:
    """Return the oracle type for a strategy, or none for regular methods."""
    strategy_name = str(strategy)
    if strategy_name == REALIZED_INVENTORY_ORACLE:
        return "realized_inventory_oracle"
    if strategy_name == FULL_OUTCOME_ORACLE:
        return "full_outcome_oracle"
    if strategy_name == REALIZED_DEMAND_ORACLE:
        return "perfect_demand_oracle"
    return "none"


def ensure_strategy_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deployability, oracle, and fallback metadata columns."""
    output = frame.copy()
    if output.empty or "strategy" not in output.columns:
        return output

    deployable_values = output["strategy"].map(strategy_deployable)
    oracle_values = output["strategy"].map(strategy_oracle_type)

    if "deployable" not in output.columns:
        output["deployable"] = deployable_values
    else:
        output["deployable"] = output["deployable"].where(output["deployable"].notna(), deployable_values).astype(bool)

    if "oracle_type" not in output.columns:
        output["oracle_type"] = oracle_values
    else:
        output["oracle_type"] = output["oracle_type"].where(output["oracle_type"].notna(), oracle_values).fillna("none").astype(str)

    if "fallback_used" not in output.columns:
        output["fallback_used"] = False
    else:
        output["fallback_used"] = output["fallback_used"].fillna(False).astype(bool)

    if "fallback_type" not in output.columns:
        output["fallback_type"] = "none"
    else:
        output["fallback_type"] = output["fallback_type"].fillna("none").astype(str)

    if "fallback_reason" not in output.columns:
        output["fallback_reason"] = "none"
    else:
        output["fallback_reason"] = output["fallback_reason"].fillna("none").astype(str)

    return output


def summarize_strategy_metadata(group: pd.DataFrame) -> Dict[str, object]:
    """Return strategy-level metadata aggregated from decision rows."""
    normalized = ensure_strategy_metadata(group)
    fallback_types = _unique_nonempty_values(normalized["fallback_type"])
    fallback_reasons = _unique_nonempty_values(normalized["fallback_reason"])
    return {
        "deployable": bool(normalized["deployable"].iloc[0]) if not normalized.empty else True,
        "oracle_type": str(normalized["oracle_type"].iloc[0]) if not normalized.empty else "none",
        "fallback_used": bool(normalized["fallback_used"].any()) if not normalized.empty else False,
        "fallback_type": "; ".join(fallback_types) if fallback_types else "none",
        "fallback_reason": "; ".join(fallback_reasons) if fallback_reasons else "none",
    }


def _unique_nonempty_values(series: pd.Series) -> List[str]:
    """Return stable nonempty fallback values excluding the explicit none value."""
    values = []
    for value in series.astype(str):
        stripped = value.strip()
        if stripped and stripped.lower() != "none" and stripped not in values:
            values.append(stripped)
    return values
