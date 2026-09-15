"""Audit Full-Outcome Oracle DP usage as an internal diagnostic.

The IEOM manuscript centers on deployable strategies plus two diagnostic
references: Realized-Inventory Oracle DP and the perfect-demand diagnostic
reference. Full-Outcome Oracle DP remains useful in the repository, but it must
not enter paper-facing deployable rankings or headline IEOM claims.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_ieom_validation_extensions import build_objective_component_sensitivity, build_statistical_validation  # noqa: E402


FULL_OUTCOME_TOKEN = "full_outcome_oracle_dp_feasibility_selector"
FULL_OUTCOME_TYPE = "full_outcome_oracle"
REALIZED_INVENTORY_TOKENS = ("oracle_dp_feasibility_selector", "realized_inventory_oracle")
PERFECT_DEMAND_TOKENS = ("oracle_realized_demand", "perfect_demand_oracle", "perfect-demand")
ORACLE_TEXT_TOKENS = (
    FULL_OUTCOME_TOKEN,
    FULL_OUTCOME_TYPE,
    "oracle_dp_feasibility_selector",
    "realized_inventory_oracle",
    "oracle_realized_demand",
    "perfect_demand_oracle",
    "perfect-demand",
    "oracle",
)


def main() -> None:
    """Write internal Full-Outcome Oracle diagnostic artifacts."""
    internal_dir = REPO_ROOT / "outputs" / "internal"
    internal_dir.mkdir(parents=True, exist_ok=True)

    artifacts = audit_artifacts()
    diagnostic_path = internal_dir / "full_outcome_oracle_diagnostic.csv"
    artifacts.to_csv(diagnostic_path, index=False)

    key_numbers = compute_filtered_key_numbers()
    summary_path = internal_dir / "full_outcome_oracle_audit_summary.md"
    summary_path.write_text(build_summary(artifacts, key_numbers), encoding="utf-8")


def audit_artifacts() -> pd.DataFrame:
    """Return artifact-level oracle usage across outputs, paper assets, and code."""
    records: List[Dict[str, object]] = []
    search_roots = [
        REPO_ROOT / "results" / "final",
        REPO_ROOT / "outputs" / "tables",
        REPO_ROOT / "outputs" / "internal",
        REPO_ROOT / "paper" / "tables",
        REPO_ROOT / "paper_submission_ieom",
        REPO_ROOT / "scripts",
        REPO_ROOT / "src",
    ]
    for path in iter_files(search_roots):
        if path.suffix.lower() not in {".csv", ".tex", ".md", ".py"}:
            continue
        records.append(audit_file(path))
    return pd.DataFrame(records).sort_values(["scope", "artifact_path"]).reset_index(drop=True)


def iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield existing files under each root."""
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
        else:
            yield from (path for path in root.rglob("*") if path.is_file())


def audit_file(path: Path) -> Dict[str, object]:
    """Return oracle token counts and scope for one file."""
    relative = path.relative_to(REPO_ROOT)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lower_text = text.lower()
    csv_row_counts: Dict[str, int] = {}
    if path.suffix.lower() == ".csv":
        csv_row_counts = audit_csv_rows(path)

    return {
        "artifact_path": str(relative),
        "scope": artifact_scope(relative),
        "paper_facing": is_paper_facing(relative),
        "ieom_paper_facing": is_ieom_paper_facing(relative),
        "contains_full_outcome_oracle": FULL_OUTCOME_TOKEN in lower_text or FULL_OUTCOME_TYPE in lower_text,
        "full_outcome_text_count": lower_text.count(FULL_OUTCOME_TOKEN) + lower_text.count(FULL_OUTCOME_TYPE),
        "full_outcome_rows": csv_row_counts.get("full_outcome_rows", 0),
        "contains_realized_inventory_oracle": any(token in lower_text for token in REALIZED_INVENTORY_TOKENS),
        "realized_inventory_rows": csv_row_counts.get("realized_inventory_rows", 0),
        "contains_perfect_demand_reference": any(token in lower_text for token in PERFECT_DEMAND_TOKENS),
        "perfect_demand_rows": csv_row_counts.get("perfect_demand_rows", 0),
        "oracle_like_text_count": sum(lower_text.count(token) for token in ORACLE_TEXT_TOKENS),
        "oracle_like_rows": csv_row_counts.get("oracle_like_rows", 0),
    }


def audit_csv_rows(path: Path) -> Dict[str, int]:
    """Return row counts for oracle-like CSV records."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    searchable_columns = [column for column in ["strategy", "method_name", "oracle_type"] if column in frame.columns]
    if not searchable_columns:
        return {}
    values = pd.Series("", index=frame.index)
    for column in searchable_columns:
        values = values + " " + frame[column].astype(str).str.lower()

    full_outcome = values.str.contains(FULL_OUTCOME_TOKEN, regex=False) | values.str.contains(FULL_OUTCOME_TYPE, regex=False)
    realized_inventory = values.apply(lambda value: any(token in value for token in REALIZED_INVENTORY_TOKENS))
    perfect_demand = values.apply(lambda value: any(token in value for token in PERFECT_DEMAND_TOKENS))
    oracle_like = values.str.contains("oracle", regex=False)
    return {
        "full_outcome_rows": int(full_outcome.sum()),
        "realized_inventory_rows": int(realized_inventory.sum()),
        "perfect_demand_rows": int(perfect_demand.sum()),
        "oracle_like_rows": int(oracle_like.sum()),
    }


def artifact_scope(relative: Path) -> str:
    """Classify an artifact for audit reporting."""
    parts = relative.parts
    if parts[:2] == ("paper_submission_ieom", "tables"):
        return "ieom_table"
    if parts and parts[0] == "paper_submission_ieom":
        return "ieom_manuscript"
    if parts[:2] == ("paper", "tables"):
        return "general_paper_table"
    if parts[:2] == ("outputs", "tables"):
        return "experiment_output"
    if parts[:2] == ("results", "final"):
        return "canonical_result"
    if parts[:2] == ("outputs", "internal"):
        return "internal_output"
    if parts and parts[0] == "scripts":
        return "script"
    if parts and parts[0] == "src":
        return "source"
    return "other"


def is_paper_facing(relative: Path) -> bool:
    """Return whether a file is a paper-facing artifact."""
    return relative.parts[:2] in {
        ("paper", "tables"),
        ("paper_submission_ieom", "tables"),
        ("paper_submission_ieom", "sections"),
        ("paper_submission_ieom", "figures"),
    } or str(relative) == "paper_submission_ieom/main.tex"


def is_ieom_paper_facing(relative: Path) -> bool:
    """Return whether a file is directly used by the IEOM manuscript."""
    return relative.parts[:2] in {
        ("paper_submission_ieom", "tables"),
        ("paper_submission_ieom", "sections"),
        ("paper_submission_ieom", "figures"),
    } or str(relative) == "paper_submission_ieom/main.tex"


def compute_filtered_key_numbers() -> Dict[str, object]:
    """Return paper-facing IEOM key numbers under hindsight-oracle filtering."""
    try:
        statistical = build_statistical_validation()
        objective_detail, objective_summary = build_objective_component_sensitivity()
    except RuntimeError as error:
        # The legacy validation-extension inputs live under ignored
        # ``outputs/tables`` and are not part of a clean submission clone. The
        # active manuscript instead uses the versioned canonical artifacts in
        # ``results/final``, which are scanned directly above.
        return {
            "recomputed": False,
            "unavailable_reason": str(error),
            "evaluation_groups": "not recomputed",
            "mismatch_rate": "not recomputed",
            "mean_peg_rate_reduction": "not recomputed",
            "median_peg_rate_reduction": "not recomputed",
            "objective_component_sensitivity_rows": "not recomputed",
            "objective_component_sensitivity_summary": [],
            "objective_component_detail_oracle_rows": "not recomputed",
        }
    lookup = dict(zip(statistical["Metric"], statistical["Estimate"]))
    return {
        "recomputed": True,
        "unavailable_reason": "",
        "evaluation_groups": lookup.get("Evaluation groups"),
        "mismatch_rate": lookup.get("Accuracy-first mismatch rate"),
        "mean_peg_rate_reduction": lookup.get("Mean PEG-rate reduction"),
        "median_peg_rate_reduction": lookup.get("Median PEG-rate reduction"),
        "objective_component_sensitivity_rows": len(objective_summary),
        "objective_component_sensitivity_summary": objective_summary.to_dict("records"),
        "objective_component_detail_oracle_rows": int(
            objective_detail["selected_strategy"].astype(str).str.contains("oracle", case=False, na=False).sum()
        ),
    }


def build_summary(artifacts: pd.DataFrame, key_numbers: Dict[str, object]) -> str:
    """Return a Markdown summary for the internal audit."""
    full_outcome_anywhere = bool(artifacts["contains_full_outcome_oracle"].any())
    full_outcome_ieom = artifacts[
        artifacts["ieom_paper_facing"] & artifacts["contains_full_outcome_oracle"]
    ]
    full_outcome_paper = artifacts[
        artifacts["paper_facing"] & artifacts["contains_full_outcome_oracle"]
    ]
    full_outcome_outputs = artifacts[
        (artifacts["scope"] == "experiment_output") & artifacts["contains_full_outcome_oracle"]
    ]
    full_outcome_internal = artifacts[
        (artifacts["scope"] == "internal_output") & artifacts["contains_full_outcome_oracle"]
    ]

    lines = [
        "# Full-Outcome Oracle DP Internal Diagnostic Audit",
        "",
        "Full-Outcome Oracle DP is retained only as an internal repository diagnostic.",
        "It is non-deployable because it can use realized test-period forecast-error and inventory-outcome information.",
        "",
        "## Summary",
        "",
        "- Full-Outcome Oracle DP available in repository outputs: {}".format("yes" if full_outcome_anywhere else "no"),
        "- Full-Outcome Oracle DP appears in paper-facing outputs after filtering: {}".format(
            "yes" if not full_outcome_paper.empty else "no"
        ),
        "- Full-Outcome Oracle DP appears in IEOM paper-facing outputs after filtering: {}".format(
            "yes" if not full_outcome_ieom.empty else "no"
        ),
        "- Full-Outcome Oracle DP appears in experiment outputs: {}".format("yes" if not full_outcome_outputs.empty else "no"),
        "- Full-Outcome Oracle DP appears in internal diagnostic outputs: {}".format(
            "yes" if not full_outcome_internal.empty else "no"
        ),
        "- Full-Outcome Oracle DP is used in deployable rankings: no",
        "- Legacy validation-extension key numbers recomputed: {}".format(
            "yes" if key_numbers["recomputed"] else "no (ignored source tables are absent from this clone)"
        ),
        "",
        "## Filtered IEOM Key Numbers",
        "",
        "- Evaluation groups: {}".format(key_numbers["evaluation_groups"]),
        "- Accuracy-first mismatch rate: {}".format(key_numbers["mismatch_rate"]),
        "- Mean PEG-rate reduction: {}".format(key_numbers["mean_peg_rate_reduction"]),
        "- Median PEG-rate reduction: {}".format(key_numbers["median_peg_rate_reduction"]),
        "- Objective-component sensitivity selected-oracle rows: {}".format(
            key_numbers["objective_component_detail_oracle_rows"]
        ),
        "",
        "## Full-Outcome Oracle Locations",
        "",
    ]
    if not key_numbers["recomputed"]:
        lines.extend(
            [
                "- Recompute note: {}".format(key_numbers["unavailable_reason"]),
                "- The active, versioned canonical artifacts were still scanned directly for oracle leakage.",
            ]
        )
    if full_outcome_outputs.empty and full_outcome_paper.empty and full_outcome_internal.empty:
        lines.append("- None found.")
    else:
        rows = artifacts[artifacts["contains_full_outcome_oracle"]][
            ["artifact_path", "scope", "full_outcome_rows", "paper_facing", "ieom_paper_facing"]
        ]
        for row in rows.itertuples(index=False):
            lines.append(
                "- `{}` ({}, rows={}, paper_facing={}, ieom_paper_facing={})".format(
                    row.artifact_path,
                    row.scope,
                    row.full_outcome_rows,
                    row.paper_facing,
                    row.ieom_paper_facing,
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The IEOM manuscript should continue to report deployable strategy rankings plus the two stated diagnostic references only: Realized-Inventory Oracle DP and the perfect-demand diagnostic reference.",
            "Full-Outcome Oracle DP may remain in internal diagnostic outputs and implementation code, but it should not support the IEOM mismatch, PEG-rate, Table 2, or Table 3 claims.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
