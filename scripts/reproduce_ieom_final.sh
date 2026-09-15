#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cleanup_latex_build() {
  if command -v latexmk >/dev/null 2>&1; then
    (cd paper_submission_ieom && latexmk -c >/dev/null 2>&1) || true
  fi
  rm -f \
    paper_submission_ieom/main.pdf \
    paper_submission_ieom/main.xdv \
    paper_submission_ieom/main_mode_a.pdf \
    paper_submission_ieom/main_mode_a.xdv \
    paper_submission_ieom/supplementary_validation_appendix.pdf \
    paper_submission_ieom/supplementary_validation_appendix.xdv
}
trap cleanup_latex_build EXIT

PYTHONPATH=src:scripts python3 -m pytest -q

if [[ "${1:-}" == "--full" ]]; then
  PYTHONPATH=src:scripts python3 - <<'PY'
from data_loaders.favorita_loader import validate_favorita_files
from data_loaders.m5_loader import validate_m5_files
from data_loaders.walmart_loader import validate_walmart_files
validate_favorita_files("data/raw/favorita")
validate_m5_files("data/raw/m5")
validate_walmart_files("data/raw/walmart")
print("Required public-data files validated for a full experiment run.")
PY
  # Generate the candidate forecasts and static policy records consumed by the
  # held-out dynamic experiments. Keeping them under results/working makes the
  # dependency explicit and avoids relying on untracked historical outputs.
  PYTHONPATH=src:scripts python3 scripts/run_favorita_minimal_pipeline.py \
    --run-mode quick \
    --max-series 100 \
    --output-dir results/working/favorita_base \
    --paper-table-dir results/working/favorita_base/paper_tables \
    --paper-figure-dir results/working/favorita_base/paper_figures \
    --asset-manifest-path results/working/favorita_base/asset_manifest.md
  PYTHONPATH=src:scripts python3 scripts/run_dynamic_policy_experiments.py --max-series 100
  PYTHONPATH=src:scripts python3 scripts/analyze_dynamic_policy_results.py
  PYTHONPATH=src:scripts python3 scripts/run_transfer_regime_experiments.py
  PYTHONPATH=src:scripts python3 scripts/analyze_transfer_regime_results.py
  PYTHONPATH=src:scripts python3 scripts/run_dp_approximation_sensitivity.py
  PYTHONPATH=src:scripts python3 scripts/audit_policy_mechanisms.py
  PYTHONPATH=src:scripts python3 scripts/run_targeted_north_star_analyses.py
  PYTHONPATH=src:scripts python3 scripts/run_high_impact_validation.py
else
  echo "Public verification uses tracked canonical results; raw datasets are not required."
fi

PYTHONPATH=src:scripts python3 scripts/build_ieom_final_assets.py
PYTHONPATH=src:scripts python3 scripts/write_result_lineage.py

if command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1; then
  (
    cd paper_submission_ieom
    if [[ "${IEOM_SUPPLEMENT_MODE:-B}" == "A" ]]; then
      latexmk -xelatex -jobname=main_mode_a -interaction=nonstopmode -halt-on-error mode_a_supplement_enabled.tex
      cp main_mode_a.pdf "IEOM_Irvine_Planning_Capability_Operational_Value_Validation.pdf"
    else
      latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
      cp main.pdf "IEOM_Irvine_Planning_Capability_Operational_Value_Validation.pdf"
    fi
    latexmk -xelatex -interaction=nonstopmode -halt-on-error supplementary_validation_appendix.tex
    cp supplementary_validation_appendix.pdf "IEOM_Irvine_Supplementary_Validation_Appendix.pdf"
  )
else
  echo "XeLaTeX toolchain not found; skipped manuscript rebuild."
fi

PYTHONPATH=src:scripts python3 scripts/write_ieom_manifests.py
PYTHONPATH=src:scripts python3 scripts/audit_ieom_submission.py

echo "Canonical IEOM outputs regenerated under results/final and figures/final."
