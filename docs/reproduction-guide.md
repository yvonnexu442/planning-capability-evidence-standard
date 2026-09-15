# Reproduction guide

Run commands from the repository root after installing dependencies and preparing the public datasets described in `data/README.md`.

## 1. Primary replay experiments

```bash
python scripts/run_dynamic_policy_experiments.py
```

## 2. Validation-selected transfer

```bash
python scripts/run_transfer_regime_experiments.py
python scripts/analyze_transfer_regime_results.py
```

## 3. Operating-stress and approximation analyses

```bash
python scripts/run_dp_approximation_sensitivity.py
python scripts/run_switch_budget_sensitivity.py
python scripts/run_ieom_validation_extensions.py
```

## 4. Reduced exhaustive benchmark and governance comparison

```bash
python scripts/run_high_impact_validation.py
python scripts/audit_policy_mechanisms.py
python scripts/audit_full_outcome_oracle.py
```

## 5. Retrospective proxy analysis

```bash
python scripts/run_targeted_north_star_analyses.py
python scripts/analyze_dynamic_policy_results.py
```

## 6. Canonical tables, figures, and lineage

```bash
python scripts/build_ieom_final_assets.py
python scripts/write_result_lineage.py
```

The convenience script `scripts/reproduce_ieom_final.sh` records the project-level sequence. Review it before launching a full run because dataset downloads and computational requirements vary by environment.

## Frozen study controls

The manuscript's parameters, seeds, tolerances, cost coefficients, sampling rules, and information boundaries are encoded in `configs/default.yaml`, the experiment scripts, and the audit records. The canonical outputs under `results/canonical/` provide comparison targets.

## Validation

Compare regenerated outputs against `results/canonical/` and review checksum and lineage records under `audits/`. Small presentation-format differences are not evidence of operational divergence; use the paper's unrounded plan and cost tolerances.
