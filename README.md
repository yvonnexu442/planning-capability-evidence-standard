# Planning Capability Evidence Standard

This repository contains the code, frozen configurations, canonical result tables, and audit records supporting IEOM Paper 64, *When Does an Added Planning Capability Create Operational Value? An Observable Evidence Standard*.

## Scope

The package supports the paper's forecast-to-plan replay experiments, validation-selected transfer analysis, operating-stress analysis, reduced exhaustive benchmark, governance comparison, and retrospective proxy analysis. Favorita, M5, and Walmart source datasets are not redistributed.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- Locally downloaded public datasets arranged as described in `data/README.md`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction

See `docs/reproduction-guide.md` for the staged workflow. Canonical outputs used in the paper are preserved under `results/canonical/`; parameter and result-lineage checks are under `audits/`.

## Expected outputs

The workflow regenerates the principal replay summaries, transfer results, stress and sensitivity summaries, reduced exact-benchmark results, governance diagnostics, proxy-regret summaries, and final figure/table inputs.

## Runtime

Runtime depends on dataset availability, machine resources, and selected analyses. The reduced benchmark is intended as a tractable fidelity check; full replay and stress analyses are substantially more expensive. No universal runtime estimate is claimed.

## Limitations

The repository does not redistribute source datasets. Results depend on the frozen sampling, tolerances, costs, seeds, and information boundaries documented in `configs/`, the scripts, and `audits/`. The reduced exact benchmark does not establish universal or full-horizon equivalence.

## License

Code is released under the MIT License. Dataset licenses and terms remain those of the original providers.
