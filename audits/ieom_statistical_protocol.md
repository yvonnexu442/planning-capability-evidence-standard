# IEOM Statistical Protocol

| Comparison | Status | Estimand | Pairing/resampling unit | Block lengths | Replicates | Interval |
|---|---|---|---|---|---:|---|
| Ensemble plus rate limiting vs accuracy-first | Primary | Mean paired dynamic-cost difference | Planning unit; circular time blocks within unit | Favorita/M5: 7, 14, 28 days; Walmart: 4, 8, 13 weeks | 1,000 | Percentile 95% CI |
| Ensemble plus rate limiting vs static feasibility-aware | Primary | Mean paired dynamic-cost difference | Favorita planning unit; circular time blocks | 7, 14, 28 days | 1,000 | Percentile 95% CI |
| Approximate DP vs ensemble control in predefined dynamic regimes | Primary | Mean paired dynamic-cost difference | Planning unit; circular time blocks | Dataset cadence-specific | 1,000 | Percentile 95% CI |
| Budgeted DP vs DP in governance stress | Primary | Mean paired dynamic/governance-cost difference | Planning unit; circular time blocks | Dataset cadence-specific | 1,000 | Percentile 95% CI |
| Other dataset/grain/policy contrasts | Secondary or exploratory | Descriptive mean effects | Shared underlying units acknowledged | Reported where available | Varies | Effect sizes, not independent-study p-values |

Scenario configurations are not treated as independent datasets. Headline uncertainty statements must be traceable to `results/final/time_block_bootstrap.csv`. Walmart ensemble-versus-accuracy intervals include zero and are described as inconclusive.
