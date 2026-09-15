# IEOM Parameter Provenance

| Parameter | Source or rationale | Selection status | Tested range or regimes | Sensitivity evidence | Manuscript role |
|---|---|---|---|---|---|
| Lead time | Operational stress dimension | Fixed scenario grid | 0, 2, 7 days; additional dynamic regimes at 2, 4, 7 | `dynamic_operational_summary.csv`; DP stress results | Regime definition |
| Shortage/holding ratio | Service versus inventory trade-off | Fixed before test | 2, 5, 10 | 27-scenario Favorita grid | Cost sensitivity |
| Capacity ratio | Flexible/moderate/tight operational regimes | Fixed before test | 2.0, 1.25, 0.75 of validation mean | 27-scenario grid; tight-capacity DP result | Execution stress |
| Rate-limit cap | Practitioner control | Strict 0.40; alternative selected on validation | 0.10, 0.20, 0.40, 0.75 | `rate_limiter_mechanism_summary.csv` | Conditional safeguard |
| Safety-stock multiplier | Existing planning conversion | Fixed | 0.50 | Shared across policies; not claimed optimal | Forecast-to-plan conversion |
| Switching cost | Governance stress dimension | Fixed per scenario | 0.25, 0.50, 2.00 | Dynamic regime results | Switching burden |
| Switch budget K | Governance rule | Fixed K=2 or floor(validation q25) | Dataset/regime-specific frozen K | `switch_budget_mechanism_summary.csv` | Hard governance constraint |
| Frozen horizon | Planning cadence constraint | Fixed before test | 1, 2, 4 periods | Frozen-horizon regime | Multi-period relevance |
| Backlog persistence | Backorder-state assumption | Fixed stress regime | 1.0 in reported dynamic tests | Persistent-backlog regime | Carryover stress |
| State rounding | Finite-state approximation | Fixed before test | max(0.1 validation mean, 0.25) | Declared approximation; no exact-DP claim | Computational control |
| Beam width | Computational cap | Fixed before test | 96 in stress tests | Runtime/complexity report | Computational control |

No important conclusion is attributed to a single calibrated dollar value. Scenario parameters define comparative operating regimes. Supplemental DataCo-derived profiling supports no current manuscript result and is not used as capacity or cost calibration.
