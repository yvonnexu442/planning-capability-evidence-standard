# Tractable exact-DP benchmark method

## Prespecified scope

The benchmark uses the frozen M5 item-store panel, sorted by planning-unit identifier, and selects every fifth unit (20 units). It uses the first eight test periods and the first three candidate model names in lexical order. These choices were fixed before examining benchmark outcomes. The two regimes are the manuscript's favorable long-lead and adverse tight-capacity cases.

## Exact and approximate branches

The exact branch enumerates every feasible model path (3^8 paths before frozen-horizon restrictions) without state rounding, state merging, or beam pruning. Each path is evaluated with the same forecast-state transition and internal selection objective as the live approximate selector. The comparison branch uses the manuscript setting (beam width 96 and validation-scaled state rounding). Realized demand is supplied only after each path is fixed, through the shared execution simulator.

This benchmark establishes exactness only for the reduced three-model, eight-period problems. It does not establish convergence or exact optimality for the full 28-period implementation.

## Prespecified outputs

The audit reports the approximate-minus-exact internal objective gap, model-path agreement, executed-order agreement, realized cost relative to the same simple reference, directional agreement in each regime, and consistently measured wall-clock runtime. No benchmark setting is selected from its result.

## Generated-result check

- Comparisons: 40
- Relative objective gap median / P90 / maximum: 0 / 0 / 0.00081081666
- Model-path agreement: 97.5%
- Executed-plan agreement: 100.0%
- Long-lead direction agreement: 100.0%
- Tight-capacity direction agreement: 100.0%
- Reduced long-lead exact effect versus reference: -0.4544% (full-horizon favorable direction not retained)
- Reduced tight-capacity exact effect versus reference: -0.0555% (full-horizon adverse direction retained)
- Median exact / approximate runtime: 0.9690 / 0.0345 seconds (same runs and machine)

Canonical unit-level results are in `results/final/exact_benchmark_results.csv`.
