"""Write the canonical manuscript result-to-implementation lineage.

The mapping is intentionally explicit: it distinguishes the inventory-state
beam-search DP used by manuscript evidence from the separate feasibility-state
selector retained only for supplemental comparisons.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final"

PRIMARY_DP = "decision_layer.dynamic_inventory_policies.select_dynamic_policy"
PRIMARY_STATE = "inventory, backlog, on-order pipeline, prior model/order/plan, cumulative switches; rounded-state beam search"
NO_SELECTOR = "not applicable"


ROWS = [
    ("Table 1", "Methods 3.4", "paper_submission_ieom/tables/information_access_table.tex", "source-code interfaces; reports/final/ieom_information_boundary_audit.md", "multiple interfaces, explicitly separated", "declared information available at action-selection time", "paper_submission_ieom/tables/information_access_table.tex", "Information-boundary disclosure"),
    ("Table 2", "Methods 3.4", "paper_submission_ieom/tables/transfer_scope_table.tex", "scripts/run_transfer_regime_experiments.py; results/final/cross_dataset_frozen_configurations.csv", "inline equal-weight and validation-weighted ensemble formulas in scripts/run_transfer_regime_experiments.py", "frozen policy definition, weights, and rate cap", "paper_submission_ieom/tables/transfer_scope_table.tex", "Protocol definition"),
    ("Table 3", "Experimental Design", "paper_submission_ieom/tables/dataset_roles_table.tex", "data loaders; public dataset metadata; results/final/README.md", NO_SELECTOR, "dataset grain, sampled panel, chronology, and role", "paper_submission_ieom/tables/dataset_roles_table.tex", "Dataset and provenance disclosure"),
    ("Table 4", "Results 5.1", "scripts/analyze_transfer_regime_results.py; scripts/audit_policy_mechanisms.py; scripts/build_ieom_final_assets.py", "results/working/favorita_primary_period_results.csv; results/working/cross_dataset_period_results.csv; results/final/mechanism_decomposition_summary.csv", "inline equal-weight ensemble plus decision_layer.practitioner_baselines.rate_limit_signal", "executed order path and unrounded dynamic cost", "results/final/time_block_bootstrap.csv; results/final/mechanism_comparison_audit.csv; paper_submission_ieom/tables/mechanism_table.tex", "Composite-policy effect and component attribution"),
    ("Table 5", "Results 5.3.1", "scripts/run_transfer_regime_experiments.py; scripts/audit_policy_mechanisms.py", "results/working/dynamic_policy_stress_results.csv; results/working/dynamic_policy_period_results.csv", PRIMARY_DP, PRIMARY_STATE, "results/final/switch_budget_mechanism_summary.csv; results/final/nonbinding_governance_independent_summary.csv; paper_submission_ieom/tables/governance_table.tex", "Independent unconstrained/budgeted governance comparison"),
    ("Table 6", "Results 5.4", "paper_submission_ieom/tables/lifecycle_decision_matrix.tex", "observable evidence standard and empirical evidence patterns", NO_SELECTOR, "capability evidence record fields", "paper_submission_ieom/tables/lifecycle_decision_matrix.tex", "Bounded interpretation matrix; not an automated classifier"),
    ("Figure 1", "Methods", "paper_submission_ieom/figures/framework_diagram.tex", "observable evidence standard", NO_SELECTOR, "presence, opportunity, plan change, consequence", "paper_submission_ieom/figures/framework_diagram.tex", "Thesis diagram"),
    ("Figure 2", "Results 5.2", "scripts/run_targeted_north_star_analyses.py", "results/working/cross_dataset_forecast_panel.csv; frozen regime settings", PRIMARY_DP, PRIMARY_STATE, "results/final/coupling_strength_sweep_summary.csv; figures/final/coupling_strength_vs_policy_value.pdf", "Operating-dependence stress sweep"),
    ("Figure 3", "Results 5.3.2", "scripts/analyze_transfer_regime_results.py; scripts/build_ieom_final_assets.py", "results/working/cross_dataset_period_results.csv", "static proxy compared with executed policies from scripts/run_transfer_regime_experiments.py", "static proxy ranks versus shared dynamic rollout costs", "results/final/static_dynamic_ranking_diagnostics.csv; results/final/static_dynamic_regret_quantiles.csv; figures/final/static_dynamic_screening.pdf", "Dataset-specific static-versus-dynamic regret diagnostic"),
    ("Figure 4", "Results 5.4", "paper_submission_ieom/figures/decision_flowchart.tex", "observable evidence standard; capability evidence record; Table 6", NO_SELECTOR, "evidence generation, primary status, risk overlay, bounded action", "paper_submission_ieom/figures/decision_flowchart.tex", "Lifecycle translation diagram"),
    ("Headline: 15/681 plan changes", "Abstract; Results 5.1", "scripts/audit_policy_mechanisms.py", "results/working/cross_dataset_period_results.csv", "inline equal-weight ensemble plus decision_layer.practitioner_baselines.rate_limit_signal", "executed order path under shared rollout", "results/final/rate_limiter_mechanism_summary.csv", "Rate-of-change control attribution"),
    ("Headline: long-lead benefit and tight-capacity adverse case", "Abstract; Results 5.2", "scripts/run_targeted_north_star_analyses.py; scripts/run_dp_approximation_sensitivity.py", "results/working/cross_dataset_forecast_panel.csv", PRIMARY_DP, PRIMARY_STATE, "results/final/coupling_strength_sweep_summary.csv; results/final/dp_approximation_sensitivity_summary.csv", "Implementation-level multi-period evidence"),
    ("Headline: 19/599 theoretically nonrestrictive path disagreements", "Abstract; Results 5.3.1", "scripts/run_transfer_regime_experiments.py; scripts/audit_policy_mechanisms.py", "results/working/dynamic_policy_stress_results.csv; results/working/dynamic_policy_period_results.csv", PRIMARY_DP, PRIMARY_STATE, "results/final/nonbinding_governance_independent_summary.csv; results/final/nonbinding_governance_disagreements.csv", "Independent implementation-level agreement audit"),
    ("Headline: 55.0% winner agreement and 79.5% cost equivalence", "Abstract; Results 5.3.2", "scripts/analyze_transfer_regime_results.py", "results/working/cross_dataset_period_results.csv", "static proxy compared with live transfer-policy formulas", "static proxy ranks versus shared dynamic rollout costs", "results/final/static_dynamic_ranking_diagnostics.csv", "Lightweight-diagnostic tail audit"),
    ("Validation-selected frozen transfer", "Results 5.2.1; Supplement S4", "scripts/run_high_impact_validation.py", "validation partitions from the frozen M5 and Walmart panels; results/working/cross_dataset_frozen_transfer.csv", "live transfer formulas in scripts/run_transfer_regime_experiments.py", "validation-only strategy selection followed by frozen test replay", "results/final/validation_selected_transfer_results.csv", "Deployable validation-selected comparison; retrospective envelope retained only as a secondary coverage diagnostic"),
    ("Tractable exact-DP benchmark", "Methods 4.3; Results 5.2.2; Supplement S2--S3", "scripts/run_high_impact_validation.py", "frozen M5 item-store panel; first eight test periods; first three lexical candidate models", "independent exhaustive enumeration compared with decision_layer.dynamic_inventory_policies.select_dynamic_policy", "unrounded inventory/backlog/pipeline forecast state; no beam or state merging in exact branch", "results/final/exact_benchmark_results.csv; reports/final/exact_benchmark_method.md", "Implementation-fidelity audit on reduced problems; does not establish full-horizon exact optimality or regime directions"),
    ("Comparative-cost coefficient robustness", "Results 5.2.2; Supplement S6", "scripts/run_high_impact_validation.py", "frozen M5 tight-capacity panel; static-screen and governance period outputs", PRIMARY_DP, PRIMARY_STATE, "results/final/coefficient_robustness_results.csv", "Prespecified half/base/double weights; DP paths recomputed for tight capacity and fixed-path accounting disclosed for diagnostic/governance analyses"),
    ("Supplement S3 exact table", "Supplement S3", "scripts/run_high_impact_validation.py", "frozen M5 item-store panel; deterministic 20-unit subset; first eight test periods; first three lexical candidate models", "independent exhaustive enumeration and " + PRIMARY_DP, "unrounded forecast-state transition; approximate branch adds rounded-state merging and beam pruning", "results/final/exact_benchmark_results.csv", "Implementation fidelity and reduced-scope regime directions"),
    ("Supplement S4 transfer table", "Supplement S4", "scripts/run_high_impact_validation.py", "frozen M5 and Walmart validation/test panels; results/working/cross_dataset_frozen_transfer.csv; results/working/cross_dataset_period_results.csv", "live transfer formulas in scripts/run_transfer_regime_experiments.py", "validation-only strategy selection; shared execution rollout", "results/final/validation_selected_transfer_results.csv", "Validation-selected transfer, retained ties, label agreement, and executed-plan agreement"),
    ("Supplement S5 envelope table", "Supplement S5", "scripts/run_transfer_regime_experiments.py", "frozen transfer protocols; test outcomes", "live transfer formulas in scripts/run_transfer_regime_experiments.py", "shared execution rollout; retrospective test envelope", "results/final/cross_dataset_frozen_transfer.csv", "Secondary hindsight policy-family coverage"),
    ("Supplement S6 coefficient tables", "Supplement S6", "scripts/run_high_impact_validation.py", "frozen M5 tight-capacity forecasts; static-screen and independently generated governance period outputs", PRIMARY_DP, PRIMARY_STATE, "results/final/coefficient_robustness_results.csv", "End-to-end tight-capacity reruns and disclosed fixed-path diagnostic/governance accounting"),
    ("Supplement S7 evidence record", "Supplement S7", "scripts/run_targeted_north_star_analyses.py; scripts/run_dp_approximation_sensitivity.py; scripts/run_high_impact_validation.py", "canonical full-horizon, sensitivity, and exact-benchmark results", PRIMARY_DP, PRIMARY_STATE, "results/final/coupling_strength_sweep_summary.csv; results/final/dp_approximation_sensitivity_summary.csv; results/final/exact_benchmark_results.csv", "Completed evidence, interpretation, and bounded action"),
    ("Supplemental evaluation-practice comparison", "Repository only", "paper_submission_ieom/tables/supplementary/evidence_standard_comparison.tex", "literature sources", NO_SELECTOR, "conceptual evidence-field comparison", "paper_submission_ieom/tables/supplementary/evidence_standard_comparison.tex", "Expository comparison omitted for page discipline"),
    ("Supplemental composite-policy comparison", "Repository only", "scripts/analyze_transfer_regime_results.py", "results/working/favorita_primary_period_results.csv; results/working/cross_dataset_period_results.csv", "inline equal-weight ensemble plus decision_layer.practitioner_baselines.rate_limit_signal", "shared dynamic-inventory rollout state", "results/final/time_block_bootstrap.csv; paper_submission_ieom/tables/supplementary/strong_baseline_dynamic_table.tex", "Detailed policy costs supporting Table 4 Panel A"),
    ("Supplemental protocol-envelope table", "Repository only", "scripts/run_transfer_regime_experiments.py", "results/working/cross_dataset_forecast_panel.csv; validation partitions", "inline equal-weight and validation-weighted ensemble formulas in scripts/run_transfer_regime_experiments.py", "shared dynamic-inventory rollout state", "results/final/cross_dataset_frozen_transfer.csv; paper_submission_ieom/tables/supplementary/transfer_results_table.tex", "Retrospective policy-family coverage summarized in manuscript prose"),
    ("Supplemental feasibility-aware comparisons", "Repository only; no manuscript table, figure, or headline", "scripts/run_favorita_minimal_pipeline.py; scripts/run_dynamic_policy_experiments.py", "Favorita candidate forecasts and validation-derived expected losses", "decision_layer.feasibility_dp_selector.DPFeasibilitySelector and BudgetedDPFeasibilitySelector", "feasibility state: candidate model, previous model, expected loss, transition/switch terms; no inventory/backlog/pipeline state", "results/final/dynamic_operational_summary.csv; results/final/time_block_bootstrap.csv rows labelled feasibility_aware", "Supplemental only; structurally distinct from the manuscript inventory-state DP"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "manuscript_object",
        "manuscript_location",
        "generating_script",
        "input_files",
        "selector_implementation",
        "state_representation",
        "output_artifact",
        "interpretive_scope",
    ]
    frame = pd.DataFrame(ROWS, columns=columns)
    frame.to_csv(OUT / "result_implementation_lineage.csv", index=False)

    lines = [
        "# Manuscript Result-to-Implementation Lineage",
        "",
        "This is the human-readable companion to `result_implementation_lineage.csv`. The manuscript's multi-period, governance, and approximation-sensitivity evidence uses only `decision_layer.dynamic_inventory_policies.select_dynamic_policy`, an inventory/backlog/pipeline beam search. The structurally different feasibility selector supports supplemental `feasibility_aware` comparisons only and supports no manuscript table, figure, or headline result.",
        "",
        "| Object | Location | Generating script | Selector implementation | Output artifact | Scope |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        values = [
            row.manuscript_object,
            row.manuscript_location,
            row.generating_script,
            row.selector_implementation,
            row.output_artifact,
            row.interpretive_scope,
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    lines.extend([
        "",
        "## Information-boundary implementation",
        "",
        "Both selector interfaces call `decision_layer.no_leakage.require_no_future_outcomes` at runtime. For the manuscript DP, candidate forecast paths enter action selection; realized demand enters only the separate dynamic-inventory simulator after the model path and plans are returned. Regression tests reject realized-outcome fields at the primary selector boundary.",
        "",
        "## Ensemble implementation",
        "",
        "The reported equal-weight and validation-weighted ensembles are the live functions in `scripts/run_transfer_regime_experiments.py`. The unused earlier helper is retained under `archive/legacy/ensemble_selector.py` and supports no current result.",
        "",
        "## Replay cost accounting",
        "",
        "The base replay weights are explicit in `scripts/analyze_transfer_regime_results.py`. M5/Walmart transfer, multi-period, governance, targeted-strength, and approximation-sensitivity rollouts explicitly set the capacity-shortfall weight to zero and preserve the violation as a separate operational outcome. These settings match the block-specific disclosure after Equation (7) and do not alter the frozen results.",
        "",
    ])
    (OUT / "result_implementation_lineage.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
