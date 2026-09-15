"""Run a tractable approximation-sensitivity audit for the evaluated DP policy.

The audit varies beam width and state rounding on a frozen M5 subset for the
long-lead and tight-capacity regimes. It is a robustness check, not an exact-DP
benchmark and not a new policy-selection experiment.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from decision_layer.dynamic_inventory_policies import DynamicPolicyConfig, select_dynamic_policy  # noqa: E402
from decision_layer.practitioner_baselines import rate_limit_signal  # noqa: E402
from planning_environment.dynamic_inventory_simulator import DynamicInventoryConfig, simulate_dynamic_inventory  # noqa: E402
from run_transfer_regime_experiments import STRICT_RATE, m5_panels  # noqa: E402
from utils.config import load_config  # noqa: E402


def dynamic_cost(actual, plan, lead, capacity, mean, rate, switch=None):
    """Return the shared rollout cost for a fixed plan path."""
    simulation = simulate_dynamic_inventory(
        actual,
        plan,
        DynamicInventoryConfig(
            lead_time=lead,
            holding_cost_rate=1.0,
            shortage_cost_rate=5.0,
            order_change_cost_rate=0.05,
            expedite_cost_rate=0.0,
            replenishment_capacity=capacity,
            max_order_change_rate=rate,
            initial_on_hand=(lead + 1) * mean,
            initial_order_quantity=mean,
        ),
        model_switch=np.zeros(len(actual)) if switch is None else switch,
        switching_cost_rate=0.5,
    )
    return float(simulation["total_dynamic_cost"].sum())


def main():
    """Evaluate nine approximation settings on two prespecified M5 regimes."""
    panel = m5_panels(100, load_config(ROOT / "configs/default.yaml"))["item_store"]
    validation, test = panel[panel["split"] == "validation"], panel[panel["split"] == "test"]
    regimes = {"long_lead": (7, 1.25, 0.50), "tight_capacity": (4, 0.75, 0.15)}
    rows = []
    for series_id, group in test.groupby("series_id", sort=True):
        pivot = group.pivot(index="date", columns="model_name", values="forecast").sort_index()
        actual = group.drop_duplicates("date").sort_values("date")["actual"].to_numpy(float)
        history = validation[validation["series_id"] == series_id].drop_duplicates("date")["actual"]
        mean = max(float(history.mean()), 1.0)
        safety = 0.5 * float(history.std() or 0.0)
        candidates = {str(column): pivot[column].to_numpy(float) for column in pivot.columns}
        ensemble = pivot.mean(axis=1).to_numpy(float)
        for regime, (lead, capacity_multiplier, rate) in regimes.items():
            capacity = mean * capacity_multiplier
            baseline_plan = rate_limit_signal(ensemble * (lead + 1) + safety, STRICT_RATE)
            baseline_cost = dynamic_cost(actual, baseline_plan, lead, capacity, mean, rate)
            for beam_width in (48, 96, 192):
                for rounding_multiplier in (0.05, 0.10, 0.20):
                    config = DynamicPolicyConfig(
                        lead_time=lead,
                        shortage_cost_rate=5.0,
                        switching_cost_rate=0.5,
                        capacity=capacity,
                        max_order_change_rate=rate,
                        frozen_horizon=1 if regime == "long_lead" else 2,
                        backlog_persistence=1.0,
                        state_rounding=max(mean * rounding_multiplier, 0.25),
                        beam_width=beam_width,
                    )
                    output = select_dynamic_policy(candidates, safety, (lead + 1) * mean, mean, config, "dp")
                    treatment_cost = dynamic_cost(actual, output["plan"], lead, capacity, mean, rate, output["switch"])
                    rows.append({
                        "planning_unit": series_id,
                        "regime": regime,
                        "beam_width": beam_width,
                        "rounding_multiplier": rounding_multiplier,
                        "baseline_cost": baseline_cost,
                        "approximate_dp_cost": treatment_cost,
                        "relative_cost_change_pct": 100.0 * (treatment_cost - baseline_cost) / max(baseline_cost, 1e-12),
                    })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["regime", "beam_width", "rounding_multiplier"], as_index=False).agg(
        planning_units=("planning_unit", "nunique"),
        mean_baseline_cost=("baseline_cost", "mean"),
        mean_approximate_dp_cost=("approximate_dp_cost", "mean"),
        mean_relative_cost_change_pct=("relative_cost_change_pct", "mean"),
        median_relative_cost_change_pct=("relative_cost_change_pct", "median"),
        favorable_share=("relative_cost_change_pct", lambda values: float((values < 0).mean())),
    )
    summary["aggregate_cost_change_pct"] = 100.0 * (
        summary["mean_approximate_dp_cost"] - summary["mean_baseline_cost"]
    ) / summary["mean_baseline_cost"].clip(lower=1e-12)
    output_dir = ROOT / "results" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "dp_approximation_sensitivity_unit.csv", index=False)
    summary.to_csv(output_dir / "dp_approximation_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
