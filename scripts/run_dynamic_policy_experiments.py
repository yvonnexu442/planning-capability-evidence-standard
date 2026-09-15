"""Generate held-out dynamic inventory-policy evidence.

The runner reuses saved Favorita candidate forecasts. Practitioner control
parameters are selected on validation dates only and frozen for test rollout.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decision_layer.practitioner_baselines import (  # noqa: E402
    BASELINE_METADATA,
    exponential_smooth_signal,
    frozen_horizon_signal,
    rate_limit_signal,
)
from planning_environment.dynamic_inventory_simulator import (  # noqa: E402
    DynamicInventoryConfig,
    simulate_dynamic_inventory,
    summarize_dynamic_inventory,
)


RESULT_DIR = REPO_ROOT / "results" / "working"
BASE_TABLE_DIR = RESULT_DIR / "favorita_base" / "tables"
FORECAST_PATH = BASE_TABLE_DIR / "favorita_forecasts.csv"
DECISION_PATH = BASE_TABLE_DIR / "favorita_decision_records.csv"
SEED = 42


def wape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    denominator = float(np.abs(actual_array).sum())
    return 0.0 if denominator <= 0.0 else float(np.abs(actual_array - forecast_array).sum() / denominator)


def mase(actual: Sequence[float], forecast: Sequence[float], scale: float) -> float:
    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))
    return float(errors.mean() / max(float(scale), 1e-8))


def load_inputs(max_series: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the explicitly generated Favorita forecast and policy tables."""
    for path in (FORECAST_PATH, DECISION_PATH):
        if not path.exists():
            raise FileNotFoundError(
                "Missing {}. Run the Favorita base pipeline before the dynamic experiments.".format(path)
            )
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    decisions = pd.read_csv(DECISION_PATH, parse_dates=["date"])
    required_splits = {"validation", "test"}
    if not required_splits.issubset(set(forecasts["split"].unique())):
        raise ValueError("Saved Favorita forecasts must contain validation and test rows.")
    series = sorted(forecasts["series_id"].astype(str).unique())[:max_series]
    forecasts = forecasts[forecasts["series_id"].astype(str).isin(series)].copy()
    decisions = decisions[decisions["series_id"].astype(str).isin(series)].copy()
    return forecasts, decisions


def select_accuracy_model(validation: pd.DataFrame) -> str:
    records = []
    for model, group in validation.groupby("model_name"):
        actual = group.drop_duplicates(["series_id", "date"])["actual"]
        forecast = group["forecast"]
        records.append((wape(actual, forecast), str(model)))
    return min(records)[1]


def safety_stock_lookup(validation: pd.DataFrame, multiplier: float = 0.5) -> Dict[str, float]:
    actual = validation[["series_id", "date", "actual"]].drop_duplicates(["series_id", "date"])
    return (actual.groupby("series_id")["actual"].std().fillna(0.0).clip(lower=0.0) * multiplier).to_dict()


def naive_scale_lookup(validation: pd.DataFrame) -> Dict[str, float]:
    actual = validation[["series_id", "date", "actual"]].drop_duplicates(["series_id", "date"]).sort_values(["series_id", "date"])
    return actual.groupby("series_id")["actual"].apply(lambda x: float(x.diff().abs().dropna().mean() or 1.0)).to_dict()


def base_signals(
    split_frame: pd.DataFrame,
    accuracy_model: str,
    safety: Mapping[str, float],
) -> Dict[str, Dict[str, np.ndarray]]:
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for series_id, group in split_frame.groupby("series_id"):
        group = group.sort_values(["date", "model_name"])
        actual_rows = group.drop_duplicates("date").sort_values("date")
        accuracy = group[group["model_name"] == accuracy_model].sort_values("date")
        if len(accuracy) != len(actual_rows):
            continue
        ensemble = group.groupby("date", as_index=False).agg(forecast=("forecast", "mean")).sort_values("date")
        buffer = float(safety.get(series_id, 0.0))
        output[str(series_id)] = {
            "date": actual_rows["date"].to_numpy(),
            "actual": actual_rows["actual"].to_numpy(dtype=float),
            "accuracy_forecast": accuracy["forecast"].to_numpy(dtype=float),
            "accuracy_target": np.maximum(accuracy["forecast"].to_numpy(dtype=float) + buffer, 0.0),
            "ensemble_forecast": ensemble["forecast"].to_numpy(dtype=float),
            "ensemble_target": np.maximum(ensemble["forecast"].to_numpy(dtype=float) + buffer, 0.0),
        }
    return output


def validation_cost(signal: np.ndarray, actual: np.ndarray) -> float:
    config = DynamicInventoryConfig(
        lead_time=2,
        holding_cost_rate=1.0,
        shortage_cost_rate=5.0,
        replenishment_capacity=None,
        max_order_change_rate=None,
        initial_on_hand=float(signal[0] * 3.0) if len(signal) else 0.0,
        initial_order_quantity=float(np.mean(actual)) if len(actual) else None,
    )
    return summarize_dynamic_inventory(simulate_dynamic_inventory(actual, signal * 3.0, config))["total_dynamic_cost"]


def tune_controls(validation_signals: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, float]:
    grids = {
        "rate_limit": [0.10, 0.20, 0.40],
        "alpha": [0.25, 0.50, 0.75],
        "freeze_periods": [2, 7, 14],
    }
    selected: Dict[str, float] = {}
    for name, candidates in grids.items():
        scored = []
        for candidate in candidates:
            costs = []
            for values in validation_signals.values():
                signal = values["accuracy_target"]
                if name == "rate_limit":
                    controlled = rate_limit_signal(signal, float(candidate))
                elif name == "alpha":
                    controlled = exponential_smooth_signal(signal, float(candidate))
                else:
                    controlled = frozen_horizon_signal(signal, int(candidate))
                costs.append(validation_cost(controlled, values["actual"]))
            scored.append((float(np.mean(costs)), float(candidate)))
        selected[name] = min(scored)[1]
    return selected


def scenario_design() -> pd.DataFrame:
    rows = []
    for lead_time in (0, 2, 7):
        for shortage_ratio in (2.0, 5.0, 10.0):
            for capacity_name, capacity_ratio, rate in (
                ("flexible", 2.0, None),
                ("moderate", 1.25, 0.50),
                ("tight", 0.75, 0.20),
            ):
                rows.append(
                    {
                        "scenario_id": "L{}_B{}_{}".format(lead_time, int(shortage_ratio), capacity_name),
                        "lead_time": lead_time,
                        "holding_cost_rate": 1.0,
                        "shortage_cost_rate": shortage_ratio,
                        "capacity_regime": capacity_name,
                        "replenishment_capacity_ratio": capacity_ratio,
                        "max_order_change_rate": rate,
                    }
                )
    return pd.DataFrame(rows)


def feasibility_signal_lookup(decisions: pd.DataFrame) -> Dict[str, Mapping[str, np.ndarray]]:
    selected = decisions[decisions["strategy"] == "feasibility_aware_selector"].copy()
    output = {}
    for series_id, group in selected.groupby("series_id"):
        group = group.sort_values("date")
        models = group["selected_model"].astype(str).to_numpy()
        switches = np.r_[0.0, (models[1:] != models[:-1]).astype(float)]
        output[str(series_id)] = {
            "forecast": group["forecast"].to_numpy(dtype=float),
            "target": group["planning_signal"].to_numpy(dtype=float),
            "switch": switches,
        }
    return output


def strategy_signals(
    values: Mapping[str, np.ndarray],
    controls: Mapping[str, float],
    feasibility: Mapping[str, np.ndarray],
) -> Iterable[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    zeros = np.zeros_like(values["actual"], dtype=float)
    yield "accuracy_first", values["accuracy_forecast"], values["accuracy_target"], zeros
    yield "accuracy_rate_limited", values["accuracy_forecast"], rate_limit_signal(values["accuracy_target"], controls["rate_limit"]), zeros
    yield "accuracy_smoothed", values["accuracy_forecast"], exponential_smooth_signal(values["accuracy_target"], controls["alpha"]), zeros
    yield "accuracy_frozen_horizon", values["accuracy_forecast"], frozen_horizon_signal(values["accuracy_target"], int(controls["freeze_periods"])), zeros
    yield "accuracy_order_up_to", values["accuracy_forecast"], values["accuracy_target"], zeros
    yield "simple_ensemble", values["ensemble_forecast"], values["ensemble_target"], zeros
    yield "simple_ensemble_rate_limited", values["ensemble_forecast"], rate_limit_signal(values["ensemble_target"], controls["rate_limit"]), zeros
    if feasibility:
        yield "feasibility_aware", feasibility["forecast"], feasibility["target"], feasibility["switch"]


def run(max_series: int) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    forecasts, decisions = load_inputs(max_series)
    validation = forecasts[forecasts["split"] == "validation"].copy()
    test = forecasts[forecasts["split"] == "test"].copy()
    accuracy_model = select_accuracy_model(validation)
    safety = safety_stock_lookup(validation)
    scale = naive_scale_lookup(validation)
    validation_signals = base_signals(validation, accuracy_model, safety)
    test_signals = base_signals(test, accuracy_model, safety)
    controls = tune_controls(validation_signals)
    scenarios = scenario_design()
    feasibility = feasibility_signal_lookup(decisions)

    scenarios.to_csv(RESULT_DIR / "scenario_design.csv", index=False)
    split_manifest = pd.DataFrame(
        [
            {
                "dataset": "Favorita",
                "split": split,
                "start_date": group["date"].min().date().isoformat(),
                "end_date": group["date"].max().date().isoformat(),
                "periods": group["date"].nunique(),
            }
            for split, group in forecasts.groupby("split")
        ]
    )
    split_manifest.to_csv(RESULT_DIR / "split_manifest.csv", index=False)
    (RESULT_DIR / "frozen_configuration.json").write_text(
        json.dumps(
            {
                "random_seed": SEED,
                "accuracy_model_selected_on_validation": accuracy_model,
                "safety_stock_multiplier": 0.5,
                "validation_selected_controls": controls,
                "max_series": max_series,
                "test_outcomes_used_for_tuning": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    records: List[Dict[str, object]] = []
    runtime_records: List[Dict[str, object]] = []
    for scenario in scenarios.to_dict("records"):
        for series_id, values in test_signals.items():
            mean_demand = float(validation_signals[series_id]["actual"].mean())
            capacity = float(scenario["replenishment_capacity_ratio"] * max(mean_demand, 1.0))
            base_config = DynamicInventoryConfig(
                lead_time=int(scenario["lead_time"]),
                holding_cost_rate=float(scenario["holding_cost_rate"]),
                shortage_cost_rate=float(scenario["shortage_cost_rate"]),
                order_change_cost_rate=0.05,
                expedite_cost_rate=0.10,
                replenishment_capacity=capacity,
                max_order_change_rate=None if pd.isna(scenario["max_order_change_rate"]) else float(scenario["max_order_change_rate"]),
                backlog_mode="backorder",
            )
            for strategy, forecast, target, switches in strategy_signals(values, controls, feasibility.get(series_id, {})):
                protection_periods = int(scenario["lead_time"]) + 1
                order_up_to_target = np.asarray(target, dtype=float) * protection_periods
                config = replace(
                    base_config,
                    initial_on_hand=float(order_up_to_target[0]) if len(order_up_to_target) else 0.0,
                    initial_order_quantity=mean_demand,
                )
                started = time.perf_counter()
                simulation = simulate_dynamic_inventory(
                    values["actual"],
                    order_up_to_target,
                    config,
                    model_switch=switches,
                    switching_cost_rate=0.05,
                )
                elapsed = time.perf_counter() - started
                summary = summarize_dynamic_inventory(simulation)
                record = {
                    "dataset": "Favorita",
                    "planning_unit": series_id,
                    "split": "test",
                    "strategy": strategy,
                    "scenario_id": scenario["scenario_id"],
                    "lead_time": scenario["lead_time"],
                    "capacity_regime": scenario["capacity_regime"],
                    "shortage_cost_rate": scenario["shortage_cost_rate"],
                    "WAPE": wape(values["actual"], forecast),
                    "MASE": mase(values["actual"], forecast, scale.get(series_id, 1.0)),
                    **summary,
                }
                records.append(record)
                runtime_records.append(
                    {
                        "dataset": "Favorita",
                        "planning_unit": series_id,
                        "strategy": strategy,
                        "scenario_id": scenario["scenario_id"],
                        "runtime_seconds": elapsed,
                    }
                )

    results = pd.DataFrame(records)
    results["dynamic_regret"] = results["total_dynamic_cost"] - results.groupby(["planning_unit", "scenario_id"])["total_dynamic_cost"].transform("min")
    results.to_csv(RESULT_DIR / "dynamic_inventory_results.csv", index=False)
    results.to_csv(RESULT_DIR / "strong_baseline_results.csv", index=False)
    pd.DataFrame(runtime_records).to_csv(RESULT_DIR / "runtime_complexity.csv", index=False)

    metadata = pd.DataFrame.from_dict(BASELINE_METADATA, orient="index").reset_index().rename(columns={"index": "strategy"})
    metadata.loc[len(metadata)] = ["feasibility_aware", False, True, True, True]
    metadata.to_csv(RESULT_DIR / "strong_baseline_metadata.csv", index=False)
    print("Wrote {} unit-scenario-strategy rows to {}".format(len(results), RESULT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-series", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(max_series=args.max_series)
