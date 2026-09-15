"""Supplemental feasibility-state selectors for finite-horizon planning.

These selectors support repository-only ``feasibility_aware`` comparisons. They
are structurally distinct from the inventory/backlog/pipeline beam search in
``dynamic_inventory_policies.py`` and support no current manuscript exhibit or
headline result. See ``reports/final/result_implementation_lineage.*``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from decision_layer.no_leakage import require_no_future_outcomes
from evaluation.planning_utility import planning_loss_weight
from planning_environment.planning_actions import forecast_to_inventory_target


LossKey = Tuple[str, str]


@dataclass(frozen=True)
class SelectorState:
    """State retained by dynamic selectors."""

    cost: float
    path: Tuple[str, ...]
    rows: Tuple[int, ...]
    switches: int
    previous_plan: float


class GreedyFeasibilitySelector:
    """Select the lowest one-step expected operational-cost model per period.

    The selector is deployable: candidate scoring uses validation-derived
    forecast and inventory-cost estimates plus current forecast-implied plan
    movement. It does not use realized test-period demand.
    """

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        calibration_group_column: str = "series_id",
        strategy_name: str = "greedy_feasibility_selector",
    ) -> None:
        self.expected_losses = expected_losses
        self.global_expected_losses = global_expected_losses or {}
        self.weights = weights or {}
        self.switch_penalty = float(switch_penalty)
        self.max_plan_change_rate = float(max_plan_change_rate)
        self.calibration_group_column = calibration_group_column
        self.strategy_name = strategy_name

    def select(self, candidate_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Return selected forecast rows for all series and periods."""
        require_no_future_outcomes(candidate_forecasts, "{}.select".format(self.__class__.__name__))
        selected_rows: List[pd.Series] = []
        for _, series_frame in candidate_forecasts.sort_values(["series_id", "date", "model_name"]).groupby("series_id", sort=False):
            previous_model = None
            previous_plan = None
            for _, period_frame in series_frame.groupby("date", sort=False):
                scored_candidates = []
                for row in _period_candidates(period_frame):
                    score, plan = self._stage_cost(row, previous_model, previous_plan)
                    scored_candidates.append((score, str(row["model_name"]), plan, row))
                _, _, selected_plan, selected = min(scored_candidates, key=lambda value: (value[0], value[1]))
                selected_rows.append(selected)
                previous_model = str(selected["model_name"])
                previous_plan = selected_plan
        return _selected_frame(selected_rows, self.strategy_name)

    def _stage_cost(
        self,
        row: pd.Series,
        previous_model: Optional[str],
        previous_plan: Optional[float],
    ) -> Tuple[float, float]:
        plan = _planning_signal(row)
        base_cost = self._base_expected_cost(row)
        plan_change_pct, execution_violation = _transition_burden(
            current_plan=plan,
            previous_plan=previous_plan,
            max_plan_change_rate=self.max_plan_change_rate,
        )
        switch_cost = 0.0 if previous_model is None or previous_model == str(row["model_name"]) else self.switch_penalty
        score = (
            base_cost
            + planning_loss_weight(self.weights, "lambda_volatility") * plan_change_pct
            + planning_loss_weight(self.weights, "lambda_switch") * switch_cost
            + planning_loss_weight(self.weights, "lambda_execution") * execution_violation
        )
        return float(score), plan

    def _base_expected_cost(self, row: pd.Series) -> float:
        model_name = str(row["model_name"])
        calibration_group = str(row.get(self.calibration_group_column, row.get("series_id", "global")))
        losses = self.expected_losses.get(
            (calibration_group, model_name),
            self.global_expected_losses.get(model_name, self.expected_losses.get(("global", model_name), {})),
        )
        return (
            planning_loss_weight(self.weights, "alpha_forecast") * float(losses.get("wape", 1.0))
            + planning_loss_weight(self.weights, "beta_inventory") * float(losses.get("inventory_cost_per_demand_unit", 0.0))
        )


class DPFeasibilitySelector(GreedyFeasibilitySelector):
    """Finite-horizon dynamic-programming selector for cumulative loss."""

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        calibration_group_column: str = "series_id",
        strategy_name: str = "dp_feasibility_selector",
    ) -> None:
        super().__init__(
            expected_losses=expected_losses,
            global_expected_losses=global_expected_losses,
            weights=weights,
            switch_penalty=switch_penalty,
            max_plan_change_rate=max_plan_change_rate,
            calibration_group_column=calibration_group_column,
            strategy_name=strategy_name,
        )

    def select(self, candidate_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Return selected forecast rows from finite-horizon DP."""
        require_no_future_outcomes(candidate_forecasts, "{}.select".format(self.__class__.__name__))
        selected_rows: List[pd.Series] = []
        for _, series_frame in candidate_forecasts.sort_values(["series_id", "date", "model_name"]).groupby("series_id", sort=False):
            selected_rows.extend(self._select_series(series_frame))
        return _selected_frame(selected_rows, self.strategy_name)

    def _select_series(self, series_frame: pd.DataFrame) -> List[pd.Series]:
        dp: Dict[str, SelectorState] = {}
        for _, period_frame in series_frame.groupby("date", sort=False):
            candidates = _period_candidates(period_frame)
            next_dp: Dict[str, SelectorState] = {}
            if not dp:
                for row in candidates:
                    model_name = str(row["model_name"])
                    stage_cost, plan = self._stage_cost(row, previous_model=None, previous_plan=None)
                    next_dp[model_name] = SelectorState(
                        cost=stage_cost,
                        path=(model_name,),
                        rows=(int(row.name),),
                        switches=0,
                        previous_plan=plan,
                    )
            else:
                for row in candidates:
                    model_name = str(row["model_name"])
                    best_state = None
                    for previous_model, state in dp.items():
                        stage_cost, plan = self._stage_cost(row, previous_model=previous_model, previous_plan=state.previous_plan)
                        switches = state.switches + int(previous_model != model_name)
                        candidate_state = SelectorState(
                            cost=state.cost + stage_cost,
                            path=state.path + (model_name,),
                            rows=state.rows + (int(row.name),),
                            switches=switches,
                            previous_plan=plan,
                        )
                        best_state = _min_state(best_state, candidate_state)
                    if best_state is not None:
                        next_dp[model_name] = best_state
            dp = next_dp
        if not dp:
            return []
        final_state = min(dp.values(), key=_state_sort_key)
        return [series_frame.loc[index] for index in final_state.rows]


class BudgetedDPFeasibilitySelector(DPFeasibilitySelector):
    """Finite-horizon DP selector with a hard maximum switch budget."""

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        max_switches: int = 2,
        calibration_group_column: str = "series_id",
        strategy_name: str = "budgeted_dp_feasibility_selector",
    ) -> None:
        super().__init__(
            expected_losses=expected_losses,
            global_expected_losses=global_expected_losses,
            weights=weights,
            switch_penalty=switch_penalty,
            max_plan_change_rate=max_plan_change_rate,
            calibration_group_column=calibration_group_column,
            strategy_name=strategy_name,
        )
        self.max_switches = int(max_switches)

    def select(self, candidate_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Return selected rows, using incumbent-stays fallback on budget failure."""
        require_no_future_outcomes(candidate_forecasts, "{}.select".format(self.__class__.__name__))
        selected_rows: List[pd.Series] = []
        for _, series_frame in candidate_forecasts.sort_values(["series_id", "date", "model_name"]).groupby("series_id", sort=False):
            rows = self._select_series(series_frame)
            if rows:
                selected_rows.extend(_annotate_fallback(rows, used=False, fallback_type="none", reason="none"))
            else:
                selected_rows.extend(self._incumbent_stays_fallback_series(series_frame))
        return _selected_frame(selected_rows, self.strategy_name)

    def _select_series(self, series_frame: pd.DataFrame) -> List[pd.Series]:
        dp: Dict[Tuple[str, int], SelectorState] = {}
        for _, period_frame in series_frame.groupby("date", sort=False):
            candidates = _period_candidates(period_frame)
            next_dp: Dict[Tuple[str, int], SelectorState] = {}
            if not dp:
                for row in candidates:
                    model_name = str(row["model_name"])
                    stage_cost, plan = self._stage_cost(row, previous_model=None, previous_plan=None)
                    state = SelectorState(
                        cost=stage_cost,
                        path=(model_name,),
                        rows=(int(row.name),),
                        switches=0,
                        previous_plan=plan,
                    )
                    next_dp[(model_name, 0)] = _min_state(next_dp.get((model_name, 0)), state)
            else:
                for row in candidates:
                    model_name = str(row["model_name"])
                    for (previous_model, used_switches), state in dp.items():
                        new_switches = used_switches + int(previous_model != model_name)
                        if new_switches > self.max_switches:
                            continue
                        stage_cost, plan = self._stage_cost(row, previous_model=previous_model, previous_plan=state.previous_plan)
                        candidate_state = SelectorState(
                            cost=state.cost + stage_cost,
                            path=state.path + (model_name,),
                            rows=state.rows + (int(row.name),),
                            switches=new_switches,
                            previous_plan=plan,
                        )
                        key = (model_name, new_switches)
                        next_dp[key] = _min_state(next_dp.get(key), candidate_state)
            dp = next_dp
            if not dp:
                warnings.warn(
                    "BudgetedDPFeasibilitySelector: no feasible path under max_switches={}. "
                    "Falling back to incumbent-stays policy.".format(self.max_switches),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return []
        if not dp:
            return []
        final_state = min(dp.values(), key=_state_sort_key)
        return [series_frame.loc[index] for index in final_state.rows]

    def _incumbent_stays_fallback_series(self, series_frame: pd.DataFrame) -> List[pd.Series]:
        """Return an incumbent-stays fallback path for a single series.

        The fallback keeps the pipeline evaluable when no path remains under the
        configured switch budget. It chooses the first period's lowest-cost
        candidate as the incumbent and holds that model whenever it is available.
        """
        period_frames = list(series_frame.sort_values(["date", "model_name"]).groupby("date", sort=False))
        if not period_frames:
            return []

        _, first_period = period_frames[0]
        first_scores = []
        for row in _period_candidates(first_period):
            score, plan = self._stage_cost(row, previous_model=None, previous_plan=None)
            first_scores.append((score, str(row["model_name"]), plan, row))
        _, incumbent_model, previous_plan, selected = min(first_scores, key=lambda value: (value[0], value[1]))

        fallback_rows: List[pd.Series] = [selected]
        incumbent_missing = False
        for date_value, period_frame in period_frames[1:]:
            candidates = _period_candidates(period_frame)
            incumbent_matches = [row for row in candidates if str(row["model_name"]) == incumbent_model]
            if incumbent_matches:
                selected = incumbent_matches[0]
                _, previous_plan = self._stage_cost(
                    selected,
                    previous_model=incumbent_model,
                    previous_plan=previous_plan,
                )
            else:
                incumbent_missing = True
                warnings.warn(
                    "BudgetedDPFeasibilitySelector: incumbent model '{}' is missing for date {}. "
                    "Incumbent-stays fallback is potentially non-strict.".format(incumbent_model, date_value),
                    RuntimeWarning,
                    stacklevel=2,
                )
                scored_candidates = []
                for row in candidates:
                    score, plan = self._stage_cost(row, previous_model=incumbent_model, previous_plan=previous_plan)
                    scored_candidates.append((score, str(row["model_name"]), plan, row))
                _, _, previous_plan, selected = min(scored_candidates, key=lambda value: (value[0], value[1]))
            fallback_rows.append(selected)

        reason = "no_budget_feasible_path_under_max_switches_{}; incumbent_model_held_when_available".format(
            self.max_switches
        )
        if incumbent_missing:
            reason = "{}; incumbent_missing_in_later_period_potentially_non_strict".format(reason)
        return _annotate_fallback(
            fallback_rows,
            used=True,
            fallback_type="incumbent_stays",
            reason=reason,
        )


class OracleDPFeasibilitySelector(DPFeasibilitySelector):
    """Non-deployable Realized-Inventory Oracle DP selector.

    This selector uses realized test-period inventory outcomes, giving the DP
    access to future outcome information that is unavailable in deployment. It
    is included as a diagnostic upper-bound benchmark for the value of
    inventory-outcome information and must never be reported as deployable.

    It is not a perfect-forecast oracle. Forecast loss still uses the
    validation-derived estimate, while inventory cost is looked up by period
    using ``(series_id, model_name, date)`` whenever possible.
    """

    ORACLE_LABEL = "[REALIZED-INVENTORY ORACLE - non-deployable]"
    ORACLE_TYPE = "realized_inventory_oracle"

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        realized_inventory_costs: Mapping[Tuple[Any, ...], float],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        calibration_group_column: str = "series_id",
        strategy_name: str = "oracle_dp_feasibility_selector",
    ) -> None:
        super().__init__(
            expected_losses=expected_losses,
            global_expected_losses=global_expected_losses,
            weights=weights,
            switch_penalty=switch_penalty,
            max_plan_change_rate=max_plan_change_rate,
            calibration_group_column=calibration_group_column,
            strategy_name=strategy_name,
        )
        self.realized_inventory_costs = dict(realized_inventory_costs)

    def _base_expected_cost(self, row: pd.Series) -> float:
        """Use realized test inventory cost instead of validation inventory cost."""
        model_name = str(row["model_name"])
        calibration_group = str(row.get(self.calibration_group_column, row.get("series_id", "global")))
        realized_inventory = self._period_realized_inventory_cost(row, model_name, calibration_group)
        losses = self.expected_losses.get(
            (calibration_group, model_name),
            self.global_expected_losses.get(model_name, self.expected_losses.get(("global", model_name), {})),
        )
        return (
            planning_loss_weight(self.weights, "alpha_forecast") * float(losses.get("wape", 1.0))
            + planning_loss_weight(self.weights, "beta_inventory") * float(realized_inventory)
        )

    def _period_realized_inventory_cost(
        self,
        row: pd.Series,
        model_name: str,
        calibration_group: str,
    ) -> float:
        """Return period-specific realized inventory cost for one candidate."""
        series_id = str(row.get("series_id", calibration_group))
        date_value = _normalized_date_key(row.get("date", None))
        lookup_keys = [
            (series_id, model_name, date_value),
            (calibration_group, model_name, date_value),
            ("global", model_name, date_value),
            # Backward-compatible aggregate fallbacks are kept only for older
            # cached artifacts; current pipelines generate period-level keys.
            (series_id, model_name),
            (calibration_group, model_name),
            ("global", model_name),
        ]
        for key in lookup_keys:
            if key in self.realized_inventory_costs:
                return float(self.realized_inventory_costs[key])
        return 0.0


class FullOutcomeOracleDPFeasibilitySelector(OracleDPFeasibilitySelector):
    """Non-deployable DP selector with realized test forecast and inventory outcomes.

    This is the strongest candidate-model oracle in the benchmark ladder. It is
    still constrained to choose among the supplied forecast candidates, but its
    stage cost can use realized test-period forecast error and realized
    inventory outcome information. It is therefore not deployable and should be
    used only as an upper-bound diagnostic.

    Under the repository's default objective, ``alpha_forecast=0.0``, so the
    realized forecast-error term is diagnostic unless an explicit sensitivity
    setting gives it positive weight.
    """

    ORACLE_LABEL = "[FULL-OUTCOME ORACLE DP - non-deployable]"
    ORACLE_TYPE = "full_outcome_oracle"

    def __init__(
        self,
        expected_losses: Mapping[LossKey, Mapping[str, float]],
        realized_inventory_costs: Mapping[Tuple[Any, ...], float],
        realized_forecast_losses: Mapping[Tuple[Any, ...], float],
        global_expected_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        switch_penalty: float = 0.02,
        max_plan_change_rate: float = 0.20,
        calibration_group_column: str = "series_id",
        strategy_name: str = "full_outcome_oracle_dp_feasibility_selector",
    ) -> None:
        super().__init__(
            expected_losses=expected_losses,
            realized_inventory_costs=realized_inventory_costs,
            global_expected_losses=global_expected_losses,
            weights=weights,
            switch_penalty=switch_penalty,
            max_plan_change_rate=max_plan_change_rate,
            calibration_group_column=calibration_group_column,
            strategy_name=strategy_name,
        )
        self.realized_forecast_losses = dict(realized_forecast_losses)

    def _base_expected_cost(self, row: pd.Series) -> float:
        """Use realized forecast error and inventory outcome for one stage."""
        model_name = str(row["model_name"])
        calibration_group = str(row.get(self.calibration_group_column, row.get("series_id", "global")))
        realized_inventory = self._period_realized_inventory_cost(row, model_name, calibration_group)
        realized_forecast = self._period_realized_forecast_loss(row, model_name, calibration_group)
        return (
            planning_loss_weight(self.weights, "alpha_forecast") * float(realized_forecast)
            + planning_loss_weight(self.weights, "beta_inventory") * float(realized_inventory)
        )

    def _period_realized_forecast_loss(
        self,
        row: pd.Series,
        model_name: str,
        calibration_group: str,
    ) -> float:
        """Return period-specific realized forecast loss for one candidate."""
        series_id = str(row.get("series_id", calibration_group))
        date_value = _normalized_date_key(row.get("date", None))
        lookup_keys = [
            (series_id, model_name, date_value),
            (calibration_group, model_name, date_value),
            ("global", model_name, date_value),
            (series_id, model_name),
            (calibration_group, model_name),
            ("global", model_name),
        ]
        for key in lookup_keys:
            if key in self.realized_forecast_losses:
                return float(self.realized_forecast_losses[key])

        losses = self.expected_losses.get(
            (calibration_group, model_name),
            self.global_expected_losses.get(model_name, self.expected_losses.get(("global", model_name), {})),
        )
        return float(losses.get("wape", 1.0))


def _period_candidates(period_frame: pd.DataFrame) -> List[pd.Series]:
    """Return deterministic candidate rows for a single period."""
    return [
        row
        for _, row in period_frame.sort_values("model_name").drop_duplicates(["model_name"], keep="first").iterrows()
    ]


def _selected_frame(rows: Sequence[pd.Series], strategy_name: str) -> pd.DataFrame:
    """Return selected rows as a strategy decision frame."""
    frame = pd.DataFrame([row.to_dict() for row in rows])
    if frame.empty:
        return frame
    frame["selected_model"] = frame["model_name"].astype(str)
    frame["strategy"] = strategy_name
    if "fallback_used" not in frame.columns:
        frame["fallback_used"] = False
    else:
        frame["fallback_used"] = frame["fallback_used"].fillna(False).astype(bool)
    if "fallback_type" not in frame.columns:
        frame["fallback_type"] = "none"
    else:
        frame["fallback_type"] = frame["fallback_type"].fillna("none").astype(str)
    if "fallback_reason" not in frame.columns:
        frame["fallback_reason"] = "none"
    else:
        frame["fallback_reason"] = frame["fallback_reason"].fillna("none").astype(str)
    return frame


def _annotate_fallback(
    rows: Sequence[pd.Series],
    used: bool,
    fallback_type: str,
    reason: str,
) -> List[pd.Series]:
    """Return row copies with explicit Budgeted-DP fallback metadata."""
    annotated_rows: List[pd.Series] = []
    for row in rows:
        annotated = row.copy()
        annotated["fallback_used"] = bool(used)
        annotated["fallback_type"] = fallback_type
        annotated["fallback_reason"] = reason
        annotated_rows.append(annotated)
    return annotated_rows


def _normalized_date_key(value: object) -> object:
    """Return a stable date key for realized inventory lookup."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _planning_signal(row: pd.Series) -> float:
    """Return forecast-implied inventory target for a candidate row."""
    return float(forecast_to_inventory_target([float(row["forecast"])], [float(row.get("safety_stock", 0.0))])[0])


def _transition_burden(
    current_plan: float,
    previous_plan: Optional[float],
    max_plan_change_rate: float,
) -> Tuple[float, float]:
    """Return plan-change rate and absolute execution violation burden.

    Both terms use the same denominator convention as _expected_plan_burden in
    the pipeline scripts: plan_change_pct is normalized (plan_change_abs / scale)
    and execution_violation is the absolute overshoot of the capacity threshold.
    This keeps lambda_execution in _stage_cost operating on the same quantity
    as execution_adaptation_penalty_total in the evaluation layer.
    """
    if previous_plan is None:
        return 0.0, 0.0
    previous_value = float(previous_plan)
    plan_change_abs = abs(float(current_plan) - previous_value)
    scale = max(abs(previous_value), 1e-8)
    plan_change_pct = plan_change_abs / scale
    execution_capacity = abs(previous_value) * float(max_plan_change_rate)
    execution_violation = max(plan_change_abs - execution_capacity, 0.0)
    return float(plan_change_pct), float(execution_violation)


def _state_sort_key(state: SelectorState) -> Tuple[float, int, Tuple[str, ...]]:
    """Return deterministic state sort key."""
    rounded_cost = float(np.round(state.cost, 12))
    return rounded_cost, int(state.switches), state.path


def _min_state(left: Optional[SelectorState], right: SelectorState) -> SelectorState:
    """Return the lower-cost deterministic state."""
    if left is None:
        return right
    return min([left, right], key=_state_sort_key)
