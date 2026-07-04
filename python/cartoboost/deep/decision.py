from __future__ import annotations

from typing import Any

import numpy as np

from ..config import FallbackMode, Objective
from ._native import dumps, loads, require_native
from .choice import ChoiceSetTransformer
from .flow import flow_uncertainty_report


class ConstrainedDecisionOptimizer:
    def __init__(
        self,
        *,
        objective: Objective = Objective.EXPECTED_UTILITY,
        constraints: dict[str, float] | None = None,
        fallback: FallbackMode = FallbackMode.RAISE,
        risk_aversion: float = 0.0,
    ) -> None:
        self.objective = objective
        self.constraints = dict(constraints or {})
        self.fallback = fallback
        self.risk_aversion = float(risk_aversion)

    def select(
        self, candidate_frame: list[dict[str, Any]], predictions: Any | None = None
    ) -> list[dict[str, Any]]:
        select = require_native("deep_constrained_decision_select_value")
        candidates = _merge_predictions(candidate_frame, predictions)
        return loads(
            select(
                dumps(candidates),
                str(self.objective),
                dumps(self.constraints),
                str(self.fallback),
                self.risk_aversion,
            )
        )

    def flow_uncertainty_report(
        self, candidate_frame: list[dict[str, Any]], predictions: Any | None = None
    ) -> dict[str, Any]:
        candidates = _merge_predictions(candidate_frame, predictions)
        utility = np.asarray(
            [row.get("expected_utility", row.get("candidate_value", 0.0)) for row in candidates],
            dtype=float,
        )
        baseline = np.full_like(utility, float(np.mean(utility)))
        residual = utility - baseline
        hidden = np.column_stack(
            [
                np.asarray([row.get("candidate_value", 0.0) for row in candidates], dtype=float),
                np.arange(len(candidates), dtype=float),
            ]
        )
        return flow_uncertainty_report(
            residual,
            model_hidden_state=hidden,
            surface="ConstrainedDecisionOptimizer",
        )

    def choice_set_report(
        self, candidate_frame: list[dict[str, Any]], predictions: Any | None = None
    ) -> dict[str, Any]:
        candidates = _merge_predictions(candidate_frame, predictions)
        report = ChoiceSetTransformer(outside_option=True).score(candidates)
        report["surface"] = "ConstrainedDecisionOptimizer"
        return report

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "objective": self.objective,
            "constraints": dict(self.constraints),
            "fallback": self.fallback,
            "risk_aversion": self.risk_aversion,
        }


def _merge_predictions(
    candidate_frame: list[dict[str, Any]], predictions: Any | None
) -> list[dict[str, Any]]:
    if predictions is None:
        return [dict(row) for row in candidate_frame]
    if not isinstance(predictions, list):
        predictions = list(predictions)
    by_candidate = {
        str(row["candidate_id"]): row
        for row in predictions
        if isinstance(row, dict) and "candidate_id" in row
    }
    merged = []
    for row in candidate_frame:
        out = dict(row)
        pred = by_candidate.get(str(row.get("candidate_id")))
        if pred is not None:
            for key, value in pred.items():
                if key not in {"decision_id", "candidate_id", "candidate_value"}:
                    out[key] = value
        merged.append(out)
    return merged
