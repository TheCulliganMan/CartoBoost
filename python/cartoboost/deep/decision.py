from __future__ import annotations

from typing import Any

from ..config import FallbackMode, Objective
from ._native import dumps, loads, require_native


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
        del predictions
        select = require_native("deep_constrained_decision_select_value")
        return loads(
            select(
                dumps(candidate_frame),
                str(self.objective),
                dumps(self.constraints),
                str(self.fallback),
            )
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "objective": self.objective,
            "constraints": dict(self.constraints),
            "fallback": self.fallback,
            "risk_aversion": self.risk_aversion,
        }
