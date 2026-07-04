from __future__ import annotations

from typing import Any

from ._native import dumps, loads, require_native


class ChoiceSetTransformer:
    """Native-backed candidate competition scorer."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        monotone_candidate_value: str | None = None,
        outside_option: bool = False,
        outside_utility: float = 0.0,
    ) -> None:
        self.temperature = float(temperature)
        self.monotone_candidate_value = monotone_candidate_value
        self.outside_option = bool(outside_option)
        self.outside_utility = float(outside_utility)

    def score(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        report = require_native("deep_choice_set_transformer_report_value")
        rows = self._with_outside_options(candidates)
        return loads(
            report(
                dumps(rows),
                self.temperature,
                self.monotone_candidate_value,
            )
        )

    def predict_proba(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.score(candidates)["predictions"]

    def counterfactual_best(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.score(candidates)["counterfactual_best"]

    def calibration_report(self, candidates: list[dict[str, Any]]) -> dict[str, float]:
        return self.score(candidates)["calibration"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "temperature": self.temperature,
            "monotone_candidate_value": self.monotone_candidate_value,
            "outside_option": self.outside_option,
            "outside_utility": self.outside_utility,
        }

    def _with_outside_options(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = [dict(row) for row in candidates]
        if not self.outside_option:
            return rows
        decisions = sorted({str(row.get("decision_id", "")) for row in rows})
        for decision_id in decisions:
            rows.append(
                {
                    "decision_id": decision_id,
                    "candidate_id": "__outside__",
                    "candidate_value": 0.0,
                    "expected_utility": self.outside_utility,
                    "response_probability": 0.0,
                    "candidate_features": [],
                    "context_features": [],
                    "entity_or_pair_embeddings": [],
                    "nest_id": "__outside__",
                    "outside_option": True,
                    "chosen": False,
                }
            )
        return rows


UtilityNet = ChoiceSetTransformer
NestedChoiceHead = ChoiceSetTransformer
CounterfactualCandidateScorer = ChoiceSetTransformer
