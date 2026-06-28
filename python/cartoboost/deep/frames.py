from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _records_from_pandas(df: Any) -> list[dict[str, Any]]:
    if not hasattr(df, "to_dict"):
        raise TypeError("expected a pandas-like DataFrame")
    return list(df.to_dict(orient="records"))


@dataclass(frozen=True)
class EntityPanelFrame:
    y: np.ndarray
    timestamps: list[Any]
    entity_ids: list[str]
    historical_x: np.ndarray | None = None
    known_future_x: np.ndarray | None = None
    static_x: np.ndarray | None = None
    freq: str | None = None

    @classmethod
    def from_pandas(
        cls,
        df: Any,
        *,
        timestamp_col: str,
        entity_col: str,
        target_col: str,
        freq: str | None = None,
        historical_covariates: list[str] | None = None,
        known_future_covariates: list[str] | None = None,
        static_covariates: list[str] | None = None,
    ) -> EntityPanelFrame:
        rows = _records_from_pandas(df)
        timestamps = sorted({row[timestamp_col] for row in rows})
        entities = sorted({str(row[entity_col]) for row in rows})
        t_pos = {value: idx for idx, value in enumerate(timestamps)}
        e_pos = {value: idx for idx, value in enumerate(entities)}
        y = np.full((len(timestamps), len(entities)), np.nan, dtype=float)
        for row in rows:
            y[t_pos[row[timestamp_col]], e_pos[str(row[entity_col])]] = float(row[target_col])
        if np.isnan(y).any():
            raise ValueError("entity panel contains missing timestamp/entity target cells")
        hist = _panel_covariates(
            rows, timestamps, entities, timestamp_col, entity_col, historical_covariates
        )
        future = _panel_covariates(
            rows, timestamps, entities, timestamp_col, entity_col, known_future_covariates
        )
        static = None
        if static_covariates:
            static = np.zeros((len(entities), len(static_covariates)), dtype=float)
            first = {str(row[entity_col]): row for row in rows}
            for entity, idx in e_pos.items():
                static[idx] = [float(first[entity][col]) for col in static_covariates]
        return cls(y, timestamps, entities, hist, future, static, freq)


@dataclass(frozen=True)
class DirectionalPairFrame:
    rows: list[dict[str, Any]]

    @classmethod
    def from_pandas(
        cls,
        df: Any,
        *,
        timestamp_col: str | None = None,
        source_col: str,
        target_col: str,
        target_value_col: str | None = None,
        numeric_covariates: list[str] | None = None,
        known_future_covariates: list[str] | None = None,
    ) -> DirectionalPairFrame:
        feature_cols = [*(numeric_covariates or []), *(known_future_covariates or [])]
        rows = []
        for row in _records_from_pandas(df):
            rows.append(
                {
                    "source_id": str(row[source_col]),
                    "target_id": str(row[target_col]),
                    "timestamp": None if timestamp_col is None else int(row[timestamp_col]),
                    "features": [float(row[col]) for col in feature_cols],
                    "target": None if target_value_col is None else float(row[target_value_col]),
                }
            )
        return cls(rows)


@dataclass(frozen=True)
class GraphTemporalFrame:
    y: np.ndarray
    timestamps: list[Any]
    node_ids: list[str]
    edges: list[tuple[int, int]]
    edge_weights: list[float]
    node_covariates: np.ndarray | None = None
    static_node_covariates: np.ndarray | None = None
    directed: bool = True


@dataclass(frozen=True)
class ResponseCurveFrame:
    rows: list[dict[str, Any]]

    @classmethod
    def from_pandas(
        cls,
        df: Any,
        *,
        feature_cols: list[str],
        candidate_value_col: str,
        response_col: str | None = None,
        group_col: str | None = None,
        candidate_id_col: str | None = None,
    ) -> ResponseCurveFrame:
        rows = []
        for idx, row in enumerate(_records_from_pandas(df)):
            rows.append(
                {
                    "features": [float(row[col]) for col in feature_cols],
                    "candidate_value": float(row[candidate_value_col]),
                    "response": None if response_col is None else float(row[response_col]),
                    "group_id": None if group_col is None else str(row[group_col]),
                    "candidate_id": str(row[candidate_id_col]) if candidate_id_col else str(idx),
                }
            )
        return cls(rows)


def _panel_covariates(
    rows: list[dict[str, Any]],
    timestamps: list[Any],
    entities: list[str],
    timestamp_col: str,
    entity_col: str,
    cols: list[str] | None,
) -> np.ndarray | None:
    if not cols:
        return None
    t_pos = {value: idx for idx, value in enumerate(timestamps)}
    e_pos = {value: idx for idx, value in enumerate(entities)}
    values = np.zeros((len(timestamps), len(entities), len(cols)), dtype=float)
    for row in rows:
        values[t_pos[row[timestamp_col]], e_pos[str(row[entity_col])]] = [
            float(row[col]) for col in cols
        ]
    return values
