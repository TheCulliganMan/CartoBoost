from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ._artifacts import (
    ArtifactPersistenceMixin,
    require_artifact_payload,
    versioned_artifact_payload,
)
from ._native import (
    geo_causal_design_summary,
    geo_causal_representation_report_value,
    geo_causal_spatial_placebos,
    geo_causal_spillover_diagnostics,
    geo_causal_synthetic_did_summary,
)

__all__ = [
    "GeoCausalPanel",
    "SyntheticDIDEstimator",
    "GeoLiftEstimator",
    "GeoExperimentDesigner",
    "InvariantRiskEncoder",
    "DomainAdversarialGeoEncoder",
    "CounterfactualRepresentationNet",
    "TreatmentEffectRepresentationHead",
    "SpatialPlaceboTester",
]


class GeoCausalPanel:
    """Geographic causal panel with unit, time, outcome, treatment, and spatial context."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] | Any,
        *,
        unit_col: str = "unit_id",
        time_col: str = "time",
        outcome_col: str = "outcome",
        treatment_col: str = "treatment",
        covariate_cols: Sequence[str] | None = None,
        latitude_col: str | None = "latitude",
        longitude_col: str | None = "longitude",
        region_col: str | None = "region_id",
        spatial_weights: Sequence[tuple[str, str, float]] | None = None,
    ) -> None:
        self.rows = _coerce_rows(
            rows,
            unit_col=unit_col,
            time_col=time_col,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            covariate_cols=covariate_cols,
            latitude_col=latitude_col,
            longitude_col=longitude_col,
            region_col=region_col,
        )
        self.spatial_weights = [(str(a), str(b), float(w)) for a, b, w in (spatial_weights or [])]


class InvariantRiskEncoder:
    """Native-backed representation supplement for geo-causal workflows."""

    causal_warning = (
        "Representation learning does not prove causal identification; use it only as a "
        "supplement to an identified design."
    )

    def fit_report(
        self,
        features: Any,
        outcomes: Any,
        regions: Any,
        *,
        heldout_region: str,
    ) -> dict[str, Any]:
        rows = _matrix(features, "features")
        y = [float(value) for value in _flatten(outcomes)]
        region_values = [str(value) for value in _flatten(regions)]
        return json.loads(
            geo_causal_representation_report_value(
                rows,
                y,
                region_values,
                str(heldout_region),
            )
        )

    def transform(
        self,
        features: Any,
        outcomes: Any,
        regions: Any,
        *,
        heldout_region: str,
    ) -> list[list[float]]:
        return self.fit_report(
            features,
            outcomes,
            regions,
            heldout_region=heldout_region,
        )["transformed_features"]


DomainAdversarialGeoEncoder = InvariantRiskEncoder
CounterfactualRepresentationNet = InvariantRiskEncoder
TreatmentEffectRepresentationHead = InvariantRiskEncoder


class SyntheticDIDEstimator(ArtifactPersistenceMixin):
    """Synthetic difference-in-differences estimator for geographic interventions."""

    def __init__(self, *, intervention_time: str, seed: int = 13) -> None:
        self.intervention_time = str(intervention_time)
        self.seed = int(seed)
        self._panel: GeoCausalPanel | None = None
        self._summary: dict[str, Any] | None = None

    def fit(
        self,
        panel: GeoCausalPanel | Iterable[Mapping[str, Any]] | Any,
    ) -> SyntheticDIDEstimator:
        self._panel = panel if isinstance(panel, GeoCausalPanel) else GeoCausalPanel(panel)
        self._summary = json.loads(
            geo_causal_synthetic_did_summary(
                self._panel.rows,
                self._panel.spatial_weights,
                self.intervention_time,
                self.seed,
                0,
            )
        )
        return self

    def estimate_effect(self) -> float:
        return float(self._require_summary()["effect"])

    def predict(self, _panel: Any | None = None) -> float:
        return self.estimate_effect()

    def score(self, _panel: Any | None = None) -> float:
        summary = self._require_summary()
        return float(abs(summary["effect"]))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"intervention_time": self.intervention_time, "seed": self.seed}

    def set_params(self, **params: Any) -> SyntheticDIDEstimator:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self.seed = int(self.seed)
        self.intervention_time = str(self.intervention_time)
        self._panel = None
        self._summary = None
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": "SyntheticDIDEstimator",
            "params": self.get_params(),
            "fitted": self._summary is not None,
            "summary": None if self._summary is None else dict(self._summary),
        }

    def save(self, path: str | Path) -> None:
        panel = self._require_panel()
        payload = versioned_artifact_payload(
            "SyntheticDIDEstimator",
            params=self.get_params(),
            rows=panel.rows,
            spatial_weights=panel.spatial_weights,
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SyntheticDIDEstimator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "SyntheticDIDEstimator")
        obj = cls(**dict(payload["params"]))
        panel = GeoCausalPanel(payload["rows"], spatial_weights=payload["spatial_weights"])
        return obj.fit(panel)

    def placebo_test(self, n: int = 100) -> list[float]:
        panel = self._require_panel()
        values = geo_causal_spatial_placebos(
            panel.rows,
            panel.spatial_weights,
            self.intervention_time,
            self.seed,
            int(n),
        )
        summary = self._require_summary()
        summary["placebo_estimates"] = list(values)
        return list(values)

    def summary(self) -> dict[str, Any]:
        return dict(self._require_summary())

    def plot(self, kind: str = "placebo", ax: Any | None = None) -> Any:
        summary = self._require_summary()
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install cartoboost[visualization] to use geo-causal plots") from exc
        ax = ax or plt.subplots()[1]
        if kind == "placebo":
            values = summary.get("placebo_estimates") or []
            ax.hist(values, bins=min(20, max(1, len(values))), color="#4c78a8", alpha=0.75)
            ax.axvline(summary["effect"], color="#d62728", linewidth=2, label="estimated effect")
            ax.set_xlabel("Placebo effect")
            ax.set_ylabel("Count")
            ax.legend()
        elif kind == "weights":
            weights = summary.get("unit_weights", {})
            ax.bar(list(weights), list(weights.values()), color="#59a14f")
            ax.set_xlabel("Control geo")
            ax.set_ylabel("Synthetic DID unit weight")
        else:
            raise ValueError("kind must be 'placebo' or 'weights'")
        return ax

    def _require_panel(self) -> GeoCausalPanel:
        if self._panel is None:
            raise ValueError("fit(panel) must be called first")
        return self._panel

    def _require_summary(self) -> dict[str, Any]:
        if self._summary is None:
            raise ValueError("fit(panel) must be called first")
        return self._summary


class GeoExperimentDesigner(ArtifactPersistenceMixin):
    """GeoLift-style helper for choosing balanced candidate test geos."""

    def __init__(self, *, intervention_time: str, seed: int = 13) -> None:
        self.intervention_time = str(intervention_time)
        self.seed = int(seed)

    def fit(
        self,
        panel: GeoCausalPanel | Iterable[Mapping[str, Any]] | Any,
    ) -> GeoExperimentDesigner:
        self.panel = panel if isinstance(panel, GeoCausalPanel) else GeoCausalPanel(panel)
        return self

    def summary(self, candidate_count: int = 1, placebo_n: int = 100) -> dict[str, Any]:
        panel = getattr(self, "panel", None)
        if panel is None:
            raise ValueError("fit(panel) must be called first")
        return json.loads(
            geo_causal_design_summary(
                panel.rows,
                panel.spatial_weights,
                self.intervention_time,
                self.seed,
                int(candidate_count),
                int(placebo_n),
            )
        )

    def predict(self, candidate_count: int = 1, placebo_n: int = 100) -> dict[str, Any]:
        return self.summary(candidate_count=candidate_count, placebo_n=placebo_n)

    def score(self, candidate_count: int = 1, placebo_n: int = 100) -> float:
        return float(
            self.summary(candidate_count=candidate_count, placebo_n=placebo_n)[
                "estimated_detectable_lift"
            ]
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"intervention_time": self.intervention_time, "seed": self.seed}

    def set_params(self, **params: Any) -> GeoExperimentDesigner:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self.seed = int(self.seed)
        self.intervention_time = str(self.intervention_time)
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": "GeoExperimentDesigner",
            "params": self.get_params(),
            "fitted": hasattr(self, "panel"),
        }

    def save(self, path: str | Path) -> None:
        panel = getattr(self, "panel", None)
        if panel is None:
            raise ValueError("fit(panel) must be called before save")
        payload = versioned_artifact_payload(
            "GeoExperimentDesigner",
            params=self.get_params(),
            rows=panel.rows,
            spatial_weights=panel.spatial_weights,
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GeoExperimentDesigner:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "GeoExperimentDesigner")
        obj = cls(**dict(payload["params"]))
        return obj.fit(GeoCausalPanel(payload["rows"], spatial_weights=payload["spatial_weights"]))

    def plot(self, candidate_count: int = 1, placebo_n: int = 100, ax: Any | None = None) -> Any:
        design = self.summary(candidate_count=candidate_count, placebo_n=placebo_n)
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install cartoboost[visualization] to use geo-causal plots") from exc
        ax = ax or plt.subplots()[1]
        values = design.get("placebo_estimates") or []
        ax.hist(values, bins=min(20, max(1, len(values))), color="#4c78a8", alpha=0.75)
        ax.axvline(design["estimated_detectable_lift"], color="#d62728", linewidth=2)
        ax.set_xlabel("Placebo effect")
        ax.set_ylabel("Count")
        return ax


GeoLiftEstimator = GeoExperimentDesigner


class SpatialPlaceboTester(ArtifactPersistenceMixin):
    """Deterministic spatial placebo runner for geographic panel experiments."""

    def __init__(self, *, intervention_time: str, seed: int = 13) -> None:
        self.intervention_time = str(intervention_time)
        self.seed = int(seed)

    def fit(
        self,
        panel: GeoCausalPanel | Iterable[Mapping[str, Any]] | Any,
    ) -> SpatialPlaceboTester:
        self.panel = panel if isinstance(panel, GeoCausalPanel) else GeoCausalPanel(panel)
        return self

    def placebo_test(self, n: int = 100) -> list[float]:
        panel = getattr(self, "panel", None)
        if panel is None:
            raise ValueError("fit(panel) must be called first")
        return list(
            geo_causal_spatial_placebos(
                panel.rows,
                panel.spatial_weights,
                self.intervention_time,
                self.seed,
                int(n),
            )
        )

    def summary(self) -> dict[str, Any]:
        panel = getattr(self, "panel", None)
        if panel is None:
            raise ValueError("fit(panel) must be called first")
        return json.loads(geo_causal_spillover_diagnostics(panel.rows, panel.spatial_weights))

    def predict(self) -> dict[str, Any]:
        return self.summary()

    def score(self) -> float:
        return float(len(self.summary().get("warnings", [])))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"intervention_time": self.intervention_time, "seed": self.seed}

    def set_params(self, **params: Any) -> SpatialPlaceboTester:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self.seed = int(self.seed)
        self.intervention_time = str(self.intervention_time)
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": "SpatialPlaceboTester",
            "params": self.get_params(),
            "fitted": hasattr(self, "panel"),
        }

    def save(self, path: str | Path) -> None:
        panel = getattr(self, "panel", None)
        if panel is None:
            raise ValueError("fit(panel) must be called before save")
        payload = versioned_artifact_payload(
            "SpatialPlaceboTester",
            params=self.get_params(),
            rows=panel.rows,
            spatial_weights=panel.spatial_weights,
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SpatialPlaceboTester:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "SpatialPlaceboTester")
        obj = cls(**dict(payload["params"]))
        return obj.fit(GeoCausalPanel(payload["rows"], spatial_weights=payload["spatial_weights"]))


def _coerce_rows(
    rows: Iterable[Mapping[str, Any]] | Any,
    *,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: Sequence[str] | None,
    latitude_col: str | None,
    longitude_col: str | None,
    region_col: str | None,
) -> list[tuple[str, str, float, bool, dict[str, float], float | None, float | None, str | None]]:
    to_dict = getattr(rows, "to_dict", None)
    if callable(to_dict):
        records = to_dict(orient="records")
    else:
        records = list(rows)
    if not records:
        raise ValueError("GeoCausalPanel requires at least one row")
    covariate_cols = list(covariate_cols or [])
    coerced = []
    for idx, row in enumerate(records):
        if isinstance(row, (list, tuple)) and len(row) == 8:
            unit_id, time_value, outcome, treatment, covariates, latitude, longitude, region_id = (
                row
            )
            coerced.append(
                (
                    str(unit_id),
                    str(time_value),
                    float(outcome),
                    bool(treatment),
                    _coerce_covariate_mapping(covariates),
                    _optional_float(latitude),
                    _optional_float(longitude),
                    None if region_id is None else str(region_id),
                )
            )
            continue
        for col in (unit_col, time_col, outcome_col, treatment_col):
            if col not in row:
                raise ValueError(f"row {idx} is missing required column {col!r}")
        covariates = {}
        for col in covariate_cols:
            value = row.get(col)
            if value is not None:
                covariates[str(col)] = float(value)
        latitude = _optional_float(row.get(latitude_col)) if latitude_col else None
        longitude = _optional_float(row.get(longitude_col)) if longitude_col else None
        region_id = (
            None if region_col is None or row.get(region_col) is None else str(row[region_col])
        )
        coerced.append(
            (
                str(row[unit_col]),
                str(row[time_col]),
                float(row[outcome_col]),
                bool(row[treatment_col]),
                covariates,
                latitude,
                longitude,
                region_id,
            )
        )
    return coerced


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _coerce_covariate_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("normalized geo-causal covariates must be a mapping")
    return {str(key): float(item) for key, item in value.items()}


def _matrix(values: Any, name: str) -> list[list[float]]:
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        values = to_numpy()
    rows = values.tolist() if hasattr(values, "tolist") else values
    out = []
    for row in rows:
        row_values = [float(value) for value in row]
        if not row_values or any(value != value for value in row_values):
            raise ValueError(f"{name} must contain finite rows")
        out.append(row_values)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    width = len(out[0])
    if any(len(row) != width for row in out):
        raise ValueError(f"{name} must have fixed width")
    return out


def _flatten(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return [values]
