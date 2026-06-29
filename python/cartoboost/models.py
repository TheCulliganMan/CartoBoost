"""Unified model registry and geo-aware model orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._artifacts import (
    dump_model_artifact,
    load_model_artifact,
    require_artifact_payload,
    versioned_artifact_payload,
)
from .classifier import CartoBoostClassifier
from .metrics import residual_morans_i
from .ranker import CartoBoostRanker
from .regressor import CartoBoostRegressor

Factory = Callable[..., Any]


@dataclass(frozen=True)
class ModelMetadata:
    """Typed metadata for a registered model family."""

    name: str
    namespace: str
    task_types: tuple[str, ...]
    capabilities: tuple[str, ...] = ()
    stable: bool = True
    artifact_format: str = "cartoboost"
    optional_dependencies: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model metadata name must be non-empty")
        if not self.namespace.strip():
            raise ValueError("model metadata namespace must be non-empty")
        if not self.task_types:
            raise ValueError("model metadata task_types must be non-empty")
        object.__setattr__(self, "task_types", tuple(str(value) for value in self.task_types))
        object.__setattr__(self, "capabilities", tuple(str(value) for value in self.capabilities))
        object.__setattr__(
            self,
            "optional_dependencies",
            tuple(str(value) for value in self.optional_dependencies),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "task_types": list(self.task_types),
            "capabilities": list(self.capabilities),
            "stable": self.stable,
            "artifact_format": self.artifact_format,
            "optional_dependencies": list(self.optional_dependencies),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Constructor plus typed metadata for a model family."""

    metadata: ModelMetadata
    factory: Factory
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def namespace(self) -> str:
        return self.metadata.namespace

    def create(self, **overrides: Any) -> Any:
        return self.factory(**{**dict(self.params), **overrides})


class ModelRegistry:
    """Duplicate-safe registry for all stable CartoBoost model namespaces."""

    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: ModelSpec, *, override: bool = False) -> ModelSpec:
        key = _registry_key(spec.namespace, spec.name)
        if key in self._specs and not override:
            raise ValueError(f"model '{key}' is already registered")
        self._specs[key] = spec
        return spec

    def get(self, name: str, *, namespace: str | None = None) -> ModelSpec:
        if namespace is not None:
            key = _registry_key(namespace, name)
            try:
                return self._specs[key]
            except KeyError as exc:
                raise KeyError(f"model '{key}' is not registered") from exc
        matches = [spec for spec in self._specs.values() if spec.name == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"model '{name}' is not registered")
        raise KeyError(f"model '{name}' is ambiguous; pass namespace=")

    def create(self, name: str, *, namespace: str | None = None, **params: Any) -> Any:
        return self.get(name, namespace=namespace).create(**params)

    def names(self, *, namespace: str | None = None) -> tuple[str, ...]:
        return tuple(
            key if namespace is None else spec.name
            for key, spec in self._specs.items()
            if namespace is None or spec.namespace == namespace
        )

    def metadata(self, *, namespace: str | None = None) -> tuple[ModelMetadata, ...]:
        return tuple(
            spec.metadata
            for spec in self._specs.values()
            if namespace is None or spec.namespace == namespace
        )

    def specs(self, *, namespace: str | None = None) -> tuple[ModelSpec, ...]:
        return tuple(
            spec
            for spec in self._specs.values()
            if namespace is None or spec.namespace == namespace
        )

    @classmethod
    def defaults(cls) -> ModelRegistry:
        registry = cls()
        for spec in default_model_specs():
            registry.register(spec)
        return registry


@dataclass(frozen=True)
class GeoTaskContext:
    """Structure inspected by ``AutoGeoModel`` before choosing candidates."""

    task_type: str = "regression"
    has_coords: bool = False
    has_graph: bool = False
    has_panel_id: bool = False
    has_time_index: bool = False
    target_sparsity: float = 0.0
    leakage_constraints: tuple[str, ...] = ("spatial_block",)
    validation_strategy: str = "spatial_holdout"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.target_sparsity) <= 1.0:
            raise ValueError("target_sparsity must be between 0 and 1")
        object.__setattr__(
            self,
            "leakage_constraints",
            tuple(str(value) for value in self.leakage_constraints),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "has_coords": self.has_coords,
            "has_graph": self.has_graph,
            "has_panel_id": self.has_panel_id,
            "has_time_index": self.has_time_index,
            "target_sparsity": self.target_sparsity,
            "leakage_constraints": list(self.leakage_constraints),
            "validation_strategy": self.validation_strategy,
        }


class AutoGeoModel:
    """Leakage-aware selector over CartoBoost geo model families."""

    def __init__(
        self,
        *,
        registry: ModelRegistry | None = None,
        metric: str = "rmse",
        random_state: int = 13,
        conformal_alpha: float = 0.1,
        max_escalation_level: int = 3,
    ) -> None:
        self.registry = registry or ModelRegistry.defaults()
        self.metric = metric
        self.random_state = int(random_state)
        self.conformal_alpha = float(conformal_alpha)
        self.max_escalation_level = int(max_escalation_level)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "registry": self.registry,
            "metric": self.metric,
            "random_state": self.random_state,
            "conformal_alpha": self.conformal_alpha,
            "max_escalation_level": self.max_escalation_level,
        }

    def set_params(self, **params: Any) -> AutoGeoModel:
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def candidate_families(self, context: GeoTaskContext) -> list[str]:
        _reject_leaky_spatial_validation(context)
        families = ["models.mean_baseline", "models.cartoboost_regressor"]
        if context.has_coords and self.max_escalation_level >= 1:
            families.append("geo.residual_nngp")
        if context.has_graph and self.max_escalation_level >= 2:
            families.append("graph.graph_residual")
        if context.has_time_index and context.has_panel_id and self.max_escalation_level >= 2:
            families.append("forecasting.cartoboost_lag")
        if "intervals" in context.leakage_constraints and self.max_escalation_level >= 3:
            families.append("prob.conformal_interval")
        return families

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        coords: Any | None = None,
        graph: Any | None = None,
        panel_id: Any | None = None,
        time_index: Any | None = None,
        validation: Mapping[str, Sequence[int]] | None = None,
        leakage_constraints: Sequence[str] = ("spatial_block",),
        validation_strategy: str = "spatial_holdout",
    ) -> AutoGeoModel:
        x_array = _as_2d_float_array(X, "X")
        y_array = _as_1d_float_array(y, "y")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        context = GeoTaskContext(
            task_type="regression",
            has_coords=coords is not None,
            has_graph=graph is not None,
            has_panel_id=panel_id is not None,
            has_time_index=time_index is not None,
            target_sparsity=float(np.mean(y_array == 0.0)),
            leakage_constraints=tuple(leakage_constraints),
            validation_strategy=validation_strategy,
        )
        _reject_leaky_spatial_validation(context)
        train_idx, holdout_idx = _resolve_holdout(
            len(y_array),
            validation=validation,
            validation_strategy=validation_strategy,
        )
        coords_array = None if coords is None else _as_2d_float_array(coords, "coords")
        if coords_array is not None and coords_array.shape[0] != y_array.shape[0]:
            raise ValueError("coords must contain the same number of rows as X")

        evaluations: list[dict[str, Any]] = []
        fitted: dict[str, Any] = {}
        for family in self.candidate_families(context):
            fit_result = _fit_autogeo_family(
                family,
                x_array,
                y_array,
                coords_array,
                train_idx,
                holdout_idx,
                random_state=self.random_state,
            )
            if fit_result is None:
                evaluations.append(
                    {
                        "family": family,
                        "status": "skipped",
                        "reason": "requires a specialized frame supplied through benchmark harness",
                    }
                )
                continue
            model, pred = fit_result
            residuals = y_array[holdout_idx] - np.asarray(pred, dtype=float)
            diagnostics = _diagnostics(y_array[holdout_idx], pred)
            if coords_array is not None and residuals.shape[0] > 2 and np.std(residuals) > 0.0:
                diagnostics["residual_morans_i"] = float(
                    residual_morans_i(coords_array[holdout_idx], residuals)
                )
            evaluations.append(
                {
                    "family": family,
                    "status": "fit",
                    "artifact_supported": _artifact_supported(model),
                    "metric": diagnostics[self.metric],
                    "diagnostics": diagnostics,
                }
            )
            fitted[family] = model

        scored = [
            row
            for row in evaluations
            if row["status"] == "fit" and row.get("artifact_supported") is True
        ]
        if not scored:
            raise RuntimeError("AutoGeoModel found no serializable fit candidates")
        best = min(
            scored,
            key=lambda row: (
                row["metric"],
                abs(row["diagnostics"].get("residual_morans_i", 0.0)),
            ),
        )
        self.context_ = context
        self.evaluations_ = evaluations
        self.selected_family_ = str(best["family"])
        final_idx = np.unique(np.concatenate([train_idx, holdout_idx]))
        final_fit = _fit_autogeo_family(
            self.selected_family_,
            x_array,
            y_array,
            coords_array,
            final_idx,
            holdout_idx,
            random_state=self.random_state,
        )
        self.selected_model_ = fitted[self.selected_family_] if final_fit is None else final_fit[0]
        self.metadata_ = {
            "model": "AutoGeoModel",
            "context": context.to_dict(),
            "selected_family": self.selected_family_,
            "selection_metric": self.metric,
            "evaluations": evaluations,
            "leakage_safe": True,
        }
        return self

    def predict(self, X: Any, *, coords: Any | None = None, **kwargs: Any) -> np.ndarray:
        model = self._require_model()
        if self.selected_family_ == "geo.residual_nngp":
            if coords is None:
                raise ValueError("coords are required for the selected geo residual model")
            return np.asarray(model.predict(X, coords=coords, **kwargs), dtype=float)
        return np.asarray(model.predict(X), dtype=float)

    def score(self, X: Any, y: Any, *, coords: Any | None = None) -> float:
        pred = self.predict(X, coords=coords)
        return _diagnostics(y, pred)[self.metric]

    def save(self, path: str | Path) -> None:
        if not hasattr(self, "selected_model_"):
            raise ValueError("AutoGeoModel is not fitted")
        payload = versioned_artifact_payload(
            "AutoGeoModel",
            params={
                "metric": self.metric,
                "random_state": self.random_state,
                "conformal_alpha": self.conformal_alpha,
                "max_escalation_level": self.max_escalation_level,
            },
            metadata=self.metadata_,
            selected_family=self.selected_family_,
            selected_model=_dump_supported_model(self.selected_model_),
        )
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AutoGeoModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "AutoGeoModel")
        obj = cls(**payload["params"])
        obj.metadata_ = dict(payload["metadata"])
        obj.context_ = GeoTaskContext(**obj.metadata_["context"])
        obj.evaluations_ = list(obj.metadata_["evaluations"])
        obj.selected_family_ = str(payload["selected_family"])
        obj.selected_model_ = _load_supported_model(payload["selected_model"])
        return obj

    def _require_model(self) -> Any:
        if not hasattr(self, "selected_model_"):
            raise ValueError("AutoGeoModel is not fitted")
        return self.selected_model_


class GeoModelStack:
    """Layered tabular, spatial residual, graph residual, and interval stack."""

    def __init__(
        self,
        *,
        base_model: Any | None = None,
        spatial_residual_model: Any | None = None,
        graph_residual_model: Any | None = None,
        interval_model: Any | None = None,
    ) -> None:
        self.base_model = base_model or CartoBoostRegressor()
        self.spatial_residual_model = spatial_residual_model
        self.graph_residual_model = graph_residual_model
        self.interval_model = interval_model

    def fit(
        self, X: Any, y: Any, *, coords: Any | None = None, graph: Any | None = None
    ) -> GeoModelStack:
        x_array = _as_2d_float_array(X, "X")
        y_array = _as_1d_float_array(y, "y")
        self.base_model.fit(x_array, y_array)
        base_pred = np.asarray(self.base_model.predict(x_array), dtype=float)
        self.layers_ = [
            _layer_summary("tabular_booster", y_array, base_pred, "fit"),
        ]
        running_pred = base_pred.copy()
        if self.spatial_residual_model is not None:
            if coords is None:
                raise ValueError("coords are required when spatial_residual_model is configured")
            residual = y_array - running_pred
            self.spatial_residual_model.fit(None, residual, coords=coords)
            spatial_delta = np.asarray(
                self.spatial_residual_model.predict(None, coords=coords),
                dtype=float,
            )
            running_pred = running_pred + spatial_delta
            self.layers_.append(_layer_summary("residual_spatial", y_array, running_pred, "fit"))
        if self.graph_residual_model is not None:
            if graph is None:
                raise ValueError("graph is required when graph_residual_model is configured")
            self.layers_.append(
                {
                    "layer": "graph_residual",
                    "status": "configured",
                    "value_added": None,
                    "note": "graph residual model is delegated to the supplied graph estimator",
                }
            )
        if self.interval_model is not None:
            self.interval_model.fit(running_pred, y_array)
            self.layers_.append(
                {
                    "layer": "conformal_interval",
                    "status": "fit",
                    "value_added": None,
                    "note": "interval layer adds calibration evidence rather than point accuracy",
                }
            )
        self.metadata_ = {"model": "GeoModelStack", "layers": self.layers_}
        return self

    def predict(self, X: Any, *, coords: Any | None = None, graph: Any | None = None) -> np.ndarray:
        del graph
        pred = np.asarray(self.base_model.predict(X), dtype=float)
        if self.spatial_residual_model is not None:
            if coords is None:
                raise ValueError("coords are required for spatial residual predictions")
            pred = pred + np.asarray(
                self.spatial_residual_model.predict(None, coords=coords),
                dtype=float,
            )
        return pred

    def score(self, X: Any, y: Any, *, coords: Any | None = None) -> float:
        return _diagnostics(y, self.predict(X, coords=coords))["rmse"]

    def explain_layers(self) -> dict[str, Any]:
        if not hasattr(self, "metadata_"):
            raise ValueError("GeoModelStack is not fitted")
        return dict(self.metadata_)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "base_model": self.base_model,
            "spatial_residual_model": self.spatial_residual_model,
            "graph_residual_model": self.graph_residual_model,
            "interval_model": self.interval_model,
        }

    def set_params(self, **params: Any) -> GeoModelStack:
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def save(self, path: str | Path) -> None:
        if not hasattr(self, "metadata_"):
            raise ValueError("GeoModelStack is not fitted")
        payload = versioned_artifact_payload(
            "GeoModelStack",
            metadata=self.metadata_,
            base_model=_dump_supported_model(self.base_model),
            spatial_residual_model=None
            if self.spatial_residual_model is None
            else _dump_supported_model(self.spatial_residual_model),
            has_graph_residual_model=self.graph_residual_model is not None,
            interval_model=None
            if self.interval_model is None
            else _dump_supported_model(self.interval_model),
        )
        if self.graph_residual_model is not None:
            raise ValueError("GeoModelStack save currently requires a serializable graph layer")
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GeoModelStack:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        require_artifact_payload(payload, "GeoModelStack")
        obj = cls(
            base_model=_load_supported_model(payload["base_model"]),
            spatial_residual_model=None
            if payload["spatial_residual_model"] is None
            else _load_supported_model(payload["spatial_residual_model"]),
            interval_model=None
            if payload["interval_model"] is None
            else _load_supported_model(payload["interval_model"]),
        )
        obj.metadata_ = dict(payload["metadata"])
        obj.layers_ = list(obj.metadata_["layers"])
        return obj


def default_model_specs() -> tuple[ModelSpec, ...]:
    from .forecasting import AutoForecaster, CartoBoostLagForecaster
    from .forecasting.graph_st import DCRNNForecaster
    from .forecasting.probabilistic import ConformalIntervalRegressor, SpatialConformalRegressor
    from .geo_causal import GeoExperimentDesigner, SpatialPlaceboTester, SyntheticDIDEstimator
    from .geostats import NearestNeighborGPRegressor, ResidualNNGPRegressor
    from .spatial_econometrics import (
        SpatialDurbinRegressor,
        SpatialErrorRegressor,
        SpatialLagRegressor,
    )

    return (
        _spec(
            "cartoboost_regressor",
            "models",
            CartoBoostRegressor,
            ("regression",),
            ("fit", "predict", "score", "save", "load"),
        ),
        _spec(
            "cartoboost_classifier",
            "models",
            CartoBoostClassifier,
            ("classification",),
            ("fit", "predict", "score", "save", "load"),
        ),
        _spec(
            "cartoboost_ranker",
            "models",
            CartoBoostRanker,
            ("ranking",),
            ("fit", "predict", "save", "load"),
        ),
        _spec(
            "auto_geo_model",
            "models",
            AutoGeoModel,
            ("regression",),
            ("selector", "leakage_safe", "diagnostics", "save", "load"),
        ),
        _spec(
            "geo_model_stack",
            "models",
            GeoModelStack,
            ("regression",),
            ("stacking", "diagnostics", "save", "load"),
        ),
        _spec(
            "auto_forecaster", "forecasting", AutoForecaster, ("forecasting",), ("fit", "predict")
        ),
        _spec(
            "cartoboost_lag",
            "forecasting",
            CartoBoostLagForecaster,
            ("forecasting",),
            ("fit", "predict"),
        ),
        _spec(
            "dcrnn",
            "graph",
            DCRNNForecaster,
            ("forecasting",),
            ("graph", "spatiotemporal", "save", "load"),
        ),
        _spec("nngp", "geo", NearestNeighborGPRegressor, ("regression",), ("coords", "intervals")),
        _spec(
            "residual_nngp",
            "geo",
            ResidualNNGPRegressor,
            ("regression",),
            ("coords", "residual_model"),
        ),
        _spec(
            "spatial_lag",
            "geo",
            SpatialLagRegressor,
            ("regression",),
            ("spatial_econometrics", "save", "load"),
        ),
        _spec(
            "spatial_error",
            "geo",
            SpatialErrorRegressor,
            ("regression",),
            ("spatial_econometrics", "save", "load"),
        ),
        _spec(
            "spatial_durbin",
            "geo",
            SpatialDurbinRegressor,
            ("regression",),
            ("spatial_econometrics", "save", "load"),
        ),
        _spec(
            "synthetic_did",
            "causal",
            SyntheticDIDEstimator,
            ("causal_panel",),
            ("fit", "placebo", "summary"),
        ),
        _spec(
            "geo_lift_design",
            "causal",
            GeoExperimentDesigner,
            ("causal_panel",),
            ("design", "placebo", "summary"),
        ),
        _spec(
            "spatial_placebo",
            "causal",
            SpatialPlaceboTester,
            ("causal_panel",),
            ("placebo", "summary"),
        ),
        _spec(
            "conformal_interval",
            "prob",
            ConformalIntervalRegressor,
            ("regression",),
            ("intervals", "coverage", "width"),
        ),
        _spec(
            "spatial_conformal",
            "prob",
            SpatialConformalRegressor,
            ("regression",),
            ("intervals", "regions", "coverage", "width"),
        ),
    )


def model_card(model: Any) -> dict[str, Any]:
    metadata = getattr(model, "metadata_", None)
    if metadata is None and hasattr(model, "get_metadata"):
        metadata = model.get_metadata()
    params = model.get_params() if hasattr(model, "get_params") else {}
    return {
        "class": model.__class__.__name__,
        "params": _jsonable(params),
        "metadata": _jsonable(metadata or {}),
        "lifecycle": {
            "fit": hasattr(model, "fit"),
            "predict": hasattr(model, "predict"),
            "score": hasattr(model, "score"),
            "save": hasattr(model, "save"),
            "load": hasattr(model.__class__, "load"),
            "get_params": hasattr(model, "get_params"),
            "set_params": hasattr(model, "set_params"),
        },
    }


def save_model_card(model: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(model_card(model), indent=2, sort_keys=True), encoding="utf-8")


class _MeanRegressor:
    def fit(self, X: Any, y: Any) -> _MeanRegressor:
        del X
        values = _as_1d_float_array(y, "y")
        self.mean_ = float(np.mean(values))
        return self

    def predict(self, X: Any) -> np.ndarray:
        rows = _as_2d_float_array(X, "X").shape[0]
        return np.full(rows, self.mean_, dtype=float)

    def score(self, X: Any, y: Any) -> float:
        return _diagnostics(y, self.predict(X))["rmse"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {}

    def set_params(self, **params: Any) -> _MeanRegressor:
        if params:
            raise ValueError(f"unknown parameters: {sorted(params)}")
        return self


def _spec(
    name: str,
    namespace: str,
    factory: Factory,
    task_types: tuple[str, ...],
    capabilities: tuple[str, ...],
) -> ModelSpec:
    return ModelSpec(
        metadata=ModelMetadata(
            name=name,
            namespace=namespace,
            task_types=task_types,
            capabilities=capabilities,
        ),
        factory=factory,
    )


def _registry_key(namespace: str, name: str) -> str:
    return f"{namespace.strip()}.{name.strip()}"


def _dump_supported_model(model: Any) -> dict[str, Any]:
    if isinstance(model, _MeanRegressor):
        return {"class": "_MeanRegressor", "mean": model.mean_}
    return dump_model_artifact(model, purpose="AutoGeoModel artifacts")


def _artifact_supported(model: Any) -> bool:
    return isinstance(model, _MeanRegressor) or callable(getattr(model, "save", None))


def _load_supported_model(payload: Mapping[str, Any]) -> Any:
    model_class = payload.get("class")
    if model_class == "_MeanRegressor":
        model = _MeanRegressor()
        model.mean_ = float(payload["mean"])
        return model
    return load_model_artifact(payload)


def _reject_leaky_spatial_validation(context: GeoTaskContext) -> None:
    if context.has_coords and context.validation_strategy in {
        "random",
        "random_row",
        "random_row_cv",
    }:
        raise ValueError("spatial model selection cannot use random row CV")


def _resolve_holdout(
    n_rows: int,
    *,
    validation: Mapping[str, Sequence[int]] | None,
    validation_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 4:
        raise ValueError("AutoGeoModel requires at least four rows")
    if validation is not None:
        train = np.asarray(validation.get("train", ()), dtype=np.int64)
        holdout = np.asarray(
            validation.get("holdout", validation.get("validation", ())), dtype=np.int64
        )
        if train.size == 0 or holdout.size == 0:
            raise ValueError("validation must include non-empty train and holdout indices")
    else:
        if validation_strategy in {"random", "random_row", "random_row_cv"}:
            raise ValueError("random row validation is not accepted")
        split = max(1, int(round(n_rows * 0.8)))
        split = min(split, n_rows - 1)
        train = np.arange(0, split, dtype=np.int64)
        holdout = np.arange(split, n_rows, dtype=np.int64)
    if np.intersect1d(train, holdout).size:
        raise ValueError("train and holdout indices must not overlap")
    if train.min(initial=0) < 0 or holdout.min(initial=0) < 0:
        raise ValueError("validation indices must be non-negative")
    if train.max(initial=0) >= n_rows or holdout.max(initial=0) >= n_rows:
        raise ValueError("validation indices exceed row count")
    return train, holdout


def _diagnostics(y_true: Any, y_pred: Any) -> dict[str, float]:
    truth = _as_1d_float_array(y_true, "y_true")
    pred = _as_1d_float_array(y_pred, "y_pred")
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    residual = truth - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum(np.abs(truth)))
    return {
        "mae": mae,
        "rmse": rmse,
        "wape": float(np.sum(np.abs(residual)) / denom) if denom > 0.0 else float("nan"),
        "bias": float(np.mean(residual)),
    }


def _autogeo_booster(random_state: int) -> CartoBoostRegressor:
    return CartoBoostRegressor(
        n_estimators=96,
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=3,
        min_gain=0.0,
        splitters=["axis", "diagonal_2d", "gaussian_2d"],
        random_state=random_state,
    )


def _fit_autogeo_family(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray | None,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    *,
    random_state: int,
) -> tuple[Any, np.ndarray] | None:
    if family == "models.mean_baseline":
        model = _MeanRegressor().fit(X[train_idx], y[train_idx])
        return model, model.predict(X[predict_idx])
    if family == "models.cartoboost_regressor":
        model = _autogeo_booster(random_state)
        model.fit(X[train_idx], y[train_idx])
        return model, np.asarray(model.predict(X[predict_idx]), dtype=float)
    if family == "geo.residual_nngp":
        if coords is None:
            return None
        from .geostats import ResidualNNGPRegressor

        model = ResidualNNGPRegressor(_autogeo_booster(random_state))
        model.fit(X[train_idx], y[train_idx], coords=coords[train_idx])
        return model, np.asarray(
            model.predict(X[predict_idx], coords=coords[predict_idx]), dtype=float
        )
    return None


def _layer_summary(
    layer: str, y_true: np.ndarray, y_pred: np.ndarray, status: str
) -> dict[str, Any]:
    diagnostics = _diagnostics(y_true, y_pred)
    return {
        "layer": layer,
        "status": status,
        "diagnostics": diagnostics,
        "value_added": None,
    }


def _as_2d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return repr(value)


__all__ = [
    "AutoGeoModel",
    "GeoModelStack",
    "GeoTaskContext",
    "ModelMetadata",
    "ModelRegistry",
    "ModelSpec",
    "default_model_specs",
    "model_card",
    "save_model_card",
]
