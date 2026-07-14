"""Rust-backed model registry and stable model metadata."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

Factory = Callable[..., Any]
MODEL_TIERS = frozenset({"stable", "supported", "experimental"})
MODEL_BACKENDS = frozenset(
    {"rust_native", "rust_native_with_python_facade", "python_orchestration"}
)
MODEL_EVIDENCE_LEVELS = frozenset({"real_data", "synthetic", "api_only", "experimental_only"})
MODEL_ARTIFACT_VERSION = 2
STABLE_MODEL_KEYS = frozenset(
    {
        "models.cartoboost_regressor",
        "models.cartoboost_classifier",
        "models.cartoboost_ranker",
        "forecasting.auto_forecaster",
        "forecasting.cartoboost_lag",
    }
)
EXPERIMENTAL_MODEL_KEYS = frozenset({"graph.dcrnn"})


@dataclass(frozen=True)
class ModelMetadata:
    """Typed metadata for a registered model family."""

    name: str
    namespace: str
    task_types: tuple[str, ...]
    capabilities: tuple[str, ...] = ()
    stable: bool | None = None
    tier: str = "stable"
    artifact_format: str = "cartoboost"
    artifact_version: int = MODEL_ARTIFACT_VERSION
    backend: str = "rust_native"
    evidence_level: str = "real_data"
    optional_dependencies: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.namespace.strip():
            raise ValueError("model metadata name and namespace must be non-empty")
        if not self.task_types:
            raise ValueError("model metadata task_types must be non-empty")
        if self.tier not in MODEL_TIERS:
            raise ValueError(f"model metadata tier must be one of {sorted(MODEL_TIERS)}")
        stable = self.tier == "stable" if self.stable is None else bool(self.stable)
        if stable != (self.tier == "stable"):
            raise ValueError("model metadata stable flag must agree with tier")
        if self.artifact_version < 1:
            raise ValueError("model metadata artifact_version must be positive")
        if self.backend not in MODEL_BACKENDS:
            raise ValueError(f"model metadata backend must be one of {sorted(MODEL_BACKENDS)}")
        if self.evidence_level not in MODEL_EVIDENCE_LEVELS:
            raise ValueError(
                f"model metadata evidence_level must be one of {sorted(MODEL_EVIDENCE_LEVELS)}"
            )
        object.__setattr__(self, "stable", stable)
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
            "tier": self.tier,
            "artifact_format": self.artifact_format,
            "artifact_version": self.artifact_version,
            "backend": self.backend,
            "evidence_level": self.evidence_level,
            "optional_dependencies": list(self.optional_dependencies),
            "dependencies": list(self.optional_dependencies),
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

    @property
    def key(self) -> str:
        return _registry_key(self.namespace, self.name)

    def create(self, **overrides: Any) -> Any:
        return self.factory(**{**dict(self.params), **overrides})


class ModelRegistry:
    """Duplicate-safe registry for stable, supported, and experimental models."""

    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: ModelSpec, *, override: bool = False) -> ModelSpec:
        key = _registry_key(spec.namespace, spec.name)
        if key in self._specs and not override:
            raise ValueError(f"model {key!r} is already registered")
        self._specs[key] = spec
        return spec

    def get(self, name: str, *, namespace: str | None = None) -> ModelSpec:
        if namespace is not None:
            key = _registry_key(namespace, name)
            try:
                return self._specs[key]
            except KeyError as exc:
                raise KeyError(f"model {key!r} is not registered") from exc
        matches = [spec for spec in self._specs.values() if spec.name == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"model {name!r} is not registered")
        raise KeyError(f"model {name!r} is ambiguous; pass namespace=")

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

    def manifest(self, *, tier: str | None = None) -> tuple[dict[str, Any], ...]:
        if tier is not None and tier not in MODEL_TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        rows = []
        for spec in self._specs.values():
            if tier is not None and spec.metadata.tier != tier:
                continue
            row = spec.metadata.to_dict()
            row["key"] = spec.key
            row["factory"] = getattr(spec.factory, "__name__", repr(spec.factory))
            rows.append(row)
        return tuple(rows)

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
        canonical = {str(row.get("key")): row for row in native_model_manifest() if row.get("key")}
        if set(canonical) != set(registry.names()):
            raise RuntimeError(
                "Rust model manifest and Python registry disagree on model keys: "
                f"rust_only={sorted(set(canonical) - set(registry.names()))}, "
                f"python_only={sorted(set(registry.names()) - set(canonical))}"
            )
        for spec in registry.specs():
            row = canonical[spec.key]
            metadata = spec.metadata
            expected = {
                "tier": metadata.tier,
                "backend": metadata.backend,
                "task": metadata.task_types[0],
                "artifact_version": metadata.artifact_version,
                "dependencies": list(metadata.optional_dependencies),
                "evidence_level": metadata.evidence_level,
            }
            actual = {key: row.get(key) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    f"Rust model manifest metadata mismatch for {spec.key}: "
                    f"expected={expected}, actual={actual}"
                )
        return registry

    @classmethod
    def stable_defaults(cls) -> ModelRegistry:
        return cls(spec for spec in cls.defaults().specs() if spec.metadata.tier == "stable")

    def by_tier(self, tier: str) -> ModelRegistry:
        if tier not in MODEL_TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        return ModelRegistry(spec for spec in self._specs.values() if spec.metadata.tier == tier)


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
            _load("regressor", "CartoBoostRegressor"),
            ("regression",),
            ("fit", "predict", "score", "save", "load"),
        ),
        _spec(
            "cartoboost_classifier",
            "models",
            _load("classifier", "CartoBoostClassifier"),
            ("classification",),
            ("fit", "predict", "score", "save", "load"),
        ),
        _spec(
            "cartoboost_ranker",
            "models",
            _load("ranker", "CartoBoostRanker"),
            ("ranking",),
            ("fit", "predict", "save", "load"),
        ),
        _spec(
            "auto_forecaster",
            "forecasting",
            AutoForecaster,
            ("forecasting",),
            ("fit", "predict", "score", "save", "load", "get_params", "set_params"),
        ),
        _spec(
            "cartoboost_lag",
            "forecasting",
            CartoBoostLagForecaster,
            ("forecasting",),
            ("fit", "predict", "score", "save", "load", "get_params", "set_params"),
        ),
        _spec(
            "dcrnn",
            "graph",
            DCRNNForecaster,
            ("forecasting",),
            ("graph", "spatiotemporal", "save", "load"),
            optional_dependencies=("torch",),
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
            optional_dependencies=("libpysal", "spreg"),
        ),
        _spec(
            "spatial_error",
            "geo",
            SpatialErrorRegressor,
            ("regression",),
            ("spatial_econometrics", "save", "load"),
            optional_dependencies=("libpysal", "spreg"),
        ),
        _spec(
            "spatial_durbin",
            "geo",
            SpatialDurbinRegressor,
            ("regression",),
            ("spatial_econometrics", "save", "load"),
            optional_dependencies=("libpysal", "spreg"),
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


def _load(module: str, name: str) -> Factory:
    from importlib import import_module

    return getattr(import_module(f". {module}".replace(" ", ""), __package__), name)


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
            name: hasattr(model, name) for name in ("fit", "predict", "score", "save", "set_params")
        }
        | {"load": hasattr(model.__class__, "load"), "get_params": hasattr(model, "get_params")},
    }


def save_model_card(model: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(model_card(model), indent=2, sort_keys=True), encoding="utf-8")


def _spec(
    name: str,
    namespace: str,
    factory: Factory,
    task_types: tuple[str, ...],
    capabilities: tuple[str, ...],
    *,
    tier: str | None = None,
    backend: str = "rust_native",
    evidence_level: str = "real_data",
    optional_dependencies: tuple[str, ...] = (),
    notes: str = "",
) -> ModelSpec:
    key = f"{namespace}.{name}"
    tier = tier or (
        "stable"
        if key in STABLE_MODEL_KEYS
        else "experimental"
        if key in EXPERIMENTAL_MODEL_KEYS
        else "supported"
    )
    if evidence_level == "real_data" and tier != "stable":
        evidence_level = "experimental_only" if tier == "experimental" else "synthetic"
    if backend == "rust_native" and tier != "stable":
        backend = "python_orchestration"
    return ModelSpec(
        metadata=ModelMetadata(
            name=name,
            namespace=namespace,
            task_types=task_types,
            capabilities=capabilities,
            stable=tier == "stable",
            tier=tier,
            backend=backend,
            evidence_level=evidence_level,
            optional_dependencies=optional_dependencies,
            notes=notes,
        ),
        factory=factory,
    )


def model_manifest(*, tier: str | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in ModelRegistry.defaults().manifest(tier=tier)]


def native_model_manifest() -> list[dict[str, Any]]:
    try:
        from . import _native
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cartoboost._native is required to read the Rust model manifest") from exc
    manifest_fn = getattr(_native, "model_manifest_json", None)
    if manifest_fn is None:
        raise ImportError("installed CartoBoost native extension lacks model_manifest_json")
    payload = json.loads(manifest_fn())
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Rust model manifest must be a list of objects")
    return payload


def _registry_key(namespace: str, name: str) -> str:
    return f"{namespace.strip()}.{name.strip()}"


def _as_2d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite two-dimensional array")
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
    "EXPERIMENTAL_MODEL_KEYS",
    "ModelMetadata",
    "ModelRegistry",
    "ModelSpec",
    "MODEL_ARTIFACT_VERSION",
    "MODEL_BACKENDS",
    "MODEL_EVIDENCE_LEVELS",
    "MODEL_TIERS",
    "STABLE_MODEL_KEYS",
    "default_model_specs",
    "model_manifest",
    "native_model_manifest",
    "model_card",
    "save_model_card",
]
