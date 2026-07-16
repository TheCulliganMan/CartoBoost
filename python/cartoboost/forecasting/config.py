"""Strict TOML configuration for forecasting runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Backend
from .registry import ForecastModelSpec, ForecastRegistry

ROOT_FIELDS = {
    "allow_unknown",
    "freq",
    "horizon",
    "target_column",
    "time_column",
    "panel_columns",
    "feature_config",
    "params",
    "backtest",
    "artifact",
    "reconciliation",
    "models",
    "metadata",
}
MODEL_FIELDS = {
    "name",
    "model",
    "model_type",
    "params",
    "optional_dependencies",
    "metadata",
}
RECONCILIATION_FIELDS = {
    "backend",
    "method",
    "hierarchy",
    "summing_matrix",
    "residual_covariance",
    "covariance_method",
    "series_id_column",
    "parent_column",
    "child_column",
    "non_negative",
    "params",
    "metadata",
}
RECONCILIATION_METHODS = {
    "bottom_up",
    "bottom_up_reconciler",
    "min_trace",
    "min_trace_reconciler",
}


@dataclass(frozen=True)
class ReconciliationConfig:
    """Portable hierarchy reconciliation configuration for native reconcilers."""

    method: str
    hierarchy: Mapping[str, Any] = field(default_factory=dict)
    summing_matrix: Any | None = None
    residual_covariance: Any | None = None
    covariance_method: str | None = None
    series_id_column: str | None = None
    parent_column: str | None = None
    child_column: str | None = None
    non_negative: bool = False
    backend: Backend | str = Backend.CPU
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = self.method.strip()
        if method not in RECONCILIATION_METHODS:
            known = ", ".join(sorted(RECONCILIATION_METHODS))
            raise ValueError(f"unknown reconciliation method {self.method!r}; known: {known}")
        if (self.parent_column is None) != (self.child_column is None):
            raise ValueError("parent_column and child_column must be provided together")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def model_name(self) -> str:
        if self.method == "bottom_up":
            return "bottom_up_reconciler"
        if self.method == "min_trace":
            return "min_trace_reconciler"
        return self.method

    def to_params(self) -> dict[str, Any]:
        params = dict(self.params)
        params.update(
            {
                "hierarchy": dict(self.hierarchy),
                "summing_matrix": self.summing_matrix,
                "series_id_column": self.series_id_column,
                "parent_column": self.parent_column,
                "child_column": self.child_column,
                "non_negative": self.non_negative,
                "backend": str(self.backend),
                "metadata": dict(self.metadata),
            }
        )
        if self.model_name == "min_trace_reconciler":
            params["residual_covariance"] = self.residual_covariance
            if self.covariance_method is not None:
                params["covariance_method"] = self.covariance_method
        return {key: value for key, value in params.items() if value is not None}

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "hierarchy": dict(self.hierarchy),
            "summing_matrix": self.summing_matrix,
            "residual_covariance": self.residual_covariance,
            "covariance_method": self.covariance_method,
            "series_id_column": self.series_id_column,
            "parent_column": self.parent_column,
            "child_column": self.child_column,
            "non_negative": self.non_negative,
            "backend": str(self.backend),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }


@dataclass
class ForecastingConfig:
    """Portable forecasting configuration parsed from TOML or dictionaries."""

    horizon: int
    freq: str | None = None
    models: tuple[ForecastModelSpec, ...] = ()
    target_column: str | None = None
    time_column: str | None = None
    panel_columns: tuple[str, ...] = ()
    feature_config: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    backtest: Mapping[str, Any] = field(default_factory=dict)
    artifact: Mapping[str, Any] = field(default_factory=dict)
    reconciliation: ReconciliationConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allow_unknown: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "panel_columns", tuple(self.panel_columns))
        object.__setattr__(self, "feature_config", dict(self.feature_config))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "backtest", dict(self.backtest))
        object.__setattr__(self, "artifact", dict(self.artifact))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_path(cls, path: str | Path) -> ForecastingConfig:
        return cls.from_toml(Path(path).read_text())

    @classmethod
    def from_toml(cls, text: str) -> ForecastingConfig:
        return cls.from_mapping(_loads_toml(text))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ForecastingConfig:
        data = dict(values)
        allow_unknown = bool(data.get("allow_unknown", False))
        unknown = sorted(set(data).difference(ROOT_FIELDS))
        metadata = dict(data.get("metadata", {}))
        if unknown and not allow_unknown:
            raise ValueError(f"unknown forecasting config field(s): {', '.join(unknown)}")
        if unknown:
            metadata["unknown_fields"] = {key: data[key] for key in unknown}
        if "horizon" not in data:
            raise ValueError("forecasting config requires 'horizon'")
        models = tuple(
            _model_spec_from_mapping(item, allow_unknown=allow_unknown)
            for item in data.get("models", ())
        )
        return cls(
            horizon=data["horizon"],
            freq=data.get("freq"),
            models=models,
            target_column=data.get("target_column"),
            time_column=data.get("time_column"),
            panel_columns=tuple(data.get("panel_columns", ())),
            feature_config=dict(data.get("feature_config", {})),
            params=dict(data.get("params", {})),
            backtest=dict(data.get("backtest", {})),
            artifact=dict(data.get("artifact", {})),
            reconciliation=_reconciliation_config_from_mapping(
                data["reconciliation"], allow_unknown=allow_unknown
            )
            if "reconciliation" in data
            else None,
            metadata=metadata,
            allow_unknown=allow_unknown,
        )

    def registry(self, *, include_defaults: bool = True) -> ForecastRegistry:
        registry = ForecastRegistry.defaults() if include_defaults else ForecastRegistry()
        for spec in self.models:
            if spec.factory is None and spec.name in registry.names():
                base = registry.get(spec.name)
                spec = ForecastModelSpec(
                    name=spec.name,
                    factory=base.factory,
                    params={**dict(base.params), **dict(spec.params)},
                    optional_dependencies=spec.optional_dependencies or base.optional_dependencies,
                    metadata={**dict(base.metadata), **dict(spec.metadata)},
                )
            registry.register(spec, override=True)
        return registry

    def construct_models(self, registry: ForecastRegistry | None = None) -> dict[str, Any]:
        active_registry = registry or self.registry(include_defaults=True)
        if self.models:
            names = [spec.name for spec in self.models]
        else:
            names = []
        return {name: active_registry.create(name) for name in names}

    def construct_reconciler(self, registry: ForecastRegistry | None = None) -> Any | None:
        if self.reconciliation is None:
            return None
        active_registry = registry or self.registry(include_defaults=True)
        _register_reconciliation_specs(active_registry)
        return active_registry.create(
            self.reconciliation.model_name,
            **self.reconciliation.to_params(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_unknown": self.allow_unknown,
            "freq": self.freq,
            "horizon": self.horizon,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "panel_columns": list(self.panel_columns),
            "feature_config": dict(self.feature_config),
            "params": dict(self.params),
            "backtest": dict(self.backtest),
            "artifact": dict(self.artifact),
            "reconciliation": None
            if self.reconciliation is None
            else self.reconciliation.to_dict(),
            "metadata": dict(self.metadata),
            "models": [
                {
                    "name": spec.name,
                    "params": dict(spec.params),
                    "optional_dependencies": list(spec.optional_dependencies),
                    "metadata": dict(spec.metadata),
                }
                for spec in self.models
            ],
        }


def _model_spec_from_mapping(
    values: Mapping[str, Any], *, allow_unknown: bool
) -> ForecastModelSpec:
    data = dict(values)
    unknown = sorted(set(data).difference(MODEL_FIELDS))
    metadata = dict(data.get("metadata", {}))
    if unknown and not allow_unknown:
        raise ValueError(f"unknown model config field(s): {', '.join(unknown)}")
    if unknown:
        metadata["unknown_fields"] = {key: data[key] for key in unknown}
    name = data.get("name") or data.get("model") or data.get("model_type")
    if not name:
        raise ValueError("each model config requires 'name', 'model', or 'model_type'")
    return ForecastModelSpec(
        name=str(name),
        params=dict(data.get("params", {})),
        optional_dependencies=tuple(_string_sequence(data.get("optional_dependencies", ()))),
        metadata=metadata,
    )


def _reconciliation_config_from_mapping(
    values: Mapping[str, Any], *, allow_unknown: bool
) -> ReconciliationConfig:
    data = dict(values)
    unknown = sorted(set(data).difference(RECONCILIATION_FIELDS))
    metadata = dict(data.get("metadata", {}))
    if unknown and not allow_unknown:
        raise ValueError(f"unknown reconciliation config field(s): {', '.join(unknown)}")
    if unknown:
        metadata["unknown_fields"] = {key: data[key] for key in unknown}
    if "method" not in data:
        raise ValueError("reconciliation config requires 'method'")
    return ReconciliationConfig(
        method=str(data["method"]),
        hierarchy=dict(data.get("hierarchy", {})),
        summing_matrix=data.get("summing_matrix"),
        residual_covariance=data.get("residual_covariance"),
        covariance_method=data.get("covariance_method"),
        series_id_column=data.get("series_id_column"),
        parent_column=data.get("parent_column"),
        child_column=data.get("child_column"),
        non_negative=bool(data.get("non_negative", False)),
        backend=Backend(str(data.get("backend", "cpu"))),
        params=dict(data.get("params", {})),
        metadata=metadata,
    )


def _string_sequence(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _register_reconciliation_specs(registry: ForecastRegistry) -> None:
    from .ensemble import BottomUpReconciler, MinTraceReconciler

    registry.register(
        ForecastModelSpec("bottom_up_reconciler", factory=BottomUpReconciler),
        override=True,
    )
    registry.register(
        ForecastModelSpec("min_trace_reconciler", factory=MinTraceReconciler),
        override=True,
    )


def _loads_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TOML forecasting config parsing requires Python 3.11+ "
                "or the optional 'tomli' package"
            ) from exc
    return tomllib.loads(text)
