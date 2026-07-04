"""Unified model registry and geo-aware model orchestration."""

from __future__ import annotations

import hashlib
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
    has_source_id: bool = False
    has_target_id: bool = False
    has_candidate_set: bool = False
    has_multi_view_graph: bool = False
    has_repeated_entities: bool = False
    history_length: int = 0
    cold_start_fraction: float = 0.0
    requires_uncertainty: bool = False
    requires_decision_output: bool = False
    target_sparsity: float = 0.0
    leakage_constraints: tuple[str, ...] = ("spatial_block",)
    validation_strategy: str = "spatial_holdout"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.target_sparsity) <= 1.0:
            raise ValueError("target_sparsity must be between 0 and 1")
        if int(self.history_length) < 0:
            raise ValueError("history_length must be non-negative")
        if not 0.0 <= float(self.cold_start_fraction) <= 1.0:
            raise ValueError("cold_start_fraction must be between 0 and 1")
        object.__setattr__(
            self,
            "leakage_constraints",
            tuple(str(value) for value in self.leakage_constraints),
        )
        object.__setattr__(self, "history_length", int(self.history_length))
        object.__setattr__(self, "cold_start_fraction", float(self.cold_start_fraction))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "has_coords": self.has_coords,
            "has_graph": self.has_graph,
            "has_panel_id": self.has_panel_id,
            "has_time_index": self.has_time_index,
            "has_source_id": self.has_source_id,
            "has_target_id": self.has_target_id,
            "has_candidate_set": self.has_candidate_set,
            "has_multi_view_graph": self.has_multi_view_graph,
            "has_repeated_entities": self.has_repeated_entities,
            "history_length": self.history_length,
            "cold_start_fraction": self.cold_start_fraction,
            "requires_uncertainty": self.requires_uncertainty,
            "requires_decision_output": self.requires_decision_output,
            "target_sparsity": self.target_sparsity,
            "leakage_constraints": list(self.leakage_constraints),
            "validation_strategy": self.validation_strategy,
        }


@dataclass(frozen=True)
class DataContract:
    """Validated AutoGeoModel inputs and split metadata."""

    X: np.ndarray
    y: np.ndarray
    coords: np.ndarray | None
    graph: Any | None
    panel_id: tuple[str, ...] | None
    time_index: tuple[str, ...] | None
    source_id: tuple[str, ...] | None
    target_id: tuple[str, ...] | None
    known_future_covariates: tuple[str, ...]
    feature_roles: Mapping[str, Any]
    schema_hash: str
    split_manifest: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.y.shape[0]),
            "n_features": int(self.X.shape[1]),
            "has_coords": self.coords is not None,
            "has_graph": self.graph is not None,
            "has_panel_id": self.panel_id is not None,
            "has_time_index": self.time_index is not None,
            "has_source_id": self.source_id is not None,
            "has_target_id": self.target_id is not None,
            "known_future_covariates": list(self.known_future_covariates),
            "feature_roles": _jsonable(dict(self.feature_roles)),
            "schema_hash": self.schema_hash,
            "split_manifest": _jsonable(dict(self.split_manifest)),
        }


@dataclass(frozen=True)
class ModelEvidenceCard:
    """Evidence summary produced by AutoGeoModel."""

    selected_family: str
    all_candidates: tuple[str, ...]
    candidates_tried: tuple[str, ...]
    candidates_skipped: tuple[Mapping[str, Any], ...]
    baseline_comparison: Mapping[str, Any] | None
    claim_falsifier_baselines: tuple[str, ...]
    split_manifest: Mapping[str, Any]
    split_hash: str
    leakage_policy: Mapping[str, Any]
    residual_diagnostics: Mapping[str, Any] | None
    spatial_diagnostics: Mapping[str, Any] | None
    temporal_diagnostics: Mapping[str, Any] | None
    interval_diagnostics: Mapping[str, Any] | None
    diagnostics: Mapping[str, Any]
    uncertainty_report: Mapping[str, Any] | None
    save_load_parity: Mapping[str, Any]
    feature_roles: Mapping[str, Any]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_family": self.selected_family,
            "all_candidates": list(self.all_candidates),
            "candidates_tried": list(self.candidates_tried),
            "candidates_skipped": [dict(row) for row in self.candidates_skipped],
            "skipped_candidates_with_reasons": [dict(row) for row in self.candidates_skipped],
            "baseline_comparison": None
            if self.baseline_comparison is None
            else dict(self.baseline_comparison),
            "claim_falsifier_baselines": list(self.claim_falsifier_baselines),
            "split_manifest": _jsonable(dict(self.split_manifest)),
            "split_hash": self.split_hash,
            "leakage_policy": dict(self.leakage_policy),
            "residual_diagnostics": None
            if self.residual_diagnostics is None
            else dict(self.residual_diagnostics),
            "spatial_diagnostics": None
            if self.spatial_diagnostics is None
            else dict(self.spatial_diagnostics),
            "temporal_diagnostics": None
            if self.temporal_diagnostics is None
            else dict(self.temporal_diagnostics),
            "interval_diagnostics": None
            if self.interval_diagnostics is None
            else dict(self.interval_diagnostics),
            "diagnostics": _jsonable(dict(self.diagnostics)),
            "uncertainty_report": None
            if self.uncertainty_report is None
            else dict(self.uncertainty_report),
            "save_load_parity": _jsonable(dict(self.save_load_parity)),
            "feature_roles": _jsonable(dict(self.feature_roles)),
            "limitations": list(self.limitations),
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
            families.append("geo.nngp")
        if context.has_graph and self.max_escalation_level >= 2:
            families.append("graph.dcrnn")
        if context.has_time_index and context.has_panel_id and self.max_escalation_level >= 2:
            families.append("forecasting.cartoboost_lag")
            families.append("forecasting.auto_forecaster")
        if "intervals" in context.leakage_constraints and self.max_escalation_level >= 3:
            families.append("prob.conformal_interval")
        if self.max_escalation_level >= 3:
            if context.has_source_id and context.has_target_id:
                families.append("deep.pair_embedding_mlp")
            if context.has_time_index and context.has_panel_id:
                families.append("deep.temporal_ssm")
                families.append("deep.inverted_transformer")
                if context.history_length >= 16 or context.cold_start_fraction > 0.0:
                    families.append("deep.retrieval_augmented")
            if context.has_graph and context.has_time_index:
                families.append("deep.delay_aware_graph_transformer")
            if context.has_multi_view_graph:
                families.append("deep.multi_view_spatial_attention")
            if context.has_repeated_entities or context.has_panel_id:
                families.append("deep.regime_moe")
            if "response_curve" in context.leakage_constraints:
                families.append("deep.monotone_basis_response")
            if context.has_candidate_set or context.requires_decision_output:
                families.append("deep.choice_set_transformer")
            if context.requires_uncertainty:
                families.append("deep.flow_uncertainty_head")
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
        source_id: Any | None = None,
        target_id: Any | None = None,
        known_future_covariates: Sequence[str] | None = None,
        feature_roles: Mapping[str, Any] | None = None,
        interval_evidence: bool = False,
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
            has_source_id=source_id is not None,
            has_target_id=target_id is not None,
            has_candidate_set=_feature_role_enabled(feature_roles, "candidate_set"),
            has_multi_view_graph=_feature_role_enabled(feature_roles, "multi_view_graph"),
            has_repeated_entities=_has_repeated_entities(panel_id, source_id, target_id),
            history_length=_history_length(panel_id, time_index, len(y_array)),
            cold_start_fraction=_cold_start_fraction(panel_id, source_id, target_id),
            requires_uncertainty=interval_evidence or "intervals" in leakage_constraints,
            requires_decision_output=_feature_role_enabled(feature_roles, "decision_output"),
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
        contract = _build_data_contract(
            x_array,
            y_array,
            coords=coords_array,
            graph=graph,
            panel_id=panel_id,
            time_index=time_index,
            source_id=source_id,
            target_id=target_id,
            known_future_covariates=known_future_covariates,
            feature_roles=feature_roles,
            train_idx=train_idx,
            holdout_idx=holdout_idx,
            validation_strategy=validation_strategy,
        )

        evaluations: list[dict[str, Any]] = []
        fitted: dict[str, Any] = {}
        candidate_families = self.candidate_families(context)
        if interval_evidence and "prob.conformal_interval" not in candidate_families:
            candidate_families.append("prob.conformal_interval")
        for adapter in _autogeo_adapters(
            candidate_families,
            conformal_alpha=self.conformal_alpha,
            interval_evidence=interval_evidence or "intervals" in leakage_constraints,
        ):
            fit_result = adapter.fit(
                contract,
                random_state=self.random_state,
            )
            if fit_result.status == "skipped":
                evaluations.append(fit_result.to_evaluation())
                continue
            model = fit_result.model
            pred = fit_result.prediction
            residuals = y_array[holdout_idx] - np.asarray(pred, dtype=float)
            diagnostics = _diagnostics(y_array[holdout_idx], pred)
            if coords_array is not None and residuals.shape[0] > 2 and np.std(residuals) > 0.0:
                diagnostics["residual_morans_i"] = float(
                    residual_morans_i(coords_array[holdout_idx], residuals)
                )
            evaluations.append(
                {
                    "family": adapter.family,
                    "status": "fit",
                    "artifact_supported": _artifact_supported(model),
                    "prediction_parity": _prediction_parity(model, contract),
                    "metric": diagnostics[self.metric],
                    "diagnostics": {**diagnostics, **fit_result.diagnostics},
                }
            )
            fitted[adapter.family] = model

        scored = [
            row
            for row in evaluations
            if row["status"] == "fit"
            and row.get("artifact_supported") is True
            and row.get("prediction_parity", {}).get("passed") is True
        ]
        if not scored:
            raise RuntimeError("AutoGeoModel found no serializable parity-safe fit candidates")
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
        final_contract = _replace_contract_split(contract, final_idx, holdout_idx)
        final_fit = _adapter_by_family(
            self.selected_family_,
            conformal_alpha=self.conformal_alpha,
            interval_evidence=interval_evidence or "intervals" in leakage_constraints,
        ).fit(
            final_contract,
            random_state=self.random_state,
        )
        self.selected_model_ = (
            fitted[self.selected_family_] if final_fit.status != "fit" else final_fit.model
        )
        final_parity = _prediction_parity(self.selected_model_, final_contract)
        if final_parity.get("passed") is not True:
            raise RuntimeError(
                "AutoGeoModel selected model failed save/load prediction parity: "
                f"{final_parity.get('reason', final_parity)}"
            )
        evidence_card = _model_evidence_card(
            selected_family=self.selected_family_,
            evaluations=evaluations,
            contract=contract,
            context=context,
            all_candidates=candidate_families,
        )
        self.metadata_ = {
            "model": "AutoGeoModel",
            "context": context.to_dict(),
            "data_contract": contract.to_dict(),
            "selected_family": self.selected_family_,
            "selection_metric": self.metric,
            "evaluations": evaluations,
            "selected_model_prediction_parity": final_parity,
            "evidence_card": evidence_card.to_dict(),
            "leakage_safe": True,
        }
        return self

    def predict(self, X: Any, *, coords: Any | None = None, **kwargs: Any) -> np.ndarray:
        model = self._require_model()
        if self.selected_family_ in {"geo.residual_nngp", "geo.nngp"}:
            if coords is None:
                raise ValueError("coords are required for the selected spatial model")
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
    card = {
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
    if isinstance(metadata, Mapping) and "evidence_card" in metadata:
        card["evidence_card"] = _jsonable(metadata["evidence_card"])
    return card


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


@dataclass(frozen=True)
class _AdapterResult:
    family: str
    status: str
    model: Any | None = None
    prediction: np.ndarray | None = None
    reason_code: str = ""
    reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_evaluation(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "diagnostics": _jsonable(dict(self.diagnostics)),
        }


class _AutoGeoAdapter:
    family: str
    required_fields: tuple[str, ...] = ()

    def fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        missing = [field for field in self.required_fields if getattr(contract, field) is None]
        if missing:
            return _AdapterResult(
                self.family,
                "skipped",
                reason_code="missing_required_fields",
                reason=f"requires {', '.join(missing)}",
            )
        try:
            return self._fit(contract, random_state=random_state)
        except (ImportError, NotImplementedError, AttributeError) as exc:
            return _AdapterResult(
                self.family,
                "skipped",
                reason_code="unsupported_runtime",
                reason=str(exc),
            )
        except ValueError as exc:
            return _AdapterResult(
                self.family,
                "skipped",
                reason_code="contract_invalid",
                reason=str(exc),
            )

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        raise NotImplementedError


class _MeanAdapter(_AutoGeoAdapter):
    family = "models.mean_baseline"

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        del random_state
        train, holdout = _contract_split(contract)
        model = _MeanRegressor().fit(contract.X[train], contract.y[train])
        return _AdapterResult(self.family, "fit", model, model.predict(contract.X[holdout]))


class _CartoBoostAdapter(_AutoGeoAdapter):
    family = "models.cartoboost_regressor"

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        train, holdout = _contract_split(contract)
        model = _autogeo_booster(random_state)
        model.fit(contract.X[train], contract.y[train])
        return _AdapterResult(
            self.family,
            "fit",
            model,
            np.asarray(model.predict(contract.X[holdout]), dtype=float),
        )


class _ResidualNNGPAdapter(_AutoGeoAdapter):
    family = "geo.residual_nngp"
    required_fields = ("coords",)

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        from .geostats import ResidualNNGPRegressor

        train, holdout = _contract_split(contract)
        model = ResidualNNGPRegressor(_autogeo_booster(random_state))
        model.fit(contract.X[train], contract.y[train], coords=contract.coords[train])
        pred = model.predict(contract.X[holdout], coords=contract.coords[holdout])
        return _AdapterResult(
            self.family,
            "fit",
            model,
            np.asarray(pred, dtype=float),
            diagnostics={"spatial_adapter": "residual_nngp"},
        )


class _NNGPAdapter(_AutoGeoAdapter):
    family = "geo.nngp"
    required_fields = ("coords",)

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        del random_state
        from .geostats import NearestNeighborGPRegressor

        train, holdout = _contract_split(contract)
        model = NearestNeighborGPRegressor()
        model.fit(None, contract.y[train], coords=contract.coords[train])
        pred = model.predict(None, coords=contract.coords[holdout])
        return _AdapterResult(
            self.family,
            "fit",
            model,
            np.asarray(pred, dtype=float),
            diagnostics={"spatial_adapter": "nngp"},
        )


class _ForecastAdapter(_AutoGeoAdapter):
    family = "forecasting.cartoboost_lag"
    required_fields = ("panel_id", "time_index")

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        del random_state
        from .forecasting import CartoBoostLagForecaster

        return _fit_forecaster_family(
            self.family,
            CartoBoostLagForecaster(),
            contract,
            diagnostics={"temporal_adapter": "cartoboost_lag"},
        )


class _AutoForecasterAdapter(_AutoGeoAdapter):
    family = "forecasting.auto_forecaster"
    required_fields = ("panel_id", "time_index")

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        del random_state
        from .forecasting import AutoForecaster

        if "load" not in AutoForecaster.__dict__:
            return _AdapterResult(
                self.family,
                "skipped",
                reason_code="missing_serialization_contract",
                reason="AutoForecaster does not expose a load(path) artifact contract",
            )
        return _fit_forecaster_family(
            self.family,
            AutoForecaster(),
            contract,
            diagnostics={"temporal_adapter": "auto_forecaster"},
        )


class _GraphAdapter(_AutoGeoAdapter):
    family = "graph.dcrnn"
    required_fields = ("graph",)

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        del random_state
        if contract.graph.__class__.__name__ != "GraphTemporalFrame":
            return _AdapterResult(
                self.family,
                "skipped",
                reason_code="unsupported_graph_contract",
                reason="graph adapter requires a GraphTemporalFrame",
            )
        from .forecasting.graph_st import DCRNNForecaster

        _train, holdout = _contract_split(contract)
        model = DCRNNForecaster(epochs=3)
        model.fit(contract.graph)
        pred = np.asarray(model.predict(max(1, int(holdout.shape[0]))), dtype=float).reshape(-1)
        if pred.shape[0] < holdout.shape[0]:
            raise ValueError("graph adapter returned fewer predictions than holdout rows")
        return _AdapterResult(
            self.family,
            "fit",
            model,
            pred[: holdout.shape[0]],
            diagnostics={"graph_adapter": "dcrnn", "horizon": int(holdout.shape[0])},
        )


class _DeepUnavailableAdapter(_AutoGeoAdapter):
    required_fields: tuple[str, ...] = ()

    def __init__(self, family: str, *, reason: str, diagnostics: Mapping[str, Any]) -> None:
        self.family = family
        self.reason = reason
        self.diagnostics = dict(diagnostics)

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        del contract, random_state
        return _AdapterResult(
            self.family,
            "skipped",
            reason_code="deep_candidate_not_registered_for_autogeo_selection",
            reason=self.reason,
            diagnostics=self.diagnostics,
        )


class _ConformalAdapter(_AutoGeoAdapter):
    family = "prob.conformal_interval"

    def __init__(self, *, alpha: float, enabled: bool) -> None:
        self.alpha = float(alpha)
        self.enabled = bool(enabled)

    def fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        if not self.enabled:
            return _AdapterResult(
                self.family,
                "skipped",
                reason_code="interval_evidence_not_requested",
                reason="requires interval_evidence=True or leakage constraint 'intervals'",
            )
        return super().fit(contract, random_state=random_state)

    def _fit(self, contract: DataContract, *, random_state: int) -> _AdapterResult:
        from .forecasting.probabilistic import ConformalIntervalRegressor

        train, holdout = _contract_split(contract)
        model = ConformalIntervalRegressor(_autogeo_booster(random_state), alpha=self.alpha)
        model.fit(
            contract.X[train],
            contract.y[train],
            contract.X[holdout],
            contract.y[holdout],
            train_end_exclusive=int(train.max()) + 1,
            calibration_start=int(holdout.min()),
            calibration_end_exclusive=int(holdout.max()) + 1,
            test_start=int(holdout.max()) + 1,
        )
        pred = np.asarray(model.predict(contract.X[holdout]), dtype=float)
        interval = model.predict_interval(contract.X[holdout], test_start=int(holdout.max()) + 1)
        width = np.asarray(interval.upper, dtype=float) - np.asarray(interval.lower, dtype=float)
        return _AdapterResult(
            self.family,
            "fit",
            model,
            pred,
            diagnostics={
                "interval_adapter": "split_conformal",
                "mean_interval_width": float(np.mean(width)),
            },
        )


def _autogeo_adapters(
    families: Sequence[str], *, conformal_alpha: float, interval_evidence: bool
) -> list[_AutoGeoAdapter]:
    adapters: dict[str, _AutoGeoAdapter] = {
        "models.mean_baseline": _MeanAdapter(),
        "models.cartoboost_regressor": _CartoBoostAdapter(),
        "geo.residual_nngp": _ResidualNNGPAdapter(),
        "geo.nngp": _NNGPAdapter(),
        "forecasting.cartoboost_lag": _ForecastAdapter(),
        "forecasting.auto_forecaster": _AutoForecasterAdapter(),
        "graph.graph_residual": _GraphAdapter(),
        "graph.dcrnn": _GraphAdapter(),
        "prob.conformal_interval": _ConformalAdapter(
            alpha=conformal_alpha,
            enabled=interval_evidence,
        ),
    }
    for family, diagnostics in _deep_candidate_diagnostics().items():
        adapters[family] = _DeepUnavailableAdapter(
            family,
            reason=(
                f"{family} is eligible from the AutoGeoModel data contract, but no "
                "parity-safe AutoGeoModel adapter is registered yet"
            ),
            diagnostics=diagnostics,
        )
    return [adapters[family] for family in families if family in adapters]


def _adapter_by_family(
    family: str, *, conformal_alpha: float, interval_evidence: bool
) -> _AutoGeoAdapter:
    adapters = _autogeo_adapters(
        [family], conformal_alpha=conformal_alpha, interval_evidence=interval_evidence
    )
    if not adapters:
        raise ValueError(f"unknown AutoGeoModel family {family!r}")
    return adapters[0]


def _fit_forecaster_family(
    family: str,
    model: Any,
    contract: DataContract,
    *,
    diagnostics: Mapping[str, Any],
) -> _AdapterResult:
    _train, holdout = _contract_split(contract)
    frame = _forecast_frame_from_contract(contract)
    model.fit(frame)
    horizon = int(holdout.shape[0])
    pred = np.asarray(model.predict(horizon), dtype=float).reshape(-1)
    if pred.shape[0] < horizon:
        raise ValueError(f"{family} returned fewer predictions than holdout rows")
    return _AdapterResult(
        family,
        "fit",
        model,
        pred[:horizon],
        diagnostics={**dict(diagnostics), "horizon": horizon},
    )


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


def _build_data_contract(
    X: np.ndarray,
    y: np.ndarray,
    *,
    coords: np.ndarray | None,
    graph: Any | None,
    panel_id: Any | None,
    time_index: Any | None,
    source_id: Any | None,
    target_id: Any | None,
    known_future_covariates: Sequence[str] | None,
    feature_roles: Mapping[str, Any] | None,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    validation_strategy: str,
) -> DataContract:
    return DataContract(
        X=X,
        y=y,
        coords=coords,
        graph=graph,
        panel_id=None if panel_id is None else _as_string_tuple(panel_id, "panel_id", y.shape[0]),
        time_index=None
        if time_index is None
        else _as_string_tuple(time_index, "time_index", y.shape[0]),
        source_id=None
        if source_id is None
        else _as_string_tuple(source_id, "source_id", y.shape[0]),
        target_id=None
        if target_id is None
        else _as_string_tuple(target_id, "target_id", y.shape[0]),
        known_future_covariates=tuple(str(value) for value in (known_future_covariates or ())),
        feature_roles={} if feature_roles is None else dict(feature_roles),
        schema_hash=_schema_hash(
            X, coords, panel_id, time_index, source_id, target_id, feature_roles
        ),
        split_manifest={
            "strategy": validation_strategy,
            "train": [int(value) for value in train_idx],
            "holdout": [int(value) for value in holdout_idx],
            "split_hash": _split_hash(train_idx, holdout_idx, validation_strategy),
        },
    )


def _replace_contract_split(
    contract: DataContract, train_idx: np.ndarray, holdout_idx: np.ndarray
) -> DataContract:
    return DataContract(
        X=contract.X,
        y=contract.y,
        coords=contract.coords,
        graph=contract.graph,
        panel_id=contract.panel_id,
        time_index=contract.time_index,
        source_id=contract.source_id,
        target_id=contract.target_id,
        known_future_covariates=contract.known_future_covariates,
        feature_roles=contract.feature_roles,
        schema_hash=contract.schema_hash,
        split_manifest={
            "strategy": contract.split_manifest["strategy"],
            "train": [int(value) for value in train_idx],
            "holdout": [int(value) for value in holdout_idx],
            "split_hash": _split_hash(
                train_idx,
                holdout_idx,
                str(contract.split_manifest["strategy"]),
            ),
        },
    )


def _contract_split(contract: DataContract) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(contract.split_manifest["train"], dtype=np.int64),
        np.asarray(contract.split_manifest["holdout"], dtype=np.int64),
    )


def _as_string_tuple(values: Any, name: str, n_rows: int) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.shape[0] != n_rows:
        raise ValueError(f"{name} must contain the same number of rows as X")
    return tuple(str(value) for value in array.tolist())


def _schema_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(_jsonable(part)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _split_hash(train_idx: np.ndarray, holdout_idx: np.ndarray, strategy: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(strategy).encode("utf-8"))
    digest.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    digest.update(np.asarray(holdout_idx, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _prediction_parity(model: Any, contract: DataContract) -> dict[str, Any]:
    if not _artifact_supported(model):
        return {"passed": False, "reason": "model does not expose save/load"}
    try:
        before = _predict_for_parity(model, contract)
        loaded = _load_supported_model(_dump_supported_model(model))
        after = _predict_for_parity(loaded, contract)
        size = min(before.shape[0], after.shape[0])
        before = before[:size]
        after = after[:size]
        passed = bool(size > 0 and np.allclose(before, after, rtol=1e-8, atol=1e-8))
        return {"passed": passed, "max_abs_diff": float(np.max(np.abs(before - after)))}
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "reason": str(exc)}


def _predict_for_parity(model: Any, contract: DataContract) -> np.ndarray:
    _train, holdout = _contract_split(contract)
    name = model.__class__.__name__
    if name in {"ResidualNNGPRegressor", "NearestNeighborGPRegressor"}:
        return np.asarray(
            model.predict(contract.X[holdout], coords=contract.coords[holdout]),
            dtype=float,
        ).reshape(-1)
    if name in {"CartoBoostLagForecaster", "AutoForecaster"}:
        return np.asarray(model.predict(int(holdout.shape[0])), dtype=float).reshape(-1)
    if name == "DCRNNForecaster":
        return np.asarray(model.predict(max(1, int(holdout.shape[0]))), dtype=float).reshape(-1)
    return np.asarray(model.predict(contract.X[holdout]), dtype=float).reshape(-1)


def _forecast_frame_from_contract(contract: DataContract) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("forecasting adapter requires pandas") from exc
    from .forecasting import ForecastFrame

    frame = pd.DataFrame(
        {
            "__time__": pd.to_datetime(list(contract.time_index)),
            "__panel__": list(contract.panel_id),
            "__target__": contract.y,
        }
    )
    return ForecastFrame.from_pandas(
        frame,
        timestamp_col="__time__",
        target_col="__target__",
        series_id_col="__panel__",
        freq=None,
        allow_irregular=True,
    )


def _model_evidence_card(
    *,
    selected_family: str,
    evaluations: Sequence[Mapping[str, Any]],
    contract: DataContract,
    context: GeoTaskContext,
    all_candidates: Sequence[str],
) -> ModelEvidenceCard:
    fit_rows = [row for row in evaluations if row.get("status") == "fit"]
    skipped = [row for row in evaluations if row.get("status") == "skipped"]
    baseline = next((row for row in fit_rows if row.get("family") == "models.mean_baseline"), None)
    selected = next((row for row in fit_rows if row.get("family") == selected_family), None)
    baseline_comparison = None
    if baseline is not None and selected is not None:
        baseline_metric = float(baseline["metric"])
        selected_metric = float(selected["metric"])
        baseline_comparison = {
            "baseline_family": baseline["family"],
            "baseline_metric": baseline_metric,
            "selected_metric": selected_metric,
            "relative_improvement": (baseline_metric - selected_metric) / baseline_metric
            if baseline_metric > 0.0
            else float("nan"),
        }
    selected_diagnostics = None if selected is None else dict(selected.get("diagnostics", {}))
    spatial_diagnostics = _candidate_diagnostics(
        fit_rows,
        families={"geo.residual_nngp", "geo.nngp"},
        keys_containing=("moran", "spatial"),
    )
    if contract.coords is not None and selected_diagnostics is not None:
        spatial_diagnostics = {
            **{
                key: value
                for key, value in selected_diagnostics.items()
                if "moran" in key.lower() or "spatial" in key.lower()
            },
            **spatial_diagnostics,
        }
    temporal_diagnostics = _candidate_diagnostics(
        fit_rows,
        families={"forecasting.cartoboost_lag", "forecasting.auto_forecaster", "graph.dcrnn"},
        keys_containing=("temporal", "horizon", "graph"),
    )
    interval_diagnostics = _candidate_diagnostics(
        fit_rows,
        families={"prob.conformal_interval"},
        keys_containing=("interval", "conformal"),
    )
    uncertainty_report = None
    if context.requires_uncertainty:
        uncertainty_report = {
            "requested": True,
            "candidate": "deep.flow_uncertainty_head"
            if "deep.flow_uncertainty_head" in all_candidates
            else "prob.conformal_interval",
            "status": _candidate_status(evaluations, "deep.flow_uncertainty_head")
            or _candidate_status(evaluations, "prob.conformal_interval"),
            "interval_diagnostics": interval_diagnostics,
        }
    save_load_parity = {
        str(row.get("family")): dict(row.get("prediction_parity", {}))
        for row in fit_rows
        if "prediction_parity" in row
    }
    return ModelEvidenceCard(
        selected_family=selected_family,
        all_candidates=tuple(str(value) for value in all_candidates),
        candidates_tried=tuple(str(row["family"]) for row in fit_rows),
        candidates_skipped=tuple(
            {
                "family": row.get("family"),
                "reason_code": row.get("reason_code"),
                "reason": row.get("reason"),
            }
            for row in skipped
        ),
        baseline_comparison=baseline_comparison,
        claim_falsifier_baselines=tuple(_claim_falsifier_baselines(context, all_candidates)),
        split_manifest=contract.split_manifest,
        split_hash=str(contract.split_manifest["split_hash"]),
        leakage_policy={
            "validation_strategy": context.validation_strategy,
            "constraints": list(context.leakage_constraints),
            "random_row_cv_allowed": False
            if context.has_coords
            else context.validation_strategy not in {"random", "random_row", "random_row_cv"},
        },
        residual_diagnostics=selected_diagnostics,
        spatial_diagnostics=spatial_diagnostics or None,
        temporal_diagnostics=temporal_diagnostics or None,
        interval_diagnostics=interval_diagnostics or None,
        diagnostics={
            "selected": selected_diagnostics,
            "fit_candidates": len(fit_rows),
            "skipped_candidates": len(skipped),
        },
        uncertainty_report=uncertainty_report,
        save_load_parity=save_load_parity,
        feature_roles=contract.feature_roles,
        limitations=tuple(_autogeo_limitations(contract, evaluations)),
    )


def _candidate_status(evaluations: Sequence[Mapping[str, Any]], family: str) -> str | None:
    row = next((item for item in evaluations if item.get("family") == family), None)
    return None if row is None else str(row.get("status"))


def _claim_falsifier_baselines(context: GeoTaskContext, all_candidates: Sequence[str]) -> list[str]:
    baselines = ["models.mean_baseline", "models.cartoboost_regressor"]
    if any(str(candidate).startswith("deep.") for candidate in all_candidates):
        baselines.append("current_stable_cartoboost")
    if context.has_graph:
        baselines.append("static_adjacency_graph_baseline")
    if context.has_time_index and context.has_panel_id:
        baselines.append("non_graph_temporal_baseline")
    if context.requires_uncertainty:
        baselines.append("conformal_interval_baseline")
    return baselines


def _candidate_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    families: set[str],
    keys_containing: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        family = str(row.get("family"))
        if family not in families:
            continue
        for key, value in dict(row.get("diagnostics", {})).items():
            lowered = str(key).lower()
            if any(token in lowered for token in keys_containing):
                result[f"{family}.{key}"] = value
    return result


def _autogeo_limitations(
    contract: DataContract, evaluations: Sequence[Mapping[str, Any]]
) -> list[str]:
    skipped_codes = {
        str(row.get("reason_code")) for row in evaluations if row.get("status") == "skipped"
    }
    limitations: list[str] = []
    if contract.graph is not None and "unsupported_graph_contract" in skipped_codes:
        limitations.append("graph evidence requires a GraphTemporalFrame contract")
    if contract.time_index is None or contract.panel_id is None:
        limitations.append("forecasting candidates require both panel_id and time_index")
    if contract.coords is None:
        limitations.append("spatial diagnostics are unavailable without coords")
    return limitations


def _feature_role_enabled(feature_roles: Mapping[str, Any] | None, key: str) -> bool:
    if not feature_roles:
        return False
    value = feature_roles.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value) > 0
    return bool(value)


def _history_length(panel_id: Any | None, time_index: Any | None, n_rows: int) -> int:
    if time_index is None:
        return int(n_rows)
    if panel_id is None:
        return int(len(set(np.asarray(time_index).tolist())))
    panels = np.asarray(panel_id).tolist()
    counts: dict[str, int] = {}
    for value in panels:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def _has_repeated_entities(*columns: Any | None) -> bool:
    for column in columns:
        if column is None:
            continue
        values = [str(value) for value in np.asarray(column).tolist()]
        if len(set(values)) < len(values):
            return True
    return False


def _cold_start_fraction(*columns: Any | None) -> float:
    values: list[str] = []
    for column in columns:
        if column is None:
            continue
        values.extend(str(value) for value in np.asarray(column).tolist())
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return float(sum(1 for value in values if counts[value] == 1) / len(values))


def _deep_candidate_diagnostics() -> dict[str, dict[str, Any]]:
    return {
        "deep.pair_embedding_mlp": {
            "deep_component": "PairEmbeddingMLP",
            "required_contract": ["source_id", "target_id"],
        },
        "deep.temporal_ssm": {
            "deep_component": "TemporalSSMForecaster",
            "required_contract": ["panel_id", "time_index"],
        },
        "deep.inverted_transformer": {
            "deep_component": "InvertedTemporalTransformer",
            "required_contract": ["panel_id", "time_index"],
        },
        "deep.delay_aware_graph_transformer": {
            "deep_component": "DelayAwareGraphTransformer",
            "required_contract": ["graph", "time_index"],
        },
        "deep.regime_moe": {
            "deep_component": "RegimeMoE",
            "required_contract": ["repeated_entities"],
        },
        "deep.retrieval_augmented": {
            "deep_component": "RetrievalAugmentedForecaster",
            "required_contract": ["panel_id", "time_index", "history"],
        },
        "deep.monotone_basis_response": {
            "deep_component": "MonotoneBasisResponseModel",
            "required_contract": ["response_curve"],
        },
        "deep.choice_set_transformer": {
            "deep_component": "ChoiceSetTransformer",
            "required_contract": ["candidate_set"],
        },
        "deep.flow_uncertainty_head": {
            "deep_component": "ConditionalFlowDistributionHead",
            "required_contract": ["uncertainty_request"],
        },
        "deep.multi_view_spatial_attention": {
            "deep_component": "MultiViewSpatialAttention",
            "required_contract": ["multi_view_graph"],
        },
    }


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
    "DataContract",
    "GeoModelStack",
    "GeoTaskContext",
    "ModelEvidenceCard",
    "ModelMetadata",
    "ModelRegistry",
    "ModelSpec",
    "default_model_specs",
    "model_card",
    "save_model_card",
]
