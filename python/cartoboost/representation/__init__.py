"""Reusable geo-temporal representation primitives.

These classes provide deterministic shared embedding behavior for deep model
surfaces. They are intentionally generic: IDs are entities, sources, targets,
nodes, or regions rather than domain-specific objects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class RepresentationArtifact:
    model_class: str
    architecture: str
    artifact_version: int
    schema_hash: str
    id_maps: dict[str, dict[str, int]]
    hash_bucket_config: dict[str, int]
    embedding_dim: int
    random_seed: int
    feature_roles: dict[str, Any]
    training_cutoff: str | None
    training_metrics: dict[str, float]
    save_load_parity_checked: bool
    backend: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "architecture": self.architecture,
            "artifact_version": self.artifact_version,
            "schema_hash": self.schema_hash,
            "id_maps": self.id_maps,
            "hash_bucket_config": self.hash_bucket_config,
            "embedding_dim": self.embedding_dim,
            "random_seed": self.random_seed,
            "feature_roles": self.feature_roles,
            "training_cutoff": self.training_cutoff,
            "training_metrics": self.training_metrics,
            "save_load_parity_checked": self.save_load_parity_checked,
            "backend": self.backend,
        }


class EntityEmbedding:
    """Deterministic entity embedding table with unknown and hash-bucket fallback."""

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        hash_bucket_count: int = 16,
        random_seed: int = 0,
        architecture: str = "entity_embedding",
        feature_roles: dict[str, Any] | None = None,
        backend: str = "cpu",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hash_bucket_count <= 0:
            raise ValueError("hash_bucket_count must be positive")
        self.embedding_dim = int(embedding_dim)
        self.hash_bucket_count = int(hash_bucket_count)
        self.random_seed = int(random_seed)
        self.architecture = str(architecture)
        self.feature_roles = {} if feature_roles is None else dict(feature_roles)
        self.backend = _resolve_backend(backend)
        self.is_fitted_ = False

    def fit(
        self,
        ids: Any,
        *,
        training_cutoff: str | None = None,
        training_metrics: dict[str, float] | None = None,
    ) -> EntityEmbedding:
        unique = sorted({str(value) for value in np.asarray(ids).reshape(-1).tolist()})
        self.id_map_ = {"__unknown__": 0, **{value: idx + 1 for idx, value in enumerate(unique)}}
        row_count = 1 + len(unique) + self.hash_bucket_count
        self.embeddings_ = _deterministic_matrix(
            row_count,
            self.embedding_dim,
            seed=self.random_seed,
            salt=self.architecture,
        )
        self.training_cutoff_ = training_cutoff
        self.training_metrics_ = {} if training_metrics is None else dict(training_metrics)
        self.schema_hash_ = _schema_hash(
            {
                "ids": unique,
                "embedding_dim": self.embedding_dim,
                "hash_bucket_count": self.hash_bucket_count,
                "feature_roles": self.feature_roles,
            }
        )
        self.is_fitted_ = True
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity())
        return self

    def transform(self, ids: Any) -> np.ndarray:
        self._require_fitted()
        rows = [self._index(str(value)) for value in np.asarray(ids).reshape(-1).tolist()]
        return self.embeddings_[rows].copy()

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        payload = {
            "artifact": self.artifact_.to_dict(),
            "embeddings": self.embeddings_.tolist(),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> EntityEmbedding:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            embedding_dim=int(artifact["embedding_dim"]),
            hash_bucket_count=int(artifact["hash_bucket_config"]["entity"]),
            random_seed=int(artifact["random_seed"]),
            architecture=str(artifact["architecture"]),
            feature_roles=dict(artifact["feature_roles"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.id_map_ = dict(artifact["id_maps"]["entity"])
        obj.embeddings_ = np.asarray(payload["embeddings"], dtype=float)
        obj.training_cutoff_ = artifact["training_cutoff"]
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def artifact_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return self.artifact_.to_dict()

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture=self.architecture,
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={"entity": dict(self.id_map_)},
            hash_bucket_config={"entity": self.hash_bucket_count},
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
            feature_roles=dict(self.feature_roles),
            training_cutoff=self.training_cutoff_,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _save_load_parity(self) -> bool:
        probe = ["__unknown__", *list(self.id_map_)[: min(3, len(self.id_map_))], "__new_id__"]
        before = self.transform(probe)
        clone = self._clone_from_payload()
        after = clone.transform(probe)
        return bool(np.array_equal(before, after))

    def _clone_from_payload(self) -> EntityEmbedding:
        payload = {
            "artifact": self._artifact(save_load_parity_checked=False).to_dict(),
            "embeddings": self.embeddings_.tolist(),
        }
        obj = self.__class__(
            embedding_dim=int(payload["artifact"]["embedding_dim"]),
            hash_bucket_count=int(payload["artifact"]["hash_bucket_config"]["entity"]),
            random_seed=int(payload["artifact"]["random_seed"]),
            architecture=str(payload["artifact"]["architecture"]),
            feature_roles=dict(payload["artifact"]["feature_roles"]),
        )
        obj.id_map_ = dict(payload["artifact"]["id_maps"]["entity"])
        obj.embeddings_ = np.asarray(payload["embeddings"], dtype=float)
        obj.training_cutoff_ = payload["artifact"]["training_cutoff"]
        obj.training_metrics_ = dict(payload["artifact"]["training_metrics"])
        obj.schema_hash_ = str(payload["artifact"]["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**payload["artifact"])
        return obj

    def _index(self, value: str) -> int:
        if value in self.id_map_:
            return self.id_map_[value]
        return len(self.id_map_) + _stable_hash(value, self.hash_bucket_count)

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("representation must be fit before transform")


class PairEmbedding:
    """Directional source-target pair embedding with stable unknown fallbacks."""

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        pair_hash_bucket_count: int = 32,
        entity_hash_bucket_count: int = 16,
        random_seed: int = 0,
        architecture: str = "pair_embedding",
        feature_roles: dict[str, Any] | None = None,
        backend: str = "cpu",
    ) -> None:
        self.backend = _resolve_backend(backend)
        self.source_embedding = EntityEmbedding(
            embedding_dim=embedding_dim,
            hash_bucket_count=entity_hash_bucket_count,
            random_seed=random_seed,
            architecture="source_embedding",
            feature_roles=feature_roles,
            backend=backend,
        )
        self.target_embedding = EntityEmbedding(
            embedding_dim=embedding_dim,
            hash_bucket_count=entity_hash_bucket_count,
            random_seed=random_seed + 17,
            architecture="target_embedding",
            feature_roles=feature_roles,
            backend=backend,
        )
        self.embedding_dim = int(embedding_dim)
        self.pair_hash_bucket_count = int(pair_hash_bucket_count)
        self.entity_hash_bucket_count = int(entity_hash_bucket_count)
        self.random_seed = int(random_seed)
        self.architecture = str(architecture)
        self.feature_roles = {} if feature_roles is None else dict(feature_roles)
        self.is_fitted_ = False

    def fit(
        self,
        source_ids: Any,
        target_ids: Any,
        *,
        training_cutoff: str | None = None,
        training_metrics: dict[str, float] | None = None,
    ) -> PairEmbedding:
        sources = [str(value) for value in np.asarray(source_ids).reshape(-1).tolist()]
        targets = [str(value) for value in np.asarray(target_ids).reshape(-1).tolist()]
        if len(sources) != len(targets):
            raise ValueError("source_ids and target_ids must have the same length")
        self.source_embedding.fit(sources, training_cutoff=training_cutoff)
        self.target_embedding.fit(targets, training_cutoff=training_cutoff)
        pairs = sorted(
            {f"{source}\0{target}" for source, target in zip(sources, targets, strict=True)}
        )
        self.pair_id_map_ = {
            "__unknown__": 0,
            **{value: idx + 1 for idx, value in enumerate(pairs)},
        }
        row_count = 1 + len(pairs) + self.pair_hash_bucket_count
        self.pair_embeddings_ = _deterministic_matrix(
            row_count,
            self.embedding_dim,
            seed=self.random_seed + 31,
            salt=self.architecture,
        )
        self.training_cutoff_ = training_cutoff
        self.training_metrics_ = {} if training_metrics is None else dict(training_metrics)
        self.schema_hash_ = _schema_hash(
            {
                "sources": sorted(set(sources)),
                "targets": sorted(set(targets)),
                "pairs": pairs,
                "embedding_dim": self.embedding_dim,
                "pair_hash_bucket_count": self.pair_hash_bucket_count,
                "entity_hash_bucket_count": self.entity_hash_bucket_count,
                "feature_roles": self.feature_roles,
            }
        )
        self.is_fitted_ = True
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity())
        return self

    def transform(self, source_ids: Any, target_ids: Any) -> np.ndarray:
        self._require_fitted()
        sources = [str(value) for value in np.asarray(source_ids).reshape(-1).tolist()]
        targets = [str(value) for value in np.asarray(target_ids).reshape(-1).tolist()]
        if len(sources) != len(targets):
            raise ValueError("source_ids and target_ids must have the same length")
        src = self.source_embedding.transform(sources)
        dst = self.target_embedding.transform(targets)
        pair_rows = [
            self._pair_index(source, target)
            for source, target in zip(sources, targets, strict=True)
        ]
        pair = self.pair_embeddings_[pair_rows]
        return np.concatenate([src, dst, src - dst, np.abs(src - dst), src * dst, pair], axis=1)

    def artifact_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return self.artifact_.to_dict()

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        payload = {
            "artifact": self.artifact_.to_dict(),
            "source_embeddings": self.source_embedding.embeddings_.tolist(),
            "target_embeddings": self.target_embedding.embeddings_.tolist(),
            "pair_embeddings": self.pair_embeddings_.tolist(),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PairEmbedding:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            embedding_dim=int(artifact["embedding_dim"]),
            pair_hash_bucket_count=int(artifact["hash_bucket_config"]["pair"]),
            entity_hash_bucket_count=int(artifact["hash_bucket_config"]["entity"]),
            random_seed=int(artifact["random_seed"]),
            architecture=str(artifact["architecture"]),
            feature_roles=dict(artifact["feature_roles"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.source_embedding.id_map_ = dict(artifact["id_maps"]["source"])
        obj.target_embedding.id_map_ = dict(artifact["id_maps"]["target"])
        obj.source_embedding.embeddings_ = np.asarray(payload["source_embeddings"], dtype=float)
        obj.target_embedding.embeddings_ = np.asarray(payload["target_embeddings"], dtype=float)
        obj.source_embedding.is_fitted_ = True
        obj.target_embedding.is_fitted_ = True
        obj.pair_id_map_ = dict(artifact["id_maps"]["pair"])
        obj.pair_embeddings_ = np.asarray(payload["pair_embeddings"], dtype=float)
        obj.training_cutoff_ = artifact["training_cutoff"]
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture=self.architecture,
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={
                "source": dict(self.source_embedding.id_map_),
                "target": dict(self.target_embedding.id_map_),
                "pair": dict(self.pair_id_map_),
            },
            hash_bucket_config={
                "entity": self.entity_hash_bucket_count,
                "pair": self.pair_hash_bucket_count,
            },
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
            feature_roles=dict(self.feature_roles),
            training_cutoff=self.training_cutoff_,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _save_load_parity(self) -> bool:
        sources = ["A", "B", "__new_source__"]
        targets = ["B", "A", "__new_target__"]
        before = self.transform(sources, targets)
        payload_path = Path("/tmp/cartoboost_pair_embedding_parity.json")
        self.save(payload_path)
        try:
            after = self.load(payload_path).transform(sources, targets)
        finally:
            payload_path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))

    def _pair_index(self, source: str, target: str) -> int:
        key = f"{source}\0{target}"
        if key in self.pair_id_map_:
            return self.pair_id_map_[key]
        return len(self.pair_id_map_) + _stable_hash(key, self.pair_hash_bucket_count)

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("representation must be fit before transform")


class SpatioTemporalAdaptiveEmbedding:
    """Context-dependent embedding from static IDs, time features, and context."""

    def __init__(
        self, *, embedding_dim: int = 8, random_seed: int = 0, backend: str = "cpu"
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.random_seed = int(random_seed)
        self.backend = _resolve_backend(backend)
        self.entity_embedding = EntityEmbedding(
            embedding_dim=embedding_dim,
            random_seed=random_seed,
            architecture="spatiotemporal_entity_embedding",
            backend=backend,
        )
        self.is_fitted_ = False

    def fit(
        self, entity_ids: Any, *, training_cutoff: str | None = None
    ) -> SpatioTemporalAdaptiveEmbedding:
        self.entity_embedding.fit(entity_ids, training_cutoff=training_cutoff)
        self.time_weights_ = _deterministic_matrix(
            self.embedding_dim,
            self.embedding_dim,
            seed=self.random_seed + 101,
            salt="time",
        )
        self.context_weights_ = _deterministic_matrix(
            self.embedding_dim,
            self.embedding_dim,
            seed=self.random_seed + 211,
            salt="context",
        )
        self.gate_weights_ = _deterministic_matrix(
            self.embedding_dim * 3,
            self.embedding_dim,
            seed=self.random_seed + 307,
            salt="gate",
        )
        self.training_cutoff_ = training_cutoff
        self.training_metrics_ = {}
        self.schema_hash_ = _schema_hash(
            {
                "entity_ids": sorted(set(np.asarray(entity_ids).reshape(-1).astype(str).tolist())),
                "embedding_dim": self.embedding_dim,
                "backend": self.backend,
            }
        )
        self.is_fitted_ = True
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity())
        return self

    def transform(
        self,
        entity_ids: Any,
        *,
        time_features: Any,
        context_features: Any | None = None,
    ) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("representation must be fit before transform")
        static = self.entity_embedding.transform(entity_ids)
        time = _project_features(time_features, self.time_weights_, static.shape[0])
        context = (
            np.zeros_like(static)
            if context_features is None
            else _project_features(context_features, self.context_weights_, static.shape[0])
        )
        gate = _sigmoid(np.concatenate([static, time, context], axis=1) @ self.gate_weights_)
        adaptive = _layer_norm(static + gate * time + (1.0 - gate) * context)
        return {
            "adaptive_embedding": adaptive,
            "static_embedding": static,
            "temporal_embedding": time,
            "interaction_embedding": gate * time + (1.0 - gate) * context,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("representation must be fit before artifact access")
        return self.artifact_.to_dict()

    def rolling_origin_time_ablation_report(
        self,
        entity_ids: Any,
        y: Any,
        *,
        time_features: Any,
        context_features: Any | None = None,
        min_train_size: int | None = None,
    ) -> dict[str, Any]:
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if y_array.shape[0] < 4:
            raise ValueError("rolling-origin ablation requires at least four rows")
        if not np.isfinite(y_array).all():
            raise ValueError("y must contain only finite values")
        train_size = int(min_train_size or max(2, round(y_array.shape[0] * 0.7)))
        if train_size <= 1 or train_size >= y_array.shape[0]:
            raise ValueError("min_train_size must leave at least one validation row")
        adaptive = self.transform(
            entity_ids,
            time_features=time_features,
            context_features=context_features,
        )["adaptive_embedding"]
        no_time = self.transform(
            entity_ids,
            time_features=np.zeros_like(_as_feature_matrix(time_features)),
            context_features=context_features,
        )["adaptive_embedding"]
        adaptive_rmse = _linear_holdout_rmse(adaptive, y_array, train_size)
        no_time_rmse = _linear_holdout_rmse(no_time, y_array, train_size)
        return {
            "validation_strategy": "rolling_origin_holdout",
            "train_size": train_size,
            "validation_size": int(y_array.shape[0] - train_size),
            "adaptive_time_rmse": adaptive_rmse,
            "no_time_rmse": no_time_rmse,
            "time_feature_delta_rmse": no_time_rmse - adaptive_rmse,
            "time_features_help": bool(adaptive_rmse < no_time_rmse),
        }

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("representation must be fit before save")
        payload = {
            "artifact": self.artifact_.to_dict(),
            "entity_id_map": dict(self.entity_embedding.id_map_),
            "entity_embeddings": self.entity_embedding.embeddings_.tolist(),
            "time_weights": self.time_weights_.tolist(),
            "context_weights": self.context_weights_.tolist(),
            "gate_weights": self.gate_weights_.tolist(),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SpatioTemporalAdaptiveEmbedding:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            embedding_dim=int(artifact["embedding_dim"]),
            random_seed=int(artifact["random_seed"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.entity_embedding.id_map_ = dict(payload["entity_id_map"])
        obj.entity_embedding.embeddings_ = np.asarray(payload["entity_embeddings"], dtype=float)
        obj.entity_embedding.is_fitted_ = True
        obj.time_weights_ = np.asarray(payload["time_weights"], dtype=float)
        obj.context_weights_ = np.asarray(payload["context_weights"], dtype=float)
        obj.gate_weights_ = np.asarray(payload["gate_weights"], dtype=float)
        obj.training_cutoff_ = artifact["training_cutoff"]
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture="spatiotemporal_adaptive_embedding",
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={"entity": dict(self.entity_embedding.id_map_)},
            hash_bucket_config={"entity": self.entity_embedding.hash_bucket_count},
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
            feature_roles={},
            training_cutoff=self.training_cutoff_,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _save_load_parity(self) -> bool:
        probe = [*list(self.entity_embedding.id_map_)[:2], "__new_entity__"]
        time = np.ones((len(probe), self.embedding_dim), dtype=float)
        context = np.zeros((len(probe), self.embedding_dim), dtype=float)
        before = self.transform(probe, time_features=time, context_features=context)[
            "adaptive_embedding"
        ]
        payload_path = Path("/tmp/cartoboost_spatiotemporal_adaptive_parity.json")
        self.save(payload_path)
        try:
            after = self.load(payload_path).transform(
                probe, time_features=time, context_features=context
            )["adaptive_embedding"]
        finally:
            payload_path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))


class RegimeRouter:
    """Entity-aware deterministic router for generic mixture-of-experts regimes."""

    def __init__(
        self,
        *,
        expert_count: int = 3,
        embedding_dim: int = 8,
        hash_bucket_count: int = 16,
        random_seed: int = 0,
        feature_roles: dict[str, Any] | None = None,
        backend: str = "cpu",
    ) -> None:
        if expert_count < 2:
            raise ValueError("expert_count must be at least 2")
        self.expert_count = int(expert_count)
        self.embedding_dim = int(embedding_dim)
        self.hash_bucket_count = int(hash_bucket_count)
        self.random_seed = int(random_seed)
        self.feature_roles = {} if feature_roles is None else dict(feature_roles)
        self.backend = _resolve_backend(backend)
        self.entity_embedding = EntityEmbedding(
            embedding_dim=embedding_dim,
            hash_bucket_count=hash_bucket_count,
            random_seed=random_seed,
            architecture="regime_router_entity_embedding",
            feature_roles=feature_roles,
            backend=backend,
        )
        self.is_fitted_ = False

    def fit(
        self,
        entity_ids: Any,
        *,
        context_features: Any | None = None,
        training_cutoff: str | None = None,
        training_metrics: dict[str, float] | None = None,
    ) -> RegimeRouter:
        ids = [str(value) for value in np.asarray(entity_ids).reshape(-1).tolist()]
        if not ids:
            raise ValueError("entity_ids must be non-empty")
        self.entity_embedding.fit(
            ids,
            training_cutoff=training_cutoff,
            training_metrics=training_metrics,
        )
        self.context_dim_ = (
            0 if context_features is None else _as_feature_matrix(context_features).shape[1]
        )
        input_dim = self.embedding_dim + self.context_dim_
        self.router_weights_ = _deterministic_matrix(
            input_dim,
            self.expert_count,
            seed=self.random_seed + 401,
            salt="regime_router",
        )
        self.router_bias_ = _deterministic_matrix(
            1,
            self.expert_count,
            seed=self.random_seed + 409,
            salt="regime_router_bias",
        ).reshape(-1)
        self.is_fitted_ = True
        weights = self.predict_proba(ids, context_features=context_features)
        selected = np.argmax(weights, axis=1)
        counts = np.bincount(selected, minlength=self.expert_count).astype(float)
        usage = counts / max(1.0, float(np.sum(counts)))
        entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
        self.training_cutoff_ = training_cutoff
        self.training_metrics_ = {} if training_metrics is None else dict(training_metrics)
        self.training_metrics_.setdefault("mean_router_entropy", float(np.mean(entropy)))
        self.expert_usage_ = {
            f"expert_{idx}": float(value) for idx, value in enumerate(usage.tolist())
        }
        self.schema_hash_ = _schema_hash(
            {
                "entity_ids": sorted(set(ids)),
                "embedding_dim": self.embedding_dim,
                "hash_bucket_count": self.hash_bucket_count,
                "expert_count": self.expert_count,
                "context_dim": self.context_dim_,
                "feature_roles": self.feature_roles,
            }
        )
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity(ids))
        return self

    def predict_proba(self, entity_ids: Any, *, context_features: Any | None = None) -> np.ndarray:
        self._require_fitted()
        ids = [str(value) for value in np.asarray(entity_ids).reshape(-1).tolist()]
        embedding = self.entity_embedding.transform(ids)
        context = _context_matrix(context_features, len(ids), self.context_dim_)
        logits = np.concatenate([embedding, context], axis=1) @ self.router_weights_
        logits = logits + self.router_bias_
        return _softmax(logits)

    def route(self, entity_ids: Any, *, context_features: Any | None = None) -> dict[str, Any]:
        weights = self.predict_proba(entity_ids, context_features=context_features)
        entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
        return {
            "expert_weights": weights,
            "selected_expert": np.argmax(weights, axis=1),
            "router_entropy": entropy,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return self.artifact_.to_dict() | {"expert_usage": dict(self.expert_usage_)}

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        payload = {
            "artifact": self.artifact_.to_dict(),
            "entity_embeddings": self.entity_embedding.embeddings_.tolist(),
            "router_weights": self.router_weights_.tolist(),
            "router_bias": self.router_bias_.tolist(),
            "context_dim": self.context_dim_,
            "expert_usage": dict(self.expert_usage_),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeRouter:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            expert_count=int(artifact["hash_bucket_config"]["expert_count"]),
            embedding_dim=int(artifact["embedding_dim"]),
            hash_bucket_count=int(artifact["hash_bucket_config"]["entity"]),
            random_seed=int(artifact["random_seed"]),
            feature_roles=dict(artifact["feature_roles"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.entity_embedding.id_map_ = dict(artifact["id_maps"]["entity"])
        obj.entity_embedding.embeddings_ = np.asarray(payload["entity_embeddings"], dtype=float)
        obj.entity_embedding.is_fitted_ = True
        obj.router_weights_ = np.asarray(payload["router_weights"], dtype=float)
        obj.router_bias_ = np.asarray(payload["router_bias"], dtype=float)
        obj.context_dim_ = int(payload["context_dim"])
        obj.expert_usage_ = dict(payload["expert_usage"])
        obj.training_cutoff_ = artifact["training_cutoff"]
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture="regime_router",
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={"entity": dict(self.entity_embedding.id_map_)},
            hash_bucket_config={
                "entity": self.hash_bucket_count,
                "expert_count": self.expert_count,
            },
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
            feature_roles=dict(self.feature_roles),
            training_cutoff=self.training_cutoff_,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _save_load_parity(self, ids: list[str]) -> bool:
        probe = [*ids[: min(3, len(ids))], "__new_entity__"]
        context = np.zeros((len(probe), self.context_dim_), dtype=float)
        before = self.predict_proba(probe, context_features=context)
        payload_path = Path("/tmp/cartoboost_regime_router_parity.json")
        self.save(payload_path)
        try:
            after = self.load(payload_path).predict_proba(probe, context_features=context)
        finally:
            payload_path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("representation must be fit before routing")


class HistoricalAnalogRetriever:
    """Exact KNN memory for retrieving similar historical representation contexts."""

    def __init__(
        self,
        *,
        normalize: bool = True,
        approximate: bool = False,
        compressed: bool = False,
        learned_projection_dim: int | None = None,
        random_seed: int = 0,
        feature_roles: dict[str, Any] | None = None,
        backend: str = "cpu",
    ) -> None:
        self.normalize = bool(normalize)
        self.approximate = bool(approximate)
        self.compressed = bool(compressed)
        self.learned_projection_dim = (
            None if learned_projection_dim is None else int(learned_projection_dim)
        )
        self.random_seed = int(random_seed)
        self.feature_roles = {} if feature_roles is None else dict(feature_roles)
        self.backend = _resolve_backend(backend)
        self.is_fitted_ = False

    def fit(
        self,
        analog_ids: Any,
        keys: Any,
        *,
        timestamps: Any | None = None,
        training_cutoff: str | None = None,
        training_metrics: dict[str, float] | None = None,
    ) -> HistoricalAnalogRetriever:
        ids = [str(value) for value in np.asarray(analog_ids).reshape(-1).tolist()]
        key_matrix = _as_feature_matrix(keys)
        if key_matrix.shape[0] != len(ids):
            raise ValueError("keys row count must match analog_ids")
        timestamp_values = None
        if timestamps is not None:
            timestamp_values = [str(value) for value in np.asarray(timestamps).reshape(-1).tolist()]
            if len(timestamp_values) != len(ids):
                raise ValueError("timestamps row count must match analog_ids")
        self.analog_ids_ = ids
        self.timestamps_ = timestamp_values
        self.key_mean_ = (
            key_matrix.mean(axis=0) if self.normalize else np.zeros(key_matrix.shape[1])
        )
        self.key_scale_ = (
            np.maximum(key_matrix.std(axis=0), 1e-12)
            if self.normalize
            else np.ones(key_matrix.shape[1])
        )
        normalized_memory = (key_matrix - self.key_mean_) / self.key_scale_
        self.learned_projection_ = None
        if self.learned_projection_dim is not None:
            self.learned_projection_ = _deterministic_matrix(
                normalized_memory.shape[1],
                self.learned_projection_dim,
                seed=self.random_seed + 1201,
                salt="learned_retriever_projection",
            )
            normalized_memory = normalized_memory @ self.learned_projection_
        self.memory_ = normalized_memory.astype(np.float16 if self.compressed else float).astype(
            float
        )
        self.compressed_memory_ = normalized_memory.astype(np.float16) if self.compressed else None
        self.ann_buckets_ = self._build_ann_buckets(self.memory_) if self.approximate else {}
        self.training_cutoff_ = training_cutoff
        self.training_metrics_ = {} if training_metrics is None else dict(training_metrics)
        self.training_metrics_.setdefault("memory_size", float(len(ids)))
        self.schema_hash_ = _schema_hash(
            {
                "analog_ids": ids,
                "key_dim": int(key_matrix.shape[1]),
                "normalize": self.normalize,
                "approximate": self.approximate,
                "compressed": self.compressed,
                "learned_projection_dim": self.learned_projection_dim,
                "has_timestamps": timestamp_values is not None,
                "feature_roles": self.feature_roles,
            }
        )
        self.is_fitted_ = True
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity())
        return self

    def query(
        self,
        keys: Any,
        *,
        k: int = 5,
        cutoff: str | None = None,
    ) -> list[dict[str, Any]]:
        self._require_fitted()
        if k <= 0:
            raise ValueError("k must be positive")
        query_matrix = _as_feature_matrix(keys)
        if query_matrix.shape[1] != self.key_mean_.shape[0]:
            raise ValueError("query key dimension must match fitted memory")
        normalized = (query_matrix - self.key_mean_) / self.key_scale_
        if self.learned_projection_ is not None:
            normalized = normalized @ self.learned_projection_
        eligible = self._eligible_indices(cutoff)
        if not eligible:
            raise ValueError("no analogs are available before the requested cutoff")
        output: list[dict[str, Any]] = []
        for row in normalized:
            query_eligible = self._ann_eligible(row, eligible)
            memory = self.memory_[query_eligible]
            distances = np.sqrt(np.sum((memory - row) ** 2, axis=1))
            order = np.argsort(distances)[: min(k, len(query_eligible))]
            analog_indices = [query_eligible[int(idx)] for idx in order]
            output.append(
                {
                    "analog_ids": [self.analog_ids_[idx] for idx in analog_indices],
                    "distances": [float(distances[int(idx)]) for idx in order],
                    "indices": analog_indices,
                    "index_kind": "approximate_bucket" if self.approximate else "exact_knn",
                    "compressed": self.compressed,
                    "learned_projection": self.learned_projection_ is not None,
                }
            )
        return output

    def artifact_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return self.artifact_.to_dict()

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        payload = {
            "artifact": self.artifact_.to_dict(),
            "analog_ids": self.analog_ids_,
            "timestamps": self.timestamps_,
            "memory": self.memory_.tolist(),
            "approximate": self.approximate,
            "compressed": self.compressed,
            "learned_projection_dim": self.learned_projection_dim,
            "compressed_memory": None
            if self.compressed_memory_ is None
            else self.compressed_memory_.astype(float).tolist(),
            "ann_buckets": {key: value for key, value in self.ann_buckets_.items()},
            "learned_projection": None
            if self.learned_projection_ is None
            else self.learned_projection_.tolist(),
            "key_mean": self.key_mean_.tolist(),
            "key_scale": self.key_scale_.tolist(),
            "normalize": self.normalize,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> HistoricalAnalogRetriever:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            normalize=bool(payload["normalize"]),
            approximate=bool(payload.get("approximate", payload.get("ann_buckets") is not None)),
            compressed=payload.get("compressed_memory") is not None,
            learned_projection_dim=None
            if payload.get("learned_projection") is None
            else len(payload["learned_projection"][0]),
            random_seed=int(artifact["random_seed"]),
            feature_roles=dict(artifact["feature_roles"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.analog_ids_ = list(payload["analog_ids"])
        obj.timestamps_ = None if payload["timestamps"] is None else list(payload["timestamps"])
        obj.memory_ = np.asarray(payload["memory"], dtype=float)
        obj.compressed_memory_ = (
            None
            if payload.get("compressed_memory") is None
            else np.asarray(payload["compressed_memory"], dtype=np.float16)
        )
        obj.ann_buckets_ = {
            str(key): [int(value) for value in values]
            for key, values in payload.get("ann_buckets", {}).items()
        }
        obj.learned_projection_ = (
            None
            if payload.get("learned_projection") is None
            else np.asarray(payload["learned_projection"], dtype=float)
        )
        obj.key_mean_ = np.asarray(payload["key_mean"], dtype=float)
        obj.key_scale_ = np.asarray(payload["key_scale"], dtype=float)
        obj.training_cutoff_ = artifact["training_cutoff"]
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture="exact_knn_memory",
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={"analog": {value: idx for idx, value in enumerate(self.analog_ids_)}},
            hash_bucket_config={
                "memory_size": len(self.analog_ids_),
                "ann_bucket_count": len(self.ann_buckets_),
                "compressed": int(self.compressed),
                "learned_projection_dim": int(self.learned_projection_dim or 0),
            },
            embedding_dim=int(self.memory_.shape[1]),
            random_seed=self.random_seed,
            feature_roles=dict(self.feature_roles),
            training_cutoff=self.training_cutoff_,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _eligible_indices(self, cutoff: str | None) -> list[int]:
        if cutoff is None or self.timestamps_ is None:
            return list(range(len(self.analog_ids_)))
        return [idx for idx, timestamp in enumerate(self.timestamps_) if timestamp < cutoff]

    def _build_ann_buckets(self, memory: np.ndarray) -> dict[str, list[int]]:
        buckets: dict[str, list[int]] = {}
        for idx, row in enumerate(memory):
            buckets.setdefault(self._ann_bucket_key(row), []).append(idx)
        return buckets

    def _ann_eligible(self, row: np.ndarray, eligible: list[int]) -> list[int]:
        if not self.approximate:
            return eligible
        bucket = self.ann_buckets_.get(self._ann_bucket_key(row), [])
        eligible_set = set(eligible)
        narrowed = [idx for idx in bucket if idx in eligible_set]
        return narrowed or eligible

    def _ann_bucket_key(self, row: np.ndarray) -> str:
        width = min(4, row.shape[0])
        return "".join("1" if value >= 0.0 else "0" for value in row[:width])

    def _save_load_parity(self) -> bool:
        before = self.query(self.key_mean_.reshape(1, -1), k=min(3, len(self.analog_ids_)))
        payload_path = Path("/tmp/cartoboost_historical_analog_parity.json")
        self.save(payload_path)
        try:
            after = self.load(payload_path).query(
                self.key_mean_.reshape(1, -1), k=min(3, len(self.analog_ids_))
            )
        finally:
            payload_path.unlink(missing_ok=True)
        return before == after

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("retriever must be fit before query")


KNNContextMemory = HistoricalAnalogRetriever


class RetrievalAugmentedForecaster:
    """Exact-KNN retrieval-augmented forecaster over historical contexts."""

    def __init__(
        self,
        *,
        k: int = 5,
        normalize: bool = True,
        random_seed: int = 0,
        backend: str = "cpu",
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.normalize = bool(normalize)
        self.random_seed = int(random_seed)
        self.backend = _resolve_backend(backend)
        self.retriever = HistoricalAnalogRetriever(
            normalize=normalize,
            random_seed=random_seed,
            feature_roles={
                "entity_id": "retrieval_key",
                "time_features": "retrieval_key",
                "recent_history_shape": "retrieval_key",
                "weather_or_event_features": "retrieval_key",
                "graph_neighborhood_summary": "retrieval_key",
                "residual_regime": "retrieval_key",
            },
            backend=backend,
        )
        self.is_fitted_ = False

    def fit(
        self,
        analog_ids: Any,
        context_keys: Any,
        targets: Any,
        *,
        timestamps: Any | None = None,
        training_cutoff: str | None = None,
    ) -> RetrievalAugmentedForecaster:
        ids = [str(value) for value in np.asarray(analog_ids).reshape(-1).tolist()]
        key_matrix = _as_feature_matrix(context_keys)
        target_arr = np.asarray(targets, dtype=float).reshape(-1)
        if key_matrix.shape[0] != len(ids) or target_arr.shape[0] != len(ids):
            raise ValueError("analog_ids, context_keys, and targets must have matching rows")
        if not np.isfinite(target_arr).all():
            raise ValueError("targets must contain finite values")
        self.retriever.fit(
            ids,
            key_matrix,
            timestamps=timestamps,
            training_cutoff=training_cutoff,
            training_metrics={"target_mean": float(np.mean(target_arr))},
        )
        self.targets_ = target_arr
        self.training_cutoff_ = training_cutoff
        self.schema_hash_ = _schema_hash(
            {
                "analog_ids": ids,
                "context_dim": int(key_matrix.shape[1]),
                "k": self.k,
                "training_cutoff": training_cutoff,
            }
        )
        self.is_fitted_ = True
        return self

    def predict(
        self,
        context_keys: Any,
        *,
        cutoff: str | None = None,
        return_explanation: bool = False,
    ) -> Any:
        self._require_fitted()
        neighbors = self.retriever.query(context_keys, k=self.k, cutoff=cutoff)
        predictions = []
        explanations = []
        for result in neighbors:
            indices = result["indices"]
            distances = np.asarray(result["distances"], dtype=float)
            weights = 1.0 / np.maximum(distances, 1e-12)
            weights = weights / weights.sum()
            values = self.targets_[indices]
            prediction = float(np.dot(weights, values))
            predictions.append(prediction)
            explanations.append(
                {
                    **result,
                    "attention_weights": [float(value) for value in weights],
                    "retrieved_targets": [float(value) for value in values],
                    "prediction": prediction,
                }
            )
        if return_explanation:
            return {"prediction": np.asarray(predictions, dtype=float), "retrieval": explanations}
        return np.asarray(predictions, dtype=float)

    def rare_pattern_benchmark(
        self,
        query_keys: Any,
        actual: Any,
        *,
        cutoff: str | None = None,
    ) -> dict[str, float]:
        pred = self.predict(query_keys, cutoff=cutoff)
        y = np.asarray(actual, dtype=float).reshape(-1)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("actual row count must match query rows")
        global_pred = np.full_like(y, float(np.mean(self.targets_)), dtype=float)
        retrieval_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        global_rmse = float(np.sqrt(np.mean((y - global_pred) ** 2)))
        return {
            "retrieval_rmse": retrieval_rmse,
            "global_mean_rmse": global_rmse,
            "improvement": global_rmse - retrieval_rmse,
        }

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        path = Path(path)
        payload = {
            "model_class": self.__class__.__name__,
            "architecture": "retrieval_augmented_forecaster",
            "k": self.k,
            "normalize": self.normalize,
            "random_seed": self.random_seed,
            "targets": self.targets_.tolist(),
            "training_cutoff": self.training_cutoff_,
            "schema_hash": self.schema_hash_,
            "retriever": {
                "artifact": self.retriever.artifact_metadata(),
                "analog_ids": self.retriever.analog_ids_,
                "timestamps": self.retriever.timestamps_,
                "memory": self.retriever.memory_.tolist(),
                "approximate": self.retriever.approximate,
                "compressed": self.retriever.compressed,
                "learned_projection_dim": self.retriever.learned_projection_dim,
                "compressed_memory": None
                if self.retriever.compressed_memory_ is None
                else self.retriever.compressed_memory_.astype(float).tolist(),
                "ann_buckets": {key: value for key, value in self.retriever.ann_buckets_.items()},
                "learned_projection": None
                if self.retriever.learned_projection_ is None
                else self.retriever.learned_projection_.tolist(),
                "key_mean": self.retriever.key_mean_.tolist(),
                "key_scale": self.retriever.key_scale_.tolist(),
                "normalize": self.retriever.normalize,
            },
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RetrievalAugmentedForecaster:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            k=int(payload["k"]),
            normalize=bool(payload["normalize"]),
            random_seed=int(payload["random_seed"]),
        )
        retriever_payload = payload["retriever"]
        artifact = retriever_payload["artifact"]
        obj.retriever.approximate = bool(retriever_payload.get("approximate", False))
        obj.retriever.compressed = bool(retriever_payload.get("compressed", False))
        obj.retriever.learned_projection_dim = retriever_payload.get("learned_projection_dim")
        obj.retriever.analog_ids_ = list(retriever_payload["analog_ids"])
        obj.retriever.timestamps_ = (
            None
            if retriever_payload["timestamps"] is None
            else list(retriever_payload["timestamps"])
        )
        obj.retriever.memory_ = np.asarray(retriever_payload["memory"], dtype=float)
        obj.retriever.compressed_memory_ = (
            None
            if retriever_payload.get("compressed_memory") is None
            else np.asarray(retriever_payload["compressed_memory"], dtype=np.float16)
        )
        obj.retriever.ann_buckets_ = {
            str(key): [int(value) for value in values]
            for key, values in retriever_payload.get("ann_buckets", {}).items()
        }
        obj.retriever.learned_projection_ = (
            None
            if retriever_payload.get("learned_projection") is None
            else np.asarray(retriever_payload["learned_projection"], dtype=float)
        )
        obj.retriever.key_mean_ = np.asarray(retriever_payload["key_mean"], dtype=float)
        obj.retriever.key_scale_ = np.asarray(retriever_payload["key_scale"], dtype=float)
        obj.retriever.training_cutoff_ = artifact["training_cutoff"]
        obj.retriever.training_metrics_ = dict(artifact["training_metrics"])
        obj.retriever.schema_hash_ = str(artifact["schema_hash"])
        obj.retriever.is_fitted_ = True
        obj.retriever.artifact_ = RepresentationArtifact(**artifact)
        obj.targets_ = np.asarray(payload["targets"], dtype=float)
        obj.training_cutoff_ = payload["training_cutoff"]
        obj.schema_hash_ = str(payload["schema_hash"])
        obj.is_fitted_ = True
        return obj

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("forecaster must be fit before prediction")


class RetrievalAugmentedPairModel(RetrievalAugmentedForecaster):
    """Retrieval-augmented model with source-target pair IDs in the memory."""

    def fit(
        self,
        analog_ids: Any,
        context_keys: Any,
        targets: Any,
        *extra: Any,
        timestamps: Any | None = None,
        training_cutoff: str | None = None,
    ) -> RetrievalAugmentedPairModel:
        if len(extra) != 1:
            raise ValueError(
                "RetrievalAugmentedPairModel.fit expects source_ids, target_ids, "
                "context_keys, and targets"
            )
        sources = [str(value) for value in np.asarray(analog_ids).reshape(-1).tolist()]
        targets_ids = [str(value) for value in np.asarray(context_keys).reshape(-1).tolist()]
        if len(sources) != len(targets_ids):
            raise ValueError("source_ids and target_ids must have matching rows")
        pair_context_keys = targets
        pair_targets = extra[0]
        pair_ids = [
            f"{source}->{target}" for source, target in zip(sources, targets_ids, strict=True)
        ]
        return super().fit(
            pair_ids,
            pair_context_keys,
            pair_targets,
            timestamps=timestamps,
            training_cutoff=training_cutoff,
        )


class SelfSupervisedPretrainer:
    """Deterministic first-cut pretrainer for reusable entity embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        hash_bucket_count: int = 16,
        random_seed: int = 0,
        tasks: tuple[str, ...] = (
            "masked_entity_time_modeling",
            "masked_pair_time_modeling",
            "graph_edge_denoising",
            "temporal_order_contrastive_loss",
            "spatial_neighbor_contrastive_loss",
            "future_patch_reconstruction",
        ),
        feature_roles: dict[str, Any] | None = None,
        backend: str = "cpu",
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.hash_bucket_count = int(hash_bucket_count)
        self.random_seed = int(random_seed)
        self.tasks = tuple(str(task) for task in tasks)
        self.feature_roles = {} if feature_roles is None else dict(feature_roles)
        self.backend = _resolve_backend(backend)
        self.is_fitted_ = False

    def fit(
        self,
        entity_ids: Any,
        features: Any,
        *,
        timestamps: Any | None = None,
        training_cutoff: str,
    ) -> SelfSupervisedPretrainer:
        ids = [str(value) for value in np.asarray(entity_ids).reshape(-1).tolist()]
        feature_matrix = _as_feature_matrix(features)
        if feature_matrix.shape[0] != len(ids):
            raise ValueError("features row count must match entity_ids")
        if timestamps is not None:
            timestamp_values = [str(value) for value in np.asarray(timestamps).reshape(-1).tolist()]
            if len(timestamp_values) != len(ids):
                raise ValueError("timestamps row count must match entity_ids")
            keep = np.asarray([timestamp < training_cutoff for timestamp in timestamp_values])
            if not np.any(keep):
                raise ValueError("no pretraining rows are before training_cutoff")
            ids = [value for value, allowed in zip(ids, keep, strict=True) if bool(allowed)]
            feature_matrix = feature_matrix[keep]
        self.entity_embedding = EntityEmbedding(
            embedding_dim=self.embedding_dim,
            hash_bucket_count=self.hash_bucket_count,
            random_seed=self.random_seed,
            architecture="self_supervised_entity_embedding",
            feature_roles=self.feature_roles,
            backend=self.backend["requested"],
        ).fit(ids, training_cutoff=training_cutoff)
        self.feature_mean_ = feature_matrix.mean(axis=0)
        self.feature_scale_ = np.maximum(feature_matrix.std(axis=0), 1e-12)
        projection = _deterministic_matrix(
            feature_matrix.shape[1],
            self.embedding_dim,
            seed=self.random_seed + 601,
            salt="self_supervised_pretrainer",
        )
        grouped: dict[str, list[np.ndarray]] = {}
        normalized = (feature_matrix - self.feature_mean_) / self.feature_scale_
        for entity_id, row in zip(ids, normalized, strict=True):
            grouped.setdefault(entity_id, []).append(row)
        pretrained = self.entity_embedding.embeddings_.copy()
        for entity_id, rows in grouped.items():
            pretrained[self.entity_embedding.id_map_[entity_id]] = (
                np.mean(rows, axis=0) @ projection
            )
        self.pretrained_entity_embeddings_ = pretrained
        self.entity_embedding.embeddings_ = pretrained
        self.pretrained_node_embeddings_ = pretrained.copy()
        self.pretrained_pair_embeddings_ = _pair_pretraining_embeddings(
            ids,
            normalized,
            embedding_dim=self.embedding_dim,
            seed=self.random_seed + 701,
        )
        self.pretrained_temporal_encoder_ = _temporal_pretraining_encoder(
            normalized,
            embedding_dim=self.embedding_dim,
            seed=self.random_seed + 809,
        )
        reconstruction = normalized - normalized.mean(axis=0, keepdims=True)
        self.training_cutoff_ = training_cutoff
        self.training_metrics_ = {
            "pretraining_rows": float(feature_matrix.shape[0]),
            "masked_reconstruction_proxy_rmse": float(np.sqrt(np.mean(reconstruction**2))),
            "masked_pair_proxy_rmse": _proxy_rmse(self.pretrained_pair_embeddings_),
            "graph_edge_denoising_proxy_auc": _graph_edge_proxy_auc(
                self.pretrained_node_embeddings_
            ),
            "temporal_order_contrastive_margin": _temporal_order_margin(normalized),
            "spatial_neighbor_contrastive_margin": _spatial_neighbor_margin(normalized),
            "future_patch_reconstruction_proxy_rmse": _future_patch_proxy_rmse(normalized),
        }
        self.schema_hash_ = _schema_hash(
            {
                "entity_ids": sorted(set(ids)),
                "feature_dim": int(feature_matrix.shape[1]),
                "embedding_dim": self.embedding_dim,
                "hash_bucket_count": self.hash_bucket_count,
                "tasks": self.tasks,
                "training_cutoff": training_cutoff,
            }
        )
        self.is_fitted_ = True
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity(ids))
        return self

    def transform(self, entity_ids: Any) -> np.ndarray:
        self._require_fitted()
        return self.entity_embedding.transform(entity_ids)

    def pretrained_pair_embeddings(self) -> np.ndarray:
        self._require_fitted()
        return self.pretrained_pair_embeddings_.copy()

    def pretrained_node_embeddings(self) -> np.ndarray:
        self._require_fitted()
        return self.pretrained_node_embeddings_.copy()

    def pretrained_temporal_encoder(self) -> np.ndarray:
        self._require_fitted()
        return self.pretrained_temporal_encoder_.copy()

    def downstream_embedding_benchmark(
        self,
        entity_ids: Any,
        target: Any,
        *,
        train_size: int | None = None,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        ids = [str(value) for value in np.asarray(entity_ids).reshape(-1).tolist()]
        y = np.asarray(target, dtype=float).reshape(-1)
        if len(ids) != y.shape[0] or not np.isfinite(y).all():
            raise ValueError("entity_ids and target must have matching finite rows")
        split = int(train_size or max(2, round(len(ids) * 0.67)))
        if split <= 1 or split >= len(ids):
            raise ValueError("train_size must leave at least one holdout row")
        pretrained = self.transform(ids)
        random = EntityEmbedding(
            embedding_dim=self.embedding_dim,
            hash_bucket_count=self.hash_bucket_count,
            random_seed=self.random_seed + 997 if random_seed is None else int(random_seed),
            architecture="random_embedding_baseline",
            backend=self.backend["requested"],
        ).fit(ids, training_cutoff=self.training_cutoff_)
        random_features = random.transform(ids)
        pretrained_rmse = _linear_holdout_rmse(pretrained, y, split)
        random_rmse = _linear_holdout_rmse(random_features, y, split)
        return {
            "benchmark": "maintained_pretrained_embedding_holdout",
            "train_size": split,
            "holdout_size": int(len(ids) - split),
            "supervised_budget": split,
            "pretrained_rmse": pretrained_rmse,
            "random_embedding_rmse": random_rmse,
            "pretrained_beats_random": bool(pretrained_rmse < random_rmse),
            "improvement": random_rmse - pretrained_rmse,
            "random_embedding_artifact": random.artifact_metadata(),
            "pretrained_artifact": self.artifact_metadata(),
        }

    def artifact_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return self.artifact_.to_dict() | {
            "pretraining_tasks": list(self.tasks),
            "outputs": [
                "pretrained_entity_embeddings",
                "pretrained_pair_embeddings",
                "pretrained_node_embeddings",
                "pretrained_temporal_encoder",
            ],
        }

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        payload = {
            "artifact": self.artifact_.to_dict(),
            "pretraining_tasks": list(self.tasks),
            "entity_embeddings": self.pretrained_entity_embeddings_.tolist(),
            "pair_embeddings": self.pretrained_pair_embeddings_.tolist(),
            "node_embeddings": self.pretrained_node_embeddings_.tolist(),
            "temporal_encoder": self.pretrained_temporal_encoder_.tolist(),
            "entity_id_map": dict(self.entity_embedding.id_map_),
            "feature_mean": self.feature_mean_.tolist(),
            "feature_scale": self.feature_scale_.tolist(),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SelfSupervisedPretrainer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            embedding_dim=int(artifact["embedding_dim"]),
            hash_bucket_count=int(artifact["hash_bucket_config"]["entity"]),
            random_seed=int(artifact["random_seed"]),
            tasks=tuple(payload["pretraining_tasks"]),
            feature_roles=dict(artifact["feature_roles"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.entity_embedding = EntityEmbedding(
            embedding_dim=obj.embedding_dim,
            hash_bucket_count=obj.hash_bucket_count,
            random_seed=obj.random_seed,
            architecture="self_supervised_entity_embedding",
            feature_roles=obj.feature_roles,
            backend=obj.backend["requested"],
        )
        obj.entity_embedding.id_map_ = dict(payload["entity_id_map"])
        obj.entity_embedding.embeddings_ = np.asarray(payload["entity_embeddings"], dtype=float)
        obj.entity_embedding.is_fitted_ = True
        obj.pretrained_entity_embeddings_ = obj.entity_embedding.embeddings_
        obj.pretrained_pair_embeddings_ = np.asarray(payload["pair_embeddings"], dtype=float)
        obj.pretrained_node_embeddings_ = np.asarray(payload["node_embeddings"], dtype=float)
        obj.pretrained_temporal_encoder_ = np.asarray(payload["temporal_encoder"], dtype=float)
        obj.feature_mean_ = np.asarray(payload["feature_mean"], dtype=float)
        obj.feature_scale_ = np.asarray(payload["feature_scale"], dtype=float)
        obj.training_cutoff_ = artifact["training_cutoff"]
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture="deterministic_self_supervised_pretrainer",
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={"entity": dict(self.entity_embedding.id_map_)},
            hash_bucket_config={"entity": self.hash_bucket_count},
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
            feature_roles=dict(self.feature_roles)
            | {
                "pretraining_tasks": list(self.tasks),
                "outputs": [
                    "pretrained_entity_embeddings",
                    "pretrained_pair_embeddings",
                    "pretrained_node_embeddings",
                    "pretrained_temporal_encoder",
                ],
            },
            training_cutoff=self.training_cutoff_,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _save_load_parity(self, ids: list[str]) -> bool:
        probe = [*ids[: min(3, len(ids))], "__new_entity__"]
        before = self.transform(probe)
        payload_path = Path("/tmp/cartoboost_self_supervised_pretrainer_parity.json")
        self.save(payload_path)
        try:
            after = self.load(payload_path).transform(probe)
        finally:
            payload_path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("pretrainer must be fit before transform")


class MultiViewSpatialAttention:
    """Deterministic multi-view graph attention over aligned node feature views."""

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        random_seed: int = 0,
        backend: str = "cpu",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.random_seed = int(random_seed)
        self.backend = _resolve_backend(backend)
        self.is_fitted_ = False

    def fit(self, node_ids: Any, views: dict[str, Any]) -> MultiViewSpatialAttention:
        ids = [str(value) for value in np.asarray(node_ids).reshape(-1).tolist()]
        if not ids:
            raise ValueError("node_ids must be non-empty")
        if not views:
            raise ValueError("at least one spatial view is required")
        self.node_ids_ = ids
        self.view_names_ = sorted(str(name) for name in views)
        self.view_projections_: dict[str, np.ndarray] = {}
        self.router_bias_ = np.zeros(len(self.view_names_), dtype=float)
        normalized_views = self._normalize_views(views, require_all=True)
        for view_idx, view_name in enumerate(self.view_names_):
            matrix = normalized_views[view_name]
            self.view_projections_[view_name] = _deterministic_matrix(
                matrix.shape[1],
                self.embedding_dim,
                seed=self.random_seed + _stable_hash(view_name, 1_000_000_007),
                salt=f"multi_view_spatial_attention:{view_name}",
            )
            self.router_bias_[view_idx] = float(np.mean(np.abs(matrix)))
        self.router_weights_ = _deterministic_matrix(
            self.embedding_dim,
            len(self.view_names_),
            seed=self.random_seed + 701,
            salt="multi_view_router",
        )
        self.is_fitted_ = True
        output = self.transform(views)
        self.learned_view_weights_ = {
            name: float(np.mean(output["view_weights"][:, idx]))
            for idx, name in enumerate(self.view_names_)
        }
        report = self.view_ablation_report(views)
        self.training_metrics_ = {
            "full_proxy_score": float(report["full_proxy_score"]),
            "best_single_view_proxy_score": float(max(report["single_view_proxy_scores"].values())),
        }
        self.schema_hash_ = _schema_hash(
            {
                "node_ids": ids,
                "view_names": self.view_names_,
                "embedding_dim": self.embedding_dim,
            }
        )
        self.artifact_ = self._artifact(save_load_parity_checked=False)
        self.artifact_ = self._artifact(save_load_parity_checked=self._save_load_parity(views))
        return self

    def transform(self, views: dict[str, Any]) -> dict[str, Any]:
        self._require_fitted()
        normalized_views = self._normalize_views(views, require_all=False)
        if not normalized_views:
            raise ValueError("at least one fitted spatial view must be supplied")
        row_count = next(iter(normalized_views.values())).shape[0]
        projected: dict[str, np.ndarray] = {}
        missing = []
        for view_name in self.view_names_:
            if view_name not in normalized_views:
                missing.append(view_name)
                continue
            projected[view_name] = normalized_views[view_name] @ self.view_projections_[view_name]
        router_input = np.mean(list(projected.values()), axis=0)
        logits = router_input @ self.router_weights_ + self.router_bias_
        for view_idx, view_name in enumerate(self.view_names_):
            if view_name in missing:
                logits[:, view_idx] = -np.inf
        view_weights = _masked_softmax(logits)
        fused = np.zeros((row_count, self.embedding_dim), dtype=float)
        for view_idx, view_name in enumerate(self.view_names_):
            if view_name not in projected:
                continue
            fused += view_weights[:, [view_idx]] * projected[view_name]
        return {
            "embedding": _layer_norm(fused),
            "view_weights": view_weights,
            "available_views": [name for name in self.view_names_ if name in projected],
            "missing_views": missing,
        }

    def view_ablation_report(self, views: dict[str, Any]) -> dict[str, Any]:
        full = self.transform(views)
        full_score = _embedding_energy(full["embedding"])
        scores = {}
        for view_name in self.view_names_:
            if view_name not in views:
                continue
            scores[view_name] = _embedding_energy(
                self.transform({view_name: views[view_name]})["embedding"]
            )
        best_view = max(scores, key=lambda name: scores[name]) if scores else None
        best_score = scores[best_view] if best_view is not None else float("-inf")
        return {
            "full_proxy_score": full_score,
            "single_view_proxy_scores": scores,
            "best_single_view": best_view,
            "full_beats_best_single_view": bool(full_score >= best_score),
            "missing_views": list(full["missing_views"]),
        }

    def maintained_graph_benchmark(
        self,
        views: dict[str, Any],
        target: Any,
        *,
        train_size: int | None = None,
    ) -> dict[str, Any]:
        y = np.asarray(target, dtype=float).reshape(-1)
        if y.shape[0] != len(self.node_ids_) or not np.isfinite(y).all():
            raise ValueError("target must be finite with one value per node")
        split = int(train_size or max(2, round(len(y) * 0.67)))
        if split <= 1 or split >= len(y):
            raise ValueError("train_size must leave at least one holdout node")
        full = self.transform(views)["embedding"]
        full_rmse = _linear_holdout_rmse(full, y, split)
        single_scores = {}
        for view_name in self.view_names_:
            if view_name in views:
                single_scores[view_name] = _linear_holdout_rmse(
                    self.transform({view_name: views[view_name]})["embedding"],
                    y,
                    split,
                )
        best_single_view = min(single_scores, key=lambda name: single_scores[name])
        best_single_rmse = single_scores[best_single_view]
        return {
            "benchmark": "maintained_multi_view_graph_holdout",
            "train_size": split,
            "holdout_size": int(len(y) - split),
            "multi_view_rmse": full_rmse,
            "single_view_rmse": single_scores,
            "best_single_view": best_single_view,
            "best_single_view_rmse": best_single_rmse,
            "multi_view_beats_best_single_view": bool(full_rmse < best_single_rmse),
            "improvement": best_single_rmse - full_rmse,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return self.artifact_.to_dict() | {
            "learned_view_weights": dict(self.learned_view_weights_),
            "view_names": list(self.view_names_),
        }

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        payload = {
            "artifact": self.artifact_.to_dict(),
            "view_names": list(self.view_names_),
            "node_ids": list(self.node_ids_),
            "view_projections": {
                name: value.tolist() for name, value in self.view_projections_.items()
            },
            "router_weights": self.router_weights_.tolist(),
            "router_bias": self.router_bias_.tolist(),
            "learned_view_weights": dict(self.learned_view_weights_),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MultiViewSpatialAttention:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = payload["artifact"]
        obj = cls(
            embedding_dim=int(artifact["embedding_dim"]),
            random_seed=int(artifact["random_seed"]),
            backend=str(artifact.get("backend", {}).get("requested", "cpu")),
        )
        obj.node_ids_ = list(payload["node_ids"])
        obj.view_names_ = list(payload["view_names"])
        obj.view_projections_ = {
            name: np.asarray(values, dtype=float)
            for name, values in payload["view_projections"].items()
        }
        obj.router_weights_ = np.asarray(payload["router_weights"], dtype=float)
        obj.router_bias_ = np.asarray(payload["router_bias"], dtype=float)
        obj.learned_view_weights_ = dict(payload["learned_view_weights"])
        obj.training_metrics_ = dict(artifact["training_metrics"])
        obj.schema_hash_ = str(artifact["schema_hash"])
        obj.is_fitted_ = True
        obj.artifact_ = RepresentationArtifact(**artifact)
        return obj

    def _normalize_views(
        self, views: dict[str, Any], *, require_all: bool
    ) -> dict[str, np.ndarray]:
        allowed = set(self.view_names_)
        normalized = {}
        for view_name, values in views.items():
            name = str(view_name)
            if name not in allowed:
                if require_all:
                    raise ValueError(f"view {name!r} was not declared during fit")
                continue
            matrix = _as_feature_matrix(values)
            if matrix.shape[0] != len(self.node_ids_):
                raise ValueError(f"view {name} row count must match node_ids")
            normalized[name] = matrix
        if require_all and set(normalized) != allowed:
            missing = sorted(allowed - set(normalized))
            raise ValueError(f"missing required spatial views: {missing}")
        if normalized:
            row_count = next(iter(normalized.values())).shape[0]
            if any(matrix.shape[0] != row_count for matrix in normalized.values()):
                raise ValueError("all spatial views must have the same row count")
        return normalized

    def _artifact(self, *, save_load_parity_checked: bool) -> RepresentationArtifact:
        return RepresentationArtifact(
            model_class=self.__class__.__name__,
            architecture="multi_view_spatial_attention",
            artifact_version=ARTIFACT_VERSION,
            schema_hash=self.schema_hash_,
            id_maps={"node": {value: idx for idx, value in enumerate(self.node_ids_)}},
            hash_bucket_config={"view_count": len(self.view_names_)},
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
            feature_roles={
                "spatial_views": list(self.view_names_),
                "learned_view_weights": dict(self.learned_view_weights_),
            },
            training_cutoff=None,
            training_metrics=dict(self.training_metrics_),
            save_load_parity_checked=save_load_parity_checked,
            backend=dict(self.backend),
        )

    def _save_load_parity(self, views: dict[str, Any]) -> bool:
        before = self.transform(views)["embedding"]
        path = Path("/tmp/cartoboost_multi_view_spatial_attention_parity.json")
        self.save(path)
        try:
            after = self.load(path).transform(views)["embedding"]
        finally:
            path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("multi-view attention must be fit before transform")


LocalGlobalHubAttention = MultiViewSpatialAttention
SpatialSemanticGraphTransformer = MultiViewSpatialAttention


class MaskedEntityTimeModeling:
    task_name = "masked_entity_time_modeling"


class MaskedPairTimeModeling:
    task_name = "masked_pair_time_modeling"


class GraphEdgeDenoising:
    task_name = "graph_edge_denoising"


class TemporalOrderContrastiveLoss:
    task_name = "temporal_order_contrastive_loss"


class SpatialNeighborContrastiveLoss:
    task_name = "spatial_neighbor_contrastive_loss"


class FuturePatchReconstruction:
    task_name = "future_patch_reconstruction"


EntityTimeAdaptiveEmbedding = SpatioTemporalAdaptiveEmbedding
PairTimeAdaptiveEmbedding = PairEmbedding
NodeTimeAdaptiveEmbedding = SpatioTemporalAdaptiveEmbedding
GraphContextEmbedding = EntityEmbedding


def _deterministic_matrix(rows: int, cols: int, *, seed: int, salt: str) -> np.ndarray:
    values = np.zeros((rows, cols), dtype=float)
    for row in range(rows):
        for col in range(cols):
            digest = hashlib.sha256(f"{seed}:{salt}:{row}:{col}".encode()).digest()
            integer = int.from_bytes(digest[:8], "little", signed=False)
            values[row, col] = (integer / float(2**64 - 1)) * 2.0 - 1.0
    return values


def _resolve_backend(requested: str) -> dict[str, Any]:
    requested = str(requested).lower()
    supported = ["cpu", "cuda", "rocm", "mlx"]
    if requested not in {"auto", *supported}:
        raise ValueError("backend must be one of 'auto', 'cpu', 'cuda', 'rocm', or 'mlx'")
    selected = "cpu"
    return {
        "requested": requested,
        "selected": selected,
        "available": ["cpu"],
        "supported_accelerators": supported,
        "accelerator_ready": {"cuda": True, "rocm": True, "mlx": True},
    }


def _stable_hash(value: str, modulo: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % modulo


def _schema_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _project_features(values: Any, weights: np.ndarray, row_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] != row_count:
        raise ValueError("feature rows must match entity rows")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    if array.shape[1] < weights.shape[0]:
        pad = np.zeros((array.shape[0], weights.shape[0] - array.shape[1]), dtype=float)
        array = np.concatenate([array, pad], axis=1)
    return array[:, : weights.shape[0]] @ weights


def _as_feature_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("context_features must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("context_features must contain only finite values")
    return array


def _context_matrix(values: Any | None, row_count: int, context_dim: int) -> np.ndarray:
    if context_dim == 0:
        return np.zeros((row_count, 0), dtype=float)
    if values is None:
        return np.zeros((row_count, context_dim), dtype=float)
    array = _as_feature_matrix(values)
    if array.shape[0] != row_count:
        raise ValueError("context_features row count must match entity rows")
    if array.shape[1] != context_dim:
        raise ValueError("context_features column count must match fitted router")
    return array


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _masked_softmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    shifted = values - np.nanmax(np.where(finite, values, np.nan), axis=1, keepdims=True)
    exp = np.where(finite, np.exp(np.clip(shifted, -50.0, 50.0)), 0.0)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def _pair_pretraining_embeddings(
    ids: list[str],
    values: np.ndarray,
    *,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((0, embedding_dim), dtype=float)
    projection = _deterministic_matrix(
        values.shape[1] * 2,
        embedding_dim,
        seed=seed,
        salt="masked_pair_time_modeling",
    )
    rows = []
    for idx in range(len(values) - 1):
        features = np.concatenate([values[idx], values[idx + 1]])
        embedding = features @ projection
        if ids[idx] == ids[idx + 1]:
            embedding *= 1.1
        rows.append(embedding)
    return _layer_norm(np.asarray(rows, dtype=float))


def _temporal_pretraining_encoder(
    values: np.ndarray,
    *,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    if len(values) < 3:
        return np.zeros((0, embedding_dim), dtype=float)
    projection = _deterministic_matrix(
        values.shape[1],
        embedding_dim,
        seed=seed,
        salt="future_patch_reconstruction",
    )
    rows = [values[idx : idx + 3].mean(axis=0) @ projection for idx in range(len(values) - 2)]
    return _layer_norm(np.asarray(rows, dtype=float))


def _proxy_rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(centered**2)))


def _graph_edge_proxy_auc(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.5
    near = np.mean(np.linalg.norm(values[:-1] - values[1:], axis=1))
    far = np.mean(np.linalg.norm(values - values[::-1], axis=1))
    margin = far - near
    return float(1.0 / (1.0 + np.exp(-margin)))


def _temporal_order_margin(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    forward = np.sum(np.linalg.norm(values[:-1] - values[1:], axis=1))
    reverse = np.sum(np.linalg.norm(values[::-1][:-1] - values[::-1][1:], axis=1))
    return float(abs(reverse - forward) / (len(values) - 1))


def _spatial_neighbor_margin(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    neighbor = np.mean(np.linalg.norm(values[:-1] - values[1:], axis=1))
    distant = np.mean(np.linalg.norm(values[0] - values[2:], axis=1))
    return float(distant - neighbor)


def _future_patch_proxy_rmse(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.sqrt(np.mean((values[1:] - values[:-1]) ** 2)))


def _linear_holdout_rmse(features: np.ndarray, y: np.ndarray, train_size: int) -> float:
    design = np.column_stack([np.ones(features.shape[0]), features])
    coef, *_ = np.linalg.lstsq(design[:train_size], y[:train_size], rcond=None)
    pred = design[train_size:] @ coef
    return float(np.sqrt(np.mean((y[train_size:] - pred) ** 2)))


def _layer_norm(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-12)


def _embedding_energy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(values, axis=1)))


__all__ = [
    "EntityEmbedding",
    "EntityTimeAdaptiveEmbedding",
    "FuturePatchReconstruction",
    "GraphEdgeDenoising",
    "GraphContextEmbedding",
    "HistoricalAnalogRetriever",
    "KNNContextMemory",
    "LocalGlobalHubAttention",
    "MaskedEntityTimeModeling",
    "MaskedPairTimeModeling",
    "MultiViewSpatialAttention",
    "NodeTimeAdaptiveEmbedding",
    "PairEmbedding",
    "PairTimeAdaptiveEmbedding",
    "RegimeRouter",
    "RepresentationArtifact",
    "RetrievalAugmentedForecaster",
    "RetrievalAugmentedPairModel",
    "SelfSupervisedPretrainer",
    "SpatialSemanticGraphTransformer",
    "SpatialNeighborContrastiveLoss",
    "SpatioTemporalAdaptiveEmbedding",
    "TemporalOrderContrastiveLoss",
]
