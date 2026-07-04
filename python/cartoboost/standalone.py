"""Standalone neural and graph models backed by native Rust implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._native import (
    StandaloneGraphSageLinkPredictor as _NativeGraphSageLinkPredictor,
)
from ._native import (
    StandaloneGraphSageRegressor as _NativeGraphSageRegressor,
)
from ._native import (
    StandaloneHeteroGraphSageLinkPredictor as _NativeHeteroGraphSageLinkPredictor,
)
from ._native import (
    StandaloneHeteroGraphSageRegressor as _NativeHeteroGraphSageRegressor,
)
from ._native import (
    StandaloneHinSageLinkPredictor as _NativeHinSageLinkPredictor,
)
from ._native import (
    StandaloneHinSageRegressor as _NativeHinSageRegressor,
)
from ._native import (
    StandaloneNeuralEmbeddingRegressor as _NativeNeuralEmbeddingRegressor,
)
from ._native import (
    StandaloneNode2VecLinkPredictor as _NativeNode2VecLinkPredictor,
)
from ._native import (
    StandaloneNode2VecRegressor as _NativeNode2VecRegressor,
)
from .config import Backend, ChoiceStrEnum, FallbackMode
from .graph.eval import link_prediction_report

__all__ = [
    "NeuralEmbeddingStandaloneRegressor",
    "Node2VecStandaloneRegressor",
    "GraphSageStandaloneRegressor",
    "HeteroGraphSageStandaloneRegressor",
    "HinSageStandaloneRegressor",
    "Node2VecLinkPredictor",
    "GraphSageLinkPredictor",
    "HeteroGraphSageLinkPredictor",
    "HinSageLinkPredictor",
]


class NeuralEmbeddingStandaloneRegressor:
    """Supervised regressor over learned ID embeddings and optional dense features."""

    def __init__(
        self,
        *,
        dim: int = 16,
        fallback: FallbackMode = FallbackMode.GLOBAL_MEAN_VECTOR,
        random_state: int | None = 42,
        support_prior_strength: float = 1.0,
        n_estimators: int = 80,
        learning_rate: float = 0.07,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        min_gain: float = 0.0,
    ) -> None:
        self._native = _NativeNeuralEmbeddingRegressor(
            dim=int(dim),
            fallback=str(fallback),
            random_state=None if random_state is None else int(random_state),
            support_prior_strength=float(support_prior_strength),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            min_gain=float(min_gain),
        )

    def fit(
        self,
        ids: Any,
        y: Any,
        *,
        dense: Any | None = None,
    ) -> NeuralEmbeddingStandaloneRegressor:
        self._native.fit(_u64_list(ids, "ids"), _f64_list(y, "y"), _dense_optional(dense))
        return self

    def predict(self, ids: Any, *, dense: Any | None = None) -> np.ndarray:
        return np.asarray(
            self._native.predict(_u64_list(ids, "ids"), _dense_optional(dense)),
            dtype=np.float64,
        )

    def score(self, ids: Any, y: Any, *, dense: Any | None = None) -> float:
        actual = np.asarray(y, dtype=np.float64)
        pred = self.predict(ids, dense=dense)
        residual = actual - pred
        total = float(np.sum((actual - np.mean(actual)) ** 2))
        if total == 0.0:
            return 0.0
        return 1.0 - float(np.sum(residual**2)) / total

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> NeuralEmbeddingStandaloneRegressor:
        instance = cls(dim=1)
        instance._native = _NativeNeuralEmbeddingRegressor.load_artifact_json(str(path))
        return instance


class Node2VecStandaloneRegressor:
    """Standalone Node2Vec regressor for node or source-target row modeling."""

    def __init__(
        self,
        *,
        dim: int = 16,
        walk_length: int = 16,
        walks_per_node: int = 8,
        window_size: int = 5,
        epochs: int = 3,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        negative_samples: int = 5,
        p: float = 1.0,
        q: float = 1.0,
        seed: int = 0xA2B2_C2D2_E2F2_1234,
        l2_regularization: float = 0.0,
        normalize: bool = True,
        n_estimators: int = 80,
        booster_learning_rate: float = 0.07,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        min_gain: float = 0.0,
    ) -> None:
        self._native = _NativeNode2VecRegressor(
            dim=int(dim),
            walk_length=int(walk_length),
            walks_per_node=int(walks_per_node),
            window_size=int(window_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            min_learning_rate=float(min_learning_rate),
            negative_samples=int(negative_samples),
            p=float(p),
            q=float(q),
            seed=int(seed),
            l2_regularization=float(l2_regularization),
            normalize=bool(normalize),
            n_estimators=int(n_estimators),
            booster_learning_rate=float(booster_learning_rate),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            min_gain=float(min_gain),
        )

    def fit(
        self,
        *,
        node_count: int,
        edges: Any,
        row_nodes: Any,
        y: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
        edge_weights: Any | None = None,
    ) -> Node2VecStandaloneRegressor:
        self._native.fit(
            int(node_count),
            _edge_pairs(edges),
            _usize_list(row_nodes, "row_nodes"),
            _f64_list(y, "y"),
            None if row_targets is None else _usize_list(row_targets, "row_targets"),
            _dense_optional(dense),
            None if edge_weights is None else _f32_list(edge_weights, "edge_weights"),
        )
        return self

    def predict(
        self,
        row_nodes: Any,
        *,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._native.predict(
                _usize_list(row_nodes, "row_nodes"),
                None if row_targets is None else _usize_list(row_targets, "row_targets"),
                _dense_optional(dense),
            ),
            dtype=np.float64,
        )

    def score(
        self,
        row_nodes: Any,
        y: Any,
        *,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> float:
        return _r2(
            np.asarray(y, dtype=np.float64),
            self.predict(row_nodes, row_targets=row_targets, dense=dense),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Node2VecStandaloneRegressor:
        instance = cls(dim=1)
        instance._native = _NativeNode2VecRegressor.load_artifact_json(str(path))
        return instance


class GraphSageStandaloneRegressor:
    """Standalone GraphSAGE regressor for node or source-target row modeling."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (16,),
        epochs: int = 20,
        learning_rate: float = 0.05,
        negative_samples: int = 4,
        seed: int = 0x5A17_9A4E_7F33_C0DE,
        add_self_loop: bool = True,
        l2_regularization: float = 1e-5,
        n_estimators: int = 80,
        booster_learning_rate: float = 0.07,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        min_gain: float = 0.0,
        backend: Backend = Backend.CPU,
    ) -> None:
        self._native = _NativeGraphSageRegressor(
            input_dim=int(input_dim),
            hidden_dims=[int(value) for value in hidden_dims],
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            negative_samples=int(negative_samples),
            seed=int(seed),
            add_self_loop=bool(add_self_loop),
            l2_regularization=float(l2_regularization),
            n_estimators=int(n_estimators),
            booster_learning_rate=float(booster_learning_rate),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            min_gain=float(min_gain),
            backend=_choice_value(backend),
        )

    def fit(
        self,
        *,
        node_features: Any,
        edges: Any,
        row_nodes: Any,
        y: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> GraphSageStandaloneRegressor:
        self._native.fit(
            _f32_matrix(node_features, "node_features"),
            _edge_pairs(edges),
            _usize_list(row_nodes, "row_nodes"),
            _f64_list(y, "y"),
            None if row_targets is None else _usize_list(row_targets, "row_targets"),
            _dense_optional(dense),
        )
        return self

    def predict(
        self,
        *,
        node_features: Any,
        row_nodes: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._native.predict(
                _f32_matrix(node_features, "node_features"),
                _usize_list(row_nodes, "row_nodes"),
                None if row_targets is None else _usize_list(row_targets, "row_targets"),
                _dense_optional(dense),
            ),
            dtype=np.float64,
        )

    def score(
        self,
        *,
        node_features: Any,
        row_nodes: Any,
        y: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> float:
        return _r2(
            np.asarray(y, dtype=np.float64),
            self.predict(
                node_features=node_features,
                row_nodes=row_nodes,
                row_targets=row_targets,
                dense=dense,
            ),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> GraphSageStandaloneRegressor:
        instance = cls(input_dim=1)
        instance._native = _NativeGraphSageRegressor.load_artifact_json(str(path))
        return instance


class HeteroGraphSageStandaloneRegressor:
    """Standalone heterogeneous GraphSAGE regressor."""

    def __init__(
        self,
        *,
        input_dim: int,
        relation_count: int,
        hidden_dims: list[int] | tuple[int, ...] = (16,),
        epochs: int = 20,
        learning_rate: float = 0.05,
        negative_samples: int = 4,
        seed: int = 0x0D1A_2A3B_4C5D_6E7F,
        l2_regularization: float = 1e-5,
        n_estimators: int = 80,
        booster_learning_rate: float = 0.07,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        min_gain: float = 0.0,
        backend: Backend = Backend.CPU,
    ) -> None:
        self._native = _NativeHeteroGraphSageRegressor(
            input_dim=int(input_dim),
            relation_count=int(relation_count),
            hidden_dims=[int(value) for value in hidden_dims],
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            negative_samples=int(negative_samples),
            seed=int(seed),
            l2_regularization=float(l2_regularization),
            n_estimators=int(n_estimators),
            booster_learning_rate=float(booster_learning_rate),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            min_gain=float(min_gain),
            backend=_choice_value(backend),
        )

    def fit(
        self,
        *,
        node_features: Any,
        edges: Any,
        row_nodes: Any,
        y: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> HeteroGraphSageStandaloneRegressor:
        self._native.fit(
            _f32_matrix(node_features, "node_features"),
            _typed_edges(edges),
            _usize_list(row_nodes, "row_nodes"),
            _f64_list(y, "y"),
            None if row_targets is None else _usize_list(row_targets, "row_targets"),
            _dense_optional(dense),
        )
        return self

    def predict(
        self,
        *,
        node_features: Any,
        row_nodes: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._native.predict(
                _f32_matrix(node_features, "node_features"),
                _usize_list(row_nodes, "row_nodes"),
                None if row_targets is None else _usize_list(row_targets, "row_targets"),
                _dense_optional(dense),
            ),
            dtype=np.float64,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> HeteroGraphSageStandaloneRegressor:
        instance = cls(input_dim=1, relation_count=1)
        instance._native = _NativeHeteroGraphSageRegressor.load_artifact_json(str(path))
        return instance


class HinSageStandaloneRegressor:
    """Standalone HinSAGE regressor with explicit node and edge type schema."""

    def __init__(
        self,
        *,
        input_dim: int,
        node_type_count: int,
        edge_type_triples: Any,
        hidden_dims: list[int] | tuple[int, ...] = (16,),
        epochs: int = 20,
        learning_rate: float = 0.05,
        negative_samples: int = 4,
        seed: int = 0xA11C_E5A6_5EED_1234,
        l2_regularization: float = 1e-5,
        neighbor_samples: list[int] | tuple[int, ...] | None = None,
        n_estimators: int = 80,
        booster_learning_rate: float = 0.07,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        min_gain: float = 0.0,
        backend: Backend = Backend.CPU,
    ) -> None:
        self._native = _NativeHinSageRegressor(
            input_dim=int(input_dim),
            node_type_count=int(node_type_count),
            edge_type_triples=_typed_edges(edge_type_triples),
            hidden_dims=[int(value) for value in hidden_dims],
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            negative_samples=int(negative_samples),
            seed=int(seed),
            l2_regularization=float(l2_regularization),
            neighbor_samples=None
            if neighbor_samples is None
            else [int(value) for value in neighbor_samples],
            n_estimators=int(n_estimators),
            booster_learning_rate=float(booster_learning_rate),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            min_gain=float(min_gain),
            backend=_choice_value(backend),
        )

    def fit(
        self,
        *,
        node_features: Any,
        node_types: Any,
        edges: Any,
        row_nodes: Any,
        y: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> HinSageStandaloneRegressor:
        self._native.fit(
            _f32_matrix(node_features, "node_features"),
            _usize_list(node_types, "node_types"),
            _typed_edges(edges),
            _usize_list(row_nodes, "row_nodes"),
            _f64_list(y, "y"),
            None if row_targets is None else _usize_list(row_targets, "row_targets"),
            _dense_optional(dense),
        )
        return self

    def predict(
        self,
        *,
        node_features: Any,
        row_nodes: Any,
        row_targets: Any | None = None,
        dense: Any | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._native.predict(
                _f32_matrix(node_features, "node_features"),
                _usize_list(row_nodes, "row_nodes"),
                None if row_targets is None else _usize_list(row_targets, "row_targets"),
                _dense_optional(dense),
            ),
            dtype=np.float64,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> HinSageStandaloneRegressor:
        instance = cls(input_dim=1, node_type_count=1, edge_type_triples=[(0, 0, 0)])
        instance._native = _NativeHinSageRegressor.load_artifact_json(str(path))
        return instance


class Node2VecLinkPredictor:
    """Standalone Node2Vec edge scorer."""

    def __init__(self, **kwargs: Any) -> None:
        self._native = _NativeNode2VecLinkPredictor(**kwargs)

    def fit(
        self,
        *,
        node_count: int,
        edges: Any,
        edge_weights: Any | None = None,
    ) -> Node2VecLinkPredictor:
        self._native.fit(
            int(node_count),
            _edge_pairs(edges),
            None if edge_weights is None else _f32_list(edge_weights, "edge_weights"),
        )
        return self

    def predict_scores(self, pairs: Any) -> np.ndarray:
        return np.asarray(self._native.predict_scores(_edge_pairs(pairs)), dtype=np.float64)

    def report(
        self, pairs: Any, labels: Any, *, query_ids: Any | None = None, k: int = 10
    ) -> dict[str, float]:
        return link_prediction_report(
            _int_list(labels, "labels"),
            self.predict_scores(pairs).tolist(),
            None if query_ids is None else list(query_ids),
            k=k,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Node2VecLinkPredictor:
        instance = cls()
        instance._native = _NativeNode2VecLinkPredictor.load_artifact_json(str(path))
        return instance


class GraphSageLinkPredictor:
    """Standalone GraphSAGE edge scorer."""

    def __init__(self, *, input_dim: int, **kwargs: Any) -> None:
        self._native = _NativeGraphSageLinkPredictor(input_dim=int(input_dim), **kwargs)

    def fit(self, *, node_features: Any, edges: Any) -> GraphSageLinkPredictor:
        self._native.fit(_f32_matrix(node_features, "node_features"), _edge_pairs(edges))
        return self

    def predict_scores(self, *, node_features: Any, pairs: Any) -> np.ndarray:
        return np.asarray(
            self._native.predict_scores(
                _f32_matrix(node_features, "node_features"),
                _edge_pairs(pairs),
            ),
            dtype=np.float64,
        )

    def report(
        self,
        *,
        node_features: Any,
        pairs: Any,
        labels: Any,
        query_ids: Any | None = None,
        k: int = 10,
    ) -> dict[str, float]:
        return link_prediction_report(
            _int_list(labels, "labels"),
            self.predict_scores(node_features=node_features, pairs=pairs).tolist(),
            None if query_ids is None else list(query_ids),
            k=k,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> GraphSageLinkPredictor:
        instance = cls(input_dim=1)
        instance._native = _NativeGraphSageLinkPredictor.load_artifact_json(str(path))
        return instance


class HeteroGraphSageLinkPredictor:
    """Standalone heterogeneous GraphSAGE edge scorer."""

    def __init__(self, *, input_dim: int, relation_count: int, **kwargs: Any) -> None:
        self._native = _NativeHeteroGraphSageLinkPredictor(
            input_dim=int(input_dim),
            relation_count=int(relation_count),
            **kwargs,
        )

    def fit(self, *, node_features: Any, edges: Any) -> HeteroGraphSageLinkPredictor:
        self._native.fit(_f32_matrix(node_features, "node_features"), _typed_edges(edges))
        return self

    def predict_scores(self, *, node_features: Any, pairs: Any) -> np.ndarray:
        return np.asarray(
            self._native.predict_scores(
                _f32_matrix(node_features, "node_features"),
                _edge_pairs(pairs),
            ),
            dtype=np.float64,
        )

    def report(
        self,
        *,
        node_features: Any,
        pairs: Any,
        labels: Any,
        query_ids: Any | None = None,
        k: int = 10,
    ) -> dict[str, float]:
        return link_prediction_report(
            _int_list(labels, "labels"),
            self.predict_scores(node_features=node_features, pairs=pairs).tolist(),
            None if query_ids is None else list(query_ids),
            k=k,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> HeteroGraphSageLinkPredictor:
        instance = cls(input_dim=1, relation_count=1)
        instance._native = _NativeHeteroGraphSageLinkPredictor.load_artifact_json(str(path))
        return instance


class HinSageLinkPredictor:
    """Standalone HinSAGE edge scorer."""

    def __init__(
        self,
        *,
        input_dim: int,
        node_type_count: int,
        edge_type_triples: Any,
        **kwargs: Any,
    ) -> None:
        self._native = _NativeHinSageLinkPredictor(
            input_dim=int(input_dim),
            node_type_count=int(node_type_count),
            edge_type_triples=_typed_edges(edge_type_triples),
            **kwargs,
        )

    def fit(self, *, node_features: Any, node_types: Any, edges: Any) -> HinSageLinkPredictor:
        self._native.fit(
            _f32_matrix(node_features, "node_features"),
            _usize_list(node_types, "node_types"),
            _typed_edges(edges),
        )
        return self

    def predict_scores(self, *, node_features: Any, pairs: Any) -> np.ndarray:
        return np.asarray(
            self._native.predict_scores(
                _f32_matrix(node_features, "node_features"),
                _edge_pairs(pairs),
            ),
            dtype=np.float64,
        )

    def report(
        self,
        *,
        node_features: Any,
        pairs: Any,
        labels: Any,
        query_ids: Any | None = None,
        k: int = 10,
    ) -> dict[str, float]:
        return link_prediction_report(
            _int_list(labels, "labels"),
            self.predict_scores(node_features=node_features, pairs=pairs).tolist(),
            None if query_ids is None else list(query_ids),
            k=k,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._native.save_artifact_json(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> HinSageLinkPredictor:
        instance = cls(input_dim=1, node_type_count=1, edge_type_triples=[(0, 0, 0)])
        instance._native = _NativeHinSageLinkPredictor.load_artifact_json(str(path))
        return instance


def _r2(actual: np.ndarray, pred: np.ndarray) -> float:
    residual = actual - pred
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total == 0.0:
        return 0.0
    return 1.0 - float(np.sum(residual**2)) / total


def _choice_value(value: str | ChoiceStrEnum) -> str:
    if isinstance(value, ChoiceStrEnum):
        return value.value
    return str(value)


def _dense_optional(values: Any | None) -> list[list[float]] | None:
    if values is None:
        return None
    return _f64_matrix(values, "dense")


def _f64_matrix(values: Any, name: str) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array).tolist()


def _f32_matrix(values: Any, name: str) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array).tolist()


def _u64_list(values: Any, name: str) -> list[int]:
    array = np.asarray(values, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return [int(value) for value in array]


def _usize_list(values: Any, name: str) -> list[int]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative node ids")
    return [int(value) for value in array]


def _int_list(values: Any, name: str) -> list[int]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return [int(value) for value in array]


def _f64_list(values: Any, name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return [float(value) for value in array]


def _f32_list(values: Any, name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return [float(value) for value in array]


def _edge_pairs(values: Any) -> list[tuple[int, int]]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("edges must be a 2D array-like with shape (n_edges, 2)")
    if np.any(array < 0):
        raise ValueError("edges must contain non-negative node ids")
    return [(int(source), int(target)) for source, target in array]


def _typed_edges(values: Any) -> list[tuple[int, int, int]]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("typed edges must be a 2D array-like with shape (n_edges, 3)")
    if np.any(array < 0):
        raise ValueError("typed edges must contain non-negative ids")
    return [(int(source), int(target), int(relation)) for source, target, relation in array]
