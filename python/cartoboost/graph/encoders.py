"""Graph encoders for CartoBoost-friendly feature construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._native import (
    GraphSageEncoder as _NativeGraphSageEncoder,
)
from .._native import (
    HeteroGraphSageEncoder as _NativeHeteroGraphSageEncoder,
)
from .._native import (
    HinSageEncoder as _NativeHinSageEncoder,
)
from .._native import (
    Node2VecEncoder as _NativeNode2VecEncoder,
)
from .._native import (
    graph_compute_directional_features as _native_compute_directional_features,
)
from ..config import Backend
from .builder import (
    HeterogeneousGraph,
    HomogeneousGraph,
    ensure_node_features_shape,
    materialize_source_target_pair_nodes,
    normalize_heterogeneous_graph,
    normalize_homogeneous_graph,
)
from .config import GraphEmbeddingsConfig, GraphFeatureConfig
from .features import GraphFeatureBundle
from .schema import DirectionalityConfig

DEFAULT_GRAPHSAGE_SEED = 0x5A17_9A4E_7F33_C0DE
DEFAULT_HETEROGRAPHSAGE_SEED = 0x0D1A_2A3B_4C5D_6E7F
DEFAULT_HINSAGE_SEED = 0xA11C_E5A6_5EED_1234
DEFAULT_NODE2VEC_SEED = 0xA2B2_C2D2_E2F2_1234


def _backend_value(backend: Backend | str) -> str:
    value = backend.value if hasattr(backend, "value") else str(backend)
    normalized = str(value).strip().lower()
    return {"native": "cpu", "hip": "rocm", "dml": "directml"}.get(
        normalized,
        normalized,
    )


def _graph_embeddings_mapping(
    cfg: Mapping[str, Any] | GraphEmbeddingsConfig | GraphFeatureConfig,
) -> Mapping[str, Any]:
    if isinstance(cfg, GraphFeatureConfig):
        return cfg.transformer_config()["graph_embeddings"]
    if isinstance(cfg, GraphEmbeddingsConfig):
        return cfg.as_graph_embeddings_dict()["graph_embeddings"]
    if not isinstance(cfg, Mapping):
        raise TypeError("graph config must be a mapping or graph config dataclass")
    graph_cfg = cfg.get("graph_embeddings")
    if graph_cfg is None:
        graph_cfg = cfg.get("graph_sage", {})
    if not isinstance(graph_cfg, Mapping):
        raise TypeError("graph configuration must be a mapping")
    return graph_cfg


def _compute_homogeneous_directional_features(
    directionality: Mapping[str, Any] | DirectionalityConfig,
    node_count: int,
    edges: Sequence[tuple[int, int]],
    embeddings: np.ndarray,
    feature_prefix: str = "graph",
    edge_weights: Sequence[float] | None = None,
    edge_timestamps: Sequence[float] | None = None,
    backend: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    directionality_cfg = DirectionalityConfig.from_config(directionality)
    if not directionality_cfg.compute_asymmetry_features:
        return np.empty((node_count, 0), dtype=np.float32), []

    args = (
        int(node_count),
        [(int(source), int(target)) for source, target in edges],
        np.asarray(embeddings, dtype=np.float32).tolist(),
        None if edge_weights is None else [float(value) for value in edge_weights],
        None if edge_timestamps is None else [float(value) for value in edge_timestamps],
        str(feature_prefix),
        list(directionality_cfg.directional_features),
    )
    values, names = _native_compute_directional_features(*args, backend)
    return np.asarray(values, dtype=np.float32), list(names)


def _compute_hetero_directional_features(
    directionality: Mapping[str, Any] | DirectionalityConfig,
    node_count: int,
    edges: Sequence[tuple[int, int, int]],
    embeddings: np.ndarray,
    feature_prefix: str = "graph",
    edge_weights: Sequence[float] | None = None,
    edge_timestamps: Sequence[float] | None = None,
    backend: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    return _compute_homogeneous_directional_features(
        directionality,
        node_count,
        [(source, target) for source, target, _relation in edges],
        embeddings,
        feature_prefix=feature_prefix,
        edge_weights=edge_weights,
        edge_timestamps=edge_timestamps,
        backend=backend,
    )


def _align_edge_values(
    values: Sequence[float] | None,
    source_edge_count: int,
    normalized_edge_count: int,
) -> list[float] | None:
    if values is None:
        return None
    if len(values) != source_edge_count:
        raise ValueError("edge value length must match edge count")
    if normalized_edge_count == source_edge_count:
        return [float(value) for value in values]
    if source_edge_count == 0 or normalized_edge_count % source_edge_count != 0:
        raise ValueError("edge value length cannot be aligned to normalized graph edges")
    repeat = normalized_edge_count // source_edge_count
    return [float(value) for value in values for _ in range(repeat)]


def _expand_pair_node_values(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in values for _ in range(3)]


def _directionality_from_graph_config(
    graph_cfg: Mapping[str, Any],
    _encoder_cfg: Mapping[str, Any],
) -> DirectionalityConfig:
    raw_directionality = graph_cfg.get("directionality", {})
    directionality_payload = DirectionalityConfig.from_config(raw_directionality).as_dict()
    prefix = str(directionality_payload.get("directional_feature_prefix", "graph"))
    configured_features = directionality_payload.get("directional_features", ())
    if configured_features:
        directionality_payload["directional_features"] = [
            _normalize_directional_feature_name(str(feature), prefix)
            for feature in configured_features
        ]

    outputs = graph_cfg.get("outputs", {})
    if outputs is not None:
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs block must be a mapping")
        output_features = outputs.get("directional_features", ())
        if output_features:
            directionality_payload["directional_features"] = [
                _normalize_directional_feature_name(str(feature), prefix)
                for feature in output_features
            ]
            directionality_payload["compute_asymmetry_features"] = True

    return DirectionalityConfig.from_config(directionality_payload)


def _normalize_directional_feature_name(feature: str, prefix: str) -> str:
    if feature.startswith(f"{prefix}_"):
        return feature
    return f"{prefix}_{feature}"


@dataclass(frozen=True)
class GraphSageConfig:
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    epochs: int = 20
    learning_rate: float = 0.05
    negative_samples: int = 4
    seed: int = DEFAULT_GRAPHSAGE_SEED
    add_self_loop: bool = True
    l2_regularization: float = 1e-5
    backend: Backend | str = Backend.CPU


def _coerce_dim(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _coerce_hidden_dims(values: Sequence[int]) -> list[int]:
    if not values:
        return []
    hidden_dims = [int(v) for v in values]
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("hidden_dims must contain only positive values")
    return hidden_dims


class GraphSageFeatureEncoder:
    """High-level wrapper around the Rust GraphSageEncoder."""

    def __init__(self, config: GraphSageConfig) -> None:
        self.config = config
        self._encoder = _NativeGraphSageEncoder(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            epochs=config.epochs,
            learning_rate=float(config.learning_rate),
            negative_samples=int(config.negative_samples),
            seed=int(config.seed),
            add_self_loop=bool(config.add_self_loop),
            l2_regularization=float(config.l2_regularization),
            backend=_backend_value(config.backend),
        )
        self.graph: HomogeneousGraph | None = None

    def fit(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
    ) -> GraphFeatureBundle:
        graph = normalize_homogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )
        feature_rows = ensure_node_features_shape(node_features, graph.node_count)
        emb = self._encoder.fit(graph.node_count, graph.edges, feature_rows)
        self.graph = graph
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(graph.node_ids),
            feature_names=[f"graph_sage_homo_{index:02d}" for index in range(len(emb[0]))],
            provenance={"encoder": "graphsage", "directed": directed},
        )

    def encode(
        self,
        node_features: Sequence[Sequence[float]],
    ) -> GraphFeatureBundle:
        if self.graph is None:
            raise RuntimeError("GraphSageFeatureEncoder must be fitted first")
        feature_rows = ensure_node_features_shape(node_features, len(self.graph.node_ids))
        emb = self._encoder.encode_graph(
            self.graph.node_count,
            self.graph.edges,
            feature_rows,
        )
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(self.graph.node_ids),
            feature_names=[f"graph_sage_homo_{index:02d}" for index in range(len(emb[0]))],
            provenance={"encoder": "graphsage", "directed": self.graph.directed},
        )

    def loss_curve(self) -> list[float]:
        return self._encoder.loss_curve()

    def save_artifact_json(self, path: str) -> None:
        self._encoder.save_artifact_json(path)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | GraphSageConfig,
    ) -> GraphSageFeatureEncoder:
        if isinstance(config, GraphSageConfig):
            return cls(config)
        input_dim = _coerce_dim(int(config["input_dim"]), "input_dim")
        hidden_dims = _coerce_hidden_dims(config.get("hidden_dims", [16]))
        return cls(
            GraphSageConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                epochs=int(config.get("epochs", 20)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                negative_samples=int(config.get("negative_samples", 4)),
                seed=int(config.get("seed", DEFAULT_GRAPHSAGE_SEED)),
                add_self_loop=bool(config.get("add_self_loop", True)),
                l2_regularization=float(config.get("l2_regularization", 1e-5)),
                backend=Backend(str(config.get("backend", "cpu"))),
            )
        )


@dataclass(frozen=True)
class HeteroGraphSageConfig:
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    epochs: int = 20
    learning_rate: float = 0.05
    negative_samples: int = 4
    seed: int = DEFAULT_HETEROGRAPHSAGE_SEED
    l2_regularization: float = 1e-5
    backend: Backend | str = Backend.CPU


@dataclass(frozen=True)
class HinSageConfig:
    input_dim: int
    node_type_count: int
    edge_type_triples: list[tuple[int, int, int]]
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    epochs: int = 20
    learning_rate: float = 0.05
    negative_samples: int = 4
    seed: int = DEFAULT_HINSAGE_SEED
    l2_regularization: float = 1e-5
    neighbor_samples: list[int] = field(default_factory=list)
    backend: Backend | str = Backend.CPU


@dataclass(frozen=True)
class Node2VecConfig:
    dim: int = 16
    walk_length: int = 16
    walks_per_node: int = 8
    window_size: int = 5
    epochs: int = 3
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    negative_samples: int = 5
    p: float = 1.0
    q: float = 1.0
    seed: int = DEFAULT_NODE2VEC_SEED
    l2_regularization: float = 0.0
    normalize: bool = True
    backend: Backend | str = Backend.CPU


class Node2VecFeatureEncoder:
    """Thin wrapper around the native Rust node2vec encoder."""

    def __init__(self, config: Node2VecConfig) -> None:
        self.config = config
        kwargs = dict(
            dim=config.dim,
            walk_length=config.walk_length,
            walks_per_node=config.walks_per_node,
            window_size=config.window_size,
            epochs=config.epochs,
            learning_rate=float(config.learning_rate),
            min_learning_rate=float(config.min_learning_rate),
            negative_samples=config.negative_samples,
            p=float(config.p),
            q=float(config.q),
            seed=int(config.seed),
            l2_regularization=float(config.l2_regularization),
            normalize=bool(config.normalize),
            backend=_backend_value(config.backend),
        )
        self._encoder = _NativeNode2VecEncoder(**kwargs)
        self.graph: HomogeneousGraph | None = None

    def fit(
        self,
        edges: Sequence[tuple[Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        edge_weights: Sequence[float] | None = None,
    ) -> GraphFeatureBundle:
        graph = normalize_homogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )
        weights = _align_edge_values(edge_weights, len(edges), len(graph.edges))
        emb = self._encoder.fit(
            graph.node_count,
            graph.edges,
            weights,
        )
        self.graph = graph
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(graph.node_ids),
            feature_names=[f"node2vec_{index:02d}" for index in range(self.config.dim)],
            provenance={
                "encoder": "node2vec",
                "directed": directed,
                "weighted": edge_weights is not None,
                "p": float(self.config.p),
                "q": float(self.config.q),
                "walk_length": int(self.config.walk_length),
                "walks_per_node": int(self.config.walks_per_node),
                "window_size": int(self.config.window_size),
                "epochs": int(self.config.epochs),
                "native": "rust",
                "backend": _backend_value(self.config.backend),
            },
        )

    def encode(self) -> GraphFeatureBundle:
        if self.graph is None:
            raise RuntimeError("Node2VecFeatureEncoder must be fitted first")
        emb = self._encoder.encode()
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(self.graph.node_ids),
            feature_names=[f"node2vec_{index:02d}" for index in range(self.config.dim)],
            provenance={"encoder": "node2vec", "native": "rust"},
        )

    def loss_curve(self) -> list[float]:
        return self._encoder.loss_curve()

    def save_artifact_json(self, path: str) -> None:
        self._encoder.save_artifact_json(path)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Node2VecFeatureEncoder:
        return cls(
            Node2VecConfig(
                dim=int(config.get("dim", config.get("output_dim", 16))),
                walk_length=int(config.get("walk_length", 16)),
                walks_per_node=int(config.get("walks_per_node", 8)),
                window_size=int(config.get("window_size", 5)),
                epochs=int(config.get("epochs", 3)),
                learning_rate=float(config.get("learning_rate", 0.025)),
                min_learning_rate=float(config.get("min_learning_rate", 0.0001)),
                negative_samples=int(config.get("negative_samples", 5)),
                p=float(config.get("p", 1.0)),
                q=float(config.get("q", 1.0)),
                seed=int(config.get("seed", DEFAULT_NODE2VEC_SEED)),
                l2_regularization=float(config.get("l2_regularization", 0.0)),
                normalize=bool(config.get("normalize", True)),
                backend=Backend(str(config.get("backend", "cpu"))),
            )
        )


class HinSageFeatureEncoder:
    """Thin wrapper around the native Rust HinSageEncoder."""

    def __init__(self, config: HinSageConfig) -> None:
        self.config = config
        self._encoder = _NativeHinSageEncoder(
            input_dim=config.input_dim,
            node_type_count=config.node_type_count,
            edge_type_triples=list(config.edge_type_triples),
            hidden_dims=list(config.hidden_dims),
            epochs=config.epochs,
            learning_rate=float(config.learning_rate),
            negative_samples=int(config.negative_samples),
            seed=int(config.seed),
            l2_regularization=float(config.l2_regularization),
            neighbor_samples=list(config.neighbor_samples),
            backend=_backend_value(config.backend),
        )

    def fit(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[int, int, int]],
        node_types: Sequence[int],
    ) -> GraphFeatureBundle:
        feature_rows = ensure_node_features_shape(node_features, len(node_types))
        emb = self._encoder.fit(
            [int(node_type) for node_type in node_types],
            [(int(source), int(target), int(relation)) for source, target, relation in edges],
            feature_rows,
        )
        return GraphFeatureBundle(
            embeddings=emb,
            feature_names=[f"hinsage_{index:02d}" for index in range(len(emb[0]))],
            node_ids=list(range(len(node_types))),
            provenance={
                "encoder": "hinsage",
                "node_type_count": self.config.node_type_count,
                "edge_type_triples": list(self.config.edge_type_triples),
                "neighbor_samples": list(self.config.neighbor_samples),
            },
        )

    def encode(self, node_features: Sequence[Sequence[float]]) -> GraphFeatureBundle:
        emb = self._encoder.encode([list(row) for row in node_features])
        return GraphFeatureBundle(
            embeddings=emb,
            feature_names=[f"hinsage_{index:02d}" for index in range(len(emb[0]))],
            node_ids=list(range(len(emb))),
            provenance={"encoder": "hinsage"},
        )

    def link_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        pairs: Sequence[tuple[int, int]],
    ) -> GraphFeatureBundle:
        values = self._encoder.link_embeddings(
            [list(row) for row in embeddings],
            [(int(source), int(target)) for source, target in pairs],
        )
        width = len(values[0]) if values else 0
        return GraphFeatureBundle(
            embeddings=values,
            feature_names=[f"hinsage_link_{index:02d}" for index in range(width)],
            node_ids=list(pairs),
            provenance={"encoder": "hinsage_link"},
        )

    def loss_curve(self) -> list[float]:
        return self._encoder.loss_curve()

    def save_artifact_json(self, path: str) -> None:
        self._encoder.save_artifact_json(path)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | HinSageConfig,
    ) -> HinSageFeatureEncoder:
        if isinstance(config, HinSageConfig):
            return cls(config)
        input_dim = _coerce_dim(int(config["input_dim"]), "input_dim")
        node_type_count = _coerce_dim(int(config["node_type_count"]), "node_type_count")
        triples = config.get("edge_type_triples")
        if not triples:
            raise ValueError("edge_type_triples is required for HinSAGE")
        edge_type_triples = [tuple(int(value) for value in triple) for triple in triples]
        hidden_dims = _coerce_hidden_dims(config.get("hidden_dims", [16]))
        neighbor_samples = [int(value) for value in config.get("neighbor_samples", [])]
        return cls(
            HinSageConfig(
                input_dim=input_dim,
                node_type_count=node_type_count,
                edge_type_triples=edge_type_triples,  # type: ignore[arg-type]
                hidden_dims=hidden_dims,
                epochs=int(config.get("epochs", 20)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                negative_samples=int(config.get("negative_samples", 4)),
                seed=int(config.get("seed", DEFAULT_HINSAGE_SEED)),
                l2_regularization=float(config.get("l2_regularization", 1e-5)),
                neighbor_samples=neighbor_samples,
                backend=Backend(str(config.get("backend", "cpu"))),
            )
        )


class HeteroGraphSageFeatureEncoder:
    """High-level wrapper around the Rust HeteroGraphSageEncoder."""

    def __init__(self, config: HeteroGraphSageConfig) -> None:
        self.config = config
        self._encoder = _NativeHeteroGraphSageEncoder(
            input_dim=config.input_dim,
            relation_count=1,
            hidden_dims=list(config.hidden_dims),
            epochs=config.epochs,
            learning_rate=float(config.learning_rate),
            negative_samples=int(config.negative_samples),
            seed=int(config.seed),
            l2_regularization=float(config.l2_regularization),
            backend=_backend_value(config.backend),
        )
        self.graph: HeterogeneousGraph | None = None

    def fit(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[Any, Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        relation_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        materialize_reverse_edges: bool = False,
        reverse_relation_suffix: str = "_reverse",
        reverse_relation_map: Mapping[Any, Any] | None = None,
    ) -> GraphFeatureBundle:
        graph = normalize_heterogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            relation_ids=relation_ids,
            node_count=node_count,
            directed=directed,
            materialize_reverse_edges=materialize_reverse_edges,
            reverse_relation_suffix=reverse_relation_suffix,
            reverse_relation_map=reverse_relation_map,
        )
        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if self._encoder.relation_count != graph.relation_count:
            # Keep relation count stable with the fitted topology.
            self._encoder = _NativeHeteroGraphSageEncoder(
                input_dim=self.config.input_dim,
                relation_count=graph.relation_count,
                hidden_dims=list(self.config.hidden_dims),
                epochs=self.config.epochs,
                learning_rate=float(self.config.learning_rate),
                negative_samples=int(self.config.negative_samples),
                seed=int(self.config.seed),
                l2_regularization=float(self.config.l2_regularization),
                backend=_backend_value(self.config.backend),
            )

        feature_rows = ensure_node_features_shape(node_features, graph.node_count)
        emb = self._encoder.fit(graph.node_count, graph.edges, feature_rows)
        self.graph = graph
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(graph.node_ids),
            feature_names=[f"graph_sage_hetero_{index:02d}" for index in range(len(emb[0]))],
            provenance={
                "encoder": "hetero_graphsage",
                "directed": directed,
                "relation_ids": list(graph.relation_ids),
            },
        )

    def encode(
        self,
        node_features: Sequence[Sequence[float]],
    ) -> GraphFeatureBundle:
        if self.graph is None:
            raise RuntimeError("HeteroGraphSageFeatureEncoder must be fitted first")
        feature_rows = ensure_node_features_shape(node_features, len(self.graph.node_ids))
        emb = self._encoder.encode_graph(
            self.graph.node_count,
            self.graph.edges,
            feature_rows,
        )
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(self.graph.node_ids),
            feature_names=[f"graph_sage_hetero_{index:02d}" for index in range(len(emb[0]))],
            provenance={
                "encoder": "hetero_graphsage",
                "directed": self.graph.directed,
                "relation_ids": list(self.graph.relation_ids),
            },
        )

    def loss_curve(self) -> list[float]:
        return self._encoder.loss_curve()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | HeteroGraphSageConfig,
    ) -> HeteroGraphSageFeatureEncoder:
        if isinstance(config, HeteroGraphSageConfig):
            return cls(config)
        input_dim = _coerce_dim(int(config["input_dim"]), "input_dim")
        hidden_dims = _coerce_hidden_dims(config.get("hidden_dims", [16]))
        return cls(
            HeteroGraphSageConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                epochs=int(config.get("epochs", 20)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                negative_samples=int(config.get("negative_samples", 4)),
                seed=int(config.get("seed", DEFAULT_HETEROGRAPHSAGE_SEED)),
                l2_regularization=float(config.get("l2_regularization", 1e-5)),
                backend=Backend(str(config.get("backend", "cpu"))),
            )
        )


class GraphFeatureTransformer:
    """Small configuration-first API for graph feature extraction."""

    def __init__(
        self,
        *,
        use_hetero: bool = False,
        use_hinsage: bool = False,
        use_node2vec: bool = False,
        sage_kwargs: Mapping[str, Any] | None = None,
        hetero_kwargs: Mapping[str, Any] | None = None,
        hinsage_kwargs: Mapping[str, Any] | None = None,
        node2vec_kwargs: Mapping[str, Any] | None = None,
        directionality: Mapping[str, Any] | None = None,
        backend: str = "cpu",
    ) -> None:
        self.use_hetero = bool(use_hetero)
        self.use_hinsage = bool(use_hinsage)
        self.use_node2vec = bool(use_node2vec)
        self.sage_kwargs = dict(sage_kwargs or {})
        self.hetero_kwargs = dict(hetero_kwargs or {})
        self.hinsage_kwargs = dict(hinsage_kwargs or {})
        self.node2vec_kwargs = dict(node2vec_kwargs or {})
        self.directionality = DirectionalityConfig.from_config(directionality)
        self.backend = str(backend)
        self.encoder: (
            GraphSageFeatureEncoder
            | HeteroGraphSageFeatureEncoder
            | HinSageFeatureEncoder
            | Node2VecFeatureEncoder
            | None
        ) = None
        self._target_input_dim: int | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any] | GraphEmbeddingsConfig | GraphFeatureConfig,
    ) -> GraphFeatureTransformer:
        graph_cfg = _graph_embeddings_mapping(cfg)

        encoder_cfg = graph_cfg.get("encoder", graph_cfg)
        if not isinstance(encoder_cfg, Mapping):
            raise TypeError("graph encoder configuration must be a mapping")

        family = str(encoder_cfg.get("family", "graphsage")).lower()
        if family not in {"graphsage", "hinsage", "sage", "node2vec"}:
            raise ValueError(f"unsupported graph family {family!r}")

        accelerator_backend = _backend_value(graph_cfg.get("backend", "native"))
        encoder_options = dict(encoder_cfg)
        encoder_options.setdefault("backend", accelerator_backend)
        use_hetero = bool(encoder_cfg.get("hetero", graph_cfg.get("hetero", False)))
        directionality = _directionality_from_graph_config(graph_cfg, encoder_cfg)
        use_hinsage = family == "hinsage" and {
            "node_type_count",
            "edge_type_triples",
        }.issubset(encoder_cfg)
        if family == "node2vec":
            return cls(
                use_node2vec=True,
                node2vec_kwargs=encoder_options,
                directionality=directionality,
                backend=accelerator_backend,
            )
        if use_hinsage:
            return cls(
                use_hetero=True,
                use_hinsage=True,
                hinsage_kwargs=encoder_options,
                directionality=directionality,
                backend=accelerator_backend,
            )
        if use_hetero:
            return cls(
                use_hetero=True,
                hetero_kwargs=encoder_options,
                directionality=directionality,
                backend=accelerator_backend,
            )
        return cls(
            use_hetero=False,
            sage_kwargs=encoder_options,
            directionality=directionality,
            backend=accelerator_backend,
        )

    def fit_transform(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[Any, Any]] | Sequence[tuple[Any, Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        relation_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        edge_weights: Sequence[float] | None = None,
        edge_timestamps: Sequence[float] | None = None,
        node_types: Sequence[int] | None = None,
    ) -> GraphFeatureBundle:
        feature_rows = [list(row) for row in node_features]
        if not feature_rows:
            raise ValueError("node_features must not be empty")
        self._target_input_dim = len(feature_rows[0])
        self._validate_input_dim()

        if self.use_node2vec:
            return self._fit_node2vec(
                feature_rows,
                edges=edges,  # type: ignore[arg-type]
                node_ids=node_ids,
                node_count=node_count,
                directed=directed,
                directionality=self.directionality,
                edge_weights=edge_weights,
                edge_timestamps=edge_timestamps,
            )
        if self.use_hinsage:
            return self._fit_hinsage(
                feature_rows,
                edges=edges,  # type: ignore[arg-type]
                node_types=node_types,
                directionality=self.directionality,
                edge_weights=edge_weights,
                edge_timestamps=edge_timestamps,
            )
        if self.use_hetero:
            return self._fit_hetero(
                feature_rows,
                edges=edges,  # type: ignore[arg-type]
                node_ids=node_ids,
                relation_ids=relation_ids,
                node_count=node_count,
                directed=directed,
                directionality=self.directionality,
                edge_weights=edge_weights,
                edge_timestamps=edge_timestamps,
            )
        return self._fit_homo(
            feature_rows,
            edges=edges,  # type: ignore[arg-type]
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
            directionality=self.directionality,
            edge_weights=edge_weights,
            edge_timestamps=edge_timestamps,
        )

    def _ensure_encoder(
        self,
    ) -> (
        GraphSageFeatureEncoder
        | HeteroGraphSageFeatureEncoder
        | HinSageFeatureEncoder
        | Node2VecFeatureEncoder
    ):
        if self._target_input_dim is None:
            raise RuntimeError("GraphFeatureTransformer.fit_transform has not been initialized")
        if self.encoder is not None:
            return self.encoder

        if self.use_node2vec:
            self.encoder = Node2VecFeatureEncoder.from_config(self.node2vec_kwargs)
            return self.encoder

        if self.use_hinsage:
            cfg = dict(self.hinsage_kwargs)
            cfg.setdefault("input_dim", self._target_input_dim)
            self.encoder = HinSageFeatureEncoder.from_config(cfg)
            return self.encoder

        if self.use_hetero:
            cfg = dict(self.hetero_kwargs)
            cfg.setdefault("input_dim", self._target_input_dim)
            self.encoder = HeteroGraphSageFeatureEncoder.from_config(
                cfg,
            )
            return self.encoder

        cfg = dict(self.sage_kwargs)
        cfg.setdefault("input_dim", self._target_input_dim)
        if self._target_input_dim is None:
            raise RuntimeError("input dimension could not be inferred")
        self.encoder = GraphSageFeatureEncoder.from_config(cfg)
        return self.encoder

    def _validate_input_dim(self) -> None:
        if self._target_input_dim is None or self._target_input_dim <= 0:
            raise ValueError("node features must have at least one column")
        if self.use_node2vec:
            configured = None
        elif self.use_hinsage:
            configured = self.hinsage_kwargs.get("input_dim")
        elif self.use_hetero:
            configured = self.hetero_kwargs.get("input_dim")
        else:
            configured = self.sage_kwargs.get("input_dim")
        if configured is None:
            return

        configured_dim = _coerce_dim(int(configured), "configured input_dim")
        if configured_dim != self._target_input_dim:
            raise ValueError(
                f"configured input_dim {configured_dim} does not match feature width "
                f"{self._target_input_dim}"
            )

    def _fit_node2vec(
        self,
        feature_rows: list[list[float]],
        edges: Sequence[tuple[Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        directionality: Mapping[str, Any] | None = None,
        edge_weights: Sequence[float] | None = None,
        edge_timestamps: Sequence[float] | None = None,
    ) -> GraphFeatureBundle:
        encoder = self._ensure_encoder()
        if not isinstance(encoder, Node2VecFeatureEncoder):
            raise RuntimeError("configured encoder mismatch; expected node2vec")
        graph = normalize_homogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )
        ensure_node_features_shape(feature_rows, graph.node_count)
        directionality = DirectionalityConfig.from_config(directionality)
        directionality.validate(directed=directed)
        bundle = encoder.fit(
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
            edge_weights=edge_weights,
        )
        directional_features, directional_names = _compute_homogeneous_directional_features(
            directionality=directionality,
            node_count=graph.node_count,
            edges=graph.edges,
            embeddings=bundle.embeddings,
            feature_prefix=directionality.directional_feature_prefix,
            edge_weights=_align_edge_values(edge_weights, len(edges), len(graph.edges)),
            edge_timestamps=_align_edge_values(edge_timestamps, len(edges), len(graph.edges)),
            backend=self.backend,
        )
        if directional_features.size == 0:
            return bundle
        return bundle.with_directional_features(directional_features, directional_names)

    def _fit_homo(
        self,
        feature_rows: list[list[float]],
        edges: Sequence[tuple[Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        directionality: Mapping[str, Any] | None = None,
        edge_weights: Sequence[float] | None = None,
        edge_timestamps: Sequence[float] | None = None,
    ) -> GraphFeatureBundle:
        encoder = self._ensure_encoder()
        if not isinstance(encoder, GraphSageFeatureEncoder):
            raise RuntimeError("configured encoder mismatch; expected homogeneous")
        directionality = DirectionalityConfig.from_config(directionality)
        directionality.validate(directed=directed)
        directional_feature_prefix = directionality.directional_feature_prefix
        graph = normalize_homogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )
        bundle = encoder.fit(
            node_features=feature_rows,
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )
        directional_features, directional_names = _compute_homogeneous_directional_features(
            directionality=directionality,
            node_count=graph.node_count,
            edges=graph.edges,
            embeddings=bundle.embeddings,
            feature_prefix=directional_feature_prefix,
            edge_weights=_align_edge_values(edge_weights, len(edges), len(graph.edges)),
            edge_timestamps=_align_edge_values(edge_timestamps, len(edges), len(graph.edges)),
            backend=self.backend,
        )
        if directional_features.size == 0:
            return bundle
        return bundle.with_directional_features(directional_features, directional_names)

    def _fit_hinsage(
        self,
        feature_rows: list[list[float]],
        edges: Sequence[tuple[int, int, int]],
        *,
        node_types: Sequence[int] | None,
        directionality: Mapping[str, Any] | None = None,
        edge_weights: Sequence[float] | None = None,
        edge_timestamps: Sequence[float] | None = None,
    ) -> GraphFeatureBundle:
        if node_types is None:
            raise ValueError("node_types is required for HinSAGE")
        encoder = self._ensure_encoder()
        if not isinstance(encoder, HinSageFeatureEncoder):
            raise RuntimeError("configured encoder mismatch; expected HinSAGE")
        bundle = encoder.fit(
            node_features=feature_rows,
            edges=edges,
            node_types=node_types,
        )
        directionality = DirectionalityConfig.from_config(directionality)
        directional_features, directional_names = _compute_hetero_directional_features(
            directionality=directionality,
            node_count=len(node_types),
            edges=[(int(source), int(target), int(relation)) for source, target, relation in edges],
            embeddings=bundle.embeddings,
            feature_prefix=directionality.directional_feature_prefix,
            edge_weights=edge_weights,
            edge_timestamps=edge_timestamps,
            backend=self.backend,
        )
        if directional_features.size == 0:
            return bundle
        return bundle.with_directional_features(directional_features, directional_names)

    def _fit_hetero(
        self,
        feature_rows: list[list[float]],
        edges: Sequence[tuple[Any, Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        relation_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        directionality: Mapping[str, Any] | None = None,
        edge_weights: Sequence[float] | None = None,
        edge_timestamps: Sequence[float] | None = None,
    ) -> GraphFeatureBundle:
        encoder = self._ensure_encoder()
        if not isinstance(encoder, HeteroGraphSageFeatureEncoder):
            raise RuntimeError("configured encoder mismatch; expected hetero")

        directionality = DirectionalityConfig.from_config(directionality)
        directionality.validate(directed=directed)
        materialize_reverse_edges = directionality.materialize_reverse_edges
        reverse_relation_suffix = directionality.reverse_relation_suffix
        reverse_relation_map = directionality.reverse_relation_map
        if reverse_relation_map is not None and not isinstance(reverse_relation_map, Mapping):
            raise TypeError("reverse_relation_map must be a mapping when provided")

        resolved_edges = list(edges)
        resolved_edge_weights = list(edge_weights) if edge_weights is not None else None
        resolved_edge_timestamps = list(edge_timestamps) if edge_timestamps is not None else None
        resolved_node_ids = list(node_ids) if node_ids is not None else None
        if directionality.create_od_pair_nodes:
            if resolved_node_ids is None:
                raise ValueError(
                    "create_od_pair_nodes requires node_ids so generated pair "
                    "nodes can be aligned with feature rows",
                )
            pair_nodes = materialize_source_target_pair_nodes(resolved_edges)
            resolved_edges = pair_nodes.edges
            resolved_edge_weights = _expand_pair_node_values(resolved_edge_weights)
            resolved_edge_timestamps = _expand_pair_node_values(resolved_edge_timestamps)
            feature_width = len(feature_rows[0])
            for pair_node in pair_nodes.pair_node_ids:
                if pair_node not in resolved_node_ids:
                    resolved_node_ids.append(pair_node)
                    feature_rows.append([0.0] * feature_width)

        graph = normalize_heterogeneous_graph(
            edges=resolved_edges,
            node_ids=resolved_node_ids,
            relation_ids=relation_ids,
            node_count=node_count,
            directed=directed,
            materialize_reverse_edges=materialize_reverse_edges,
            reverse_relation_suffix=reverse_relation_suffix,
            reverse_relation_map=reverse_relation_map,
        )
        bundle = encoder.fit(
            node_features=feature_rows,
            edges=resolved_edges,
            node_ids=resolved_node_ids,
            relation_ids=relation_ids,
            node_count=node_count,
            directed=directed,
            materialize_reverse_edges=materialize_reverse_edges,
            reverse_relation_suffix=reverse_relation_suffix,
            reverse_relation_map=reverse_relation_map,
        )
        directional_features, directional_names = _compute_hetero_directional_features(
            directionality=directionality,
            node_count=graph.node_count,
            edges=graph.edges,
            embeddings=bundle.embeddings,
            feature_prefix=directionality.directional_feature_prefix,
            edge_weights=_align_edge_values(
                resolved_edge_weights,
                len(resolved_edges),
                len(graph.edges),
            ),
            edge_timestamps=_align_edge_values(
                resolved_edge_timestamps,
                len(resolved_edges),
                len(graph.edges),
            ),
            backend=self.backend,
        )
        if directional_features.size == 0:
            return bundle
        return bundle.with_directional_features(directional_features, directional_names)
