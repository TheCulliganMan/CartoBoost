"""Configuration helpers for graph feature pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .schema import (
    DirectedMetaPath,
    DirectionalityConfig,
    GraphSchema,
    as_edge_types,
)


class GraphEncoderFamily(str, Enum):
    GRAPHSAGE = "graphsage"
    HINSAGE = "hinsage"
    SAGE = "sage"
    NODE2VEC = "node2vec"


class GraphBackend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    HIP = "hip"
    METAL = "metal"
    DIRECTML = "directml"
    DML = "dml"
    WEBGPU = "webgpu"
    # Backward-compatible name for the original Rust/CPU implementation.
    NATIVE = "native"


class GraphTaskMode(str, Enum):
    PRECOMPUTE_FEATURES = "precompute_features"


class DirectionalFeature(str, Enum):
    SOURCE_TARGET_EMBEDDING = "source_target_embedding"
    TARGET_SOURCE_EMBEDDING = "target_source_embedding"
    FORWARD_REVERSE_SIMILARITY_DELTA = "forward_reverse_similarity_delta"
    SOURCE_TARGET_AFFINITY = "source_target_affinity"
    TARGET_SOURCE_AFFINITY = "target_source_affinity"
    SOURCE_OUTBOUND_STRENGTH = "source_outbound_strength"
    TARGET_INBOUND_STRENGTH = "target_inbound_strength"
    FORWARD_FLOW_WEIGHT = "forward_flow_weight"
    REVERSE_FLOW_WEIGHT = "reverse_flow_weight"
    FLOW_ASYMMETRY = "flow_asymmetry"
    SOURCE_OUT_DEGREE_WEIGHTED = "source_out_degree_weighted"
    TARGET_IN_DEGREE_WEIGHTED = "target_in_degree_weighted"
    FLOW_IMBALANCE_RATIO = "flow_imbalance_ratio"
    DIRECTED_TEMPORAL_DRIFT = "directed_temporal_drift"
    OD_FORWARD_SIMILARITY = "od_forward_similarity"
    OD_REVERSE_SIMILARITY = "od_reverse_similarity"
    ORIGIN_OUTBOUND_STRENGTH = "origin_outbound_strength"
    DESTINATION_INBOUND_STRENGTH = "destination_inbound_strength"
    FORWARD_FLOW_VOLUME_30D = "forward_flow_volume_30d"
    REVERSE_FLOW_VOLUME_30D = "reverse_flow_volume_30d"
    DIRECTIONAL_MARKET_DRIFT = "directional_market_drift"
    DIRECTIONAL_ACCEPTANCE_RATE = "directional_acceptance_rate"
    DIRECTIONAL_PRICE_PRESSURE = "directional_price_pressure"


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


@dataclass(frozen=True)
class GraphEncoderConfig:
    """Typed graph encoder configuration for public Python APIs."""

    family: GraphEncoderFamily | str = GraphEncoderFamily.GRAPHSAGE
    input_dim: int | None = None
    dim: int | None = None
    hidden_dims: tuple[int, ...] = (16,)
    walk_length: int = 16
    walks_per_node: int = 8
    window_size: int = 5
    epochs: int = 20
    learning_rate: float = 0.05
    min_learning_rate: float = 0.0001
    negative_samples: int = 4
    p: float = 1.0
    q: float = 1.0
    seed: int | None = None
    l2_regularization: float = 1e-5
    normalize: bool = True
    hetero: bool = False
    add_self_loop: bool = True
    node_type_count: int | None = None
    edge_type_triples: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    neighbor_samples: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | GraphEncoderConfig | None,
    ) -> GraphEncoderConfig:
        if isinstance(config, GraphEncoderConfig):
            return config
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise TypeError("encoder config must be a mapping or GraphEncoderConfig")

        triples = config.get("edge_type_triples", ())
        if triples is None:
            edge_type_triples: tuple[tuple[int, int, int], ...] = ()
        else:
            edge_type_triples = tuple(
                _coerce_int_triple(triple, "edge_type_triples") for triple in triples
            )

        hidden_dims = config.get("hidden_dims", (16,))
        if hidden_dims is None:
            hidden_dims_tuple: tuple[int, ...] = ()
        elif isinstance(hidden_dims, str) or not isinstance(hidden_dims, Sequence):
            raise TypeError("hidden_dims must be a sequence")
        else:
            hidden_dims_tuple = tuple(int(value) for value in hidden_dims)

        neighbor_samples = config.get("neighbor_samples", ())
        if neighbor_samples is None:
            neighbor_samples_tuple: tuple[int, ...] = ()
        elif isinstance(neighbor_samples, str) or not isinstance(
            neighbor_samples,
            Sequence,
        ):
            raise TypeError("neighbor_samples must be a sequence")
        else:
            neighbor_samples_tuple = tuple(int(value) for value in neighbor_samples)

        return cls(
            family=GraphEncoderFamily(
                str(_enum_value(config.get("family", GraphEncoderFamily.GRAPHSAGE)))
            ),
            input_dim=(int(config["input_dim"]) if config.get("input_dim") is not None else None),
            dim=int(config["dim"]) if config.get("dim") is not None else None,
            hidden_dims=hidden_dims_tuple,
            walk_length=int(config.get("walk_length", 16)),
            walks_per_node=int(config.get("walks_per_node", 8)),
            window_size=int(config.get("window_size", 5)),
            epochs=int(config.get("epochs", 20)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            min_learning_rate=float(config.get("min_learning_rate", 0.0001)),
            negative_samples=int(config.get("negative_samples", 4)),
            p=float(config.get("p", 1.0)),
            q=float(config.get("q", 1.0)),
            seed=int(config["seed"]) if config.get("seed") is not None else None,
            l2_regularization=float(config.get("l2_regularization", 1e-5)),
            normalize=bool(config.get("normalize", True)),
            hetero=bool(config.get("hetero", False)),
            add_self_loop=bool(config.get("add_self_loop", True)),
            node_type_count=(
                int(config["node_type_count"])
                if config.get("node_type_count") is not None
                else None
            ),
            edge_type_triples=edge_type_triples,
            neighbor_samples=neighbor_samples_tuple,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family": str(_enum_value(self.family)),
            "hidden_dims": list(self.hidden_dims),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "negative_samples": self.negative_samples,
            "l2_regularization": self.l2_regularization,
            "hetero": self.hetero,
        }
        if self.input_dim is not None:
            payload["input_dim"] = self.input_dim
        if self.dim is not None:
            payload["dim"] = self.dim
        if self.seed is not None:
            payload["seed"] = self.seed
        if str(_enum_value(self.family)) == GraphEncoderFamily.NODE2VEC.value:
            payload.update(
                {
                    "walk_length": self.walk_length,
                    "walks_per_node": self.walks_per_node,
                    "window_size": self.window_size,
                    "min_learning_rate": self.min_learning_rate,
                    "p": self.p,
                    "q": self.q,
                    "normalize": self.normalize,
                }
            )
        if self.add_self_loop is not True:
            payload["add_self_loop"] = self.add_self_loop
        if self.node_type_count is not None:
            payload["node_type_count"] = self.node_type_count
        if self.edge_type_triples:
            payload["edge_type_triples"] = list(self.edge_type_triples)
        if self.neighbor_samples:
            payload["neighbor_samples"] = list(self.neighbor_samples)
        return payload


@dataclass(frozen=True)
class GraphOutputsConfig:
    """Typed graph feature output selection."""

    directional_features: tuple[DirectionalFeature | str, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | GraphOutputsConfig | None,
    ) -> GraphOutputsConfig:
        if isinstance(config, GraphOutputsConfig):
            return config
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise TypeError("outputs config must be a mapping or GraphOutputsConfig")
        features = config.get("directional_features", ())
        if features is None:
            directional_features: tuple[str, ...] = ()
        elif isinstance(features, str):
            directional_features = (features,)
        else:
            directional_features = tuple(str(_enum_value(feature)) for feature in features)
        return cls(directional_features=directional_features)

    def as_dict(self) -> dict[str, Any]:
        if not self.directional_features:
            return {}
        return {
            "directional_features": [
                str(_enum_value(feature)) for feature in self.directional_features
            ]
        }


@dataclass(frozen=True)
class GraphEmbeddingsConfig:
    """Typed replacement for the nested ``graph_embeddings`` mapping."""

    enabled: bool = True
    backend: GraphBackend | str = GraphBackend.NATIVE
    task_mode: GraphTaskMode | str = GraphTaskMode.PRECOMPUTE_FEATURES
    encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    directionality: DirectionalityConfig = field(default_factory=DirectionalityConfig)
    outputs: GraphOutputsConfig = field(default_factory=GraphOutputsConfig)
    graph: GraphSchema | None = None
    metapaths: tuple[DirectedMetaPath, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | GraphEmbeddingsConfig,
    ) -> GraphEmbeddingsConfig:
        if isinstance(config, GraphEmbeddingsConfig):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("graph_embeddings config must be a mapping or GraphEmbeddingsConfig")

        root = config.get("graph_embeddings", config)
        if not isinstance(root, Mapping):
            raise TypeError("graph_embeddings config must be a mapping")

        graph_block = root.get("graph")
        graph_schema = None
        root_directionality = root.get("directionality", {})
        if graph_block:
            if isinstance(graph_block, GraphSchema):
                graph_schema = graph_block.validate()
                if graph_schema.directionality is not None:
                    root_directionality = graph_schema.directionality
            elif isinstance(graph_block, Mapping):
                graph_schema = _graph_schema_from_mapping(
                    graph_block,
                    root_directionality=root_directionality,
                )
            else:
                raise TypeError("graph block must be a mapping or GraphSchema")

        encoder_block = root.get("encoder", root.get("walk_embeddings", {}))
        encoder = GraphEncoderConfig.from_config(encoder_block)
        directionality = DirectionalityConfig.from_config(root_directionality)
        outputs = GraphOutputsConfig.from_config(root.get("outputs", {}))
        metapaths = tuple(
            _coerce_metapath(value) for value in _metapath_values(root, encoder.as_dict())
        )
        if graph_schema is not None:
            for metapath in metapaths:
                metapath.validate(graph_schema)

        return cls(
            enabled=bool(root.get("enabled", True)),
            backend=GraphBackend(str(_enum_value(root.get("backend", GraphBackend.NATIVE)))),
            task_mode=GraphTaskMode(
                str(_enum_value(root.get("task_mode", GraphTaskMode.PRECOMPUTE_FEATURES)))
            ),
            encoder=encoder,
            directionality=directionality,
            outputs=outputs,
            graph=graph_schema,
            metapaths=metapaths,
        )

    def as_graph_embeddings_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "backend": str(_enum_value(self.backend)),
            "task_mode": str(_enum_value(self.task_mode)),
            "encoder": self.encoder.as_dict(),
            "directionality": self.directionality.as_dict(),
            "outputs": self.outputs.as_dict(),
        }
        if self.graph is not None:
            payload["graph"] = {
                "directed": self.graph.directed,
                "node_types": list(self.graph.node_types),
                "edge_types": [edge.as_tuple() for edge in self.graph.edge_types],
                "directionality": (
                    self.graph.directionality.as_dict()
                    if self.graph.directionality is not None
                    else self.directionality.as_dict()
                ),
            }
            if self.graph.timestamp_col is not None:
                payload["graph"]["timestamp_col"] = self.graph.timestamp_col
        if self.metapaths:
            payload["metapaths"] = [list(metapath.steps) for metapath in self.metapaths]
        return {"graph_embeddings": payload}


@dataclass(frozen=True)
class GraphFeatureConfig:
    """Validated graph feature configuration.

    This mirrors the optional YAML-style ``graph_embeddings`` block while
    preserving CartoBoost's current dense-plus-sparse booster contract.
    """

    enabled: bool = True
    backend: GraphBackend | str = GraphBackend.NATIVE
    task_mode: GraphTaskMode | str = GraphTaskMode.PRECOMPUTE_FEATURES
    graph_schema: GraphSchema | None = None
    encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    root_directionality: DirectionalityConfig = field(default_factory=DirectionalityConfig)
    metapaths: tuple[DirectedMetaPath, ...] = field(default_factory=tuple)
    outputs: GraphOutputsConfig = field(default_factory=GraphOutputsConfig)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | GraphEmbeddingsConfig | GraphFeatureConfig,
    ) -> GraphFeatureConfig:
        if isinstance(config, GraphFeatureConfig):
            return config
        graph_embeddings = GraphEmbeddingsConfig.from_config(config)
        return cls(
            enabled=graph_embeddings.enabled,
            backend=graph_embeddings.backend,
            task_mode=graph_embeddings.task_mode,
            graph_schema=graph_embeddings.graph,
            encoder=graph_embeddings.encoder,
            root_directionality=graph_embeddings.directionality,
            metapaths=graph_embeddings.metapaths,
            outputs=graph_embeddings.outputs,
        )

    @property
    def directionality(self) -> DirectionalityConfig:
        if self.graph_schema is None or self.graph_schema.directionality is None:
            return self.root_directionality
        return self.graph_schema.directionality

    def transformer_config(self) -> dict[str, Any]:
        """Return the subset accepted by ``GraphFeatureTransformer.from_config``."""
        encoder = self.encoder.as_dict()
        if "family" not in encoder:
            encoder["family"] = "graphsage"
        return {
            "graph_embeddings": {
                "enabled": self.enabled,
                "backend": str(_enum_value(self.backend)),
                "task_mode": str(_enum_value(self.task_mode)),
                "encoder": encoder,
                "directionality": self.directionality.as_dict(),
                "outputs": self.outputs.as_dict(),
            },
        }


def _coerce_int_triple(value: Any, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} entries must be 3-item sequences")
    if len(value) != 3:
        raise ValueError(f"{field_name} entries must have 3 values")
    return (int(value[0]), int(value[1]), int(value[2]))


def _graph_schema_from_mapping(
    graph_block: Mapping[str, Any],
    *,
    root_directionality: Any = None,
) -> GraphSchema:
    node_types = graph_block.get("node_types", [])
    if isinstance(node_types, str) or not isinstance(node_types, Sequence):
        raise TypeError("graph.node_types must be a sequence")

    edge_values = graph_block.get("edge_types", [])
    if isinstance(edge_values, str) or not isinstance(edge_values, Sequence):
        raise TypeError("graph.edge_types must be a sequence")

    directionality_block = graph_block.get("directionality", root_directionality or {})
    directionality = DirectionalityConfig.from_config(directionality_block)
    return GraphSchema(
        node_types=[str(value) for value in node_types],
        edge_types=as_edge_types([tuple(value) for value in edge_values]),
        directed=bool(graph_block.get("directed", True)),
        timestamp_col=(
            str(graph_block["timestamp_col"]) if graph_block.get("timestamp_col") else None
        ),
        directionality=directionality,
    ).validate()


def _metapath_values(
    root: Mapping[str, Any],
    encoder: Mapping[str, Any],
) -> Sequence[Any]:
    walk_embeddings = root.get("walk_embeddings", {})
    values: Any = ()
    if isinstance(walk_embeddings, Mapping):
        values = walk_embeddings.get("metapaths", walk_embeddings.get("meta_paths", ()))
    if not values:
        values = root.get("metapaths", root.get("meta_paths", ()))
    if not values:
        values = encoder.get("metapaths", encoder.get("meta_paths", ()))
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError("metapaths must be a sequence")
    return values


def _coerce_metapath(value: Any) -> DirectedMetaPath:
    if isinstance(value, DirectedMetaPath):
        return value
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("metapath must be a node/relation/node sequence")
    return DirectedMetaPath(tuple(str(step) for step in value))
