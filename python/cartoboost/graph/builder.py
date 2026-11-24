"""Utilities for normalizing edge streams for local graph encoders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._native import (
    graph_materialize_source_target_pair_nodes as _native_materialize_source_target_pair_nodes,
)


def _ensure_unique(values: Iterable[Any], label: str) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        try:
            if value in seen:
                raise ValueError(f"duplicate {label}: {value!r}")
            seen.add(value)
        except TypeError as exc:
            raise ValueError(
                f"values for {label} must be hashable, got {value!r}",
            ) from exc
        ordered.append(value)
    return ordered


@dataclass(frozen=True)
class HomogeneousGraph:
    """Normalized homogeneous edge list."""

    node_ids: list[Any]
    node_to_index: dict[Any, int]
    node_count: int
    edges: list[tuple[int, int]]
    directed: bool = True


@dataclass(frozen=True)
class HeterogeneousGraph:
    """Normalized typed-edge list."""

    node_ids: list[Any]
    node_to_index: dict[Any, int]
    relation_ids: list[Any]
    relation_to_index: dict[Any, int]
    node_count: int
    relation_count: int
    edges: list[tuple[int, int, int]]
    directed: bool = True


@dataclass(frozen=True)
class SourceTargetPairNodes:
    """Augmented typed edges with materialized source-target pair nodes."""

    edges: list[tuple[Any, Any, Any]]
    pair_node_ids: list[Any]


def normalize_homogeneous_graph(
    edges: Sequence[tuple[Any, Any]],
    *,
    node_ids: Sequence[Any] | None = None,
    node_count: int | None = None,
    directed: bool = True,
) -> HomogeneousGraph:
    """Normalize a homogeneous edge list into integer node indices."""
    if node_ids is not None:
        normal_nodes = _ensure_unique(node_ids, "node id")
        node_to_index = {node: index for index, node in enumerate(normal_nodes)}
        resolved_node_count = len(node_ids)
    elif node_count is not None:
        if node_count <= 0:
            raise ValueError("node_count must be positive when provided")
        normal_nodes = list(range(node_count))
        node_to_index = {node: node for node in normal_nodes}
        resolved_node_count = node_count
    else:
        detected = []
        node_to_index: dict[Any, int] = {}
        for source, target in edges:
            for node in (source, target):
                if node not in node_to_index:
                    node_to_index[node] = len(detected)
                    detected.append(node)
        if not detected:
            raise ValueError("node_count must be provided for empty edge sets")
        normal_nodes = detected
        resolved_node_count = len(detected)

    normalized_edges: list[tuple[int, int]] = []
    for source, target in edges:
        if source not in node_to_index or target not in node_to_index:
            raise ValueError("edge endpoint not represented by node_ids/node_count")
        src = node_to_index[source]
        dst = node_to_index[target]
        normalized_edges.append((src, dst))
        if not directed and src != dst:
            normalized_edges.append((dst, src))

    if normalized_edges == []:
        # Keep an explicit empty graph with no edges for edge-case stability.
        normalized_edges = []

    return HomogeneousGraph(
        node_ids=normal_nodes,
        node_to_index=node_to_index,
        node_count=resolved_node_count,
        edges=normalized_edges,
        directed=directed,
    )


def normalize_heterogeneous_graph(
    edges: Sequence[tuple[Any, Any, Any]],
    *,
    node_ids: Sequence[Any] | None = None,
    relation_ids: Sequence[Any] | None = None,
    node_count: int | None = None,
    directed: bool = True,
    materialize_reverse_edges: bool = False,
    reverse_relation_suffix: str = "_reverse",
    reverse_relation_map: Mapping[Any, Any] | None = None,
) -> HeterogeneousGraph:
    """Normalize a typed edge list into integer node/relation indices."""
    if node_ids is not None:
        normal_nodes = _ensure_unique(node_ids, "node id")
        node_to_index = {node: index for index, node in enumerate(normal_nodes)}
        resolved_node_count = len(node_ids)
    elif node_count is not None:
        if node_count <= 0:
            raise ValueError("node_count must be positive when provided")
        normal_nodes = list(range(node_count))
        node_to_index = {node: node for node in normal_nodes}
        resolved_node_count = node_count
    else:
        detected = []
        node_to_index = {}
        for source, target, _relation in edges:
            for node in (source, target):
                if node not in node_to_index:
                    node_to_index[node] = len(detected)
                    detected.append(node)
        if not detected:
            raise ValueError("node_count must be provided for empty edge sets")
        normal_nodes = detected
        resolved_node_count = len(detected)

    if relation_ids is not None:
        normal_relations = _ensure_unique(relation_ids, "relation id")
        relation_to_index = {rel: index for index, rel in enumerate(normal_relations)}
    else:
        normal_relations = []
        relation_to_index: dict[Any, int] = {}
        for _source, _target, relation in edges:
            if relation not in relation_to_index:
                relation_to_index[relation] = len(normal_relations)
                normal_relations.append(relation)
    resolved_relation_count = len(normal_relations)

    if resolved_relation_count == 0:
        raise ValueError("relation set is empty")

    normalized_edges: list[tuple[int, int, int]] = []
    relation_lookup = reverse_relation_map or {}
    for source, target, relation in edges:
        if source not in node_to_index or target not in node_to_index:
            raise ValueError("edge endpoint not represented by node_ids/node_count")
        if relation not in relation_to_index:
            if relation_ids is not None:
                raise ValueError(f"relation {relation!r} not in relation_ids")
            relation_to_index[relation] = len(normal_relations)
            normal_relations.append(relation)
            resolved_relation_count = len(normal_relations)

        source_idx = node_to_index[source]
        target_idx = node_to_index[target]
        relation_idx = relation_to_index[relation]
        normalized_edges.append((source_idx, target_idx, relation_idx))
        if not directed:
            normalized_edges.append((target_idx, source_idx, relation_idx))
        elif materialize_reverse_edges:
            if relation in relation_lookup:
                reverse_relation = relation_lookup[relation]
            elif isinstance(relation, str):
                reverse_relation = f"{relation}{reverse_relation_suffix}"
            else:
                raise ValueError(
                    f"cannot materialize reverse edge for non-string relation {relation!r}",
                )

            if reverse_relation not in relation_to_index:
                relation_to_index[reverse_relation] = len(normal_relations)
                normal_relations.append(reverse_relation)
                resolved_relation_count = len(normal_relations)
            reverse_relation_idx = relation_to_index[reverse_relation]
            normalized_edges.append((target_idx, source_idx, reverse_relation_idx))

    return HeterogeneousGraph(
        node_ids=normal_nodes,
        node_to_index=node_to_index,
        relation_ids=normal_relations,
        relation_to_index=relation_to_index,
        node_count=resolved_node_count,
        relation_count=resolved_relation_count,
        edges=normalized_edges,
        directed=directed,
    )


def materialize_source_target_pair_nodes(
    edges: Sequence[tuple[Any, Any, Any]],
    *,
    source_to_pair_relation: Any = "source_to_pair",
    pair_to_target_relation: Any = "pair_to_target",
    pair_node_prefix: str = "od_pair",
    include_original_edges: bool = True,
) -> SourceTargetPairNodes:
    """Create explicit source-target pair nodes for directed OD-style facts."""
    endpoint_lookup: dict[str, Any] = {}
    pair_node_lookup: dict[str, tuple[str, str, str]] = {}
    native_input_edges = []
    for source, target, relation in edges:
        source_key = str(source)
        target_key = str(target)
        for key, node in ((source_key, source), (target_key, target)):
            if key in endpoint_lookup and (
                type(endpoint_lookup[key]) is not type(node) or endpoint_lookup[key] != node
            ):
                raise ValueError("distinct graph node IDs have colliding string representations")
            endpoint_lookup[key] = node
        pair_id = f"{pair_node_prefix}:{source_key}:{target_key}"
        logical_pair = (pair_node_prefix, source_key, target_key)
        if pair_id in pair_node_lookup and pair_node_lookup[pair_id] != logical_pair:
            raise ValueError("distinct source-target pairs render the same pair node ID")
        pair_node_lookup[pair_id] = logical_pair
        native_input_edges.append((source_key, target_key, str(relation)))

    if pair_node_lookup.keys() & endpoint_lookup.keys():
        raise ValueError("generated pair node ID collides with an original node ID")

    native_edges, native_pair_ids = _native_materialize_source_target_pair_nodes(
        native_input_edges,
        str(source_to_pair_relation),
        str(pair_to_target_relation),
        str(pair_node_prefix),
        bool(include_original_edges),
    )
    augmented: list[tuple[Any, Any, Any]] = []
    for source, target, relation in native_edges:
        augmented.append(
            (
                pair_node_lookup.get(source, endpoint_lookup.get(source, source)),
                pair_node_lookup.get(target, endpoint_lookup.get(target, target)),
                relation,
            )
        )
    return SourceTargetPairNodes(
        edges=augmented,
        pair_node_ids=[pair_node_lookup[pair_id] for pair_id in native_pair_ids],
    )


def ensure_node_features_shape(
    features: Iterable[Sequence[float]],
    node_count: int,
) -> list[list[float]]:
    """Validate and materialize a dense feature table."""
    dense = [list(row) for row in features]
    if len(dense) != node_count:
        raise ValueError(
            f"expected {node_count} feature rows, got {len(dense)}",
        )
    return dense
