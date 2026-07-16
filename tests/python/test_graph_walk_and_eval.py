from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from cartoboost.graph import (
    DirectedMetaPath,
    DirectionalFeature,
    DirectionalityConfig,
    EdgeType,
    GraphEmbeddingsConfig,
    GraphEncoderConfig,
    GraphEncoderFamily,
    GraphFeatureConfig,
    GraphFeatureTransformer,
    GraphOutputsConfig,
    GraphSchema,
    HinSageFeatureEncoder,
    MetaPathWalkGenerator,
    Node2VecFeatureEncoder,
    SignedEdgeSampler,
    TemporalWalkGenerator,
    binary_auc,
    binary_average_precision,
    link_prediction_report,
    materialize_source_target_pair_nodes,
    mean_reciprocal_rank,
    normalize_heterogeneous_graph,
    top_k_metrics,
)


def test_meta_path_generator_respects_relation_constraints() -> None:
    edges = [
        (0, 1, "pickup-trip"),
        (1, 2, "trip-dropoff"),
        (2, 3, "dropoff-pickup"),
    ]
    generator = MetaPathWalkGenerator(
        metapath=("pickup-trip", "trip-dropoff", "dropoff-pickup"),
        walk_length=4,
        walks_per_node=2,
        seed=7,
    )
    walks = generator.generate(edges=edges, start_nodes=[0, 1])
    assert len(walks) == 4
    for walk in walks:
        assert walk[0] in {0, 1}


def test_directed_metapath_validates_schema_and_drives_relation_walk() -> None:
    schema = GraphSchema(
        node_types=["pickup_zone", "dropoff_zone", "taxi_trip"],
        edge_types=[
            EdgeType("taxi_trip", "picked_up_in", "pickup_zone"),
            EdgeType("pickup_zone", "flows_to", "dropoff_zone"),
        ],
        directionality=DirectionalityConfig(preserve_source_target_roles=True),
    ).validate()
    path = DirectedMetaPath(
        (
            "taxi_trip",
            "picked_up_in",
            "pickup_zone",
            "flows_to",
            "dropoff_zone",
        ),
    ).validate(schema)

    generator = MetaPathWalkGenerator(metapath=path, walk_length=3, walks_per_node=1)
    walks = generator.generate(
        start_nodes=[0],
        edges=[(0, 1, "picked_up_in"), (1, 2, "flows_to")],
    )
    assert walks == [[0, 1, 2]]


def test_directionality_config_rejects_undirected_source_target_roles() -> None:
    schema = GraphSchema(
        node_types=["market"],
        edge_types=[EdgeType("market", "connected_to", "market")],
        directed=False,
        directionality=DirectionalityConfig(preserve_source_target_roles=True),
    )
    try:
        schema.validate()
    except ValueError as error:
        assert "requires a directed graph schema" in str(error)
    else:
        raise AssertionError("expected source-target role validation to fail")


def test_temporal_walk_generator_enforces_time_monotonicity() -> None:
    edges = [
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 3, 3.0),
        (2, 4, 4.0),
    ]
    generator = TemporalWalkGenerator(walk_length=4, walks_per_node=1, seed=13)
    walks = generator.generate(temporal_edges=edges, start_nodes=[0])
    assert walks
    for walk in walks:
        assert walk[0] == 0
        assert walk[-1] in {1, 2, 3, 4}


def test_signed_sampler_avoids_false_negatives() -> None:
    edges = [(0, 1), (0, 2)]
    sampler = SignedEdgeSampler(negative_samples=2, seed=3)
    sampled = sampler.sample(node_count=4, edges=edges)
    sampled_edge_set = {edge for values in sampled.values() for edge in values}
    assert not sampled_edge_set.intersection(set(edges))


def test_signed_sampler_returns_fewer_negatives_when_source_is_dense() -> None:
    edges = [(0, 0), (0, 1), (0, 2)]
    sampler = SignedEdgeSampler(negative_samples=2, seed=3)

    sampled = sampler.sample(node_count=3, edges=edges)

    assert sampled == {edge: [] for edge in edges}


def test_graph_metrics_match_simple_reference() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2]
    assert binary_auc(labels, scores) > 0.8
    assert binary_average_precision(labels, scores) > 0.9

    top_k = top_k_metrics(labels, scores, query_ids=["a", "a", "b", "b"], k=1)
    assert top_k["recall_at_k"] == 1.0
    assert top_k["hit_rate_at_k"] == 1.0

    mrr = mean_reciprocal_rank(labels, scores, query_ids=["a", "a", "b", "b"])
    assert mrr > 0.8

    report = link_prediction_report(labels, scores, query_ids=["a", "a", "b", "b"], k=1)
    assert report["auc"] > 0.8
    assert report["average_precision"] > 0.9
    assert report["mean_reciprocal_rank"] > 0.8


def test_graph_feature_transformer_homogeneous_fit_transform_smoke() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {"graph_sage": {"hidden_dims": [2], "epochs": 2, "input_dim": 2}},
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
    )
    assert bundle.embeddings.shape == (3, 2)


def test_graph_feature_transformer_rejects_mismatched_input_dim() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {"graph_sage": {"input_dim": 3, "hidden_dims": [2], "epochs": 2}},
    )
    try:
        transformer.fit_transform(node_features=features, edges=edges, node_count=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected input_dim mismatch to raise")


def test_native_node2vec_feature_encoder_fits_directed_weighted_graph() -> None:
    encoder = Node2VecFeatureEncoder.from_config(
        {
            "dim": 4,
            "walk_length": 5,
            "walks_per_node": 3,
            "window_size": 2,
            "epochs": 2,
            "negative_samples": 2,
            "seed": 13,
        },
    )
    bundle = encoder.fit(
        edges=[(0, 1), (1, 2), (2, 0), (2, 3)],
        node_count=4,
        directed=True,
        edge_weights=[1.0, 2.0, 1.0, 3.0],
    )

    assert bundle.embeddings.shape == (4, 4)
    assert bundle.feature_names == [
        "node2vec_00",
        "node2vec_01",
        "node2vec_02",
        "node2vec_03",
    ]
    assert bundle.provenance["encoder"] == "node2vec"
    assert bundle.provenance["native"] == "rust"
    assert len(encoder.loss_curve()) == 2


def test_native_node2vec_encode_returns_fitted_embeddings_and_node_ids() -> None:
    encoder = Node2VecFeatureEncoder.from_config(
        {
            "dim": 3,
            "walk_length": 4,
            "walks_per_node": 2,
            "window_size": 1,
            "epochs": 1,
            "negative_samples": 1,
            "seed": 19,
        },
    )
    fitted = encoder.fit(
        edges=[("a", "b"), ("b", "c")],
        node_ids=["a", "b", "c"],
        directed=True,
    )
    encoded = encoder.encode()

    assert encoded.node_ids == ["a", "b", "c"]
    assert encoded.feature_names == fitted.feature_names
    np.testing.assert_allclose(encoded.embeddings, fitted.embeddings)


def test_native_node2vec_rejects_invalid_config_and_inputs() -> None:
    try:
        Node2VecFeatureEncoder.from_config({"dim": 0})
    except ValueError as error:
        assert "dim must be positive" in str(error)
    else:
        raise AssertionError("expected invalid node2vec dim to raise")

    encoder = Node2VecFeatureEncoder.from_config(
        {
            "dim": 2,
            "walk_length": 4,
            "walks_per_node": 1,
            "window_size": 1,
            "epochs": 1,
            "negative_samples": 1,
        },
    )
    try:
        encoder.encode()
    except RuntimeError as error:
        assert "must be fitted" in str(error)
    else:
        raise AssertionError("expected encode before fit to raise")

    try:
        encoder.fit(edges=[(0, 1)], node_count=2, edge_weights=[1.0, 2.0])
    except ValueError as error:
        assert "edge value length" in str(error)
    else:
        raise AssertionError("expected edge weight length mismatch to raise")

    try:
        encoder.fit(edges=[(0, 1)], node_count=2, edge_weights=[-1.0])
    except ValueError as error:
        assert "edge_weights" in str(error)
    else:
        raise AssertionError("expected negative edge weight to raise")


def test_native_node2vec_writes_artifact_json() -> None:
    encoder = Node2VecFeatureEncoder.from_config(
        {
            "dim": 2,
            "walk_length": 4,
            "walks_per_node": 2,
            "window_size": 1,
            "epochs": 1,
            "negative_samples": 1,
            "seed": 23,
        },
    )
    encoder.fit(edges=[(0, 1), (1, 2)], node_count=3)

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = f"{directory}/node2vec.json"
        encoder.save_artifact_json(artifact_path)
        with open(artifact_path, encoding="utf-8") as file:
            payload = json.loads(file.read())

    assert payload["artifact_type"] == "cartoboost.neural.node2vec_encoder"
    assert payload["artifact_version"] == 1
    assert payload["node_count"] == 3
    assert payload["output_dim"] == 2
    assert len(payload["embeddings"]) == 3


def test_graph_feature_transformer_node2vec_undirected_weights_expand() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "node2vec",
                    "dim": 2,
                    "walk_length": 4,
                    "walks_per_node": 1,
                    "window_size": 1,
                    "epochs": 1,
                    "negative_samples": 1,
                    "seed": 29,
                },
                "outputs": {
                    "directional_features": [
                        "source_outbound_strength",
                        "target_inbound_strength",
                    ],
                },
                "directionality": {
                    "preserve_source_target_roles": False,
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=[[0.1], [0.2]],
        edges=[(0, 1)],
        node_count=2,
        directed=False,
        edge_weights=[7.0],
    )

    assert bundle.provenance["directed"] is False
    assert bundle.embeddings.shape == (2, 4)
    assert list(bundle.embeddings[:, -2]) == [7.0, 7.0]
    assert list(bundle.embeddings[:, -1]) == [7.0, 7.0]


def test_graph_feature_transformer_node2vec_augments_booster_dense_schema() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "node2vec",
                    "dim": 3,
                    "walk_length": 4,
                    "walks_per_node": 2,
                    "window_size": 1,
                    "epochs": 1,
                    "negative_samples": 1,
                    "seed": 7,
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=[[0.1], [0.2], [0.3]],
        edges=[(0, 1), (1, 2)],
        node_count=3,
        directed=True,
    )
    dense = bundle.augment_dense([[1.0], [2.0], [3.0]])

    assert bundle.embeddings.shape == (3, 3)
    assert dense.shape == (3, 4)
    assert bundle.feature_schema_payload()["names"] == [
        "node2vec_00",
        "node2vec_01",
        "node2vec_02",
    ]
    assert bundle.feature_schema_payload()["kinds"] == ["Numeric", "Numeric", "Numeric"]


def test_graph_feature_transformer_node2vec_emits_directional_features() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "node2vec",
                    "dim": 2,
                    "walk_length": 4,
                    "walks_per_node": 2,
                    "window_size": 1,
                    "epochs": 1,
                    "negative_samples": 1,
                    "seed": 11,
                },
                "outputs": {
                    "directional_features": [
                        "source_outbound_strength",
                        "target_inbound_strength",
                        "flow_imbalance_ratio",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=[[0.1], [0.2], [0.3]],
        edges=[(0, 1), (1, 0), (0, 2)],
        node_count=3,
        directed=True,
        edge_weights=[2.0, 5.0, 3.0],
    )

    assert bundle.feature_names[-3:] == [
        "graph_source_outbound_strength",
        "graph_target_inbound_strength",
        "graph_flow_imbalance_ratio",
    ]
    assert bundle.embeddings.shape == (3, 5)


def test_graph_transformer_accepts_graph_embeddings_config_namespace() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "hetero": False,
                    "input_dim": 2,
                    "hidden_dims": [2],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
    )
    assert bundle.embeddings.shape == (3, 2)


def test_graph_transformer_accepts_dataclass_graph_embeddings_config() -> None:
    transformer = GraphFeatureTransformer.from_config(
        GraphEmbeddingsConfig(
            encoder=GraphEncoderConfig(
                family=GraphEncoderFamily.HINSAGE,
                input_dim=2,
                node_type_count=2,
                edge_type_triples=((0, 0, 1), (1, 1, 0)),
                hidden_dims=(2,),
                epochs=2,
                neighbor_samples=(1, 0),
            ),
            directionality=DirectionalityConfig(
                preserve_source_target_roles=True,
                compute_asymmetry_features=True,
                directional_features=(DirectionalFeature.SOURCE_TARGET_EMBEDDING,),
            ),
        )
    )
    bundle = transformer.fit_transform(
        node_features=[[1.0, 0.0], [0.0, 1.0], [0.4, 0.2]],
        edges=[(0, 1, 0), (0, 2, 0), (1, 0, 1)],
        node_types=[0, 1, 1],
    )

    assert bundle.embeddings.shape[0] == 3
    assert bundle.provenance["encoder"] == "hinsage"
    assert bundle.feature_names[-1] == "graph_source_target_embedding"


def test_graph_feature_transformer_directional_features_disabled_by_default() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
    )
    assert bundle.embeddings.shape == (3, 2)
    assert len(bundle.feature_names) == 2
    assert "dir_source_target_affinity" not in bundle.feature_names


def test_graph_feature_transformer_directional_features_enabled() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7], [0.4, 0.6]]
    edges = [(0, 1), (1, 2), (2, 1)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_feature_prefix": "dir",
                    "directional_features": [
                        "dir_source_target_affinity",
                        "dir_target_source_affinity",
                        "dir_flow_imbalance_ratio",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=4,
        directed=True,
    )
    assert bundle.embeddings.shape == (4, 5)
    assert len(bundle.feature_names) == 5
    assert bundle.feature_names == [
        "graph_sage_homo_00",
        "graph_sage_homo_01",
        "dir_source_target_affinity",
        "dir_target_source_affinity",
        "dir_flow_imbalance_ratio",
    ]


def test_graph_feature_transformer_directional_features_use_weights_and_time() -> None:
    features = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    edges = [(0, 1), (1, 0), (0, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_features": [
                        "graph_forward_flow_weight",
                        "graph_reverse_flow_weight",
                        "graph_directed_temporal_drift",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        edge_weights=[2.0, 5.0, 3.0],
        edge_timestamps=[10.0, 20.0, 40.0],
    )
    forward_idx = bundle.feature_names.index("graph_forward_flow_weight")
    reverse_idx = bundle.feature_names.index("graph_reverse_flow_weight")
    drift_idx = bundle.feature_names.index("graph_directed_temporal_drift")

    assert bundle.embeddings[0, forward_idx] == 5.0
    assert bundle.embeddings[0, reverse_idx] == 5.0
    assert bundle.embeddings[0, drift_idx] == 8.0


def test_graph_feature_transformer_directional_features_reject_invalid_names() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_features": ["unknown_feature"],
                },
            },
        },
    )
    try:
        transformer.fit_transform(node_features=features, edges=edges, node_count=3)
    except ValueError as error:
        assert "no recognized feature names" in str(error)
    else:
        raise AssertionError("expected invalid directional feature selection to raise")


def test_graph_feature_transformer_outputs_directional_features_contract() -> None:
    features = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    edges = [(0, 1), (1, 0), (0, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {"directional_feature_prefix": "graph"},
                "outputs": {
                    "directional_features": [
                        "source_target_embedding",
                        "target_source_embedding",
                        "forward_reverse_similarity_delta",
                        "source_outbound_strength",
                        "target_inbound_strength",
                        "flow_imbalance_ratio",
                        "directed_temporal_drift",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        edge_weights=[2.0, 5.0, 3.0],
        edge_timestamps=[10.0, 20.0, 40.0],
    )
    assert bundle.feature_names[-7:] == [
        "graph_source_target_embedding",
        "graph_target_source_embedding",
        "graph_forward_reverse_similarity_delta",
        "graph_source_outbound_strength",
        "graph_target_inbound_strength",
        "graph_flow_imbalance_ratio",
        "graph_directed_temporal_drift",
    ]


def test_graph_feature_transformer_emits_geotemporal_directional_aliases() -> None:
    features = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    edges = [(0, 1), (1, 0), (0, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_features": [
                        "graph_od_forward_similarity",
                        "graph_od_reverse_similarity",
                        "graph_origin_outbound_strength",
                        "graph_destination_inbound_strength",
                        "graph_forward_flow_volume_30d",
                        "graph_reverse_flow_volume_30d",
                        "graph_directional_market_drift",
                        "graph_directional_acceptance_rate",
                        "graph_directional_price_pressure",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        edge_weights=[2.0, 5.0, 3.0],
        edge_timestamps=[10.0, 20.0, 40.0],
    )
    assert bundle.feature_names[-9:] == [
        "graph_od_forward_similarity",
        "graph_od_reverse_similarity",
        "graph_origin_outbound_strength",
        "graph_destination_inbound_strength",
        "graph_forward_flow_volume_30d",
        "graph_reverse_flow_volume_30d",
        "graph_directional_market_drift",
        "graph_directional_acceptance_rate",
        "graph_directional_price_pressure",
    ]


def test_graph_feature_transformer_hetero_directional_features_enabled() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [
        (0, 1, "pickup_trip"),
        (1, 2, "trip_dropoff"),
    ]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "hetero": True,
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {"compute_asymmetry_features": True},
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        directed=True,
    )
    assert bundle.embeddings.shape[0] == 3
    assert bundle.embeddings.shape[1] == len(bundle.feature_names)
    assert bundle.feature_names[0] == "graph_sage_hetero_00"
    assert "graph_source_target_embedding" in bundle.feature_names
    assert "graph_target_source_embedding" in bundle.feature_names
    assert "graph_forward_reverse_similarity_delta" in bundle.feature_names
    assert sum(name.startswith("graph_") for name in bundle.feature_names) >= 14


def test_normalize_heterogeneous_graph_supports_reverse_relation_materialization() -> None:
    graph = normalize_heterogeneous_graph(
        edges=[(0, 1, "pickup_to_dropoff"), (1, 2, "pickup_to_dropoff")],
        directed=True,
        materialize_reverse_edges=True,
        reverse_relation_suffix="_reverse",
    )
    assert len(graph.edges) == 4
    assert "pickup_to_dropoff" in graph.relation_ids
    assert "pickup_to_dropoff_reverse" in graph.relation_ids

    forward_relation = graph.relation_to_index["pickup_to_dropoff"]
    reverse_relation = graph.relation_to_index["pickup_to_dropoff_reverse"]
    assert (0, 1, forward_relation) in graph.edges
    assert (1, 0, reverse_relation) in graph.edges
    assert (2, 1, reverse_relation) in graph.edges


def test_source_target_pair_nodes_keep_forward_and_reverse_distinct() -> None:
    materialized = materialize_source_target_pair_nodes(
        [
            ("Chicago", "Dallas", "flows_to"),
            ("Dallas", "Chicago", "flows_to"),
        ],
    )
    assert ("od_pair", "Chicago", "Dallas") in materialized.pair_node_ids
    assert ("od_pair", "Dallas", "Chicago") in materialized.pair_node_ids
    assert (
        "Chicago",
        ("od_pair", "Chicago", "Dallas"),
        "source_to_pair",
    ) in materialized.edges
    assert (
        ("od_pair", "Dallas", "Chicago"),
        "Chicago",
        "pair_to_target",
    ) in materialized.edges


def test_graph_feature_transformer_can_create_od_pair_nodes_with_zero_features() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    node_ids = ["taxi_trip_a", "pickup", "dropoff"]
    edges = [
        ("taxi_trip_a", "pickup", "picked_up_in"),
        ("pickup", "dropoff", "flows_to"),
    ]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "hetero": True,
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "create_od_pair_nodes": True,
                    "compute_asymmetry_features": True,
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_ids=node_ids,
        directed=True,
    )
    assert ("od_pair", "pickup", "dropoff") in bundle.node_ids
    assert bundle.embeddings.shape[0] == 5
    assert "graph_flow_imbalance_ratio" in bundle.feature_names


def test_graph_feature_config_parses_directional_yaml_shape() -> None:
    config = GraphFeatureConfig.from_config(
        {
            "graph_embeddings": {
                "enabled": True,
                "backend": "native",
                "task_mode": "precompute_features",
                "graph": {
                    "directed": True,
                    "node_types": [
                        "pickup_zone",
                        "dropoff_zone",
                        "od_pair",
                        "entity",
                        "time_bucket",
                    ],
                    "edge_types": [
                        ["pickup_zone", "flows_to", "dropoff_zone"],
                        ["dropoff_zone", "reverse_flows_to", "pickup_zone"],
                        ["entity", "active_from", "pickup_zone"],
                        ["entity", "active_to", "dropoff_zone"],
                        ["entity", "observed_on", "od_pair"],
                    ],
                    "directionality": {
                        "materialize_reverse_edges": True,
                        "preserve_source_target_roles": True,
                        "create_od_pair_nodes": True,
                        "compute_asymmetry_features": True,
                    },
                },
                "walk_embeddings": {
                    "algorithm": "metapath2vec",
                    "metapaths": [
                        [
                            "pickup_zone",
                            "flows_to",
                            "dropoff_zone",
                            "reverse_flows_to",
                            "pickup_zone",
                        ],
                    ],
                },
                "outputs": {
                    "directional_features": [
                        "source_target_embedding",
                        "target_source_embedding",
                    ],
                },
            },
        },
    )

    assert config.graph_schema is not None
    assert config.graph_schema.directed is True
    assert config.directionality.materialize_reverse_edges is True
    assert config.directionality.create_od_pair_nodes is True
    assert config.metapaths[0].relations == ("flows_to", "reverse_flows_to")
    assert config.transformer_config()["graph_embeddings"]["directionality"][
        "compute_asymmetry_features"
    ]


def test_graph_feature_config_accepts_dataclass_config() -> None:
    config = GraphFeatureConfig.from_config(
        GraphEmbeddingsConfig(
            encoder=GraphEncoderConfig(
                family="graphsage",
                input_dim=2,
                hidden_dims=(2,),
                epochs=2,
            ),
            directionality=DirectionalityConfig(
                compute_asymmetry_features=True,
                directional_feature_prefix="graph",
            ),
            outputs=GraphOutputsConfig(
                directional_features=(DirectionalFeature.SOURCE_TARGET_EMBEDDING,),
            ),
        )
    )

    assert config.encoder.family == "graphsage"
    transformer_config = config.transformer_config()["graph_embeddings"]
    assert transformer_config["encoder"]["input_dim"] == 2
    assert transformer_config["outputs"]["directional_features"] == ["source_target_embedding"]


def test_graph_backend_accepts_all_accelerator_names() -> None:
    for backend in (
        "auto",
        "cpu",
        "cuda",
        "rocm",
        "hip",
        "metal",
        "directml",
        "dml",
        "webgpu",
        "native",
    ):
        config = GraphEmbeddingsConfig.from_config(
            {
                "graph_embeddings": {
                    "backend": backend,
                    "encoder": {"family": "graphsage", "input_dim": 2},
                }
            }
        )
        assert config.backend.value == backend


def test_top_level_graph_backend_propagates_to_accelerated_encoder() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "backend": "webgpu",
                "encoder": {"family": "graphsage", "input_dim": 2},
            }
        }
    )
    assert transformer.sage_kwargs["backend"] == "webgpu"

    native_transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "backend": "native",
                "encoder": {"family": "graphsage", "input_dim": 2},
            }
        }
    )
    assert native_transformer.sage_kwargs["backend"] == "cpu"


def test_node2vec_rejects_non_cpu_top_level_backend_until_fused_kernel_exists() -> None:
    with pytest.raises(ValueError, match="fused accelerator kernel"):
        GraphFeatureTransformer.from_config(
            {
                "graph_embeddings": {
                    "backend": "cuda",
                    "encoder": {"family": "node2vec"},
                }
            }
        )


def test_graph_feature_config_preserves_node2vec_settings_and_root_directionality() -> None:
    config = GraphFeatureConfig.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "node2vec",
                    "dim": 6,
                    "walk_length": 7,
                    "walks_per_node": 3,
                    "window_size": 2,
                    "epochs": 2,
                    "negative_samples": 2,
                    "p": 0.5,
                    "q": 2.0,
                    "normalize": False,
                },
                "directionality": {"compute_asymmetry_features": True},
            },
        },
    )
    transformer_config = config.transformer_config()["graph_embeddings"]

    assert transformer_config["encoder"]["family"] == "node2vec"
    assert transformer_config["encoder"]["dim"] == 6
    assert transformer_config["encoder"]["walk_length"] == 7
    assert transformer_config["encoder"]["walks_per_node"] == 3
    assert transformer_config["encoder"]["window_size"] == 2
    assert transformer_config["encoder"]["p"] == 0.5
    assert transformer_config["encoder"]["q"] == 2.0
    assert transformer_config["encoder"]["normalize"] is False
    assert transformer_config["directionality"]["compute_asymmetry_features"] is True


def test_graph_feature_bundle_exposes_schema_and_training_metadata() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_sage": {
                "input_dim": 2,
                "hidden_dims": [2],
                "epochs": 2,
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=[[0.1, 0.2], [0.2, 0.3]],
        edges=[(0, 1)],
        node_count=2,
    )

    assert bundle.feature_schema_payload()["names"] == [
        "graph_sage_homo_00",
        "graph_sage_homo_01",
    ]
    assert bundle.feature_schema_payload()["kinds"] == ["Numeric", "Numeric"]
    assert bundle.training_config_metadata()["provenance"]["encoder"] == "graphsage"


def test_native_hinsage_feature_encoder_requires_typed_schema_and_builds_link_features() -> None:
    encoder = HinSageFeatureEncoder.from_config(
        {
            "input_dim": 2,
            "node_type_count": 3,
            "edge_type_triples": [(0, 0, 2), (1, 1, 2), (2, 2, 0)],
            "hidden_dims": [3],
            "epochs": 3,
            "negative_samples": 1,
            "neighbor_samples": [2, 0, 0],
        }
    )
    node_types = [0, 1, 2, 2, 2]
    edges = [
        (0, 2, 0),
        (0, 3, 0),
        (0, 4, 0),
        (1, 2, 1),
        (2, 0, 2),
    ]
    features = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.2],
        [0.7, 0.4],
        [0.9, 0.6],
    ]

    bundle = encoder.fit(features, edges, node_types)
    link_bundle = encoder.link_embeddings(bundle.embeddings, [(0, 2), (1, 2)])

    assert bundle.embeddings.shape == (5, 3)
    assert bundle.provenance["encoder"] == "hinsage"
    assert bundle.provenance["neighbor_samples"] == [2, 0, 0]
    assert link_bundle.embeddings.shape == (2, 12)
    assert link_bundle.feature_names[0] == "hinsage_link_00"
    assert len(encoder.loss_curve()) == 3


def test_graph_feature_transformer_uses_true_hinsage_when_schema_is_provided() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "input_dim": 2,
                    "node_type_count": 2,
                    "edge_type_triples": [(0, 0, 1), (1, 1, 0)],
                    "hidden_dims": [2],
                    "epochs": 2,
                    "neighbor_samples": [1, 0],
                }
            }
        }
    )
    bundle = transformer.fit_transform(
        node_features=[[1.0, 0.0], [0.0, 1.0], [0.4, 0.2]],
        edges=[(0, 1, 0), (0, 2, 0), (1, 0, 1)],
        node_types=[0, 1, 1],
    )

    assert bundle.embeddings.shape == (3, 2)
    assert bundle.provenance["encoder"] == "hinsage"
