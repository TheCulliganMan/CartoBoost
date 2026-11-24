from __future__ import annotations

import tempfile

import cartoboost.graph as cb_graph
import cartoboost.neural as cb_neural
import numpy as np
import pytest
from cartoboost import (
    GraphSageLinkPredictor,
    GraphSageStandaloneRegressor,
    HeteroGraphSageLinkPredictor,
    HeteroGraphSageStandaloneRegressor,
    HinSageLinkPredictor,
    HinSageStandaloneRegressor,
    NeuralEmbeddingStandaloneRegressor,
    Node2VecEncoder,
    Node2VecLinkPredictor,
    Node2VecStandaloneRegressor,
)


def _node_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )


def _graph_fixture():
    return {
        "node_features": _node_features(),
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "typed_edges": [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)],
        "node_types": [0, 1, 0, 1],
        "row_nodes": np.array([0, 1, 2, 3]),
        "row_targets": np.array([1, 2, 3, 0]),
        "y": np.array([1.2, 1.7, 2.1, 0.5]),
        "edge_type_triples": [(0, 0, 1), (1, 1, 0)],
    }


def test_graph_and_neural_exposure_surfaces() -> None:
    assert cb_graph.Node2VecStandaloneRegressor is Node2VecStandaloneRegressor
    assert cb_graph.GraphSageStandaloneRegressor is GraphSageStandaloneRegressor
    assert cb_graph.HeteroGraphSageStandaloneRegressor is HeteroGraphSageStandaloneRegressor
    assert cb_graph.HinSageStandaloneRegressor is HinSageStandaloneRegressor
    assert cb_graph.Node2VecLinkPredictor is Node2VecLinkPredictor
    assert cb_graph.GraphSageLinkPredictor is GraphSageLinkPredictor
    assert cb_graph.HeteroGraphSageLinkPredictor is HeteroGraphSageLinkPredictor
    assert cb_graph.HinSageLinkPredictor is HinSageLinkPredictor
    assert cb_neural.NeuralEmbeddingStandaloneRegressor is NeuralEmbeddingStandaloneRegressor
    assert Node2VecEncoder(dim=2).output_dim == 2
    assert cb_graph.Node2VecEncoder is Node2VecEncoder
    from cartoboost import _native

    for name in ("GraphSageEncoder", "HeteroGraphSageEncoder", "HinSageEncoder"):
        assert getattr(cb_graph, name) is getattr(_native, name)


def test_neural_embedding_standalone_regressor_predicts_and_reloads() -> None:
    ids = np.array([1, 2, 1, 3, 2, 3], dtype=np.uint64)
    dense = np.column_stack([ids.astype(float) / 3.0])
    y = np.array([1.0, 2.0, 1.1, 3.0, 2.1, 3.2])
    model = NeuralEmbeddingStandaloneRegressor(
        dim=3,
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=1,
        random_state=7,
    ).fit(ids, y, dense=dense)

    pred = model.predict(ids, dense=dense)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/neural.json"
        model.save(path)
        loaded = NeuralEmbeddingStandaloneRegressor.load(path)
        np.testing.assert_allclose(loaded.predict(ids, dense=dense), pred)


def test_standalone_regressors_reject_non_finite_inputs() -> None:
    model = NeuralEmbeddingStandaloneRegressor(
        dim=2,
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
    )

    with pytest.raises(ValueError, match="y must contain only finite values"):
        model.fit([1, 2], [1.0, np.nan])

    with pytest.raises(ValueError, match="dense must contain only finite values"):
        model.fit([1, 2], [1.0, 2.0], dense=[[0.0], [np.inf]])

    graph = GraphSageStandaloneRegressor(
        input_dim=2,
        hidden_dims=(2,),
        epochs=1,
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
    )
    with pytest.raises(ValueError, match="node_features must contain only finite values"):
        graph.fit(
            node_features=[[1.0, 0.0], [np.nan, 1.0]],
            edges=[(0, 1)],
            row_nodes=[0],
            y=[1.0],
        )


def test_node2vec_standalone_pair_regressor_predicts() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    row_nodes = np.array([0, 1, 2, 3])
    row_targets = np.array([1, 2, 3, 0])
    dense = np.column_stack([row_nodes.astype(float)])
    y = np.array([1.0, 1.5, 2.0, 0.7])

    model = Node2VecStandaloneRegressor(
        dim=4,
        walk_length=5,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=11,
    ).fit(
        node_count=4,
        edges=edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        dense=dense,
        y=y,
    )

    pred = model.predict(row_nodes, row_targets=row_targets, dense=dense)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_graphsage_standalone_pair_regressor_predicts() -> None:
    node_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    row_nodes = np.array([0, 1, 2, 3])
    row_targets = np.array([1, 2, 3, 0])
    y = np.array([1.2, 1.7, 2.1, 0.5])

    model = GraphSageStandaloneRegressor(
        input_dim=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=13,
    ).fit(
        node_features=node_features,
        edges=edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        y=y,
    )

    pred = model.predict(
        node_features=node_features,
        row_nodes=row_nodes,
        row_targets=row_targets,
    )
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_hetero_and_hinsage_standalone_regressors_predict_and_reload() -> None:
    node_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )
    typed_edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]
    row_nodes = np.array([0, 1, 2, 3])
    row_targets = np.array([1, 2, 3, 0])
    y = np.array([1.2, 1.7, 2.1, 0.5])

    hetero = HeteroGraphSageStandaloneRegressor(
        input_dim=2,
        relation_count=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=23,
    ).fit(
        node_features=node_features,
        edges=typed_edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        y=y,
    )
    hetero_pred = hetero.predict(
        node_features=node_features,
        row_nodes=row_nodes,
        row_targets=row_targets,
    )
    assert hetero_pred.shape == y.shape
    assert np.all(np.isfinite(hetero_pred))

    hinsage = HinSageStandaloneRegressor(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=[(0, 0, 1), (1, 1, 0)],
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=29,
    ).fit(
        node_features=node_features,
        node_types=[0, 1, 0, 1],
        edges=typed_edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        y=y,
    )
    pred = hinsage.predict(
        node_features=node_features,
        row_nodes=row_nodes,
        row_targets=row_targets,
    )
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/hinsage.json"
        hinsage.save(path)
        loaded = HinSageStandaloneRegressor.load(path)
        np.testing.assert_allclose(
            loaded.predict(
                node_features=node_features,
                row_nodes=row_nodes,
                row_targets=row_targets,
            ),
            pred,
        )


def test_standalone_link_predictors_report_metrics() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    candidates = [(0, 1), (0, 2), (1, 2), (1, 3)]
    labels = [1, 0, 1, 0]

    node2vec = Node2VecLinkPredictor(
        dim=4,
        walk_length=5,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        negative_samples=1,
        seed=17,
    ).fit(node_count=4, edges=edges)
    node2vec_report = node2vec.report(candidates, labels, query_ids=[0, 0, 1, 1], k=1)
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(node2vec_report)

    node_features = np.eye(4, 2, dtype=np.float32)
    sage = GraphSageLinkPredictor(
        input_dim=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=19,
    ).fit(node_features=node_features, edges=edges)
    sage_report = sage.report(
        node_features=node_features,
        pairs=candidates,
        labels=labels,
        query_ids=[0, 0, 1, 1],
        k=1,
    )
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(sage_report)

    typed_edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]
    hetero = HeteroGraphSageLinkPredictor(
        input_dim=2,
        relation_count=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=31,
    ).fit(node_features=node_features, edges=typed_edges)
    hetero_report = hetero.report(
        node_features=node_features,
        pairs=candidates,
        labels=labels,
        query_ids=[0, 0, 1, 1],
        k=1,
    )
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(hetero_report)

    hinsage = HinSageLinkPredictor(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=[(0, 0, 1), (1, 1, 0)],
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=37,
    ).fit(
        node_features=node_features,
        node_types=[0, 1, 0, 1],
        edges=typed_edges,
    )
    hinsage_report = hinsage.report(
        node_features=node_features,
        pairs=candidates,
        labels=labels,
        query_ids=[0, 0, 1, 1],
        k=1,
    )
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(hinsage_report)


def test_hinsage_native_encoder_exposes_graph_aware_encode() -> None:
    from cartoboost import HinSageEncoder

    node_features = np.eye(4, 2, dtype=np.float32)
    node_types = [0, 1, 0, 1]
    edge_type_triples = [(0, 0, 1), (1, 1, 0)]
    edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]

    encoder = HinSageEncoder(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=edge_type_triples,
        hidden_dims=[4],
        epochs=2,
        negative_samples=1,
    )
    fitted = np.asarray(encoder.fit(node_types, edges, node_features))
    encoded = np.asarray(encoder.encode_graph(node_types, edges, node_features))
    np.testing.assert_allclose(encoded, fitted)


def test_standalone_regressor_scores_and_all_roundtrips(tmp_path) -> None:
    fixture = _graph_fixture()
    ids = np.array([1, 2, 1, 3, 2, 3], dtype=np.uint64)
    dense = np.column_stack([ids.astype(float) / 3.0])
    y = np.array([1.0, 2.0, 1.1, 3.0, 2.1, 3.2])

    neural = NeuralEmbeddingStandaloneRegressor(
        dim=3,
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=1,
        random_state=41,
    ).fit(ids, y, dense=dense)
    neural_pred = neural.predict(ids, dense=dense)
    assert np.isfinite(neural.score(ids, y, dense=dense))
    neural_path = neural.save(tmp_path / "neural.json")
    np.testing.assert_allclose(
        NeuralEmbeddingStandaloneRegressor.load(neural_path).predict(ids, dense=dense),
        neural_pred,
    )

    node2vec = Node2VecStandaloneRegressor(
        dim=4,
        walk_length=5,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        negative_samples=1,
        seed=43,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
    ).fit(
        node_count=4,
        edges=fixture["edges"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
        y=fixture["y"],
    )
    node2vec_pred = node2vec.predict(fixture["row_nodes"], row_targets=fixture["row_targets"])
    assert np.isfinite(
        node2vec.score(
            fixture["row_nodes"],
            fixture["y"],
            row_targets=fixture["row_targets"],
        )
    )
    node2vec_path = node2vec.save(tmp_path / "node2vec.json")
    np.testing.assert_allclose(
        Node2VecStandaloneRegressor.load(node2vec_path).predict(
            fixture["row_nodes"],
            row_targets=fixture["row_targets"],
        ),
        node2vec_pred,
    )

    sage = GraphSageStandaloneRegressor(
        input_dim=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=47,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
    ).fit(
        node_features=fixture["node_features"],
        edges=fixture["edges"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
        y=fixture["y"],
    )
    sage_pred = sage.predict(
        node_features=fixture["node_features"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
    )
    assert np.isfinite(
        sage.score(
            node_features=fixture["node_features"],
            row_nodes=fixture["row_nodes"],
            row_targets=fixture["row_targets"],
            y=fixture["y"],
        )
    )
    sage_path = sage.save(tmp_path / "graphsage.json")
    np.testing.assert_allclose(
        GraphSageStandaloneRegressor.load(sage_path).predict(
            node_features=fixture["node_features"],
            row_nodes=fixture["row_nodes"],
            row_targets=fixture["row_targets"],
        ),
        sage_pred,
    )

    hetero = HeteroGraphSageStandaloneRegressor(
        input_dim=2,
        relation_count=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=53,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
    ).fit(
        node_features=fixture["node_features"],
        edges=fixture["typed_edges"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
        y=fixture["y"],
    )
    hetero_pred = hetero.predict(
        node_features=fixture["node_features"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
    )
    hetero_path = hetero.save(tmp_path / "hetero.json")
    np.testing.assert_allclose(
        HeteroGraphSageStandaloneRegressor.load(hetero_path).predict(
            node_features=fixture["node_features"],
            row_nodes=fixture["row_nodes"],
            row_targets=fixture["row_targets"],
        ),
        hetero_pred,
    )

    hinsage = HinSageStandaloneRegressor(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=fixture["edge_type_triples"],
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        neighbor_samples=(2, 2),
        seed=59,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
    ).fit(
        node_features=fixture["node_features"],
        node_types=fixture["node_types"],
        edges=fixture["typed_edges"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
        y=fixture["y"],
    )
    hinsage_pred = hinsage.predict(
        node_features=fixture["node_features"],
        row_nodes=fixture["row_nodes"],
        row_targets=fixture["row_targets"],
    )
    hinsage_path = hinsage.save(tmp_path / "hinsage.json")
    np.testing.assert_allclose(
        HinSageStandaloneRegressor.load(hinsage_path).predict(
            node_features=fixture["node_features"],
            row_nodes=fixture["row_nodes"],
            row_targets=fixture["row_targets"],
        ),
        hinsage_pred,
    )


def test_standalone_link_predictors_all_save_load_and_score(tmp_path) -> None:
    fixture = _graph_fixture()
    candidates = [(0, 1), (0, 2), (1, 2), (1, 3)]
    labels = [1, 0, 1, 0]
    query_ids = [0, 0, 1, 1]

    node2vec = Node2VecLinkPredictor(
        dim=4,
        walk_length=5,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        negative_samples=1,
        seed=61,
    ).fit(node_count=4, edges=fixture["edges"])
    node2vec_scores = node2vec.predict_scores(candidates)
    assert np.isfinite(node2vec_scores).all()
    assert "mean_reciprocal_rank" in node2vec.report(candidates, labels, query_ids=query_ids, k=1)
    node2vec_path = node2vec.save(tmp_path / "node2vec-link.json")
    np.testing.assert_allclose(
        Node2VecLinkPredictor.load(node2vec_path).predict_scores(candidates),
        node2vec_scores,
    )

    sage = GraphSageLinkPredictor(
        input_dim=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=67,
    ).fit(node_features=fixture["node_features"], edges=fixture["edges"])
    sage_scores = sage.predict_scores(node_features=fixture["node_features"], pairs=candidates)
    assert np.isfinite(sage_scores).all()
    assert "mean_reciprocal_rank" in sage.report(
        node_features=fixture["node_features"],
        pairs=candidates,
        labels=labels,
        query_ids=query_ids,
        k=1,
    )
    sage_path = sage.save(tmp_path / "graphsage-link.json")
    np.testing.assert_allclose(
        GraphSageLinkPredictor.load(sage_path).predict_scores(
            node_features=fixture["node_features"],
            pairs=candidates,
        ),
        sage_scores,
    )

    hetero = HeteroGraphSageLinkPredictor(
        input_dim=2,
        relation_count=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=71,
    ).fit(node_features=fixture["node_features"], edges=fixture["typed_edges"])
    hetero_scores = hetero.predict_scores(
        node_features=fixture["node_features"],
        pairs=candidates,
    )
    assert np.isfinite(hetero_scores).all()
    assert "mean_reciprocal_rank" in hetero.report(
        node_features=fixture["node_features"],
        pairs=candidates,
        labels=labels,
        query_ids=query_ids,
        k=1,
    )
    hetero_path = hetero.save(tmp_path / "hetero-link.json")
    np.testing.assert_allclose(
        HeteroGraphSageLinkPredictor.load(hetero_path).predict_scores(
            node_features=fixture["node_features"],
            pairs=candidates,
        ),
        hetero_scores,
    )

    hinsage = HinSageLinkPredictor(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=fixture["edge_type_triples"],
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        neighbor_samples=(2, 2),
        seed=73,
    ).fit(
        node_features=fixture["node_features"],
        node_types=fixture["node_types"],
        edges=fixture["typed_edges"],
    )
    hinsage_scores = hinsage.predict_scores(
        node_features=fixture["node_features"],
        pairs=candidates,
    )
    assert np.isfinite(hinsage_scores).all()
    assert "mean_reciprocal_rank" in hinsage.report(
        node_features=fixture["node_features"],
        pairs=candidates,
        labels=labels,
        query_ids=query_ids,
        k=1,
    )
    hinsage_path = hinsage.save(tmp_path / "hinsage-link.json")
    np.testing.assert_allclose(
        HinSageLinkPredictor.load(hinsage_path).predict_scores(
            node_features=fixture["node_features"],
            pairs=candidates,
        ),
        hinsage_scores,
    )


def test_standalone_wrapper_argument_validation() -> None:
    fixture = _graph_fixture()
    with pytest.raises(ValueError, match="ids must be 1D"):
        NeuralEmbeddingStandaloneRegressor(dim=2).fit([[1, 2]], [1.0, 2.0])
    with pytest.raises(ValueError, match="edges must be a 2D array-like"):
        Node2VecStandaloneRegressor(dim=2).fit(
            node_count=2,
            edges=[(0, 1, 2)],
            row_nodes=[0],
            y=[1.0],
        )
    with pytest.raises(ValueError, match="edges must contain non-negative"):
        GraphSageLinkPredictor(input_dim=2).fit(
            node_features=fixture["node_features"],
            edges=[(0, -1)],
        )
    with pytest.raises(ValueError, match="typed edges must be a 2D array-like"):
        HeteroGraphSageLinkPredictor(input_dim=2, relation_count=2).fit(
            node_features=fixture["node_features"],
            edges=[(0, 1)],
        )
    with pytest.raises(ValueError, match="node_types must contain non-negative"):
        HinSageLinkPredictor(
            input_dim=2,
            node_type_count=2,
            edge_type_triples=fixture["edge_type_triples"],
        ).fit(
            node_features=fixture["node_features"],
            node_types=[0, -1, 0, 1],
            edges=fixture["typed_edges"],
        )


@pytest.mark.parametrize("ids", [[1.5], [-1], [float("nan")], [float("inf")], [2**64]])
def test_standalone_embedding_rejects_invalid_ids(ids):
    with pytest.raises(ValueError, match="ids"):
        NeuralEmbeddingStandaloneRegressor(dim=2).fit(ids, [1.0])


def test_standalone_integer_conversions_preserve_exact_values():
    from cartoboost.standalone import _int_list, _u64_list, _usize_list

    assert _u64_list([2**63 + 1, 2**64 - 1], "ids") == [2**63 + 1, 2**64 - 1]
    assert _u64_list(np.array([2**64 - 1], dtype=np.uint64), "ids") == [2**64 - 1]
    assert _int_list([-(2**63), 2**63 - 1], "labels") == [-(2**63), 2**63 - 1]
    assert _usize_list([0.0, 2.0], "row_nodes") == [0, 2]
    with pytest.raises(ValueError, match="finite integers"):
        _usize_list([0.5], "row_nodes")
    with pytest.raises(ValueError, match="finite integers"):
        _int_list([1.5], "labels")
    with pytest.raises(ValueError, match="outside the supported range"):
        _int_list(np.array([2**63], dtype=np.uint64), "labels")


def test_standalone_integer_conversions_preserve_integer_strings():
    from cartoboost.standalone import _edge_pairs, _int_list, _typed_edges, _u64_list, _usize_list

    assert _u64_list([str(2**64 - 1)], "ids") == [2**64 - 1]
    assert _usize_list(np.array(["0", "2"]), "ids") == [0, 2]
    assert _int_list(["-2", b"3"], "labels") == [-2, 3]
    assert _edge_pairs([["0", "1"]]) == [(0, 1)]
    assert _typed_edges([["0", "1", "2"]]) == [(0, 1, 2)]
    for values in [["1.5"], ["nan"], ["inf"], ["-1"], [str(2**64)]]:
        with pytest.raises(ValueError):
            _u64_list(values, "ids")


@pytest.mark.parametrize("typed", [False, True])
def test_standalone_edges_reject_fractional_ids(typed):
    from cartoboost.standalone import _edge_pairs, _typed_edges

    with pytest.raises(ValueError, match="finite integers"):
        if typed:
            _typed_edges([[0, 1, 0.5]])
        else:
            _edge_pairs([[0, 1.5]])


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_standalone_integer_tables_preserve_large_mixed_ids(kind):
    from cartoboost.standalone import _edge_pairs, _int_list, _u64_list

    large = [2**64 - 1, 2**63 + 1]
    small = [1, 2]
    if kind == "pandas":
        pd = pytest.importorskip("pandas")
        ids = pd.Series(large, dtype="uint64")
        edges = pd.DataFrame({"pickup": ids, "dropoff": pd.Series(small, dtype="int64")})
        signed = pd.Series([-(2**63), 2**63 - 1], dtype="int64")
    else:
        pl = pytest.importorskip("polars")
        ids = pl.Series("pickup", large, dtype=pl.UInt64)
        edges = pl.DataFrame([ids, pl.Series("dropoff", small, dtype=pl.Int64)])
        signed = pl.Series([-(2**63), 2**63 - 1], dtype=pl.Int64)
    assert _u64_list(ids, "ids") == large
    assert _edge_pairs(edges) == list(zip(large, small, strict=True))
    assert _int_list(signed, "labels") == [-(2**63), 2**63 - 1]
    with pytest.raises(ValueError, match="supported range"):
        _int_list(ids, "labels")
    with pytest.raises(ValueError, match="non-negative"):
        _u64_list(signed, "ids")


def test_standalone_nullable_integer_tables_validate_original_values():
    pd = pytest.importorskip("pandas")
    from cartoboost.standalone import _u64_list

    assert _u64_list(pd.Series([2**64 - 1, 1], dtype="UInt64"), "ids") == [2**64 - 1, 1]
    for values in [
        pd.Series([2**64 - 1, None], dtype="UInt64"),
        pd.Series([1, None], dtype="Int64"),
        pd.Series([1.0, 1.5], dtype="Float64"),
    ]:
        with pytest.raises(ValueError, match="finite integers"):
            _u64_list(values, "ids")


_ARRAY_STANDALONE_FAMILIES = [
    "embedding",
    "node2vec",
    "graphsage",
    "hetero",
    "hinsage",
    "node2vec_link",
    "graphsage_link",
    "hetero_link",
    "hinsage_link",
]


def _native_transfer_case(family, dense_width=1):
    graph = _graph_fixture()
    link = family.endswith("_link")
    base = family.removesuffix("_link")
    classes = {
        "embedding": NeuralEmbeddingStandaloneRegressor,
        "node2vec": Node2VecStandaloneRegressor,
        "graphsage": GraphSageStandaloneRegressor,
        "hetero": HeteroGraphSageStandaloneRegressor,
        "hinsage": HinSageStandaloneRegressor,
        "node2vec_link": Node2VecLinkPredictor,
        "graphsage_link": GraphSageLinkPredictor,
        "hetero_link": HeteroGraphSageLinkPredictor,
        "hinsage_link": HinSageLinkPredictor,
    }
    config = {} if link else dict(n_estimators=3, max_depth=2, min_samples_leaf=1)
    if base == "embedding":
        config.update(dim=3, random_state=31)
    elif base == "node2vec":
        config.update(
            dim=3,
            walk_length=3,
            walks_per_node=1,
            window_size=2,
            epochs=1,
            negative_samples=1,
            seed=31,
        )
    else:
        config.update(input_dim=2, hidden_dims=[3], epochs=1, negative_samples=1, seed=31)
        if base == "hetero":
            config.update(relation_count=2)
        elif base == "hinsage":
            config.update(node_type_count=2, edge_type_triples=graph["edge_type_triples"])
    ids = np.array([2**64 - 1, 2**63 + 1, 1, 2**64 - 1], dtype=np.uint64)
    nodes = np.asarray(graph["row_nodes"], dtype=np.uintp)
    targets = np.asarray(graph["row_targets"], dtype=np.uintp)
    dense = np.arange(4 * dense_width, dtype=np.float64).reshape(4, dense_width) / 7.0
    fit = {}
    query = {}
    if base == "embedding":
        fit["ids"] = ids
        query["ids"] = ids
    else:
        fit["edges"] = np.asarray(
            graph["typed_edges" if base in {"hetero", "hinsage"} else "edges"], dtype=np.uintp
        )
        if base == "node2vec":
            fit["node_count"] = 4
            fit["edge_weights"] = np.ones(4, dtype=np.float32)
        else:
            fit["node_features"] = graph["node_features"]
            query["node_features"] = graph["node_features"]
        if base == "hinsage":
            fit["node_types"] = np.asarray(graph["node_types"], dtype=np.uintp)
        if link:
            query["pairs"] = np.column_stack([nodes, targets])
        else:
            fit.update(row_nodes=nodes, row_targets=targets)
            query.update(row_nodes=nodes, row_targets=targets)
    if not link:
        fit.update(y=graph["y"], dense=dense)
        query["dense"] = dense
    return classes[family], config, fit, query


def _transfer_layout(value, layout):
    if not isinstance(value, np.ndarray):
        return value
    if layout == "fortran":
        return np.asfortranarray(value)
    if layout == "reversed":
        return value[::-1].copy()[::-1]
    if layout == "strided":
        storage = np.empty((value.shape[0] * 2, *value.shape[1:]), dtype=value.dtype)
        storage[::2] = value
        return storage[::2]
    return value.copy()


def _transfer_lists(arguments):
    result = {}
    for name, value in arguments.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
            if name in {"edges", "pairs"}:
                value = [tuple(row) for row in value]
        result[name] = value
    return result


@pytest.mark.parametrize("family", _ARRAY_STANDALONE_FAMILIES)
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "reversed", "strided"])
@pytest.mark.parametrize("dense_width", [0, 1])
def test_all_standalone_array_endpoints_preserve_values_artifacts_and_layouts(
    family, layout, dense_width, tmp_path
):
    model_class, config, fit, query = _native_transfer_case(family, dense_width)
    legacy = model_class(**config)._native
    candidate = model_class(**config)._native
    legacy.fit(**_transfer_lists(fit))
    candidate.fit_arrays(**{key: _transfer_layout(value, layout) for key, value in fit.items()})
    operation = "predict_scores" if family.endswith("_link") else "predict"
    expected = np.asarray(getattr(legacy, operation)(**_transfer_lists(query)))
    query_arrays = {key: _transfer_layout(value, layout) for key, value in query.items()}
    actual = getattr(candidate, operation + "_arrays")(**query_arrays)
    assert actual.dtype == np.float64
    assert actual.shape == (4,)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    singleton = []
    for index in range(4):
        one = {
            key: value if key == "node_features" else value[index : index + 1]
            for key, value in query_arrays.items()
        }
        singleton.append(getattr(candidate, operation + "_arrays")(**one)[0])
    np.testing.assert_array_equal(actual, singleton)
    old_path, new_path = tmp_path / "legacy.json", tmp_path / "array.json"
    legacy.save_artifact_json(str(old_path))
    candidate.save_artifact_json(str(new_path))
    assert old_path.read_bytes() == new_path.read_bytes()
    restored = type(candidate).load_artifact_json(str(new_path))
    np.testing.assert_array_equal(
        getattr(restored, operation + "_arrays")(**query_arrays), expected
    )
    assert restored.backend == candidate.backend == legacy.backend
    if family.endswith("_link"):
        query_arrays["pairs"] = np.empty((0, 2), dtype=np.uintp)
        assert getattr(candidate, operation + "_arrays")(**query_arrays).shape == (0,)


@pytest.mark.parametrize("family", _ARRAY_STANDALONE_FAMILIES)
def test_all_standalone_array_endpoints_preserve_invalid_input_errors(family):
    model_class, config, fit, query = _native_transfer_case(family)
    model = model_class(**config)._native
    operation = "predict_scores" if family.endswith("_link") else "predict"
    try:
        initial = getattr(model, operation)(**_transfer_lists(query))
    except ValueError:
        with pytest.raises(ValueError):
            getattr(model, operation + "_arrays")(**query)
    else:
        np.testing.assert_array_equal(getattr(model, operation + "_arrays")(**query), initial)
    model.fit_arrays(**fit)
    if family == "embedding":
        query["dense"] = np.full((4, 1), np.nan)
    elif family.endswith("_link"):
        query["pairs"] = np.array([[0, 100]], dtype=np.uintp)
    else:
        query["row_nodes"] = np.array([100, 1, 2, 3], dtype=np.uintp)
    with pytest.raises(ValueError):
        getattr(model, operation + "_arrays")(**query)


@pytest.mark.parametrize("family", _ARRAY_STANDALONE_FAMILIES)
def test_all_standalone_public_wrappers_match_native_legacy_prediction(family):
    model_class, config, fit, query = _native_transfer_case(family)
    model = model_class(**config).fit(**fit)
    operation = "predict_scores" if family.endswith("_link") else "predict"
    expected = getattr(model._native, operation)(**_transfer_lists(query))
    np.testing.assert_array_equal(getattr(model, operation)(**query), expected)
