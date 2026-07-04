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
