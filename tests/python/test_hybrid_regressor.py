import numpy as np
import pytest
from cartoboost import (
    CartoBoostRegressor,
    NeuralEmbeddingRegressor,
    benchmark_neural_vs_cartoboost,
)
from sklearn.base import clone


@pytest.mark.parametrize("as_id_column", [False, True])
def test_neural_pipeline_preserves_large_integer_ids(as_id_column):
    ids = [2**63 + 1, 2**63 + 2, 2**64 - 1]
    model = NeuralEmbeddingRegressor(
        dim=2,
        use_residual=False,
        id_column=1 if as_id_column else None,
        final_model_kwargs={"n_estimators": 1, "min_samples_leaf": 1},
    )
    X = [[0.0, ids[0]], [1.0, ids[1]], [2.0, ids[2]]] if as_id_column else [[0.0], [1.0], [2.0]]
    kwargs = {} if as_id_column else {"ids": np.asarray(ids, dtype=np.uint64)}
    model.fit(X, [0.0, 1.0, 2.0], **kwargs)

    assert model.neural_transformers[0].known_ids() == set(ids)
    assert np.isfinite(model.predict(X, **kwargs)).all()


def test_neural_pipeline_keeps_sparse_schema_after_augmented_dense_features():
    model = NeuralEmbeddingRegressor(
        dim=2,
        use_residual=False,
        final_model_kwargs={"n_estimators": 1, "min_samples_leaf": 1},
    ).fit(
        [[0.0], [1.0], [2.0]],
        [0.0, 1.0, 2.0],
        ids=[1, 2, 3],
        sparse_sets={"zones": [[10], [20], [30]]},
        feature_schema={"dense": ["distance"], "sparse_sets": ["zones"]},
    )
    assert model.model.feature_schema_["names"] == [
        "distance",
        "neural_embedding_00",
        "neural_embedding_01",
        "zones",
    ]
    assert model.model.feature_schema_["kinds"] == ["Numeric", "Numeric", "Numeric", "SparseSet"]


def test_neural_pipeline_uses_configured_transformer_and_clones_its_settings():
    from cartoboost.neural import NeuralEmbeddingFeatures

    transformer = NeuralEmbeddingFeatures(dim=3, fallback="zero_vector", support_prior_strength=7.0)
    model = NeuralEmbeddingRegressor(
        neural_transformer=transformer,
        use_residual=False,
        final_model_kwargs={"n_estimators": 1, "min_samples_leaf": 1},
    ).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], ids=[1, 2, 3])

    assert model.neural_transformers[0] is transformer
    assert transformer.known_ids() == {1, 2, 3}
    assert model.transform([[1.0]], ids=[99]).shape == (1, 4)
    np.testing.assert_array_equal(transformer.transform([99]), np.zeros((1, 3)))
    cloned = clone(model)
    assert cloned.neural_transformer is not transformer
    assert cloned.neural_transformer.dim == 3
    assert cloned.neural_transformer.fallback == "zero_vector"
    assert cloned.neural_transformer.support_prior_strength == 7.0
    assert cloned.neural_transformers == []


def test_neural_benchmark_accepts_explicit_dimension_and_drop_setting():
    report = benchmark_neural_vs_cartoboost(
        [[0.0], [1.0], [2.0], [3.0]],
        [0.0, 1.0, 2.0, 3.0],
        ids=[1, 2, 1, 2],
        split_ratio=0.5,
        neural_kwargs={
            "dim": 2,
            "drop_id_column": False,
            "use_residual": False,
            "final_model_kwargs": {"n_estimators": 1},
        },
        cartoboost_kwargs={"n_estimators": 1},
    )
    assert report["n_rows"] == 4
    assert np.isfinite(report["hybrid_mae"])


@pytest.mark.parametrize("with_graph", [False, True])
def test_oof_residual_embeddings_do_not_use_held_out_targets(with_graph):
    class RecordingRegressor(CartoBoostRegressor):
        def fit(self, X, y, **kwargs):
            self.training_X = np.array(X, copy=True)
            return super().fit(X, y, **kwargs)

    class RecordingPipeline(NeuralEmbeddingRegressor):
        def _build_final_model(self):
            return RecordingRegressor(**self.final_model_kwargs)

    rng = np.random.default_rng(12)
    X = rng.normal(size=(24, 2))
    y = rng.normal(size=24)
    ids = np.arange(24, dtype=np.uint64) % 5
    held_out = np.array_split(np.random.default_rng(42).permutation(24), 3)[0]
    changed_y = y.copy()
    changed_y[held_out] += rng.normal(size=len(held_out)) * 100.0
    base_params = {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.5, "min_samples_leaf": 1}
    if with_graph:
        base_params.update(
            graph_indptr=list(range(25)),
            graph_indices=[(row + 1) % 24 for row in range(24)],
            graph_weights=[1.0 + row / 100.0 for row in range(24)],
            graph_smoothing=0.3,
        )
    params = dict(
        dim=3,
        oof_folds=3,
        random_state=42,
        base_model_kwargs=base_params,
        final_model_kwargs={"n_estimators": 1},
    )
    fit_kwargs = dict(
        ids=ids,
        sample_weight=np.linspace(0.5, 2.0, 24),
        sparse_sets={"zones": [[row % 3] for row in range(24)]},
        feature_schema={"dense": ["x", "y"], "sparse_sets": ["zones"]},
    )
    original = RecordingPipeline(**params).fit(X, y, **fit_kwargs)
    changed = RecordingPipeline(**params).fit(X, changed_y, **fit_kwargs)

    np.testing.assert_array_equal(
        original.model.training_X[held_out, 2:], changed.model.training_X[held_out, 2:]
    )
    assert np.isfinite(original.predict(X, ids=ids, sparse_sets=fit_kwargs["sparse_sets"])).all()


def test_oof_graph_subset_preserves_internal_edge_order_and_weights():
    from cartoboost.neural.pipeline import _subset_graph_params

    params = {
        "graph_indptr": [0, 3, 4, 6],
        "graph_indices": [0, 1, 2, 2, 0, 2],
        "graph_weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "graph_smoothing": 0.3,
    }
    subset = _subset_graph_params(params, np.array([0, 2]), 3)
    assert subset["graph_indptr"] == [0, 2, 4]
    assert subset["graph_indices"] == [0, 1, 0, 1]
    assert subset["graph_weights"] == [1.0, 3.0, 5.0, 6.0]
    assert subset["graph_smoothing"] == 0.3
    assert params["graph_indptr"] == [0, 3, 4, 6]


def test_regressor_can_fit_on_augmented_neural_features():
    row_count = 180
    rng = np.random.default_rng(7)

    x_base = np.column_stack(
        [
            np.linspace(0.0, 1.0, row_count),
            np.sin(np.linspace(0.0, 4.0, row_count)),
        ]
    )
    neural = rng.normal(size=(row_count, 4)).astype(np.float64)
    y = 1.5 * x_base[:, 0] + 0.7 * x_base[:, 1] + 2.0 * neural[:, 0] + 0.2 * neural[:, 1]

    model = CartoBoostRegressor(
        n_estimators=12,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=1,
    ).fit(np.hstack([x_base, neural]), y)

    predictions = model.predict(np.hstack([x_base, neural]))
    mae = float(np.mean(np.abs(predictions - y)))

    assert mae < 0.8


def test_neural_embedding_regressor_train_predict_with_ids_array():
    rng = np.random.default_rng(7)
    rows = 200
    ids = (rng.integers(0, 40, size=rows)).astype(np.uint64)

    x = np.column_stack(
        [
            rng.normal(size=rows),
            rng.normal(size=rows),
            rng.normal(size=rows),
        ]
    )

    cell_signal = (ids.astype(np.float64) % 10) * 0.25
    y = 1.2 * x[:, 0] - 0.7 * x[:, 1] + cell_signal + rng.normal(0.0, 0.05, size=rows)

    regressor = NeuralEmbeddingRegressor(
        dim=4,
        use_residual=True,
        random_state=11,
        final_model_kwargs={
            "n_estimators": 25,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "min_gain": 0.0,
        },
    )
    regressor.fit(x, y, ids=ids)

    pred = regressor.predict(x, ids=ids)
    train_mae = float(np.mean(np.abs(pred - y)))
    transformed = regressor.transform(x, ids=ids)

    assert pred.shape == (rows,)
    assert transformed.shape == (rows, x.shape[1] + regressor.dim)
    assert train_mae < 0.8
    assert regressor.timings["final_fit_ms"] > 0.0


def test_neural_embedding_regressor_rejects_non_finite_numeric_inputs():
    model = NeuralEmbeddingRegressor(
        dim=2,
        base_model_kwargs={"n_estimators": 2, "max_depth": 1, "min_samples_leaf": 1},
    )
    ids = np.asarray([1, 2, 3], dtype=np.uint64)

    with pytest.raises(ValueError, match="X must contain only finite values"):
        model.fit([[0.0], [np.nan], [2.0]], [0.0, 1.0, 2.0], ids=ids)

    with pytest.raises(ValueError, match="y must contain only finite values"):
        model.fit([[0.0], [1.0], [2.0]], [0.0, np.inf, 2.0], ids=ids)

    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(
            [[0.0], [1.0], [2.0]],
            [0.0, 1.0, 2.0],
            ids=ids,
            sample_weight=[1.0, np.nan, 1.0],
        )


def test_neural_embedding_regressor_with_id_column():
    rng = np.random.default_rng(9)
    rows = 150
    ids = (rng.integers(0, 30, size=rows)).astype(np.float64)

    x = np.column_stack(
        [
            np.linspace(-1.0, 1.0, rows),
            np.linspace(0.5, 1.5, rows),
            ids,
        ]
    )
    y = 1.1 * x[:, 0] + 0.3 * x[:, 1] + (ids % 5) * 0.5 + 0.02 * rng.normal(size=rows)

    regressor = NeuralEmbeddingRegressor(
        dim=3,
        id_column=2,
        drop_id_column=True,
        use_residual=True,
        final_model_kwargs={
            "n_estimators": 18,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_gain": 0.0,
        },
    )
    regressor.fit(x, y)

    pred = regressor.predict(x)
    mae = float(np.mean(np.abs(pred - y)))

    assert pred.shape == (rows,)
    assert mae < 0.9


def test_neural_embedding_regressor_appends_feature_schema_for_final_fit():
    rng = np.random.default_rng(15)
    rows = 120
    ids = (rng.integers(1, 60, size=rows)).astype(np.uint64)

    x = np.column_stack([rng.normal(size=rows), rng.normal(size=rows)])
    y = 1.0 * x[:, 0] + 0.4 * x[:, 1] + (ids % 10) * 0.3

    regressor = NeuralEmbeddingRegressor(
        dim=2,
        use_residual=True,
        final_model_kwargs={
            "n_estimators": 20,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_gain": 0.0,
        },
    )
    regressor.fit(
        x,
        y,
        ids=ids,
        feature_schema={"names": ["base_0", "base_1"], "kinds": ["Numeric", "Numeric"]},
    )

    pred = regressor.predict(x, ids=ids)
    assert pred.shape == (rows,)
    assert regressor.model.feature_schema_["names"][-2:] == [
        "neural_embedding_00",
        "neural_embedding_01",
    ]
    assert regressor.model.feature_schema_["names"][:2] == ["base_0", "base_1"]


def test_neural_embedding_regressor_supports_oof_and_multi_key_ids():
    rng = np.random.default_rng(23)
    rows = 180
    pickup = rng.integers(1, 30, size=rows, dtype=np.uint64)
    dropoff = rng.integers(1, 40, size=rows, dtype=np.uint64)
    ids = np.column_stack([pickup, dropoff])
    x = rng.normal(size=(rows, 3))
    y = (
        0.7 * x[:, 0]
        - 0.2 * x[:, 1]
        + 0.25 * (pickup % 5).astype(float)
        + 0.15 * (dropoff % 7).astype(float)
    )

    regressor = NeuralEmbeddingRegressor(
        dim=2,
        oof_folds=3,
        random_state=5,
        support_prior_strength=2.0,
        final_model_kwargs={
            "n_estimators": 16,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_gain": 0.0,
        },
    )
    regressor.fit(x, y, ids=ids)
    transformed = regressor.transform(x, ids=ids)
    pred = regressor.predict(x, ids=ids)

    assert len(regressor.neural_transformers) == 2
    assert transformed.shape == (rows, x.shape[1] + 4)
    assert pred.shape == (rows,)


def test_neural_embedding_regressor_predict_uses_fallback_context():
    rng = np.random.default_rng(31)
    rows = 120
    ids = rng.integers(1, 20, size=rows, dtype=np.uint64)
    x = rng.normal(size=(rows, 2))
    y = x[:, 0] + 0.4 * (ids % 3).astype(float)

    regressor = NeuralEmbeddingRegressor(
        dim=2,
        random_state=8,
        final_model_kwargs={
            "n_estimators": 12,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_gain": 0.0,
        },
    ).fit(x, y, ids=ids)

    cold_x = rng.normal(size=(3, 2))
    pred = regressor.predict(
        cold_x,
        ids=np.array([100, 101, 102], dtype=np.uint64),
        fallback_ids=np.array([[1, 2], [99, 3], [98, 97]], dtype=np.uint64),
        neighbor_ids=[[4, 5], [], [6]],
    )

    assert pred.shape == (3,)


def test_neural_embedding_regressor_clone_starts_with_fresh_native_transformer():
    rng = np.random.default_rng(41)
    rows = 90
    ids = rng.integers(1, 18, size=rows, dtype=np.uint64)
    x = rng.normal(size=(rows, 2))
    y = x[:, 0] + 0.3 * (ids % 4).astype(float)

    regressor = NeuralEmbeddingRegressor(
        dim=2,
        id_column=2,
        random_state=13,
        final_model_kwargs={
            "n_estimators": 8,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_gain": 0.0,
        },
    ).fit(np.column_stack([x, ids.astype(float)]), y)

    cloned = clone(regressor)

    assert cloned is not regressor
    assert cloned.get_params()["neural_transformer"] is None
    assert cloned.neural_transformer is not regressor.neural_transformer
    assert cloned.neural_transformers == []

    cloned.fit(np.column_stack([x, ids.astype(float)]), y)
    assert cloned.predict(np.column_stack([x, ids.astype(float)])).shape == (rows,)


def test_benchmark_neural_vs_cartoboost_smoke():
    rng = np.random.default_rng(12)
    rows = 256
    ids = (rng.integers(1, 100, size=rows)).astype(np.uint64)
    x = rng.normal(size=(rows, 5))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.1 * (ids % 7) + 0.1 * rng.normal(size=rows)

    results = benchmark_neural_vs_cartoboost(
        x,
        y,
        ids=ids,
        split_ratio=0.75,
        cartoboost_kwargs={
            "n_estimators": 8,
            "learning_rate": 0.2,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "min_gain": 0.0,
        },
    )

    assert results["structured_mae"] >= 0.0
    assert results["hybrid_mae"] >= 0.0
    assert results["n_rows"] == rows
    assert results["n_features"] == x.shape[1]
