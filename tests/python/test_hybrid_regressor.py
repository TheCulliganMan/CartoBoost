import numpy as np
from cartoboost import (
    CartoBoostRegressor,
    NeuralEmbeddingRegressor,
    benchmark_neural_vs_cartoboost,
)
from sklearn.base import clone


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
