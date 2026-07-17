"""Neural embedding regression example.

This example models repeated categorical location/customer IDs with
NeuralEmbeddingRegressor, then uses a fixed train-only guard to avoid neural
embeddings on cold-ID holdouts.

    uv run python examples/05_neural_embedding_regression.py
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from cartoboost import CartoBoostRegressor, NeuralEmbeddingRegressor

SPLITTERS = ["axis_histogram:256"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--ids", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--embedding-dim", type=int, default=8)
    return parser.parse_args()


def synthetic_rows(rows: int, ids: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    entity_id = rng.integers(0, ids, size=rows, dtype=np.uint64)
    x = rng.normal(size=(rows, 4))
    id_float = entity_id.astype(float)
    id_effect = 1.3 * np.sin(id_float / 4.0) + 0.7 * np.cos(id_float / 9.0)
    y = (
        1.2 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.35 * x[:, 2] ** 2
        + id_effect
        + rng.normal(0.0, 0.18, size=rows)
    )
    return x.astype(float), entity_id, y.astype(float)


def random_split(row_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(row_count)
    test_count = max(1, int(row_count * 0.2))
    return order[test_count:], order[:test_count]


def cold_id_split(ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 101)
    unique_ids = rng.permutation(np.unique(ids))
    test_id_count = max(1, int(len(unique_ids) * 0.2))
    test_ids = {int(value) for value in unique_ids[:test_id_count]}
    test_mask = np.asarray([int(value) in test_ids for value in ids])
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def inner_validation(row_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 997)
    order = rng.permutation(row_count)
    validation_count = max(1, int(row_count * 0.2))
    return order[validation_count:], order[:validation_count]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - residual / total if total > 0.0 else 0.0


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "r2": r2(actual, predicted),
    }


def dense_model(n_estimators: int) -> CartoBoostRegressor:
    return CartoBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=5,
        min_gain=0.0,
        split_policy="structured",
    )


def neural_model(n_estimators: int, embedding_dim: int, seed: int) -> NeuralEmbeddingRegressor:
    kwargs = {
        "n_estimators": n_estimators,
        "learning_rate": 0.08,
        "max_depth": 4,
        "min_samples_leaf": 5,
        "min_gain": 0.0,
        "split_policy": "structured",
    }
    return NeuralEmbeddingRegressor(
        dim=embedding_dim,
        random_state=seed,
        drop_id_column=False,
        id_column=None,
        oof_folds=5,
        support_prior_strength=0.5,
        base_model_kwargs={**kwargs, "n_estimators": max(10, n_estimators // 2)},
        final_model_kwargs=kwargs,
    )


def fit_predict_dense(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_estimators: int,
) -> np.ndarray:
    model = dense_model(n_estimators)
    model.fit(x[train_idx], y[train_idx])
    return model.predict(x[test_idx])


def fit_predict_neural(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_estimators: int,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    model = neural_model(n_estimators, embedding_dim, seed)
    model.fit(x[train_idx], y[train_idx], ids=ids[train_idx])
    return model.predict(x[test_idx], ids=ids[test_idx])


def guarded_prediction(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_estimators: int,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x = x[train_idx]
    train_y = y[train_idx]
    train_ids = ids[train_idx]
    test_ids = ids[test_idx]
    train_id_set = {int(value) for value in train_ids}
    cold_id_fraction = float(np.mean([int(value) not in train_id_set for value in test_ids]))

    inner_train, inner_valid = inner_validation(len(train_y), seed)
    dense_probe = dense_model(n_estimators)
    neural_probe = neural_model(n_estimators, embedding_dim, seed)
    dense_probe.fit(train_x[inner_train], train_y[inner_train])
    neural_probe.fit(train_x[inner_train], train_y[inner_train], ids=train_ids[inner_train])
    dense_validation_rmse = rmse(
        train_y[inner_valid],
        dense_probe.predict(train_x[inner_valid]),
    )
    neural_validation_rmse = rmse(
        train_y[inner_valid],
        neural_probe.predict(train_x[inner_valid], ids=train_ids[inner_valid]),
    )

    use_neural = cold_id_fraction < 0.5 and neural_validation_rmse <= dense_validation_rmse * 0.99
    if use_neural:
        prediction = fit_predict_neural(
            x,
            y,
            ids,
            train_idx,
            test_idx,
            n_estimators,
            embedding_dim,
            seed,
        )
        selected = "neural_embedding"
    else:
        prediction = fit_predict_dense(x, y, train_idx, test_idx, n_estimators)
        selected = "dense"
    return prediction, {
        "selected": selected,
        "cold_id_fraction": cold_id_fraction,
        "dense_validation_rmse": dense_validation_rmse,
        "neural_validation_rmse": neural_validation_rmse,
    }


def evaluate_split(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dense_pred = fit_predict_dense(x, y, train_idx, test_idx, args.n_estimators)
    neural_pred = fit_predict_neural(
        x,
        y,
        ids,
        train_idx,
        test_idx,
        args.n_estimators,
        args.embedding_dim,
        args.seed,
    )
    guarded_pred, guard = guarded_prediction(
        x,
        y,
        ids,
        train_idx,
        test_idx,
        n_estimators=args.n_estimators,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    return {
        "split": name,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "dense": metrics(y[test_idx], dense_pred),
        "neural_embedding": {
            **metrics(y[test_idx], neural_pred),
            "embedding_dim": int(args.embedding_dim),
        },
        "guarded": {
            **metrics(y[test_idx], guarded_pred),
            **guard,
        },
    }


def main() -> None:
    args = parse_args()
    x, ids, y = synthetic_rows(args.rows, args.ids, args.seed)
    random_train, random_test = random_split(len(y), args.seed)
    cold_train, cold_test = cold_id_split(ids, args.seed)
    print(
        json.dumps(
            {
                "task": "neural_embedding_regression",
                "rows": int(len(y)),
                "id_count": int(len(np.unique(ids))),
                "random": evaluate_split(
                    "random",
                    x,
                    y,
                    ids,
                    random_train,
                    random_test,
                    args,
                ),
                "cold_id_holdout": evaluate_split(
                    "cold_id_holdout",
                    x,
                    y,
                    ids,
                    cold_train,
                    cold_test,
                    args,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
