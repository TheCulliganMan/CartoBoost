"""Taxi origin-destination graph regression example.

Run with synthetic taxi-shaped data:

    uv run python examples/03_taxi_od_graph_regression.py

Run with a local TLC-shaped CSV or Parquet file:

    uv run --group bench python examples/03_taxi_od_graph_regression.py \
        --input data/yellow_tripdata_2024-01.parquet --sample-size 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from cartoboost import CartoBoostRegressor
from cartoboost.graph import (
    DirectionalFeature,
    DirectionalityConfig,
    GraphEmbeddingsConfig,
    GraphEncoderConfig,
    GraphEncoderFamily,
    GraphFeatureTransformer,
)

QUALITATIVE_SPLITTERS = ["axis_histogram:256", "diagonal_2d", "gaussian_2d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-dim", type=int, default=8)
    parser.add_argument("--graph-epochs", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=80)
    return parser.parse_args()


def synthetic_taxi_rows(seed: int, sample_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pickup = rng.integers(1, 33, size=sample_size)
    dropoff = ((pickup + rng.integers(1, 10, size=sample_size) - 1) % 32) + 1
    hour = rng.integers(0, 24, size=sample_size)
    dayofweek = rng.integers(0, 7, size=sample_size)
    passenger_count = 1.0 + ((pickup + dropoff) % 3).astype(float)
    route_span = np.abs(dropoff - pickup).astype(float)
    trip_distance = 0.8 + 0.42 * route_span + rng.gamma(1.2, 0.15, size=sample_size)
    night = ((hour >= 22) | (hour <= 2)).astype(float)
    airport_like = ((pickup % 11 == 0) | (dropoff % 13 == 0)).astype(float)
    seconds = (
        220.0
        + 135.0 * trip_distance
        + 70.0 * night
        + 45.0 * airport_like
        + 8.0 * pickup
        - 3.0 * dropoff
        + rng.normal(0.0, 30.0, size=sample_size)
    )
    return {
        "PULocationID": pickup.astype(int),
        "DOLocationID": dropoff.astype(int),
        "trip_distance": trip_distance.astype(float),
        "passenger_count": passenger_count.astype(float),
        "hour": hour.astype(float),
        "dayofweek": dayofweek.astype(float),
        "target": np.log1p(np.maximum(seconds, 1.0)).astype(float),
    }


def load_taxi_rows(path: Path, seed: int, sample_size: int) -> dict[str, np.ndarray]:
    pandas = __import__("pandas")
    frame = pandas.read_parquet(path) if path.suffix == ".parquet" else pandas.read_csv(path)
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    pickup_time = pandas.to_datetime(frame["tpep_pickup_datetime"])
    dropoff_time = pandas.to_datetime(frame["tpep_dropoff_datetime"])
    duration_seconds = (dropoff_time - pickup_time).dt.total_seconds().to_numpy(dtype=float)
    valid = (
        np.isfinite(duration_seconds)
        & (duration_seconds > 0.0)
        & (duration_seconds < 6.0 * 3600.0)
        & (frame["trip_distance"].to_numpy(dtype=float) > 0.0)
    )
    frame = frame.loc[valid]
    pickup_time = pickup_time.loc[valid]
    duration_seconds = duration_seconds[valid]
    return {
        "PULocationID": frame["PULocationID"].to_numpy(dtype=int),
        "DOLocationID": frame["DOLocationID"].to_numpy(dtype=int),
        "trip_distance": frame["trip_distance"].to_numpy(dtype=float),
        "passenger_count": frame["passenger_count"].fillna(1.0).to_numpy(dtype=float),
        "hour": pickup_time.dt.hour.to_numpy(dtype=float),
        "dayofweek": pickup_time.dt.dayofweek.to_numpy(dtype=float),
        "target": np.log1p(duration_seconds).astype(float),
    }


def train_test_split(row_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(row_count)
    test_count = max(1, int(row_count * 0.2))
    return order[test_count:], order[:test_count]


def dense_features(rows: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            rows["trip_distance"],
            rows["passenger_count"],
            rows["hour"],
            rows["dayofweek"],
        ]
    ).astype(float)


def graph_augmented_features(
    rows: dict[str, np.ndarray],
    train_idx: np.ndarray,
    *,
    graph_dim: int,
    graph_epochs: int,
    seed: int,
) -> np.ndarray:
    pickup = rows["PULocationID"].astype(int)
    dropoff = rows["DOLocationID"].astype(int)
    node_count = int(max(pickup.max(), dropoff.max()) + 1)
    train_edges = sorted({(int(pickup[index]), int(dropoff[index])) for index in train_idx})
    pickup_counts = np.bincount(pickup[train_idx], minlength=node_count).astype(float)
    dropoff_counts = np.bincount(dropoff[train_idx], minlength=node_count).astype(float)
    node_features = np.column_stack(
        [
            np.arange(node_count, dtype=float) / float(max(node_count - 1, 1)),
            np.log1p(pickup_counts),
            np.log1p(dropoff_counts),
            (pickup_counts + dropoff_counts > 0.0).astype(float),
        ]
    )
    transformer = GraphFeatureTransformer.from_config(
        GraphEmbeddingsConfig(
            encoder=GraphEncoderConfig(
                family=GraphEncoderFamily.GRAPHSAGE,
                input_dim=node_features.shape[1],
                hidden_dims=(graph_dim,),
                epochs=graph_epochs,
                seed=seed,
            ),
            directionality=DirectionalityConfig(
                compute_asymmetry_features=True,
                directional_feature_prefix="taxi_graph",
                directional_features=(
                    DirectionalFeature.SOURCE_TARGET_AFFINITY,
                    DirectionalFeature.FLOW_IMBALANCE_RATIO,
                ),
            ),
        )
    )
    bundle = transformer.fit_transform(
        node_features=node_features,
        edges=train_edges,
        node_count=node_count,
        directed=True,
    )
    embeddings = bundle.embeddings.astype(float)
    return np.hstack([dense_features(rows), embeddings[pickup], embeddings[dropoff]])


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - residual / total if total > 0.0 else 0.0


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_estimators: int,
) -> Any:
    model = CartoBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=2,
        split_policy="structured",
    )
    model.fit(x[train_idx], y[train_idx])
    return model.predict(x[test_idx])


def main() -> None:
    args = parse_args()
    rows = (
        load_taxi_rows(args.input, args.seed, args.sample_size)
        if args.input is not None
        else synthetic_taxi_rows(args.seed, args.sample_size)
    )
    train_idx, test_idx = train_test_split(len(rows["target"]), args.seed)
    y = rows["target"]
    dense = dense_features(rows)
    graph = graph_augmented_features(
        rows,
        train_idx,
        graph_dim=args.graph_dim,
        graph_epochs=args.graph_epochs,
        seed=args.seed,
    )
    dense_pred = fit_predict(dense, y, train_idx, test_idx, args.n_estimators)
    graph_pred = fit_predict(graph, y, train_idx, test_idx, args.n_estimators)
    print(
        json.dumps(
            {
                "rows": int(len(y)),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "dense": {"rmse": rmse(y[test_idx], dense_pred), "r2": r2(y[test_idx], dense_pred)},
                "graph_augmented": {
                    "rmse": rmse(y[test_idx], graph_pred),
                    "r2": r2(y[test_idx], graph_pred),
                    "feature_count": int(graph.shape[1]),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
