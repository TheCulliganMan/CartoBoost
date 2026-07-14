"""Pickup-zone demand graph example for taxi data.

This example builds a zone graph from pickup demand rows. It demonstrates the
case LightGBM cannot express directly without manually materialized graph
features: a GraphSAGE encoder learns zone embeddings from zone adjacency and
zone-hour demand signals, then CartoBoost fits on the augmented rows.

    uv run python examples/04_taxi_pickup_zone_graph.py
"""

from __future__ import annotations

import json

import numpy as np
from cartoboost import CartoBoostRegressor
from cartoboost.graph import (
    GraphEmbeddingsConfig,
    GraphEncoderConfig,
    GraphEncoderFamily,
    GraphFeatureTransformer,
)

QUALITATIVE_SPLITTERS = ["axis_histogram:256", "diagonal_2d", "gaussian_2d"]


def build_pickup_demand() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rows: list[list[float]] = []
    target: list[float] = []
    zones = np.arange(1, 17)
    for zone in zones:
        for hour in range(24):
            for dayofweek in range(7):
                commute = 1.0 if hour in {7, 8, 17, 18} else 0.0
                weekend = 1.0 if dayofweek >= 5 else 0.0
                business_core = 1.0 if zone in {4, 5, 6, 7} else 0.0
                demand = 9.0 + 1.4 * zone + 7.0 * commute + 3.2 * weekend + 5.0 * business_core
                rows.append([float(zone), float(hour), float(dayofweek)])
                target.append(np.log1p(demand))
    adjacency = []
    for zone in zones:
        if zone > zones.min():
            adjacency.append((int(zone), int(zone - 1)))
        if zone < zones.max():
            adjacency.append((int(zone), int(zone + 1)))
        adjacency.append((int(zone), int(zone)))
    return (
        np.asarray(rows, dtype=float),
        np.asarray(target, dtype=float),
        zones.astype(int),
        adjacency,
    )


def spatial_holdout_indices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    holdout_zones = {4, 8, 12, 16}
    test_mask = np.asarray([int(zone) in holdout_zones for zone in x[:, 0]])
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def graph_features(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    zones: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    node_count = int(zones.max() + 1)
    train_zones = x[train_idx, 0].astype(int)
    sums = np.bincount(train_zones, weights=y[train_idx], minlength=node_count)
    counts = np.bincount(train_zones, minlength=node_count).astype(float)
    node_features = np.column_stack(
        [
            np.arange(node_count, dtype=float) / float(max(node_count - 1, 1)),
            np.divide(sums, np.maximum(counts, 1.0)),
            np.log1p(counts),
            (counts > 0.0).astype(float),
        ]
    )
    transformer = GraphFeatureTransformer.from_config(
        GraphEmbeddingsConfig(
            encoder=GraphEncoderConfig(
                family=GraphEncoderFamily.GRAPHSAGE,
                input_dim=node_features.shape[1],
                hidden_dims=(6,),
                epochs=6,
                seed=42,
            )
        )
    )
    bundle = transformer.fit_transform(
        node_features=node_features,
        edges=edges,
        node_count=node_count,
        directed=True,
    )
    zone_embeddings = bundle.embeddings.astype(float)[x[:, 0].astype(int)]
    return np.hstack([x, zone_embeddings])


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    model = CartoBoostRegressor(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=2,
        split_policy="structured",
    )
    model.fit(x[train_idx], y[train_idx])
    return model.predict(x[test_idx])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 0.0,
    }


def main() -> None:
    x, y, zones, edges = build_pickup_demand()
    train_idx, test_idx = spatial_holdout_indices(x)
    dense_pred = fit_predict(x, y, train_idx, test_idx)
    graph_x = graph_features(x, y, train_idx, zones, edges)
    graph_pred = fit_predict(graph_x, y, train_idx, test_idx)
    print(
        json.dumps(
            {
                "task": "pickup_zone_demand_spatial_holdout",
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "dense": metrics(y[test_idx], dense_pred),
                "graph_augmented": {
                    **metrics(y[test_idx], graph_pred),
                    "feature_count": int(graph_x.shape[1]),
                    "graph_edges": int(len(edges)),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
