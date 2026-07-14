#!/usr/bin/env python3
"""Trace CartoBoost wall time across splitters, regressors, neural, and graph paths."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

DEFAULT_OUTPUT = ROOT / "target" / "walltime_trace.json"


class Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def time(self, case: str, stage: str, **metadata: Any) -> Any:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(
                {
                    "case": case,
                    "stage": stage,
                    "wall_ms": (time.perf_counter() - started) * 1000.0,
                    **metadata,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--graph-epochs", type=int, default=4)
    parser.add_argument("--graph-dim", type=int, default=8)
    parser.add_argument("--neural-dim", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def synthetic_dense(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(rows, dtype=np.float64)
    x = np.column_stack(
        [
            np.sin(idx * 0.017) * 10.0,
            np.cos(idx * 0.023) * 10.0,
            idx % 24,
            idx % 17,
            rng.normal(size=rows),
            rng.normal(size=rows),
            rng.normal(size=rows),
            rng.normal(size=rows),
        ]
    ).astype(np.float64)
    y = (
        np.where(x[:, 0] + x[:, 1] > 0.0, 1.0, -1.0)
        + np.where((x[:, 2] < 6.0) | (x[:, 2] > 20.0), 0.5, -0.5)
        + np.where((np.arange(rows) % 31) == 3, 1.5, 0.0)
        + 0.2 * x[:, 4]
    ).astype(np.float64)
    meta = {
        "ids": (np.arange(rows, dtype=np.uint64) % 97).astype(np.uint64),
        "sparse_sets": {
            "route_cells": [[int(row % 31), int((row // 7) % 31)] for row in range(rows)]
        },
    }
    return x, y, meta


def synthetic_graph(rows: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 123)
    node_count = max(48, int(math.sqrt(rows) * 3))
    source = rng.integers(0, node_count, size=rows)
    target = (source + rng.integers(1, max(3, node_count // 6), size=rows)) % node_count
    edges = sorted({(int(src), int(dst)) for src, dst in zip(source, target, strict=True)})
    axis = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
    node_features = np.column_stack(
        [np.sin(axis), np.cos(axis), np.linspace(-1.0, 1.0, node_count)]
    ).astype(np.float64)
    return {
        "node_count": node_count,
        "source": source,
        "target": target,
        "edges": edges,
        "node_features": node_features,
    }


def regressor_kwargs(args: argparse.Namespace, splitters: list[str]) -> dict[str, Any]:
    return {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_samples_leaf": 2,
        "min_gain": 0.0,
        "splitters": splitters,
    }


def trace_cartoboost_splitters(
    trace: Trace,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
) -> None:
    from cartoboost import CartoBoostRegressor

    splitter_cases = {
        "regressor:auto": None,
        "splitter:axis": ["axis"],
        "splitter:axis_histogram": ["axis_histogram:256"],
        "splitter:diagonal_2d": ["diagonal_2d"],
        "splitter:gaussian_2d": ["gaussian_2d"],
        "splitter:periodic": ["periodic:24"],
        "splitter:sparse_set": ["sparse_set"],
        "splitter:mixed": [
            "axis",
            "diagonal_2d",
            "gaussian_2d",
            "periodic:24",
            "sparse_set",
        ],
    }
    probes = x[: min(200, len(x))]
    for case, splitters in splitter_cases.items():
        kwargs = regressor_kwargs(args, splitters or ["auto"])
        if splitters is None:
            kwargs["splitters"] = None
        model = CartoBoostRegressor(**kwargs)
        uses_sparse = splitters == ["sparse_set"] or case.endswith("mixed")
        sparse_sets = meta["sparse_sets"] if uses_sparse else None
        with trace.time(case, "fit", rows=len(x), splitters=splitters):
            model.fit(x, y, sparse_sets=sparse_sets)
        with trace.time(case, "predict", rows=len(probes)):
            model.predict(
                probes,
                sparse_sets=None
                if sparse_sets is None
                else {name: rows[: len(probes)] for name, rows in sparse_sets.items()},
            )
        trace.events.append(
            {
                "case": case,
                "stage": "metadata",
                "wall_ms": 0.0,
                "training_config": getattr(model, "training_config_", None),
            }
        )


def trace_neural(
    trace: Trace,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
) -> None:
    from cartoboost import NeuralEmbeddingRegressor
    from cartoboost.neural.features import NeuralEmbeddingFeatures

    case = "net:embedding_table"
    features = NeuralEmbeddingFeatures(dim=args.neural_dim, random_state=args.seed)
    with trace.time(case, "fit_transform", rows=len(x), dim=args.neural_dim):
        embedding = features.fit_transform(ids, y)
    with trace.time(case, "transform", rows=len(x), dim=args.neural_dim):
        features.transform(ids)

    case = "regressor:neural_embedding"
    model = NeuralEmbeddingRegressor(
        dim=args.neural_dim,
        id_column=None,
        drop_id_column=False,
        random_state=args.seed,
        base_model_kwargs=regressor_kwargs(args, ["axis_histogram:256"]),
        final_model_kwargs=regressor_kwargs(args, ["axis_histogram:256"]),
    )
    with trace.time(case, "fit", rows=len(x), dim=args.neural_dim):
        model.fit(x, y, ids=ids)
    with trace.time(case, "predict", rows=min(200, len(x))):
        model.predict(x[: min(200, len(x))], ids=ids[: min(200, len(ids))])
    for stage, wall_ms in model.timings.items():
        trace.events.append({"case": case, "stage": stage, "wall_ms": wall_ms})
    trace.events.append(
        {
            "case": "net:embedding_table",
            "stage": "output_shape",
            "wall_ms": 0.0,
            "rows": int(embedding.shape[0]),
            "cols": int(embedding.shape[1]),
        }
    )


def graph_transformer(family: str, args: argparse.Namespace, input_dim: int) -> Any:
    from cartoboost.graph import GraphFeatureTransformer

    if family == "node2vec":
        return GraphFeatureTransformer.from_config(
            {
                "encoder": {
                    "family": "node2vec",
                    "dim": int(args.graph_dim),
                    "walk_length": 16,
                    "walks_per_node": 4,
                    "window_size": 4,
                    "epochs": int(args.graph_epochs),
                    "negative_samples": 3,
                    "seed": int(args.seed),
                }
            }
        )
    if family == "hetero_graphsage":
        return GraphFeatureTransformer.from_config(
            {
                "encoder": {
                    "family": "graphsage",
                    "hetero": True,
                    "input_dim": input_dim,
                    "hidden_dims": [int(args.graph_dim)],
                    "epochs": int(args.graph_epochs),
                    "seed": int(args.seed),
                }
            }
        )
    if family == "hinsage":
        return GraphFeatureTransformer.from_config(
            {
                "encoder": {
                    "family": "hinsage",
                    "input_dim": input_dim,
                    "node_type_count": 1,
                    "edge_type_triples": [(0, 0, 0)],
                    "hidden_dims": [int(args.graph_dim)],
                    "epochs": int(args.graph_epochs),
                    "neighbor_samples": [8],
                    "seed": int(args.seed),
                }
            }
        )
    return GraphFeatureTransformer.from_config(
        {
            "encoder": {
                "family": "graphsage",
                "input_dim": input_dim,
                "hidden_dims": [int(args.graph_dim)],
                "epochs": int(args.graph_epochs),
                "seed": int(args.seed),
            }
        }
    )


def trace_graphs(
    trace: Trace,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    graph: dict[str, Any],
) -> None:
    from cartoboost import CartoBoostRegressor

    families = ["node2vec", "graphsage", "hetero_graphsage", "hinsage"]
    for family in families:
        case = f"graph:{family}"
        try:
            edges = graph["edges"]
            if family == "hetero_graphsage":
                from cartoboost.graph import (
                    HeteroGraphSageConfig,
                    HeteroGraphSageFeatureEncoder,
                )

                encoder = HeteroGraphSageFeatureEncoder(
                    HeteroGraphSageConfig(
                        input_dim=int(graph["node_features"].shape[1]),
                        hidden_dims=[int(args.graph_dim)],
                        epochs=int(args.graph_epochs),
                        seed=int(args.seed),
                    )
                )
                typed_edges = [(source, target, 0) for source, target in edges]
                with trace.time(case, "feature_fit_transform", edges=len(edges)):
                    bundle = encoder.fit(
                        graph["node_features"],
                        typed_edges,
                        relation_ids=[0],
                        node_count=int(graph["node_count"]),
                        directed=True,
                    )
            elif family == "hinsage":
                from cartoboost.graph import HinSageConfig, HinSageFeatureEncoder

                encoder = HinSageFeatureEncoder(
                    HinSageConfig(
                        input_dim=int(graph["node_features"].shape[1]),
                        node_type_count=1,
                        edge_type_triples=[(0, 0, 0)],
                        hidden_dims=[int(args.graph_dim)],
                        epochs=int(args.graph_epochs),
                        neighbor_samples=[8],
                        seed=int(args.seed),
                    )
                )
                typed_edges = [(source, target, 0) for source, target in edges]
                with trace.time(case, "feature_fit_transform", edges=len(edges)):
                    bundle = encoder.fit(
                        graph["node_features"],
                        typed_edges,
                        [0] * int(graph["node_count"]),
                    )
            else:
                transformer = graph_transformer(family, args, int(graph["node_features"].shape[1]))
                fit_kwargs: dict[str, Any] = {
                    "node_features": graph["node_features"],
                    "node_count": graph["node_count"],
                    "directed": True,
                    "edges": edges,
                }
                with trace.time(case, "feature_fit_transform", edges=len(edges)):
                    bundle = transformer.fit_transform(**fit_kwargs)
        except Exception as exc:  # noqa: BLE001
            trace.events.append(
                {
                    "case": case,
                    "stage": "error",
                    "wall_ms": 0.0,
                    "error": str(exc),
                }
            )
            continue
        embeddings = bundle.embeddings.astype(np.float64)
        augmented = np.hstack(
            [
                x,
                embeddings[graph["source"]],
                embeddings[graph["target"]],
            ]
        )
        model = CartoBoostRegressor(**regressor_kwargs(args, ["axis_histogram:256"]))
        with trace.time(case, "downstream_regressor_fit", rows=len(augmented)):
            model.fit(augmented, y)
        with trace.time(case, "downstream_regressor_predict", rows=min(200, len(augmented))):
            model.predict(augmented[: min(200, len(augmented))])
        trace.events.append(
            {
                "case": case,
                "stage": "feature_shape",
                "wall_ms": 0.0,
                "rows": int(embeddings.shape[0]),
                "cols": int(embeddings.shape[1]),
                "feature_names": list(bundle.feature_names),
            }
        )


def summarize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    for event in events:
        totals[event["case"]] = totals.get(event["case"], 0.0) + float(event["wall_ms"])
    return [
        {"case": case, "total_wall_ms": total}
        for case, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def main() -> None:
    args = parse_args()
    trace = Trace()
    with trace.time("setup", "dense_data", rows=args.rows):
        x, y, meta = synthetic_dense(args.rows, args.seed)
    with trace.time("setup", "graph_data", rows=args.rows):
        graph = synthetic_graph(args.rows, args.seed)

    trace_cartoboost_splitters(trace, args, x, y, meta)
    trace_neural(trace, args, x, y, meta["ids"])
    trace_graphs(trace, args, x, y, graph)

    payload = {
        "artifact_version": 1,
        "rows": int(args.rows),
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "events": trace.events,
        "summary": summarize(trace.events),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"][:12]}, indent=2))


if __name__ == "__main__":
    main()
