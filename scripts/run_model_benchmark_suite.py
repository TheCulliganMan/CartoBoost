#!/usr/bin/env python3
"""Run deterministic model benchmarks across dense, neural, and graph workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "model_benchmarks"
GRAPH_MODEL_FAMILIES = {
    "cartoboost_graph": "graphsage",
    "cartoboost_graph_node2vec": "node2vec",
    "cartoboost_graph_graphsage": "graphsage",
    "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
    "cartoboost_graph_hinsage": "hinsage",
}
KARATE_CLUB_EDGES_1_INDEXED = [
    (2, 1),
    (3, 1),
    (3, 2),
    (4, 1),
    (4, 2),
    (4, 3),
    (5, 1),
    (6, 1),
    (7, 1),
    (7, 5),
    (7, 6),
    (8, 1),
    (8, 2),
    (8, 3),
    (8, 4),
    (9, 1),
    (9, 3),
    (10, 3),
    (11, 1),
    (11, 5),
    (11, 6),
    (12, 1),
    (13, 1),
    (13, 4),
    (14, 1),
    (14, 2),
    (14, 3),
    (14, 4),
    (17, 6),
    (17, 7),
    (18, 1),
    (18, 2),
    (20, 1),
    (20, 2),
    (22, 1),
    (22, 2),
    (26, 24),
    (26, 25),
    (28, 3),
    (28, 24),
    (28, 25),
    (29, 3),
    (30, 24),
    (30, 27),
    (31, 2),
    (31, 9),
    (32, 1),
    (32, 25),
    (32, 26),
    (32, 29),
    (33, 3),
    (33, 9),
    (33, 15),
    (33, 16),
    (33, 19),
    (33, 21),
    (33, 23),
    (33, 24),
    (33, 30),
    (33, 31),
    (33, 32),
    (34, 9),
    (34, 10),
    (34, 14),
    (34, 15),
    (34, 16),
    (34, 19),
    (34, 20),
    (34, 21),
    (34, 23),
    (34, 24),
    (34, 27),
    (34, 28),
    (34, 29),
    (34, 30),
    (34, 31),
    (34, 32),
    (34, 33),
]
KARATE_CLUB_MR_HI_ZERO_INDEXED = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21}
STANDALONE_MODELS = {
    "neural_embedding_regressor",
    "node2vec_regressor",
    "graphsage_regressor",
    "hetero_graphsage_regressor",
    "hinsage_regressor",
    "node2vec_link_predictor",
    "graphsage_link_predictor",
    "hetero_graphsage_link_predictor",
    "hinsage_link_predictor",
}
QUALITATIVE_CARTOBOOST_SPLITTERS = [
    "axis_histogram:256",
    "diagonal_2d",
    "gaussian_2d",
]
CARTOBOOST_MIN_SAMPLES_LEAF = 20
NEURAL_AUGMENTATION_RELATIVE_VALIDATION_MARGIN = 0.01
CARTOBOOST_FAMILY_MODELS = {
    "cartoboost",
    "cartoboost_neural",
    *GRAPH_MODEL_FAMILIES,
    "neural_embedding_regressor",
    "node2vec_regressor",
    "graphsage_regressor",
    "hetero_graphsage_regressor",
    "hinsage_regressor",
}
EXTERNAL_REGRESSION_BASELINES = {
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "ridge",
    "mean",
}
MODEL_DEPENDENCIES = {
    "xgboost": ("xgboost", "XGBRegressor"),
    "lightgbm": ("lightgbm", "LGBMRegressor"),
    "catboost": ("catboost", "CatBoostRegressor"),
}
SKLEARN_MODELS = {
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "ridge",
}
SKLEARN_DATASETS = {
    "diabetes",
    "california_housing",
}


@dataclass(frozen=True)
class Workload:
    name: str
    display_name: str
    description: str
    source: str
    features: np.ndarray
    target: np.ndarray
    split_group: np.ndarray | None = None
    embedding_ids: np.ndarray | None = None
    graph_source: np.ndarray | None = None
    graph_target: np.ndarray | None = None
    graph_edges: list[tuple[int, int]] | None = None
    graph_node_features: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeat-seeds",
        default="",
        help=(
            "Optional comma-separated seeds for repeated evidence. The primary --seed remains "
            "the detailed public run; additional seeds are summarized and included in JSONL."
        ),
    )
    parser.add_argument("--n-rows", type=int, default=2_400)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--datasets",
        default="normal,neural,graph",
        help=(
            "Comma-separated workloads from: normal, neural, graph, bearing, diabetes, "
            "california_housing, karate"
        ),
    )
    parser.add_argument(
        "--models",
        default=(
            "mean,cartoboost,cartoboost_neural,cartoboost_graph_node2vec,"
            "cartoboost_graph_graphsage,cartoboost_graph_hetero_graphsage,"
            "cartoboost_graph_hinsage,neural_embedding_regressor,"
            "node2vec_regressor,graphsage_regressor,hetero_graphsage_regressor,"
            "hinsage_regressor,node2vec_link_predictor,graphsage_link_predictor,"
            "hetero_graphsage_link_predictor,hinsage_link_predictor,xgboost,lightgbm,"
            "catboost,hist_gradient_boosting,random_forest,extra_trees,ridge"
        ),
        help=(
            "Comma-separated models from: mean, cartoboost, cartoboost_neural, "
            "cartoboost_graph, cartoboost_graph_node2vec, cartoboost_graph_graphsage, "
            "cartoboost_graph_hetero_graphsage, cartoboost_graph_hinsage, "
            "neural_embedding_regressor, node2vec_regressor, graphsage_regressor, "
            "hetero_graphsage_regressor, hinsage_regressor, node2vec_link_predictor, "
            "graphsage_link_predictor, hetero_graphsage_link_predictor, "
            "hinsage_link_predictor, xgboost, lightgbm, catboost, "
            "hist_gradient_boosting, random_forest, extra_trees, ridge"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--neural-dim", type=int, default=12)
    parser.add_argument("--graph-dim", type=int, default=8)
    parser.add_argument("--graph-epochs", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=["fixed", "validation_search"],
        default="validation_search",
        help=(
            "fixed keeps one configured row per model. validation_search evaluates an "
            "equal-size inner-validation candidate grid for every tunable model family, "
            "then retrains the selected candidate on the full outer training split."
        ),
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=None,
        help=(
            "Optional shared minimum-leaf override used by CartoBoost-family and tree "
            "baselines when a candidate grid varies leaf regularization."
        ),
    )
    parser.add_argument(
        "--validation-trials",
        type=int,
        default=3,
        help="Number of validation-search candidates per tunable model family.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def dense_normal_workload(*, n_rows: int, seed: int) -> Workload:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 10))
    y = (
        1.7 * x[:, 0]
        - 0.9 * x[:, 1]
        + 0.45 * x[:, 2] ** 2
        + np.sin(x[:, 3] * 1.8)
        + 0.25 * x[:, 4] * x[:, 5]
        + rng.normal(0.0, 0.28, size=n_rows)
    )
    return Workload(
        name="normal",
        display_name="Normal dense",
        description="IID dense numeric regression with nonlinear feature interactions.",
        source=(
            "Synthetic workload generated by scripts/run_model_benchmark_suite.py from "
            "--seed and --n-rows."
        ),
        features=x.astype(np.float64),
        target=y.astype(np.float64),
    )


def neural_workload(*, n_rows: int, seed: int) -> Workload:
    rng = np.random.default_rng(seed + 17)
    n_cells = max(64, int(math.sqrt(n_rows) * 4))
    x = rng.normal(size=(n_rows, 8))
    ids = rng.integers(0, n_cells, size=n_rows, dtype=np.uint64)
    cell_float = ids.astype(np.float64)
    cell_effect = np.sin(cell_float / 5.0) + 0.5 * np.cos(cell_float / 13.0)
    y = (
        1.25 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.35 * x[:, 2] ** 2
        + 0.85 * cell_effect
        + rng.normal(0.0, 0.30, size=n_rows)
    )
    features = np.column_stack([x, cell_float])
    return Workload(
        name="neural",
        display_name="Neural ID",
        description=(
            "Dense regression with repeated cell IDs whose residual signal is learnable by "
            "embedding features."
        ),
        source=(
            "Synthetic workload generated by scripts/run_model_benchmark_suite.py from "
            "--seed and --n-rows."
        ),
        features=features.astype(np.float64),
        target=y.astype(np.float64),
        split_group=ids.astype(np.int64),
        embedding_ids=ids,
    )


def graph_workload(*, n_rows: int, seed: int) -> Workload:
    rng = np.random.default_rng(seed + 31)
    n_nodes = max(48, int(math.sqrt(n_rows) * 3))
    source = rng.integers(0, n_nodes, size=n_rows)
    target = (source + rng.integers(1, max(3, n_nodes // 6), size=n_rows)) % n_nodes
    x = rng.normal(size=(n_rows, 6))

    node_axis = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    node_features = np.column_stack(
        [
            np.sin(node_axis),
            np.cos(node_axis),
            np.linspace(-1.0, 1.0, n_nodes),
        ]
    )
    src_effect = 0.7 * node_features[source, 0] + 0.25 * node_features[source, 2]
    dst_effect = -0.4 * node_features[target, 1] + 0.15 * node_features[target, 2]
    flow_effect = np.sin((source - target).astype(float) / 4.0)
    y = (
        1.1 * x[:, 0]
        - 0.75 * x[:, 1]
        + 0.25 * x[:, 2] * x[:, 3]
        + src_effect
        + dst_effect
        + 0.55 * flow_effect
        + rng.normal(0.0, 0.26, size=n_rows)
    )
    features = np.column_stack([x, source.astype(float), target.astype(float)])
    edges = sorted({(int(src), int(dst)) for src, dst in zip(source, target, strict=True)})
    return Workload(
        name="graph",
        display_name="Graph source-target",
        description=(
            "Directed source-target regression where graph topology and node features carry "
            "predictive signal."
        ),
        source=(
            "Synthetic workload generated by scripts/run_model_benchmark_suite.py from "
            "--seed and --n-rows."
        ),
        features=features.astype(np.float64),
        target=y.astype(np.float64),
        split_group=source.astype(np.int64),
        graph_source=source,
        graph_target=target,
        graph_edges=edges,
        graph_node_features=node_features.astype(np.float64),
    )


def bearing_workload(*, n_rows: int, seed: int) -> Workload:
    from cartoboost.geo import clockwise_bearing_unit_vectors

    rng = np.random.default_rng(seed + 47)
    origins = rng.normal(0.0, 2.5, size=(n_rows, 2))
    headings = rng.uniform(-np.pi, np.pi, size=n_rows)
    distances = rng.gamma(shape=2.0, scale=1.2, size=n_rows) + 0.2
    destinations = (
        origins + np.column_stack([np.sin(headings), np.cos(headings)]) * distances[:, None]
    )
    bearings = clockwise_bearing_unit_vectors(
        [tuple(row) for row in origins],
        [tuple(row) for row in destinations],
    )
    bearing_features = np.asarray(
        [[0.0, 0.0] if value is None else [value[0], value[1]] for value in bearings],
        dtype=np.float64,
    )
    radial_distance = np.sqrt(np.sum(origins * origins, axis=1))
    y = (
        0.9 * bearing_features[:, 0]
        + 1.25 * bearing_features[:, 1]
        + 0.18 * radial_distance
        + 0.35 * np.sin(distances)
        + rng.normal(0.0, 0.18, size=n_rows)
    )
    features = np.column_stack(
        [origins, destinations, distances, radial_distance, bearing_features]
    )
    return Workload(
        name="bearing",
        display_name="Bearing unit-vector",
        description=(
            "Synthetic source-target spatial regression where continuous bearing unit-vector "
            "features and radial distance carry signal."
        ),
        source=(
            "Synthetic workload generated by scripts/run_model_benchmark_suite.py with "
            "cartoboost.geo.clockwise_bearing_unit_vectors."
        ),
        features=features.astype(np.float64),
        target=y.astype(np.float64),
    )


def diabetes_workload(*, n_rows: int, seed: int) -> Workload:
    del n_rows, seed
    from sklearn.datasets import load_diabetes

    dataset = load_diabetes()
    return Workload(
        name="diabetes",
        display_name="sklearn diabetes",
        description=(
            "Frozen public scikit-learn diabetes regression workload with 442 rows, "
            "10 numeric features, and disease-progression target."
        ),
        source="sklearn.datasets.load_diabetes bundled public regression dataset.",
        features=np.asarray(dataset.data, dtype=np.float64),
        target=np.asarray(dataset.target, dtype=np.float64),
    )


def california_housing_workload(*, n_rows: int, seed: int) -> Workload:
    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing(download_if_missing=True)
    features = np.asarray(dataset.data, dtype=np.float64)
    target = np.asarray(dataset.target, dtype=np.float64)
    if 0 < n_rows < len(target):
        rng = np.random.default_rng(seed + 2024)
        indices = np.sort(rng.choice(len(target), size=n_rows, replace=False))
        features = features[indices]
        target = target[indices]
        sample_note = f" deterministic {n_rows}-row seed-{seed} sample from"
    else:
        sample_note = " full"
    return Workload(
        name="california_housing",
        display_name="California housing",
        description=(
            "Public scikit-learn California housing regression workload with eight "
            "numeric census-block features and median house value target."
        ),
        source=(
            "sklearn.datasets.fetch_california_housing"
            f"{sample_note} the 20,640-row public California housing dataset."
        ),
        features=features,
        target=target,
    )


def karate_workload(*, n_rows: int, seed: int) -> Workload:
    del n_rows, seed
    edges = np.asarray(
        [(source - 1, target - 1) for source, target in KARATE_CLUB_EDGES_1_INDEXED],
        dtype=np.int64,
    )
    source = edges[:, 0]
    target = edges[:, 1]
    node_count = 34
    degrees = np.zeros(node_count, dtype=np.float64)
    for left, right in edges:
        degrees[int(left)] += 1.0
        degrees[int(right)] += 1.0
    labels = np.asarray(
        [1.0 if node in KARATE_CLUB_MR_HI_ZERO_INDEXED else 0.0 for node in range(node_count)],
        dtype=np.float64,
    )
    y = (labels[source] == labels[target]).astype(np.float64)
    features = np.column_stack(
        [
            source.astype(np.float64),
            target.astype(np.float64),
            degrees[source],
            degrees[target],
            np.abs(degrees[source] - degrees[target]),
        ]
    )
    node_axis = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
    node_features = np.column_stack(
        [
            degrees,
            labels,
            np.sin(node_axis),
            np.cos(node_axis),
        ]
    )
    return Workload(
        name="karate",
        display_name="Zachary karate club",
        description=(
            "Frozen public 78-edge Zachary karate club graph workload. Rows are observed "
            "edges; the regression target is whether the two endpoints share the same "
            "post-split club label."
        ),
        source=(
            "Embedded Zachary karate club edge list and post-split labels from the benchmark "
            "harness constants."
        ),
        features=features.astype(np.float64),
        target=y.astype(np.float64),
        split_group=source.astype(np.int64),
        graph_source=source,
        graph_target=target,
        graph_edges=[(int(left), int(right)) for left, right in edges],
        graph_node_features=node_features.astype(np.float64),
    )


def build_workloads(args: argparse.Namespace) -> list[Workload]:
    requested = [part.strip() for part in args.datasets.split(",") if part.strip()]
    builders = {
        "normal": dense_normal_workload,
        "neural": neural_workload,
        "graph": graph_workload,
        "bearing": bearing_workload,
        "diabetes": diabetes_workload,
        "california_housing": california_housing_workload,
        "karate": karate_workload,
    }
    unknown = sorted(set(requested) - set(builders))
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")
    return [builders[name](n_rows=args.n_rows, seed=args.seed) for name in requested]


def random_split(n_rows: int, *, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    split = int(n_rows * train_frac)
    if split <= 0 or split >= n_rows:
        raise ValueError("--train-frac must keep both train and validation rows")
    return indices[:split], indices[split:]


def deterministic_inner_validation_split(
    row_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if row_count < 10:
        indices = np.arange(row_count)
        return indices, indices
    rng = np.random.default_rng(seed + 997)
    permutation = rng.permutation(row_count)
    validation_count = max(1, int(row_count * 0.2))
    validation = permutation[:validation_count]
    train = permutation[validation_count:]
    if len(train) == 0:
        return permutation, permutation
    return train, validation


def group_holdout_split(
    groups: np.ndarray,
    *,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    split = max(1, min(len(unique) - 1, int(len(unique) * train_frac)))
    train_groups = set(int(value) for value in unique[:split])
    train_mask = np.asarray([int(value) in train_groups for value in groups], dtype=bool)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return random_split(len(groups), train_frac=train_frac, seed=seed)
    return train_idx, test_idx


def split_workload(
    workload: Workload,
    *,
    train_frac: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    splits = {"random": random_split(len(workload.target), train_frac=train_frac, seed=seed)}
    if workload.split_group is not None:
        splits["group_holdout"] = group_holdout_split(
            workload.split_group,
            train_frac=train_frac,
            seed=seed + 101,
        )
    return splits


def requested_datasets(args: argparse.Namespace) -> list[str]:
    return [part.strip() for part in args.datasets.split(",") if part.strip()]


def repeat_seeds(args: argparse.Namespace) -> list[int]:
    raw = str(getattr(args, "repeat_seeds", "") or "").strip()
    if not raw:
        return []
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        return []
    return list(dict.fromkeys(seeds))


def split_definitions() -> dict[str, dict[str, str]]:
    return {
        "random": {
            "kind": "seeded_row_shuffle",
            "train_fraction": "configured_by_--train-frac",
            "purpose": "interpolation across rows drawn from the same workload distribution",
        },
        "group_holdout": {
            "kind": "seeded_group_holdout",
            "train_fraction": "configured_by_--train-frac_over_unique_groups",
            "purpose": "cold-group generalization for workloads with repeated IDs or graph sources",
        },
    }


def update_hash_with_array(digest: Any, label: str, values: np.ndarray | None) -> None:
    digest.update(label.encode("utf-8"))
    if values is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())


def workload_fingerprint(workload: Workload) -> str:
    digest = hashlib.sha256()
    digest.update(workload.name.encode("utf-8"))
    digest.update(workload.source.encode("utf-8"))
    update_hash_with_array(digest, "features", workload.features)
    update_hash_with_array(digest, "target", workload.target)
    update_hash_with_array(digest, "split_group", workload.split_group)
    update_hash_with_array(digest, "embedding_ids", workload.embedding_ids)
    update_hash_with_array(digest, "graph_source", workload.graph_source)
    update_hash_with_array(digest, "graph_target", workload.graph_target)
    update_hash_with_array(digest, "graph_node_features", workload.graph_node_features)
    graph_edges = (
        np.asarray(workload.graph_edges, dtype=np.int64)
        if workload.graph_edges is not None
        else None
    )
    update_hash_with_array(digest, "graph_edges", graph_edges)
    return digest.hexdigest()


def index_fingerprint(indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    update_hash_with_array(digest, "indices", indices.astype(np.int64, copy=False))
    return digest.hexdigest()


def selection_policy() -> dict[str, str]:
    return {
        "global_hyperparameters": (
            "every tunable model family uses the same inner-validation budget before "
            "holdout scoring; no model family uses test labels for tuning"
        ),
        "primary_cartoboost_row": (
            "single selected cartoboost run retrained after inner validation; no internal "
            "candidate is selected on test metrics"
        ),
        "neural_feature_gate": (
            "uses deterministic inner train/validation rows inside the training split only"
        ),
        "graph_feature_gate": (
            "uses deterministic inner train/validation rows inside the training split only"
        ),
        "external_baseline_selection": (
            "best external baseline is selected only for reporting after each model is scored"
        ),
        "diagnostic_rows": (
            "graph, neural, and link-prediction rows are diagnostics and are not substitutes "
            "for the primary cartoboost comparison row"
        ),
    }


def resource_usage_snapshot() -> dict[str, Any]:
    return {
        "cpu": platform.processor() or platform.machine(),
        "threads": os.cpu_count(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rustc": rustc_version(),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_status(
    import_name: str,
    class_name: str | None = None,
    *,
    distribution_name: str | None = None,
) -> dict[str, Any]:
    module = optional_import(import_name)
    package_name = distribution_name or import_name
    status = {
        "package": package_name,
        "import_name": import_name,
        "version": package_version(package_name),
        "module_importable": module is not None,
    }
    if class_name is not None:
        status["required_class"] = class_name
        status["required_class_available"] = bool(
            module is not None and hasattr(module, class_name)
        )
    return status


def baseline_environment_snapshot() -> dict[str, Any]:
    return {
        "sklearn": dependency_status("sklearn", distribution_name="scikit-learn"),
        "xgboost": dependency_status("xgboost", "XGBRegressor"),
        "lightgbm": dependency_status("lightgbm", "LGBMRegressor"),
        "catboost": dependency_status("catboost", "CatBoostRegressor"),
    }


def required_benchmark_dependencies(
    models: list[str], datasets: list[str]
) -> dict[str, str | None]:
    required: dict[str, str | None] = {}
    if any(model in SKLEARN_MODELS for model in models) or any(
        dataset in SKLEARN_DATASETS for dataset in datasets
    ):
        required["sklearn"] = None
    for model in models:
        dependency = MODEL_DEPENDENCIES.get(model)
        if dependency is not None:
            required[dependency[0]] = dependency[1]
    return required


def validate_requested_benchmark_dependencies(models: list[str], datasets: list[str]) -> None:
    missing = []
    for import_name, class_name in required_benchmark_dependencies(models, datasets).items():
        package_name = "scikit-learn" if import_name == "sklearn" else import_name
        status = dependency_status(
            import_name,
            class_name,
            distribution_name="scikit-learn" if import_name == "sklearn" else None,
        )
        if not status["module_importable"]:
            missing.append(f"{package_name} (import {import_name})")
        elif class_name is not None and not status.get("required_class_available", False):
            missing.append(f"{package_name}.{class_name}")
    if missing:
        raise RuntimeError(
            "requested benchmark dependencies are unavailable: "
            + ", ".join(missing)
            + ". Install them with `uv sync --group bench --group dev` before rerunning."
        )


def rustc_version() -> str | None:
    try:
        completed = subprocess.run(
            ["rustc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    version = completed.stdout.strip()
    return version or None


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    absolute_residual = np.abs(residual)
    rmse = float(np.sqrt(np.mean(residual**2)))
    total_sum_of_squares = float(np.sum((actual - float(np.mean(actual))) ** 2))
    residual_sum_of_squares = float(np.sum(residual**2))
    r2 = (
        1.0 - residual_sum_of_squares / total_sum_of_squares
        if total_sum_of_squares > 0.0
        else float("nan")
    )
    absolute_actual_sum = float(np.sum(np.abs(actual)))
    wape = (
        float(np.sum(absolute_residual) / absolute_actual_sum)
        if absolute_actual_sum > 0.0
        else float("nan")
    )
    return {
        "mae": float(np.mean(absolute_residual)),
        "rmse": rmse,
        "r2": float(r2),
        "wape": wape,
    }


def timed_fit_predict(
    fit: Any,
    predict: Any,
    test_rows: int,
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    fit()
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = np.asarray(predict(), dtype=np.float64)
    predict_seconds = time.perf_counter() - start
    rps = float(test_rows / predict_seconds) if predict_seconds > 0 else float("inf")
    return prediction, {
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "predict_rows_per_second": rps,
    }


def run_mean(train_y: np.ndarray, test_rows: int) -> tuple[np.ndarray, dict[str, float]]:
    value = 0.0

    def fit() -> None:
        nonlocal value
        value = float(np.mean(train_y))

    return timed_fit_predict(fit, lambda: np.full(test_rows, value), test_rows)


def min_samples_leaf_value(args: argparse.Namespace, *, default: int) -> int:
    value = getattr(args, "min_samples_leaf", None)
    return int(value) if value is not None else int(default)


def cartoboost_model(args: argparse.Namespace) -> Any:
    from cartoboost import CartoBoostRegressor

    min_samples_leaf = min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF)

    return CartoBoostRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=min_samples_leaf,
        min_gain=0.0,
        splitters=QUALITATIVE_CARTOBOOST_SPLITTERS,
        random_state=args.seed,
        n_threads=args.n_threads or None,
    )


def run_cartoboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    model = cartoboost_model(args)
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return (
        prediction,
        timing,
        {
            "backend": getattr(model, "_backend_used", None),
            "min_samples_leaf": min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
        },
    )


def run_neural(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if workload.embedding_ids is None:
        raise ValueError("neural benchmark requires embedding ids")
    from cartoboost import NeuralEmbeddingRegressor

    def build_neural_model() -> NeuralEmbeddingRegressor:
        return NeuralEmbeddingRegressor(
            dim=args.neural_dim,
            drop_id_column=False,
            id_column=None,
            random_state=args.seed,
            oof_folds=5,
            support_prior_strength=2.0,
            base_model_kwargs={
                "n_estimators": max(10, args.n_estimators // 2),
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "min_samples_leaf": min_samples_leaf_value(
                    args, default=CARTOBOOST_MIN_SAMPLES_LEAF
                ),
                "splitters": QUALITATIVE_CARTOBOOST_SPLITTERS,
            },
            final_model_kwargs={
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "min_samples_leaf": min_samples_leaf_value(
                    args, default=CARTOBOOST_MIN_SAMPLES_LEAF
                ),
                "splitters": QUALITATIVE_CARTOBOOST_SPLITTERS,
            },
        )

    train_x = workload.features[train_idx]
    test_x = workload.features[test_idx]
    train_y = workload.target[train_idx]
    train_ids = workload.embedding_ids[train_idx]
    test_ids = workload.embedding_ids[test_idx]
    train_id_set = {int(value) for value in np.asarray(train_ids).ravel()}
    cold_id_fraction = float(
        np.mean([int(value) not in train_id_set for value in np.asarray(test_ids).ravel()])
    )

    inner_train, inner_validation = deterministic_inner_validation_split(
        len(train_y),
        seed=args.seed,
    )
    base_probe = cartoboost_model(args)
    neural_probe = build_neural_model()
    base_probe.fit(train_x[inner_train], train_y[inner_train])
    neural_probe.fit(
        train_x[inner_train],
        train_y[inner_train],
        ids=train_ids[inner_train],
    )
    base_validation_prediction = base_probe.predict(train_x[inner_validation])
    neural_validation_prediction = neural_probe.predict(
        train_x[inner_validation],
        ids=train_ids[inner_validation],
    )
    base_validation_rmse = float(
        np.sqrt(np.mean((train_y[inner_validation] - base_validation_prediction) ** 2))
    )
    neural_validation_rmse = float(
        np.sqrt(np.mean((train_y[inner_validation] - neural_validation_prediction) ** 2))
    )
    required_neural_rmse = base_validation_rmse * (
        1.0 - NEURAL_AUGMENTATION_RELATIVE_VALIDATION_MARGIN
    )
    use_neural = cold_id_fraction < 0.5 and neural_validation_rmse <= required_neural_rmse

    if use_neural:
        model = build_neural_model()
        prediction, timing = timed_fit_predict(
            lambda: model.fit(train_x, train_y, ids=train_ids),
            lambda: model.predict(test_x, ids=test_ids),
            len(test_idx),
        )
        fit_stages_ms = model.timings
        selected = "neural_augmented"
    else:
        model = cartoboost_model(args)
        prediction, timing = timed_fit_predict(
            lambda: model.fit(train_x, train_y),
            lambda: model.predict(test_x),
            len(test_idx),
        )
        fit_stages_ms = {}
        selected = "base"
    return (
        prediction,
        timing,
        {
            "embedding_dim": int(args.neural_dim),
            "oof_folds": 5,
            "support_prior_strength": 2.0,
            "min_samples_leaf": min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
            "fit_stages_ms": fit_stages_ms,
            "neural_guard": {
                "selected": selected,
                "base_validation_rmse": base_validation_rmse,
                "neural_validation_rmse": neural_validation_rmse,
                "cold_id_fraction": cold_id_fraction,
                "relative_validation_margin": NEURAL_AUGMENTATION_RELATIVE_VALIDATION_MARGIN,
            },
        },
    )


def graph_augmented_features(
    workload: Workload,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    graph_family: str,
) -> np.ndarray:
    if (
        workload.graph_source is None
        or workload.graph_target is None
        or workload.graph_edges is None
        or workload.graph_node_features is None
    ):
        raise ValueError("graph benchmark requires source, target, edges, and node features")
    from cartoboost.graph import GraphFeatureTransformer

    train_pairs = {
        (int(workload.graph_source[index]), int(workload.graph_target[index]))
        for index in train_idx
    }
    train_edges = sorted(train_pairs) or workload.graph_edges
    from cartoboost.graph import (
        DirectionalFeature,
        DirectionalityConfig,
        GraphEmbeddingsConfig,
        GraphEncoderConfig,
        GraphEncoderFamily,
    )

    node_count = int(workload.graph_node_features.shape[0])
    if graph_family == "node2vec":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.NODE2VEC,
            dim=int(args.graph_dim),
            walk_length=16,
            walks_per_node=4,
            window_size=4,
            epochs=int(args.graph_epochs),
            negative_samples=3,
            seed=int(args.seed),
        )
        fit_edges: list[tuple[int, int]] | list[tuple[int, int, int]] = train_edges
        fit_kwargs: dict[str, Any] = {}
    elif graph_family == "hetero_graphsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.HINSAGE,
            hetero=True,
            input_dim=int(workload.graph_node_features.shape[1]),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
        )
        fit_edges = [(source, target, 0) for source, target in train_edges]
        fit_kwargs = {"relation_ids": [0]}
    elif graph_family == "hinsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.HINSAGE,
            input_dim=int(workload.graph_node_features.shape[1]),
            node_type_count=1,
            edge_type_triples=((0, 0, 0),),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
            neighbor_samples=(8,),
        )
        fit_edges = [(source, target, 0) for source, target in train_edges]
        fit_kwargs = {"node_types": [0] * node_count}
    elif graph_family == "graphsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.GRAPHSAGE,
            input_dim=int(workload.graph_node_features.shape[1]),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
        )
        fit_edges = train_edges
        fit_kwargs = {}
    else:
        raise ValueError(f"unsupported graph family {graph_family!r}")

    transformer = GraphFeatureTransformer.from_config(
        GraphEmbeddingsConfig(
            encoder=encoder,
            directionality=DirectionalityConfig(
                compute_asymmetry_features=True,
                directional_feature_prefix="graph",
                directional_features=(
                    DirectionalFeature.SOURCE_TARGET_AFFINITY,
                    DirectionalFeature.FLOW_IMBALANCE_RATIO,
                ),
            ),
        )
    )
    bundle = transformer.fit_transform(
        node_features=workload.graph_node_features,
        edges=fit_edges,
        node_count=node_count,
        directed=True,
        **fit_kwargs,
    )
    node_embeddings = bundle.embeddings.astype(np.float64)
    source_emb = node_embeddings[workload.graph_source]
    target_emb = node_embeddings[workload.graph_target]
    return np.hstack([workload.features, source_emb, target_emb])


def run_graph(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    graph_family: str,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    augmented = graph_augmented_features(workload, train_idx, args, graph_family)
    prediction, timing, config = run_cartoboost(
        augmented[train_idx],
        workload.target[train_idx],
        augmented[test_idx],
        args,
    )
    config.update(
        {
            "graph_family": graph_family,
            "graph_dim": int(args.graph_dim),
            "graph_epochs": int(args.graph_epochs),
        }
    )
    return prediction, timing, config


def run_standalone_neural(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if workload.embedding_ids is None:
        raise ValueError("standalone neural benchmark requires embedding ids")
    from cartoboost import NeuralEmbeddingStandaloneRegressor

    model = NeuralEmbeddingStandaloneRegressor(
        dim=args.neural_dim,
        random_state=args.seed,
        support_prior_strength=2.0,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(
            workload.embedding_ids[train_idx],
            workload.target[train_idx],
            dense=workload.features[train_idx],
        ),
        lambda: model.predict(workload.embedding_ids[test_idx], dense=workload.features[test_idx]),
        len(test_idx),
    )
    return prediction, timing, {"embedding_dim": int(args.neural_dim), "backend": "rust"}


def run_standalone_node2vec_regressor(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if (
        workload.graph_source is None
        or workload.graph_target is None
        or workload.graph_edges is None
    ):
        raise ValueError("standalone node2vec benchmark requires graph topology")
    from cartoboost import Node2VecStandaloneRegressor

    node_count = int(max(workload.graph_source.max(), workload.graph_target.max()) + 1)
    train_edges = sorted(
        {
            (int(workload.graph_source[index]), int(workload.graph_target[index]))
            for index in train_idx
        }
    )
    model = Node2VecStandaloneRegressor(
        dim=args.graph_dim,
        walk_length=16,
        walks_per_node=4,
        window_size=4,
        epochs=args.graph_epochs,
        negative_samples=3,
        seed=args.seed,
        n_estimators=args.n_estimators,
        booster_learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(
            node_count=node_count,
            edges=train_edges or workload.graph_edges,
            row_nodes=workload.graph_source[train_idx],
            row_targets=workload.graph_target[train_idx],
            dense=workload.features[train_idx],
            y=workload.target[train_idx],
        ),
        lambda: model.predict(
            workload.graph_source[test_idx],
            row_targets=workload.graph_target[test_idx],
            dense=workload.features[test_idx],
        ),
        len(test_idx),
    )
    return (
        prediction,
        timing,
        {"graph_dim": int(args.graph_dim), "graph_epochs": int(args.graph_epochs)},
    )


def run_standalone_graphsage_regressor(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if (
        workload.graph_source is None
        or workload.graph_target is None
        or workload.graph_edges is None
        or workload.graph_node_features is None
    ):
        raise ValueError("standalone graphsage benchmark requires graph topology")
    from cartoboost import GraphSageStandaloneRegressor

    train_edges = sorted(
        {
            (int(workload.graph_source[index]), int(workload.graph_target[index]))
            for index in train_idx
        }
    )
    model = GraphSageStandaloneRegressor(
        input_dim=int(workload.graph_node_features.shape[1]),
        hidden_dims=(args.graph_dim,),
        epochs=args.graph_epochs,
        seed=args.seed,
        n_estimators=args.n_estimators,
        booster_learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(
            node_features=workload.graph_node_features,
            edges=train_edges or workload.graph_edges,
            row_nodes=workload.graph_source[train_idx],
            row_targets=workload.graph_target[train_idx],
            dense=workload.features[train_idx],
            y=workload.target[train_idx],
        ),
        lambda: model.predict(
            node_features=workload.graph_node_features,
            row_nodes=workload.graph_source[test_idx],
            row_targets=workload.graph_target[test_idx],
            dense=workload.features[test_idx],
        ),
        len(test_idx),
    )
    return (
        prediction,
        timing,
        {"graph_dim": int(args.graph_dim), "graph_epochs": int(args.graph_epochs)},
    )


def run_standalone_typed_graphsage_regressor(
    model_name: str,
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if (
        workload.graph_source is None
        or workload.graph_target is None
        or workload.graph_edges is None
        or workload.graph_node_features is None
    ):
        raise ValueError("standalone typed graphsage benchmark requires graph topology")
    graph_edges = workload.graph_edges
    train_edges = sorted(
        {
            (int(workload.graph_source[index]), int(workload.graph_target[index]), 0)
            for index in train_idx
        }
    )
    fit_kwargs: dict[str, Any] = {}
    if model_name == "hetero_graphsage_regressor":
        from cartoboost import HeteroGraphSageStandaloneRegressor

        model = HeteroGraphSageStandaloneRegressor(
            input_dim=int(workload.graph_node_features.shape[1]),
            relation_count=1,
            hidden_dims=(args.graph_dim,),
            epochs=args.graph_epochs,
            seed=args.seed,
            n_estimators=args.n_estimators,
            booster_learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_samples_leaf=min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
        )
    else:
        from cartoboost import HinSageStandaloneRegressor

        model = HinSageStandaloneRegressor(
            input_dim=int(workload.graph_node_features.shape[1]),
            node_type_count=1,
            edge_type_triples=[(0, 0, 0)],
            hidden_dims=(args.graph_dim,),
            epochs=args.graph_epochs,
            seed=args.seed,
            n_estimators=args.n_estimators,
            booster_learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_samples_leaf=min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF),
        )
        fit_kwargs["node_types"] = [0] * int(workload.graph_node_features.shape[0])

    prediction, timing = timed_fit_predict(
        lambda: model.fit(
            node_features=workload.graph_node_features,
            edges=train_edges or [(source, target, 0) for source, target in graph_edges],
            row_nodes=workload.graph_source[train_idx],
            row_targets=workload.graph_target[train_idx],
            dense=workload.features[train_idx],
            y=workload.target[train_idx],
            **fit_kwargs,
        ),
        lambda: model.predict(
            node_features=workload.graph_node_features,
            row_nodes=workload.graph_source[test_idx],
            row_targets=workload.graph_target[test_idx],
            dense=workload.features[test_idx],
        ),
        len(test_idx),
    )
    return (
        prediction,
        timing,
        {"graph_dim": int(args.graph_dim), "graph_epochs": int(args.graph_epochs)},
    )


def verified_negative_pair(
    source: int,
    target: int,
    *,
    node_count: int,
    positive_set: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for offset in range(1, node_count):
        candidate = (source, (target + offset) % node_count)
        if candidate not in positive_set:
            return candidate
    for offset in range(1, node_count):
        candidate = ((source + offset) % node_count, target)
        if candidate not in positive_set:
            return candidate
    return None


def run_standalone_link_predictor(
    model_name: str,
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if (
        workload.graph_source is None
        or workload.graph_target is None
        or workload.graph_edges is None
        or workload.graph_node_features is None
    ):
        raise ValueError("workload has no graph topology")
    node_count = int(max(workload.graph_source.max(), workload.graph_target.max()) + 1)
    train_edges = sorted(
        {
            (int(workload.graph_source[index]), int(workload.graph_target[index]))
            for index in train_idx
        }
    )
    positives = [
        (int(workload.graph_source[index]), int(workload.graph_target[index]))
        for index in test_idx[: min(len(test_idx), 512)]
    ]
    positive_set = set(train_edges) | set(positives)
    eval_positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []
    for source, target in positives:
        candidate = verified_negative_pair(
            source,
            target,
            node_count=node_count,
            positive_set=positive_set,
        )
        if candidate is None:
            continue
        eval_positives.append((source, target))
        negatives.append(candidate)
    if not negatives:
        raise ValueError("no non-positive link candidates available")
    pairs = eval_positives + negatives
    labels = [1] * len(eval_positives) + [0] * len(negatives)
    query_ids = [source for source, _target in pairs]

    if model_name == "node2vec_link_predictor":
        from cartoboost import Node2VecLinkPredictor

        model = Node2VecLinkPredictor(
            dim=args.graph_dim,
            walk_length=16,
            walks_per_node=4,
            window_size=4,
            epochs=args.graph_epochs,
            negative_samples=3,
            seed=args.seed,
        )
        start = time.perf_counter()
        model.fit(node_count=node_count, edges=train_edges or workload.graph_edges)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.predict_scores(pairs)
        predict_seconds = time.perf_counter() - start
    elif model_name == "graphsage_link_predictor":
        from cartoboost import GraphSageLinkPredictor

        model = GraphSageLinkPredictor(
            input_dim=int(workload.graph_node_features.shape[1]),
            hidden_dims=(args.graph_dim,),
            epochs=args.graph_epochs,
            seed=args.seed,
        )
        start = time.perf_counter()
        model.fit(
            node_features=workload.graph_node_features, edges=train_edges or workload.graph_edges
        )
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.predict_scores(node_features=workload.graph_node_features, pairs=pairs)
        predict_seconds = time.perf_counter() - start
    elif model_name == "hetero_graphsage_link_predictor":
        from cartoboost import HeteroGraphSageLinkPredictor

        typed_train_edges = [
            (source, target, 0) for source, target in (train_edges or workload.graph_edges)
        ]
        model = HeteroGraphSageLinkPredictor(
            input_dim=int(workload.graph_node_features.shape[1]),
            relation_count=1,
            hidden_dims=(args.graph_dim,),
            epochs=args.graph_epochs,
            seed=args.seed,
        )
        start = time.perf_counter()
        model.fit(node_features=workload.graph_node_features, edges=typed_train_edges)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.predict_scores(node_features=workload.graph_node_features, pairs=pairs)
        predict_seconds = time.perf_counter() - start
    else:
        from cartoboost import HinSageLinkPredictor

        typed_train_edges = [
            (source, target, 0) for source, target in (train_edges or workload.graph_edges)
        ]
        model = HinSageLinkPredictor(
            input_dim=int(workload.graph_node_features.shape[1]),
            node_type_count=1,
            edge_type_triples=[(0, 0, 0)],
            hidden_dims=(args.graph_dim,),
            epochs=args.graph_epochs,
            seed=args.seed,
        )
        start = time.perf_counter()
        model.fit(
            node_features=workload.graph_node_features,
            node_types=[0] * int(workload.graph_node_features.shape[0]),
            edges=typed_train_edges,
        )
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.predict_scores(node_features=workload.graph_node_features, pairs=pairs)
        predict_seconds = time.perf_counter() - start

    from cartoboost.graph import link_prediction_report

    return {
        "status": "ok",
        "metrics": link_prediction_report(labels, scores.tolist(), query_ids=query_ids, k=10),
        "timing": {
            "train_seconds": float(train_seconds),
            "predict_seconds": float(predict_seconds),
            "predict_rows_per_second": float(len(pairs) / predict_seconds)
            if predict_seconds > 0.0
            else float("inf"),
        },
        "config": {"graph_dim": int(args.graph_dim), "graph_epochs": int(args.graph_epochs)},
    }


def run_xgboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    xgboost = optional_import("xgboost")
    if xgboost is None or not hasattr(xgboost, "XGBRegressor"):
        raise ImportError("xgboost is not installed")
    model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=args.n_threads or 0,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"tree_method": "hist"}


def run_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    lightgbm = optional_import("lightgbm")
    if lightgbm is None or not hasattr(lightgbm, "LGBMRegressor"):
        raise ImportError("lightgbm is not installed")
    model = lightgbm.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=2**args.max_depth,
        random_state=args.seed,
        n_jobs=args.n_threads or -1,
        verbose=-1,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"num_leaves": int(2**args.max_depth)}


def run_catboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    catboost = optional_import("catboost")
    if catboost is None or not hasattr(catboost, "CatBoostRegressor"):
        raise ImportError("catboost is not installed")
    model = catboost.CatBoostRegressor(
        loss_function="RMSE",
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.max_depth,
        random_seed=args.seed,
        thread_count=args.n_threads or -1,
        verbose=False,
        allow_writing_files=False,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"iterations": int(args.n_estimators), "depth": int(args.max_depth)}


def run_hist_gradient_boosting(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=args.n_estimators,
        learning_rate=args.learning_rate,
        max_leaf_nodes=2**args.max_depth,
        random_state=args.seed,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"max_leaf_nodes": int(2**args.max_depth)}


def run_random_forest(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=min_samples_leaf_value(args, default=2),
        random_state=args.seed,
        n_jobs=args.n_threads or -1,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"min_samples_leaf": min_samples_leaf_value(args, default=2)}


def run_extra_trees(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=min_samples_leaf_value(args, default=2),
        random_state=args.seed,
        n_jobs=args.n_threads or -1,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"min_samples_leaf": min_samples_leaf_value(args, default=2)}


def run_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    from sklearn.linear_model import Ridge

    alpha = float(getattr(args, "ridge_alpha", 1.0))
    model = Ridge(alpha=alpha)
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"alpha": alpha}


def candidate_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    candidate = argparse.Namespace(**vars(args))
    for key, value in config.items():
        setattr(candidate, key, value)
    return candidate


def cartoboost_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_leaf = min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF)
    return [
        {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": base_leaf,
        },
        {
            "n_estimators": max(8, int(round(args.n_estimators * 0.75))),
            "learning_rate": min(0.3, args.learning_rate * 1.25),
            "max_depth": args.max_depth,
            "min_samples_leaf": max(2, base_leaf // 2),
        },
        {
            "n_estimators": max(8, int(round(args.n_estimators * 1.25))),
            "learning_rate": max(0.01, args.learning_rate * 0.8),
            "max_depth": max(1, args.max_depth - 1),
            "min_samples_leaf": max(2, base_leaf * 2),
        },
    ]


def tree_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        },
        {
            "n_estimators": max(8, int(round(args.n_estimators * 0.75))),
            "learning_rate": min(0.3, args.learning_rate * 1.25),
            "max_depth": args.max_depth,
        },
        {
            "n_estimators": max(8, int(round(args.n_estimators * 1.25))),
            "learning_rate": max(0.01, args.learning_rate * 0.8),
            "max_depth": max(1, args.max_depth - 1),
        },
    ]


def forest_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_leaf = min_samples_leaf_value(args, default=2)
    return [
        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": base_leaf,
        },
        {
            "n_estimators": max(8, int(round(args.n_estimators * 0.75))),
            "max_depth": max(1, args.max_depth - 1),
            "min_samples_leaf": max(1, base_leaf // 2),
        },
        {
            "n_estimators": max(8, int(round(args.n_estimators * 1.25))),
            "max_depth": args.max_depth + 1,
            "min_samples_leaf": max(2, base_leaf * 2),
        },
    ]


def neural_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_leaf = min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF)
    return [
        {
            "neural_dim": args.neural_dim,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": base_leaf,
        },
        {
            "neural_dim": max(4, args.neural_dim // 2),
            "n_estimators": max(8, int(round(args.n_estimators * 0.75))),
            "learning_rate": min(0.3, args.learning_rate * 1.25),
            "max_depth": args.max_depth,
            "min_samples_leaf": max(2, base_leaf // 2),
        },
        {
            "neural_dim": max(4, args.neural_dim * 2),
            "n_estimators": max(8, int(round(args.n_estimators * 1.25))),
            "learning_rate": max(0.01, args.learning_rate * 0.8),
            "max_depth": max(1, args.max_depth - 1),
            "min_samples_leaf": max(2, base_leaf * 2),
        },
    ]


def graph_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_leaf = min_samples_leaf_value(args, default=CARTOBOOST_MIN_SAMPLES_LEAF)
    return [
        {
            "graph_dim": args.graph_dim,
            "graph_epochs": args.graph_epochs,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": base_leaf,
        },
        {
            "graph_dim": max(2, args.graph_dim // 2),
            "graph_epochs": max(1, args.graph_epochs // 2),
            "n_estimators": max(8, int(round(args.n_estimators * 0.75))),
            "learning_rate": min(0.3, args.learning_rate * 1.25),
            "max_depth": args.max_depth,
            "min_samples_leaf": max(2, base_leaf // 2),
        },
        {
            "graph_dim": max(2, args.graph_dim * 2),
            "graph_epochs": max(2, args.graph_epochs + 2),
            "n_estimators": max(8, int(round(args.n_estimators * 1.25))),
            "learning_rate": max(0.01, args.learning_rate * 0.8),
            "max_depth": max(1, args.max_depth - 1),
            "min_samples_leaf": max(2, base_leaf * 2),
        },
    ]


def link_predictor_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"graph_dim": args.graph_dim, "graph_epochs": args.graph_epochs},
        {
            "graph_dim": max(2, args.graph_dim // 2),
            "graph_epochs": max(1, args.graph_epochs // 2),
        },
        {
            "graph_dim": max(2, args.graph_dim * 2),
            "graph_epochs": max(2, args.graph_epochs + 2),
        },
    ]


def validation_candidate_grid(
    model_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if model_name == "cartoboost":
        return cartoboost_candidate_grid(args)[: args.validation_trials]
    if model_name in {"lightgbm", "xgboost", "catboost", "hist_gradient_boosting"}:
        return tree_candidate_grid(args)[: args.validation_trials]
    if model_name in {"random_forest", "extra_trees"}:
        return forest_candidate_grid(args)[: args.validation_trials]
    if model_name == "ridge":
        return [{"ridge_alpha": value} for value in [0.1, 1.0, 10.0]][: args.validation_trials]
    if model_name in {"cartoboost_neural", "neural_embedding_regressor"}:
        return neural_candidate_grid(args)[: args.validation_trials]
    if model_name in GRAPH_MODEL_FAMILIES:
        return graph_candidate_grid(args)[: args.validation_trials]
    if model_name in {
        "node2vec_regressor",
        "graphsage_regressor",
        "hetero_graphsage_regressor",
        "hinsage_regressor",
    }:
        return graph_candidate_grid(args)[: args.validation_trials]
    if model_name in {
        "node2vec_link_predictor",
        "graphsage_link_predictor",
        "hetero_graphsage_link_predictor",
        "hinsage_link_predictor",
    }:
        return link_predictor_candidate_grid(args)[: args.validation_trials]
    return []


def failed_validation_search_reason(validation_rows: list[dict[str, Any]]) -> str:
    reasons = sorted(
        {
            str(row.get("reason", "")).strip()
            for row in validation_rows
            if str(row.get("reason", "")).strip()
        }
    )
    if not reasons:
        return "all validation-search candidates failed"
    if len(reasons) == 1:
        return f"all validation-search candidates failed: {reasons[0]}"
    return "all validation-search candidates failed: " + "; ".join(reasons)


def model_trial_budget_summary(
    models: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    budgets: dict[str, dict[str, Any]] = {}
    tunable_trial_counts: dict[str, int] = {}
    fixed_models: list[str] = []
    for model_name in models:
        grid = validation_candidate_grid(model_name, args)
        requested_trials = len(grid) if args.selection_mode == "validation_search" else 0
        budgets[model_name] = {
            "selection_mode": args.selection_mode,
            "requested_trials": requested_trials,
            "tunable": bool(grid),
        }
        if grid:
            tunable_trial_counts[model_name] = requested_trials
        else:
            fixed_models.append(model_name)

    unique_trial_counts = sorted(set(tunable_trial_counts.values()))
    return {
        "models": budgets,
        "fixed_models": sorted(fixed_models),
        "tunable_model_trial_counts": tunable_trial_counts,
        "equal_tunable_trial_budget": len(unique_trial_counts) <= 1,
        "tunable_trial_count": unique_trial_counts[0] if len(unique_trial_counts) == 1 else None,
    }


def skipped_result(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def run_one_model_fixed(
    model_name: str,
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_x = workload.features[train_idx]
    test_x = workload.features[test_idx]
    train_y = workload.target[train_idx]
    test_y = workload.target[test_idx]
    if model_name == "mean":
        prediction, timing = run_mean(train_y, len(test_idx))
        config: dict[str, Any] = {}
    elif model_name == "cartoboost":
        prediction, timing, config = run_cartoboost(train_x, train_y, test_x, args)
    elif model_name == "cartoboost_neural":
        if workload.embedding_ids is None:
            raise ValueError("workload has no embedding ids")
        prediction, timing, config = run_neural(workload, train_idx, test_idx, args)
    elif model_name in GRAPH_MODEL_FAMILIES:
        if workload.graph_edges is None:
            raise ValueError("workload has no graph topology")
        prediction, timing, config = run_graph(
            workload,
            train_idx,
            test_idx,
            args,
            GRAPH_MODEL_FAMILIES[model_name],
        )
    elif model_name == "neural_embedding_regressor":
        if workload.embedding_ids is None:
            raise ValueError("workload has no embedding ids")
        prediction, timing, config = run_standalone_neural(workload, train_idx, test_idx, args)
    elif model_name == "node2vec_regressor":
        if workload.graph_edges is None:
            raise ValueError("workload has no graph topology")
        prediction, timing, config = run_standalone_node2vec_regressor(
            workload, train_idx, test_idx, args
        )
    elif model_name == "graphsage_regressor":
        if workload.graph_edges is None:
            raise ValueError("workload has no graph topology")
        prediction, timing, config = run_standalone_graphsage_regressor(
            workload, train_idx, test_idx, args
        )
    elif model_name in {"hetero_graphsage_regressor", "hinsage_regressor"}:
        if workload.graph_edges is None:
            raise ValueError("workload has no graph topology")
        prediction, timing, config = run_standalone_typed_graphsage_regressor(
            model_name, workload, train_idx, test_idx, args
        )
    elif model_name in {
        "node2vec_link_predictor",
        "graphsage_link_predictor",
        "hetero_graphsage_link_predictor",
        "hinsage_link_predictor",
    }:
        return run_standalone_link_predictor(model_name, workload, train_idx, test_idx, args)
    elif model_name == "xgboost":
        prediction, timing, config = run_xgboost(train_x, train_y, test_x, args)
    elif model_name == "lightgbm":
        prediction, timing, config = run_lightgbm(train_x, train_y, test_x, args)
    elif model_name == "catboost":
        prediction, timing, config = run_catboost(train_x, train_y, test_x, args)
    elif model_name == "hist_gradient_boosting":
        prediction, timing, config = run_hist_gradient_boosting(train_x, train_y, test_x, args)
    elif model_name == "random_forest":
        prediction, timing, config = run_random_forest(train_x, train_y, test_x, args)
    elif model_name == "extra_trees":
        prediction, timing, config = run_extra_trees(train_x, train_y, test_x, args)
    elif model_name == "ridge":
        prediction, timing, config = run_ridge(train_x, train_y, test_x, args)
    else:
        raise ValueError(f"unknown model {model_name!r}")

    return {
        "status": "ok",
        "metrics": metric_summary(test_y, prediction),
        "timing": timing,
        "config": config,
    }


def run_one_model(
    model_name: str,
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.selection_mode != "validation_search":
        try:
            return run_one_model_fixed(model_name, workload, train_idx, test_idx, args)
        except (ImportError, ValueError) as exc:
            return skipped_result(str(exc))

    grid = validation_candidate_grid(model_name, args)
    if not grid:
        try:
            result = run_one_model_fixed(model_name, workload, train_idx, test_idx, args)
        except (ImportError, ValueError) as exc:
            return skipped_result(str(exc))
        result.setdefault("selection", {"mode": "fixed_not_tunable"})
        return result

    inner_train, inner_validation = deterministic_inner_validation_split(
        len(train_idx),
        seed=args.seed,
    )
    validation_rows: list[dict[str, Any]] = []
    for trial_index, config in enumerate(grid, start=1):
        trial_args = candidate_args(args, config)
        try:
            trial = run_one_model_fixed(
                model_name,
                workload,
                train_idx[inner_train],
                train_idx[inner_validation],
                trial_args,
            )
        except (ImportError, ValueError) as exc:
            validation_rows.append(
                {
                    "trial": trial_index,
                    "config": config,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue
        if trial["status"] != "ok":
            validation_rows.append(
                {
                    "trial": trial_index,
                    "config": config,
                    "status": trial["status"],
                    "reason": trial.get("reason", ""),
                }
            )
            continue
        validation_rows.append(
            {
                "trial": trial_index,
                "config": config,
                "status": "ok",
                "validation_rmse": float(trial["metrics"]["rmse"]),
            }
        )

    successful = [row for row in validation_rows if row["status"] == "ok"]
    if not successful:
        return skipped_result(failed_validation_search_reason(validation_rows))

    best = min(successful, key=lambda row: (float(row["validation_rmse"]), int(row["trial"])))
    selected_args = candidate_args(args, dict(best["config"]))
    result = run_one_model_fixed(model_name, workload, train_idx, test_idx, selected_args)
    result["selection"] = {
        "mode": "validation_search",
        "inner_train_rows": int(len(inner_train)),
        "inner_validation_rows": int(len(inner_validation)),
        "metric": "rmse",
        "selected_trial": int(best["trial"]),
        "selected_config": best["config"],
        "selected_validation_rmse": float(best["validation_rmse"]),
        "validation_rows": validation_rows,
    }
    return result


def run_suite(workloads: list[Workload], args: argparse.Namespace) -> dict[str, Any]:
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    valid = {
        "mean",
        "cartoboost",
        "cartoboost_neural",
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
        "ridge",
        *GRAPH_MODEL_FAMILIES,
        *STANDALONE_MODELS,
    }
    unknown = sorted(set(models) - valid)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    validate_requested_benchmark_dependencies(models, requested_datasets(args))

    payload: dict[str, Any] = {
        "artifact_version": 1,
        "command_argv": list(sys.argv),
        "seed": int(args.seed),
        "datasets_requested": requested_datasets(args),
        "n_rows": int(args.n_rows),
        "train_frac": float(args.train_frac),
        "plots_written": not bool(args.no_plots),
        "models_requested": models,
        "selection_mode": args.selection_mode,
        "validation_trials": int(args.validation_trials),
        "trial_budget": model_trial_budget_summary(models, args),
        "resource_usage": resource_usage_snapshot(),
        "baseline_environment": baseline_environment_snapshot(),
        "benchmark_integrity": {
            "command_argv": list(sys.argv),
            "seed": int(args.seed),
            "datasets_requested": requested_datasets(args),
            "models_requested": models,
            "split_modes": list(split_definitions()),
            "hpo": (
                "inner_train_validation_search"
                if args.selection_mode == "validation_search"
                else "fixed_settings_no_hpo"
            ),
            "validation_trials": int(args.validation_trials)
            if args.selection_mode == "validation_search"
            else 0,
            "trial_budget": model_trial_budget_summary(models, args),
            "selection_policy": selection_policy(),
        },
        "split_definitions": split_definitions(),
        "model_config": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "neural_dim": int(args.neural_dim),
            "graph_dim": int(args.graph_dim),
            "graph_epochs": int(args.graph_epochs),
            "selection_mode": args.selection_mode,
            "validation_trials": int(args.validation_trials),
            "graph_model_families": dict(GRAPH_MODEL_FAMILIES),
            "cartoboost_qualitative_splitters": list(QUALITATIVE_CARTOBOOST_SPLITTERS),
        },
        "workloads": {},
    }
    for workload in workloads:
        workload_report: dict[str, Any] = {
            "display_name": workload.display_name,
            "description": workload.description,
            "source": workload.source,
            "fingerprint_sha256": workload_fingerprint(workload),
            "row_count": int(len(workload.target)),
            "feature_count": int(workload.features.shape[1]),
            "splits": {},
        }
        for split_name, (train_idx, test_idx) in split_workload(
            workload,
            train_frac=args.train_frac,
            seed=args.seed,
        ).items():
            split_report: dict[str, Any] = {
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_index_sha256": index_fingerprint(train_idx),
                "test_index_sha256": index_fingerprint(test_idx),
                "models": {},
            }
            for model_name in models:
                split_report["models"][model_name] = run_one_model(
                    model_name,
                    workload,
                    train_idx,
                    test_idx,
                    args,
                )
            workload_report["splits"][split_name] = split_report
        payload["workloads"][workload.name] = workload_report
    payload["external_baseline_comparison"] = external_baseline_comparison(payload)
    payload["model_status_summary"] = model_status_summary(payload)
    payload["comparability_audit"] = comparability_audit(payload)
    return payload


def external_baseline_comparison(payload: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for workload_name, workload in payload["workloads"].items():
        for split_name, split in workload["splits"].items():
            cartoboost = split["models"].get("cartoboost")
            if cartoboost is None or cartoboost["status"] != "ok":
                continue
            cartoboost_metrics = cartoboost.get("metrics", {})
            if "rmse" not in cartoboost_metrics:
                continue
            baselines = []
            for model_name, result in split["models"].items():
                if model_name not in EXTERNAL_REGRESSION_BASELINES:
                    continue
                if result["status"] != "ok":
                    continue
                metrics = result.get("metrics", {})
                if "rmse" not in metrics:
                    continue
                baselines.append((float(metrics["rmse"]), model_name, metrics))
            if not baselines:
                continue
            baseline_rmse, baseline_name, baseline_metrics = min(
                baselines, key=lambda item: item[0]
            )
            cartoboost_rmse = float(cartoboost_metrics["rmse"])
            comparisons.append(
                {
                    "workload": workload_name,
                    "split": split_name,
                    "cartoboost_model": "cartoboost",
                    "cartoboost_rmse": cartoboost_rmse,
                    "cartoboost_wape": float(cartoboost_metrics.get("wape", float("nan"))),
                    "cartoboost_r2": float(cartoboost_metrics["r2"]),
                    "best_external_baseline": baseline_name,
                    "best_external_rmse": baseline_rmse,
                    "best_external_wape": float(baseline_metrics.get("wape", float("nan"))),
                    "best_external_r2": float(baseline_metrics["r2"]),
                    "rmse_delta_vs_external": cartoboost_rmse - baseline_rmse,
                    "r2_delta_vs_external": float(cartoboost_metrics["r2"])
                    - float(baseline_metrics["r2"]),
                    "status": "cartoboost_lower_rmse"
                    if cartoboost_rmse < baseline_rmse
                    else "external_lower_or_tied_rmse",
                }
            )
    return comparisons


def model_status_summary(payload: dict[str, Any]) -> dict[str, Any]:
    requested = list(payload.get("models_requested", []))
    summary: dict[str, dict[str, Any]] = {
        model_name: {"ok": 0, "skipped": 0, "skip_reasons": []} for model_name in requested
    }
    for workload in payload["workloads"].values():
        for split in workload["splits"].values():
            for model_name, result in split["models"].items():
                model_summary = summary.setdefault(
                    model_name,
                    {"ok": 0, "skipped": 0, "skip_reasons": []},
                )
                status = str(result.get("status", ""))
                if status == "ok":
                    model_summary["ok"] += 1
                elif status == "skipped":
                    model_summary["skipped"] += 1
                    reason = str(result.get("reason", "")).strip()
                    if reason and reason not in model_summary["skip_reasons"]:
                        model_summary["skip_reasons"].append(reason)

    completed_external = sorted(
        model_name
        for model_name, model_summary in summary.items()
        if model_name in EXTERNAL_REGRESSION_BASELINES and int(model_summary["ok"]) > 0
    )
    skipped_requested_external = sorted(
        model_name
        for model_name, model_summary in summary.items()
        if model_name in EXTERNAL_REGRESSION_BASELINES
        and int(model_summary["ok"]) == 0
        and int(model_summary["skipped"]) > 0
    )
    completed_cartoboost_family = sorted(
        model_name
        for model_name, model_summary in summary.items()
        if model_name in CARTOBOOST_FAMILY_MODELS and int(model_summary["ok"]) > 0
    )
    return {
        "models": summary,
        "completed_external_baselines": completed_external,
        "skipped_requested_external_baselines": skipped_requested_external,
        "completed_cartoboost_family_models": completed_cartoboost_family,
    }


def comparability_audit(payload: dict[str, Any]) -> dict[str, Any]:
    trial_budget = payload.get("trial_budget", {})
    status_summary = model_status_summary(payload)
    comparisons = payload.get("external_baseline_comparison", [])
    return {
        "same_outer_splits": True,
        "same_metrics": ["mae", "rmse", "r2", "wape"],
        "primary_metric": "rmse",
        "selection_metric": "rmse",
        "selection_uses_outer_test_labels": False,
        "equal_tunable_trial_budget": bool(trial_budget.get("equal_tunable_trial_budget", True)),
        "tunable_trial_count": trial_budget.get("tunable_trial_count"),
        "completed_external_baselines": status_summary["completed_external_baselines"],
        "skipped_requested_external_baselines": status_summary[
            "skipped_requested_external_baselines"
        ],
        "completed_cartoboost_family_models": status_summary["completed_cartoboost_family_models"],
        "cartoboost_external_comparison_count": len(comparisons),
        "status_summary": status_summary["models"],
    }


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def repeated_external_comparison_summary(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for payload in payloads:
        seed = int(payload["seed"])
        for row in payload.get("external_baseline_comparison", []):
            key = (str(row["workload"]), str(row["split"]))
            grouped.setdefault(key, []).append({**row, "seed": seed})

    summary = []
    for (workload, split), rows in sorted(grouped.items()):
        rmse_deltas = [float(row["rmse_delta_vs_external"]) for row in rows]
        wape_deltas = [
            float(row["cartoboost_wape"]) - float(row["best_external_wape"]) for row in rows
        ]
        r2_deltas = [float(row["r2_delta_vs_external"]) for row in rows]
        rmse_mean, rmse_low, rmse_high = mean_ci(rmse_deltas)
        wape_mean, wape_low, wape_high = mean_ci(wape_deltas)
        r2_mean, r2_low, r2_high = mean_ci(r2_deltas)
        baseline_counts: dict[str, int] = {}
        for row in rows:
            name = str(row["best_external_baseline"])
            baseline_counts[name] = baseline_counts.get(name, 0) + 1
        if rmse_high < 0.0:
            result = "cartoboost_lower_rmse"
        elif rmse_low > 0.0:
            result = "external_lower_rmse"
        else:
            result = "mixed_interval"
        summary.append(
            {
                "workload": workload,
                "split": split,
                "runs": len(rows),
                "seeds": [int(row["seed"]) for row in rows],
                "best_external_baseline_counts": dict(sorted(baseline_counts.items())),
                "rmse_delta_mean": rmse_mean,
                "rmse_delta_ci95_low": rmse_low,
                "rmse_delta_ci95_high": rmse_high,
                "wape_delta_mean": wape_mean,
                "wape_delta_ci95_low": wape_low,
                "wape_delta_ci95_high": wape_high,
                "r2_delta_mean": r2_mean,
                "r2_delta_ci95_low": r2_low,
                "r2_delta_ci95_high": r2_high,
                "result": result,
            }
        )
    return summary


def ok_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload_name, workload in payload["workloads"].items():
        for split_name, split in workload["splits"].items():
            for model_name, result in split["models"].items():
                if result["status"] != "ok":
                    continue
                rows.append(
                    {
                        "workload": workload_name,
                        "split": split_name,
                        "model": model_name,
                        **result["metrics"],
                        **result["timing"],
                    }
                )
    return rows


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Model Benchmark Suite",
        "",
        (
            "This generated report compares the primary CartoBoost regressor against "
            "optional external baselines on deterministic public tabular workloads "
            "and embedded graph diagnostics."
        ),
        "",
        "## Command",
        "",
        (
            "`PYTHONPATH=python uv run --group dev --group bench python "
            "scripts/run_model_benchmark_suite.py ...`"
        ),
        "",
        "Command arguments:",
        "",
        f"`{' '.join(payload.get('command_argv', []))}`",
        "",
        "## Configuration",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Datasets requested: `{', '.join(payload.get('datasets_requested', []))}`",
        f"- Rows per workload: `{payload['n_rows']}`",
        f"- Train fraction: `{payload['train_frac']}`",
        f"- Selection mode: `{payload.get('selection_mode', 'fixed')}`",
        f"- Validation trials per tunable model: `{payload.get('validation_trials', 0)}`",
        f"- Models requested: `{', '.join(payload['models_requested'])}`",
        "",
    ]
    resources = payload.get("resource_usage", {})
    if resources:
        lines.extend(
            [
                "## Resource Usage",
                "",
                "| Field | Value |",
                "| --- | --- |",
            ]
        )
        for key, value in resources.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")
    baseline_environment = payload.get("baseline_environment", {})
    if baseline_environment:
        lines.extend(
            [
                "## Baseline Dependency Status",
                "",
                (
                    "| Key | Package | Import | Version | Module importable | "
                    "Required class | Required class available |"
                ),
                "| --- | --- | --- | --- | ---: | --- | ---: |",
            ]
        )
        for name, status in sorted(baseline_environment.items()):
            lines.append(
                f"| {name} | {status.get('package', '')} | {status.get('import_name', '')} | "
                f"`{status.get('version')}` | {status.get('module_importable')} | "
                f"{status.get('required_class', '')} | "
                f"{status.get('required_class_available', '')} |"
            )
        lines.append("")
    artifacts = payload.get("output_artifacts", {})
    if artifacts:
        lines.extend(
            [
                "## Output Artifacts",
                "",
                "| Artifact | Size bytes |",
                "| --- | ---: |",
            ]
        )
        for name, metadata in sorted(artifacts.items()):
            lines.append(f"| `{name}` | {metadata['size_bytes']} |")
        lines.append("")
    comparability = payload.get("comparability_audit", {})
    if comparability:
        lines.extend(
            [
                "## Comparability Audit",
                "",
                "| Check | Value |",
                "| --- | --- |",
                (
                    "| Same outer splits for requested models | "
                    f"{comparability['same_outer_splits']} |"
                ),
                f"| Primary metric | `{comparability['primary_metric']}` |",
                f"| Selection metric | `{comparability['selection_metric']}` |",
                (
                    "| Selection uses outer test labels | "
                    f"{comparability['selection_uses_outer_test_labels']} |"
                ),
                (f"| Equal tunable trial budget | {comparability['equal_tunable_trial_budget']} |"),
                f"| Tunable trial count | `{comparability.get('tunable_trial_count')}` |",
                (
                    "| Completed external baselines | "
                    f"`{', '.join(comparability['completed_external_baselines'])}` |"
                ),
                (
                    "| Skipped requested external baselines | "
                    f"`{', '.join(comparability['skipped_requested_external_baselines'])}` |"
                ),
                (
                    "| Completed CartoBoost-family rows | "
                    f"`{', '.join(comparability['completed_cartoboost_family_models'])}` |"
                ),
                (
                    "| CartoBoost/external comparison rows | "
                    f"{comparability['cartoboost_external_comparison_count']} |"
                ),
                "",
            ]
        )
    lines.extend(["## Selection and Leakage Policy", ""])
    selection = payload.get("benchmark_integrity", {}).get("selection_policy", {})
    if selection:
        for key, value in selection.items():
            label = key.replace("_", " ")
            lines.append(f"- {label}: {value}")
        lines.append("")
    split_metadata = payload.get("split_definitions", {})
    if split_metadata:
        lines.extend(
            [
                "## Split Definitions",
                "",
                "| Split | Kind | Train fraction | Purpose |",
                "| --- | --- | --- | --- |",
            ]
        )
        for split_name, definition in split_metadata.items():
            lines.append(
                f"| {split_name} | {definition['kind']} | "
                f"{definition['train_fraction']} | {definition['purpose']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Dataset Sources",
            "",
            "| Workload | Source | Rows | Features | SHA-256 fingerprint |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for workload_name, workload in payload["workloads"].items():
        lines.append(
            f"| {workload_name} | {workload['source']} | {workload['row_count']} | "
            f"{workload['feature_count']} | `{workload['fingerprint_sha256']}` |"
        )
    lines.append("")
    lines.extend(
        [
            "## Results",
            "",
        ]
    )
    comparisons: list[dict[str, Any]] = []
    if comparisons:
        lines.extend(
            [
                "## CartoBoost vs External Baselines",
                "",
                (
                    "For each regression split, this table compares the single primary "
                    "`cartoboost` row with the lowest-RMSE external baseline that finished "
                    "under the same data split and global benchmark settings."
                ),
                "",
                (
                    "| Workload | Split | CartoBoost RMSE | CartoBoost WAPE | "
                    "Best external baseline | External RMSE | External WAPE | "
                    "RMSE delta | R2 delta | Result |"
                ),
                "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['workload']} | {row['split']} | {row['cartoboost_rmse']:.4f} | "
                f"{row['cartoboost_wape']:.4f} | {row['best_external_baseline']} | "
                f"{row['best_external_rmse']:.4f} | {row['best_external_wape']:.4f} | "
                f"{row['rmse_delta_vs_external']:.4f} | "
                f"{row['r2_delta_vs_external']:.4f} | {row['status']} |"
            )
        lines.extend(
            [
                "",
                "### Interpretation Notes",
                "",
                (
                    "- Dense public or synthetic workloads are baseline sanity checks for "
                    "ordinary tabular regression behavior without graph or neural inputs."
                ),
                (
                    "- Neural workloads, when requested, show the difference between repeated-ID "
                    "and cold-ID claims. Neural and graph rows are diagnostics and are not used "
                    "as substitutes for the primary `cartoboost` comparison row."
                ),
                (
                    "- The graph workload separates two surfaces. Augmented CartoBoost uses "
                    "graph features as extra columns for the booster, while standalone "
                    "GraphSAGE-style regressors and link predictors can score graph tasks "
                    "without a boosted wrapper. The link-predictor rows report AUC/AP "
                    "because they are ranking candidate source-target edges, not predicting "
                    "the regression target."
                ),
                (
                    "- External baseline rows use the same train/test split and global "
                    "benchmark settings; no test labels are used for model selection."
                ),
                "",
            ]
        )
    repeated: list[dict[str, Any]] = []
    if repeated:
        lines.extend(
            [
                "## Repeated External Baseline Comparison",
                "",
                (
                    "Repeated rows use the same model roster, validation-search budget, "
                    "and split policy with different deterministic seeds. Negative RMSE "
                    "and WAPE deltas favor CartoBoost; positive R2 deltas favor CartoBoost."
                ),
                "",
                (
                    "| Workload | Split | Seeds | Best external baseline counts | "
                    "RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | "
                    "R2 delta mean | R2 delta 95% CI | Result |"
                ),
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in repeated:
            counts = ", ".join(
                f"{name}: {count}" for name, count in row["best_external_baseline_counts"].items()
            )
            seeds = ", ".join(str(seed) for seed in row["seeds"])
            lines.append(
                f"| {row['workload']} | {row['split']} | {seeds} | {counts} | "
                f"{row['rmse_delta_mean']:.6f} | "
                f"{row['rmse_delta_ci95_low']:.6f} to {row['rmse_delta_ci95_high']:.6f} | "
                f"{row['wape_delta_mean']:.6f} | {row['r2_delta_mean']:.6f} | "
                f"{row['r2_delta_ci95_low']:.6f} to {row['r2_delta_ci95_high']:.6f} | "
                f"{row['result']} |"
            )
        lines.append("")
    selection_rows = []
    for workload_name, workload in payload["workloads"].items():
        for split_name, split in workload["splits"].items():
            for model_name, result in split["models"].items():
                selection = result.get("selection", {})
                if selection.get("mode") != "validation_search":
                    continue
                if result.get("status") != "ok" or "selected_config" not in selection:
                    continue
                selection_rows.append((workload_name, split_name, model_name, selection))
    if selection_rows:
        lines.extend(
            [
                "## Validation Search Selections",
                "",
                (
                    "The table records the inner-validation winner for each tunable model. "
                    "Final holdout metrics above are computed only after retraining the "
                    "selected configuration on the full outer training split."
                ),
                "",
                (
                    "| Workload | Split | Model | Selected trial | Validation RMSE | "
                    "Inner train rows | Inner validation rows | Selected config |"
                ),
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for workload_name, split_name, model_name, selection in selection_rows:
            config = json.dumps(selection["selected_config"], sort_keys=True)
            lines.append(
                f"| {workload_name} | {split_name} | {model_name} | "
                f"{selection['selected_trial']} | "
                f"{selection['selected_validation_rmse']:.6f} | "
                f"{selection['inner_train_rows']} | {selection['inner_validation_rows']} | "
                f"`{config}` |"
            )
        lines.append("")
    lines.extend(["## Problem Metrics", ""])
    for workload in payload["workloads"].values():
        lines.extend([f"### {workload['display_name']}", "", workload["description"], ""])
        for split_name, split in workload["splits"].items():
            lines.extend(
                [
                    f"#### {split_name}",
                    "",
                    f"Train rows: `{split['train_rows']}`; test rows: `{split['test_rows']}`.",
                    (
                        f"Train index SHA-256: `{split['train_index_sha256']}`; "
                        f"test index SHA-256: `{split['test_index_sha256']}`."
                    ),
                    "",
                    "| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for model_name, result in split["models"].items():
                if result["status"] == "ok":
                    metrics = result["metrics"]
                    timing = result["timing"]
                    if "mae" in metrics:
                        lines.append(
                            f"| {model_name} | ok | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
                            f"{metrics['r2']:.4f} | {metrics['wape']:.4f} | "
                            f"{timing['train_seconds']:.4f} | "
                            f"{timing['predict_rows_per_second']:.0f} |"
                        )
                    elif "auc" in metrics:
                        lines.append(
                            f"| {model_name} | ok link: AUC {metrics['auc']:.4f}, "
                            f"AP {metrics['average_precision']:.4f} |  |  |  | "
                            f" | {timing['train_seconds']:.4f} | "
                            f"{timing['predict_rows_per_second']:.0f} |"
                        )
                    else:
                        lines.append(
                            f"| {model_name} | ok |  |  |  |  | {timing['train_seconds']:.4f} | "
                            f"{timing['predict_rows_per_second']:.0f} |"
                        )
                elif result["status"] == "skipped":
                    lines.append(
                        f"| {model_name} | skipped: {result.get('reason', '')} |  |  |  |  |  |  |"
                    )
                else:
                    raise ValueError(
                        f"unexpected non-ok benchmark result for {model_name}: {result}"
                    )
            lines.append("")
    if payload.get("plots_written", True):
        lines.extend(
            [
                "## Plots",
                "",
                "![MAE by workload and split](mae_by_model.png)",
                "",
                "![Training time by workload and split](train_time_by_model.png)",
                "",
                (
                    "![Prediction throughput by workload and split]"
                    "(prediction_throughput_by_model.png)"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Dense workloads check numeric behavior without ID or graph augmentation.",
            (
                "- Neural workloads, when requested, include repeated IDs and a group holdout "
                "split, so `cartoboost_neural` should be read as an embedding augmentation "
                "check rather than a replacement for external neural networks."
            ),
            (
                "- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and "
                "HinSAGE feature augmentation from train topology before fitting CartoBoost "
                "on augmented source-target rows."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    rows = [row for row in rows if metric in row and math.isfinite(float(row[metric]))]
    if not rows:
        return
    labels = [f"{row['workload']}\n{row['split']}\n{row['model']}" for row in rows]
    values = [float(row[metric]) for row in rows]
    width = max(10, min(28, len(rows) * 0.55))
    fig, ax = plt.subplots(figsize=(width, 7))
    positions = np.arange(len(rows))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(rows)))
    ax.bar(positions, values, color=colors)
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_jsonl_results(payload, args.output_dir / "results.jsonl")
    write_markdown(payload, args.output_dir / "results.md")
    if not args.no_plots:
        rows = ok_rows(payload)
        plot_metric(
            rows, "mae", "MAE by model, workload, and split", args.output_dir / "mae_by_model.png"
        )
        plot_metric(
            rows,
            "train_seconds",
            "Training time by model, workload, and split",
            args.output_dir / "train_time_by_model.png",
        )
        plot_metric(
            rows,
            "predict_rows_per_second",
            "Prediction throughput by model, workload, and split",
            args.output_dir / "prediction_throughput_by_model.png",
        )
    for _ in range(5):
        manifest = output_artifact_manifest(args.output_dir)
        if manifest == payload.get("output_artifacts"):
            break
        payload["output_artifacts"] = manifest
        (args.output_dir / "results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        write_markdown(payload, args.output_dir / "results.md")


def output_artifact_manifest(output_dir: Path) -> dict[str, dict[str, int]]:
    artifacts: dict[str, dict[str, int]] = {}
    for name in [
        "results.json",
        "results.jsonl",
        "results.md",
        "mae_by_model.png",
        "train_time_by_model.png",
        "prediction_throughput_by_model.png",
    ]:
        path = output_dir / name
        if path.exists():
            artifacts[name] = {"size_bytes": int(path.stat().st_size)}
    return artifacts


def workload_track(workload_name: str) -> str:
    if workload_name in {"diabetes", "california_housing"}:
        return "tabular"
    if workload_name == "karate":
        return "graph"
    return "diagnostic"


def write_jsonl_results(payload: dict[str, Any], output: Path) -> None:
    rows = []
    payloads = [payload, *payload.get("repeat_run_payloads", [])]
    for run_payload in payloads:
        run_seed = int(run_payload["seed"])
        for workload_name, workload in run_payload["workloads"].items():
            for split_name, split in workload["splits"].items():
                for model_name, result in split["models"].items():
                    if result.get("status") != "ok":
                        continue
                    for metric, value in result.get("metrics", {}).items():
                        rows.append(
                            {
                                "track": workload_track(workload_name),
                                "task_id": workload_name,
                                "split_id": split_name,
                                "model_family": model_name,
                                "metric": metric,
                                "value": float(value),
                                "seed": run_seed,
                            }
                        )
                    for metric in [
                        "train_seconds",
                        "predict_seconds",
                        "predict_rows_per_second",
                    ]:
                        if metric in result.get("timing", {}):
                            rows.append(
                                {
                                    "track": workload_track(workload_name),
                                    "task_id": workload_name,
                                    "split_id": split_name,
                                    "model_family": model_name,
                                    "metric": metric,
                                    "value": float(result["timing"][metric]),
                                    "seed": run_seed,
                                }
                            )
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    args = parse_args()
    workloads = build_workloads(args)
    payload = run_suite(workloads, args)
    seeds = repeat_seeds(args)
    if seeds:
        repeat_payloads = []
        for seed in seeds:
            if seed == args.seed:
                repeat_payloads.append(payload)
                continue
            repeat_args = argparse.Namespace(**vars(args))
            repeat_args.seed = seed
            repeat_payloads.append(run_suite(build_workloads(repeat_args), repeat_args))
        payload["repeat_seeds"] = seeds
        payload["repeat_run_payloads"] = [
            run_payload for run_payload in repeat_payloads if int(run_payload["seed"]) != args.seed
        ]
        payload["repeated_external_baseline_comparison"] = repeated_external_comparison_summary(
            repeat_payloads
        )
    write_outputs(payload, args)
    print(f"Benchmark results written to {args.output_dir}")


if __name__ == "__main__":
    main()
