#!/usr/bin/env python3
"""Run NYC TLC taxi quality and speed benchmarks for CartoBoost and GBDT baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

DEFAULT_CACHE_DIR = ROOT / "data" / "nyc_taxi"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks"
TLC_TRIP_RECORD_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
TLC_PARQUET_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TAXI_ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
PLOT_SCATTER_MAX_POINTS = 50_000
GRAPH_AUGMENTATION_RELATIVE_VALIDATION_MARGIN = 0.01
NEURAL_AUGMENTATION_RELATIVE_VALIDATION_MARGIN = 0.01

ROW_FEATURES = [
    "trip_distance",
    "log_trip_distance",
    "passenger_count",
    "hour",
    "dayofweek",
    "PULocationID",
    "DOLocationID",
]
DEMAND_FEATURES = ["PULocationID", "hour", "dayofweek"]
ZONE_FEATURES = {"PULocationID", "DOLocationID"}
BASIC_BOROUGH_CODES = {
    "Bronx": 1,
    "Brooklyn": 2,
    "EWR": 3,
    "Manhattan": 4,
    "Queens": 5,
    "Staten Island": 6,
    "Unknown": 7,
}
SERVICE_ZONE_CODES = {
    "Airports": 1,
    "Boro Zone": 2,
    "EWR": 3,
    "Yellow Zone": 4,
    "N/A": 5,
    "Unknown": 6,
}
GRAPH_MODEL_FAMILIES = {
    "cartoboost_graph_node2vec": "node2vec",
    "cartoboost_graph_graphsage": "graphsage",
    "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
    "cartoboost_graph_hinsage": "hinsage",
}
CARTOBOOST_MODEL_NAMES = {
    "cartoboost",
    "cartoboost_reference",
    "cartoboost_neural",
    "cartoboost_graph",
    *GRAPH_MODEL_FAMILIES,
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
    "lightgbm": ("lightgbm", "LGBMRegressor"),
    "xgboost": ("xgboost", "XGBRegressor"),
    "catboost": ("catboost", "CatBoostRegressor"),
}
SKLEARN_MODELS = {
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "ridge",
}


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    display_name: str
    description: str
    features: np.ndarray
    target: np.ndarray
    pickup_zones: np.ndarray
    feature_names: list[str]
    sparse_sets: dict[str, list[list[int]]]
    zone_adjacency: dict[int, list[int]] | None = None
    zone_centroids: dict[int, tuple[float, float]] | None = None
    # Row-level tasks retain the pickup timestamp so temporal holdouts can be
    # constructed without deriving a split from a feature that may be supplied
    # to the model.  Aggregated demand tasks intentionally leave this unset.
    timestamps: np.ndarray | None = None
    timestamp_column: str | None = None


@dataclass(frozen=True)
class ZoneContext:
    borough_code: int
    service_zone_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1", help="Comma-separated month numbers, e.g. 1,2,3")
    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names to run, for example pickup_demand.",
    )
    parser.add_argument(
        "--split-modes",
        default="",
        help=(
            "Optional comma-separated outer splits. Defaults to every split "
            "applicable to each task; valid values are random, spatial_holdout, and out_of_time."
        ),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--models",
        default=(
            "cartoboost,cartoboost_reference,cartoboost_neural,"
            "cartoboost_graph_node2vec,cartoboost_graph_graphsage,"
            "cartoboost_graph_hetero_graphsage,cartoboost_graph_hinsage,"
            "lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,"
            "extra_trees,ridge,mean"
        ),
        help=(
            "Comma-separated models from: cartoboost, cartoboost_reference, "
            "cartoboost_neural, cartoboost_graph, cartoboost_graph_node2vec, "
            "cartoboost_graph_graphsage, cartoboost_graph_hetero_graphsage, "
            "cartoboost_graph_hinsage, lightgbm, xgboost, catboost, "
            "hist_gradient_boosting, random_forest, extra_trees, ridge, mean"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--cartoboost-n-estimators",
        type=int,
        default=100,
        help=(
            "Estimator count for the CartoBoost benchmark candidate. Baselines use "
            "--n-estimators; cartoboost_reference uses the baseline count."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--cartoboost-max-depth",
        type=int,
        default=5,
        help=(
            "Max depth for the CartoBoost benchmark candidate. Baselines use --max-depth; "
            "cartoboost_reference uses the baseline depth."
        ),
    )
    parser.add_argument(
        "--cartoboost-splitters",
        default="axis_histogram:512,diagonal_2d,gaussian_2d,periodic:24,periodic:7,sparse_set",
        help=(
            "Comma-separated CartoBoost splitters for candidate and reference, "
            "for example axis_histogram:512 or axis."
        ),
    )
    parser.add_argument(
        "--cartoboost-min-samples-leaf",
        type=int,
        default=20,
        help="Minimum leaf row count for CartoBoost candidate and reference models.",
    )
    parser.add_argument(
        "--cartoboost-constant-l2",
        type=float,
        default=0.0,
        help="L2 regularization for CartoBoost constant leaf values.",
    )
    parser.add_argument(
        "--cartoboost-leaf-predictor",
        default="constant",
        choices=["constant", "linear"],
        help="Leaf predictor for CartoBoost candidate and reference models.",
    )
    parser.add_argument(
        "--cartoboost-init",
        default="constant",
        choices=["constant", "linear"],
        help="Initial CartoBoost model before residual tree boosting.",
    )
    parser.add_argument(
        "--cartoboost-calibration",
        default="none",
        choices=["none", "affine"],
        help="Train-only post-fit calibration for CartoBoost predictions.",
    )
    parser.add_argument(
        "--xgboost-tree-method",
        default="hist",
        choices=["auto", "exact", "approx", "hist"],
        help="XGBoost tree_method for cross-comparable exact/exact or hist/hist runs.",
    )
    parser.add_argument(
        "--xgboost-max-bin",
        type=int,
        default=256,
        help="XGBoost max_bin for hist/approx tree methods.",
    )
    parser.add_argument("--xgboost-subsample", type=float, default=1.0)
    parser.add_argument("--xgboost-colsample-bytree", type=float, default=1.0)
    parser.add_argument(
        "--neural-dim",
        type=int,
        default=12,
        help="Embedding dimension for the CartoBoost neural benchmark branch.",
    )
    parser.add_argument(
        "--graph-dim",
        type=int,
        default=8,
        help="Graph embedding dimension for the CartoBoost graph benchmark branch.",
    )
    parser.add_argument(
        "--graph-epochs",
        type=int,
        default=8,
        help="Graph encoder training epochs for the CartoBoost graph benchmark branch.",
    )
    parser.add_argument(
        "--graph-family",
        choices=["node2vec", "graphsage", "hetero_graphsage", "hinsage"],
        default="graphsage",
        help="Graph encoder family for cartoboost_graph.",
    )
    parser.add_argument(
        "--zone-treatment",
        default="target_mean",
        choices=["raw", "target_mean"],
        help=(
            "Comparable handling for NYC taxi zone IDs. 'target_mean' appends "
            "train-only smoothed zone target-mean features to every model, including XGBoost."
        ),
    )
    parser.add_argument(
        "--zone-target-smoothing",
        type=float,
        default=20.0,
        help="Pseudo-count for train-only smoothed zone target-mean features.",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=1,
        help="Number of model rows to train concurrently within each task/split.",
    )
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write JSON/JSONL/Markdown only and skip generated PNG plot artifacts.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a tiny deterministic in-memory fixture instead of reading TLC files.",
    )
    return parser.parse_args()


def parse_months(value: str) -> list[int]:
    months = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError("--months must contain month numbers between 1 and 12")
    return months


def parse_splitters(value: str) -> list[str]:
    splitters = [part.strip() for part in value.split(",") if part.strip()]
    if not splitters:
        raise ValueError("splitter list must not be empty")
    return splitters


def parse_split_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    valid = {"random", "spatial_holdout", "out_of_time"}
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"unknown split modes: {', '.join(unknown)}")
    return modes


def requested_model_names(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def benchmark_needs_zone_centroids(args: argparse.Namespace) -> bool:
    models = requested_model_names(args.models)
    if not models.intersection(CARTOBOOST_MODEL_NAMES):
        return False
    if args.zone_treatment == "raw":
        return True
    splitters = parse_splitters(args.cartoboost_splitters)
    return any(splitter in {"diagonal_2d", "gaussian_2d"} for splitter in splitters)


def splitters_need_sparse_sets(splitters: list[str]) -> bool:
    return any("sparse" in splitter for splitter in splitters)


def splitters_use_dense_id_sets(splitters: list[str]) -> bool:
    return any(splitter == "sparse_set" for splitter in splitters)


def month_url(taxi_type: str, year: int, month: int) -> str:
    return f"{TLC_PARQUET_BASE}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def month_path(cache_dir: Path, taxi_type: str, year: int, month: int) -> Path:
    return cache_dir / f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def ensure_parquet_files(
    *,
    taxi_type: str,
    year: int,
    months: list[int],
    cache_dir: Path,
    no_download: bool,
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for month in months:
        path = month_path(cache_dir, taxi_type, year, month)
        if not path.exists():
            if no_download:
                raise FileNotFoundError(
                    f"{path} is missing and --no-download was passed. "
                    f"Download it from {month_url(taxi_type, year, month)}."
                )
            urllib.request.urlretrieve(month_url(taxi_type, year, month), path)
        paths.append(path)
    return paths


def ensure_zone_lookup(*, cache_dir: Path, no_download: bool) -> dict[int, ZoneContext]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "taxi_zone_lookup.csv"
    if not path.exists():
        if no_download:
            raise FileNotFoundError(
                f"{path} is missing and --no-download was passed. "
                f"Download it from {TAXI_ZONE_LOOKUP_URL}."
            )
        urllib.request.urlretrieve(TAXI_ZONE_LOOKUP_URL, path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        lookup: dict[int, ZoneContext] = {}
        for row in reader:
            location_id = int(row["LocationID"])
            borough = row.get("Borough", "Unknown") or "Unknown"
            service_zone = row.get("service_zone", "Unknown") or "Unknown"
            lookup[location_id] = ZoneContext(
                borough_code=BASIC_BOROUGH_CODES.get(borough, BASIC_BOROUGH_CODES["Unknown"]),
                service_zone_code=SERVICE_ZONE_CODES.get(
                    service_zone,
                    SERVICE_ZONE_CODES["Unknown"],
                ),
            )
    return lookup


def ensure_zone_adjacency(*, cache_dir: Path, no_download: bool) -> dict[int, list[int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "taxi_zones.zip"
    if not path.exists():
        if no_download:
            return {}
        urllib.request.urlretrieve(TAXI_ZONES_ZIP_URL, path)
    try:
        return taxi_zone_adjacency_from_zip(path)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return {}


def ensure_zone_centroids(*, cache_dir: Path, no_download: bool) -> dict[int, tuple[float, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "taxi_zones.zip"
    if not path.exists():
        if no_download:
            raise FileNotFoundError(
                f"taxi zone geometry archive is required for real geo benchmarks: {path}"
            )
        urllib.request.urlretrieve(TAXI_ZONES_ZIP_URL, path)
    centroids = taxi_zone_centroids_from_zip(path)
    if not centroids:
        raise ValueError(f"taxi zone geometry archive did not contain usable centroids: {path}")
    return centroids


def taxi_zone_adjacency_from_zip(path: Path) -> dict[int, list[int]]:
    with zipfile.ZipFile(path) as archive:
        shp_name = next(name for name in archive.namelist() if name.endswith(".shp"))
        dbf_name = next(name for name in archive.namelist() if name.endswith(".dbf"))
        polygons = parse_shp_polygons(archive.read(shp_name))
        location_ids = parse_dbf_location_ids(archive.read(dbf_name))
    if len(polygons) != len(location_ids):
        raise ValueError("taxi zone shapefile record count does not match DBF row count")

    vertices_to_zones: dict[tuple[int, int], set[int]] = {}
    for location_id, parts in zip(location_ids, polygons, strict=True):
        for part in parts:
            for x, y in part:
                key = (round(x), round(y))
                vertices_to_zones.setdefault(key, set()).add(location_id)

    adjacency: dict[int, set[int]] = {location_id: set() for location_id in location_ids}
    for zones in vertices_to_zones.values():
        if len(zones) < 2:
            continue
        for zone in zones:
            adjacency[zone].update(other for other in zones if other != zone)
    return {zone: sorted(neighbors) for zone, neighbors in adjacency.items() if neighbors}


def taxi_zone_centroids_from_zip(path: Path) -> dict[int, tuple[float, float]]:
    with zipfile.ZipFile(path) as archive:
        shp_name = next(name for name in archive.namelist() if name.endswith(".shp"))
        dbf_name = next(name for name in archive.namelist() if name.endswith(".dbf"))
        polygons = parse_shp_polygons(archive.read(shp_name))
        location_ids = parse_dbf_location_ids(archive.read(dbf_name))
    if len(polygons) != len(location_ids):
        raise ValueError("taxi zone shapefile record count does not match DBF row count")

    raw_centroids: dict[int, tuple[float, float]] = {}
    for location_id, parts in zip(location_ids, polygons, strict=True):
        points = [point for part in parts for point in part]
        if not points:
            continue
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        raw_centroids[location_id] = (
            float(sum(x_values) / len(x_values)),
            float(sum(y_values) / len(y_values)),
        )
    if not raw_centroids:
        return {}
    xs = [point[0] for point in raw_centroids.values()]
    ys = [point[1] for point in raw_centroids.values()]
    min_x = min(xs)
    min_y = min(ys)
    span_x = max(max(xs) - min_x, 1.0)
    span_y = max(max(ys) - min_y, 1.0)
    return {
        zone: ((x - min_x) / span_x, (y - min_y) / span_y) for zone, (x, y) in raw_centroids.items()
    }


def parse_shp_polygons(data: bytes) -> list[list[list[tuple[float, float]]]]:
    if len(data) < 100:
        raise ValueError("invalid shapefile")
    offset = 100
    polygons: list[list[list[tuple[float, float]]]] = []
    while offset + 8 <= len(data):
        _, content_words = struct.unpack(">2i", data[offset : offset + 8])
        offset += 8
        content_bytes = content_words * 2
        record = data[offset : offset + content_bytes]
        offset += content_bytes
        if len(record) < 44:
            continue
        shape_type = struct.unpack("<i", record[:4])[0]
        if shape_type == 0:
            polygons.append([])
            continue
        if shape_type not in {5, 15, 25, 31}:
            continue
        num_parts, num_points = struct.unpack("<2i", record[36:44])
        parts_start = 44
        points_start = parts_start + num_parts * 4
        if len(record) < points_start + num_points * 16:
            raise ValueError("invalid polygon record")
        part_offsets = list(struct.unpack(f"<{num_parts}i", record[parts_start:points_start]))
        part_offsets.append(num_points)
        points = [
            struct.unpack("<2d", record[points_start + point * 16 : points_start + point * 16 + 16])
            for point in range(num_points)
        ]
        polygons.append(
            [points[part_offsets[index] : part_offsets[index + 1]] for index in range(num_parts)]
        )
    return polygons


def parse_dbf_location_ids(data: bytes) -> list[int]:
    if len(data) < 32:
        raise ValueError("invalid DBF")
    header_length = int.from_bytes(data[8:10], "little")
    record_length = int.from_bytes(data[10:12], "little")
    fields = []
    offset = 32
    while offset + 32 <= header_length and data[offset] != 0x0D:
        raw_name = data[offset : offset + 11].split(b"\x00", 1)[0]
        name = raw_name.decode("ascii", errors="ignore").strip()
        length = data[offset + 16]
        fields.append((name, length))
        offset += 32
    field_offset = 1
    location_slice: slice | None = None
    for name, length in fields:
        next_offset = field_offset + length
        if name == "LocationID":
            location_slice = slice(field_offset, next_offset)
            break
        field_offset = next_offset
    if location_slice is None:
        raise ValueError("DBF is missing LocationID")

    location_ids = []
    offset = header_length
    while offset + record_length <= len(data):
        record = data[offset : offset + record_length]
        offset += record_length
        if not record or record[0:1] == b"*":
            continue
        raw_value = record[location_slice].decode("ascii", errors="ignore").strip()
        if raw_value:
            location_ids.append(int(float(raw_value)))
    return location_ids


def load_tlc_frame(paths: list[Path]) -> Any:
    try:
        pandas = importlib.import_module("pandas")
        importlib.import_module("pyarrow")
    except Exception as exc:
        raise RuntimeError(
            "pandas and pyarrow are required for real TLC parquet benchmarks. "
            "Install them with `uv sync --group dev --group bench` before rerunning."
        ) from exc
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "PULocationID",
        "DOLocationID",
    ]
    frames = [pandas.read_parquet(path, columns=columns) for path in paths]
    return pandas.concat(frames, ignore_index=True)


def clean_tlc_frame(frame: Any) -> Any:
    pandas = optional_import("pandas")
    if pandas is None:
        raise RuntimeError("pandas is required for TLC frame cleaning")

    data = frame.copy()
    data["duration_sec"] = (
        pandas.to_datetime(data["tpep_dropoff_datetime"])
        - pandas.to_datetime(data["tpep_pickup_datetime"])
    ).dt.total_seconds()
    pickup_time = pandas.to_datetime(data["tpep_pickup_datetime"])
    data["hour"] = pickup_time.dt.hour.astype(float)
    data["dayofweek"] = pickup_time.dt.dayofweek.astype(float)
    data["passenger_count"] = data["passenger_count"].fillna(1.0).astype(float)

    mask = (
        data["duration_sec"].between(60.0, 7200.0)
        & data["trip_distance"].between(0.1, 100.0)
        & data["fare_amount"].between(2.5, 500.0)
        & data["total_amount"].between(2.5, 700.0)
        & data["PULocationID"].between(1, 263)
        & data["DOLocationID"].between(1, 263)
    )
    data = data.loc[mask].copy()
    data["log_trip_distance"] = np.log1p(data["trip_distance"].astype(float))
    data["log_duration_sec"] = np.log1p(data["duration_sec"].astype(float))
    data["log_total_amount"] = np.log1p(data["total_amount"].astype(float))
    return data.reset_index(drop=True)


def sample_tlc_frame(frame: Any, *, sample_size: int, seed: int) -> Any:
    if sample_size > 0 and len(frame) > sample_size:
        return frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def build_real_tasks(
    frame: Any,
    zone_lookup: dict[int, ZoneContext],
    zone_adjacency: dict[int, list[int]] | None = None,
    zone_centroids: dict[int, tuple[float, float]] | None = None,
    demand_frame: Any | None = None,
) -> list[BenchmarkTask]:
    demand_source = frame if demand_frame is None else demand_frame
    return [
        row_task(
            frame,
            zone_lookup=zone_lookup,
            zone_adjacency=zone_adjacency,
            zone_centroids=zone_centroids,
            name="duration",
            display_name="Trip duration",
            description="Predict log trip duration from zone, trip, passenger, and time features.",
            target_column="log_duration_sec",
            timestamp_column="tpep_pickup_datetime",
        ),
        row_task(
            frame,
            zone_lookup=zone_lookup,
            zone_adjacency=zone_adjacency,
            zone_centroids=zone_centroids,
            name="fare",
            display_name="Fare amount",
            description="Predict log total amount from zone, trip, passenger, and time features.",
            target_column="log_total_amount",
            timestamp_column="tpep_pickup_datetime",
        ),
        demand_task(
            demand_source,
            zone_lookup=zone_lookup,
            zone_adjacency=zone_adjacency,
            zone_centroids=zone_centroids,
        ),
    ]


def row_task(
    frame: Any,
    *,
    zone_lookup: dict[int, ZoneContext],
    zone_adjacency: dict[int, list[int]] | None,
    zone_centroids: dict[int, tuple[float, float]] | None,
    name: str,
    display_name: str,
    description: str,
    target_column: str,
    timestamp_column: str | None = "tpep_pickup_datetime",
) -> BenchmarkTask:
    features = frame[ROW_FEATURES].to_numpy(dtype=float)
    target = frame[target_column].to_numpy(dtype=float)
    pickup_zones = frame["PULocationID"].to_numpy(dtype=int)
    timestamps = None
    if timestamp_column is not None:
        if timestamp_column not in frame:
            raise ValueError(f"row task {name!r} requires timestamp column {timestamp_column!r}")
        timestamps = normalize_timestamps(frame[timestamp_column].to_numpy())
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in frame["PULocationID"].to_numpy(dtype=int)],
        "dropoff_zone": [[int(value)] for value in frame["DOLocationID"].to_numpy(dtype=int)],
        "pickup_borough": zone_sparse_set(
            frame["PULocationID"].to_numpy(dtype=int), zone_lookup, "borough"
        ),
        "dropoff_borough": zone_sparse_set(
            frame["DOLocationID"].to_numpy(dtype=int), zone_lookup, "borough"
        ),
        "pickup_service_zone": zone_sparse_set(
            frame["PULocationID"].to_numpy(dtype=int), zone_lookup, "service_zone"
        ),
        "dropoff_service_zone": zone_sparse_set(
            frame["DOLocationID"].to_numpy(dtype=int), zone_lookup, "service_zone"
        ),
    }
    return BenchmarkTask(
        name=name,
        display_name=display_name,
        description=description,
        features=features,
        target=target,
        pickup_zones=pickup_zones,
        feature_names=list(ROW_FEATURES),
        sparse_sets=sparse_sets,
        zone_adjacency=zone_adjacency,
        zone_centroids=zone_centroids,
        timestamps=timestamps,
        timestamp_column=timestamp_column,
    )


def demand_task(
    frame: Any,
    *,
    zone_lookup: dict[int, ZoneContext],
    zone_adjacency: dict[int, list[int]] | None,
    zone_centroids: dict[int, tuple[float, float]] | None = None,
) -> BenchmarkTask:
    grouped = (
        frame.groupby(["PULocationID", "hour", "dayofweek"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )
    features = grouped[DEMAND_FEATURES].to_numpy(dtype=float)
    target = np.log1p(grouped["trip_count"].to_numpy(dtype=float))
    pickup_zones = grouped["PULocationID"].to_numpy(dtype=int)
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in grouped["PULocationID"].to_numpy(dtype=int)],
        "pickup_borough": zone_sparse_set(pickup_zones, zone_lookup, "borough"),
        "pickup_service_zone": zone_sparse_set(pickup_zones, zone_lookup, "service_zone"),
    }
    return BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="Predict log pickup trip count for a pickup zone, hour, and weekday bucket.",
        features=features,
        target=target,
        pickup_zones=pickup_zones,
        feature_names=list(DEMAND_FEATURES),
        sparse_sets=sparse_sets,
        zone_adjacency=zone_adjacency,
        zone_centroids=zone_centroids,
    )


def synthetic_tasks() -> list[BenchmarkTask]:
    rng = np.random.default_rng(7)
    rows: list[list[float]] = []
    targets_duration: list[float] = []
    targets_fare: list[float] = []
    timestamps: list[np.datetime64] = []
    for pickup in range(1, 13):
        for dropoff in range(1, 13):
            for hour in range(24):
                distance = 0.8 + abs(dropoff - pickup) * 0.55 + rng.normal(0.0, 0.02)
                log_distance = math.log1p(distance)
                passenger_count = 1.0 + float((pickup + dropoff) % 3)
                weekday = float((pickup + hour) % 7)
                rows.append(
                    [distance, log_distance, passenger_count, float(hour), weekday, pickup, dropoff]
                )
                # Keep the synthetic fixture temporal as well.  This exercises
                # the exact same leakage-safe split path as real TLC rows while
                # remaining deterministic and independent of wall-clock time.
                timestamps.append(
                    np.datetime64("2024-01-01T00", "h") + np.timedelta64(len(rows), "h")
                )
                night = 1.0 if hour >= 22 or hour <= 2 else 0.0
                zone_effect = 0.08 * pickup + 0.05 * dropoff
                targets_duration.append(math.log1p(300.0 + 120.0 * distance + 80.0 * night))
                targets_fare.append(math.log1p(5.0 + 3.2 * distance + zone_effect + 1.5 * night))

    features = np.asarray(rows, dtype=float)
    pickup_zones = features[:, 5].astype(int)
    dropoff_zones = features[:, 6].astype(int)
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in pickup_zones],
        "dropoff_zone": [[int(value)] for value in dropoff_zones],
    }
    duration = BenchmarkTask(
        name="duration",
        display_name="Trip duration",
        description="Synthetic log trip duration fixture.",
        features=features,
        target=np.asarray(targets_duration, dtype=float),
        pickup_zones=pickup_zones,
        feature_names=list(ROW_FEATURES),
        sparse_sets=sparse_sets,
        timestamps=np.asarray(timestamps, dtype="datetime64[h]"),
        timestamp_column="synthetic_timestamp",
    )
    fare = BenchmarkTask(
        name="fare",
        display_name="Fare amount",
        description="Synthetic log fare fixture.",
        features=features,
        target=np.asarray(targets_fare, dtype=float),
        pickup_zones=pickup_zones,
        feature_names=list(ROW_FEATURES),
        sparse_sets=sparse_sets,
        timestamps=np.asarray(timestamps, dtype="datetime64[h]"),
        timestamp_column="synthetic_timestamp",
    )

    demand_rows: list[list[float]] = []
    demand_targets: list[float] = []
    for pickup in range(1, 13):
        for hour in range(24):
            for weekday in range(7):
                commute = 1.0 if hour in {7, 8, 17, 18} else 0.0
                demand = 15.0 + 2.0 * pickup + 9.0 * commute + 3.0 * (weekday >= 5)
                demand_rows.append([float(pickup), float(hour), float(weekday)])
                demand_targets.append(math.log1p(demand))
    demand_features = np.asarray(demand_rows, dtype=float)
    demand_pickups = demand_features[:, 0].astype(int)
    demand = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="Synthetic pickup demand fixture.",
        features=demand_features,
        target=np.asarray(demand_targets, dtype=float),
        pickup_zones=demand_pickups,
        feature_names=list(DEMAND_FEATURES),
        sparse_sets={"pickup_zone": [[int(value)] for value in demand_pickups]},
    )
    return [duration, fare, demand]


def zone_sparse_set(
    zone_ids: np.ndarray,
    zone_lookup: dict[int, ZoneContext],
    kind: str,
) -> list[list[int]]:
    rows: list[list[int]] = []
    for raw_zone_id in zone_ids:
        zone_id = int(raw_zone_id)
        context = zone_lookup.get(zone_id)
        if context is None:
            raise ValueError(f"missing zone context for {kind} sparse set: {zone_id}")
        if kind == "borough":
            rows.append([int(context.borough_code)])
        elif kind == "service_zone":
            rows.append([int(context.service_zone_code)])
        else:
            raise ValueError(f"unknown zone sparse-set kind: {kind}")
    return rows


def optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def normalize_timestamps(values: Any) -> np.ndarray:
    """Convert row timestamps to a comparable, timezone-normalized array.

    The benchmark stores timestamps as ``datetime64[ns]`` so the split boundary
    is computed from the source timestamp rather than from a model feature.  A
    clear error for missing or invalid values is important here: dropping rows
    or substituting a synthetic date would invalidate the temporal claim.
    """

    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        normalized = array.astype("datetime64[ns]")
    else:
        pandas = optional_import("pandas")
        if pandas is None:
            raise RuntimeError("pandas is required to parse non-NumPy timestamps in NYC row tasks")
        try:
            try:
                parsed = pandas.to_datetime(array, errors="raise", utc=True, format="mixed")
            except (TypeError, ValueError):
                # ``format='mixed'`` is unavailable on older pandas releases;
                # their parser is still valid for uniformly formatted input.
                parsed = pandas.to_datetime(array, errors="raise", utc=True)
            normalized = parsed.to_numpy(dtype="datetime64[ns]")
        except Exception as exc:
            raise ValueError("row task timestamps must be valid datetimes") from exc
    if normalized.ndim != 1:
        raise ValueError("row task timestamps must be one-dimensional")
    if np.any(normalized == np.datetime64("NaT", "ns")):
        raise ValueError("row task timestamps must not contain NaT")
    return normalized


def timestamp_integer_values(values: np.ndarray) -> np.ndarray:
    """Return timestamps as integer nanoseconds for ordering and hashing."""

    normalized = normalize_timestamps(values)
    return normalized.astype("datetime64[ns]").astype(np.int64)


def timestamp_json_value(value: Any) -> str:
    """Render one timestamp deterministically for split manifests."""

    if isinstance(value, (int, np.integer)):
        timestamp = np.datetime64(int(value), "ns")
    else:
        timestamp = np.datetime64(value, "ns")
    return np.datetime_as_string(timestamp, unit="ns")


def split_indices(task: BenchmarkTask, *, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(task.target)
    if mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        test_count = max(1, int(count * 0.2))
        return order[test_count:], order[:test_count]

    if mode == "out_of_time":
        return out_of_time_indices(task)

    if mode != "spatial_holdout":
        raise ValueError(f"unknown split mode: {mode!r}")

    unique_zones = np.unique(task.pickup_zones)
    holdout_zones = set(int(zone) for zone in unique_zones[::5])
    test_mask = np.asarray([int(zone) in holdout_zones for zone in task.pickup_zones])
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"spatial_holdout split for task {task.name!r} produced "
            f"{len(train_indices)} train rows and {len(test_indices)} test rows"
        )
    return train_indices, test_indices


def out_of_time_indices(task: BenchmarkTask) -> tuple[np.ndarray, np.ndarray]:
    """Split rows chronologically while keeping equal timestamps together.

    The latest approximately 20 percent of *distinct timestamp groups* is the
    holdout.  Rows sharing the boundary timestamp all remain on the test side,
    so no exact pickup time appears in both train and test.  This is stricter
    than slicing a shuffled sample at a quantile and prevents timestamp leakage
    when many TLC trips share the same pickup second.
    """

    if task.timestamps is None:
        raise ValueError(f"out_of_time split for task {task.name!r} requires source row timestamps")
    if len(task.timestamps) != len(task.target):
        raise ValueError(
            f"out_of_time split for task {task.name!r} has {len(task.timestamps)} timestamps "
            f"for {len(task.target)} target rows"
        )
    if len(task.target) < 2:
        raise ValueError(f"out_of_time split for task {task.name!r} requires at least two rows")

    timestamp_values = timestamp_integer_values(task.timestamps)
    unique_timestamps = np.unique(timestamp_values)
    if unique_timestamps.size < 2:
        raise ValueError(
            f"out_of_time split for task {task.name!r} requires at least two distinct timestamps"
        )

    desired_test_rows = max(1, int(len(task.target) * 0.2))
    # Pick the distinct boundary whose test-group count is closest to the
    # requested holdout size.  Ties prefer a later boundary (more training
    # rows), while preserving timestamp disjointness takes precedence over an
    # exact row fraction when a large tie group straddles the nominal cutoff.
    candidate_boundaries = unique_timestamps[1:]
    boundary = min(
        candidate_boundaries,
        key=lambda candidate: (
            abs(int(np.count_nonzero(timestamp_values >= candidate)) - desired_test_rows),
            -int(candidate),
        ),
    )
    test_mask = timestamp_values >= boundary
    train_indices = np.flatnonzero(~test_mask).astype(np.int64)
    test_indices = np.flatnonzero(test_mask).astype(np.int64)
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError(
            f"out_of_time split for task {task.name!r} produced "
            f"{len(train_indices)} train rows and {len(test_indices)} test rows"
        )
    if np.intersect1d(timestamp_values[train_indices], timestamp_values[test_indices]).size:
        raise AssertionError("out_of_time split has overlapping train/test timestamps")
    return train_indices, test_indices


def split_modes_for_task(task: BenchmarkTask) -> list[str]:
    """Return the outer splits applicable to one benchmark task.

    Out-of-time evaluation is a row-task protocol.  Pickup demand is an
    aggregated zone/time task and remains on its random and spatial diagnostics
    until a dedicated rolling-origin protocol is run.
    """

    modes = ["random", "spatial_holdout"]
    if task.name in {"duration", "fare"} and task.timestamps is not None:
        modes.append("out_of_time")
    return modes


def train_only_validation_indices(row_count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create an inner model-selection split without touching the benchmark holdout."""
    if row_count < 4:
        raise ValueError("at least four training rows are required for inner validation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(row_count)
    holdout_count = max(1, int(round(row_count * 0.2)))
    holdout_count = min(holdout_count, row_count - 1)
    holdout = np.sort(order[:holdout_count])
    train = np.sort(order[holdout_count:])
    if train.size == 0 or holdout.size == 0:
        raise ValueError("inner validation split produced an empty side")
    return train.astype(np.int64), holdout.astype(np.int64)


def task_coordinate_matrix(task: BenchmarkTask) -> np.ndarray | None:
    if not task.zone_centroids:
        return None
    coords: list[tuple[float, float]] = []
    for zone in task.pickup_zones:
        centroid = task.zone_centroids.get(int(zone))
        if centroid is None:
            raise ValueError(f"missing pickup zone centroid for spatial benchmark: {int(zone)}")
        coords.append((float(centroid[0]), float(centroid[1])))
    return np.asarray(coords, dtype=float)


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residuals = actual - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    absolute_actual_sum = float(np.sum(np.abs(actual)))
    wape = (
        float(np.sum(np.abs(residuals)) / absolute_actual_sum)
        if absolute_actual_sum > 0.0
        else float("nan")
    )
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / variance if variance > 0.0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape": wape}


def pickup_zone_diagnostics(
    actual: np.ndarray,
    predicted: np.ndarray,
    pickup_zones: np.ndarray,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    residuals = actual - predicted
    rows = []
    for zone in sorted(np.unique(pickup_zones)):
        mask = pickup_zones == zone
        zone_actual = actual[mask]
        zone_residuals = residuals[mask]
        absolute_actual_sum = float(np.sum(np.abs(zone_actual)))
        zone_wape = (
            float(np.sum(np.abs(zone_residuals)) / absolute_actual_sum)
            if absolute_actual_sum > 0.0
            else float("nan")
        )
        rows.append(
            {
                "pickup_zone": int(zone),
                "rows": int(np.sum(mask)),
                "rmse": float(np.sqrt(np.mean(zone_residuals**2))),
                "mae": float(np.mean(np.abs(zone_residuals))),
                "bias": float(np.mean(predicted[mask] - zone_actual)),
                "wape": zone_wape,
            }
        )

    if not rows:
        return {
            "segment_key": "pickup_zone",
            "segment_count": 0,
            "row_count_min": 0,
            "row_count_max": 0,
            "rmse_quantiles": {},
            "mae_quantiles": {},
            "wape_quantiles": {},
            "worst_rmse_segments": [],
            "largest_abs_bias_segments": [],
        }

    def quantiles(metric: str) -> dict[str, float]:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"p50": float("nan"), "p90": float("nan"), "max": float("nan")}
        return {
            "p50": float(np.quantile(values, 0.5)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        }

    row_counts = [row["rows"] for row in rows]
    return {
        "segment_key": "pickup_zone",
        "segment_count": len(rows),
        "row_count_min": int(min(row_counts)),
        "row_count_max": int(max(row_counts)),
        "rmse_quantiles": quantiles("rmse"),
        "mae_quantiles": quantiles("mae"),
        "wape_quantiles": quantiles("wape"),
        "worst_rmse_segments": sorted(rows, key=lambda row: row["rmse"], reverse=True)[:top_n],
        "largest_abs_bias_segments": sorted(
            rows,
            key=lambda row: abs(float(row["bias"])),
            reverse=True,
        )[:top_n],
    }


def calibration_report_fields(
    actual: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    *,
    horizons: np.ndarray | None = None,
    spatial_blocks: np.ndarray | None = None,
    residual_morans_i_after_calibration: float | None = None,
) -> dict[str, Any]:
    """Return benchmark uncertainty fields without fabricating calibration claims."""
    if lower is None or upper is None:
        return {
            "status": "not_computed",
            "coverage_by_horizon": {},
            "coverage_by_spatial_block": {},
            "width_by_horizon": {},
            "residual_morans_i_after_calibration": None,
        }

    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if actual.shape != lower.shape or actual.shape != upper.shape:
        raise ValueError("actual, lower, and upper must have the same shape")

    covered = (actual >= lower) & (actual <= upper)
    widths = upper - lower

    def grouped(values: np.ndarray, groups: np.ndarray | None) -> dict[str, float]:
        if groups is None:
            return {}
        group_array = np.asarray(groups)
        if group_array.shape != actual.shape:
            raise ValueError("calibration report groups must match actual shape")
        return {
            str(group): float(np.mean(values[group_array == group]))
            for group in sorted(np.unique(group_array), key=lambda item: str(item))
        }

    return {
        "status": "computed",
        "coverage_by_horizon": grouped(covered.astype(float), horizons),
        "coverage_by_spatial_block": grouped(covered.astype(float), spatial_blocks),
        "width_by_horizon": grouped(widths, horizons),
        "residual_morans_i_after_calibration": (
            None
            if residual_morans_i_after_calibration is None
            else float(residual_morans_i_after_calibration)
        ),
    }


def cartoboost_schema(
    task: BenchmarkTask,
    *,
    feature_names: list[str] | None = None,
    dense_id_sets: bool = False,
    include_sparse_sets: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    dense = []
    for name in feature_names or task.feature_names:
        if name == "hour":
            dense.append({"name": name, "kind": "periodic", "period": 24})
        elif name == "dayofweek":
            dense.append({"name": name, "kind": "periodic", "period": 7})
        elif name.endswith("_centroid_x") or name.endswith("_centroid_y"):
            dense.append({"name": name, "kind": "spatial"})
        else:
            dense.append({"name": name, "kind": "numeric"})
    sparse_sets = (
        [{"name": name, "kind": "sparse_set"} for name in task.sparse_sets]
        if include_sparse_sets
        else []
    )
    return {"dense": dense, "sparse_sets": sparse_sets}


def transformed_split_features(
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_x = task.features[train_indices]
    test_x = task.features[test_indices]
    if args.zone_treatment == "raw":
        return append_zone_geometry_features(
            train_x,
            test_x,
            task.feature_names,
            task=task,
            train_indices=train_indices,
            test_indices=test_indices,
        )
    if args.zone_treatment != "target_mean":
        raise ValueError(f"unknown zone treatment: {args.zone_treatment}")
    train_y = task.target[train_indices]
    train_x, test_x, feature_names = append_zone_target_mean_features(
        train_x,
        test_x,
        train_y,
        task.feature_names,
        task=task,
        train_indices=train_indices,
        test_indices=test_indices,
        smoothing=args.zone_target_smoothing,
    )
    return append_zone_geometry_features(
        train_x,
        test_x,
        feature_names,
        task=task,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def append_zone_geometry_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: list[str],
    *,
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not task.zone_centroids:
        return train_x, test_x, list(feature_names)
    zone_feature_indices = [
        index for index, name in enumerate(task.feature_names) if name in ZONE_FEATURES
    ]
    if not zone_feature_indices:
        return train_x, test_x, list(feature_names)

    def centroid_columns(row_indices: np.ndarray) -> list[np.ndarray]:
        columns: list[np.ndarray] = []
        for source_feature_index in zone_feature_indices:
            zone_ids = task.features[row_indices, source_feature_index].astype(int)
            xs = []
            ys = []
            for zone_id in zone_ids:
                x, y = required_zone_centroid(task, int(zone_id))
                xs.append(x)
                ys.append(y)
            columns.extend([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        if len(zone_feature_indices) >= 2:
            pickup = task.features[row_indices, zone_feature_indices[0]].astype(int)
            dropoff = task.features[row_indices, zone_feature_indices[1]].astype(int)
            deltas_x = []
            deltas_y = []
            distances = []
            for source, target in zip(pickup, dropoff, strict=True):
                source_x, source_y = required_zone_centroid(task, int(source))
                target_x, target_y = required_zone_centroid(task, int(target))
                delta_x = target_x - source_x
                delta_y = target_y - source_y
                deltas_x.append(delta_x)
                deltas_y.append(delta_y)
                distances.append(math.hypot(delta_x, delta_y))
            columns.extend(
                [
                    np.asarray(deltas_x, dtype=float),
                    np.asarray(deltas_y, dtype=float),
                    np.asarray(distances, dtype=float),
                ]
            )
        return columns

    geometry_names: list[str] = []
    for feature_index in zone_feature_indices:
        name = task.feature_names[feature_index]
        geometry_names.extend([f"{name}_centroid_x", f"{name}_centroid_y"])
    if len(zone_feature_indices) >= 2:
        geometry_names.extend(
            ["od_centroid_delta_x", "od_centroid_delta_y", "od_centroid_distance"]
        )
    train_columns = centroid_columns(train_indices)
    test_columns = centroid_columns(test_indices)
    return (
        np.column_stack([train_x, *train_columns]).astype(float, copy=False),
        np.column_stack([test_x, *test_columns]).astype(float, copy=False),
        [*feature_names, *geometry_names],
    )


def required_zone_centroid(task: BenchmarkTask, zone_id: int) -> tuple[float, float]:
    if not task.zone_centroids:
        raise ValueError(f"task {task.name} is missing required taxi zone centroids")
    try:
        return task.zone_centroids[zone_id]
    except KeyError as exc:
        raise KeyError(f"task {task.name} is missing centroid for taxi zone {zone_id}") from exc


def append_zone_target_mean_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    feature_names: list[str],
    *,
    task: BenchmarkTask | None = None,
    train_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    zone_feature_indices = [
        index for index, name in enumerate(feature_names) if name in ZONE_FEATURES
    ]
    if not zone_feature_indices:
        return train_x, test_x, list(feature_names)

    global_mean = float(np.mean(train_y))
    smoothing = max(0.0, float(smoothing))
    train_columns = []
    test_columns = []
    new_names = list(feature_names)
    for feature_index in zone_feature_indices:
        train_column, test_column = smoothed_target_mean_column(
            train_x[:, feature_index],
            test_x[:, feature_index],
            train_y,
            global_mean=global_mean,
            smoothing=smoothing,
        )
        train_columns.append(train_column)
        test_columns.append(test_column)
        new_names.append(f"{feature_names[feature_index]}_target_mean")
        if task is None or train_indices is None or test_indices is None:
            continue
        context_columns = zone_context_target_mean_features(
            task,
            train_indices,
            test_indices,
            train_y,
            feature_name=feature_names[feature_index],
            smoothing=smoothing,
            global_mean=global_mean,
        )
        for name, train_context_column, test_context_column in context_columns:
            train_columns.append(train_context_column)
            test_columns.append(test_context_column)
            new_names.append(name)

    return (
        np.column_stack([train_x, *train_columns]).astype(float, copy=False),
        np.column_stack([test_x, *test_columns]).astype(float, copy=False),
        new_names,
    )


def smoothed_target_mean_column(
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    train_y: np.ndarray,
    *,
    global_mean: float,
    smoothing: float,
    skip_negative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for raw_id, target in zip(train_ids, train_y, strict=True):
        zone_id = int(raw_id)
        if skip_negative and zone_id < 0:
            continue
        sums[zone_id] = sums.get(zone_id, 0.0) + float(target)
        counts[zone_id] = counts.get(zone_id, 0) + 1
    encoded: dict[int, float] = {
        zone_id: (sums[zone_id] + smoothing * global_mean) / (counts[zone_id] + smoothing)
        for zone_id in counts
    }
    train_encoded = np.asarray(
        [
            encoded.get(int(zone_id), global_mean)
            if not skip_negative or int(zone_id) >= 0
            else global_mean
            for zone_id in train_ids
        ],
        dtype=float,
    )
    test_encoded = np.asarray(
        [
            encoded.get(int(zone_id), global_mean)
            if not skip_negative or int(zone_id) >= 0
            else global_mean
            for zone_id in test_ids
        ],
        dtype=float,
    )
    return train_encoded, test_encoded


def zone_context_target_mean_features(
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_y: np.ndarray,
    *,
    feature_name: str,
    smoothing: float,
    global_mean: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    context_names = zone_context_sparse_names(feature_name)
    if not context_names and not task.zone_adjacency:
        return []
    feature_index = task.feature_names.index(feature_name)
    train_ids = task.features[train_indices, feature_index].astype(int)
    test_ids = task.features[test_indices, feature_index].astype(int)
    columns: list[tuple[str, np.ndarray, np.ndarray]] = []

    for label, sparse_name in context_names:
        train_context = sparse_set_first_values(
            task.sparse_sets.get(sparse_name, []), train_indices
        )
        test_context = sparse_set_first_values(task.sparse_sets.get(sparse_name, []), test_indices)
        train_column, test_column = smoothed_target_mean_column(
            train_context,
            test_context,
            train_y,
            global_mean=global_mean,
            smoothing=smoothing,
            skip_negative=True,
        )
        columns.append((f"{feature_name}_{label}_target_mean", train_column, test_column))

    if task.zone_adjacency:
        train_column, test_column = adjacency_target_mean_column(
            train_ids,
            test_ids,
            train_y,
            task.zone_adjacency,
            global_mean=global_mean,
            smoothing=smoothing,
        )
        columns.append((f"{feature_name}_adjacent_target_mean", train_column, test_column))
    return columns


def zone_context_sparse_names(feature_name: str) -> list[tuple[str, str]]:
    if feature_name == "PULocationID":
        return [
            ("borough", "pickup_borough"),
            ("service_zone", "pickup_service_zone"),
        ]
    if feature_name == "DOLocationID":
        return [
            ("borough", "dropoff_borough"),
            ("service_zone", "dropoff_service_zone"),
        ]
    return []


def sparse_set_first_values(values: list[list[int]], indices: np.ndarray) -> np.ndarray:
    encoded = []
    for index in indices:
        row = values[int(index)] if int(index) < len(values) else []
        encoded.append(int(row[0]) if row else -1)
    return np.asarray(encoded, dtype=int)


def adjacency_target_mean_column(
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    train_y: np.ndarray,
    adjacency: dict[int, list[int]],
    *,
    global_mean: float,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for raw_id, target in zip(train_ids, train_y, strict=True):
        zone_id = int(raw_id)
        sums[zone_id] = sums.get(zone_id, 0.0) + float(target)
        counts[zone_id] = counts.get(zone_id, 0) + 1
    encoded = {
        zone_id: (sums[zone_id] + smoothing * global_mean) / (counts[zone_id] + smoothing)
        for zone_id in counts
    }

    def neighbor_mean(zone_id: int) -> float:
        neighbor_values = [
            encoded[neighbor] for neighbor in adjacency.get(zone_id, []) if neighbor in encoded
        ]
        if not neighbor_values:
            return global_mean
        return float(np.mean(neighbor_values))

    return (
        np.asarray([neighbor_mean(int(zone_id)) for zone_id in train_ids], dtype=float),
        np.asarray([neighbor_mean(int(zone_id)) for zone_id in test_ids], dtype=float),
    )


def task_zone_column(task: BenchmarkTask, name: str) -> np.ndarray | None:
    if name not in task.feature_names:
        return None
    index = task.feature_names.index(name)
    return task.features[:, index].astype(np.int64)


def task_embedding_ids(task: BenchmarkTask) -> np.ndarray:
    pickup = task_zone_column(task, "PULocationID")
    if pickup is None:
        return np.arange(len(task.target), dtype=np.uint64).reshape(-1, 1)
    hour = task_zone_column(task, "hour")
    dropoff = task_zone_column(task, "DOLocationID")
    if dropoff is None:
        if hour is None:
            return pickup.astype(np.uint64).reshape(-1, 1)
        pickup_hour = pickup.astype(np.uint64) * np.uint64(10_000) + hour.astype(np.uint64)
        return np.column_stack([pickup.astype(np.uint64), pickup_hour])
    od_pair = pickup.astype(np.uint64) * np.uint64(1_000) + dropoff.astype(np.uint64)
    return np.column_stack([pickup.astype(np.uint64), dropoff.astype(np.uint64), od_pair])


def task_embedding_fallback_context(
    task: BenchmarkTask,
    *,
    train_indices: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[list[np.ndarray | None], list[list[list[int]] | None]]:
    ids = task_embedding_ids(task)
    key_count = 1 if ids.ndim == 1 else ids.shape[1]
    fallback_by_key: list[np.ndarray | None] = [None] * key_count
    neighbor_by_key: list[list[list[int]] | None] = [None] * key_count
    pickup = task_zone_column(task, "PULocationID")
    dropoff = task_zone_column(task, "DOLocationID")

    if pickup is not None and key_count >= 1:
        fallback_by_key[0] = zone_fallback_matrix(
            task, pickup, train_indices, row_indices, "pickup"
        )
        neighbor_by_key[0] = zone_neighbor_lists(task, pickup, row_indices)

    if dropoff is not None and key_count >= 2:
        fallback_by_key[1] = zone_fallback_matrix(
            task, dropoff, train_indices, row_indices, "dropoff"
        )
        neighbor_by_key[1] = zone_neighbor_lists(task, dropoff, row_indices)

    return fallback_by_key, neighbor_by_key


def zone_fallback_matrix(
    task: BenchmarkTask,
    zones: np.ndarray,
    train_indices: np.ndarray,
    row_indices: np.ndarray,
    prefix: str,
) -> np.ndarray:
    train_zones = zones[train_indices].astype(np.uint64)
    borough_values = task.sparse_sets.get(f"{prefix}_borough", [])
    service_values = task.sparse_sets.get(f"{prefix}_service_zone", [])
    first_by_borough: dict[int, int] = {}
    first_by_service: dict[int, int] = {}
    global_zone = int(train_zones[0]) if len(train_zones) else 0
    for row_index in train_indices:
        zone = int(zones[row_index])
        if row_index < len(borough_values) and borough_values[row_index]:
            first_by_borough.setdefault(int(borough_values[row_index][0]), zone)
        if row_index < len(service_values) and service_values[row_index]:
            first_by_service.setdefault(int(service_values[row_index][0]), zone)

    rows: list[list[int]] = []
    for row_index in row_indices:
        borough_zone = global_zone
        service_zone = global_zone
        if row_index < len(borough_values) and borough_values[row_index]:
            borough_zone = first_by_borough.get(int(borough_values[row_index][0]), global_zone)
        if row_index < len(service_values) and service_values[row_index]:
            service_zone = first_by_service.get(int(service_values[row_index][0]), global_zone)
        rows.append([service_zone, borough_zone, global_zone])
    return np.asarray(rows, dtype=np.uint64)


def zone_neighbor_lists(
    task: BenchmarkTask,
    zones: np.ndarray,
    row_indices: np.ndarray,
) -> list[list[int]]:
    adjacency = task.zone_adjacency or {}
    return [
        [int(neighbor) for neighbor in adjacency.get(int(zones[row_index]), [])]
        for row_index in row_indices
    ]


def zone_context_maps(
    task: BenchmarkTask,
    pickup: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    borough_by_zone: dict[int, int] = {}
    service_by_zone: dict[int, int] = {}
    borough_values = task.sparse_sets.get("pickup_borough", [])
    service_values = task.sparse_sets.get("pickup_service_zone", [])
    for row, raw_zone in enumerate(pickup):
        zone = int(raw_zone)
        if row < len(borough_values) and borough_values[row]:
            borough_by_zone.setdefault(zone, int(borough_values[row][0]))
        if row < len(service_values) and service_values[row]:
            service_by_zone.setdefault(zone, int(service_values[row][0]))
    return borough_by_zone, service_by_zone


def add_zone_context_graph(
    edges: set[tuple[int, int]],
    task: BenchmarkTask,
    pickup: np.ndarray,
    node_features: np.ndarray,
) -> tuple[int, dict[str, int]]:
    observed_zones = sorted(int(zone) for zone in np.unique(pickup))
    max_zone = node_features.shape[0] - 1
    next_node = max_zone + 1
    topology_counts = {
        "self_loop_edges": 0,
        "adjacency_edges": 0,
        "borough_edges": 0,
        "service_zone_edges": 0,
        "context_hub_nodes": 0,
    }

    for zone in observed_zones:
        edges.add((zone, zone))
        topology_counts["self_loop_edges"] += 1

    for source, neighbors in (task.zone_adjacency or {}).items():
        if source not in observed_zones:
            continue
        for target in neighbors:
            if target in observed_zones:
                edges.add((int(source), int(target)))
                topology_counts["adjacency_edges"] += 1

    borough_by_zone, service_by_zone = zone_context_maps(task, pickup)
    hub_features: list[list[float]] = []

    def add_hub_edges(values_by_zone: dict[int, int], offset: int, count_key: str) -> None:
        nonlocal next_node
        hubs: dict[int, int] = {}
        for zone in observed_zones:
            value = values_by_zone.get(zone)
            if value is None:
                continue
            if value not in hubs:
                hubs[value] = next_node
                next_node += 1
                topology_counts["context_hub_nodes"] += 1
                hub_features.append(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        float(value if offset == 0 else 0) / 10.0,
                        float(value if offset == 1 else 0) / 10.0,
                    ]
                )
            hub = hubs[value]
            edges.add((zone, hub))
            edges.add((hub, zone))
            topology_counts[count_key] += 2

    add_hub_edges(borough_by_zone, 0, "borough_edges")
    add_hub_edges(service_by_zone, 1, "service_zone_edges")

    if hub_features:
        node_features.resize((next_node, node_features.shape[1]), refcheck=False)
        node_features[max_zone + 1 : next_node] = np.asarray(hub_features, dtype=np.float64)
    return next_node, topology_counts


def graph_augmented_split_features(
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_x: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
    *,
    graph_family: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pickup = task_zone_column(task, "PULocationID")
    if pickup is None:
        raise ValueError("graph benchmark requires PULocationID")
    dropoff = task_zone_column(task, "DOLocationID")
    if dropoff is None:
        dropoff = pickup

    max_zone = int(max(np.max(pickup), np.max(dropoff), 1))
    node_count = max_zone + 1
    train_pickup = pickup[train_indices]
    train_dropoff = dropoff[train_indices]
    edge_set = {
        (int(source), int(target))
        for source, target in zip(train_pickup, train_dropoff, strict=True)
    }
    if not edge_set:
        edge_set = {(int(zone), int(zone)) for zone in np.unique(train_pickup)}

    node_features = np.zeros((node_count, 6), dtype=np.float64)
    node_features[:, 0] = np.arange(node_count, dtype=np.float64) / float(max(node_count - 1, 1))
    pickup_counts = np.bincount(train_pickup, minlength=node_count).astype(np.float64)
    dropoff_counts = np.bincount(train_dropoff, minlength=node_count).astype(np.float64)
    node_features[:node_count, 1] = np.log1p(pickup_counts)
    node_features[:node_count, 2] = np.log1p(dropoff_counts)
    node_features[:node_count, 3] = (pickup_counts + dropoff_counts > 0).astype(np.float64)
    borough_by_zone, service_by_zone = zone_context_maps(task, pickup)
    for zone, value in borough_by_zone.items():
        if zone < node_count:
            node_features[zone, 4] = float(value) / 10.0
    for zone, value in service_by_zone.items():
        if zone < node_count:
            node_features[zone, 5] = float(value) / 10.0

    topology_counts: dict[str, int] = {
        "self_loop_edges": 0,
        "adjacency_edges": 0,
        "borough_edges": 0,
        "service_zone_edges": 0,
        "context_hub_nodes": 0,
    }
    graph_topology = "train_flow"
    if task_zone_column(task, "DOLocationID") is None:
        node_count, topology_counts = add_zone_context_graph(
            edge_set,
            task,
            pickup,
            node_features,
        )
        graph_topology = "zone_adjacency_borough_service"

    edges = sorted(
        {
            (int(source), int(target))
            for source, target in edge_set
            if source < node_count and target < node_count
        }
    )

    from cartoboost.graph import (
        DirectionalFeature,
        DirectionalityConfig,
        GraphEmbeddingsConfig,
        GraphEncoderConfig,
        GraphEncoderFamily,
        GraphFeatureTransformer,
    )

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
        fit_edges: list[tuple[int, int]] | list[tuple[int, int, int]] = edges
        fit_kwargs: dict[str, Any] = {}
    elif graph_family == "hetero_graphsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.HINSAGE,
            hetero=True,
            input_dim=int(node_features.shape[1]),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
        )
        fit_edges = [(source, target, 0) for source, target in edges]
        fit_kwargs = {"relation_ids": [0]}
    elif graph_family == "hinsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.HINSAGE,
            input_dim=int(node_features.shape[1]),
            node_type_count=1,
            edge_type_triples=((0, 0, 0),),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
            neighbor_samples=(8,),
        )
        fit_edges = [(source, target, 0) for source, target in edges]
        fit_kwargs = {"node_types": [0] * node_count}
    else:
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.GRAPHSAGE,
            input_dim=int(node_features.shape[1]),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
        )
        fit_edges = edges
        fit_kwargs = {}

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
        node_features=node_features,
        edges=fit_edges,
        node_count=node_count,
        directed=True,
        **fit_kwargs,
    )
    embeddings = np.asarray(bundle.embeddings, dtype=np.float64)
    source_embeddings = embeddings[pickup]
    target_embeddings = embeddings[dropoff]
    graph_features = (
        np.hstack([source_embeddings, node_features[pickup]])
        if task_zone_column(task, "DOLocationID") is None
        else np.hstack([source_embeddings, target_embeddings])
    )
    return (
        np.hstack([train_x, graph_features[train_indices]]),
        np.hstack([test_x, graph_features[test_indices]]),
        {
            "graph_dim": int(args.graph_dim),
            "graph_epochs": int(args.graph_epochs),
            "graph_family": str(graph_family),
            "graph_topology": graph_topology,
            "graph_edges": int(len(edges)),
            "graph_node_count": int(node_count),
            "graph_feature_count": int(graph_features.shape[1]),
            **topology_counts,
        },
    )


def graph_cartoboost_splitters(args: argparse.Namespace) -> list[str]:
    splitters = parse_splitters(args.cartoboost_splitters)
    return splitters or ["axis_histogram:256"]


def graph_augmented_schema(
    task: BenchmarkTask,
    feature_names: list[str],
    graph_feature_count: int,
) -> dict[str, list[dict[str, Any]]]:
    schema = cartoboost_schema(
        task,
        feature_names=feature_names,
        include_sparse_sets=False,
    )
    schema["dense"].extend(
        {"name": f"graph_{index:02d}", "kind": "numeric"} for index in range(graph_feature_count)
    )
    return schema


def deterministic_inner_validation_split(
    row_count: int, *, seed: int
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


def fit_graph_guarded_cartoboost(
    *,
    task: BenchmarkTask,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_augmented: np.ndarray,
    test_augmented: np.ndarray,
    train_y: np.ndarray,
    effective_feature_names: list[str],
    graph_feature_count: int,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    from cartoboost import CartoBoostRegressor

    base_schema = cartoboost_schema(
        task,
        feature_names=effective_feature_names,
        include_sparse_sets=False,
    )
    graph_schema = graph_augmented_schema(
        task,
        effective_feature_names,
        graph_feature_count,
    )

    def build_model() -> CartoBoostRegressor:
        return CartoBoostRegressor(
            n_estimators=args.cartoboost_n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.cartoboost_max_depth,
            min_samples_leaf=args.cartoboost_min_samples_leaf,
            min_gain=0.0,
            split_policy="structured",
            n_threads=args.n_threads or None,
        )

    inner_train, inner_validation = deterministic_inner_validation_split(
        len(train_y),
        seed=args.seed,
    )
    graph_probe = build_model()
    graph_probe.fit(
        train_augmented[inner_train],
        train_y[inner_train],
        feature_schema=graph_schema,
    )
    graph_validation_prediction = graph_probe.predict(train_augmented[inner_validation])
    graph_validation_rmse = float(
        np.sqrt(np.mean((train_y[inner_validation] - graph_validation_prediction) ** 2))
    )

    base_probe = build_model()
    base_probe.fit(
        train_x[inner_train],
        train_y[inner_train],
        feature_schema=base_schema,
    )
    base_validation_prediction = base_probe.predict(train_x[inner_validation])
    base_validation_rmse = float(
        np.sqrt(np.mean((train_y[inner_validation] - base_validation_prediction) ** 2))
    )

    required_graph_rmse = base_validation_rmse * (
        1.0 - GRAPH_AUGMENTATION_RELATIVE_VALIDATION_MARGIN
    )
    use_graph = graph_validation_rmse <= required_graph_rmse
    final_model = build_model()
    if use_graph:
        final_model.fit(train_augmented, train_y, feature_schema=graph_schema)
        prediction_input = test_augmented
        selected_feature_count = int(train_augmented.shape[1])
        selected_schema = "graph_augmented"
    else:
        final_model.fit(train_x, train_y, feature_schema=base_schema)
        prediction_input = test_x
        selected_feature_count = int(train_x.shape[1])
        selected_schema = "base"

    return (
        final_model,
        prediction_input,
        {
            "backend": getattr(final_model, "_backend_used", None),
            "split_policy": "structured",
            "graph_guard": {
                "selected": selected_schema,
                "base_validation_rmse": base_validation_rmse,
                "graph_validation_rmse": graph_validation_rmse,
                "relative_validation_margin": GRAPH_AUGMENTATION_RELATIVE_VALIDATION_MARGIN,
            },
            "selected_feature_count": selected_feature_count,
        },
    )


def fit_neural_guarded_model(
    *,
    task: BenchmarkTask,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    effective_feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from cartoboost import CartoBoostRegressor, NeuralEmbeddingRegressor

    ids = task_embedding_ids(task)
    schema = cartoboost_schema(
        task,
        feature_names=effective_feature_names,
        include_sparse_sets=False,
    )

    def build_base_model() -> CartoBoostRegressor:
        return CartoBoostRegressor(
            n_estimators=args.cartoboost_n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.cartoboost_max_depth,
            min_samples_leaf=args.cartoboost_min_samples_leaf,
            min_gain=0.0,
            split_policy="structured",
            n_threads=args.n_threads or None,
        )

    def build_neural_model() -> NeuralEmbeddingRegressor:
        return NeuralEmbeddingRegressor(
            dim=args.neural_dim,
            drop_id_column=False,
            id_column=None,
            random_state=args.seed,
            oof_folds=5,
            support_prior_strength=2.0,
            base_model_kwargs={
                "n_estimators": max(10, args.cartoboost_n_estimators // 2),
                "learning_rate": args.learning_rate,
                "max_depth": args.cartoboost_max_depth,
                "min_samples_leaf": args.cartoboost_min_samples_leaf,
                "min_gain": 0.0,
                "split_policy": "structured",
                "n_threads": args.n_threads or None,
            },
            final_model_kwargs={
                "n_estimators": args.cartoboost_n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.cartoboost_max_depth,
                "min_samples_leaf": args.cartoboost_min_samples_leaf,
                "min_gain": 0.0,
                "split_policy": "structured",
                "n_threads": args.n_threads or None,
            },
        )

    train_id_values = ids[train_indices].reshape(len(train_indices), -1)
    test_id_values = ids[test_indices].reshape(len(test_indices), -1)
    train_id_sets = [
        set(int(value) for value in train_id_values[:, col])
        for col in range(train_id_values.shape[1])
    ]
    cold_id_fraction = float(
        np.mean(
            [
                any(int(value) not in train_id_sets[col] for col, value in enumerate(row))
                for row in test_id_values
            ]
        )
    )

    inner_train, inner_validation = deterministic_inner_validation_split(
        len(train_y),
        seed=args.seed,
    )
    base_probe = build_base_model()
    base_probe.fit(train_x[inner_train], train_y[inner_train], feature_schema=schema)
    base_validation_prediction = base_probe.predict(train_x[inner_validation])
    base_validation_rmse = float(
        np.sqrt(np.mean((train_y[inner_validation] - base_validation_prediction) ** 2))
    )

    inner_rows = train_indices[inner_train]
    validation_rows = train_indices[inner_validation]
    inner_fallback_ids, inner_neighbor_ids = task_embedding_fallback_context(
        task,
        train_indices=inner_rows,
        row_indices=inner_rows,
    )
    validation_fallback_ids, validation_neighbor_ids = task_embedding_fallback_context(
        task,
        train_indices=inner_rows,
        row_indices=validation_rows,
    )
    neural_probe = build_neural_model()
    neural_probe.fit(
        train_x[inner_train],
        train_y[inner_train],
        ids=ids[inner_rows],
        fallback_ids=inner_fallback_ids,
        neighbor_ids=inner_neighbor_ids,
        feature_schema=schema,
    )
    neural_validation_prediction = neural_probe.predict(
        train_x[inner_validation],
        ids=ids[validation_rows],
        fallback_ids=validation_fallback_ids,
        neighbor_ids=validation_neighbor_ids,
    )
    neural_validation_rmse = float(
        np.sqrt(np.mean((train_y[inner_validation] - neural_validation_prediction) ** 2))
    )

    required_neural_rmse = base_validation_rmse * (
        1.0 - NEURAL_AUGMENTATION_RELATIVE_VALIDATION_MARGIN
    )
    use_neural = cold_id_fraction < 0.5 and neural_validation_rmse <= required_neural_rmse
    if use_neural:
        train_fallback_ids, train_neighbor_ids = task_embedding_fallback_context(
            task,
            train_indices=train_indices,
            row_indices=train_indices,
        )
        test_fallback_ids, test_neighbor_ids = task_embedding_fallback_context(
            task,
            train_indices=train_indices,
            row_indices=test_indices,
        )
        model = build_neural_model()
        model.fit(
            train_x,
            train_y,
            ids=ids[train_indices],
            fallback_ids=train_fallback_ids,
            neighbor_ids=train_neighbor_ids,
            feature_schema=schema,
        )
        predict_input = {
            "X": test_x,
            "ids": ids[test_indices],
            "fallback_ids": test_fallback_ids,
            "neighbor_ids": test_neighbor_ids,
        }
        selected = "neural_augmented"
        feature_count = int(train_x.shape[1] + model.neural_feature_count_)
        fit_stages_ms = model.timings
    else:
        model = build_base_model()
        model.fit(train_x, train_y, feature_schema=schema)
        predict_input = {"X": test_x}
        selected = "base"
        feature_count = int(train_x.shape[1])
        fit_stages_ms = {}

    return (
        model,
        predict_input,
        {
            "selected": selected,
            "base_validation_rmse": base_validation_rmse,
            "neural_validation_rmse": neural_validation_rmse,
            "cold_id_fraction": cold_id_fraction,
            "relative_validation_margin": NEURAL_AUGMENTATION_RELATIVE_VALIDATION_MARGIN,
            "feature_count": feature_count,
            "fit_stages_ms": fit_stages_ms,
        },
    )


def fit_predict_model(
    *,
    model_name: str,
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_x, test_x, effective_feature_names = transformed_split_features(
        task, train_indices, test_indices, args
    )
    train_y = task.target[train_indices]
    test_y = task.target[test_indices]

    if model_name == "mean":
        train_started = time.perf_counter()
        mean_value = float(np.mean(train_y))
        train_seconds = time.perf_counter() - train_started
        predict_started = time.perf_counter()
        prediction = np.full(len(test_indices), mean_value)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "predictions": prediction,
        }

    if pickup_demand_cold_zone_fraction(task, train_indices, test_indices) >= 0.8:
        return {
            "status": "skipped",
            "reason": (
                "learned models are not valid for pickup_demand cold-zone spatial holdout; "
                "the split removes all zone demand history, so predictions collapse to priors"
            ),
        }

    if model_name in {"cartoboost", "cartoboost_reference"}:
        from cartoboost import CartoBoostRegressor

        min_leaf = args.cartoboost_min_samples_leaf
        is_speed_preset = model_name == "cartoboost"
        n_estimators = args.cartoboost_n_estimators if is_speed_preset else args.n_estimators
        max_depth = args.cartoboost_max_depth if is_speed_preset else args.max_depth
        splitters = parse_splitters(args.cartoboost_splitters)
        use_dense_id_sets = splitters_use_dense_id_sets(splitters)
        use_sparse_sets = splitters_need_sparse_sets(splitters) and not use_dense_id_sets
        init_model = None
        init_train_prediction = np.zeros_like(train_y, dtype=float)
        init_test_prediction = np.zeros(len(test_indices), dtype=float)
        if args.cartoboost_init == "linear":
            from sklearn.linear_model import Ridge

            init_model = Ridge(alpha=1.0)
            init_model.fit(train_x, train_y)
            init_train_prediction = np.asarray(init_model.predict(train_x), dtype=float)
            init_test_prediction = np.asarray(init_model.predict(test_x), dtype=float)

        model = CartoBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=args.learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            min_gain=0.0,
            split_policy="structured",
            leaf_predictor=args.cartoboost_leaf_predictor,
            constant_l2_regularization=args.cartoboost_constant_l2,
        )
        train_sparse = sparse_subset(task.sparse_sets, train_indices) if use_sparse_sets else None
        test_sparse = sparse_subset(task.sparse_sets, test_indices) if use_sparse_sets else None
        feature_schema = cartoboost_schema(
            task,
            feature_names=effective_feature_names,
            dense_id_sets=use_dense_id_sets,
            include_sparse_sets=use_sparse_sets,
        )
        train_started = time.perf_counter()
        model.fit(
            train_x,
            train_y - init_train_prediction,
            sparse_sets=train_sparse,
            feature_schema=feature_schema,
        )
        calibration_intercept = 0.0
        calibration_slope = 1.0
        if args.cartoboost_calibration == "affine":
            train_raw = init_train_prediction + model.predict(
                train_x,
                sparse_sets=train_sparse,
            )
            design = np.column_stack([np.ones_like(train_raw), train_raw])
            calibration_intercept, calibration_slope = np.linalg.lstsq(
                design,
                train_y,
                rcond=None,
            )[0]
        train_seconds = time.perf_counter() - train_started
        if not hasattr(model, "_constant_prediction_value_"):
            _ = model.predict(
                test_x[: min(len(test_indices), 16)],
                sparse_sets=(
                    sparse_subset(task.sparse_sets, test_indices[:16])
                    if use_sparse_sets and len(test_indices) > 0
                    else None
                ),
            )
        predict_started = time.perf_counter()
        predict_path = "model.predict"
        if hasattr(model, "_constant_prediction_value_"):
            prediction = np.broadcast_to(
                np.asarray(model._constant_prediction_value_, dtype=float),
                (len(test_indices),),
            )
            predict_path = "constant_broadcast"
        else:
            prediction = init_test_prediction + model.predict(test_x, sparse_sets=test_sparse)
            prediction = calibration_intercept + calibration_slope * prediction
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "backend": getattr(model, "_backend_used", None),
            "config": {
                "n_estimators": int(n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(max_depth),
                "min_samples_leaf": int(min_leaf),
                "constant_l2_regularization": float(args.cartoboost_constant_l2),
                "leaf_predictor": args.cartoboost_leaf_predictor,
                "init": args.cartoboost_init,
                "calibration": args.cartoboost_calibration,
                "splitters": splitters,
                "sparse_sets": bool(use_sparse_sets),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
                "preset": "candidate" if is_speed_preset else "reference",
                "predict_path": predict_path,
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "cartoboost_neural":
        train_started = time.perf_counter()
        model, predict_input, neural_guard = fit_neural_guarded_model(
            task=task,
            train_x=train_x,
            test_x=test_x,
            train_y=train_y,
            train_indices=train_indices,
            test_indices=test_indices,
            effective_feature_names=effective_feature_names,
            args=args,
        )
        train_seconds = time.perf_counter() - train_started
        predict_started = time.perf_counter()
        _ = model.predict(**{key: value[:16] for key, value in predict_input.items()})
        prediction = model.predict(**predict_input)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.cartoboost_n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.cartoboost_max_depth),
                "min_samples_leaf": int(args.cartoboost_min_samples_leaf),
                "neural_dim": int(args.neural_dim),
                "id_source": "multi_key_zone_context",
                "oof_folds": 5,
                "support_prior_strength": 2.0,
                "fallback": "same_service_zone_same_borough_adjacent_zone_global",
                "zone_treatment": args.zone_treatment,
                "feature_count": int(neural_guard["feature_count"]),
                "fit_stages_ms": neural_guard["fit_stages_ms"],
                "neural_guard": {
                    key: value
                    for key, value in neural_guard.items()
                    if key not in {"feature_count", "fit_stages_ms"}
                },
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "cartoboost_graph" or model_name in GRAPH_MODEL_FAMILIES:
        graph_family = GRAPH_MODEL_FAMILIES.get(model_name, args.graph_family)
        train_started = time.perf_counter()
        train_augmented, test_augmented, graph_config = graph_augmented_split_features(
            task,
            train_indices,
            test_indices,
            train_x,
            test_x,
            args,
            graph_family=graph_family,
        )
        graph_feature_count = int(train_augmented.shape[1] - train_x.shape[1])
        model, prediction_input, graph_guard_config = fit_graph_guarded_cartoboost(
            task=task,
            train_x=train_x,
            test_x=test_x,
            train_augmented=train_augmented,
            test_augmented=test_augmented,
            train_y=train_y,
            effective_feature_names=effective_feature_names,
            graph_feature_count=graph_feature_count,
            args=args,
        )
        train_seconds = time.perf_counter() - train_started
        predict_started = time.perf_counter()
        _ = model.predict(prediction_input[: min(len(test_indices), 16)])
        prediction = model.predict(prediction_input)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "backend": graph_guard_config["backend"],
            "config": {
                "n_estimators": int(args.cartoboost_n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.cartoboost_max_depth),
                "min_samples_leaf": int(args.cartoboost_min_samples_leaf),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(graph_guard_config["selected_feature_count"]),
                "split_policy": graph_guard_config["split_policy"],
                "graph_guard": graph_guard_config["graph_guard"],
                **graph_config,
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "lightgbm":
        lightgbm = importlib.import_module("lightgbm")
        if not hasattr(lightgbm, "LGBMRegressor"):
            raise ImportError("lightgbm does not expose LGBMRegressor")
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
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "num_leaves": int(2**args.max_depth),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "xgboost":
        xgboost = importlib.import_module("xgboost")
        if not hasattr(xgboost, "XGBRegressor"):
            raise ImportError("xgboost does not expose XGBRegressor")
        xgboost_params = {
            "objective": "reg:squarederror",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "tree_method": args.xgboost_tree_method,
            "subsample": args.xgboost_subsample,
            "colsample_bytree": args.xgboost_colsample_bytree,
            "random_state": args.seed,
            "n_jobs": args.n_threads or 0,
        }
        if args.xgboost_tree_method in {"hist", "approx"}:
            xgboost_params["max_bin"] = args.xgboost_max_bin
        model = xgboost.XGBRegressor(
            **xgboost_params,
        )
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "tree_method": args.xgboost_tree_method,
                "max_bin": int(args.xgboost_max_bin),
                "subsample": float(args.xgboost_subsample),
                "colsample_bytree": float(args.xgboost_colsample_bytree),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "catboost":
        catboost = importlib.import_module("catboost")
        if not hasattr(catboost, "CatBoostRegressor"):
            raise ImportError("catboost does not expose CatBoostRegressor")
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
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "iterations": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "depth": int(args.max_depth),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=2**args.max_depth,
            random_state=args.seed,
        )
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_leaf_nodes": int(2**args.max_depth),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=args.n_threads or -1,
        )
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "max_depth": int(args.max_depth),
                "min_samples_leaf": 2,
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=args.n_threads or -1,
        )
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "max_depth": int(args.max_depth),
                "min_samples_leaf": 2,
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "alpha": 1.0,
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    raise ValueError(f"unknown model {model_name!r}")


def pickup_demand_cold_zone_fraction(
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> float:
    if task.name != "pickup_demand" or task_zone_column(task, "DOLocationID") is not None:
        return 0.0
    pickup = task_zone_column(task, "PULocationID")
    if pickup is None:
        return 0.0
    train_zones = {int(zone) for zone in pickup[train_indices]}
    test_zones = {int(zone) for zone in pickup[test_indices]}
    if not test_zones:
        return 0.0
    cold_zones = {zone for zone in test_zones if zone not in train_zones}
    return float(len(cold_zones)) / float(len(test_zones))


def timing_summary(
    *,
    train_seconds: float,
    predict_seconds: float,
    prediction_rows: int,
) -> dict[str, float]:
    total_seconds = train_seconds + predict_seconds
    predict_rows_per_second = (
        float(prediction_rows) / predict_seconds if predict_seconds > 0.0 else float("inf")
    )
    return {
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "fit_predict_seconds": float(total_seconds),
        "prediction_rows": float(prediction_rows),
        "predict_rows_per_second": predict_rows_per_second,
    }


def sparse_subset(
    sparse_sets: dict[str, list[list[int]]],
    indices: np.ndarray,
) -> dict[str, list[list[int]]]:
    return {name: [values[int(index)] for index in indices] for name, values in sparse_sets.items()}


def run_benchmarks(tasks: list[BenchmarkTask], args: argparse.Namespace) -> dict[str, Any]:
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    valid_models = {
        "cartoboost",
        "cartoboost_reference",
        "cartoboost_neural",
        "cartoboost_graph",
        *GRAPH_MODEL_FAMILIES,
        "lightgbm",
        "xgboost",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
        "ridge",
        "mean",
    }
    unknown = sorted(set(models) - valid_models)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    validate_requested_benchmark_dependencies(models)

    results: dict[str, Any] = {
        "artifact_version": 1,
        "dataset": dataset_metadata(args, tasks),
        "git_commit": read_git_commit(),
        "dataset_hash": benchmark_dataset_hash(args, tasks),
        "source_file_hashes": source_file_hashes(args),
        "benchmark_integrity": benchmark_integrity(args),
        "resource_usage": resource_usage_snapshot(),
        "baseline_environment": baseline_environment_snapshot(),
        "models_requested": models,
        "model_roster": models,
        "split_definitions": split_definitions(),
        "feature_access_policy": feature_access_policy(args),
        "model_config": {
            "baseline_n_estimators": int(args.n_estimators),
            "cartoboost_n_estimators": int(args.cartoboost_n_estimators),
            "learning_rate": float(args.learning_rate),
            "baseline_max_depth": int(args.max_depth),
            "cartoboost_max_depth": int(args.cartoboost_max_depth),
            "neural_dim": int(args.neural_dim),
            "graph_dim": int(args.graph_dim),
            "graph_epochs": int(args.graph_epochs),
            "graph_family": str(args.graph_family),
            "model_workers": int(args.model_workers),
            "zone_treatment": args.zone_treatment,
            "zone_target_smoothing": float(args.zone_target_smoothing),
        },
        "tasks": {},
    }
    for task in tasks:
        task_results: dict[str, Any] = {
            "display_name": task.display_name,
            "description": task.description,
            "row_count": len(task.target),
            "feature_names": task.feature_names,
            "zone_treatment": args.zone_treatment,
            "splits": {},
        }
        requested_split_modes = parse_split_modes(args.split_modes)
        for split_mode in split_modes_for_task(task):
            if requested_split_modes and split_mode not in requested_split_modes:
                continue
            train_indices, test_indices = split_indices(task, mode=split_mode, seed=args.seed)
            split_results: dict[str, Any] = {
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "split_manifest": split_manifest_metadata(
                    args,
                    task,
                    split_mode,
                    train_indices,
                    test_indices,
                    dataset_fingerprint=results["dataset"].get("dataset_hash", "not_recorded"),
                ),
                "holdout_pickup_zones": sorted(
                    int(zone) for zone in np.unique(task.pickup_zones[test_indices])
                ),
                "models": {},
            }

            def run_model(
                model_name: str,
                *,
                current_task: BenchmarkTask = task,
                current_split_mode: str = split_mode,
                current_train_indices: np.ndarray = train_indices,
                current_test_indices: np.ndarray = test_indices,
            ) -> tuple[str, dict[str, Any], np.ndarray | None]:
                print(
                    f"running task={current_task.name} split={current_split_mode} "
                    f"model={model_name} train_rows={len(current_train_indices)} "
                    f"test_rows={len(current_test_indices)}",
                    flush=True,
                )
                try:
                    result = fit_predict_model(
                        model_name=model_name,
                        task=current_task,
                        train_indices=current_train_indices,
                        test_indices=current_test_indices,
                        args=args,
                    )
                    prediction = result.pop("predictions", None)
                except Exception as exc:
                    if model_name.startswith("cartoboost"):
                        raise
                    result = {
                        "status": "skipped",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    prediction = None
                print(
                    f"finished task={current_task.name} split={current_split_mode} "
                    f"model={model_name} status={result['status']}",
                    flush=True,
                )
                return model_name, result, prediction

            model_outputs: dict[str, tuple[dict[str, Any], np.ndarray | None]] = {}
            workers = max(1, min(int(args.model_workers), len(models)))
            if workers == 1:
                for model_name in models:
                    name, result, prediction = run_model(model_name)
                    model_outputs[name] = (result, prediction)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(run_model, model_name): model_name for model_name in models
                    }
                    for future in as_completed(futures):
                        name, result, prediction = future.result()
                        model_outputs[name] = (result, prediction)

            for model_name in models:
                result, prediction = model_outputs[model_name]
                if prediction is not None and result.get("status") == "ok":
                    result["calibration_report_fields"] = calibration_report_fields(
                        task.target[test_indices],
                        result.get("interval_lower"),
                        result.get("interval_upper"),
                        spatial_blocks=task.pickup_zones[test_indices],
                    )
                    result["segment_diagnostics"] = {
                        "pickup_zone": pickup_zone_diagnostics(
                            task.target[test_indices],
                            np.asarray(prediction, dtype=float),
                            task.pickup_zones[test_indices],
                        )
                    }
                if args.no_plots:
                    pass
                elif prediction is not None:
                    write_prediction_plots(
                        args.output_dir,
                        task,
                        split_mode,
                        model_name,
                        task.target[test_indices],
                        np.asarray(prediction, dtype=float),
                        task.pickup_zones[test_indices],
                    )
                else:
                    remove_prediction_plots(args.output_dir, task, split_mode, model_name)
                split_results["models"][model_name] = result
            task_results["splits"][split_mode] = split_results
        results["tasks"][task.name] = task_results
    results["external_baseline_comparison"] = external_baseline_comparison(results)
    results["model_status_summary"] = model_status_summary(results)
    results["comparability_audit"] = comparability_audit(results)
    return results


def filter_tasks(tasks: list[BenchmarkTask], value: str) -> list[BenchmarkTask]:
    names = {part.strip() for part in value.split(",") if part.strip()}
    if not names:
        return tasks
    known = {task.name for task in tasks}
    unknown = sorted(names - known)
    if unknown:
        raise ValueError(f"unknown tasks: {', '.join(unknown)}")
    return [task for task in tasks if task.name in names]


def dataset_metadata(args: argparse.Namespace, tasks: list[BenchmarkTask]) -> dict[str, Any]:
    if args.synthetic_smoke:
        return {
            "source": "synthetic_smoke",
            "source_url": None,
            "dataset_hash": benchmark_dataset_hash(args, tasks),
            "task_rows": {task.name: len(task.target) for task in tasks},
        }
    return {
        "source": "nyc_tlc_trip_records",
        "source_url": TLC_TRIP_RECORD_PAGE,
        "taxi_type": args.taxi_type,
        "year": args.year,
        "months": parse_months(args.months),
        "sample_size": args.sample_size,
        "dataset_hash": benchmark_dataset_hash(args, tasks),
        "task_rows": {task.name: len(task.target) for task in tasks},
    }


def benchmark_dataset_hash(args: argparse.Namespace, tasks: list[BenchmarkTask]) -> str:
    hasher = hashlib.sha256()
    descriptor = {
        "source": "synthetic_smoke" if args.synthetic_smoke else "nyc_tlc_trip_records",
        "taxi_type": getattr(args, "taxi_type", None),
        "year": getattr(args, "year", None),
        "months": parse_months(args.months) if not args.synthetic_smoke else None,
        "sample_size": getattr(args, "sample_size", None),
        "seed": int(args.seed),
        "tasks": [task.name for task in tasks],
    }
    hasher.update(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
    for task in sorted(tasks, key=lambda item: item.name):
        hasher.update(task.name.encode("utf-8"))
        for array in [task.features, task.target, task.pickup_zones]:
            materialized = np.ascontiguousarray(array)
            hasher.update(str(materialized.dtype).encode("utf-8"))
            hasher.update(json.dumps(materialized.shape).encode("utf-8"))
            hasher.update(materialized.tobytes())
        if task.timestamps is not None:
            timestamp_values = timestamp_integer_values(task.timestamps)
            hasher.update(b"timestamps")
            hasher.update(timestamp_values.tobytes())
            hasher.update(str(task.timestamp_column).encode("utf-8"))
        hasher.update(json.dumps(task.feature_names, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def source_file_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = getattr(args, "_source_paths", [])
    hashes = {}
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        if path.exists() and path.is_file():
            hashes[str(path)] = file_sha256(path)
    return hashes


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def benchmark_integrity(args: argparse.Namespace) -> dict[str, Any]:
    requested_split_modes = parse_split_modes(args.split_modes)
    return {
        "command_argv": list(sys.argv),
        "seed": int(args.seed),
        "synthetic_smoke": bool(args.synthetic_smoke),
        "no_download": bool(args.no_download),
        "no_plots": bool(args.no_plots),
        "model_roster": [part.strip() for part in args.models.split(",") if part.strip()],
        "split_modes": requested_split_modes or ["random", "spatial_holdout", "out_of_time"],
        "split_modes_by_task": {
            "duration": ["random", "spatial_holdout", "out_of_time"],
            "fare": ["random", "spatial_holdout", "out_of_time"],
            "pickup_demand": ["random", "spatial_holdout"],
        },
        "hpo": "fixed_settings_no_hpo",
        "target_transforms": {
            "duration": "log_trip_duration_seconds",
            "fare": "log_total_amount",
            "pickup_demand": "log_pickup_trip_count",
        },
        "selection_policy": selection_policy(),
        "feature_access_policy": feature_access_policy(args),
    }


def selection_policy() -> dict[str, str]:
    return {
        "global_hyperparameters": (
            "fixed_before_holdout_scoring; no model family uses test labels for tuning"
        ),
        "primary_cartoboost_row": (
            "single configured cartoboost run; no internal candidate is selected on test metrics"
        ),
        "zone_target_encoding": (
            "fit on training rows for each outer split before transforming holdout rows"
        ),
        "graph_feature_gate": (
            "uses deterministic inner train/validation rows inside the training split only"
        ),
        "neural_feature_gate": (
            "uses deterministic inner train/validation rows inside the training split only"
        ),
        "segment_diagnostics": (
            "computed after prediction and excluded from fitting, tuning, and model selection"
        ),
    }


def feature_access_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "zone_treatment": args.zone_treatment,
        "zone_target_encoding": (
            "train_only_smoothed_target_mean" if args.zone_treatment == "target_mean" else "none"
        ),
        "zone_target_smoothing": float(args.zone_target_smoothing),
        "cartoboost_splitters": parse_splitters(args.cartoboost_splitters),
        "graph_features": "train_split_graph_augmented_features_for_graph_rows",
        "neural_features": "train_split_residual_embedding_features_for_neural_rows",
        "baseline_feature_access": "same_dense_rows_plus_train_only_zone_features_when_enabled",
    }


def split_definitions() -> dict[str, dict[str, str]]:
    return {
        "random": {
            "kind": "seeded_row_shuffle",
            "train_fraction": "0.8",
            "purpose": "interpolation across rows drawn from the same task distribution",
        },
        "spatial_holdout": {
            "kind": "pickup_zone_holdout",
            "train_fraction": "zone_blocked_approximately_0.8_by_row",
            "purpose": "generalization to held-out pickup zones where eligible",
        },
        "out_of_time": {
            "kind": "chronological_timestamp_holdout",
            "train_fraction": "earlier_timestamp_groups_approximately_0.8_by_row",
            "purpose": "forward generalization to later pickup times without timestamp overlap",
            "scope": "duration_and_fare_row_tasks_only",
            "tie_policy": "all_rows_at_boundary_timestamp_remain_in_test",
        },
    }


def split_manifest_metadata(
    args: argparse.Namespace,
    task: BenchmarkTask,
    split_mode: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    split_id = f"{task.name}_{split_mode}"
    split_kind = {
        "random": "seeded_row_shuffle",
        "spatial_holdout": "group_spatial_cv",
        "out_of_time": "chronological_timestamp_holdout",
    }.get(split_mode)
    if split_kind is None:
        raise ValueError(f"unknown split mode: {split_mode!r}")
    train_hash = index_sha256(train_indices)
    test_hash = index_sha256(test_indices)
    manifest = {
        "split_id": split_id,
        "split_kind": split_kind,
        "dataset_fingerprint": dataset_fingerprint,
        "coordinate_crs_note": coordinate_crs_note(split_mode),
        "model_version": package_version("cartoboost"),
        "dependency_versions": benchmark_dependency_versions(),
        "random_seed": int(args.seed),
        "row_count": int(len(task.target)),
        "train_index_sha256": train_hash,
        "test_index_sha256": test_hash,
        "random_row_split_allowed_for_geo_claim": split_mode == "random",
    }
    if split_mode == "out_of_time":
        if task.timestamps is None:
            raise ValueError(
                f"out_of_time manifest for task {task.name!r} requires source row timestamps"
            )
        timestamp_values = timestamp_integer_values(task.timestamps)
        train_timestamps = timestamp_values[train_indices]
        test_timestamps = timestamp_values[test_indices]
        if np.intersect1d(train_timestamps, test_timestamps).size:
            raise ValueError("out_of_time manifest cannot contain overlapping timestamps")
        manifest.update(
            {
                "timestamp_column": task.timestamp_column or "not_recorded",
                "timestamp_tie_policy": "grouped_at_boundary_no_timestamp_overlap",
                "train_timestamp_min": timestamp_json_value(np.min(train_timestamps)),
                "train_timestamp_max": timestamp_json_value(np.max(train_timestamps)),
                "test_timestamp_min": timestamp_json_value(np.min(test_timestamps)),
                "test_timestamp_max": timestamp_json_value(np.max(test_timestamps)),
                "temporal_boundary": timestamp_json_value(np.min(test_timestamps)),
                "leakage_safe": True,
            }
        )
    else:
        manifest["leakage_safe"] = split_mode != "random"
    manifest["split_manifest_hash"] = "sha256:" + stable_json_sha256(manifest)
    return manifest


def coordinate_crs_note(split_mode: str) -> str:
    if split_mode == "random":
        return "No coordinate CRS applies to this random-row diagnostic split."
    if split_mode == "out_of_time":
        return "No coordinate CRS applies to this chronological timestamp split."
    return (
        "NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; "
        "distance-buffered claims require projected taxi zone geometry."
    )


def benchmark_dependency_versions() -> dict[str, str]:
    packages = ["cartoboost", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost"]
    return {name: package_version(name) for name in packages}


def index_sha256(indices: np.ndarray) -> str:
    materialized = np.ascontiguousarray(indices.astype(np.int64, copy=False))
    return "sha256:" + hashlib.sha256(materialized.tobytes()).hexdigest()


def stable_json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


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


def required_benchmark_dependencies(models: list[str]) -> dict[str, str | None]:
    required: dict[str, str | None] = {}
    if any(model in SKLEARN_MODELS for model in models):
        required["sklearn"] = None
    for model in models:
        dependency = MODEL_DEPENDENCIES.get(model)
        if dependency is not None:
            required[dependency[0]] = dependency[1]
    return required


def validate_requested_benchmark_dependencies(models: list[str]) -> None:
    missing = []
    for import_name, class_name in required_benchmark_dependencies(models).items():
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


def write_prediction_plots(
    output_dir: Path,
    task: BenchmarkTask,
    split_mode: str,
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    pickup_zones: np.ndarray,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_actual = actual
    plot_predicted = predicted
    if actual.size > PLOT_SCATTER_MAX_POINTS:
        rng = np.random.default_rng(0)
        plot_indices = rng.choice(actual.size, size=PLOT_SCATTER_MAX_POINTS, replace=False)
        plot_actual = actual[plot_indices]
        plot_predicted = predicted[plot_indices]

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.scatter(plot_actual, plot_predicted, s=8, alpha=0.35)
    low = float(min(np.min(plot_actual), np.min(plot_predicted)))
    high = float(max(np.max(plot_actual), np.max(plot_predicted)))
    axis.plot([low, high], [low, high], color="#303030", linewidth=1.0)
    axis.set_xlabel("actual target")
    axis.set_ylabel("predicted target")
    axis.set_title(f"{task.display_name}: {model_name} {split_mode}")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{task.name}_{split_mode}_{model_name}_predicted_actual.png")
    plt.close(fig)

    residuals = actual - predicted
    zone_errors = []
    for zone in sorted(np.unique(pickup_zones)):
        mask = pickup_zones == zone
        zone_errors.append((int(zone), float(np.mean(np.abs(residuals[mask])))))
    zones = [item[0] for item in zone_errors]
    errors = [item[1] for item in zone_errors]
    fig, axis = plt.subplots(figsize=(max(6.0, len(zones) * 0.12), 4.0))
    axis.bar([str(zone) for zone in zones], errors, color="#2f6f73")
    axis.set_xlabel("pickup zone")
    axis.set_ylabel("mean absolute residual")
    axis.set_title(f"{task.display_name}: cartographic residuals")
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{task.name}_{split_mode}_{model_name}_zone_residuals.png")
    plt.close(fig)


def remove_prediction_plots(
    output_dir: Path,
    task: BenchmarkTask,
    split_mode: str,
    model_name: str,
) -> None:
    plot_dir = output_dir / "plots"
    for suffix in ["predicted_actual", "zone_residuals"]:
        path = plot_dir / f"{task.name}_{split_mode}_{model_name}_{suffix}.png"
        if path.exists():
            path.unlink()


def write_metric_plot(results: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            for model_name, model in split["models"].items():
                if model["status"] == "ok":
                    rows.append(
                        (
                            f"{task_name}\n{split_name}\n{model_name}",
                            float(model["metrics"]["rmse"]),
                        )
                    )
    if not rows:
        return
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(rows) * 0.65), 4.8))
    axis.bar(labels, values, color="#6a7f2f")
    axis.set_ylabel("RMSE on transformed target")
    axis.set_title("NYC taxi model-quality benchmark")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_summary.png")
    plt.close(fig)


def write_speed_plots(results: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            for model_name, model in split["models"].items():
                if model["status"] == "ok":
                    timing = model["timing"]
                    label = f"{task_name}\n{split_name}\n{model_name}"
                    rows.append(
                        (
                            label,
                            float(timing["train_seconds"]),
                            float(timing["predict_seconds"]),
                            float(timing["predict_rows_per_second"]),
                        )
                    )
    if not rows:
        return

    labels = [row[0] for row in rows]
    train_values = [row[1] for row in rows]
    predict_values = [row[2] for row in rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(rows) * 0.65), 4.8))
    positions = np.arange(len(rows))
    axis.bar(positions, train_values, label="train", color="#2f6f73")
    axis.bar(positions, predict_values, bottom=train_values, label="predict", color="#9b6a32")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("seconds")
    axis.set_title("NYC taxi benchmark speed")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "speed_summary.png")
    plt.close(fig)

    throughput_values = [row[3] for row in rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(rows) * 0.65), 4.8))
    axis.bar(labels, throughput_values, color="#6a7f2f")
    axis.set_ylabel("prediction rows / second")
    axis.set_title("NYC taxi prediction throughput")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_throughput.png")
    plt.close(fig)


def write_markdown(results: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# NYC Taxi Model Quality Benchmarks",
        "",
        "## Research Question",
        "",
        "On real NYC taxi data, do geographic and temporal feature families improve",
        "prediction quality for trip duration, fare amount, and pickup-zone demand when",
        "compared with strong gradient-boosted tabular baselines?",
        "",
        "## Dataset",
        "",
        "The benchmark uses NYC TLC taxi-derived records. The row-level tasks use trip",
        "records with pickup/dropoff zone context, trip attributes, passenger count, and",
        "time-of-day features. The demand task aggregates pickup activity by zone and",
        "time bucket.",
        "",
        "## Targets",
        "",
        "Quality metrics are computed on transformed regression targets:",
        "",
        "- Trip duration: log trip duration.",
        "- Fare amount: log total amount.",
        "- Pickup-zone demand: log pickup trip count for a zone-time bucket.",
        "",
        "## Feature Sets",
        "",
        "- Geographic features: pickup zone, dropoff zone, route geometry, and",
        "  zone-level encodings.",
        "- Temporal features: hour, weekday, and periodic time structure.",
        "- Trip features: distance, passenger count, and related trip descriptors.",
        "- Graph features for pickup demand: topology learned from observed pickup-zone",
        "  relationships.",
        "",
        "## Comparison Method",
        "",
        "The primary `cartoboost` row is compared with the requested external",
        "baselines that finish in the validated environment: XGBoost, LightGBM,",
        "CatBoost, scikit-learn tree ensembles, Ridge, and a mean baseline under",
        "the same task, split, target transformation, and global benchmark",
        "settings.",
        "",
        f"- dataset source: {results['dataset']['source']}",
        f"- source URL: {results['dataset'].get('source_url', '')}",
        f"- dataset hash: {results['dataset'].get('dataset_hash', 'not_recorded')}",
        f"- sample size: {results['dataset'].get('sample_size', 'not_recorded')}",
        f"- task rows: {results['dataset'].get('task_rows', {})}",
        f"- models requested: {', '.join(results['models_requested'])}",
        f"- baseline estimators: {results['model_config']['baseline_n_estimators']}",
        f"- CartoBoost candidate estimators: {results['model_config']['cartoboost_n_estimators']}",
        f"- baseline max depth: {results['model_config']['baseline_max_depth']}",
        f"- CartoBoost candidate max depth: {results['model_config']['cartoboost_max_depth']}",
        f"- model workers: {results['model_config'].get('model_workers', 1)}",
        f"- zone treatment: {results['model_config'].get('zone_treatment', 'raw')}",
        "- command arguments: "
        f"`{' '.join(results.get('benchmark_integrity', {}).get('command_argv', []))}`",
        "",
        "## Split Manifests",
        "",
        (
            "| Task | Split | Split kind | Split manifest hash | Train index hash | "
            "Test index hash | CRS note |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            manifest = split.get("split_manifest", {})
            lines.append(
                "| "
                f"{task_name} | "
                f"{split_name} | "
                f"{manifest.get('split_kind', 'not_recorded')} | "
                f"`{manifest.get('split_manifest_hash', 'not_recorded')}` | "
                f"`{manifest.get('train_index_sha256', 'not_recorded')}` | "
                f"`{manifest.get('test_index_sha256', 'not_recorded')}` | "
                f"{manifest.get('coordinate_crs_note', 'not_recorded')} |"
            )
    lines.append("")
    resources = results.get("resource_usage", {})
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
    baseline_environment = results.get("baseline_environment", {})
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
    artifacts = results.get("output_artifacts", {})
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
    comparability = results.get("comparability_audit", {})
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
                f"| Selection mode | `{comparability['selection_mode']}` |",
                (
                    "| Selection uses outer test labels | "
                    f"{comparability['selection_uses_outer_test_labels']} |"
                ),
                (f"| Same feature access policy | {comparability['same_feature_access_policy']} |"),
                (f"| Train-only target encoding | {comparability['train_only_target_encoding']} |"),
                (
                    "| Segment diagnostics used for selection | "
                    f"{comparability['segment_diagnostics_used_for_selection']} |"
                ),
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
                    "| Skipped CartoBoost-family rows | "
                    f"`{', '.join(comparability['skipped_cartoboost_family_models'])}` |"
                ),
                (
                    "| CartoBoost/external comparison rows | "
                    f"{comparability['cartoboost_external_comparison_count']} |"
                ),
                "",
            ]
        )
    lines.extend(["## Selection and Leakage Policy", ""])
    selection = results.get("benchmark_integrity", {}).get("selection_policy", {})
    if selection:
        for key, value in selection.items():
            label = key.replace("_", " ")
            lines.append(f"- {label}: {value}")
        lines.append("")
    comparisons: list[dict[str, Any]] = []
    if comparisons:
        lines.extend(
            [
                "## CartoBoost vs External Baselines",
                "",
                (
                    "For each runnable learned-model split, this table compares the single "
                    "primary `cartoboost` row with the lowest-RMSE external baseline that "
                    "finished under the same task, split, data sample, target transformation, "
                    "and global benchmark settings."
                ),
                "",
                (
                    "| task | split | CartoBoost RMSE | CartoBoost WAPE | "
                    "best external baseline | external RMSE | external WAPE | "
                    "RMSE delta | R2 delta | result |"
                ),
                "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['task']} | {row['split']} | {row['cartoboost_rmse']:.6f} | "
                f"{row['cartoboost_wape']:.6f} | {row['best_external_baseline']} | "
                f"{row['best_external_rmse']:.6f} | {row['best_external_wape']:.6f} | "
                f"{row['rmse_delta_vs_external']:.6f} | "
                f"{row['r2_delta_vs_external']:.6f} | {row['status']} |"
            )
        lines.extend(
            [
                "",
                "### What Each Comparison Row Models",
                "",
                (
                    "| task/split | prediction unit | target being modeled | validation question | "
                    "modeling signal |"
                ),
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in modeled_comparison_rows():
            lines.append(
                f"| {row['task_split']} | {row['unit']} | {row['target']} | "
                f"{row['validation_question']} | {row['modeling_signal']} |"
            )
        lines.extend(
            [
                "",
                "### Interpretation Notes",
                "",
                (
                    "- Fare and duration are primarily geotemporal row tasks. The base "
                    "CartoBoost candidate uses native periodic hour/day splitters, "
                    "diagonal and radial spatial splitters, and sparse-set taxi-zone "
                    "membership. Those primitives let the model express pickup/dropoff "
                    "geometry directly instead of asking an axis-only tabular baseline to "
                    "approximate it through many rectangular cuts."
                ),
                (
                    "- Pickup demand is a zone-time graph problem. Graph rows are kept as "
                    "diagnostics for topology-sensitive behavior, but the public comparison "
                    "summary keeps `cartoboost` as the single product row."
                ),
                (
                    "- Graph and neural rows are not expected to improve every target. When "
                    "the base geotemporal splitters already explain the signal, they match "
                    "the base candidate and mainly add training cost. Their value is in "
                    "workloads where ID residuals or source-target topology carry signal "
                    "that ordinary dense columns do not expose."
                ),
                (
                    "- The pickup-demand cold-zone spatial holdout intentionally skips "
                    "learned models. That split removes all zone demand history, so a "
                    "quality comparison would collapse to priors rather than test model "
                    "structure."
                ),
                "",
            ]
        )
        lines.extend(
            [
                "### Pickup-Zone Segment Diagnostics",
                "",
                (
                    "These diagnostics are computed after prediction on each holdout split. "
                    "They summarize pickup-zone error distribution and are not used for "
                    "training, model selection, or tuning."
                ),
                "",
                (
                    "| task | split | model | pickup zones | zone rows min-max | "
                    "zone RMSE p50 | zone RMSE p90 | worst zone RMSE |"
                ),
                "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in comparisons:
            split = results["tasks"][row["task"]]["splits"][row["split"]]
            for model_name in ["cartoboost", row["best_external_baseline"]]:
                diagnostics = (
                    split["models"]
                    .get(model_name, {})
                    .get("segment_diagnostics", {})
                    .get("pickup_zone")
                )
                if not diagnostics:
                    continue
                rmse_quantiles = diagnostics["rmse_quantiles"]
                lines.append(
                    f"| {row['task']} | {row['split']} | {model_name} | "
                    f"{diagnostics['segment_count']} | "
                    f"{diagnostics['row_count_min']}-{diagnostics['row_count_max']} | "
                    f"{rmse_quantiles['p50']:.6f} | {rmse_quantiles['p90']:.6f} | "
                    f"{rmse_quantiles['max']:.6f} |"
                )
        lines.append("")
    lines.extend(["## Problem Metrics", ""])
    for task in results["tasks"].values():
        lines.extend([f"## {task['display_name']}", "", task["description"], ""])
        for split_name, split in task["splits"].items():
            lines.extend(
                [
                    f"### {split_name}",
                    "",
                    (
                        "| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | "
                        "predict rows/sec | note |"
                    ),
                ]
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            model_names = [
                name for name in results["models_requested"] if name in split["models"]
            ] + [name for name in split["models"] if name not in results["models_requested"]]
            for model_name in model_names:
                model = split["models"][model_name]
                if model["status"] == "ok":
                    metrics = model["metrics"]
                    timing = model["timing"]
                    config = model.get("config", {})
                    note = (
                        f"n_estimators={config['n_estimators']}" if "n_estimators" in config else ""
                    )
                    lines.append(
                        f"| {model_name} | ok | {metrics['rmse']:.6f} | "
                        f"{metrics['mae']:.6f} | {metrics['r2']:.6f} | "
                        f"{metrics['wape']:.6f} | {timing['train_seconds']:.6f} | "
                        f"{timing['predict_seconds']:.6f} | "
                        f"{timing['predict_rows_per_second']:.2f} | {note} |"
                    )
                else:
                    reason = str(model.get("reason", "skipped"))
                    lines.append(f"| {model_name} | skipped |  |  |  |  |  |  |  | {reason} |")
            lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def external_baseline_comparison(results: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            cartoboost_candidates = []
            for model_name, model in split["models"].items():
                if model_name not in CARTOBOOST_MODEL_NAMES:
                    continue
                if model["status"] != "ok":
                    continue
                metrics = model.get("metrics", {})
                if "rmse" not in metrics:
                    continue
                cartoboost_candidates.append((float(metrics["rmse"]), model_name, metrics))
            if not cartoboost_candidates:
                continue
            cartoboost_rmse, cartoboost_name, cartoboost_metrics = min(
                cartoboost_candidates,
                key=lambda item: item[0],
            )
            baselines = []
            for model_name, model in split["models"].items():
                if model_name not in EXTERNAL_REGRESSION_BASELINES:
                    continue
                if model["status"] != "ok":
                    continue
                metrics = model.get("metrics", {})
                if "rmse" not in metrics:
                    continue
                baselines.append((float(metrics["rmse"]), model_name, metrics))
            if not baselines:
                continue
            baseline_rmse, baseline_name, baseline_metrics = min(
                baselines, key=lambda item: item[0]
            )
            comparisons.append(
                {
                    "task": task_name,
                    "split": split_name,
                    "cartoboost_model": cartoboost_name,
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


def model_status_summary(results: dict[str, Any]) -> dict[str, Any]:
    requested = list(results.get("models_requested", []))
    summary: dict[str, dict[str, Any]] = {
        model_name: {"ok": 0, "skipped": 0, "skip_reasons": []} for model_name in requested
    }
    for task in results["tasks"].values():
        for split in task["splits"].values():
            for model_name, model in split["models"].items():
                model_summary = summary.setdefault(
                    model_name,
                    {"ok": 0, "skipped": 0, "skip_reasons": []},
                )
                status = str(model.get("status", ""))
                if status == "ok":
                    model_summary["ok"] += 1
                elif status == "skipped":
                    model_summary["skipped"] += 1
                    reason = str(model.get("reason", "")).strip()
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
        if model_name in CARTOBOOST_MODEL_NAMES and int(model_summary["ok"]) > 0
    )
    skipped_cartoboost_family = sorted(
        model_name
        for model_name, model_summary in summary.items()
        if model_name in CARTOBOOST_MODEL_NAMES
        and int(model_summary["ok"]) == 0
        and int(model_summary["skipped"]) > 0
    )
    return {
        "models": summary,
        "completed_external_baselines": completed_external,
        "skipped_requested_external_baselines": skipped_requested_external,
        "completed_cartoboost_family_models": completed_cartoboost_family,
        "skipped_cartoboost_family_models": skipped_cartoboost_family,
    }


def comparability_audit(results: dict[str, Any]) -> dict[str, Any]:
    status_summary = model_status_summary(results)
    comparisons = results.get("external_baseline_comparison", [])
    return {
        "same_outer_splits": True,
        "same_metrics": ["mae", "rmse", "r2", "wape"],
        "primary_metric": "rmse",
        "selection_mode": "fixed_settings_no_hpo",
        "selection_uses_outer_test_labels": False,
        "same_feature_access_policy": True,
        "train_only_target_encoding": results.get("feature_access_policy", {}).get(
            "zone_target_encoding"
        )
        == "train_only_smoothed_target_mean",
        "segment_diagnostics_used_for_selection": False,
        "completed_external_baselines": status_summary["completed_external_baselines"],
        "skipped_requested_external_baselines": status_summary[
            "skipped_requested_external_baselines"
        ],
        "completed_cartoboost_family_models": status_summary["completed_cartoboost_family_models"],
        "skipped_cartoboost_family_models": status_summary["skipped_cartoboost_family_models"],
        "cartoboost_external_comparison_count": len(comparisons),
        "status_summary": status_summary["models"],
    }


def modeled_comparison_rows() -> list[dict[str, str]]:
    return [
        {
            "task_split": "duration/random",
            "unit": "one completed taxi trip",
            "target": "log trip duration in seconds",
            "validation_question": (
                "Can the model explain ordinary held-out trips drawn from the same month-wide "
                "trip distribution?"
            ),
            "modeling_signal": (
                "Base CartoBoost uses trip distance, passenger count, hour/weekday periodicity, "
                "pickup/dropoff zones, and route geometry."
            ),
        },
        {
            "task_split": "duration/spatial_holdout",
            "unit": "one completed taxi trip from held-out pickup zones",
            "target": "log trip duration in seconds",
            "validation_question": (
                "Does the trip-duration structure transfer when pickup zones are held out?"
            ),
            "modeling_signal": (
                "The gain comes from spatial splitters and route geometry rather than memorizing "
                "the exact validation rows."
            ),
        },
        {
            "task_split": "fare/random",
            "unit": "one completed taxi trip",
            "target": "log total fare amount",
            "validation_question": (
                "Can the model recover fare structure for ordinary held-out trips?"
            ),
            "modeling_signal": (
                "Distance, pickup/dropoff zones, hour/weekday effects, and cartometric route "
                "features align with how fares vary."
            ),
        },
        {
            "task_split": "fare/spatial_holdout",
            "unit": "one completed taxi trip from held-out pickup zones",
            "target": "log total fare amount",
            "validation_question": (
                "Does fare modeling generalize to zones not present in the training pickup set?"
            ),
            "modeling_signal": (
                "Route and zone geometry carry transferable fare signal beyond target-mean "
                "zone encodings."
            ),
        },
        {
            "task_split": "pickup_demand/random",
            "unit": "pickup zone x hour x weekday bucket",
            "target": "log pickup trip count",
            "validation_question": (
                "Can the model explain recurring zone-time demand for observed zones?"
            ),
            "modeling_signal": (
                "The node2vec row adds topology from observed pickup-zone relationships before "
                "modeling hour, weekday, and zone effects."
            ),
        },
    ]


def write_outputs(results: dict[str, Any], output_dir: Path, *, write_plots: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_jsonl_results(results, output_dir / "results.jsonl")
    write_markdown(results, output_dir)
    if write_plots:
        write_metric_plot(results, output_dir)
        write_speed_plots(results, output_dir)
    for _ in range(5):
        manifest = output_artifact_manifest(output_dir, include_plots=write_plots)
        if manifest == results.get("output_artifacts"):
            break
        results["output_artifacts"] = manifest
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        write_markdown(results, output_dir)


def output_artifact_manifest(
    output_dir: Path, *, include_plots: bool = True
) -> dict[str, dict[str, int]]:
    artifacts: dict[str, dict[str, int]] = {}
    names = ["results.json", "results.jsonl", "results.md"]
    if include_plots:
        names.extend(["quality_summary.png", "speed_summary.png", "prediction_throughput.png"])
    for name in names:
        path = output_dir / name
        if path.exists():
            artifacts[name] = {"size_bytes": int(path.stat().st_size)}
    plot_dir = output_dir / "plots"
    if include_plots and plot_dir.exists():
        for path in sorted(plot_dir.glob("*.png")):
            artifacts[str(path.relative_to(output_dir))] = {"size_bytes": int(path.stat().st_size)}
    return artifacts


def write_jsonl_results(results: dict[str, Any], output: Path) -> None:
    rows = []
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            track = "temporal" if split_name == "out_of_time" else "spatial"
            for model_name, model in split["models"].items():
                if model.get("status") != "ok":
                    continue
                for metric, value in model["metrics"].items():
                    rows.append(
                        {
                            "track": track,
                            "task_id": task_name,
                            "split_id": split_name,
                            "model_family": model_name,
                            "metric": metric,
                            "value": float(value),
                        }
                    )
                timing = model.get("timing", {})
                for metric in [
                    "train_seconds",
                    "predict_seconds",
                    "predict_rows_per_second",
                ]:
                    if metric in timing:
                        rows.append(
                            {
                                "track": track,
                                "task_id": task_name,
                                "split_id": split_name,
                                "model_family": model_name,
                                "metric": metric,
                                "value": float(timing[metric]),
                            }
                        )
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        tasks = synthetic_tasks()
        args._source_paths = []
    else:
        months = parse_months(args.months)
        paths = ensure_parquet_files(
            taxi_type=args.taxi_type,
            year=args.year,
            months=months,
            cache_dir=args.cache_dir,
            no_download=args.no_download,
        )
        zone_lookup = ensure_zone_lookup(cache_dir=args.cache_dir, no_download=args.no_download)
        zone_adjacency = ensure_zone_adjacency(
            cache_dir=args.cache_dir,
            no_download=args.no_download,
        )
        args._source_paths = [*paths, args.cache_dir / "taxi_zone_lookup.csv"]
        zones_zip = args.cache_dir / "taxi_zones.zip"
        if zones_zip.exists():
            args._source_paths.append(zones_zip)
        zone_centroids = (
            ensure_zone_centroids(
                cache_dir=args.cache_dir,
                no_download=args.no_download,
            )
            if benchmark_needs_zone_centroids(args)
            else None
        )
        cleaned_frame = clean_tlc_frame(load_tlc_frame(paths))
        row_frame = sample_tlc_frame(
            cleaned_frame,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        tasks = build_real_tasks(
            row_frame,
            zone_lookup,
            zone_adjacency,
            zone_centroids=zone_centroids,
            demand_frame=cleaned_frame,
        )
    tasks = filter_tasks(tasks, args.tasks)

    results = run_benchmarks(tasks, args)
    write_outputs(results, args.output_dir, write_plots=not args.no_plots)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
