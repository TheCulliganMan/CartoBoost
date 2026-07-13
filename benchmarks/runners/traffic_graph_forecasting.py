"""Run fixed-origin native traffic graph forecasting evaluations on real inputs.

The runner intentionally does not download data, synthesize missing sensors, or
fill absent graph edges.  Supply the original traffic HDF5 series and adjacency
pickle explicitly, then preserve the generated JSON artifact with its source
hashes and exact origins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from cartoboost.preview.forecasting import (
    DCRNNForecaster,
    GraphTemporalFrame,
    LSTTNForecaster,
    SpatialShiftGraphonMoEForecaster,
    SpatialTemporalGraphGatedTransformerForecaster,
    STGformerForecaster,
    STGormerForecaster,
)


PROFILE_MODELS = {
    "dcrnn": DCRNNForecaster,
    "stgormer": STGormerForecaster,
    "stgformer": STGformerForecaster,
    "lsttn": LSTTNForecaster,
    "spatial_temporal_graph_gated_transformer": SpatialTemporalGraphGatedTransformerForecaster,
    "spatial_shift_graphon_moe": SpatialShiftGraphonMoEForecaster,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hdf_values(path: Path) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional real-data dependency
        raise RuntimeError(
            "traffic graph evaluation requires h5py; run this command through "
            "`uv run --with h5py -- python -m benchmarks.runners.traffic_graph_forecasting`"
        ) from exc
    with h5py.File(path, "r") as handle:
        try:
            values = np.asarray(handle["df/block0_values"], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "expected the DCRNN-style HDF5 dataframe values at df/block0_values"
            ) from exc
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("traffic HDF5 values must be a non-empty [time, node] matrix")
    if not np.isfinite(values).all():
        raise ValueError("traffic HDF5 values must be finite")
    return values


def read_npy_values(path: Path, target_feature: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("traffic NPY values must have shape [time, node, feature]")
    if target_feature < 0 or target_feature >= values.shape[2]:
        raise ValueError(
            f"target_feature {target_feature} is outside the available feature range "
            f"[0, {values.shape[2]})"
        )
    target = np.asarray(values[:, :, target_feature], dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("selected traffic NPY feature must be finite")
    return target


def read_adjacency(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, (list, tuple)) or len(payload) < 3:
        raise ValueError("expected a DCRNN adjacency pickle with matrix in element 2")
    adjacency = np.asarray(payload[2], dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0.0):
        raise ValueError("adjacency values must be finite and non-negative")
    return adjacency


def read_npy_adjacency(path: Path) -> np.ndarray:
    adjacency = np.asarray(np.load(path, mmap_mode="r"), dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("NPY adjacency matrix must be square")
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0.0):
        raise ValueError("NPY adjacency values must be finite and non-negative")
    return adjacency


def adjacency_csr(adjacency: np.ndarray) -> tuple[list[int], list[int], list[float]]:
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row in adjacency:
        for target, weight in enumerate(row):
            if weight != 0.0:
                indices.append(target)
                data.append(float(weight))
        indptr.append(len(indices))
    if not data:
        raise ValueError("adjacency must contain at least one non-zero edge")
    return indptr, indices, data


def parse_cutoffs(value: str) -> list[int]:
    try:
        cutoffs = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cutoffs must be comma-separated integers") from exc
    if not cutoffs or cutoffs != sorted(set(cutoffs)):
        raise argparse.ArgumentTypeError("cutoffs must be non-empty, unique, and sorted")
    return cutoffs


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    squared = float(np.sum(residual * residual))
    total = int(actual.size)
    centered = actual - float(np.mean(actual))
    denominator = float(np.sum(centered * centered))
    return {
        "rmse": float(np.sqrt(squared / total)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - squared / denominator if denominator > 0.0 else float("nan"),
    }


def build_model(args: argparse.Namespace) -> Any:
    model_type = PROFILE_MODELS[args.model]
    if args.model == "dcrnn":
        return model_type(
            diffusion_steps=args.graph_order,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            backend="cpu",
        )
    if args.model == "lsttn" and args.lookback < args.periodicity * 14:
        raise ValueError(
            "LSTTN evaluation requires the paper's two-week long-history context: "
            "lookback must be at least periodicity * 14"
        )
    return model_type(
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        graph_order=args.graph_order,
        experts=args.experts,
        periodicity=args.periodicity,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        horizon=args.horizon,
        backend="cpu",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_h5 is not None:
        values = read_hdf_values(args.data_h5)
        values_path = args.data_h5
        values_format = "dcrnn_hdf5_dataframe"
    else:
        values = read_npy_values(args.node_values_npy, args.target_feature)
        values_path = args.node_values_npy
        values_format = "time_node_feature_npy"
    if args.adjacency_pickle is not None:
        adjacency = read_adjacency(args.adjacency_pickle)
        adjacency_path = args.adjacency_pickle
        adjacency_format = "dcrnn_adjacency_pickle"
    else:
        adjacency = read_npy_adjacency(args.adjacency_npy)
        adjacency_path = args.adjacency_npy
        adjacency_format = "dense_adjacency_npy"
    if values.shape[1] != adjacency.shape[0]:
        raise ValueError(
            f"series has {values.shape[1]} nodes but adjacency has {adjacency.shape[0]} nodes"
        )
    if args.nodes is not None:
        if args.nodes <= 0 or args.nodes > values.shape[1]:
            raise ValueError("nodes must be between 1 and the source node count")
        values = values[:, : args.nodes]
        adjacency = adjacency[: args.nodes, : args.nodes]
    indptr, indices, data = adjacency_csr(adjacency)
    frame = GraphTemporalFrame(
        node_ids=[f"sensor_{index}" for index in range(values.shape[1])],
        timestamps=list(range(values.shape[0])),
        target=values,
        indptr=indptr,
        indices=indices,
        data=data,
        horizon=args.horizon,
        frequency=args.frequency,
    )
    rows = []
    for cutoff in args.cutoffs:
        if cutoff <= args.horizon or cutoff + args.horizon > values.shape[0]:
            raise ValueError(f"cutoff {cutoff} does not leave a full holdout horizon")
        if args.model != "dcrnn" and cutoff <= args.lookback + args.horizon:
            raise ValueError(f"cutoff {cutoff} is too early for lookback {args.lookback}")
        train = frame.train_slice(cutoff)
        actual = values[cutoff : cutoff + args.horizon]
        model = build_model(args)
        fit_start = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        prediction = np.asarray(model.predict(args.horizon), dtype=float)
        predict_seconds = time.perf_counter() - predict_start
        if prediction.shape != actual.shape or not np.isfinite(prediction).all():
            raise RuntimeError("native graph model returned invalid forecast output")
        rows.append(
            {
                "cutoff": cutoff,
                **metrics(actual, prediction),
                "fit_wallclock_seconds": fit_seconds,
                "predict_wallclock_seconds": predict_seconds,
            }
        )
    return {
        "artifact_type": "cartoboost.traffic_graph_forecasting_evaluation",
        "source_url": args.source_url,
        "traffic_values_path": str(values_path),
        "traffic_values_sha256": sha256(values_path),
        "traffic_values_format": values_format,
        "adjacency_path": str(adjacency_path),
        "adjacency_sha256": sha256(adjacency_path),
        "adjacency_format": adjacency_format,
        "model": args.model,
        "settings": {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "hidden_size": args.hidden_size,
            "attention_heads": args.attention_heads,
            "graph_order": args.graph_order,
            "experts": args.experts,
            "periodicity": args.periodicity,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "target_feature": args.target_feature,
        },
        "origins": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--data-h5", type=Path)
    data_source.add_argument("--node-values-npy", type=Path)
    adjacency_source = parser.add_mutually_exclusive_group(required=True)
    adjacency_source.add_argument("--adjacency-pickle", type=Path)
    adjacency_source.add_argument("--adjacency-npy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--model", choices=sorted(PROFILE_MODELS), required=True)
    parser.add_argument("--cutoffs", type=parse_cutoffs, required=True)
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--target-feature", type=int, default=0)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--graph-order", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--periodicity", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    args = parser.parse_args(argv)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
