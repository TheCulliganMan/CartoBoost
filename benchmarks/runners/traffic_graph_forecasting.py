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
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from cartoboost.forecasting import (
    DCRNNForecaster,
    GraphTemporalFrame,
    GraphWaveNetForecaster,
    LSTTNForecaster,
    SpatialShiftGraphonMoEForecaster,
    SpatialTemporalGraphGatedTransformerForecaster,
    STAEformerForecaster,
    STGformerForecaster,
    STGormerForecaster,
)

PROFILE_MODELS = {
    "dcrnn": DCRNNForecaster,
    "graph_wavenet": GraphWaveNetForecaster,
    "staeformer": STAEformerForecaster,
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


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("models must be a non-empty comma-separated list")
    if models == ["all"]:
        return sorted(PROFILE_MODELS)
    unknown = sorted(set(models) - set(PROFILE_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported graph forecast models: {', '.join(unknown)}")
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("models must not contain duplicates")
    return models


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


def model_lookback(args: argparse.Namespace, model_name: str) -> int:
    if model_name == "dcrnn":
        return 0
    return args.lsttn_lookback if model_name == "lsttn" else args.lookback


def backend_execution_scope(model_name: str, selected_backend: str) -> str:
    if selected_backend == "cpu":
        return "cpu_training_and_inference"
    if model_name == "lsttn":
        return f"{selected_backend}_full_graph_training_and_inference_cpu_orchestration"
    return f"{selected_backend}_forecast_head_cpu_feature_graph_and_training"


def build_model(args: argparse.Namespace, model_name: str) -> Any:
    model_type = PROFILE_MODELS[model_name]
    if model_name == "dcrnn":
        return model_type(
            diffusion_steps=args.graph_order,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            backend=args.backend,
        )
    if model_name == "graph_wavenet":
        return model_type(
            lookback=args.lookback,
            dilation_depth=args.dilation_depth,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            backend=args.backend,
        )
    if model_name == "staeformer":
        return model_type(
            lookback=args.lookback,
            attention_heads=args.attention_heads,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            backend=args.backend,
        )
    if model_name == "lsttn" and args.lsttn_lookback < args.periodicity * 14:
        raise ValueError(
            "LSTTN evaluation requires the paper's two-week long-history context: "
            "lsttn_lookback must be at least periodicity * 14"
        )
    lookback = model_lookback(args, model_name)
    return model_type(
        lookback=lookback,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        graph_order=args.graph_order,
        experts=args.experts,
        periodicity=args.periodicity,
        recent_window=args.recent_window,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        horizon=args.horizon,
        backend=args.backend,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = list(args.models)
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
    source_time_rows = int(values.shape[0])
    source_nodes = int(values.shape[1])
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
        required_context = max(model_lookback(args, model) for model in models)
        if cutoff <= required_context + args.horizon:
            raise ValueError(
                f"cutoff {cutoff} is too early for required context {required_context}"
            )
        train = frame.train_slice(cutoff)
        actual = values[cutoff : cutoff + args.horizon]
        for model_name in models:
            print(
                f"[{model_name}] fitting cutoff={cutoff} rows={cutoff} "
                f"nodes={values.shape[1]} backend={args.backend}",
                file=sys.stderr,
                flush=True,
            )
            model = build_model(args, model_name)
            fit_start = time.perf_counter()
            model.fit(train)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = np.asarray(model.predict(args.horizon), dtype=float)
            predict_seconds = time.perf_counter() - predict_start
            if prediction.shape != actual.shape or not np.isfinite(prediction).all():
                raise RuntimeError("native graph model returned invalid forecast output")
            model_metadata = model.metadata_
            selected_backend = str(model_metadata["backend"])
            row_metrics = metrics(actual, prediction)
            rows.append(
                {
                    "model": model_name,
                    "cutoff": cutoff,
                    "backend": selected_backend,
                    "backend_execution_scope": backend_execution_scope(
                        model_name, selected_backend
                    ),
                    **row_metrics,
                    "fit_wallclock_seconds": fit_seconds,
                    "predict_wallclock_seconds": predict_seconds,
                }
            )
            print(
                f"[{model_name}] rmse={row_metrics['rmse']:.6f} "
                f"mae={row_metrics['mae']:.6f} r2={row_metrics['r2']:.6f} "
                f"fit={fit_seconds:.3f}s predict={predict_seconds:.3f}s",
                file=sys.stderr,
                flush=True,
            )
    return {
        "artifact_type": "cartoboost.traffic_graph_forecasting_evaluation",
        "dataset": {
            "name": "user_supplied_traffic_graph",
            "source_time_rows": source_time_rows,
            "source_time_rows_before_preprocessing": (args.source_time_rows_before_preprocessing),
            "source_nodes": source_nodes,
            "evaluated_nodes": len(frame.node_ids),
            "directed_edges": len(data),
        },
        "task": {
            "target": "traffic_speed",
            "frequency": args.frequency,
            "forecast_horizon": args.horizon,
        },
        "split": {
            "kind": "fixed_origin_temporal_holdout",
            "origins": args.cutoffs,
            "train_rows_per_origin": args.cutoffs,
            "test_rows_per_origin": args.horizon,
        },
        "source_url": args.source_url,
        "source_artifact_sha256": args.source_artifact_sha256,
        "preprocessing": args.preprocessing,
        "traffic_values_path": str(values_path),
        "traffic_values_sha256": sha256(values_path),
        "traffic_values_format": values_format,
        "adjacency_path": str(adjacency_path),
        "adjacency_sha256": sha256(adjacency_path),
        "adjacency_format": adjacency_format,
        "model": models[0] if len(models) == 1 else "multi_model_comparison",
        "models": models,
        "settings": {
            "backend": args.backend,
            "frequency": args.frequency,
            "lookback": args.lookback,
            "lsttn_lookback": args.lsttn_lookback,
            "horizon": args.horizon,
            "hidden_size": args.hidden_size,
            "attention_heads": args.attention_heads,
            "graph_order": args.graph_order,
            "experts": args.experts,
            "periodicity": args.periodicity,
            "recent_window": args.recent_window,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dilation_depth": args.dilation_depth,
            "target_feature": args.target_feature,
            "nodes": args.nodes,
            "cutoffs": args.cutoffs,
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
    parser.add_argument("--source-artifact-sha256")
    parser.add_argument("--source-time-rows-before-preprocessing", type=int)
    parser.add_argument("--preprocessing", default="none")
    model_selection = parser.add_mutually_exclusive_group(required=True)
    model_selection.add_argument("--model", choices=sorted(PROFILE_MODELS))
    model_selection.add_argument("--models", type=parse_models)
    parser.add_argument("--cutoffs", type=parse_cutoffs, required=True)
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--target-feature", type=int, default=0)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--lsttn-lookback", type=int, default=4032)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--graph-order", type=int, default=2)
    parser.add_argument("--dilation-depth", type=int, default=3)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--periodicity", type=int, default=288)
    parser.add_argument("--recent-window", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--backend", choices=["auto", "cpu", "cuda", "rocm", "metal"], default="cpu"
    )
    args = parser.parse_args(argv)
    args.models = [args.model] if args.model is not None else args.models
    result = run(args)
    result["invocation"] = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
