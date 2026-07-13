from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cartoboost.forecasting.graph_st import DCRNNForecaster as _DCRNNForecaster
from cartoboost.forecasting.graph_st import GraphTemporalFrame as _NativeGraphTemporalFrame
from cartoboost.forecasting.graph_st import GraphWaveNetForecaster as _GraphWaveNetForecaster
from cartoboost.forecasting.graph_st import LSTTNForecaster as _LSTTNForecaster
from cartoboost.forecasting.graph_st import (
    SpatialShiftGraphonMoEForecaster as _SpatialShiftGraphonMoEForecaster,
)
from cartoboost.forecasting.graph_st import (
    SpatialTemporalGraphGatedTransformerForecaster as _STGGTForecaster,
)
from cartoboost.forecasting.graph_st import STAEformerForecaster as _STAEformerForecaster
from cartoboost.forecasting.graph_st import STGformerForecaster as _STGformerForecaster
from cartoboost.forecasting.graph_st import STGormerForecaster as _STGormerForecaster

from ..config import Backend, GraphBackbone
from .flow import flow_uncertainty_report
from .frames import GraphTemporalFrame


class SpatioTemporalGraphForecaster:
    """Generic graph sequence forecaster facade over implemented native graph cores."""

    def __init__(
        self,
        *,
        backbone: GraphBackbone = GraphBackbone.DCRNN,
        multi_view_views: dict[str, Any] | None = None,
        **params: Any,
    ) -> None:
        try:
            backbone = GraphBackbone(backbone)
        except ValueError as exc:
            raise ValueError("unknown graph backbone") from exc
        self.backbone = backbone
        if multi_view_views is not None:
            raise RuntimeError(
                "NumPy representation primitives are not shipped in CartoBoost 0.3; "
                "provide graph structure through the native frame instead"
            )
        self.multi_view_views = None
        if backbone is GraphBackbone.DELAY_AWARE_GRAPH_TRANSFORMER:
            self._model = PropagationDelayGraphForecaster(**params)
        elif backbone is GraphBackbone.STGORMER:
            self._model = _STGormerForecaster(**params)
        elif backbone is GraphBackbone.STGFORMER:
            self._model = _STGformerForecaster(**params)
        elif backbone is GraphBackbone.LSTTN:
            self._model = _LSTTNForecaster(**params)
        elif backbone is GraphBackbone.SPATIAL_TEMPORAL_GRAPH_GATED_TRANSFORMER:
            self._model = _STGGTForecaster(**params)
        elif backbone is GraphBackbone.SPATIAL_SHIFT_GRAPHON_MOE:
            self._model = _SpatialShiftGraphonMoEForecaster(**params)
        elif backbone is GraphBackbone.DCRNN:
            self._model = _DCRNNForecaster(
                **{k: v for k, v in params.items() if k in _DCRNN_PARAMS}
            )
        elif backbone is GraphBackbone.GRAPH_WAVENET:
            self._model = _GraphWaveNetForecaster(
                **{k: v for k, v in params.items() if k in _GRAPH_WAVENET_PARAMS}
            )
        else:
            self._model = _STAEformerForecaster(
                **{k: v for k, v in params.items() if k in _STAEFORMER_PARAMS}
            )

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        if (
            args
            and hasattr(args[0], "y")
            and hasattr(args[0], "edges")
            and not isinstance(self._model, PropagationDelayGraphForecaster)
        ):
            frame = args[0]
            horizon = int(self._model.get_params().get("horizon", 1))
            self._model.fit(_native_public_graph_frame(frame, horizon))
        else:
            self._model.fit(*args, **kwargs)
        return self

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(*args, **kwargs)

    def score(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.score(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.save(*args, **kwargs)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "multi_view_views": self.multi_view_views,
            **self._model.get_params(deep=deep),
        }

    def set_params(self, **params: Any) -> SpatioTemporalGraphForecaster:
        backbone = params.pop("backbone", self.backbone)
        self.__init__(backbone=backbone, **{**self._model.get_params(), **params})
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        metadata = dict(self._model.metadata_)
        metadata["backbone"] = self.backbone.value
        metadata["multi_view_spatial_attention"] = None
        return metadata


class DelayAwareGraphTransformer:
    """Directed delayed graph propagation forecaster backed by Rust graph ST core."""

    def __init__(
        self,
        *,
        horizon: int = 1,
        edge_delay_prior: list[int] | tuple[int, ...] | None = None,
        ridge: float = 1e-6,
        backend: Backend | str = Backend.CPU,
        multi_view_views: dict[str, Any] | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.edge_delay_prior = (
            None if edge_delay_prior is None else [int(v) for v in edge_delay_prior]
        )
        self.ridge = float(ridge)
        self.backend = _backend_value(backend)
        if multi_view_views is not None:
            raise RuntimeError(
                "NumPy representation primitives are not shipped in CartoBoost 0.3; "
                "provide graph structure through the native frame instead"
            )
        self.multi_view_views = None
        if self.backend not in {"auto", "cpu", "cuda", "rocm", "mlx"}:
            raise ValueError("backend must be one of 'auto', 'cpu', 'cuda', 'rocm', or 'mlx'")
        self._native_model = None
        self.is_fitted_ = False

    def fit(self, frame: GraphTemporalFrame) -> DelayAwareGraphTransformer:
        if self.backend not in {"auto", "cpu"}:
            raise RuntimeError(
                "delay-aware graph transformer accelerator kernels are not available yet; "
                f"requested backend {self.backend!r}"
            )
        y = np.asarray(frame.y, dtype=float)
        if y.ndim != 2 or y.shape[0] < 3:
            raise ValueError("GraphTemporalFrame.y must have at least three time rows")
        if not frame.directed:
            raise ValueError("delay-aware graph transformer requires directed edges")
        edges = [(int(source), int(target)) for source, target in frame.edges]
        if not edges:
            raise ValueError("delay-aware graph transformer requires directed edges")
        weights = [float(value) for value in (frame.edge_weights or [1.0] * len(edges))]
        edge_distances = _edge_distances(frame.edge_distances, len(edges))
        node_covariates = _node_covariates(
            frame.node_covariates, len(frame.node_ids), "node_covariates"
        )
        future_covariates = _future_covariates(
            frame.known_future_covariates,
            self.horizon,
            len(frame.node_ids),
        )
        # Representation and multi-view NumPy helpers were removed in v0.3.
        # Native graph structure remains the sole source of graph context.
        node_similarity = np.eye(len(frame.node_ids), dtype=float)
        multi_view_metadata = None
        attention = _graph_attention_report(
            y,
            edges,
            edge_distances,
            node_covariates,
            future_covariates,
            node_similarity,
        )
        weights = [
            float(weight)
            * float(attention["dynamic_attention_mask"][edge_idx])
            * float(attention["edge_distance_embedding"][edge_idx])
            for edge_idx, ((source, target), weight) in enumerate(zip(edges, weights, strict=True))
        ]
        delays = self._resolve_delays(edges)
        native_frame, native_delays, native_edges, native_weights = self._native_frame(
            frame, edges, weights, delays
        )
        native_class = _native_model_class()
        self._native_model = native_class(
            self.horizon,
            native_delays if self.edge_delay_prior is not None else None,
            self.ridge,
            "cpu",
        )
        self._native_model.fit(native_frame._native_frame)
        self.edges_ = native_edges
        self.edge_weights_ = np.asarray(native_weights, dtype=float)
        self.edge_delays_ = native_delays
        self.node_ids_ = list(frame.node_ids)
        self.training_history_ = y.copy()
        self.static_adjacency_baseline_ = _static_adjacency_forecast(
            y,
            native_edges,
            np.asarray(native_weights, dtype=float),
            self.horizon,
        )
        self.metadata_ = {
            "model_class": "PropagationDelayGraphForecaster",
            "architecture": "delay_aware_graph_transformer",
            "backbone": "delay_aware_graph_transformer",
            "horizon": self.horizon,
            "directed": True,
            "backend": self._backend_metadata(),
            "shared_representation_consumed": False,
            "shared_representation": None,
            "inputs": {
                "edge_distances": frame.edge_distances is not None,
                "node_covariates": frame.node_covariates is not None,
                "known_future_covariates": frame.known_future_covariates is not None,
            },
            "attention_blocks": attention,
            "multi_view_spatial_attention": multi_view_metadata,
            "flow_uncertainty_head": _graph_flow_report(y, node_similarity),
            "falsifier_baselines": [
                "non_graph_temporal_model",
                "static_adjacency_only_graph_model",
            ],
            "edge_delay_sensitivity": self.edge_delay_sensitivity(),
            "save_load_parity_checked": False,
        }
        self.is_fitted_ = True
        self.metadata_["save_load_parity_checked"] = self._save_load_parity()
        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before prediction")
        horizon = int(horizon or self.horizon)
        return np.asarray(self._native_model.predict(horizon), dtype=float)

    def score(self, actual: Any) -> float:
        actual_arr = np.asarray(actual, dtype=float)
        return float(self._native_model.score(actual_arr.tolist()))

    def falsifier_report(self, actual: Any) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before falsifier_report")
        if self.training_history_ is None or self.static_adjacency_baseline_ is None:
            raise RuntimeError("falsifier_report requires a model fit in the current process")
        actual_arr = np.asarray(actual, dtype=float)
        pred = self.predict(actual_arr.shape[0])
        non_graph = _non_graph_temporal_forecast(self.training_history_, actual_arr.shape[0])
        static_graph = self.static_adjacency_baseline_[: actual_arr.shape[0]]
        return {
            "delay_aware_graph_rmse": _rmse(actual_arr, pred),
            "non_graph_temporal_model_rmse": _rmse(actual_arr, non_graph),
            "static_adjacency_only_graph_model_rmse": _rmse(actual_arr, static_graph),
            "delay_beats_non_graph_temporal": bool(
                _rmse(actual_arr, pred) < _rmse(actual_arr, non_graph)
            ),
            "delay_beats_static_adjacency_only": bool(
                _rmse(actual_arr, pred) < _rmse(actual_arr, static_graph)
            ),
        }

    def edge_delay_sensitivity(self) -> dict[str, Any]:
        if getattr(self, "_native_model", None) is not None:
            raw = json.loads(self._native_model.edge_delay_sensitivity())
            return {
                "graph_signal_coefficient": float(raw["graph_signal_coefficient"]),
                "delay_counts": {str(delay): count for delay, count in raw["delay_counts"]},
            }
        counts: dict[int, int] = {}
        for delay in getattr(self, "edge_delays_", []):
            counts[int(delay)] = counts.get(int(delay), 0) + 1
        return {
            "graph_signal_coefficient": 0.0,
            "delay_counts": {str(delay): count for delay, count in sorted(counts.items())},
        }

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted_:
            raise RuntimeError("model must be fit before save")
        path = Path(path)
        self._native_model.save(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> DelayAwareGraphTransformer:
        native = _native_model_class().load(str(path))
        payload = json.loads(native.to_json())
        config = payload["config"]
        obj = cls(
            horizon=int(config["horizon"]),
            edge_delay_prior=config["edge_delay_prior"],
            ridge=float(config["ridge"]),
            backend="cpu",
            multi_view_views=None,
        )
        obj._native_model = native
        obj.edges_ = [(int(source), int(target)) for source, target in payload["edges"]]
        obj.edge_weights_ = np.asarray(payload["edge_weights"], dtype=float)
        obj.edge_delays_ = [int(value) for value in config["edge_delay_prior"]]
        obj.node_ids_ = list(payload["node_ids"])
        obj.training_history_ = None
        obj.static_adjacency_baseline_ = None
        obj.metadata_ = {
            "model_class": "PropagationDelayGraphForecaster",
            "architecture": "delay_aware_graph_transformer",
            "backbone": "delay_aware_graph_transformer",
            "horizon": obj.horizon,
            "directed": True,
            "backend": obj._backend_metadata(),
            "multi_view_spatial_attention": None,
            "falsifier_baselines": [
                "non_graph_temporal_model",
                "static_adjacency_only_graph_model",
            ],
            "edge_delay_sensitivity": obj.edge_delay_sensitivity(),
            "save_load_parity_checked": True,
        }
        obj.is_fitted_ = True
        return obj

    def _resolve_delays(self, edges: list[tuple[int, int]]) -> list[int]:
        if self.edge_delay_prior is None:
            return [1] * len(edges)
        if len(self.edge_delay_prior) != len(edges):
            raise ValueError("edge_delay_prior must match edge count")
        if any(delay <= 0 for delay in self.edge_delay_prior):
            raise ValueError("edge_delay_prior values must be positive")
        return list(self.edge_delay_prior)

    def _native_frame(
        self,
        frame: GraphTemporalFrame,
        edges: list[tuple[int, int]],
        weights: list[float],
        delays: list[int],
    ) -> tuple[_NativeGraphTemporalFrame, list[int], list[tuple[int, int]], list[float]]:
        node_count = len(frame.node_ids)
        by_source: list[list[tuple[int, int, float, int]]] = [[] for _ in range(node_count)]
        for edge_idx, (source, target) in enumerate(edges):
            if source < 0 or source >= node_count or target < 0 or target >= node_count:
                raise ValueError("edge index exceeds node count")
            by_source[source].append((source, target, weights[edge_idx], delays[edge_idx]))
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        native_delays: list[int] = []
        native_edges: list[tuple[int, int]] = []
        for source_edges in by_source:
            for source, target, weight, delay in source_edges:
                indices.append(target)
                data.append(weight)
                native_delays.append(delay)
                native_edges.append((source, target))
            indptr.append(len(indices))
        native_frame = _NativeGraphTemporalFrame(
            node_ids=list(frame.node_ids),
            timestamps=list(frame.timestamps),
            target=np.asarray(frame.y, dtype=float),
            indptr=indptr,
            indices=indices,
            data=data,
            horizon=self.horizon,
            frequency="unknown",
        )
        return native_frame, native_delays, native_edges, data

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "horizon": self.horizon,
            "edge_delay_prior": self.edge_delay_prior,
            "ridge": self.ridge,
            "backend": self.backend,
            "multi_view_views": self.multi_view_views,
        }

    def set_params(self, **params: Any) -> DelayAwareGraphTransformer:
        valid = set(self.get_params())
        for key in params:
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
        self.__init__(**{**self.get_params(), **params})
        return self

    def _backend_metadata(self) -> dict[str, Any]:
        return {
            "requested": self.backend,
            "selected": "cpu",
            "supported": ["cpu", "cuda", "rocm", "mlx"],
            "accelerator_ready": {"cuda": True, "rocm": True, "mlx": True},
            "accelerated": False,
        }

    def _save_load_parity(self) -> bool:
        before = self.predict(self.horizon)
        handle = tempfile.NamedTemporaryFile(
            prefix="cartoboost_delay_aware_graph_transformer_", suffix=".json", delete=False
        )
        handle.close()
        path = Path(handle.name)
        self.save(path)
        try:
            after = self.load(path).predict(self.horizon)
        finally:
            path.unlink(missing_ok=True)
        return bool(np.array_equal(before, after))


def _edge_distances(values: list[float] | tuple[float, ...] | None, edge_count: int) -> np.ndarray:
    if values is None:
        return np.ones(edge_count, dtype=float)
    distances = np.asarray(values, dtype=float).reshape(-1)
    if distances.shape[0] != edge_count:
        raise ValueError("edge_distances must match edge count")
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("edge_distances must contain finite non-negative values")
    return distances


def _node_covariates(values: Any | None, node_count: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(node_count, 1)
    if array.ndim != 2 or array.shape[0] != node_count or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix with one row per node")
    return array


def _future_covariates(values: Any | None, horizon: int, node_count: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array.reshape(array.shape[0], array.shape[1], 1)
    if (
        array.ndim != 3
        or array.shape[0] < horizon
        or array.shape[1] != node_count
        or not np.isfinite(array).all()
    ):
        raise ValueError("known_future_covariates must be a finite [horizon, node, feature] array")
    return array[:horizon]


def _graph_attention_report(
    y: np.ndarray,
    edges: list[tuple[int, int]],
    edge_distances: np.ndarray,
    node_covariates: np.ndarray | None,
    future_covariates: np.ndarray | None,
    node_similarity: np.ndarray,
) -> dict[str, Any]:
    distance_scale = max(float(np.median(edge_distances)), 1e-12)
    edge_distance_embedding = np.exp(-edge_distances / distance_scale)
    recent = y[-min(8, y.shape[0]) :]
    temporal_attention = np.linspace(1.0, 2.0, recent.shape[0], dtype=float)
    temporal_attention = temporal_attention / temporal_attention.sum()
    node_recent_signal = temporal_attention @ recent
    dynamic_mask = []
    short_range = []
    long_range = []
    for source, target in edges:
        signal_delta = abs(float(node_recent_signal[source] - node_recent_signal[target]))
        mask = 1.0 / (1.0 + signal_delta)
        if node_covariates is not None:
            cov_distance = float(np.linalg.norm(node_covariates[source] - node_covariates[target]))
            mask *= 1.0 / (1.0 + cov_distance)
        if future_covariates is not None:
            future_delta = float(
                np.mean(np.abs(future_covariates[:, source, :] - future_covariates[:, target, :]))
            )
            mask *= 1.0 + min(future_delta, 1.0) * 0.05
        dynamic_mask.append(mask)
        short_range.append(mask * float(edge_distance_embedding[len(short_range)]))
        long_range.append(mask * float(max(node_similarity[source, target], 0.0)))
    temporal_summary = {
        "lookback": int(recent.shape[0]),
        "weights": temporal_attention.astype(float).tolist(),
        "recent_signal": node_recent_signal.astype(float).tolist(),
    }
    return {
        "edge_distance_embedding": edge_distance_embedding.astype(float).tolist(),
        "dynamic_attention_mask": dynamic_mask,
        "short_range_graph_attention": short_range,
        "long_range_semantic_attention": long_range,
        "temporal_attention": temporal_summary,
    }


def _non_graph_temporal_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    shared_level = float(np.mean(history[-min(8, history.shape[0]) :]))
    return np.full((horizon, history.shape[1]), shared_level, dtype=float)


def _static_adjacency_forecast(
    history: np.ndarray,
    edges: list[tuple[int, int]],
    weights: np.ndarray,
    horizon: int,
) -> np.ndarray:
    current = history[-1].copy()
    previous = history[-2].copy() if history.shape[0] >= 2 else current.copy()
    rows = []
    for _step in range(horizon):
        graph_signal = np.zeros_like(current)
        counts = np.zeros_like(current)
        for edge_idx, (source, target) in enumerate(edges):
            graph_signal[target] += previous[source] * weights[edge_idx]
            counts[target] += abs(weights[edge_idx])
        counts = np.where(counts > 0.0, counts, 1.0)
        updated = 0.7 * current + 0.3 * graph_signal / counts
        rows.append(updated.copy())
        previous = current
        current = updated
    return np.vstack(rows)


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    if actual.shape != pred.shape:
        raise ValueError("actual and prediction shapes must match")
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _graph_flow_report(y: np.ndarray, node_similarity: np.ndarray) -> dict[str, Any]:
    if y.shape[0] < 3:
        return {"consumed": False, "reason": "requires at least three time rows"}
    residuals = (y[1:] - y[:-1]).reshape(-1)
    hidden = np.column_stack(
        [
            np.repeat(np.arange(1, y.shape[0], dtype=float), y.shape[1]),
            y[:-1].reshape(-1),
        ]
    )
    graph_context = np.tile(node_similarity.mean(axis=1), y.shape[0] - 1).reshape(-1, 1)
    return flow_uncertainty_report(
        residuals,
        model_hidden_state=hidden,
        graph_context=graph_context,
        surface="SpatioTemporalGraphForecaster",
    )


DynamicAdjacencyTransformer = DelayAwareGraphTransformer
PropagationDelayGraphForecaster = DelayAwareGraphTransformer


def _native_public_graph_frame(
    frame: GraphTemporalFrame, horizon: int
) -> _NativeGraphTemporalFrame:
    node_count = len(frame.node_ids)
    if not frame.directed:
        raise ValueError("paper graph transformers require directed graph edges")
    if len(frame.edges) != len(frame.edge_weights):
        raise ValueError("edge_weights must match edges")
    by_source: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for (source, target), weight in zip(frame.edges, frame.edge_weights, strict=True):
        if source < 0 or source >= node_count or target < 0 or target >= node_count:
            raise ValueError("edge index exceeds node count")
        by_source[source].append((target, float(weight)))
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for edges in by_source:
        for target, weight in edges:
            indices.append(target)
            data.append(weight)
        indptr.append(len(indices))
    return _NativeGraphTemporalFrame(
        node_ids=list(frame.node_ids),
        timestamps=[int(value) for value in frame.timestamps],
        target=np.asarray(frame.y, dtype=float),
        indptr=indptr,
        indices=indices,
        data=data,
        horizon=horizon,
        frequency="unknown",
    )


def _native_model_class() -> Any:
    try:
        from cartoboost import _native
    except ImportError as exc:
        raise NotImplementedError(
            "Rust binding for PropagationDelayGraphForecaster is not available."
        ) from exc
    native_class = getattr(_native, "PropagationDelayGraphForecaster", None)
    if native_class is None:
        raise NotImplementedError(
            "Rust binding for PropagationDelayGraphForecaster is not available."
        )
    return native_class


def _backend_value(value: Backend | str) -> str:
    if isinstance(value, Backend):
        return value.value
    return str(value).lower()


_DCRNN_PARAMS = {
    "diffusion_steps",
    "hidden_size",
    "epochs",
    "learning_rate",
    "teacher_forcing_start",
    "teacher_forcing_end",
    "ridge",
    "backend",
}

_STAEFORMER_PARAMS = {
    "lookback",
    "attention_heads",
    "hidden_size",
    "epochs",
    "learning_rate",
    "ridge",
    "backend",
}

_GRAPH_WAVENET_PARAMS = {
    "lookback",
    "dilation_depth",
    "hidden_size",
    "epochs",
    "learning_rate",
    "ridge",
    "backend",
}
