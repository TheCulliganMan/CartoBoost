from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._artifacts import ArtifactPersistenceMixin
from ..config import Backend, ChoiceStrEnum


class GraphTemporalFrame:
    """Graph time-series frame backed by the Rust graph forecasting core."""

    def __init__(
        self,
        *,
        node_ids: list[str],
        timestamps: list[int],
        target: Any,
        indptr: list[int],
        indices: list[int],
        data: list[float],
        horizon: int,
        frequency: str,
        covariates: Any | None = None,
        owner_mask: list[bool] | None = None,
        target_mask: Any | None = None,
        imputed_mask: Any | None = None,
        target_weights: Any | None = None,
        covariate_roles: list[str] | None = None,
    ) -> None:
        native_class = _native_class("GraphTemporalFrame")
        if native_class is None:
            raise NotImplementedError("Rust binding for GraphTemporalFrame is not available.")
        target_rows = np.ascontiguousarray(np.asarray(target, dtype=float))
        covariate_rows = (
            None
            if covariates is None
            else np.ascontiguousarray(np.asarray(covariates, dtype=float))
        )
        self._node_ids = list(map(str, node_ids))
        self._timestamps = list(map(int, timestamps))
        self._target = target_rows
        self._indptr = list(map(int, indptr))
        self._indices = list(map(int, indices))
        self._data = list(map(float, data))
        self._horizon = int(horizon)
        self._frequency = str(frequency)
        self._covariates = covariate_rows
        self._owner_mask = None if owner_mask is None else list(map(bool, owner_mask))
        self._target_mask = (
            None
            if target_mask is None
            else np.ascontiguousarray(np.asarray(target_mask, dtype=bool))
        )
        self._imputed_mask = (
            None
            if imputed_mask is None
            else np.ascontiguousarray(np.asarray(imputed_mask, dtype=bool))
        )
        self._target_weights = (
            None
            if target_weights is None
            else np.ascontiguousarray(np.asarray(target_weights, dtype=float))
        )
        self._covariate_roles = None if covariate_roles is None else list(map(str, covariate_roles))
        native_args = (
            self._node_ids,
            self._timestamps,
            self._target,
            self._indptr,
            self._indices,
            self._data,
            self._horizon,
            self._frequency,
            covariate_rows,
            self._owner_mask,
            self._target_mask,
            self._imputed_mask,
            self._target_weights,
            self._covariate_roles,
        )
        if not hasattr(native_class, "from_numpy"):
            raise RuntimeError(
                "CartoBoost native extension is incompatible: "
                "GraphTemporalFrame.from_numpy is required"
            )
        self._native_frame = native_class.from_numpy(*native_args)

    @property
    def node_ids(self) -> list[str]:
        return list(self._native_frame.node_ids)

    @property
    def horizon(self) -> int:
        return int(self._native_frame.horizon)

    @property
    def frequency(self) -> str:
        return str(self._native_frame.frequency)

    def train_slice(self, size: int) -> GraphTemporalFrame:
        if size <= 0 or size > len(self._target):
            raise ValueError("size must be positive and within the frame length")
        covariates = None if self._covariates is None else self._covariates[:size]
        return GraphTemporalFrame(
            node_ids=self._node_ids,
            timestamps=self._timestamps[:size],
            target=self._target[:size],
            indptr=self._indptr,
            indices=self._indices,
            data=self._data,
            horizon=self._horizon,
            frequency=self._frequency,
            covariates=covariates,
            owner_mask=self._owner_mask,
            target_mask=None if self._target_mask is None else self._target_mask[:size],
            imputed_mask=None if self._imputed_mask is None else self._imputed_mask[:size],
            target_weights=None if self._target_weights is None else self._target_weights[:size],
            covariate_roles=self._covariate_roles,
        )

    def splitter_data(self) -> dict[str, list[int]]:
        return {"timestamp": list(self._timestamps)}


class MarketPanelFrame:
    """Generic directional market panel for native structure learning.

    ``target_names`` names the caller-selected primary and secondary measures;
    CartoBoost does not assign business semantics to either target.
    """

    def __init__(
        self,
        *,
        lane_ids: list[str],
        timestamps: list[int],
        target_names: tuple[str, str] | list[str],
        primary: Any,
        secondary: Any,
        origin_ids: list[str],
        destination_ids: list[str],
        coordinates: Any,
        hierarchy_groups: list[list[str]] | None = None,
        calendar: Any | None = None,
        mix: Any | None = None,
        expert_priors: list[dict[str, Any]] | None = None,
        expert_labels: list[dict[str, Any]] | None = None,
        horizon: int = 1,
        frequency: str = "daily",
    ) -> None:
        native_class = _native_class("MarketPanelFrame")
        if native_class is None:
            raise NotImplementedError("Rust binding for MarketPanelFrame is not available.")
        primary_rows = np.asarray(primary, dtype=float).tolist()
        secondary_rows = np.asarray(secondary, dtype=float).tolist()
        if (
            np.asarray(primary, dtype=float).ndim != 2
            or np.asarray(secondary, dtype=float).ndim != 2
        ):
            raise ValueError("primary and secondary must be two-dimensional time by lane matrices")
        coordinate_rows = np.asarray(coordinates, dtype=float).tolist()
        calendar_rows = (
            np.zeros((len(timestamps), 0), dtype=float).tolist()
            if calendar is None
            else np.asarray(calendar, dtype=float).tolist()
        )
        mix_rows = None if mix is None else np.asarray(mix, dtype=float).tolist()
        parent_groups = (
            [[] for _ in lane_ids]
            if hierarchy_groups is None
            else [list(map(str, groups)) for groups in hierarchy_groups]
        )
        self._native_frame = native_class(
            list(map(str, lane_ids)),
            list(map(int, timestamps)),
            list(map(str, target_names)),
            primary_rows,
            secondary_rows,
            list(map(str, origin_ids)),
            list(map(str, destination_ids)),
            coordinate_rows,
            calendar_rows,
            parent_groups,
            mix_rows,
            json.dumps(expert_priors or []),
            json.dumps(expert_labels or []),
            int(horizon),
            str(frequency),
        )
        # Keep the caller-supplied panel unchanged so it can be handed to a
        # Rust graph forecaster.  This is data conversion only; the graph
        # model and every learned relationship remain native.
        self._timestamps = list(map(int, timestamps))
        self._primary = primary_rows
        self._secondary = secondary_rows
        self._horizon = int(horizon)
        self._frequency = str(frequency)

    @property
    def lane_ids(self) -> list[str]:
        return list(self._native_frame.lane_ids)

    @property
    def target_names(self) -> list[str]:
        return list(self._native_frame.target_names)

    def as_graph_temporal_frame(
        self,
        *,
        indptr: list[int],
        indices: list[int],
        data: list[float],
        target: str = "primary",
        covariates: Any | None = None,
    ) -> GraphTemporalFrame:
        """Create a native graph frame for a named observed market target.

        The adjacency is intentionally explicit: this adapter never infers,
        densifies, or fabricates a market graph.  Graph forecasters require a
        complete numeric target matrix, so a panel with unobserved values is
        rejected rather than imputed.
        """
        if target == "primary":
            values = self._primary
        elif target == "secondary":
            values = self._secondary
        else:
            raise ValueError("target must be 'primary' or 'secondary'")
        target_values = np.asarray(values, dtype=float)
        if not np.isfinite(target_values).all():
            raise ValueError(
                "market graph forecasting requires complete observed target values; "
                "remove unavailable lanes or provide an explicitly observed panel"
            )
        return GraphTemporalFrame(
            node_ids=self.lane_ids,
            timestamps=self._timestamps,
            target=target_values,
            indptr=indptr,
            indices=indices,
            data=data,
            horizon=self._horizon,
            frequency=self._frequency,
            covariates=covariates,
        )


class MarketStructureForecaster(ArtifactPersistenceMixin):
    """Sparse, explainable, time-aware smoothing for two named market targets."""

    def __init__(
        self,
        *,
        top_k: int = 8,
        neural_hidden_dim: int = 16,
        neural_epochs: int = 20,
        head_epochs: int = 80,
        head_learning_rate: float = 0.02,
        huber_delta: float = 1.0,
        quantile_levels: list[float] | tuple[float, ...] = (0.1, 0.5, 0.9),
        graph_strength: float = 0.55,
        local_strength: float = 0.35,
        correlation_floor: float = 0.10,
        shift_zscore: float = 2.0,
        calibrate_intervals: bool = True,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        native_class = _native_class("MarketStructureForecaster")
        if native_class is None:
            raise NotImplementedError(
                "Rust binding for MarketStructureForecaster is not available."
            )
        self._params = {
            "top_k": int(top_k),
            "neural_hidden_dim": int(neural_hidden_dim),
            "neural_epochs": int(neural_epochs),
            "head_epochs": int(head_epochs),
            "head_learning_rate": float(head_learning_rate),
            "huber_delta": float(huber_delta),
            "quantile_levels": list(map(float, quantile_levels)),
            "graph_strength": float(graph_strength),
            "local_strength": float(local_strength),
            "correlation_floor": float(correlation_floor),
            "shift_zscore": float(shift_zscore),
            "calibrate_intervals": bool(calibrate_intervals),
            "backend": _choice_value(backend),
        }
        self._native_model = native_class(**self._params)
        self.is_fitted_ = False

    def fit(self, frame: MarketPanelFrame) -> MarketStructureForecaster:
        native = getattr(frame, "_native_frame", None)
        if native is None:
            raise TypeError("expected a MarketPanelFrame")
        self._native_model.fit(native)
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int, *, future_calendar: Any | None = None) -> list[dict[str, Any]]:
        self._check_is_fitted()
        calendar = (
            None if future_calendar is None else np.asarray(future_calendar, dtype=float).tolist()
        )
        return list(json.loads(self._native_model.predict_json(int(horizon), calendar)))

    def nowcast(self) -> list[dict[str, Any]]:
        self._check_is_fitted()
        return list(json.loads(self._native_model.nowcast_json()))

    def weekly_rollups(
        self, horizon: int, *, future_calendar: Any | None = None
    ) -> list[dict[str, Any]]:
        """Aggregate daily native forecasts into calendar-week rows."""
        self._check_is_fitted()
        calendar = (
            None if future_calendar is None else np.asarray(future_calendar, dtype=float).tolist()
        )
        return list(json.loads(self._native_model.weekly_rollups_json(int(horizon), calendar)))

    def relationships(self) -> list[dict[str, Any]]:
        self._check_is_fitted()
        return list(json.loads(self._native_model.relationships_json()))

    def explorer_payload(self, horizon: int = 7) -> dict[str, Any]:
        """Return portable lanes, forecasts, explanations, and learned kernels."""
        self._check_is_fitted()
        return dict(json.loads(self._native_model.explorer_json(int(horizon))))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        self._native_model.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> MarketStructureForecaster:
        native_class = _native_class("MarketStructureForecaster")
        if native_class is None:
            raise NotImplementedError(
                "Rust binding for MarketStructureForecaster is not available."
            )
        obj = cls.__new__(cls)
        obj._native_model = native_class.load(str(path))
        obj._params = {"backend": obj.selected_backend_}
        obj.is_fitted_ = True
        return obj

    def to_json(self) -> str:
        self._check_is_fitted()
        return str(self._native_model.to_json())

    @classmethod
    def from_json(cls, value: str) -> MarketStructureForecaster:
        native_class = _native_class("MarketStructureForecaster")
        if native_class is None:
            raise NotImplementedError(
                "Rust binding for MarketStructureForecaster is not available."
            )
        obj = cls.__new__(cls)
        obj._native_model = native_class.from_json(value)
        obj._params = {"backend": obj.selected_backend_}
        obj.is_fitted_ = True
        return obj

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    @property
    def selected_backend_(self) -> str:
        backend = getattr(self._native_model, "backend", None)
        return str(backend()) if callable(backend) else str(self._params.get("backend", "cpu"))

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("MarketStructureForecaster must be fit before prediction")


class DCRNNForecaster(ArtifactPersistenceMixin):
    """Rust DCRNN-style graph sequence forecaster."""

    native_class_name = "DCRNNForecaster"

    def __init__(
        self,
        *,
        diffusion_steps: int = 2,
        hidden_size: int = 8,
        epochs: int = 160,
        learning_rate: float = 0.03,
        teacher_forcing_start: float = 1.0,
        teacher_forcing_end: float = 0.2,
        ridge: float = 0.0001,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        native_class = _native_class(self.native_class_name)
        if native_class is None:
            raise NotImplementedError("Rust binding for DCRNNForecaster is not available.")
        self._params = {
            "diffusion_steps": int(diffusion_steps),
            "hidden_size": int(hidden_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "teacher_forcing_start": float(teacher_forcing_start),
            "teacher_forcing_end": float(teacher_forcing_end),
            "ridge": float(ridge),
            "backend": _choice_value(backend),
        }
        self._native_model = native_class(
            self._params["diffusion_steps"],
            self._params["hidden_size"],
            self._params["epochs"],
            self._params["learning_rate"],
            self._params["teacher_forcing_start"],
            self._params["teacher_forcing_end"],
            self._params["ridge"],
            self._params["backend"],
        )
        self.is_fitted_ = False
        self._fit_frame: GraphTemporalFrame | None = None

    def fit(self, frame: GraphTemporalFrame) -> DCRNNForecaster:
        self._native_model.fit(_native_frame(frame))
        self.is_fitted_ = True
        self._fit_frame = frame
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self._native_model.predict(int(horizon)), dtype=float)

    def score(self, actual: Any, *, horizon: int | None = None) -> float:
        actual_arr = np.asarray(actual, dtype=float)
        if actual_arr.ndim != 2:
            raise ValueError("actual must be a two-dimensional horizon by node matrix")
        pred = self.predict(int(horizon or actual_arr.shape[0]))
        if pred.shape != actual_arr.shape:
            raise ValueError("prediction and actual must have the same shape")
        err = actual_arr - pred
        return float(np.sqrt(np.mean(err * err)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    def set_params(self, **params: Any) -> DCRNNForecaster:
        valid = set(self._params)
        for key in params:
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
        self.__init__(**{**self._params, **params})
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": self.__class__.__name__,
            "params": dict(self._params),
            "backend": self._backend_selected(),
            "fitted": self.is_fitted_,
        }

    def backtest(
        self,
        splitter: Any | None = None,
        *,
        frame: GraphTemporalFrame | None = None,
        train_size: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(splitter, GraphTemporalFrame):
            frame = splitter
            splitter = None
        frame = frame or self._fit_frame
        if frame is None:
            raise ValueError("backtest requires a GraphTemporalFrame or a previously fitted frame")
        if train_size is not None:
            return dict(
                json.loads(self._native_model.backtest(_native_frame(frame), int(train_size)))
            )
        if splitter is None:
            raise ValueError("backtest requires either train_size or a splitter")
        fold_results = []
        split_data = (
            frame.splitter_data()
            if getattr(splitter, "timestamp_col", None) is not None
            else np.asarray(frame._timestamps)
        )
        for fold in splitter.split(split_data):
            train_count = len(fold.train_indices)
            fitted = self._clone_unfit()
            fitted.fit(frame.train_slice(train_count))
            predictions = fitted.predict(fold.horizon)
            actual = np.asarray(
                frame._target[train_count : train_count + fold.horizon],
                dtype=float,
            )
            fold_results.append(
                {
                    "fold_id": fold.fold_id,
                    "train_size": train_count,
                    "validation_size": int(len(fold.validation_indices)),
                    "horizon": int(fold.horizon),
                    "by_horizon": _horizon_metrics(predictions, actual),
                }
            )
        return {"folds": fold_results}

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        self._native_model.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> DCRNNForecaster:
        native_class = _native_class(cls.native_class_name)
        if native_class is None:
            raise NotImplementedError("Rust binding for DCRNNForecaster is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.load(str(path))
        obj.is_fitted_ = True
        obj._fit_frame = None
        obj._params = {"backend": obj._backend_selected()}
        return obj

    def to_json(self) -> str:
        self._check_is_fitted()
        return str(self._native_model.to_json())

    @classmethod
    def from_json(cls, value: str) -> DCRNNForecaster:
        native_class = _native_class(cls.native_class_name)
        if native_class is None:
            raise NotImplementedError("Rust binding for DCRNNForecaster is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.from_json(value)
        obj.is_fitted_ = True
        obj._fit_frame = None
        obj._params = {"backend": obj._backend_selected()}
        return obj

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("DCRNNForecaster must be fitted before predict")

    def _clone_unfit(self) -> DCRNNForecaster:
        return DCRNNForecaster(**self._params)

    @property
    def selected_backend_(self) -> str:
        return self._backend_selected()

    def _backend_selected(self) -> str:
        backend = getattr(self._native_model, "backend", None)
        if backend is None:
            return str(self._params.get("backend", "cpu"))
        return str(backend())


def available_graph_st_backends() -> list[str]:
    try:
        from cartoboost import _native
    except ImportError:
        return ["cpu"]
    fn = getattr(_native, "graph_st_available_backends_value", None)
    if fn is None:
        return ["cpu"]
    return list(fn())


def _choice_value(value: str | ChoiceStrEnum) -> str:
    if isinstance(value, ChoiceStrEnum):
        return value.value
    return str(value)


class STAEformerForecaster(ArtifactPersistenceMixin):
    """Rust spatiotemporal attention graph sequence forecaster."""

    def __init__(
        self,
        *,
        lookback: int = 8,
        attention_heads: int = 4,
        hidden_size: int = 8,
        epochs: int = 120,
        learning_rate: float = 0.02,
        ridge: float = 0.0001,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        native_class = _native_class("STAEformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for STAEformerForecaster is not available.")
        self._params = {
            "lookback": int(lookback),
            "attention_heads": int(attention_heads),
            "hidden_size": int(hidden_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "ridge": float(ridge),
            "backend": _choice_value(backend),
        }
        self._native_model = native_class(
            self._params["lookback"],
            self._params["attention_heads"],
            self._params["hidden_size"],
            self._params["epochs"],
            self._params["learning_rate"],
            self._params["ridge"],
            self._params["backend"],
        )
        self.is_fitted_ = False

    def fit(self, frame: GraphTemporalFrame) -> STAEformerForecaster:
        self._native_model.fit(_native_frame(frame))
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self._native_model.predict(int(horizon)), dtype=float)

    def score(self, actual: Any, *, horizon: int | None = None) -> float:
        actual_arr = np.asarray(actual, dtype=float)
        if actual_arr.ndim != 2:
            raise ValueError("actual must be a two-dimensional horizon by node matrix")
        if horizon is not None and int(horizon) != actual_arr.shape[0]:
            actual_arr = actual_arr[: int(horizon)]
        return float(self._native_model.score(actual_arr.tolist()))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        self._native_model.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> STAEformerForecaster:
        native_class = _native_class("STAEformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for STAEformerForecaster is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.load(str(path))
        obj.is_fitted_ = True
        obj._params = {"backend": obj._backend_selected()}
        return obj

    def to_json(self) -> str:
        self._check_is_fitted()
        return str(self._native_model.to_json())

    @classmethod
    def from_json(cls, value: str) -> STAEformerForecaster:
        native_class = _native_class("STAEformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for STAEformerForecaster is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.from_json(value)
        obj.is_fitted_ = True
        obj._params = {"backend": obj._backend_selected()}
        return obj

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    def set_params(self, **params: Any) -> STAEformerForecaster:
        valid = set(self._params)
        for key in params:
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
        self.__init__(**{**self._params, **params})
        return self

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": "STAEformerForecaster",
            "params": dict(self._params),
            "backend": self._backend_selected(),
            "fitted": self.is_fitted_,
        }

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("STAEformerForecaster must be fitted before predict")

    @property
    def selected_backend_(self) -> str:
        return self._backend_selected()

    def _backend_selected(self) -> str:
        backend = getattr(self._native_model, "backend", None)
        if backend is None:
            return str(self._params.get("backend", "cpu"))
        return str(backend())


class GraphWaveNetForecaster(ArtifactPersistenceMixin):
    """Rust graph WaveNet-style dilated temporal graph forecaster."""

    def __init__(
        self,
        *,
        lookback: int = 8,
        dilation_depth: int = 3,
        hidden_size: int = 8,
        epochs: int = 120,
        learning_rate: float = 0.02,
        ridge: float = 0.0001,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        native_class = _native_class("GraphWaveNetForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for GraphWaveNetForecaster is not available.")
        self._params = {
            "lookback": int(lookback),
            "dilation_depth": int(dilation_depth),
            "hidden_size": int(hidden_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "ridge": float(ridge),
            "backend": _choice_value(backend),
        }
        self._native_model = native_class(
            self._params["lookback"],
            self._params["dilation_depth"],
            self._params["hidden_size"],
            self._params["epochs"],
            self._params["learning_rate"],
            self._params["ridge"],
            self._params["backend"],
        )
        self.is_fitted_ = False

    def fit(self, frame: GraphTemporalFrame) -> GraphWaveNetForecaster:
        self._native_model.fit(_native_frame(frame))
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self._native_model.predict(int(horizon)), dtype=float)

    def score(self, actual: Any, *, horizon: int | None = None) -> float:
        actual_arr = np.asarray(actual, dtype=float)
        if actual_arr.ndim != 2:
            raise ValueError("actual must be a two-dimensional horizon by node matrix")
        if horizon is not None and int(horizon) != actual_arr.shape[0]:
            actual_arr = actual_arr[: int(horizon)]
        return float(self._native_model.score(actual_arr.tolist()))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        self._native_model.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> GraphWaveNetForecaster:
        native_class = _native_class("GraphWaveNetForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for GraphWaveNetForecaster is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.load(str(path))
        obj.is_fitted_ = True
        obj._params = {"backend": obj._backend_selected()}
        return obj

    def to_json(self) -> str:
        self._check_is_fitted()
        return str(self._native_model.to_json())

    @classmethod
    def from_json(cls, value: str) -> GraphWaveNetForecaster:
        native_class = _native_class("GraphWaveNetForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for GraphWaveNetForecaster is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.from_json(value)
        obj.is_fitted_ = True
        obj._params = {"backend": obj._backend_selected()}
        return obj

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    @property
    def metadata_(self) -> dict[str, Any]:
        return {
            "model": "GraphWaveNetForecaster",
            "params": dict(self._params),
            "backend": self._backend_selected(),
            "fitted": self.is_fitted_,
        }

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("GraphWaveNetForecaster must be fitted before predict")

    @property
    def selected_backend_(self) -> str:
        return self._backend_selected()

    def _backend_selected(self) -> str:
        backend = getattr(self._native_model, "backend", None)
        if backend is None:
            return str(self._params.get("backend", "cpu"))
        return str(backend())


class _PaperGraphTransformerForecaster(ArtifactPersistenceMixin):
    """Shared thin Python facade for paper-derived native graph architectures."""

    _profile: str
    _architecture: str

    def __init__(
        self,
        *,
        lookback: int = 12,
        hidden_size: int = 16,
        attention_heads: int = 4,
        graph_order: int = 2,
        experts: int = 4,
        periodicity: int = 24,
        recent_window: int = 12,
        epochs: int = 80,
        learning_rate: float = 0.01,
        weight_decay: float = 0.00001,
        batch_size: int = 32,
        backend: Backend | str = Backend.CPU,
        horizon: int = 1,
    ) -> None:
        native_class = _native_class("PaperGraphTransformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for paper graph transformers is not available.")
        self._params = {
            "profile": self._profile,
            "lookback": int(lookback),
            "hidden_size": int(hidden_size),
            "attention_heads": int(attention_heads),
            "graph_order": int(graph_order),
            "experts": int(experts),
            "periodicity": int(periodicity),
            "recent_window": int(recent_window),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "backend": _choice_value(backend),
            "horizon": int(horizon),
        }
        if self._params["horizon"] <= 0:
            raise ValueError("horizon must be positive")
        if not 1 <= self._params["batch_size"] <= 32:
            raise ValueError("batch_size must be between 1 and 32")
        self._native_model = native_class(
            **{key: value for key, value in self._params.items() if key != "horizon"}
        )
        self.is_fitted_ = False

    def fit(self, frame: GraphTemporalFrame, *, checkpoint_path: str | Path | None = None):
        if checkpoint_path is None:
            self._native_model.fit(_native_frame(frame))
        else:
            self._native_model.fit_checkpointed(_native_frame(frame), str(Path(checkpoint_path)))
        self.is_fitted_ = True
        return self

    def fit_shard_round(
        self,
        frame: GraphTemporalFrame,
        *,
        shared_state_path: str | Path,
        checkpoint_path: str | Path,
        identity: dict[str, Any] | str,
        objective_weight: float,
        phase: str = "supervised",
        normalization: tuple[float, float] | None = None,
    ) -> str:
        """Emit a frozen-base shared proposal without mutating shared state."""
        if phase not in {"pretrain", "supervised", "local_adaptation"}:
            raise ValueError("phase must be pretrain, supervised, or local_adaptation")
        mean, scale = (None, None) if normalization is None else normalization
        identity_json = (
            identity if isinstance(identity, str) else json.dumps(identity, sort_keys=True)
        )
        proposal = self._native_model.fit_shard_round(
            _native_frame(frame),
            str(Path(shared_state_path)),
            str(Path(checkpoint_path)),
            identity_json,
            float(objective_weight),
            phase,
            mean,
            scale,
        )
        self.is_fitted_ = True
        return str(proposal)

    def prepare_shard_warm_start(self, identity: dict[str, Any] | str):
        """Reset cutoff-bound optimizer state after strict policy validation."""
        identity_json = (
            identity if isinstance(identity, str) else json.dumps(identity, sort_keys=True)
        )
        self._native_model.prepare_shard_warm_start(identity_json)
        return self

    @staticmethod
    def reduce_shard_rounds(rounds: list[str], expected_base_hash: int) -> str:
        native_class = _native_class("PaperGraphTransformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for paper graph transformers is not available.")
        return str(native_class.reduce_shard_rounds(list(rounds), int(expected_base_hash)))

    def predict(self, horizon: int) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self._native_model.predict(int(horizon)), dtype=float)

    def predict_owned(self, horizon: int) -> np.ndarray:
        """Return only owner-node forecasts from a sharded graph frame."""
        self._check_is_fitted()
        return np.asarray(self._native_model.predict_owned(int(horizon)), dtype=float)

    def predict_median(self, horizon: int) -> np.ndarray:
        """Return the stable native direct-decoder median."""
        self._check_is_fitted()
        return np.asarray(self._native_model.predict_median(int(horizon)), dtype=float)

    def predict_conformal(
        self,
        horizon: int,
        *,
        calibration_actual: Any,
        calibration_median: Any,
        alpha: float = 0.1,
    ) -> dict[str, Any]:
        """Build noncrossing horizon-wise intervals from raw held-out pairs."""
        self._check_is_fitted()
        actual = np.asarray(calibration_actual, dtype=float)
        median = np.asarray(calibration_median, dtype=float)
        if actual.ndim != 3 or median.shape != actual.shape:
            raise ValueError(
                "calibration_actual and calibration_median must match shape "
                "(origins, horizon, nodes)"
            )
        return dict(
            json.loads(
                self._native_model.predict_conformal_json(
                    int(horizon),
                    actual.tolist(),
                    median.tolist(),
                    float(alpha),
                )
            )
        )

    def historical_fits(self) -> tuple[int, np.ndarray]:
        """Return frozen-state one-step fits and their first history index."""
        self._check_is_fitted()
        start, values = self._native_model.historical_fits()
        return int(start), np.asarray(values, dtype=float)

    def score(self, actual: Any) -> float:
        self._check_is_fitted()
        actual_arr = np.asarray(actual, dtype=float)
        if actual_arr.ndim != 2:
            raise ValueError("actual must be a two-dimensional horizon by node matrix")
        return float(self._native_model.score(actual_arr.tolist()))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        self._native_model.save(str(path))

    def save_local(self, path: str | Path) -> None:
        """Persist only shard-local parameters; pair with :meth:`load_shard`."""
        self._check_is_fitted()
        self._native_model.save_local(str(path))

    def save_shard_pair(
        self,
        local_path: str | Path,
        shared_path: str | Path,
        manifest_path: str | Path,
    ) -> None:
        """Atomically commit compatible local/shared state via a manifest."""
        self._check_is_fitted()
        self._native_model.save_shard_pair(
            str(Path(local_path)), str(Path(shared_path)), str(Path(manifest_path))
        )

    def parameter_inventory(self) -> list[dict[str, Any]]:
        """Return ownership, optimizer ownership, size, and hashes by state segment."""
        self._check_is_fitted()
        return list(json.loads(self._native_model.parameter_inventory_json()))

    def memory_telemetry(self) -> dict[str, int]:
        """Return component-level native persistent-memory accounting in bytes."""
        self._check_is_fitted()
        return {
            key: int(value)
            for key, value in json.loads(self._native_model.memory_telemetry_json()).items()
        }

    def edge_diagnostics(self) -> list[dict[str, Any]]:
        """Expose per-edge structural, diffusion, learned-attention, and horizon evidence."""
        self._check_is_fitted()
        return list(json.loads(self._native_model.edge_diagnostics_json()))

    @classmethod
    def load(cls, path: str | Path):
        native_class = _native_class("PaperGraphTransformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for paper graph transformers is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.load(str(path))
        obj.is_fitted_ = True
        obj._params = {"profile": cls._profile, "backend": obj._backend_selected(), "horizon": 1}
        return obj

    @classmethod
    def load_shard(cls, local_path: str | Path, shared_state_path: str | Path):
        """Combine a shard-local state with its compatible shared backbone."""
        native_class = _native_class("PaperGraphTransformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for paper graph transformers is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.load_shard(str(local_path), str(shared_state_path))
        obj.is_fitted_ = True
        obj._params = {"profile": cls._profile, "backend": obj._backend_selected(), "horizon": 1}
        return obj

    @classmethod
    def load_shard_pair(
        cls, local_path: str | Path, shared_path: str | Path, manifest_path: str | Path
    ):
        """Load only a complete local/shared transaction committed by ``save_shard_pair``."""
        native_class = _native_class("PaperGraphTransformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for paper graph transformers is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.load_shard_pair(
            str(Path(local_path)), str(Path(shared_path)), str(Path(manifest_path))
        )
        obj.is_fitted_ = True
        obj._params = {"profile": cls._profile, "backend": obj._backend_selected(), "horizon": 1}
        return obj

    def to_json(self) -> str:
        self._check_is_fitted()
        return str(self._native_model.to_json())

    @classmethod
    def from_json(cls, value: str):
        native_class = _native_class("PaperGraphTransformerForecaster")
        if native_class is None:
            raise NotImplementedError("Rust binding for paper graph transformers is not available.")
        obj = cls.__new__(cls)
        obj._native_model = native_class.from_json(value)
        obj.is_fitted_ = True
        obj._params = {"profile": cls._profile, "backend": obj._backend_selected(), "horizon": 1}
        return obj

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    @property
    def metadata_(self) -> dict[str, Any]:
        metadata = {
            "model": type(self).__name__,
            "architecture": self._architecture,
            "profile": self._profile,
            "params": dict(self._params),
            "backend": self._backend_selected(),
            "fitted": self.is_fitted_,
        }
        if self.is_fitted_:
            metadata["architecture_report"] = dict(
                json.loads(self._native_model.architecture_json())
            )
        return metadata

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predict")

    @property
    def selected_backend_(self) -> str:
        return self._backend_selected()

    def _backend_selected(self) -> str:
        return str(self._native_model.backend())


class STGormerForecaster(_PaperGraphTransformerForecaster):
    """Structural-bias spatial/temporal Transformer with routed MoE feed-forwards."""

    _profile = "heterogeneous_moe"
    _architecture = "stgormer"


class STGformerForecaster(_PaperGraphTransformerForecaster):
    """Single-block recursive high-order graph-propagation Transformer."""

    _profile = "efficient_high_order"
    _architecture = "stgformer"


class LSTTNForecaster(_PaperGraphTransformerForecaster):
    """Long-history masked-subseries, periodic graph, and short-term fusion forecaster."""

    _profile = "long_short_fusion"
    _architecture = "lsttn"

    def __init__(
        self,
        *,
        lookback: int = 24 * 28,
        periodicity: int = 24,
        recent_window: int | None = None,
        horizon: int = 24 * 7,
        **kwargs: Any,
    ) -> None:
        """Use a four-week hourly context and one-week horizon by default.

        Callers with another frequency should set all temporal widths explicitly
        (for example, ``lookback=672, periodicity=24, recent_window=168`` for
        four weeks of hourly history, a daily cycle, and one recent week).
        """
        super().__init__(
            lookback=lookback,
            periodicity=periodicity,
            recent_window=min(24 * 7, lookback) if recent_window is None else recent_window,
            horizon=horizon,
            **kwargs,
        )


class SpatialTemporalGraphGatedTransformerForecaster(_PaperGraphTransformerForecaster):
    """Graph convolution with temporal attention and gated recurrent updates."""

    _profile = "gated_graph_temporal"
    _architecture = "spatial_temporal_graph_gated_transformer"


class SpatialShiftGraphonMoEForecaster(_PaperGraphTransformerForecaster):
    """Input-conditioned expert graphon mixture for spatial distribution shift."""

    _profile = "spatial_shift_graphon_moe"
    _architecture = "spatial_shift_graphon_moe"


def _native_frame(frame: GraphTemporalFrame) -> Any:
    native = getattr(frame, "_native_frame", None)
    if native is None:
        raise TypeError("expected a GraphTemporalFrame")
    return native


def _native_class(name: str) -> Any | None:
    try:
        from cartoboost import _native
    except ImportError:
        return None
    return getattr(_native, name, None)


def _horizon_metrics(prediction: np.ndarray, actual: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx, (pred_row, actual_row) in enumerate(zip(prediction, actual, strict=False), start=1):
        err = np.asarray(pred_row, dtype=float) - np.asarray(actual_row, dtype=float)
        abs_err = np.abs(err)
        denom = float(np.sum(np.abs(actual_row)))
        rows.append(
            {
                "horizon": idx,
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "wape": float(np.sum(abs_err) / denom) if denom > 0 else 0.0,
            }
        )
    return rows


__all__ = [
    "DCRNNForecaster",
    "GraphTemporalFrame",
    "GraphWaveNetForecaster",
    "LSTTNForecaster",
    "SpatialShiftGraphonMoEForecaster",
    "SpatialTemporalGraphGatedTransformerForecaster",
    "STAEformerForecaster",
    "STGformerForecaster",
    "STGormerForecaster",
]
