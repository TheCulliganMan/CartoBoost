from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

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
    ) -> None:
        native_class = _native_class("GraphTemporalFrame")
        if native_class is None:
            raise NotImplementedError("Rust binding for GraphTemporalFrame is not available.")
        target_rows = np.asarray(target, dtype=float).tolist()
        covariate_rows = (
            None if covariates is None else np.asarray(covariates, dtype=float).tolist()
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
        self._native_frame = native_class(
            self._node_ids,
            self._timestamps,
            self._target,
            self._indptr,
            self._indices,
            self._data,
            self._horizon,
            self._frequency,
            covariate_rows,
        )

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
        )

    def splitter_data(self) -> dict[str, list[int]]:
        return {"timestamp": list(self._timestamps)}


class DCRNNForecaster:
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
        backend: Backend = Backend.CPU,
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


class STAEformerForecaster:
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
        backend: Backend = Backend.CPU,
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

    def _backend_selected(self) -> str:
        backend = getattr(self._native_model, "backend", None)
        if backend is None:
            return str(self._params.get("backend", "cpu"))
        return str(backend())


class GraphWaveNetForecaster:
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
        backend: Backend = Backend.CPU,
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

    def _backend_selected(self) -> str:
        backend = getattr(self._native_model, "backend", None)
        if backend is None:
            return str(self._params.get("backend", "cpu"))
        return str(backend())


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
    "STAEformerForecaster",
]
