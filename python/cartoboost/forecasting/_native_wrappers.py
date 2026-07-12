from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
    """Thin result container for native forecasting outputs."""

    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.mean, dtype=dtype)


class NativeForecastWrapper:
    """Base class for Python forecasting wrappers over Rust/PyO3 implementations."""

    native_class_name: str

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)
        self._native_model: Any | None = None
        self.is_fitted_ = False

    def fit(self, *args: Any, **kwargs: Any) -> NativeForecastWrapper:
        native_model = self._new_native_model()
        fit = getattr(native_model, "fit", None)
        if fit is None:
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose fit()."
            )
        native_args = self._coerce_fit_args(args)
        result = fit(*native_args, **kwargs)
        self._native_model = native_model if result is None else result
        self.is_fitted_ = True
        return self

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        self._check_is_fitted()
        predict = getattr(self._native_model, "predict", None)
        if predict is None:
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose predict()."
            )
        return predict(*args, **kwargs)

    def forecast(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

    def predict_interval(self, *args: Any, **kwargs: Any) -> Any:
        self._check_is_fitted()
        method = getattr(self._native_model, "predict_interval", None)
        if method is None:
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose predict_interval()."
            )
        return method(*args, **kwargs)

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> NativeForecastWrapper:
        self._params.update(params)
        self._native_model = None
        self.is_fitted_ = False
        return self

    def score(self, values: Any, *, horizon: int | None = None) -> float:
        actual = np.asarray(values, dtype=float).reshape(-1)
        if actual.size == 0:
            raise ValueError("values must contain at least one observation")
        raw_prediction = self.predict(int(horizon or actual.size))
        if hasattr(raw_prediction, "predictions"):
            rows = raw_prediction.predictions
            rows = rows() if callable(rows) else rows
            prediction = np.asarray([row[-1] for row in rows], dtype=float).reshape(-1)
        else:
            prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
        if prediction.shape[0] != actual.shape[0]:
            raise ValueError("prediction and values must have the same length")
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        save = getattr(self._native_model, "save", None)
        if not callable(save):
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose save()."
            )
        save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> NativeForecastWrapper:
        native_class = _native_class(cls.native_class_name)
        if native_class is None:
            raise NotImplementedError(
                f"Rust binding for {cls.__name__} is not available: "
                f"cartoboost._native.{cls.native_class_name} is missing."
            )
        load = getattr(native_class, "load", None)
        if not callable(load):
            raise NotImplementedError(
                f"Rust binding {cls.native_class_name!r} does not expose load()."
            )
        obj = cls.__new__(cls)
        NativeForecastWrapper.__init__(obj)
        obj._native_model = load(str(path))
        obj.is_fitted_ = True
        return obj

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        method = getattr(self._native_model, "get_metadata", None)
        if method is not None:
            return dict(method())
        metadata_json = getattr(self._native_model, "metadata_json", None)
        if metadata_json is not None:
            return dict(json.loads(metadata_json()))
        metadata = getattr(self._native_model, "metadata_", None)
        return {} if metadata is None else dict(metadata)

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.get_metadata()

    def _new_native_model(self) -> Any:
        native_class = _native_class(self.native_class_name)
        if native_class is None:
            raise NotImplementedError(
                f"Rust binding for {self.__class__.__name__} is not available: "
                f"cartoboost._native.{self.native_class_name} is missing."
            )
        return native_class(**self._params)

    def _coerce_fit_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        first = args[0]
        native_frame = getattr(first, "_native_frame", None)
        if native_frame is not None:
            return (native_frame, *args[1:])
        if _is_native_forecast_frame(first):
            return args
        return (_native_frame_from_values(first), *args[1:])

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self._native_model is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict")

    def __getattr__(self, name: str) -> Any:
        native_model = self.__dict__.get("_native_model")
        if native_model is not None and hasattr(native_model, name):
            return getattr(native_model, name)
        raise AttributeError(name)


def _native_class(name: str) -> Any | None:
    try:
        from cartoboost import _native
    except ImportError:
        return None
    return getattr(_native, name, None)


def _is_native_forecast_frame(value: Any) -> bool:
    return value.__class__.__name__ == "ForecastFrame" and value.__class__.__module__.endswith(
        "._native"
    )


def _native_frame_from_values(values: Any) -> Any:
    native_frame_class = _native_class("ForecastFrame")
    if native_frame_class is None:
        raise NotImplementedError("Rust binding for ForecastFrame is not available.")
    if isinstance(values, dict):
        rows = []
        lengths = {len(series_values) for series_values in values.values()}
        if len(lengths) != 1:
            raise ValueError("all panel series must have the same length")
        for series_id, series_values in values.items():
            for idx, value in enumerate(series_values):
                rows.append(
                    (
                        str(series_id),
                        (datetime(1970, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%dT%H:%M:%S"),
                        float(value),
                    )
                )
        return native_frame_class(rows, "D")
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        rows = [
            (
                "__single__",
                (datetime(1970, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%dT%H:%M:%S"),
                float(value),
            )
            for idx, value in enumerate(arr)
        ]
    elif arr.ndim == 2:
        rows = []
        for idx in range(arr.shape[0]):
            timestamp = (datetime(1970, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%dT%H:%M:%S")
            for series_idx in range(arr.shape[1]):
                rows.append((str(series_idx), timestamp, float(arr[idx, series_idx])))
    else:
        raise ValueError("forecast training values must be a 1D series or 2D panel")
    return native_frame_class(rows, "D")


def _forecast_frame_to_artifact(frame: Any) -> dict[str, Any]:
    """Serialize a validated ForecastFrame into a JSON-safe training payload."""

    data = frame.to_pandas()
    records: list[dict[str, Any]] = []
    for raw in data.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for name, value in raw.items():
            if isinstance(value, np.generic):
                value = value.item()
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif value is not None:
                try:
                    if bool(np.isnan(value)):
                        value = None
                except (TypeError, ValueError):
                    pass
            record[str(name)] = value
        records.append(record)
    return {
        "metadata": frame.to_metadata(),
        "columns": [str(column) for column in data.columns],
        "records": records,
    }


def _forecast_frame_from_artifact(payload: Any) -> Any:
    """Restore a ForecastFrame from a JSON-safe training payload."""

    if not isinstance(payload, dict):
        raise ValueError("forecast artifact training_frame must be an object")
    metadata = payload.get("metadata")
    columns = payload.get("columns")
    records = payload.get("records")
    if (
        not isinstance(metadata, dict)
        or not isinstance(columns, list)
        or not isinstance(records, list)
    ):
        raise ValueError("forecast artifact training_frame is malformed")
    from .schema import ForecastFrame

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is an explicit forecast dependency.
        raise ImportError(
            "loading a forecasting artifact requires pandas; install cartoboost[pandas]"
        ) from exc
    data = pd.DataFrame.from_records(records, columns=[str(column) for column in columns])
    return ForecastFrame.from_pandas(
        data,
        timestamp_col=str(metadata["timestamp_col"]),
        target_col=str(metadata["target_col"]),
        series_id_col=metadata.get("series_id_col"),
        freq=metadata.get("freq"),
        static_covariates=metadata.get("static_covariates", []),
        known_future_covariates=metadata.get("known_future_covariates", []),
        historical_covariates=metadata.get("historical_covariates", []),
        allow_irregular=bool(metadata.get("allow_irregular", False)),
        allow_missing_targets=bool(metadata.get("allow_missing_targets", False)),
        allow_missing_covariates=bool(metadata.get("allow_missing_covariates", False)),
        sample_weight_col=metadata.get("sample_weight_col"),
    )
