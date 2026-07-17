from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


class FoundationAdapterUnavailable(ImportError):
    """Raised when an optional foundation-model dependency is not installed."""


class _FoundationAdapter:
    adapter_name = "foundation"
    dependency_name: str | None = None

    def __init__(
        self,
        *,
        model_id: str,
        model_hash: str | None = None,
        backend: Any | None = None,
        explicitly_enabled: bool = False,
    ) -> None:
        self.model_id = str(model_id)
        self.model_hash = model_hash or _stable_hash(self.model_id)
        self.backend = backend
        self.explicitly_enabled = bool(explicitly_enabled)

    def external_metadata(self) -> dict[str, Any]:
        version = None
        if self.dependency_name and importlib.util.find_spec(self.dependency_name) is not None:
            try:
                version = importlib.metadata.version(self.dependency_name)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        return {
            "adapter": self.adapter_name,
            "dependency": self.dependency_name,
            "external_version": version,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "explicitly_enabled": self.explicitly_enabled,
            "auto_geo_enabled": self.explicitly_enabled,
        }

    def missing_dependency_skip_reason(self) -> str | None:
        if self.backend is not None or self.dependency_name is None:
            return None
        if importlib.util.find_spec(self.dependency_name) is not None:
            return None
        return (
            f"{self.adapter_name} requires optional dependency {self.dependency_name!r}; "
            f"install {self.dependency_name} or provide a backend."
        )

    def predict(self, inputs: Any) -> np.ndarray:
        skip_reason = self.missing_dependency_skip_reason()
        if skip_reason is not None:
            raise FoundationAdapterUnavailable(skip_reason)
        if self.backend is None:
            raise FoundationAdapterUnavailable(
                f"{self.adapter_name} has no local backend configured; use cached outputs or "
                "provide an explicit backend."
            )
        output = self.backend(inputs)
        arr = np.asarray(output, dtype=float)
        if arr.ndim == 0 or not np.isfinite(arr).all():
            raise ValueError("foundation adapter output must be a finite array")
        return arr

    def cache_output(self, path: str | Path, inputs: Any, output: Any) -> Path:
        arr = np.asarray(output, dtype=float)
        if arr.ndim == 0 or not np.isfinite(arr).all():
            raise ValueError("cached foundation output must be a finite array")
        payload = {
            "metadata": self.external_metadata(),
            "input_hash": _json_hash(_jsonable(inputs)),
            "output": arr.tolist(),
            "output_shape": list(arr.shape),
        }
        path = Path(path)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def load_cache(path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            "metadata": dict(payload["metadata"]),
            "input_hash": str(payload["input_hash"]),
            "output": np.asarray(payload["output"], dtype=float),
            "output_shape": list(payload["output_shape"]),
        }


class ChronosAdapter(_FoundationAdapter):
    adapter_name = "chronos"
    dependency_name = "chronos"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_id="chronos", **kwargs)


class TimesFMAdapter(_FoundationAdapter):
    adapter_name = "timesfm"
    dependency_name = "timesfm"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_id="timesfm", **kwargs)


class MoiraiAdapter(_FoundationAdapter):
    adapter_name = "moirai"
    dependency_name = "uni2ts"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_id="moirai", **kwargs)


class TimeGPTAdapter(_FoundationAdapter):
    adapter_name = "timegpt"
    dependency_name = "nixtla"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_id="timegpt", **kwargs)


class TabPFNAdapter(_FoundationAdapter):
    adapter_name = "tabpfn"
    dependency_name = "tabpfn"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_id="tabpfn", **kwargs)


class FoundationForecastFeatures:
    """Cached foundation-model forecast features with reproducibility metadata."""

    def __init__(self, adapter: _FoundationAdapter) -> None:
        self.adapter = adapter

    def transform_from_cache(self, path: str | Path) -> np.ndarray:
        return self.adapter.load_cache(path)["output"]

    @staticmethod
    def benchmark_with_without_features(
        y_true: Any,
        baseline_prediction: Any,
        foundation_prediction: Any,
    ) -> dict[str, float]:
        y = np.asarray(y_true, dtype=float).reshape(-1)
        base = np.asarray(baseline_prediction, dtype=float).reshape(-1)
        foundation = np.asarray(foundation_prediction, dtype=float).reshape(-1)
        if y.shape != base.shape or y.shape != foundation.shape:
            raise ValueError("benchmark arrays must have matching shapes")
        base_rmse = float(np.sqrt(np.mean((y - base) ** 2)))
        foundation_rmse = float(np.sqrt(np.mean((y - foundation) ** 2)))
        return {
            "without_foundation_rmse": base_rmse,
            "with_foundation_rmse": foundation_rmse,
            "rmse_delta": base_rmse - foundation_rmse,
        }


class TabPFNFeatureGenerator(FoundationForecastFeatures):
    pass


class PriorFittedBaseline(TabPFNAdapter):
    pass


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "ChronosAdapter",
    "FoundationAdapterUnavailable",
    "FoundationForecastFeatures",
    "MoiraiAdapter",
    "PriorFittedBaseline",
    "TabPFNAdapter",
    "TabPFNFeatureGenerator",
    "TimeGPTAdapter",
    "TimesFMAdapter",
]
