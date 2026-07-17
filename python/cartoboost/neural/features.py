"""Neural feature tooling for CartoBoost."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .._native import NeuralEmbeddingFeatures as _NativeNeuralEmbeddingFeatures
from ..config import Backend


class ArtifactFallback(str):
    ZERO_VECTOR = "zero_vector"
    GLOBAL_MEAN = "global_mean_vector"
    PARENT_CELL = "parent_cell"


class NeuralEmbeddingFeatures:
    """Learn and persist embedding table artifacts for CartoBoost neural features."""

    def __init__(
        self,
        *,
        dim: int,
        fallback: str = ArtifactFallback.GLOBAL_MEAN,
        random_state: int | None = 42,
        parent_resolution: int | None = None,
        support_prior_strength: float = 1.0,
        backend: Backend | str = Backend.CPU,
    ) -> None:
        self.dim = int(dim)
        self.fallback = str(fallback)
        self.random_state = random_state
        self.parent_resolution = parent_resolution
        self.support_prior_strength = float(support_prior_strength)
        self.backend = str(backend)
        self._backend = _NativeNeuralEmbeddingFeatures(
            dim=self.dim,
            fallback=self.fallback,
            random_state=None if random_state is None else int(random_state),
            parent_resolution=parent_resolution,
            support_prior_strength=self.support_prior_strength,
            backend=self.backend,
        )
        self._row_cache: dict[int, np.ndarray] | None = None

    def fit(self, ids: Iterable[Any], target: Iterable[Any]) -> NeuralEmbeddingFeatures:
        id_values = _as_1d_array(ids, np.uint64, "ids")
        target_values = _as_1d_array(target, np.float64, "target")
        if id_values.size != target_values.size:
            raise ValueError("ids and target must have the same length")

        self._backend.fit(id_values, target_values)
        self._row_cache = None
        return self

    def transform(self, ids: Iterable[Any]) -> np.ndarray:
        id_values = _as_1d_array(ids, np.uint64, "ids")
        if id_values.size == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.asarray(self._backend.transform(id_values), dtype=np.float32)

    def transform_with_fallback(
        self,
        ids: Iterable[Any],
        *,
        fallback_ids: Iterable[Any] | None = None,
        neighbor_ids: Iterable[Iterable[Any]] | None = None,
    ) -> np.ndarray:
        """Transform IDs, replacing unseen rows with hierarchy or neighbor vectors.

        `fallback_ids` may be a 1D parent-ID array or a 2D array whose columns are
        tried in order, for example zone -> borough -> service-zone. `neighbor_ids`
        is a row-aligned iterable of neighbor ID iterables; known neighbor
        embeddings are averaged for graph-aware fallback.
        """

        id_values = _as_1d_array(ids, np.uint64, "ids")
        output = self.transform(id_values)
        if id_values.size == 0:
            return output

        row_cache = self._embedding_rows()
        unknown = np.array([int(row_id) not in row_cache for row_id in id_values], dtype=bool)
        if not np.any(unknown):
            return output

        if neighbor_ids is not None:
            neighbor_rows = list(neighbor_ids)
            if len(neighbor_rows) != id_values.size:
                raise ValueError("neighbor_ids length must match ids")
            for row_index in np.flatnonzero(unknown):
                vectors = [
                    row_cache[int(neighbor_id)]
                    for neighbor_id in _iter_u64_values(neighbor_rows[row_index])
                    if int(neighbor_id) in row_cache
                ]
                if vectors:
                    output[row_index] = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
                    unknown[row_index] = False

        if fallback_ids is not None and np.any(unknown):
            fallback_matrix = _as_fallback_matrix(fallback_ids, id_values.size)
            for row_index in np.flatnonzero(unknown):
                for fallback_id in fallback_matrix[row_index]:
                    vector = row_cache.get(int(fallback_id))
                    if vector is not None:
                        output[row_index] = vector
                        unknown[row_index] = False
                        break

        return output

    def fit_transform(self, ids: Iterable[Any], target: Iterable[Any]) -> np.ndarray:
        id_values = _as_1d_array(ids, np.uint64, "ids")
        target_values = _as_1d_array(target, np.float64, "target")
        if id_values.size != target_values.size:
            raise ValueError("ids and target must have the same length")

        if id_values.size == 0:
            self.fit(id_values, target_values)
            return np.empty((0, self.dim), dtype=np.float32)

        output = np.asarray(self._backend.fit_transform(id_values, target_values), dtype=np.float32)
        self._row_cache = None
        return output

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        self._backend.export(str(path))
        return path

    @classmethod
    def from_artifact(cls, path: str | Path) -> NeuralEmbeddingFeatures:
        path = Path(path)
        native = _NativeNeuralEmbeddingFeatures.from_artifact(str(path))
        instance = cls(
            dim=native.dim,
            fallback=native.fallback,
            random_state=None,
            parent_resolution=native.parent_resolution,
            support_prior_strength=native.support_prior_strength,
        )
        instance._backend = native
        instance._row_cache = None
        return instance

    def artifact_rows(self) -> list[dict[str, Any]]:
        return [
            {"id": int(row_id), "values": values}
            for row_id, values in self._backend.artifact_rows()
        ]

    def known_ids(self) -> set[int]:
        return set(self._embedding_rows())

    def _embedding_rows(self) -> dict[int, np.ndarray]:
        if self._row_cache is None:
            self._row_cache = {
                int(row_id): np.asarray(values, dtype=np.float32)
                for row_id, values in self._backend.artifact_rows()
            }
        return self._row_cache


def _as_1d_array(values: Iterable[Any], dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        raise ValueError(f"{name} must be 1D")
    if array.ndim != 1:
        array = np.ravel(array)
    return np.ascontiguousarray(array, dtype=dtype)


def _as_fallback_matrix(values: Iterable[Any], row_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("fallback_ids must be 1D or 2D")
    if array.ndim == 1:
        if array.shape[0] != row_count:
            raise ValueError("fallback_ids length must match ids")
        array = array.reshape(row_count, 1)
    elif array.ndim == 2:
        if array.shape[0] != row_count:
            raise ValueError("fallback_ids row count must match ids")
    else:
        raise ValueError("fallback_ids must be 1D or 2D")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _iter_u64_values(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if array.size == 0:
        return np.empty(0, dtype=np.uint64)
    return np.ravel(array).astype(np.uint64)
