from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import Backend, FallbackMode
from ..regressor import CartoBoostRegressor, _rust_feature_schema_payload
from .features import NeuralEmbeddingFeatures


class NeuralEmbeddingRegressor:
    """Two-stage estimator that appends offline neural embeddings as dense columns.

    The estimator can train embeddings on residuals (the default) or on the full
    target. In both cases, the final model remains `CartoBoostRegressor`.
    """

    def __init__(
        self,
        *,
        dim: int = 16,
        fallback: FallbackMode = FallbackMode.GLOBAL_MEAN_VECTOR,
        random_state: int | None = 42,
        neural_transformer: NeuralEmbeddingFeatures | None = None,
        use_residual: bool = True,
        oof_folds: int = 1,
        drop_id_column: bool = True,
        id_column: int | str | None = None,
        support_prior_strength: float = 1.0,
        backend: Backend | str = Backend.CPU,
        base_model_kwargs: dict[str, Any] | None = None,
        final_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if oof_folds < 1:
            raise ValueError("oof_folds must be at least 1")

        self.dim = int(dim)
        self.fallback = fallback
        self.random_state = random_state
        self.use_residual = use_residual
        self.oof_folds = int(oof_folds)
        self.drop_id_column = drop_id_column
        self.id_column = id_column
        self.support_prior_strength = float(support_prior_strength)
        self.backend = str(backend)
        self.neural_transformer = neural_transformer or NeuralEmbeddingFeatures(
            dim=dim,
            fallback=str(fallback),
            random_state=random_state,
            support_prior_strength=support_prior_strength,
            backend=self.backend,
        )
        self.neural_transformers: list[NeuralEmbeddingFeatures] = []
        self.base_model_kwargs = dict(base_model_kwargs or {})
        self.final_model_kwargs = dict(final_model_kwargs or self.base_model_kwargs)

        self._base_model: CartoBoostRegressor | None = None
        self._final_model: CartoBoostRegressor | None = None
        self._fit_timings_ms: dict[str, float] = {}
        self._feature_names_in_: np.ndarray | None = None
        self._n_features_in_ = 0

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        sparse_sets: Any | None = None,
        id_column: int | str | None = None,
        ids: np.ndarray | list[Any] | None = None,
        fallback_ids: Any | None = None,
        neighbor_ids: Any | None = None,
        **fit_kwargs: Any,
    ) -> NeuralEmbeddingRegressor:
        resolved_id_column = self._resolve_id_column(id_column)
        if resolved_id_column is None and ids is None:
            raise ValueError("id_column or ids is required for embedding lookup")

        dense, id_values = self._prepare_dense_and_ids(X, ids, resolved_id_column)
        target = _to_1d_float_array(y)

        if dense.shape[0] != target.shape[0]:
            raise ValueError("X and y must contain the same number of rows")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (target.shape[0],):
                raise ValueError("sample_weight length must match y")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0.0):
                raise ValueError("sample_weight must contain only finite non-negative values")

        fit_kwargs = dict(fit_kwargs)
        feature_schema = fit_kwargs.pop("feature_schema", None)
        fit_kwargs.update({"sparse_sets": sparse_sets, "sample_weight": sample_weight})

        base_fit_kwargs = dict(fit_kwargs)
        if feature_schema is not None:
            base_fit_kwargs["feature_schema"] = feature_schema

        if self.use_residual:
            base_model = self._build_base_model()
            start = time.perf_counter()
            base_model.fit(dense, target, **base_fit_kwargs)
            self._fit_timings_ms["base_fit_ms"] = _ms_since(start)

            residual = target - base_model.predict(dense, sparse_sets=sparse_sets)
            residual_target = residual
            self._base_model = base_model
        else:
            residual_target = target

        start = time.perf_counter()
        neural_features = self._fit_transform_embeddings(
            id_values,
            residual_target,
            fallback_ids=fallback_ids,
            neighbor_ids=neighbor_ids,
        )
        self._fit_timings_ms["neural_fit_ms"] = _ms_since(start)

        if neural_features.shape[0] != dense.shape[0]:
            raise ValueError("neural transformer output dimension mismatch")

        x_train = _append_features(dense, neural_features)
        final_model = self._build_final_model()
        sparse_names = _sparse_set_names(sparse_sets)
        final_feature_schema = _append_feature_schema(
            feature_schema=feature_schema,
            dense_width=dense.shape[1],
            sparse_names=sparse_names,
            neural_dim=neural_features.shape[1],
        )

        start = time.perf_counter()
        final_fit_kwargs = dict(fit_kwargs)
        if final_feature_schema is not None:
            final_fit_kwargs["feature_schema"] = final_feature_schema
        final_model.fit(x_train, target, **final_fit_kwargs)
        self._fit_timings_ms["final_fit_ms"] = _ms_since(start)
        self._final_model = final_model

        self._n_features_in_ = x_train.shape[1]
        self._feature_names_in_ = _build_neural_feature_names(
            getattr(X, "columns", None),
            dense.shape[1],
            neural_features.shape[1],
        )
        return self

    def predict(
        self,
        X: Any,
        *,
        sparse_sets: Any | None = None,
        ids: np.ndarray | list[Any] | None = None,
        id_column: int | str | None = None,
        fallback_ids: Any | None = None,
        neighbor_ids: Any | None = None,
    ) -> np.ndarray:
        if self._final_model is None:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")

        dense, id_values = self._prepare_dense_and_ids(
            X,
            ids,
            self._resolve_id_column(id_column),
        )
        neural = self._transform_embeddings(
            id_values,
            fallback_ids=fallback_ids,
            neighbor_ids=neighbor_ids,
        )
        x_pred = _append_features(dense, neural)
        return self._final_model.predict(x_pred, sparse_sets=sparse_sets)

    def transform(
        self,
        X: Any,
        *,
        ids: np.ndarray | list[Any] | None = None,
        id_column: int | str | None = None,
        fallback_ids: Any | None = None,
        neighbor_ids: Any | None = None,
    ) -> np.ndarray:
        if self._final_model is None:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")

        dense, id_values = self._prepare_dense_and_ids(
            X,
            ids,
            self._resolve_id_column(id_column),
        )
        return _append_features(
            dense,
            self._transform_embeddings(
                id_values,
                fallback_ids=fallback_ids,
                neighbor_ids=neighbor_ids,
            ),
        )

    def score(
        self,
        X: Any,
        y: Any,
        *,
        sparse_sets: Any | None = None,
        id_column: int | str | None = None,
        ids: np.ndarray | list[Any] | None = None,
        fallback_ids: Any | None = None,
        neighbor_ids: Any | None = None,
    ) -> float:
        predictions = self.predict(
            X,
            sparse_sets=sparse_sets,
            ids=ids,
            id_column=id_column,
            fallback_ids=fallback_ids,
            neighbor_ids=neighbor_ids,
        )
        return _mean_absolute_error(y, predictions)

    def benchmark(
        self,
        X: Any,
        y: Any,
        *,
        sparse_sets: Any | None = None,
        id_column: int | str | None = None,
        ids: np.ndarray | list[Any] | None = None,
    ) -> dict[str, float]:
        start = time.perf_counter()
        pred = self.predict(X, sparse_sets=sparse_sets, ids=ids, id_column=id_column)
        mae = _mean_absolute_error(y, pred)
        return {"mae": mae, "predict_ms": _ms_since(start)}

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._fit_timings_ms)

    @property
    def n_features_in_(self) -> int:
        return self._n_features_in_

    @property
    def feature_names_in_(self) -> np.ndarray | None:
        return self._feature_names_in_

    @property
    def model(self) -> CartoBoostRegressor:
        if self._final_model is None:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")
        return self._final_model

    @property
    def neural_feature_count_(self) -> int:
        return self.dim * max(1, len(self.neural_transformers))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters without fitted native transformer state."""

        return {
            "dim": self.dim,
            "fallback": self.fallback,
            "random_state": self.random_state,
            "neural_transformer": None,
            "use_residual": self.use_residual,
            "oof_folds": self.oof_folds,
            "drop_id_column": self.drop_id_column,
            "id_column": self.id_column,
            "support_prior_strength": self.support_prior_strength,
            "backend": self.backend,
            "base_model_kwargs": dict(self.base_model_kwargs),
            "final_model_kwargs": dict(self.final_model_kwargs),
        }

    def set_params(self, **params: Any) -> NeuralEmbeddingRegressor:
        valid = self.get_params(deep=False)
        unknown = sorted(set(params) - set(valid))
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
        updated = {**valid, **params}
        self.__init__(**updated)
        return self

    def __sklearn_clone__(self) -> NeuralEmbeddingRegressor:
        return type(self)(**self.get_params(deep=False))

    def _build_base_model(self) -> CartoBoostRegressor:
        return CartoBoostRegressor(**self.base_model_kwargs)

    def _build_final_model(self) -> CartoBoostRegressor:
        return CartoBoostRegressor(**self.final_model_kwargs)

    def _resolve_id_column(self, id_column: int | str | None) -> int | str | None:
        if id_column is not None:
            return id_column
        return self.id_column

    def _fit_transform_embeddings(
        self,
        ids: np.ndarray,
        target: np.ndarray,
        *,
        fallback_ids: Any | None,
        neighbor_ids: Any | None,
    ) -> np.ndarray:
        id_matrix = _ensure_id_matrix(ids)
        fallback_by_key = _split_fallback_by_key(
            fallback_ids, id_matrix.shape[1], id_matrix.shape[0]
        )
        neighbor_by_key = _split_neighbor_by_key(
            neighbor_ids, id_matrix.shape[1], id_matrix.shape[0]
        )
        self.neural_transformers = []

        blocks: list[np.ndarray] = []
        for key_index in range(id_matrix.shape[1]):
            transformer = self._new_transformer(key_index)
            key_ids = id_matrix[:, key_index]
            if self.oof_folds > 1 and key_ids.shape[0] >= self.oof_folds:
                block = self._oof_embedding_block(
                    key_ids,
                    target,
                    key_index=key_index,
                    fallback_ids=fallback_by_key[key_index],
                    neighbor_ids=neighbor_by_key[key_index],
                )
                transformer.fit(key_ids, target)
            else:
                transformer.fit(key_ids, target)
                block = transformer.transform_with_fallback(
                    key_ids,
                    fallback_ids=fallback_by_key[key_index],
                    neighbor_ids=neighbor_by_key[key_index],
                )
            self.neural_transformers.append(transformer)
            blocks.append(block)

        if len(blocks) == 1:
            self.neural_transformer = self.neural_transformers[0]
            return blocks[0]
        return np.hstack(blocks)

    def _transform_embeddings(
        self,
        ids: np.ndarray,
        *,
        fallback_ids: Any | None,
        neighbor_ids: Any | None,
    ) -> np.ndarray:
        if not self.neural_transformers:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")
        id_matrix = _ensure_id_matrix(ids)
        if id_matrix.shape[1] != len(self.neural_transformers):
            raise ValueError("ids key count must match fitted neural transformers")
        fallback_by_key = _split_fallback_by_key(
            fallback_ids, id_matrix.shape[1], id_matrix.shape[0]
        )
        neighbor_by_key = _split_neighbor_by_key(
            neighbor_ids, id_matrix.shape[1], id_matrix.shape[0]
        )
        blocks = [
            transformer.transform_with_fallback(
                id_matrix[:, key_index],
                fallback_ids=fallback_by_key[key_index],
                neighbor_ids=neighbor_by_key[key_index],
            )
            for key_index, transformer in enumerate(self.neural_transformers)
        ]
        if len(blocks) == 1:
            return blocks[0]
        return np.hstack(blocks)

    def _oof_embedding_block(
        self,
        ids: np.ndarray,
        target: np.ndarray,
        *,
        key_index: int,
        fallback_ids: Any | None,
        neighbor_ids: Any | None,
    ) -> np.ndarray:
        output = np.empty((ids.shape[0], self.dim), dtype=np.float32)
        rng = np.random.default_rng(0 if self.random_state is None else self.random_state)
        folds = np.array_split(rng.permutation(ids.shape[0]), self.oof_folds)
        for fold_index, valid_idx in enumerate(folds):
            if valid_idx.size == 0:
                continue
            train_mask = np.ones(ids.shape[0], dtype=bool)
            train_mask[valid_idx] = False
            transformer = self._new_transformer(key_index + fold_index + 1)
            transformer.fit(ids[train_mask], target[train_mask])
            output[valid_idx] = transformer.transform_with_fallback(
                ids[valid_idx],
                fallback_ids=_take_optional_rows(fallback_ids, valid_idx),
                neighbor_ids=_take_optional_rows(neighbor_ids, valid_idx),
            )
        return output

    def _new_transformer(self, salt: int) -> NeuralEmbeddingFeatures:
        random_state = None if self.random_state is None else int(self.random_state) + salt
        return NeuralEmbeddingFeatures(
            dim=self.dim,
            fallback=self.fallback,
            random_state=random_state,
            support_prior_strength=self.support_prior_strength,
            backend=self.backend,
        )

    def _prepare_dense_and_ids(
        self,
        X: Any,
        ids: np.ndarray | list[Any] | None,
        id_column: int | str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ids is not None and id_column is not None:
            raise ValueError("provide either ids or id_column, not both")

        if ids is not None:
            id_values = _to_u64_id_matrix(ids)
            dense = _to_2d_float_array(X)
            return dense, id_values

        if id_column is None:
            raise ValueError("id_column or ids is required")

        if isinstance(id_column, int):
            dense = _to_2d_float_array(X)
            if id_column < 0 or id_column >= dense.shape[1]:
                raise ValueError("id_column index is out of range")
            id_values = _to_u64_id_matrix(dense[:, id_column])

            if self.drop_id_column:
                dense = np.delete(dense, id_column, axis=1)

            return dense, id_values

        if not hasattr(X, "__getitem__"):
            raise ValueError("id_column string lookup requires object with named columns")

        try:
            id_values = _to_u64_id_matrix(X[id_column])
        except (TypeError, KeyError) as exc:
            raise ValueError(f"id_column {id_column!r} not found") from exc

        if self.drop_id_column:
            dense_obj = _drop_named_column(X, id_column)
            dense = _to_2d_float_array(dense_obj)
        else:
            dense = _to_2d_float_array(X)

        return dense, id_values


def benchmark_neural_vs_cartoboost(
    X: Any,
    y: Any,
    *,
    ids: np.ndarray | list[Any],
    split_ratio: float = 0.8,
    neural_kwargs: dict[str, Any] | None = None,
    cartoboost_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_array = _to_2d_float_array(X)
    y_array = _to_1d_float_array(y)
    id_values = _to_u64_ids(ids)

    if x_array.shape[0] != y_array.shape[0] or x_array.shape[0] != id_values.shape[0]:
        raise ValueError("X, y, and ids must have the same number of rows")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be in (0, 1)")

    split_index = int(x_array.shape[0] * split_ratio)
    if split_index <= 0 or split_index >= x_array.shape[0]:
        raise ValueError("split_ratio must keep both train and validation sets")

    x_train = x_array[:split_index]
    x_valid = x_array[split_index:]
    y_train = y_array[:split_index]
    y_valid = y_array[split_index:]
    id_train = id_values[:split_index]
    id_valid = id_values[split_index:]

    cartooboost_params = dict(cartoboost_kwargs or {})
    cartooboost = CartoBoostRegressor(**cartooboost_params)

    start = time.perf_counter()
    cartooboost.fit(x_train, y_train)
    carto_fit_ms = _ms_since(start)

    start = time.perf_counter()
    cart_pred = cartooboost.predict(x_valid)
    carto_pred_ms = _ms_since(start)
    carto_mae = _mean_absolute_error(y_valid, cart_pred)

    hybrid = NeuralEmbeddingRegressor(
        dim=16,
        drop_id_column=False,
        **(neural_kwargs or {}),
    )
    start = time.perf_counter()
    hybrid.fit(x_train, y_train, ids=id_train)
    hybrid_fit_ms = _ms_since(start)

    start = time.perf_counter()
    hybrid_pred = hybrid.predict(x_valid, ids=id_valid)
    hybrid_pred_ms = _ms_since(start)
    hybrid_mae = _mean_absolute_error(y_valid, hybrid_pred)

    return {
        "n_rows": int(x_array.shape[0]),
        "n_features": int(x_array.shape[1]),
        "structured_mae": carto_mae,
        "hybrid_mae": hybrid_mae,
        "improvement": carto_mae - hybrid_mae,
        "cartoboost_fit_ms": carto_fit_ms,
        "cartoboost_predict_ms": carto_pred_ms,
        "hybrid_fit_ms": hybrid_fit_ms,
        "hybrid_predict_ms": hybrid_pred_ms,
    }


def _to_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("X must be a 2D numeric array or 2D table")
    if array.shape[0] == 0:
        raise ValueError("X must not be empty")
    if array.shape[1] == 0:
        raise ValueError("X must include at least one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _to_1d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("y must be a 1D numeric array")
    if array.shape[0] == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("y must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _to_u64_ids(values: np.ndarray | list[Any] | tuple[Any, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("ids must be 1D")
    if array.ndim != 1:
        array = np.ravel(array)

    float_ids = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(float_ids)):
        raise ValueError("ids must be finite")

    return np.asarray(np.rint(float_ids).astype(np.uint64), dtype=np.uint64)


def _to_u64_id_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("ids must be 1D or 2D")
    if array.ndim > 2:
        raise ValueError("ids must be 1D or 2D")

    float_ids = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(float_ids)):
        raise ValueError("ids must be finite")

    id_values = np.asarray(np.rint(float_ids).astype(np.uint64), dtype=np.uint64)
    if id_values.ndim == 1:
        id_values = id_values.reshape(-1, 1)
    return np.ascontiguousarray(id_values, dtype=np.uint64)


def _ensure_id_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.uint64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("ids must be 1D or 2D")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _split_fallback_by_key(values: Any | None, n_keys: int, row_count: int) -> list[Any | None]:
    if values is None:
        return [None] * n_keys
    if isinstance(values, (list, tuple)) and len(values) == n_keys:
        try:
            return [
                None if item is None else _normalize_fallback_rows(item, row_count)
                for item in values
            ]
        except ValueError:
            pass

    array = np.asarray(values)
    if n_keys == 1:
        return [_normalize_fallback_rows(array, row_count)]
    if array.ndim == 2 and array.shape == (row_count, n_keys):
        return [array[:, key_index] for key_index in range(n_keys)]
    if array.ndim == 3 and array.shape[0] == row_count and array.shape[1] == n_keys:
        return [array[:, key_index, :] for key_index in range(n_keys)]
    raise ValueError("fallback_ids shape must align with rows and embedding keys")


def _normalize_fallback_rows(values: Any, row_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        if array.shape[0] != row_count:
            raise ValueError("fallback_ids length must match rows")
        return np.ascontiguousarray(array, dtype=np.uint64)
    if array.ndim == 2:
        if array.shape[0] != row_count:
            raise ValueError("fallback_ids row count must match rows")
        return np.ascontiguousarray(array, dtype=np.uint64)
    raise ValueError("fallback_ids for each key must be 1D or 2D")


def _split_neighbor_by_key(values: Any | None, n_keys: int, row_count: int) -> list[Any | None]:
    if values is None:
        return [None] * n_keys
    if n_keys == 1:
        if len(values) != row_count:
            raise ValueError("neighbor_ids length must match rows")
        return [values]
    if not isinstance(values, (list, tuple)) or len(values) != n_keys:
        raise ValueError("neighbor_ids for multiple keys must be a list per key")
    for item in values:
        if item is None:
            continue
        if len(item) != row_count:
            raise ValueError("each neighbor_ids key list length must match rows")
    return list(values)


def _take_optional_rows(values: Any | None, indices: np.ndarray) -> Any | None:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(index)] for index in indices]


def _drop_named_column(values: Any, column: str) -> Any:
    if hasattr(values, "drop"):
        try:
            return values.drop(columns=[column])
        except Exception:
            pass

    if hasattr(values, "drop_columns"):
        try:
            return values.drop_columns([column])
        except Exception:
            pass

    if hasattr(values, "columns"):
        columns = list(values.columns)
        try:
            index = columns.index(column)
        except ValueError as exc:
            raise ValueError(f"id_column {column!r} not found") from exc

        array = _to_2d_float_array(values)
        if 0 <= index < array.shape[1]:
            return np.delete(array, index, axis=1)

    return _to_2d_float_array(values)


def _append_features(dense: np.ndarray, neural_features: np.ndarray) -> np.ndarray:
    if dense.shape[0] != neural_features.shape[0]:
        raise ValueError("X and neural features must have the same row count")
    return np.hstack([dense, np.asarray(neural_features, dtype=np.float64)])


def _build_neural_feature_names(
    columns: list[str] | None,
    base_feature_count: int,
    dim: int,
) -> np.ndarray | None:
    if columns is None:
        return None

    base_names = list(columns)

    if len(base_names) != base_feature_count:
        return None

    neural_names = _neural_feature_names(dim)
    return np.array([*base_names, *neural_names], dtype=object)


def _ms_since(start_time: float) -> float:
    return float((time.perf_counter() - start_time) * 1000.0)


def _mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("mean absolute error requires at least one row")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must contain only finite values")
    return float(np.mean(np.abs(truth - pred)))


def _neural_feature_names(dim: int) -> list[str]:
    return [f"neural_embedding_{index:02}" for index in range(dim)]


def _sparse_set_names(sparse_sets: Any | None) -> list[str]:
    if sparse_sets is None:
        return []
    if isinstance(sparse_sets, dict):
        return list(sparse_sets.keys())
    if hasattr(sparse_sets, "columns"):
        return list(sparse_sets.columns)
    return [f"sparse_set_{index}" for index, _ in enumerate(sparse_sets)]


def _append_feature_schema(
    feature_schema: Any | None,
    dense_width: int,
    sparse_names: list[str],
    neural_dim: int,
) -> dict[str, Any] | None:
    if feature_schema is None:
        return None

    base_payload = _rust_feature_schema_payload(
        feature_schema=feature_schema,
        dense_width=dense_width,
        sparse_names=sparse_names,
    )
    return {
        "names": [*base_payload["names"], *_neural_feature_names(neural_dim)],
        "kinds": [*base_payload["kinds"], *(["Numeric"] * neural_dim)],
    }
