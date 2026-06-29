from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .config import FuzzyKernel, LeafPredictor, Objective

try:  # pragma: no cover - exercised when the optional sklearn extra is installed.
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover - lightweight fallback for core installs.

    class BaseEstimator:  # type: ignore[no-redef]
        pass


from ._artifacts import require_artifact_payload, versioned_artifact_payload
from ._native import CartoBoostRanker as _NativeRankerModel
from .regressor import (
    _as_1d_float_array,
    _as_sample_weight_array,
    _encode_sparse_columns,
    _encoded_feature_schema,
    _feature_schema_metadata,
    _fit_transform_categorical_features,
    _is_empty_sparse_sets,
    _is_valid_splitter_name,
    _json_attr,
    _normalize_sparse_sets,
    _resolve_linear_leaf_features,
    _rust_feature_schema_json,
    _sparse_names_from_feature_schema,
    _transform_categorical_features,
)
from .tensorboard import write_training_history


class CartoBoostRanker(BaseEstimator):
    """Sklearn-style grouped learning-to-rank estimator backed by Rust.

    Training uses native pairwise logistic or LambdaRank objectives. Pass
    `groups` as a shorter list of group sizes or as one query id per row, or
    use `group_col` to name a column in `X`. Rows for each query group must be
    contiguous.

    Example:
        >>> ranker = CartoBoostRanker(n_estimators=8, max_depth=1, splitters=["axis"])
        >>> X = [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]]
        >>> ranker.fit(X, [0.0, 1.0, 3.0, 0.0, 2.0, 4.0], groups=[3, 3])
        CartoBoostRanker(...)
        >>> ranker.score_groups(X, [0.0, 1.0, 3.0, 0.0, 2.0, 4.0], groups=[3, 3])["ndcg"] > 0.0
        True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        min_gain: float = 1e-8,
        objective: Objective = Objective.LAMBDARANK,
        group_col: str | int | None = None,
        splitters: list[str] | None = None,
        leaf_predictor: LeafPredictor = LeafPredictor.CONSTANT,
        linear_leaf_features: list[str] | None = None,
        fuzzy: bool = False,
        fuzzy_bandwidth: float = 0.0,
        fuzzy_kernel: FuzzyKernel = FuzzyKernel.LINEAR,
        l2_regularization: float = 1.0,
        constant_l2_regularization: float = 0.0,
        random_state: int | None = None,
        n_threads: int | None = None,
        tensorboard_log_dir: str | Path | None = None,
        tensorboard_run_name: str | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.objective = objective
        self.group_col = group_col
        self.splitters = splitters
        self.leaf_predictor = leaf_predictor
        self.linear_leaf_features = linear_leaf_features
        self.fuzzy = fuzzy
        self.fuzzy_bandwidth = fuzzy_bandwidth
        self.fuzzy_kernel = fuzzy_kernel
        self.l2_regularization = l2_regularization
        self.constant_l2_regularization = constant_l2_regularization
        self.random_state = random_state
        self.n_threads = n_threads
        self.tensorboard_log_dir = tensorboard_log_dir
        self.tensorboard_run_name = tensorboard_run_name
        self._model: Any | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-compatible constructor parameters.

        Example:
            >>> CartoBoostRanker(objective="pairwise_logit").get_params()["objective"]
            'pairwise_logit'
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "objective": self.objective,
            "group_col": self.group_col,
            "splitters": self.splitters,
            "leaf_predictor": self.leaf_predictor,
            "linear_leaf_features": self.linear_leaf_features,
            "fuzzy": self.fuzzy,
            "fuzzy_bandwidth": self.fuzzy_bandwidth,
            "fuzzy_kernel": self.fuzzy_kernel,
            "l2_regularization": self.l2_regularization,
            "constant_l2_regularization": self.constant_l2_regularization,
            "random_state": self.random_state,
            "n_threads": self.n_threads,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "tensorboard_run_name": self.tensorboard_run_name,
        }

    def set_params(self, **params: Any) -> CartoBoostRanker:
        """Set sklearn-compatible constructor parameters and clear fitted state.

        Example:
            >>> CartoBoostRanker().set_params(n_estimators=3).n_estimators
            3
        """
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self._validate_params()
        self._model = None
        return self

    def fit(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        *,
        groups: Iterable[Any] | None = None,
        group_col: str | int | None = None,
        sample_weight: Iterable[float] | None = None,
        feature_schema: Any | None = None,
        sparse_sets: Any | None = None,
    ) -> CartoBoostRanker:
        """Fit native grouped pairwise or LambdaRank trees and return ``self``.

        Example:
            >>> X = [["q1", 0.0], ["q1", 1.0], ["q2", 0.0], ["q2", 1.0]]
            >>> CartoBoostRanker(group_col=0, n_estimators=2).fit(X, [0.0, 1.0, 0.0, 2.0])
            CartoBoostRanker(...)
        """
        self._validate_params()
        group_col = self.group_col if group_col is None else group_col
        X_model, group_values, model_feature_schema = _extract_group_column_and_schema(
            X,
            group_col,
            feature_schema,
        )
        targets_array = _as_1d_float_array(y)
        dense_array, categorical_encoder, feature_names = _fit_transform_categorical_features(
            X_model,
            targets_array,
            model_feature_schema,
            sample_weight=sample_weight,
        )
        if dense_array.shape[0] != targets_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        group_sizes = _normalize_groups(
            groups if groups is not None else group_values,
            dense_array.shape[0],
            prefer_query_ids=groups is None and group_values is not None,
        )
        weights_array = _as_sample_weight_array(sample_weight, targets_array.shape[0])
        sparse_columns, sparse_names = _normalize_sparse_sets(sparse_sets, targets_array.shape[0])
        sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        encoded_feature_schema = _encoded_feature_schema(
            model_feature_schema,
            categorical_encoder,
            dense_array.shape[1],
        )
        schema_json = _rust_feature_schema_json(
            encoded_feature_schema,
            dense_array.shape[1],
            sparse_names,
        )
        schema_metadata = _feature_schema_metadata(model_feature_schema)

        self.n_features_in_ = (
            int(categorical_encoder["original_feature_count"])
            if categorical_encoder
            else dense_array.shape[1]
        )
        self.encoded_n_features_in_ = dense_array.shape[1]
        self.n_sparse_sets_in_ = len(sparse_columns)
        self.sparse_set_names_ = sparse_names
        self.groups_ = group_sizes
        self.feature_schema_ = schema_metadata
        self.categorical_encoder_ = categorical_encoder
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        model = _NativeRankerModel(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            min_gain=float(self.min_gain),
            objective=str(self.objective),
            splitters=list(self.splitters or ["auto"]),
            leaf_predictor=str(self.leaf_predictor),
            linear_leaf_features=_resolve_linear_leaf_features(
                self.linear_leaf_features,
                dense_array.shape[1],
            ),
            l2_regularization=float(self.l2_regularization),
            constant_l2_regularization=float(self.constant_l2_regularization),
            fuzzy=bool(self.fuzzy),
            fuzzy_bandwidth=float(self.fuzzy_bandwidth),
            fuzzy_kernel=str(self.fuzzy_kernel),
            n_threads=None if self.n_threads is None else int(self.n_threads),
        )
        model.fit_arrays(
            dense_array,
            targets_array,
            group_sizes,
            None
            if weights_array is None
            else np.ascontiguousarray(weights_array, dtype=np.float64),
            sparse_offsets,
            sparse_ids,
            schema_json,
        )
        self._model = model
        self.feature_schema_ = (
            json.loads(schema_json) if schema_json is not None else schema_metadata
        )
        self.metadata_ = _json_attr(model, "metadata_json")
        self.training_config_ = _json_attr(model, "training_config_json")
        self.training_history_ = _json_attr(model, "training_history_json") or []
        write_training_history(
            model,
            self.tensorboard_log_dir,
            run_name=self.tensorboard_run_name,
        )
        self.requires_sparse_sets_ = bool(
            getattr(model, "requires_sparse_sets", bool(sparse_columns))
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: Iterable[Iterable[float]], sparse_sets: Any | None = None) -> np.ndarray:
        """Return one relevance score per row.

        Example:
            >>> ranker = CartoBoostRanker(n_estimators=2, splitters=["axis"])
            >>> ranker.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 3.0], groups=[3])
            CartoBoostRanker(...)
            >>> ranker.predict([[2.0]]).shape
            (1,)
        """
        if self._model is None:
            raise RuntimeError("CartoBoostRanker is not fitted")
        X_model = _maybe_drop_prediction_group_column(
            X,
            getattr(self, "group_col", None),
            getattr(self, "n_features_in_", None),
        )
        dense_array, sparse_offsets, sparse_ids = self._prediction_inputs(X_model, sparse_sets)
        return np.asarray(
            self._model.predict_arrays(dense_array, sparse_offsets, sparse_ids),
            dtype=float,
        )

    def score_groups(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        *,
        groups: Iterable[Any] | None = None,
        group_col: str | int | None = None,
        sparse_sets: Any | None = None,
    ) -> dict[str, float]:
        """Return grouped ``ndcg``, ``map``, and ``mrr`` metrics.

        Example:
            >>> ranker = CartoBoostRanker(n_estimators=2, splitters=["axis"])
            >>> X = [[0.0], [1.0], [2.0]]
            >>> ranker.fit(X, [0.0, 1.0, 3.0], groups=[3])
            CartoBoostRanker(...)
            >>> sorted(ranker.score_groups(X, [0.0, 1.0, 3.0], groups=[3]))
            ['map', 'mrr', 'ndcg']
        """
        if self._model is None:
            raise RuntimeError("CartoBoostRanker is not fitted")
        group_col = self.group_col if group_col is None else group_col
        if groups is None:
            X_model, group_values = _extract_group_column(X, group_col)
        else:
            X_model = _maybe_drop_prediction_group_column(
                X,
                group_col,
                getattr(self, "n_features_in_", None),
            )
            group_values = None
        dense_array, sparse_offsets, sparse_ids = self._prediction_inputs(X_model, sparse_sets)
        targets_array = _as_1d_float_array(y)
        if dense_array.shape[0] != targets_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        group_sizes = _normalize_groups(
            groups if groups is not None else group_values,
            dense_array.shape[0],
            prefer_query_ids=groups is None and group_values is not None,
        )
        metrics = self._model.metrics_arrays(
            dense_array,
            targets_array,
            group_sizes,
            sparse_offsets,
            sparse_ids,
        )
        return {str(key): float(value) for key, value in metrics.items()}

    def score(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        *,
        groups: Iterable[Any] | None = None,
        group_col: str | int | None = None,
        sparse_sets: Any | None = None,
    ) -> float:
        """Return grouped NDCG for sklearn-style scoring.

        Example:
            >>> ranker = CartoBoostRanker(n_estimators=2, splitters=["axis"])
            >>> X = [[0.0], [1.0], [2.0]]
            >>> ranker.fit(X, [0.0, 1.0, 3.0], groups=[3])
            CartoBoostRanker(...)
            >>> ranker.score(X, [0.0, 1.0, 3.0], groups=[3]) >= 0.0
            True
        """

        return self.score_groups(
            X,
            y,
            groups=groups,
            group_col=group_col,
            sparse_sets=sparse_sets,
        )["ndcg"]

    def save(self, path: str | Path) -> None:
        """Write a ranker artifact, including categorical encoders when present.

        Example:
            >>> ranker = CartoBoostRanker(n_estimators=2, splitters=["axis"])
            >>> ranker.fit([[0.0], [1.0]], [0.0, 1.0], groups=[2])
            CartoBoostRanker(...)
            >>> ranker.save("route-ranker.json")
        """
        if self._model is None:
            raise RuntimeError("CartoBoostRanker is not fitted")
        path = Path(path)
        with tempfile.TemporaryDirectory() as temp_dir:
            native_path = Path(temp_dir) / "native-ranker.json"
            self._model.save(native_path)
            native_payload = json.loads(native_path.read_text(encoding="utf-8"))
        payload = versioned_artifact_payload(
            "cartoboost.ranker",
            categorical_encoder=getattr(self, "categorical_encoder_", None),
            group_col=_jsonable_group_col(self.group_col),
            native_model=native_payload,
        )
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def save_weights(self, path: str | Path, *, format: str = "auto") -> None:
        """Fail loudly for unsupported ranker weight/export artifacts.

        Example:
            >>> ranker = CartoBoostRanker(n_estimators=2, splitters=["axis"])
            >>> ranker.fit([[0.0], [1.0]], [0.0, 1.0], groups=[2])
            CartoBoostRanker(...)
            >>> ranker.save_weights("ranker.onnx", format="onnx")
            Traceback (most recent call last):
            ...
            NotImplementedError: CartoBoostRanker does not support save_weights or ONNX export; ...
        """
        if self._model is None:
            raise RuntimeError("CartoBoostRanker is not fitted")
        raise NotImplementedError(
            "CartoBoostRanker does not support save_weights or ONNX export; "
            "use save(path) for the supported ranker artifact"
        )

    @classmethod
    def load(cls, path: str | Path) -> CartoBoostRanker:
        """Load a ranker artifact written by ``save``.

        Example:
            >>> restored = CartoBoostRanker.load("route-ranker.json")
            >>> restored.predict([[1.0]]).shape
            (1,)
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_type") == "cartoboost.ranker":
            require_artifact_payload(payload, "cartoboost.ranker")
            with tempfile.TemporaryDirectory() as temp_dir:
                native_path = Path(temp_dir) / "native-ranker.json"
                native_path.write_text(
                    json.dumps(payload["native_model"], sort_keys=True),
                    encoding="utf-8",
                )
                native_model = _NativeRankerModel.load(native_path)
            estimator = cls._from_native_model(native_model)
            estimator.group_col = payload.get("group_col")
            estimator.categorical_encoder_ = payload.get("categorical_encoder")
            if estimator.categorical_encoder_:
                estimator.n_features_in_ = int(
                    estimator.categorical_encoder_["original_feature_count"]
                )
                estimator.encoded_n_features_in_ = native_model.feature_count
            return estimator
        native_model = _NativeRankerModel.load(path)
        return cls._from_native_model(native_model)

    @classmethod
    def _from_native_model(cls, native_model: Any) -> CartoBoostRanker:
        estimator = cls(
            n_estimators=native_model.n_estimators,
            learning_rate=native_model.learning_rate,
            max_depth=native_model.max_depth,
            min_samples_leaf=native_model.min_samples_leaf,
            min_gain=native_model.min_gain,
            objective=str(native_model.objective),
            splitters=list(native_model.splitters),
        )
        estimator._model = native_model
        estimator.n_features_in_ = native_model.feature_count
        estimator.encoded_n_features_in_ = native_model.feature_count
        estimator.categorical_encoder_ = None
        estimator.feature_schema_ = _json_attr(native_model, "feature_schema_json")
        estimator.sparse_set_names_ = _sparse_names_from_feature_schema(estimator.feature_schema_)
        estimator.n_sparse_sets_in_ = len(estimator.sparse_set_names_)
        estimator.metadata_ = _json_attr(native_model, "metadata_json")
        estimator.training_config_ = _json_attr(native_model, "training_config_json")
        estimator.training_history_ = _json_attr(native_model, "training_history_json") or []
        estimator.requires_sparse_sets_ = bool(getattr(native_model, "requires_sparse_sets", False))
        estimator.is_fitted_ = True
        return estimator

    def _prediction_inputs(
        self,
        X: Iterable[Iterable[float]],
        sparse_sets: Any | None,
    ) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
        expected_sparse_count = getattr(self, "n_sparse_sets_in_", 0)
        dense_array = _transform_categorical_features(
            X,
            getattr(self, "categorical_encoder_", None),
        )
        expected_dense = getattr(self, "encoded_n_features_in_", self.n_features_in_)
        if hasattr(self, "encoded_n_features_in_") and dense_array.shape[1] != expected_dense:
            raise ValueError(
                f"encoded X has {dense_array.shape[1]} features, but CartoBoostRanker was fitted "
                f"with {expected_dense} encoded features"
            )
        if not getattr(self, "requires_sparse_sets_", False):
            sparse_columns: list[list[list[int]]] = []
            sparse_offsets: list[list[int]] = []
            sparse_ids: list[list[int]] = []
        elif _is_empty_sparse_sets(sparse_sets) and expected_sparse_count == 0:
            sparse_columns = []
            sparse_offsets = []
            sparse_ids = []
        else:
            sparse_columns, _ = _normalize_sparse_sets(
                sparse_sets,
                dense_array.shape[0],
                getattr(self, "sparse_set_names_", None),
            )
            sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        if sparse_columns and len(sparse_columns) != expected_sparse_count:
            raise ValueError(
                f"sparse_sets has {len(sparse_columns)} columns, but CartoBoostRanker was "
                f"fitted with {expected_sparse_count}"
            )
        if not sparse_columns and getattr(self, "requires_sparse_sets_", False):
            raise ValueError("sparse_sets are required for prediction with this sparse-list model")
        return dense_array, sparse_offsets, sparse_ids

    def _validate_params(self) -> None:
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive")
        learning_rate = float(self.learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative")
        if int(self.min_samples_leaf) <= 0:
            raise ValueError("min_samples_leaf must be positive")
        min_gain = float(self.min_gain)
        if not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and non-negative")
        if str(self.objective) not in {"pairwise", "pairwise_logit", "lambdarank", "lambda_rank"}:
            raise ValueError("objective must be 'pairwise_logit' or 'lambdarank'")
        if self.leaf_predictor not in {LeafPredictor.CONSTANT, LeafPredictor.LINEAR}:
            raise ValueError("leaf_predictor must be 'constant' or 'linear'")
        if float(self.l2_regularization) < 0 or not math.isfinite(float(self.l2_regularization)):
            raise ValueError("l2_regularization must be finite and non-negative")
        constant_l2 = float(self.constant_l2_regularization)
        if constant_l2 < 0 or not math.isfinite(constant_l2):
            raise ValueError("constant_l2_regularization must be finite and non-negative")
        if float(self.fuzzy_bandwidth) < 0 or not math.isfinite(float(self.fuzzy_bandwidth)):
            raise ValueError("fuzzy_bandwidth must be finite and non-negative")
        if str(self.fuzzy_kernel) not in {
            "linear",
            "triangular",
            "gaussian",
            "exponential",
            "bisquare",
            "epanechnikov",
            "tricube",
        }:
            raise ValueError(
                "fuzzy_kernel must be 'linear', 'gaussian', 'exponential', "
                "'bisquare', 'epanechnikov', or 'tricube'"
            )
        if self.n_threads is not None and int(self.n_threads) <= 0:
            raise ValueError("n_threads must be positive")
        if self.splitters is not None:
            if isinstance(self.splitters, str):
                raise ValueError("splitters must be a list of splitter names")
            unknown = [
                splitter for splitter in self.splitters if not _is_valid_splitter_name(splitter)
            ]
            if unknown:
                raise ValueError(f"unknown splitter(s): {unknown}")


def _extract_group_column(values: Any, group_col: str | int | None) -> tuple[Any, Any | None]:
    if group_col is None:
        return values, None
    columns = getattr(values, "columns", None)
    if columns is not None:
        if group_col not in columns:
            raise ValueError(f"group_col {group_col!r} is not a column in X")
        groups = values[group_col]
        return values.drop(columns=[group_col]), groups
    array = np.asarray(values)
    if not isinstance(group_col, int):
        raise ValueError("group_col must be an integer index for array-like X")
    if array.ndim != 2:
        raise ValueError("X must be a 2D array when group_col is an integer index")
    if group_col < 0 or group_col >= array.shape[1]:
        raise ValueError("group_col index is out of range")
    groups = array[:, group_col]
    dense = np.delete(array, group_col, axis=1)
    return dense, groups


def _extract_group_column_and_schema(
    values: Any,
    group_col: str | int | None,
    feature_schema: Any | None,
) -> tuple[Any, Any | None, Any | None]:
    if group_col is None:
        return values, None, feature_schema
    group_index = _group_column_index(values, group_col)
    x_model, groups = _extract_group_column(values, group_col)
    return x_model, groups, _drop_dense_schema_entry(feature_schema, group_index, group_col)


def _maybe_drop_prediction_group_column(
    values: Any,
    group_col: str | int | None,
    expected_features: int | None,
) -> Any:
    if group_col is None or expected_features is None:
        return values
    columns = getattr(values, "columns", None)
    if columns is not None:
        if group_col in columns and len(columns) == int(expected_features) + 1:
            return values.drop(columns=[group_col])
        return values
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] == int(expected_features):
        return values
    if array.shape[1] == int(expected_features) + 1:
        if not isinstance(group_col, int):
            raise ValueError("group_col must be an integer index for array-like X")
        if group_col < 0 or group_col >= array.shape[1]:
            raise ValueError("group_col index is out of range")
        return np.delete(array, group_col, axis=1)
    return values


def _group_column_index(values: Any, group_col: str | int) -> int:
    columns = getattr(values, "columns", None)
    if columns is not None:
        if group_col not in columns:
            raise ValueError(f"group_col {group_col!r} is not a column in X")
        return int(list(columns).index(group_col))
    array = np.asarray(values)
    if not isinstance(group_col, int):
        raise ValueError("group_col must be an integer index for array-like X")
    if array.ndim != 2:
        raise ValueError("X must be a 2D array when group_col is an integer index")
    if group_col < 0 or group_col >= array.shape[1]:
        raise ValueError("group_col index is out of range")
    return int(group_col)


def _drop_dense_schema_entry(
    feature_schema: Any | None,
    group_index: int,
    group_col: str | int,
) -> Any | None:
    if feature_schema is None:
        return None
    if hasattr(feature_schema, "to_dict"):
        feature_schema = feature_schema.to_dict()
    if isinstance(feature_schema, dict) and "names" in feature_schema and "kinds" in feature_schema:
        names = list(feature_schema["names"])
        kinds = list(feature_schema["kinds"])
        if group_index >= len(names) or group_index >= len(kinds):
            return feature_schema
        return {
            **feature_schema,
            "names": names[:group_index] + names[group_index + 1 :],
            "kinds": kinds[:group_index] + kinds[group_index + 1 :],
        }
    if isinstance(feature_schema, dict) and (
        "dense" in feature_schema or "sparse_sets" in feature_schema
    ):
        dense = list(feature_schema.get("dense", []))
        if group_index < len(dense):
            dense = dense[:group_index] + dense[group_index + 1 :]
        return {**feature_schema, "dense": dense}
    if isinstance(feature_schema, dict):
        keys = list(feature_schema)
        if isinstance(group_col, str) and group_col in feature_schema:
            return {key: value for key, value in feature_schema.items() if key != group_col}
        if group_index < len(keys):
            dropped = keys[group_index]
            return {key: value for key, value in feature_schema.items() if key != dropped}
    return feature_schema


def _jsonable_group_col(group_col: str | int | None) -> str | int | None:
    if isinstance(group_col, np.generic):
        group_col = group_col.item()
    if group_col is None or isinstance(group_col, str | int):
        return group_col
    raise TypeError("group_col must be a string or integer selector to save a ranker artifact")


def _normalize_groups(
    values: Any | None,
    row_count: int,
    *,
    prefer_query_ids: bool = False,
) -> list[int]:
    if values is None:
        raise ValueError("provide groups or group_col for ranking")
    array = _as_group_array(values)
    if array.shape[0] == 0:
        raise ValueError("groups must not be empty")
    if not prefer_query_ids:
        size_values = _try_group_sizes(array)
        if size_values is not None and sum(size_values) == row_count:
            return size_values
    if array.shape[0] != row_count:
        sizes = [int(value) for value in array.tolist()]
        if any(size <= 0 for size in sizes):
            raise ValueError("group sizes must be positive")
        if sum(sizes) != row_count:
            raise ValueError("group sizes must sum to the number of rows")
        return sizes
    sizes = []
    seen_closed = set()
    current = array[0]
    count = 0
    for value in array.tolist():
        if value == current:
            count += 1
            continue
        seen_closed.add(current)
        if value in seen_closed:
            raise ValueError("query id groups must be contiguous")
        sizes.append(count)
        current = value
        count = 1
    sizes.append(count)
    return sizes


def _try_group_sizes(values: np.ndarray) -> list[int] | None:
    sizes = []
    for value in values.tolist():
        if isinstance(value, bool):
            return None
        try:
            size = int(value)
        except (TypeError, ValueError):
            return None
        if size != value or size <= 0:
            return None
        sizes.append(size)
    return sizes


def _as_group_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("groups must be a 1D array of group sizes or query ids")
        array = np.empty(values.shape[0], dtype=object)
        array[:] = values.tolist()
        return array
    items = list(values)
    array = np.empty(len(items), dtype=object)
    array[:] = items
    if array.ndim != 1:
        raise ValueError("groups must be a 1D array of group sizes or query ids")
    return array
