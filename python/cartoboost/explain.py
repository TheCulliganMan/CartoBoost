from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import ExplanationAlgorithm, ExplanationDecomposition
from .regressor import _tabular_column_mapping


def make_shap_explainer(
    model: Any,
    background: Any | None = None,
    *,
    sparse_sets: Any | None = None,
    sparse_id_vocabulary: dict[str, list[int]] | None = None,
    algorithm: ExplanationAlgorithm = ExplanationAlgorithm.AUTO,
    feature_names: list[str] | None = None,
    decomposition: ExplanationDecomposition = ExplanationDecomposition.FEATURES,
    **kwargs: Any,
) -> Any:
    """Build a SHAP explainer for CartoBoost predictions.

    SHAP is an optional dependency. Install ``shap`` directly before using
    this helper.
    """
    shap = _import_shap()
    _validate_shap_model(model)
    decomposition = _validate_decomposition(decomposition)
    if decomposition == ExplanationDecomposition.WEIGHTS:
        if background is None:
            raise ValueError("background is required for decomposition='weights'")
        adapter = _AdditiveWeightShapAdapter(
            model,
            background,
            sparse_sets=sparse_sets,
        )
        return _AdditiveWeightShapExplainer(shap, adapter, **kwargs)
    if sparse_sets is not None:
        if background is None:
            raise ValueError("background is required for sparse SHAP")
        adapter = _SparseSetShapAdapter(
            model,
            background,
            sparse_sets,
            feature_names=feature_names,
            sparse_id_vocabulary=sparse_id_vocabulary,
        )
        explainer = shap.Explainer(
            adapter.predict,
            adapter.background,
            algorithm=algorithm,
            feature_names=adapter.feature_names,
            **kwargs,
        )
        explainer.cartoboost_sparse_adapter = adapter
        return explainer
    if getattr(model, "requires_sparse_sets_", False):
        raise ValueError(
            "sparse_sets must be provided when explaining a model that requires sparse_sets"
        )
    names = feature_names or _model_feature_names(model)
    if background is None:
        return _NativePathDependentShapExplainer(shap, model, names, **kwargs)
    tree_ensemble = _tree_shap_ensemble(model)
    if tree_ensemble is not None:
        return _NativeTreeShapExplainer(
            shap.TreeExplainer(
                tree_ensemble,
                data=_as_2d_array(background),
                model_output="raw",
                feature_perturbation="interventional",
                feature_names=names,
            )
        )
    return shap.Explainer(
        model,
        background,
        algorithm=algorithm,
        feature_names=names,
        **kwargs,
    )


def explain_shap(
    model: Any,
    X: Any,
    *,
    background: Any | None = None,
    sparse_sets: Any | None = None,
    background_sparse_sets: Any | None = None,
    sparse_id_vocabulary: dict[str, list[int]] | None = None,
    algorithm: ExplanationAlgorithm = ExplanationAlgorithm.AUTO,
    feature_names: list[str] | None = None,
    decomposition: ExplanationDecomposition = ExplanationDecomposition.FEATURES,
    **kwargs: Any,
) -> Any:
    """Return a SHAP Explanation for CartoBoost predictions."""
    if sparse_sets is not None or background_sparse_sets is not None:
        if sparse_sets is None or background_sparse_sets is None:
            raise ValueError(
                "sparse_sets and background_sparse_sets must both be provided for sparse SHAP"
            )
        sparse_id_vocabulary = sparse_id_vocabulary or _merged_sparse_vocabulary(
            background_sparse_sets,
            sparse_sets,
            getattr(model, "sparse_set_names_", None),
        )
    explainer = make_shap_explainer(
        model,
        background,
        sparse_sets=background_sparse_sets,
        sparse_id_vocabulary=sparse_id_vocabulary,
        algorithm=algorithm,
        feature_names=feature_names,
        decomposition=decomposition,
        **kwargs,
    )
    additive_adapter = getattr(explainer, "cartoboost_additive_adapter", None)
    if additive_adapter is not None:
        return explainer(X, sparse_sets=sparse_sets)
    adapter = getattr(explainer, "cartoboost_sparse_adapter", None)
    if adapter is not None:
        return explainer(adapter.transform(X, sparse_sets))
    return explainer(X)


def _import_shap() -> Any:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "SHAP support requires the optional 'shap' package. Install it with `pip install shap`."
        ) from exc
    return shap


def _validate_shap_model(model: Any) -> None:
    if getattr(model, "_model", None) is None:
        raise RuntimeError("CartoBoostRegressor is not fitted")


def _validate_decomposition(
    decomposition: ExplanationDecomposition | str,
) -> ExplanationDecomposition:
    if isinstance(decomposition, ExplanationDecomposition):
        return decomposition
    try:
        return ExplanationDecomposition(str(decomposition))
    except ValueError as exc:
        raise ValueError("decomposition must be either 'features' or 'weights'") from exc


def _model_feature_names(model: Any) -> list[str] | None:
    names = getattr(model, "feature_name_", None)
    if names is not None:
        return [str(name) for name in names]

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    schema = getattr(model, "feature_schema_", None)
    if isinstance(schema, dict):
        schema_names = schema.get("names")
        if isinstance(schema_names, list):
            feature_count = getattr(model, "n_features_in_", len(schema_names))
            return [str(name) for name in schema_names[:feature_count]]

    return None


class _AdditiveWeightShapExplainer:
    """Exact SHAP explainer for CartoBoost's additive prediction components.

    ``predict_additive_values`` returns the initial prediction followed by one
    contribution per fitted tree.  The prediction is their sum, so each
    component is an independent additive feature.  Its SHAP value is therefore
    its value minus its background mean; the expected prediction is the sum of
    those means.  Computing this directly is exact and avoids routing a
    potentially very-wide tree-weight matrix through SHAP's generic permutation
    explainer.
    """

    def __init__(self, shap: Any, adapter: _AdditiveWeightShapAdapter, **kwargs: Any) -> None:
        self._shap = shap
        self.cartoboost_additive_adapter = adapter
        self.expected_value = float(np.sum(adapter.background_mean))
        # Retain this familiar attribute for callers that inspect explainers.
        self.feature_names = adapter.feature_names
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                "weight decomposition is computed exactly and does not accept "
                f"SHAP explainer options: {unsupported}"
            )

    def __call__(self, X: Any, *, sparse_sets: Any | None = None, **kwargs: Any) -> Any:
        values = self.cartoboost_additive_adapter.transform(X, sparse_sets=sparse_sets)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                "weight decomposition is computed exactly and does not accept "
                f"SHAP call options: {unsupported}"
            )
        return self._shap.Explanation(
            values=values - self.cartoboost_additive_adapter.background_mean,
            base_values=np.full(values.shape[0], self.expected_value, dtype=float),
            data=values,
            feature_names=self.feature_names,
        )

    def shap_values(self, X: Any, *, sparse_sets: Any | None = None) -> np.ndarray:
        """Return the exact component SHAP values as a NumPy array."""
        return np.asarray(self(X, sparse_sets=sparse_sets).values)


class _NativeTreeShapExplainer:
    """Expose SHAP's native TreeExplainer for a CartoBoost axis-tree ensemble."""

    def __init__(self, explainer: Any) -> None:
        self.explainer = explainer
        self.expected_value = explainer.expected_value

    def __call__(self, X: Any, **kwargs: Any) -> Any:
        return self.explainer(_as_2d_array(X), **kwargs)

    def shap_values(self, X: Any, **kwargs: Any) -> Any:
        return self.explainer.shap_values(_as_2d_array(X), **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.explainer, name)


class _NativePathDependentShapExplainer:
    """Format Rust-native path-dependent TreeSHAP as standard SHAP objects."""

    def __init__(
        self,
        shap: Any,
        model: Any,
        feature_names: list[str] | None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                f"native path-dependent SHAP does not accept explainer options: {unsupported}"
            )
        native_model = getattr(model, "_model", None)
        base_value = getattr(native_model, "feature_contribution_base_value", None)
        if not callable(base_value):
            raise RuntimeError(
                "cartoboost._native model is missing native feature contributions; "
                "rebuild the native extension with `maturin develop`"
            )
        self._shap = shap
        self.model = model
        self.expected_value = float(base_value())
        self.feature_names = feature_names

    def __call__(self, X: Any, **kwargs: Any) -> Any:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                f"native path-dependent SHAP does not accept call options: {unsupported}"
            )
        contributions = np.asarray(self.model.predict_feature_contributions(X), dtype=float)
        return self._shap.Explanation(
            values=contributions[:, :-1],
            base_values=np.full(contributions.shape[0], self.expected_value, dtype=float),
            data=_as_explanation_data(X),
            feature_names=self.feature_names,
        )

    def shap_values(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Return the native feature contributions without the bias column."""
        return np.asarray(self(X, **kwargs).values, dtype=float)


class _AdditiveWeightShapAdapter:
    def __init__(
        self,
        model: Any,
        background: Any,
        *,
        sparse_sets: Any | None,
    ) -> None:
        self.model = model
        self.background = self.transform(background, sparse_sets=sparse_sets)
        self.background_mean = np.mean(self.background, axis=0)
        self.feature_names = ["init_prediction"] + [
            f"tree_{idx}" for idx in range(max(0, self.background.shape[1] - 1))
        ]

    def transform(self, dense: Any, *, sparse_sets: Any | None = None) -> np.ndarray:
        values = np.asarray(
            self.model.predict_additive_values(dense, sparse_sets=sparse_sets),
            dtype=float,
        )
        if values.ndim != 2:
            raise ValueError("additive values must be a 2D matrix")
        return values

    def predict(self, additive_values: Any) -> np.ndarray:
        values = _as_2d_array(additive_values)
        return np.sum(values, axis=1)


def _tree_shap_ensemble(model: Any) -> dict[str, Any] | None:
    """Return a NumPy-backed SHAP tree model when native routing permits it."""
    native_model = getattr(model, "_model", None)
    export = getattr(native_model, "tree_shap_ensemble_json", None)
    if export is None:
        return None
    try:
        payload = json.loads(export())
    except ValueError:
        # Structured splitters and linear leaves require CartoBoost's general
        # SHAP path; the native exporter deliberately rejects lossy conversion.
        return None
    for tree in payload["trees"]:
        for name in (
            "children_left",
            "children_right",
            "children_default",
            "features",
        ):
            tree[name] = np.asarray(tree[name], dtype=np.int32)
        tree["thresholds"] = np.asarray(tree["thresholds"], dtype=np.float64)
        tree["values"] = np.asarray(tree["values"], dtype=np.float64)
        tree["node_sample_weight"] = np.asarray(tree["node_sample_weight"], dtype=np.float64)
    return payload


class _SparseSetShapAdapter:
    def __init__(
        self,
        model: Any,
        background: Any,
        sparse_sets: Any,
        *,
        feature_names: list[str] | None,
        sparse_id_vocabulary: dict[str, list[int]] | None,
    ) -> None:
        self.model = model
        self.dense_width = int(getattr(model, "n_features_in_", 0))
        if self.dense_width <= 0:
            raise RuntimeError("CartoBoostRegressor is not fitted")
        self.names, self.vocabulary = _sparse_vocabulary(
            sparse_sets,
            getattr(model, "sparse_set_names_", None),
            sparse_id_vocabulary,
        )
        self.background = self.transform(background, sparse_sets)
        dense_names = feature_names or _model_feature_names(model)
        if dense_names is None:
            dense_names = [f"feature_{idx}" for idx in range(self.dense_width)]
        sparse_names = [f"{name}={ident}" for name in self.names for ident in self.vocabulary[name]]
        self.feature_names = [*dense_names, *sparse_names]

    def transform(self, dense: Any, sparse_sets: Any) -> np.ndarray:
        rows = _as_2d_array(dense)
        columns = _normalize_sparse_columns(sparse_sets, rows.shape[0], self.names)
        encoded_columns = []
        for name, column in zip(self.names, columns, strict=True):
            vocab = self.vocabulary[name]
            encoded = np.zeros((rows.shape[0], len(vocab)), dtype=float)
            id_to_idx = {ident: idx for idx, ident in enumerate(vocab)}
            for row_idx, row_ids in enumerate(column):
                for ident in row_ids:
                    idx = id_to_idx.get(ident)
                    if idx is not None:
                        encoded[row_idx, idx] = 1.0
            encoded_columns.append(encoded)
        if not encoded_columns:
            return rows
        return np.concatenate([rows, *encoded_columns], axis=1)

    def predict(self, augmented: Any) -> np.ndarray:
        rows = _as_2d_array(augmented)
        dense = rows[:, : self.dense_width]
        offset = self.dense_width
        sparse_sets: dict[str, list[list[int]]] = {}
        for name in self.names:
            vocab = self.vocabulary[name]
            width = len(vocab)
            flags = rows[:, offset : offset + width]
            sparse_sets[name] = [
                [ident for ident, active in zip(vocab, row, strict=True) if active >= 0.5]
                for row in flags
            ]
            offset += width
        return self.model.predict(dense, sparse_sets=sparse_sets)


def _as_2d_array(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2:
        raise ValueError("SHAP inputs must be 2D")
    return rows


def _as_explanation_data(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    rows = np.asarray(values)
    if rows.ndim != 2:
        raise ValueError("SHAP inputs must be 2D")
    return rows


def _sparse_vocabulary(
    sparse_sets: Any,
    expected_names: list[str] | None,
    explicit: dict[str, list[int]] | None,
) -> tuple[list[str], dict[str, list[int]]]:
    columns, names = _normalize_sparse_columns_with_names(sparse_sets, expected_names)
    if explicit is not None:
        vocab = {
            str(name): sorted({int(value) for value in values}) for name, values in explicit.items()
        }
        missing = [name for name in names if name not in vocab]
        if missing:
            raise ValueError(f"sparse_id_vocabulary is missing columns: {missing}")
        return names, {name: vocab[name] for name in names}
    return names, {
        name: sorted({ident for row in column for ident in row})
        for name, column in zip(names, columns, strict=True)
    }


def _merged_sparse_vocabulary(
    left: Any,
    right: Any,
    expected_names: list[str] | None,
) -> dict[str, list[int]]:
    left_columns, names = _normalize_sparse_columns_with_names(left, expected_names)
    right_columns, right_names = _normalize_sparse_columns_with_names(right, names)
    if right_names != names:
        raise ValueError("sparse_sets and background_sparse_sets columns must match")
    return {
        name: sorted(
            {ident for column in (left_column, right_column) for row in column for ident in row}
        )
        for name, left_column, right_column in zip(names, left_columns, right_columns, strict=True)
    }


def _normalize_sparse_columns(
    sparse_sets: Any,
    expected_rows: int,
    expected_names: list[str],
) -> list[list[list[int]]]:
    columns, _ = _normalize_sparse_columns_with_names(sparse_sets, expected_names)
    if any(len(column) != expected_rows for column in columns):
        raise ValueError("each sparse_sets column must match the dense row count")
    return columns


def _normalize_sparse_columns_with_names(
    sparse_sets: Any,
    expected_names: list[str] | None,
) -> tuple[list[list[list[int]]], list[str]]:
    if sparse_sets is None:
        raise ValueError("sparse_sets are required for sparse SHAP")
    if isinstance(sparse_sets, dict):
        mapping = {str(name): column for name, column in sparse_sets.items()}
        names = list(expected_names or mapping.keys())
        missing = [name for name in names if name not in mapping]
        if missing:
            raise ValueError(f"sparse_sets is missing columns: {missing}")
        items = [(name, mapping[name]) for name in names]
    elif (mapping := _tabular_column_mapping(sparse_sets)) is not None:
        names = list(expected_names or mapping.keys())
        missing = [name for name in names if name not in mapping]
        if missing:
            raise ValueError(f"sparse_sets is missing columns: {missing}")
        items = [(name, mapping[name]) for name in names]
    else:
        raw_columns = list(sparse_sets)
        if expected_names is not None:
            if len(raw_columns) != len(expected_names):
                raise ValueError(
                    f"sparse_sets has {len(raw_columns)} columns, expected {len(expected_names)}"
                )
            items = list(zip(expected_names, raw_columns, strict=True))
        else:
            items = [(f"sparse_set_{idx}", column) for idx, column in enumerate(sparse_sets)]
        names = [name for name, _ in items]
    return [
        [[_normalize_sparse_id(value) for value in row] for row in _sequence_values(column)]
        for _, column in items
    ], names


def _sequence_values(values: Any) -> Any:
    if hasattr(values, "to_list"):
        return values.to_list()
    if hasattr(values, "tolist"):
        return values.tolist()
    return values


def _normalize_sparse_id(value: Any) -> int:
    ident = int(value)
    if ident < 0 or float(value) != float(ident):
        raise ValueError("sparse_sets IDs must be non-negative integers")
    return ident
