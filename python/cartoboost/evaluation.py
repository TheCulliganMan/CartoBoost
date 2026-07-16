"""Evaluation protocol helpers for blocked CartoBoost validation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .accelerators import pairwise_squared_distances, workload_decision
from .config import Backend

__all__ = [
    "environmental_blocked_cv",
    "grouped_blocked_cv",
    "out_of_time_split",
    "spatial_buffered_cv",
    "spatial_blocked_cv",
    "spatial_grouped_cv",
    "temporal_blocked_cv",
]


def spatial_blocked_cv(
    coordinates: object,
    *,
    n_splits: int = 5,
    grid_shape: tuple[int, int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices using held-out spatial grid blocks.

    Coordinates can be one-dimensional or multi-dimensional. For dimensions
    beyond the first two, blocking uses the first two columns.

    Example:
        >>> folds = list(spatial_blocked_cv([[0.0], [1.0], [2.0], [3.0]], n_splits=2))
        >>> len(folds)
        2
    """

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] == 0:
        raise ValueError("coordinates must have shape (n_samples, n_dimensions)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must contain only finite values")
    _validate_n_splits(n_splits, coords.shape[0])

    if coords.shape[1] == 1:
        yield from _ordered_blocked_cv(coords[:, 0], n_splits)
        return

    rows, cols = _resolve_grid_shape(n_splits, grid_shape)
    x_bins = _coordinate_bins(coords[:, 0], cols)
    y_bins = _coordinate_bins(coords[:, 1], rows)
    block_ids = y_bins * cols + x_bins
    if np.unique(block_ids).size < n_splits:
        ranges = np.ptp(coords[:, :2], axis=0)
        yield from _ordered_blocked_cv(coords[:, int(np.argmax(ranges))], n_splits)
        return
    yield from grouped_blocked_cv(block_ids, n_splits=n_splits)


def spatial_buffered_cv(
    coordinates: object,
    *,
    n_splits: int = 5,
    buffer_radius: float,
    coordinate_cols: tuple[object, object] | list[object] | None = None,
    grid_shape: tuple[int, int] | None = None,
    random_state: int | None = None,
    coordinate_units: str = "projected",
    allow_degree_buffer: bool = False,
    backend: Backend | str = Backend.CPU,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield spatial blocked folds with a distance buffer around each test block.

    ``buffer_radius`` is measured in the coordinate units. Use projected
    coordinates such as meters or feet for meaningful buffers. Latitude and
    longitude degree buffers raise unless ``allow_degree_buffer=True``.

    Example:
        >>> coords = [[0.0, 0.0], [10.0, 0.0]]
        >>> folds = list(spatial_buffered_cv(coords, n_splits=2, buffer_radius=1.0))
        >>> len(folds)
        2
    """

    coords = _extract_coordinates(coordinates, coordinate_cols)
    radius = _validate_projected_buffer(buffer_radius, coordinate_units, allow_degree_buffer)
    block_splits = list(spatial_blocked_cv(coords, n_splits=n_splits, grid_shape=grid_shape))
    if random_state is not None:
        rng = np.random.default_rng(int(random_state))
        rng.shuffle(block_splits)

    for train_idx, test_idx in block_splits:
        yield _apply_spatial_buffer(coords, train_idx, test_idx, radius, str(backend))


def spatial_grouped_cv(
    coordinates: object,
    groups: object,
    *,
    n_splits: int = 5,
    buffer_radius: float = 0.0,
    coordinate_cols: tuple[object, object] | list[object] | None = None,
    random_state: int | None = None,
    coordinate_units: str = "projected",
    allow_degree_buffer: bool = False,
    backend: Backend | str = Backend.CPU,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield grouped folds with optional spatial buffering around test rows.

    Whole groups are held out together. When ``buffer_radius`` is positive,
    train rows from non-test groups are also removed if they are too close to
    any test coordinate.

    Example:
        >>> coords = [[0.0], [1.0], [10.0], [11.0]]
        >>> folds = list(spatial_grouped_cv(coords, ["a", "a", "b", "b"], n_splits=2))
        >>> len(folds)
        2
    """

    coords = _extract_coordinates(coordinates, coordinate_cols)
    group_values = _as_object_vector(groups, "groups")
    if group_values.shape[0] != coords.shape[0]:
        raise ValueError("groups must be one-dimensional and match coordinates rows")
    radius = _validate_projected_buffer(buffer_radius, coordinate_units, allow_degree_buffer)
    group_splits = list(grouped_blocked_cv(group_values, n_splits=n_splits))
    if random_state is not None:
        rng = np.random.default_rng(int(random_state))
        rng.shuffle(group_splits)

    for train_idx, test_idx in group_splits:
        yield _apply_spatial_buffer(coords, train_idx, test_idx, radius, str(backend))


def environmental_blocked_cv(
    environmental_features: object,
    *,
    n_splits: int = 5,
    feature_cols: list[object] | tuple[object, ...] | None = None,
    random_state: int | None = None,
    use_sklearn: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield folds by clustering environmental covariates.

    The helper uses sklearn KMeans when ``use_sklearn=True``. This keeps the
    dependency optional and fails with a clear install hint when sklearn is not
    available.

    Example:
        >>> features = [[0.0], [1.0], [10.0], [11.0]]
        >>> folds = list(environmental_blocked_cv(features, n_splits=2, use_sklearn=False))
        >>> len(folds)
        2
    """

    features = _extract_feature_matrix(environmental_features, feature_cols)
    _validate_n_splits(n_splits, features.shape[0])
    if use_sklearn:
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "environmental_blocked_cv requires scikit-learn; install cartoboost with "
                "scikit-learn or pass use_sklearn=False"
            ) from exc
        labels = KMeans(
            n_clusters=int(n_splits),
            n_init=10,
            random_state=random_state,
        ).fit_predict(features)
    else:
        order_axis = int(np.argmax(np.ptp(features, axis=0)))
        labels = np.empty(features.shape[0], dtype=int)
        order = np.argsort(features[:, order_axis], kind="mergesort")
        for fold_id, fold_indices in enumerate(np.array_split(order, n_splits)):
            labels[fold_indices] = fold_id
    yield from grouped_blocked_cv(labels, n_splits=n_splits)


def temporal_blocked_cv(
    times: object,
    *,
    n_splits: int = 5,
    gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices from contiguous time-ordered folds.

    ``gap`` removes that many neighboring sorted rows on each side of the test
    block from the corresponding training split.

    Example:
        >>> folds = list(temporal_blocked_cv([1, 2, 3, 4], n_splits=2, gap=0))
        >>> folds[0][1].tolist()
        [0, 1]
    """

    time_values = np.asarray(times)
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional")
    _validate_n_splits(n_splits, time_values.shape[0])
    if gap < 0:
        raise ValueError("gap must be non-negative")

    order = np.argsort(time_values, kind="mergesort")
    positions = np.arange(order.size)
    for test_positions in np.array_split(positions, n_splits):
        test_idx = order[test_positions]
        blocked = np.zeros(order.size, dtype=bool)
        start = max(0, int(test_positions[0]) - gap)
        stop = min(order.size, int(test_positions[-1]) + gap + 1)
        blocked[start:stop] = True
        train_idx = order[~blocked]
        yield np.sort(train_idx), np.sort(test_idx)


def out_of_time_split(
    times: object,
    *,
    validation_size: int | None = None,
    validation_fraction: float | None = 0.2,
    cutoff: object | None = None,
    gap: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/validation indices for a future holdout window.

    By default, the latest 20% of rows by ``times`` are used for validation.
    Pass ``validation_size`` for an exact tail count, ``validation_fraction``
    for a tail fraction, or ``cutoff`` to validate on rows strictly after a
    time boundary. ``gap`` removes that many sorted rows immediately before the
    validation window from training.

    Example:
        >>> train, valid = out_of_time_split([1, 2, 3, 4], validation_size=1)
        >>> train.tolist(), valid.tolist()
        ([0, 1, 2], [3])
    """

    time_values = np.asarray(times)
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if time_values.shape[0] < 2:
        raise ValueError("times must contain at least two samples")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if validation_size is not None and validation_fraction is not None:
        raise ValueError("pass only one of validation_size or validation_fraction")
    if cutoff is not None and (validation_size is not None or validation_fraction is not None):
        raise ValueError("cutoff cannot be combined with validation_size or validation_fraction")

    order = np.argsort(time_values, kind="mergesort")
    if cutoff is not None:
        cutoff_value = _coerce_cutoff(cutoff, time_values)
        validation_mask = time_values > cutoff_value
        validation_positions = np.nonzero(validation_mask[order])[0]
        if validation_positions.size == 0:
            raise ValueError("cutoff leaves no validation samples")
        validation_start = int(validation_positions[0])
        validation_idx = order[validation_positions]
    else:
        holdout_count = _resolve_validation_count(
            time_values.shape[0],
            validation_size,
            validation_fraction,
        )
        validation_start = time_values.shape[0] - holdout_count
        validation_idx = order[validation_start:]

    train_stop = max(0, validation_start - gap)
    train_idx = order[:train_stop]
    if train_idx.size == 0:
        raise ValueError("out-of-time split leaves no training samples")
    return np.sort(train_idx), np.sort(validation_idx)


def grouped_blocked_cv(
    groups: object,
    *,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices with whole groups held out together.

    Example:
        >>> folds = list(grouped_blocked_cv(["a", "a", "b", "b"], n_splits=2))
        >>> len(folds)
        2
    """

    group_values = _as_object_vector(groups, "groups")
    _validate_n_splits(n_splits, group_values.shape[0])

    inverse, counts = _group_inverse_and_counts(group_values)
    if counts.size < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique groups")

    fold_group_indices = _balanced_group_folds(counts, n_splits)
    all_indices = np.arange(group_values.shape[0])
    for fold_groups in fold_group_indices:
        test_mask = np.isin(inverse, fold_groups)
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
        yield train_idx, test_idx


def _as_object_vector(values: object, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        items = [
            values[index].item() if hasattr(values[index], "item") else values[index]
            for index in range(values.shape[0])
        ]
    else:
        items = list(values)  # type: ignore[arg-type]
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _group_inverse_and_counts(group_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    group_to_index: dict[object, int] = {}
    inverse = np.empty(group_values.shape[0], dtype=int)
    counts: list[int] = []
    for row, value in enumerate(group_values.tolist()):
        try:
            group_index = group_to_index.get(value)
        except TypeError as exc:
            raise ValueError("groups must contain hashable values") from exc
        if group_index is None:
            group_index = len(counts)
            group_to_index[value] = group_index
            counts.append(0)
        inverse[row] = group_index
        counts[group_index] += 1
    return inverse, np.asarray(counts, dtype=int)


def _validate_n_splits(n_splits: int, n_samples: int) -> None:
    if not isinstance(n_splits, int):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed the number of samples")


def _extract_coordinates(
    values: object,
    coordinate_cols: tuple[object, object] | list[object] | None,
) -> np.ndarray:
    if coordinate_cols is None:
        try:
            coords = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "coordinates must be numeric array-like, or pass coordinate_cols for "
                "mapping/dataframe inputs"
            ) from exc
    else:
        if len(coordinate_cols) != 2:
            raise ValueError("coordinate_cols must contain exactly two column names")
        columns = [_column_values(values, column) for column in coordinate_cols]
        coords = np.asarray(columns, dtype=float).T
        coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] == 0:
        raise ValueError("coordinates must have shape (n_samples, n_dimensions)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must contain only finite values")
    return coords


def _extract_feature_matrix(
    values: object,
    feature_cols: list[object] | tuple[object, ...] | None,
) -> np.ndarray:
    if feature_cols is None:
        try:
            features = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "environmental_features must be numeric array-like, or pass feature_cols for "
                "mapping/dataframe inputs"
            ) from exc
    else:
        if len(feature_cols) == 0:
            raise ValueError("feature_cols must contain at least one column")
        columns = [_column_values(values, column) for column in feature_cols]
        features = np.asarray(columns, dtype=float).T
        features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("environmental_features must have shape (n_samples, n_features)")
    if not np.all(np.isfinite(features)):
        raise ValueError("environmental_features must contain only finite values")
    return features


def _column_values(values: object, column: object) -> object:
    if isinstance(values, dict):
        if column not in values:
            raise ValueError(f"column {column!r} is not present")
        return values[column]
    columns = getattr(values, "columns", None)
    if columns is not None:
        if column not in columns:
            raise ValueError(f"column {column!r} is not present")
        return values[column]
    array = np.asarray(values)
    if not isinstance(column, int):
        raise ValueError("column selectors must be integer indices for array-like inputs")
    if array.ndim != 2 or column < 0 or column >= array.shape[1]:
        raise ValueError(f"column index {column!r} is out of range")
    return array[:, column]


def _validate_projected_buffer(
    buffer_radius: float,
    coordinate_units: str,
    allow_degree_buffer: bool,
) -> float:
    radius = float(buffer_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("buffer_radius must be a finite non-negative value")
    units = str(coordinate_units).lower()
    if units not in {"projected", "meters", "feet", "degrees"}:
        raise ValueError("coordinate_units must be 'projected', 'meters', 'feet', or 'degrees'")
    if radius > 0.0 and units == "degrees" and not allow_degree_buffer:
        raise ValueError(
            "buffer_radius with latitude/longitude degrees is ambiguous; project coordinates "
            "to a linear CRS or pass allow_degree_buffer=True"
        )
    return radius


def _apply_spatial_buffer(
    coords: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    radius: float,
    backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if np.intersect1d(train_idx, test_idx, assume_unique=False).size:
        raise ValueError("spatial CV fold has overlapping train and test indices")
    if radius == 0.0:
        return train_idx, test_idx
    all_indices = np.arange(coords.shape[0])
    distances = _min_distances_to_test(coords, test_idx, backend)
    safe_train = all_indices[distances > radius]
    safe_train = np.intersect1d(safe_train, train_idx, assume_unique=True)
    if safe_train.size == 0:
        raise ValueError("buffer_radius removes all training samples for at least one fold")
    return safe_train, test_idx


def _min_distances_to_test(
    coords: np.ndarray,
    test_idx: np.ndarray,
    backend: str = "cpu",
) -> np.ndarray:
    test_coords = coords[np.asarray(test_idx, dtype=int)]
    distance_work = int(coords.shape[0] * test_coords.shape[0] * coords.shape[1])
    dispatch = workload_decision(backend, "pairwise_distance", distance_work, 16_384)
    if dispatch["executed"] != "cpu":
        squared = pairwise_squared_distances(
            coords,
            test_coords,
            backend=str(dispatch["executed"]),
        ).astype(float)
        return np.sqrt(np.maximum(np.min(squared, axis=1), 0.0))
    deltas = coords[:, None, :] - test_coords[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return np.min(distances, axis=1)


def _resolve_validation_count(
    n_samples: int,
    validation_size: int | None,
    validation_fraction: float | None,
) -> int:
    if validation_size is not None:
        if not isinstance(validation_size, int):
            raise ValueError("validation_size must be an integer")
        if validation_size < 1:
            raise ValueError("validation_size must be at least 1")
        if validation_size >= n_samples:
            raise ValueError("validation_size must be smaller than the number of samples")
        return validation_size

    if validation_fraction is None:
        raise ValueError("validation_fraction must be provided when validation_size is omitted")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    holdout_count = int(np.ceil(n_samples * validation_fraction))
    return min(max(holdout_count, 1), n_samples - 1)


def _coerce_cutoff(cutoff: object, time_values: np.ndarray) -> object:
    if np.issubdtype(time_values.dtype, np.datetime64):
        return np.asarray(cutoff, dtype=time_values.dtype)
    return cutoff


def _resolve_grid_shape(
    n_splits: int,
    grid_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if grid_shape is not None:
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain two integers")
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_shape values must be positive")
        if rows * cols < n_splits:
            raise ValueError("grid_shape must provide at least n_splits blocks")
        return rows, cols

    cols = int(np.ceil(np.sqrt(n_splits)))
    rows = int(np.ceil(n_splits / cols))
    return rows, cols


def _coordinate_bins(values: np.ndarray, bin_count: int) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return np.zeros(values.shape[0], dtype=int)
    scaled = (values - minimum) / (maximum - minimum)
    return np.minimum((scaled * bin_count).astype(int), bin_count - 1)


def _balanced_group_folds(counts: np.ndarray, n_splits: int) -> list[np.ndarray]:
    fold_loads = np.zeros(n_splits, dtype=int)
    fold_groups: list[list[int]] = [[] for _ in range(n_splits)]
    for group_index in np.argsort(-counts, kind="mergesort"):
        fold_index = int(np.argmin(fold_loads))
        fold_groups[fold_index].append(int(group_index))
        fold_loads[fold_index] += int(counts[group_index])
    return [np.asarray(groups, dtype=int) for groups in fold_groups]


def _ordered_blocked_cv(
    values: np.ndarray,
    n_splits: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(values, kind="mergesort")
    sorted_folds = np.array_split(order, n_splits)
    for test_idx in sorted_folds:
        yield _complement_indices(values.shape[0], test_idx), np.sort(test_idx)


def _complement_indices(n_samples: int, test_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    mask[test_idx] = False
    return np.nonzero(mask)[0]
