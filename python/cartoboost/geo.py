from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import _native
from ._native import (
    CoordinateMatrix,
    GeoSpatialWeights,
    PanelIndex,
    SplitManifest,
    TimeIndex,
    geo_buffered_spatial_cv,
    geo_clockwise_bearing_unit_vector_value,
    geo_group_spatial_cv,
    geo_initial_bearing_unit_vector_latlng_value,
    geo_rolling_origin_panel_split,
    geo_spatial_block_cv,
    geo_spatial_temporal_blocked_split,
)

__all__ = [
    "CoordinateMatrix",
    "GeoSpatialWeights",
    "PanelIndex",
    "SplitManifest",
    "TimeIndex",
    "build_geo_sparse_sets",
    "build_zip_sparse_sets",
    "clockwise_bearing_unit_vector",
    "clockwise_bearing_unit_vectors",
    "coerce_geo_to_feature_id",
    "coerce_zip_to_feature_id",
    "initial_bearing_unit_vector_latlng",
    "initial_bearing_unit_vectors_latlng",
    "local_frame_features",
    "local_frame_feature_rows",
    "route_latlng_points",
    "radial_anchor_distances",
    "radial_anchor_distance_rows",
    "rbf_anchor_features",
    "rbf_anchor_feature_rows",
    "route_feature_vector",
    "route_feature_rows",
    "buffered_spatial_cv_manifest",
    "group_spatial_cv_manifest",
    "rolling_origin_panel_split_manifest",
    "spatial_block_cv_manifest",
    "spatial_temporal_blocked_split_manifest",
]


_NON_DIGITS = re.compile(r"\D")


def spatial_block_cv_manifest(
    coords: CoordinateMatrix,
    *,
    n_folds: int,
    dataset_fingerprint: str,
    coordinate_crs_note: str,
    model_version: str,
    dependency_versions: Mapping[str, str],
    random_seed: int | None = None,
    split_id: str = "spatial_block_cv",
) -> SplitManifest:
    """Create a deterministic spatial block split manifest."""
    return geo_spatial_block_cv(
        coords,
        n_folds,
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dict(dependency_versions),
        random_seed,
        split_id,
    )


def buffered_spatial_cv_manifest(
    coords: CoordinateMatrix,
    *,
    n_folds: int,
    buffer_distance: float,
    dataset_fingerprint: str,
    coordinate_crs_note: str,
    model_version: str,
    dependency_versions: Mapping[str, str],
    random_seed: int | None = None,
    split_id: str = "buffered_spatial_cv",
) -> SplitManifest:
    """Create a spatial block manifest with training rows buffered away from each test block."""
    return geo_buffered_spatial_cv(
        coords,
        n_folds,
        buffer_distance,
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dict(dependency_versions),
        random_seed,
        split_id,
    )


def group_spatial_cv_manifest(
    groups: Sequence[str],
    *,
    n_folds: int,
    dataset_fingerprint: str,
    coordinate_crs_note: str,
    model_version: str,
    dependency_versions: Mapping[str, str],
    random_seed: int | None = None,
    split_id: str = "group_spatial_cv",
) -> SplitManifest:
    """Create a manifest that holds out complete spatial groups."""
    return geo_group_spatial_cv(
        list(groups),
        n_folds,
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dict(dependency_versions),
        random_seed,
        split_id,
    )


def rolling_origin_panel_split_manifest(
    panel: PanelIndex,
    *,
    min_train_size: int,
    horizon: int,
    step: int,
    dataset_fingerprint: str,
    coordinate_crs_note: str,
    model_version: str,
    dependency_versions: Mapping[str, str],
    random_seed: int | None = None,
    split_id: str = "rolling_origin_panel_split",
) -> SplitManifest:
    """Create a leakage-safe rolling-origin split manifest for panel rows."""
    return geo_rolling_origin_panel_split(
        panel,
        min_train_size,
        horizon,
        step,
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dict(dependency_versions),
        random_seed,
        split_id,
    )


def spatial_temporal_blocked_split_manifest(
    coords: CoordinateMatrix,
    time: TimeIndex,
    *,
    n_spatial_folds: int,
    min_train_time: int,
    horizon: int,
    dataset_fingerprint: str,
    coordinate_crs_note: str,
    model_version: str,
    dependency_versions: Mapping[str, str],
    random_seed: int | None = None,
    split_id: str = "spatial_temporal_blocked_split",
) -> SplitManifest:
    """Create a manifest that blocks both held-out places and held-out time."""
    return geo_spatial_temporal_blocked_split(
        coords,
        time,
        n_spatial_folds,
        min_train_time,
        horizon,
        dataset_fingerprint,
        coordinate_crs_note,
        model_version,
        dict(dependency_versions),
        random_seed,
        split_id,
    )


def coerce_zip_to_feature_id(value: Any, *, strict: bool = False) -> int | None:
    """Convert a ZIP-like value to a deterministic non-negative integer ID."""
    text = _coerce_zip_string(value, strict=strict)
    if text is None:
        return None
    return int(text)


def coerce_geo_to_feature_id(
    value: Any,
    *,
    namespace: str = "geo",
    strict: bool = False,
) -> int | None:
    """Convert an abstract geographic identifier to a deterministic non-negative ID."""
    return _coerce_geo_string(value, namespace=namespace, strict=strict)


def clockwise_bearing_unit_vector(
    origin: tuple[Any, Any],
    destination: tuple[Any, Any],
) -> tuple[float, float] | None:
    """Return the planar clockwise bearing as ``(east, north)`` unit components.

    Angles are measured clockwise from north, but the returned feature is a unit
    vector rather than degrees. North is ``(0, 1)``, east is ``(1, 0)``, and
    northwest is near ``(-0.707, 0.707)``. Identical points return ``None`` so
    callers must handle zero-distance routes explicitly.
    """

    ox, oy = _coordinate_pair(origin, "origin")
    dx, dy = _coordinate_pair(destination, "destination")
    result = geo_clockwise_bearing_unit_vector_value(ox, oy, dx, dy)
    return None if result is None else (float(result[0]), float(result[1]))


def clockwise_bearing_unit_vectors(
    origins: Sequence[tuple[Any, Any]],
    destinations: Sequence[tuple[Any, Any]],
) -> list[tuple[float, float] | None]:
    """Vectorized planar clockwise bearing unit features for paired coordinates."""

    origin_values = list(origins)
    destination_values = list(destinations)
    if len(origin_values) != len(destination_values):
        raise ValueError("origins and destinations must have the same number of rows")
    return [
        clockwise_bearing_unit_vector(origin, destination)
        for origin, destination in zip(origin_values, destination_values, strict=True)
    ]


def initial_bearing_unit_vector_latlng(
    origin: tuple[Any, Any],
    destination: tuple[Any, Any],
) -> tuple[float, float] | None:
    """Return the great-circle initial bearing as ``(east, north)`` components.

    Coordinates are ``(latitude, longitude)`` pairs in degrees. The result uses
    the same continuous bearing encoding as ``clockwise_bearing_unit_vector``.
    """

    origin_lat, origin_lng = _coordinate_pair(origin, "origin")
    dest_lat, dest_lng = _coordinate_pair(destination, "destination")
    result = geo_initial_bearing_unit_vector_latlng_value(
        origin_lat,
        origin_lng,
        dest_lat,
        dest_lng,
    )
    return None if result is None else (float(result[0]), float(result[1]))


def initial_bearing_unit_vectors_latlng(
    origins: Sequence[tuple[Any, Any]],
    destinations: Sequence[tuple[Any, Any]],
) -> list[tuple[float, float] | None]:
    """Vectorized great-circle initial bearing unit features for lat/lng pairs."""

    origin_values = list(origins)
    destination_values = list(destinations)
    if len(origin_values) != len(destination_values):
        raise ValueError("origins and destinations must have the same number of rows")
    return [
        initial_bearing_unit_vector_latlng(origin, destination)
        for origin, destination in zip(origin_values, destination_values, strict=True)
    ]


def route_feature_vector(
    origin: tuple[Any, Any],
    destination: tuple[Any, Any],
) -> tuple[float, float, float, float, float] | None:
    """Return ``(mid_x, mid_y, distance, bearing_east, bearing_north)``.

    The helper is for planar or projected coordinates. Identical points return
    ``None`` so zero-length routes stay explicit.
    """

    ox, oy = _coordinate_pair(origin, "origin")
    dx, dy = _coordinate_pair(destination, "destination")
    result = _native_geo_feature("geo_route_feature_vector_value")(ox, oy, dx, dy)
    return None if result is None else tuple(float(value) for value in result)  # type: ignore[return-value]


def route_feature_rows(
    origins: Sequence[tuple[Any, Any]],
    destinations: Sequence[tuple[Any, Any]],
) -> list[tuple[float, float, float, float, float] | None]:
    """Vectorized route midpoint, distance, and bearing features."""

    origin_values = list(origins)
    destination_values = list(destinations)
    if len(origin_values) != len(destination_values):
        raise ValueError("origins and destinations must have the same number of rows")
    return [
        route_feature_vector(origin, destination)
        for origin, destination in zip(origin_values, destination_values, strict=True)
    ]


def route_latlng_points(route: Any) -> list[tuple[float, float]]:
    """Extract decoded ``(latitude, longitude)`` route points.

    Accepts direct coordinate sequences plus OSRM/Valhalla-style mappings when
    geometry is already decoded into coordinate arrays. Encoded polyline strings
    raise a clear error; callers should request GeoJSON/decoded shapes upstream.
    """

    candidate = route
    coordinate_order = "latlng"
    if isinstance(route, Mapping):
        candidate, coordinate_order = _route_geometry_candidate(route)
    points = [
        _latlng_route_point(point, idx, coordinate_order=coordinate_order)
        for idx, point in enumerate(candidate)
    ]
    if not points:
        raise ValueError("route must contain at least one coordinate point")
    return points


def radial_anchor_distances(
    point: tuple[Any, Any],
    anchors: Sequence[tuple[Any, Any]],
) -> list[float]:
    """Return distances from one point to each anchor."""

    px, py = _coordinate_pair(point, "point")
    anchor_values = [
        _coordinate_pair(anchor, f"anchors[{idx}]") for idx, anchor in enumerate(anchors)
    ]
    return [
        float(value)
        for value in _native_geo_feature("geo_radial_anchor_distances_value")(px, py, anchor_values)
    ]


def radial_anchor_distance_rows(
    points: Sequence[tuple[Any, Any]],
    anchors: Sequence[tuple[Any, Any]],
) -> list[list[float]]:
    """Return one radial-distance feature row per point."""

    point_values = list(points)
    anchor_values = list(anchors)
    if not anchor_values:
        raise ValueError("anchors must contain at least one coordinate pair")
    return [radial_anchor_distances(point, anchor_values) for point in point_values]


def rbf_anchor_features(
    point: tuple[Any, Any],
    anchors: Sequence[tuple[Any, Any]],
    *,
    length_scale: float,
) -> list[float]:
    """Return Gaussian radial-basis features from one point to each anchor."""

    px, py = _coordinate_pair(point, "point")
    anchor_values = [
        _coordinate_pair(anchor, f"anchors[{idx}]") for idx, anchor in enumerate(anchors)
    ]
    return [
        float(value)
        for value in _native_geo_feature("geo_rbf_anchor_features_value")(
            px,
            py,
            anchor_values,
            _finite_float(length_scale, "length_scale"),
        )
    ]


def rbf_anchor_feature_rows(
    points: Sequence[tuple[Any, Any]],
    anchors: Sequence[tuple[Any, Any]],
    *,
    length_scale: float,
) -> list[list[float]]:
    """Return one Gaussian radial-basis feature row per point."""

    point_values = list(points)
    anchor_values = list(anchors)
    if not anchor_values:
        raise ValueError("anchors must contain at least one coordinate pair")
    return [
        rbf_anchor_features(point, anchor_values, length_scale=length_scale)
        for point in point_values
    ]


def local_frame_features(
    point: tuple[Any, Any],
    origin: tuple[Any, Any],
    axis: tuple[Any, Any],
) -> tuple[float, float] | None:
    """Project a point into a local frame as ``(along_axis, cross_axis)``."""

    px, py = _coordinate_pair(point, "point")
    ox, oy = _coordinate_pair(origin, "origin")
    ax, ay = _coordinate_pair(axis, "axis")
    result = _native_geo_feature("geo_local_frame_features_value")(px, py, ox, oy, ax, ay)
    return None if result is None else (float(result[0]), float(result[1]))


def local_frame_feature_rows(
    points: Sequence[tuple[Any, Any]],
    origin: tuple[Any, Any],
    axis: tuple[Any, Any],
) -> list[tuple[float, float] | None]:
    """Vectorized local-frame projection features."""

    return [local_frame_features(point, origin, axis) for point in points]


def build_zip_sparse_sets(
    origin_zip: Sequence[Any] | None = None,
    destination_zip: Sequence[Any] | None = None,
    *,
    include_raw: bool = True,
    zip3_only: bool = False,
    parent_prefixes: Sequence[int] | None = None,
    include_match_indicator: bool = True,
    strict: bool = False,
) -> dict[str, list[list[int]]]:
    """Build sparse-set columns for ZIP features.

    Returns sparse_set columns keyed by role/prefix so ZIP origin/destination columns
    are explicitly surfaced as geographic context.
    """
    has_origin_zip = origin_zip is not None
    has_destination_zip = destination_zip is not None
    if not has_origin_zip and not has_destination_zip:
        raise ValueError("origin_zip and destination_zip cannot both be None")

    if zip3_only:
        if include_raw:
            raise ValueError("zip3_only cannot be used with include_raw=True")
        parent_prefixes = (3,)
        include_raw = False
    parent_prefixes = _normalize_prefixes(parent_prefixes or (3, 2))
    ozip_codes: list[str | None] = (
        _coerce_zip_sequence(origin_zip, strict=strict, name="origin_zip") if has_origin_zip else []
    )
    dzip_codes: list[str | None] = (
        _coerce_zip_sequence(destination_zip, strict=strict, name="destination_zip")
        if has_destination_zip
        else []
    )
    if has_origin_zip and has_destination_zip and len(ozip_codes) != len(dzip_codes):
        raise ValueError("origin_zip and destination_zip must have the same number of rows")
    row_count = len(ozip_codes) if has_origin_zip else len(dzip_codes)
    sparse_sets: dict[str, list[list[int]]] = {}

    if has_origin_zip:
        sparse_sets.update(
            _zip_sparse_columns(
                ozip_codes,
                prefix="ozip",
                include_raw=include_raw,
                parent_prefixes=parent_prefixes,
                row_count=row_count,
            )
        )
    if has_destination_zip:
        sparse_sets.update(
            _zip_sparse_columns(
                dzip_codes,
                prefix="dzip",
                include_raw=include_raw,
                parent_prefixes=parent_prefixes,
                row_count=row_count,
            )
        )

    if include_match_indicator and has_origin_zip and has_destination_zip:
        sparse_sets["zip_match"] = [
            [1] if ozip == dzip and ozip is not None else []
            for ozip, dzip in zip(ozip_codes, dzip_codes, strict=False)
        ]

    return sparse_sets


def build_geo_sparse_sets(
    geo_features: Mapping[str, Sequence[Any]] | None = None,
    *,
    namespace: str = "geo",
    strict: bool = False,
) -> dict[str, list[list[int]]]:
    """Build sparse-set columns for abstract geographic IDs.

    The input keys become sparse column names.
    """
    if not geo_features:
        raise ValueError("geo_features cannot be empty")

    feature_items = list(geo_features.items())
    row_count = len(feature_items[0][1])
    if row_count == 0:
        raise ValueError("geo_features values must contain at least one row")

    for _, (name, values) in enumerate(feature_items[1:], start=1):
        if len(values) != row_count:
            raise ValueError(f"geo feature '{name}' has {len(values)} rows, expected {row_count}")
        if name == "":
            raise ValueError("geo feature names must be non-empty")

    sparse_sets: dict[str, list[list[int]]] = {}
    for name, values in feature_items:
        if name == "":
            raise ValueError("geo feature names must be non-empty")
        column: list[list[int]] = []
        for idx, value in enumerate(values):
            value_id = _coerce_geo_string(value, namespace=f"{namespace}:{name}", strict=strict)
            if value_id is None and strict:
                raise ValueError(f"geo feature '{name}' contains invalid value at row {idx}")
            column.append([] if value_id is None else [value_id])
        sparse_sets[name] = column
    return sparse_sets


def _zip_sparse_columns(
    codes: list[str | None],
    *,
    prefix: str,
    include_raw: bool,
    parent_prefixes: tuple[int, ...],
    row_count: int,
) -> dict[str, list[list[int]]]:
    columns: dict[str, list[list[int]]] = {}
    if include_raw:
        columns[f"{prefix}_zip5"] = [[] for _ in range(row_count)]
    for level in parent_prefixes:
        if include_raw and level == 5:
            continue
        columns[f"{prefix}_zip_p{level}"] = [[] for _ in range(row_count)]

    for idx, code in enumerate(codes):
        if code is None:
            continue
        if include_raw:
            columns[f"{prefix}_zip5"][idx].append(int(code))
        for level in parent_prefixes:
            if include_raw and level == 5:
                continue
            columns[f"{prefix}_zip_p{level}"][idx].append(int(code[:level]))

    return columns


def _normalize_prefixes(prefixes: Sequence[int]) -> tuple[int, ...]:
    cleaned = []
    for value in prefixes:
        if not isinstance(value, int):
            raise ValueError("parent_prefixes must be integers")
        if value <= 0:
            raise ValueError("parent_prefixes must be positive")
        if value > 5:
            value = 5
        cleaned.append(value)
    if not cleaned:
        raise ValueError("parent_prefixes must contain at least one level")
    # remove duplicates while keeping original order
    unique: list[int] = []
    for value in cleaned:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _coerce_zip_sequence(
    values: Sequence[Any],
    *,
    strict: bool,
    name: str,
) -> list[str | None]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one row")
    coerced: list[str | None] = []
    for idx, value in enumerate(values):
        code = _coerce_zip_string(value, strict=strict)
        if code is None and strict:
            raise ValueError(f"{name} contains invalid ZIP value at row {idx}")
        coerced.append(code)
    return coerced


def _coerce_zip_string(value: Any, strict: bool) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        if strict:
            raise ValueError("boolean ZIP values are not supported")
        return None
    if isinstance(value, int):
        if value < 0:
            if strict:
                raise ValueError("ZIP values must be non-negative")
            return None
        text = str(value)
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            if strict:
                raise ValueError("ZIP values must be finite")
            return None
        if not value.is_integer():
            if strict:
                raise ValueError("ZIP values must be integer-like")
            return None
        if value < 0:
            if strict:
                raise ValueError("ZIP values must be non-negative")
            return None
        text = str(int(value))
    else:
        text = str(value).strip()
        if not text:
            if strict:
                raise ValueError("ZIP values must be non-empty")
            return None
        text = _NON_DIGITS.sub("", text)
        if not text:
            if strict:
                raise ValueError(f"ZIP value {value!r} has no digits")
            return None

    text = text.zfill(5)[:5]
    return text


def _coordinate_pair(value: tuple[Any, Any], field_name: str) -> tuple[float, float]:
    try:
        first, second = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a two-value coordinate pair") from exc
    return _finite_float(first, f"{field_name}[0]"), _finite_float(second, f"{field_name}[1]")


def _route_geometry_candidate(route: Mapping[str, Any]) -> tuple[Any, str]:
    if "geometry" in route:
        geometry = route["geometry"]
        if isinstance(geometry, str):
            raise ValueError(
                "route geometry is an encoded polyline string; request decoded GeoJSON "
                "coordinates from OSRM/Valhalla before route-cell encoding"
            )
        if isinstance(geometry, Mapping):
            geometry_type = geometry.get("type")
            if geometry_type not in (None, "LineString"):
                raise ValueError("route geometry mapping must be a LineString")
            if "coordinates" not in geometry:
                raise ValueError("route geometry mapping must contain coordinates")
            return geometry["coordinates"], "lonlat"
        return geometry, "latlng"
    if "shape" in route:
        shape = route["shape"]
        if isinstance(shape, str):
            raise ValueError(
                "route shape is an encoded polyline string; request decoded shape points "
                "from Valhalla before route-cell encoding"
            )
        return shape, "latlng"
    if "coordinates" in route:
        return route["coordinates"], "lonlat"
    if "points" in route:
        return route["points"], "latlng"
    raise ValueError("route mapping must contain decoded geometry, shape, coordinates, or points")


def _latlng_route_point(point: Any, idx: int, *, coordinate_order: str) -> tuple[float, float]:
    field_name = f"route[{idx}]"
    if isinstance(point, Mapping):
        if "lat" in point and "lon" in point:
            return _finite_float(point["lat"], f"{field_name}.lat"), _finite_float(
                point["lon"],
                f"{field_name}.lon",
            )
        if "lat" in point and "lng" in point:
            return _finite_float(point["lat"], f"{field_name}.lat"), _finite_float(
                point["lng"],
                f"{field_name}.lng",
            )
        if "latitude" in point and "longitude" in point:
            return _finite_float(point["latitude"], f"{field_name}.latitude"), _finite_float(
                point["longitude"],
                f"{field_name}.longitude",
            )
        raise ValueError(f"{field_name} mapping must contain lat/lon coordinates")
    first, second = _coordinate_pair(point, field_name)
    if coordinate_order == "lonlat":
        latitude, longitude = second, first
    else:
        latitude, longitude = first, second
    if abs(latitude) <= 90.0 and abs(longitude) <= 180.0:
        return latitude, longitude
    raise ValueError(f"{field_name} must be a valid latitude/longitude point")


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite coordinate")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite coordinate") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite coordinate")
    return result


def _native_geo_feature(name: str) -> Any:
    func = getattr(_native, name, None)
    if func is None:
        raise RuntimeError(
            f"cartoboost._native does not expose {name}; run `uv run --group dev maturin develop` "
            "after native geo feature changes"
        )
    return func


def _coerce_geo_string(value: Any, *, namespace: str, strict: bool) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        if strict:
            raise ValueError("geo values cannot be boolean")
        return None
    if isinstance(value, int):
        if value < 0:
            if strict:
                raise ValueError("geo values must be non-negative")
            return None
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            if strict:
                raise ValueError("geo values must be finite")
            return None
        if not value.is_integer():
            if strict:
                raise ValueError("geo values must be integer-like")
            return None
        if value < 0:
            if strict:
                raise ValueError("geo values must be non-negative")
            return None
        return int(value)

    text = str(value).strip()
    if not text:
        if strict:
            raise ValueError("geo values must be non-empty")
        return None
    token = f"{namespace}:{text}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest()
    return int(digest, 16) % (10**9)
