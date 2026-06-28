from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import cache
from typing import Any

from . import _native
from .geo import route_latlng_points

__all__ = [
    "build_h3_route_sparse_sets",
    "build_h3_sparse_sets",
    "encode_h3_cells",
    "encode_h3_route_cells",
    "expand_h3_sparse_set",
    "h3_parent_id",
    "latlng_to_h3_id",
    "normalize_h3_id",
    "scaffold_h3_parent_id",
]


def normalize_h3_id(value: Any) -> int:
    """Normalize an H3-style sparse ID."""

    if isinstance(value, bool):
        raise ValueError("H3 IDs must be non-negative integers or integer strings")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("H3 IDs must be non-negative")
        return value
    if isinstance(value, str):
        return int(_native.h3_normalize_id_text(value))
    raise ValueError("H3 IDs must be non-negative integers or integer strings")


def latlng_to_h3_id(latitude: Any, longitude: Any, *, resolution: int) -> int:
    """Encode a latitude/longitude pair with the optional ``h3`` package."""

    h3 = _load_h3()
    res = _normalize_resolution(resolution, "resolution")
    lat = _normalize_coordinate(latitude, "latitude")
    lng = _normalize_coordinate(longitude, "longitude")
    if hasattr(h3, "latlng_to_cell"):
        return normalize_h3_id(h3.latlng_to_cell(lat, lng, res))
    if hasattr(h3, "geo_to_h3"):
        return normalize_h3_id(h3.geo_to_h3(lat, lng, res))
    raise RuntimeError("installed h3 package does not expose latlng_to_cell or geo_to_h3")


def h3_parent_id(cell: Any, *, parent_resolution: int) -> int:
    """Return the real H3 parent cell using the optional ``h3`` package."""

    h3 = _load_h3()
    parent = _normalize_resolution(parent_resolution, "parent_resolution")
    cell_id = normalize_h3_id(cell)
    cell_text = format(cell_id, "x")
    if hasattr(h3, "cell_to_parent"):
        return normalize_h3_id(h3.cell_to_parent(cell_text, parent))
    if hasattr(h3, "h3_to_parent"):
        return normalize_h3_id(h3.h3_to_parent(cell_text, parent))
    raise RuntimeError("installed h3 package does not expose cell_to_parent or h3_to_parent")


def encode_h3_cells(
    latitudes: Iterable[Any],
    longitudes: Iterable[Any],
    *,
    resolution: int,
) -> list[int]:
    """Encode latitude/longitude rows into H3 integer IDs."""

    lat_values = list(latitudes)
    lng_values = list(longitudes)
    if len(lat_values) != len(lng_values):
        raise ValueError("latitudes and longitudes must have the same number of rows")
    if not lat_values:
        raise ValueError("latitudes and longitudes must contain at least one row")
    return [
        latlng_to_h3_id(latitude, longitude, resolution=resolution)
        for latitude, longitude in zip(lat_values, lng_values, strict=True)
    ]


def encode_h3_route_cells(
    route: Any,
    *,
    resolution: int,
    parent_resolutions: Iterable[int] = (),
) -> list[int]:
    """Encode one decoded OSRM/Valhalla route into sorted H3 sparse IDs."""

    child_resolution = _normalize_resolution(resolution, "resolution")
    parents = [_normalize_resolution(value, "parent_resolutions") for value in parent_resolutions]
    _native.h3_validate_parent_resolutions_value(child_resolution, parents)
    cells: list[int] = []
    for latitude, longitude in route_latlng_points(route):
        child = latlng_to_h3_id(latitude, longitude, resolution=child_resolution)
        cells.append(child)
        cells.extend(h3_parent_id(child, parent_resolution=parent) for parent in parents)
    return [int(value) for value in _native.geo_assemble_route_sparse_rows_value([cells])[0]]


def build_h3_route_sparse_sets(
    routes: Iterable[Any],
    *,
    name: str = "route_h3",
    resolution: int,
    parent_resolutions: Iterable[int] = (),
) -> dict[str, list[list[int]]]:
    """Build one H3 sparse-set column from decoded route geometries."""

    if not name:
        raise ValueError("name must be non-empty")
    route_values = list(routes)
    if not route_values:
        raise ValueError("routes must contain at least one row")
    parents = tuple(parent_resolutions)
    rows = [
        encode_h3_route_cells(
            route,
            resolution=resolution,
            parent_resolutions=parents,
        )
        for route in route_values
    ]
    return {name: [list(row) for row in _native.geo_assemble_route_sparse_rows_value(rows)]}


def build_h3_sparse_sets(
    coordinates: dict[str, tuple[Iterable[Any], Iterable[Any]]],
    *,
    resolution: int,
    parent_resolutions: Iterable[int] = (),
) -> dict[str, list[list[int]]]:
    """Build sparse-set columns from named latitude/longitude coordinate pairs.

    ``coordinates`` maps output sparse-set names to ``(latitudes, longitudes)``.
    Each row contains the child H3 cell plus any requested real H3 parent cells.
    """

    if not coordinates:
        raise ValueError("coordinates cannot be empty")
    child_resolution = _normalize_resolution(resolution, "resolution")
    parents = [_normalize_resolution(value, "parent_resolutions") for value in parent_resolutions]
    _native.h3_validate_parent_resolutions_value(child_resolution, parents)

    sparse_sets: dict[str, list[list[int]]] = {}
    expected_rows: int | None = None
    for name, pair in coordinates.items():
        if not name:
            raise ValueError("coordinate names must be non-empty")
        try:
            latitudes, longitudes = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("coordinate values must be (latitudes, longitudes) pairs") from exc
        cells = encode_h3_cells(latitudes, longitudes, resolution=child_resolution)
        if expected_rows is None:
            expected_rows = len(cells)
        else:
            _native.geo_validate_equal_row_count_value(name, len(cells), expected_rows)
        parent_columns = [
            [h3_parent_id(cell, parent_resolution=parent) for cell in cells] for parent in parents
        ]
        sparse_sets[name] = [
            [int(value) for value in row]
            for row in _native.geo_assemble_sparse_column_value(cells, parent_columns)
        ]
    return sparse_sets


def expand_h3_sparse_set(
    values: Iterable[Any],
    *,
    resolution: int,
    parent_resolutions: Iterable[int] = (),
) -> list[int]:
    """Return normalized IDs plus deterministic scaffold parent IDs.

    This is a test scaffold, not real H3 parent cartometry. It creates stable
    synthetic parent IDs so sparse-set code can exercise hierarchical columns
    without depending on an H3 library.
    """

    child_resolution = _normalize_resolution(resolution, "resolution")
    parents = [_normalize_resolution(value, "parent_resolutions") for value in parent_resolutions]
    normalized = [normalize_h3_id(value) for value in values]
    return [
        int(value)
        for value in _native.h3_expand_sparse_set_value(normalized, child_resolution, parents)
    ]


def scaffold_h3_parent_id(cell: Any, resolution: int, parent_resolution: int) -> int:
    """Build a deterministic synthetic parent ID for tests.

    The output is stable for a cell/resolution pair, but it is not a real H3
    parent and must not be used for cartospatial semantics.
    """

    child = _normalize_resolution(resolution, "resolution")
    parent = _normalize_resolution(parent_resolution, "parent_resolution")
    normalized_cell = normalize_h3_id(cell)
    return int(_native.h3_scaffold_parent_id_value(normalized_cell, child, parent))


@cache
def _load_h3() -> Any:
    try:
        return importlib.import_module("h3")
    except ImportError as exc:
        raise ImportError(
            "H3 auto-encoding requires the optional 'h3' package. "
            "Install with `pip install cartoboost[h3]` or `pip install h3`."
        ) from exc


def _normalize_coordinate(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite coordinate")
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite coordinate") from exc
    return float(_native.geo_normalize_coordinate_value(coordinate, field_name))


def _normalize_resolution(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer H3 resolution")
    try:
        resolution = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer H3 resolution") from exc
    if resolution != value:
        raise ValueError(f"{field_name} must be an integer H3 resolution")
    return int(_native.h3_normalize_resolution_value(resolution, field_name))
