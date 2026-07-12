"""Native-backed validation contracts for stable CartoBoost workflows."""

from __future__ import annotations

from enum import Enum

from .config import SplitPolicy
from .geo import (
    SplitManifest,
    buffered_spatial_cv_manifest,
    group_spatial_cv_manifest,
    rolling_origin_panel_split_manifest,
    spatial_block_cv_manifest,
    spatial_temporal_blocked_split_manifest,
)


class SplitKind(str, Enum):
    """Named validation families accepted by the native split-manifest API."""

    SPATIAL = "spatial"
    GROUPED = "grouped"
    BUFFERED = "buffered"
    TEMPORAL = "temporal"
    OUT_OF_TIME = "out_of_time"


# Native-backed manifest constructors. Lower-level coordinate, panel, and time
# index types are accepted directly so validation cannot silently fall back to
# a Python implementation or alter the split protocol.
native_spatial_split = spatial_block_cv_manifest
native_buffered_spatial_split = buffered_spatial_cv_manifest
native_grouped_split = group_spatial_cv_manifest
native_temporal_split = rolling_origin_panel_split_manifest
native_spatial_temporal_split = spatial_temporal_blocked_split_manifest
native_out_of_time_split = rolling_origin_panel_split_manifest


__all__ = [
    "SplitKind",
    "SplitManifest",
    "SplitPolicy",
    "buffered_spatial_cv_manifest",
    "group_spatial_cv_manifest",
    "rolling_origin_panel_split_manifest",
    "spatial_block_cv_manifest",
    "spatial_temporal_blocked_split_manifest",
    "native_spatial_split",
    "native_buffered_spatial_split",
    "native_grouped_split",
    "native_temporal_split",
    "native_spatial_temporal_split",
    "native_out_of_time_split",
]
