from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._native import weighted_overlay as _native_weighted_overlay
from .config import OverlayKernel


@dataclass(frozen=True)
class OverlayConfig:
    weights: dict[str, float]
    zone_priority_multiplier: bool = True
    kernel: OverlayKernel = OverlayKernel.NONE
    bandwidth_meters: float | None = None
    distance_alpha: float = 0.0
    precision: int = 6


def weighted_overlay(
    *,
    points: dict[str, Any],
    zones: dict[str, Any],
    weights: dict[str, float],
    origin: tuple[float, float] | None = None,
    zone_priority_multiplier: bool = True,
    kernel: OverlayKernel = OverlayKernel.NONE,
    bandwidth_meters: float | None = None,
    distance_alpha: float = 0.0,
    precision: int = 6,
    include_debug: bool = False,
) -> dict[str, Any]:
    return _native_weighted_overlay(
        points=points,
        zones=zones,
        weights=weights,
        origin=origin,
        zone_priority_multiplier=zone_priority_multiplier,
        kernel=str(kernel),
        bandwidth_meters=bandwidth_meters,
        distance_alpha=distance_alpha,
        precision=precision,
        include_debug=include_debug,
    )
