from __future__ import annotations

from typing import Any

from ...config import ValidationObjective
from .._native_wrappers import NativeForecastWrapper


class AutoStatsBank(NativeForecastWrapper):
    """Thin wrapper for the Rust deterministic statistical expert bank."""

    native_class_name = "AutoStatsBank"

    def __init__(
        self,
        *,
        season_length: int,
        validation_window: int | None = None,
        validation_objective: ValidationObjective = ValidationObjective.MEAN_SQUARED_ERROR,
    ) -> None:
        season_length = int(season_length)
        if season_length <= 0:
            raise ValueError("season_length must be positive")
        if validation_window is not None and int(validation_window) <= 0:
            raise ValueError("validation_window must be positive when provided")
        validation_objective = str(validation_objective)
        super().__init__(
            season_length=season_length,
            validation_window=None if validation_window is None else int(validation_window),
            validation_objective=validation_objective,
        )
        self.season_length = season_length
        self.validation_window = None if validation_window is None else int(validation_window)
        self.validation_objective = validation_objective

    def get_metadata(self) -> dict[str, Any]:
        return super().get_metadata()


__all__ = ["AutoStatsBank"]
