"""Stable Python interface for CartoBoost.

CartoBoost 0.3 intentionally has a small source API.  The root package is
reserved for the three Rust-backed estimators, their shared configuration,
and the package version.  Validation, schema, and forecasting live in their
named submodules; unproven model families are available only through the lazy
``cartoboost.preview`` namespace.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any

__version__ = "0.3.5"

__all__ = [
    "CartoBoostRegressor",
    "CartoBoostClassifier",
    "CartoBoostRanker",
    "BoosterConfig",
    "__version__",
]

_STABLE_SYMBOLS = {
    "CartoBoostRegressor": ("regressor", "CartoBoostRegressor"),
    "CartoBoostClassifier": ("classifier", "CartoBoostClassifier"),
    "CartoBoostRanker": ("ranker", "CartoBoostRanker"),
    "BoosterConfig": ("config", "BoosterConfig"),
}


def __getattr__(name: str) -> _Any:
    """Load the preview namespace only when a caller explicitly requests it."""

    if name == "preview":
        module = _import_module(".preview", __name__)
        globals()[name] = module
        return module
    target = _STABLE_SYMBOLS.get(name)
    if target is not None:
        module_name, attribute = target
        module = _import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"preview"})
