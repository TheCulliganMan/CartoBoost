"""CartoBoost's unified public Python API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.3.13"

# Every shipped family is reachable directly from ``cartoboost`` and through
# its named module. There is deliberately no secondary namespace.
_MODULES = (
    "accelerators",
    "causal",
    "deep",
    "evaluation",
    "experimental",
    "explain",
    "forecasting",
    "foundation",
    "geo",
    "geo_causal",
    "geostats",
    "graph",
    "h3",
    "io",
    "metrics",
    "models",
    "neural",
    "overlay",
    "plotting",
    "prob",
    "prophet",
    "s2",
    "schema",
    "spatial_econometrics",
    "standalone",
    "tensorboard",
    "utilities",
    "validation",
)

_STABLE_SYMBOLS = {
    "CartoBoostRegressor": ("regressor", "CartoBoostRegressor"),
    "CartoBoostClassifier": ("classifier", "CartoBoostClassifier"),
    "CartoBoostRanker": ("ranker", "CartoBoostRanker"),
    "BoosterConfig": ("config", "BoosterConfig"),
}

__all__ = [
    "CartoBoostRegressor",
    "CartoBoostClassifier",
    "CartoBoostRanker",
    "BoosterConfig",
    *_MODULES,
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Resolve model families and model symbols without a tiered API."""

    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"_native", "_forecasting_catalog"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    target = _STABLE_SYMBOLS.get(name)
    if target is not None:
        module_name, attribute = target
        value = getattr(import_module(f".{module_name}", __name__), attribute)
        globals()[name] = value
        return value
    # Named modules remain the preferred import path, while this fallback
    # keeps direct top-level imports consistent across all shipped families.
    for module_name in _MODULES:
        module = import_module(f".{module_name}", __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def mean_reciprocal_rank(
    labels: Any,
    scores: Any,
    query_ids: Any | None = None,
    *,
    groups: Any | None = None,
    k: int | None = None,
) -> float:
    """Evaluate ranking scores with either graph query IDs or tabular groups.

    The positional ``query_ids`` form retains the graph metric's treatment of
    queries without positive labels. ``groups`` and ``k`` use the tabular
    ranking metric, including its zero score for queries without positives.
    """
    if query_ids is not None:
        if groups is not None or k is not None:
            raise ValueError("query_ids cannot be combined with groups or k")
        metric = import_module(".graph.eval", __name__).mean_reciprocal_rank
        return metric(labels, scores, query_ids)
    metric = import_module(".metrics", __name__).mean_reciprocal_rank
    return metric(labels, scores, groups=groups, k=k)
