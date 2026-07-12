"""Lazy, explicitly non-stable CartoBoost surfaces.

Preview objects are intentionally absent from :mod:`cartoboost` itself.  They
may change or disappear in a minor release and carry no source-API guarantee.
Both ``cartoboost.preview.geo`` and ``from cartoboost import preview`` remain
lazy: importing this namespace does not import optional model stacks.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

_MODULES = {
    name: f"cartoboost.{name}"
    for name in (
        "causal",
        "experimental",
        "graph",
        "plotting",
        "prob",
        "deep",
        "neural",
        "forecasting",
        "geo",
        "geo_causal",
        "geostats",
        "h3",
        "s2",
        "spatial_econometrics",
        "standalone",
        "utilities",
        "foundation",
        "explain",
        "overlay",
        "prophet",
        "metrics",
        "models",
    )
}

# The stable forecasting package intentionally exposes only the v0.3 contract;
# the broad legacy/native forecasting registry is routed through this lazy
# private implementation module for ``cartoboost.preview.forecasting``.
_MODULES["forecasting"] = "cartoboost._forecasting_preview"

# AutoGeoModel and GeoModelStack are deliberately not part of the 0.3 wheel.
# Their selector behavior has not met the native/evidence admission gate, so
# no implementation is exposed through the preview namespace.
_BLOCKED_SYMBOLS = {
    "models": frozenset({"AutoGeoModel", "GeoModelStack"}),
}


def _lazy_module(name: str, target: str) -> ModuleType:
    proxy = ModuleType(f"{__name__}.{name}")
    proxy.__package__ = __name__
    proxy.__doc__ = f"Lazy preview proxy for :mod:`{target}`."

    def load_attribute(attribute: str) -> Any:
        if attribute == "__all__":
            module = import_module(target)
            blocked = _BLOCKED_SYMBOLS.get(name, ())
            return [value for value in getattr(module, "__all__", ()) if value not in blocked]
        if attribute in _BLOCKED_SYMBOLS.get(name, ()):
            raise AttributeError(f"{proxy.__name__}.{attribute} is not shipped in CartoBoost 0.3")
        module = import_module(target)
        value = getattr(module, attribute)
        setattr(proxy, attribute, value)
        return value

    def module_getattr(attribute: str) -> Any:
        return load_attribute(attribute)

    def module_dir() -> list[str]:
        blocked = _BLOCKED_SYMBOLS.get(name, ())
        return sorted((set(vars(proxy)) | set(dir(import_module(target)))) - set(blocked))

    proxy.__getattr__ = module_getattr  # type: ignore[attr-defined]
    proxy.__dir__ = module_dir  # type: ignore[attr-defined]
    sys.modules[proxy.__name__] = proxy
    return proxy


# Mark this module as a package and register only lightweight proxy modules.
# The actual CartoBoost modules are imported on first attribute access.
__path__ = []  # type: ignore[var-annotated]
for _name, _target in _MODULES.items():
    globals()[_name] = _lazy_module(_name, _target)


_SYMBOLS = {
    "FeatureKind": ("schema", "FeatureKind"),
    "FeatureSchema": ("schema", "FeatureSchema"),
    "ModelMetadata": ("models", "ModelMetadata"),
    "ModelRegistry": ("models", "ModelRegistry"),
    "ModelSpec": ("models", "ModelSpec"),
    "model_manifest": ("models", "model_manifest"),
    "CartoBoostLagForecaster": ("forecasting", "CartoBoostLagForecaster"),
    "AutoForecaster": ("forecasting", "AutoForecaster"),
    "AutoForecastConfig": ("forecasting", "AutoForecastConfig"),
    "ForecastFrame": ("forecasting", "ForecastFrame"),
    "ForecastResult": ("forecasting", "ForecastResult"),
    "ForecastMetricSet": ("forecasting", "ForecastMetricSet"),
    "NaiveForecaster": ("forecasting", "NaiveForecaster"),
    "SeasonalNaiveForecaster": ("forecasting", "SeasonalNaiveForecaster"),
    "RollingOriginSplitter": ("forecasting", "RollingOriginSplitter"),
    "RollingOriginBacktester": ("forecasting", "RollingOriginBacktester"),
    "LagConfig": ("forecasting", "LagConfig"),
    "croston_forecast": ("utilities", "croston_forecast"),
    "sba_forecast": ("utilities", "sba_forecast"),
    "tsb_forecast": ("utilities", "tsb_forecast"),
    "mean_interval_width": ("metrics", "mean_interval_width"),
    "residual_morans_i": ("metrics", "residual_morans_i"),
    "SpatialGaussianProcessRegressor": ("geostats", "SpatialGaussianProcessRegressor"),
    "NearestNeighborGPRegressor": ("geostats", "NearestNeighborGPRegressor"),
    "ResidualNNGPRegressor": ("geostats", "ResidualNNGPRegressor"),
    "Prophet": ("prophet", "Prophet"),
    "NeuralEmbeddingRegressor": ("neural", "NeuralEmbeddingRegressor"),
    "NeuralEmbeddingFeatures": ("neural", "NeuralEmbeddingFeatures"),
    "GraphSageEncoder": ("_native", "GraphSageEncoder"),
    "HeteroGraphSageEncoder": ("_native", "HeteroGraphSageEncoder"),
    "HinSageEncoder": ("_native", "HinSageEncoder"),
    "Node2VecEncoder": ("_native", "Node2VecEncoder"),
    "kalman_filter": ("utilities", "kalman_filter"),
    "fit_ordinary_kriging_variogram": ("utilities", "fit_ordinary_kriging_variogram"),
    "ordinary_kriging_predict": ("utilities", "ordinary_kriging_predict"),
    "ordinary_kriging_leave_one_out": ("utilities", "ordinary_kriging_leave_one_out"),
    "ordinary_kriging_leave_one_out_diagnostics": (
        "utilities",
        "ordinary_kriging_leave_one_out_diagnostics",
    ),
    "empirical_variogram": ("utilities", "empirical_variogram"),
}

__all__ = sorted(set(_MODULES) | set(_SYMBOLS))


def __getattr__(name: str) -> Any:
    target = _SYMBOLS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(f"cartoboost.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
