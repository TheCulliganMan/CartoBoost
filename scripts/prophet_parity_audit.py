#!/usr/bin/env python3
"""Emit a machine-readable Prophet 1.2.2 public-surface audit.

Run once in the upstream Prophet environment and once in the CartoBoost
environment, then compare the JSON outputs. The audit deliberately reports
missing methods and signature differences instead of treating a wrapper as
compatible merely because a basic fit succeeds.
"""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

PUBLIC_PROPHET_METHODS = [
    "add_country_holidays",
    "add_group_component",
    "add_regressor",
    "add_seasonality",
    "calculate_initial_params",
    "construct_holiday_dataframe",
    "fit",
    "flat_growth_init",
    "flat_trend",
    "fourier_series",
    "initialize_scales",
    "linear_growth_init",
    "make_all_seasonality_features",
    "make_future_dataframe",
    "make_holiday_features",
    "make_seasonality_features",
    "parse_seasonality_args",
    "piecewise_linear",
    "piecewise_logistic",
    "plot",
    "plot_components",
    "predict",
    "predict_seasonal_components",
    "predict_trend",
    "predict_uncertainty",
    "predictive_samples",
    "preprocess",
    "regressor_column_matrix",
    "set_auto_seasonalities",
    "set_changepoints",
    "setup_dataframe",
    "validate_column_name",
    "validate_inputs",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["cartoboost", "prophet"], required=True)
    args = parser.parse_args()
    if args.engine == "cartoboost":
        from cartoboost import Prophet
    else:
        from prophet import Prophet

    signatures: dict[str, str] = {}
    missing: list[str] = []
    for name in PUBLIC_PROPHET_METHODS:
        method = getattr(Prophet, name, None)
        if method is None:
            missing.append(name)
        else:
            try:
                signatures[name] = str(inspect.signature(method))
            except (TypeError, ValueError):
                signatures[name] = "<signature unavailable>"
    print(
        json.dumps(
            {
                "engine": args.engine,
                "prophet_version": _version(args.engine),
                "constructor": str(inspect.signature(Prophet)),
                "missing_methods": missing,
                "signatures": signatures,
                "method_count": len(PUBLIC_PROPHET_METHODS) - len(missing),
                "expected_method_count": len(PUBLIC_PROPHET_METHODS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _version(engine: str) -> Any:
    if engine == "prophet":
        import prophet

        return prophet.__version__
    return "cartoboost"


if __name__ == "__main__":
    raise SystemExit(main())
