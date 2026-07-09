#!/usr/bin/env python3
"""Matched Prophet-surface benchmark for CartoBoost and upstream Prophet.

Run this script in the CartoBoost environment with ``--engine cartoboost`` and
in an environment containing ``prophet==1.2.2`` with ``--engine prophet``.
Both engines receive the same generated ``ds``/``y`` fixture and 30-step
horizon. CartoBoost additionally accepts the fixture as Polars directly;
upstream Prophet requires a pandas conversion.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["cartoboost", "prophet"], required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input", choices=["pandas", "polars"], default="polars")
    args = parser.parse_args()
    if args.rows <= args.horizon:
        raise ValueError("--rows must be greater than --horizon")
    frame = _fixture(args.rows, args.seed)
    input_frame: object = frame
    if args.input == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("--input polars requires polars") from exc
        input_frame = pl.from_pandas(frame)
    if args.engine == "cartoboost":
        from cartoboost import Prophet
    else:
        from prophet import Prophet
    fit_start = perf_counter()
    if args.engine == "cartoboost":
        model = Prophet(
            n_changepoints=25,
            yearly_seasonality=False,
            weekly_seasonality=3,
            daily_seasonality=False,
            uncertainty_samples=0,
        ).fit(input_frame)
    else:
        model = Prophet(
            n_changepoints=25,
            yearly_seasonality=False,
            weekly_seasonality=3,
            daily_seasonality=False,
            uncertainty_samples=0,
        ).fit(frame)
    fit_seconds = perf_counter() - fit_start
    future_start = perf_counter()
    future = model.make_future_dataframe(args.horizon, include_history=False)
    forecast = model.predict(future)
    predict_seconds = perf_counter() - future_start
    print(
        json.dumps(
            {
                "engine": args.engine,
                "rows": args.rows,
                "horizon": args.horizon,
                "input": args.input,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": fit_seconds + predict_seconds,
                "output_rows": len(forecast),
                "output_columns": list(forecast.columns),
            },
            sort_keys=True,
        )
    )
    return 0


def _fixture(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    return pd.DataFrame(
        {
            "ds": pd.date_range("1900-01-01", periods=rows, freq="D"),
            "y": 80.0
            + 0.015 * t
            + 12.0 * np.sin(2.0 * np.pi * t / 7.0)
            + 4.0 * np.cos(2.0 * np.pi * t / 30.0)
            + rng.normal(0.0, 1.5, rows),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
