# NYC Taxi Benchmark Assets

This directory contains generated evidence for the NYC taxi benchmark. The
maintained narrative is
[NYC Taxi Benchmarks](../../benchmarks/nyc-taxi.md).

The current maintained run uses real 2024 NYC TLC yellow taxi data, a
100,000-row sample across all 12 months, transformed trip duration, transformed
fare amount, and transformed pickup-zone demand. The feature sets include
pickup/dropoff zone context, trip descriptors, hour/day features, and graph
features for the zone-demand task.

Generated files include single-run `results.*`, repeated-run
`repeated_results.*`, top-level summary images, and per-task plots under
`plots/`.

Forecasting benchmark evidence in this directory includes:

- `forecasting_m4_suite_sample.json`: M4 sample suite over 24 series per group.
- `forecasting_m5_full_roster_sample.json`: M5 100-series full-roster
  comparison sample.
- `forecasting_m5_full.json`: M5 30,490-series CartoBoost-only full-corpus
  check.
- `forecasting_m6_full.json`: M6 100-symbol full-roster point-forecast proxy.
- `forecasting_*_plots/`: generated RMSE, horizon, forecast-line, and
  actual-vs-predicted plots for the maintained forecasting runs.

Raw TLC Parquet files are cached under `data/nyc_taxi/` and must not be
committed.
