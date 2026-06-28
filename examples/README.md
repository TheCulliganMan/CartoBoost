# CartoBoost Examples

This directory holds lightweight examples for fitting and evaluating CartoBoost
regression models. The examples use synthetic data so the temporal-spatial
patterns are easy to inspect.

Taxi graph examples:

- `forecasting/naive_seasonal_visualization.py` compares native naive and
  seasonal naive forecasters on hourly taxi-zone pickup demand and writes a
  comparison plot.
- `forecasting/theta_optimized_visualization.py` compares native theta and
  optimized theta forecasters with an explicit theta/alpha grid.
- `forecasting/ets_component_visualization.py` fits native additive ETS and
  plots fitted values, residual-ready forecasts, level/trend paths, and
  seasonal components.
- `forecasting/arima_example_visualization.py` builds a deterministic taxi
  pickup/dropoff lane panel, fits fixed ARIMA and AutoARIMA, prints held-out
  metrics, and can write a forecast plot under `target/`.
- `forecasting/cartoboost_lag_visualization.py` fits a recursive CartoBoost lag
  forecaster with lag, rolling, calendar, and trend features for pickup demand.
- `forecasting/weighted_ensemble_visualization.py` combines seasonal naive,
  theta, and Kalman components in a native weighted ensemble.
- `forecasting/kriging_example_visualization.py` builds deterministic pickup-zone
  coordinates, fits a kriging variogram, writes interpolation/variogram/LOO
  diagnostic plots, and records the selected config.
- `forecasting/kalman_diagnostics_visualization.py` writes Kalman filtered and
  smoothed state diagnostics, forecast intervals, and innovation charts.
- `03_taxi_od_graph_regression.py` builds a pickup-dropoff zone graph for trip
  duration regression. It uses synthetic taxi-shaped rows by default and can
  read a local TLC-style CSV or Parquet file with `--input`.
- `04_taxi_pickup_zone_graph.py` builds a pickup-zone adjacency graph for demand
  modeling under a spatial holdout.
- `05_neural_embedding_regression.py` fits dense and neural embedding
  regressors on repeated IDs, then demonstrates a cold-ID guard that falls back
  to dense CartoBoost when embeddings are unsupported.
- `06_geo_causal_lift.py` builds a taxi pickup-zone intervention panel, runs
  synthetic DID, GeoLift-style design, placebo checks, and spillover warnings.
- `model_lab_geo_causal_request.json` is a browser/WASM request payload for
  `runGeoCausalExperiment` in the modeling lab.
- `model_lab_graph_forecast_request.json` is a browser/WASM request payload for
  `runGraphForecast`, using directed pickup-zone flows and horizon-wise metrics.

Suggested workflow:

1. Generate regression-shaped CSV files with
   `scripts/generate_synthetic_data.py`.
2. Train CartoBoost models with the CLI or Python estimator.
3. Write predictions to CSV with `target` and `prediction` columns.
4. Summarize metrics with `scripts/validation_report.py`.
5. Plot benchmark summaries with `scripts/plot_benchmarks.py`.
