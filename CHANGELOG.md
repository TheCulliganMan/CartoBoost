# Changelog

## Unreleased

## 0.3.7 — Full Accelerator Backends

- Published the stable `cuda`, `rocm`, `metal`, and `webgpu` native backend
  feature names alongside CPython 3.14 and free-threaded CPython support.

## 0.3.6 — Full Metal LSTTN Forecasting

- Added full Metal execution for LSTTN training and inference, including the
  scalar computation graph, reverse-mode gradients, AdamW parameter updates,
  and direct long-horizon forecast evaluation.
- Added configurable long-history, periodic, recent-context, and forecast
  windows for hourly freight and traffic forecasting without fixed five-minute
  assumptions.
- Added verified 207-sensor METR-LA evidence for a 168-hour horizon, with
  source checksums, exact backend scope, comparable graph-model results, and a
  committed machine-readable artifact.
- Updated the Rust/WASM graph forecasting surface and Metal parity, lifecycle,
  long-horizon stability, and browser-profile validation.

## 0.3.5 — Spatial-Temporal Taxi Graph Forecasting

- Added Rust-native directional market-structure forecasting with graph-aware
  spatial-temporal transformer profiles, Python and WebAssembly bindings, and
  artifact-backed taxi lane visualizations.
- Added large-scale H3 pickup-to-dropoff lane exploration in the Modeling Lab
  and linked the graph forecasting guides throughout the public documentation.

- Added CartoBoost Forecasting V1: Rust-native `ForecastFrame`, deterministic
  `ForecastResult` outputs, naive/seasonal naive/theta/optimized-theta/ETS/
  AutoARIMA forecasters, rolling-origin backtesting, leakage-safe lag features,
  `CartoBoostLagForecaster`, Rust-core weighted ensembles, artifact/config
  helpers, CLI script support, taxi-focused examples/docs, and deterministic
  forecasting benchmarks including explicit `functime` and `statsforecast`
  library comparisons.
## 0.3.0 — Focused Beta Reset

- Reduced the stable Python surface to the Rust-backed regressor, classifier, ranker, and shared configuration.
- Added typed schema and validation entry points, explicit preview namespace routing, and native schema validation.
- Added release ancestry/CI firewalls, wheel and sdist smoke tests, and benchmark provenance freshness checks.
- Removed the orphan representation and state-space Rust crates and their NumPy
  duplicate modules from the distribution; no compatibility namespace remains.
