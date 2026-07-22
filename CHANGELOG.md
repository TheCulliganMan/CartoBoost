# Changelog

## 0.3.11 — Native Feature Contributions

- Added exact background-free, path-dependent TreeSHAP for supported
  hard axis-aligned regression trees, with LightGBM-compatible feature columns
  followed by a constant base-value column.
- Added `predict(..., pred_contrib=True)`,
  `predict_feature_contributions(...)`, and fitted `feature_name_` metadata,
  including aggregation of encoded categorical columns back to original input
  features.
- Added native-backed `shap.Explanation` helpers while preserving explicit
  background-based SHAP and the initial-value/per-tree additive decomposition.

## 0.3.10 — Minimal Python Installation

- Kept the standard Python installation limited to CartoBoost and NumPy.
- Kept dataframe, geospatial, visualization, explainability, optimization,
  accelerator, and foundation-model integrations available through named extras.
- Added a release packaging guard that enforces the minimal dependency contract.

## 0.3.9 — Unified Model API and Portable Model Artifacts

- Made every shipped modeling family available through the standard
  `cartoboost` and `cartoboost.forecasting` imports, including graph, spatial,
  probabilistic, causal, neural, and foundation-model adapters.
- Kept documented model classes available from the standard API, with optional
  third-party backends installed through their named extras.
- Added portable `save_weights`, `load_weights`, and pickle round-trips for
  native boosters, local forecasters, deep models, spatial estimators, and
  Python orchestration wrappers.

## 0.3.8 — Exact Tree-Component SHAP

- Made `CartoBoostRegressor` weight SHAP decomposition exact and direct. The
  `decomposition="weights"` path now returns background-centered initial-value
  and per-tree attributions without constructing SHAP's generic permutation
  explainer, and exposes `expected_value` and `shap_values(...)` for existing
  analysis pipelines.
- Added native SHAP `TreeExplainer` export for dense axis-aligned,
  constant-leaf CartoBoost models, preserving initial prediction, tree scaling,
  thresholds, missing routing, and node weights.
- Documented decomposition selection, component baseline semantics, sparse-set
  behavior, and structured-routing support.

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
- Added typed schema and validation entry points, explicit supported namespace routing, and native schema validation.
- Added release ancestry/CI firewalls, wheel and sdist smoke tests, and benchmark provenance freshness checks.
- Removed the orphan representation and state-space Rust crates and their NumPy
  duplicate modules from the distribution; no compatibility namespace remains.
