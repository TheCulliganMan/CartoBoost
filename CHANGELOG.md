# Changelog

## Unreleased

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
