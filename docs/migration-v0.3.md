# Migrating to CartoBoost 0.3

CartoBoost 0.3 reorganizes the Python API around a small stable root package.
Use this page when upgrading application code or loading an older model artifact.
The stable root package contains only
the Rust-backed `CartoBoostRegressor`, `CartoBoostClassifier`,
`CartoBoostRanker`, `BoosterConfig`, and `__version__`.

## Import Changes

Use named modules for the rest of the package:

- `cartoboost.schema` for `FeatureSchema` and typed feature specifications.
- `cartoboost.validation` for `SplitPolicy`, `SplitManifest`, leakage-safe splits,
  and native manifest constructors such as `native_spatial_split` and
  `native_out_of_time_split`.
- `cartoboost.forecasting` for `ForecastFrame`, forecasters, backtesting, and metrics.
- `cartoboost.supported` for graph, geostatistical, probabilistic, causal, neural, and
  accelerator surfaces without a compatibility promise.

## Artifact Compatibility

Stable model files now use the `cartoboost.model` v2 envelope. Valid 0.2.45
stable estimator artifacts are migrated in memory when loaded; supported and
experimental artifacts are rejected explicitly. Source imports and constructor
aliases are not migrated.

## Removed APIs

The former NumPy representation and selective state-space modules were removed.
There is no compatibility import. Replace them with a documented stable or
supported model whose validation contract matches the task.

## CLI Change

The PyPI wheel does not install a `cartoboost` executable. Run the Rust CLI from
a source checkout with `cargo run -p cartoboost-cli -- --help`.

## Upgrade Checklist

1. Replace root imports with their documented named-module imports.
2. Retrain or test-load persisted artifacts before deploying the new package.
3. Re-run the original validation split and compare predictions and metrics.
4. Update source-checkout CLI invocations to use `cargo run -p cartoboost-cli`.
5. Treat supported or experimental surfaces as explicit dependencies rather than stable compatibility promises.
