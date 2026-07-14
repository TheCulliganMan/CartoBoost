# Migrating to CartoBoost 0.3

CartoBoost 0.3 is a source-API reset. The stable root package contains only
the Rust-backed `CartoBoostRegressor`, `CartoBoostClassifier`,
`CartoBoostRanker`, `BoosterConfig`, and `__version__`.

Use the named stable modules for the rest of the product:

- `cartoboost.schema` for `FeatureSchema` and typed feature specifications.
- `cartoboost.validation` for `SplitPolicy`, `SplitManifest`, leakage-safe splits,
  and native manifest constructors such as `native_spatial_split` and
  `native_out_of_time_split`.
- `cartoboost.forecasting` for `ForecastFrame`, forecasters, backtesting, and metrics.
- `cartoboost.supported` for graph, geostatistical, probabilistic, causal, neural, and
  accelerator surfaces without a compatibility promise.

Stable model files now use the `cartoboost.model` v2 envelope. Valid 0.2.45
stable estimator artifacts are migrated in memory when loaded; supported and
experimental artifacts are rejected explicitly. Source imports and constructor
aliases are not migrated.

The former NumPy representation and selective state-space modules were removed
with their orphan Rust crates. They have no compatibility namespace; a future
return requires native bindings and real-data evidence.

The PyPI wheel does not install a `cartoboost` executable. Run the Rust CLI from
a source checkout with `cargo run -p cartoboost-cli -- --help`.
