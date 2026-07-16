# Python API Reference

> CartoBoost 0.3 keeps the root package intentionally small: the stable
> estimators are `CartoBoostRegressor`, `CartoBoostClassifier`,
> `CartoBoostRanker`, and `BoosterConfig`. Forecasting and validation use their
> named stable modules; graph, geostatistical, causal, neural, probabilistic,
> plotting, and deep surfaces are supported APIs under `cartoboost.supported`.
> See [Migrating to 0.3](../migration-v0.3.md) before updating imports.

This page lists the public Python entry points used to fit, evaluate, explain,
and save CartoBoost regression, classification, ranking, forecasting,
CartoBoost graph, and CartoBoost neural models.

The API is organized around scientific model choice: fit the same train split
as the baselines, predict the same validation rows, compute the same metrics,
and keep artifacts that make the comparison reproducible.

## Model-Choice Map

| Need | Primary entry points | Evidence to collect |
| --- | --- | --- |
| Row-level regression | `CartoBoostRegressor`, `cartoboost.schema.FeatureSchema`, sparse sets | RMSE, MAE, R2 on the same split as baselines. |
| Spatial econometrics | `cartoboost.spatial_econometrics.*` | Residual Moran's I, rho/lambda, AIC/BIC where valid, and same-split holdout metrics. |
| Scalable GP geostatistics | `cartoboost.geostats.*` | Interpolation RMSE/MAE, uncertainty calibration, duplicate-coordinate policy, and prediction variance near versus far from training points. |
| Geo-causal experiments | `cartoboost.geo_causal.*` | Treatment effect, unit/time weights, placebo distribution, spillover diagnostics, and explicit causal assumptions. |
| Binary or multiclass classification | `CartoBoostClassifier`, sparse sets | Logloss, ROC-AUC or PR-AUC, Brier score, and calibration checks on the same split as baselines. |
| Grouped ranking | `CartoBoostRanker`, grouped relevance labels | NDCG, MAP, MRR, and baseline ranking comparison by query group. |
| Demand or time-series forecasting | `ForecastFrame`, `CartoBoostLagForecaster`, typed split policy, backtester | Rolling-origin or out-of-time RMSE, MAE, WAPE, horizon metrics. |
| Spatial piecewise forecasting | `SpatialPiecewiseKrigingForecaster` | Compare base piecewise seasonal, kriging, and fused rows under the same rolling-origin folds; inspect correction, variance, neighbors, and runtime metadata. |
| Directional panel forecasting | `NeuralPanelForecaster`, `LaneNeuralPanelForecaster` | Rolling-origin RMSE, MAE, WAPE, horizon metrics, quantile diagnostics, timing, and comparison against seasonal naive plus `CartoBoostLagForecaster`. |
| Repeated-ID residual signal | `NeuralEmbeddingRegressor`, `benchmark_neural_vs_cartoboost` | Repeated-ID and cold-ID splits, with out-of-fold embeddings when possible. |
| Directed topology | `cartoboost.graph`, graph regressors, graph feature transformers | Same train-side graph construction for all rows, plus grouped or cold-source validation. |
| Diagnostics and intervals | evaluation helpers, SHAP helpers, kriging diagnostics | Residual spatial autocorrelation, interval coverage, and residual summaries by zone/hour. |

## `cartoboost.CartoBoostRegressor`

```python
CartoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    min_gain=1e-8,
    loss="l2",
    quantile_alpha=0.5,
    huber_delta=1.0,
    log_offset=1.0,
    loss_params=None,
    split_policy="auto",
    leaf_predictor="constant",
    linear_leaf_features=None,
    fuzzy=False,
    fuzzy_bandwidth=0.0,
    fuzzy_kernel="linear",
    l2_regularization=1.0,
    constant_l2_regularization=0.0,
    random_state=None,
    n_threads=None,
    monotonic_constraints=None,
    tensorboard_log_dir=None,
    tensorboard_run_name=None,
)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, feature_schema=None, sparse_sets=None, eval_set=None)` | `self` | `eval_set` is accepted but currently ignored. |
| `predict(X, sparse_sets=None)` | `numpy.ndarray` | Requires matching dense width and sparse columns. |
| `predict_additive_values(X, sparse_sets=None)` | `numpy.ndarray` | Row sums equal `predict(X)`. |
| `make_shap_explainer(background, decomposition="features", **kwargs)` | SHAP explainer | Requires optional SHAP dependency. `decomposition="weights"` returns CartoBoost's direct exact initial-value/per-tree explainer. |
| `explain_shap(X, background=..., decomposition="features", **kwargs)` | `shap.Explanation` | Convenience SHAP entry point. Weight explanations use the background component mean as their baseline and avoid permutation sampling. |
| `save(path)` | `None` | Writes a model artifact. |
| `save_weights(path, format="auto")` | `None` | Writes JSON weights or supported ONNX. |
| `CartoBoostRegressor.load(path)` | estimator | Loads model artifacts. |
| `CartoBoostRegressor.load_weights(path)` | estimator | Loads weights artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

`X`, `y`, `sample_weight`, and sparse-set tables may be NumPy arrays or
dataframe-style objects. Install `duckdb` to pass DuckDB relations directly,
or `polars` for Polars inputs.
Set `tensorboard_log_dir` to write native per-iteration training scalars to
TensorBoard event files; install `tensorboardX` for the optional
writer dependency.

Numeric model inputs must be finite. CartoBoost does not impute `NaN`, `None`,
`pd.NA`, `inf`, or `-inf` numeric feature, target, coordinate, sample-weight,
neural embedding, graph feature, or deep-model covariate values. Clean or
impute those columns before fitting or prediction. The categorical encoder is
the exception for categorical columns: missing category sentinels are converted
to a stable missing-category token.

Dense inputs may include categorical columns. Pandas categorical,
string, or object columns are encoded during fit, and columns can be marked
explicitly with `FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL` in
`feature_schema`. The fitted artifact stores the category mapping so
`predict` and `load` use the same one-hot, subset partition, ordinal, or
smoothed target-stat encoding.

For benchmark comparisons, call `fit` only on the training indices from the
chosen split and call `predict` only on the matching validation indices. If
CartoBoost receives zone, hour, distance, or target-mean
features, provide comparable encoded columns to LightGBM, XGBoost, or other
baselines before interpreting a quality delta.

## Spatial Econometrics

```python
SpatialWeights(
    n_rows,
    n_cols,
    rows,
    cols,
    values,
    row_standardize=True,
)

SpatialLagRegressor(row_standardize=True)
SpatialErrorRegressor(row_standardize=True)
SpatialDurbinRegressor(row_standardize=True)
SpatialTwoStageLeastSquares(row_standardize=True)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, spatial_weights=W)` | `self` | Fits against sparse spatial weights. |
| `predict(X, spatial_weights=W)` | `numpy.ndarray` | Requires row-compatible sparse weights. |
| `summary()` | `dict` | Includes coefficients and diagnostics. |
| `save(path)` | `None` | Writes a JSON model artifact. |
| `Class.load(path)` | estimator | Loads the native spatial model artifact. |

Diagnostics include residual Moran's I, `rho` or `lambda` when estimated,
Gaussian log likelihood, AIC/BIC where valid, sigma squared, isolated rows, and
Durbin direct/indirect/total effects. `SpatialWeights.from_neighbors(...)`
builds sparse weights from adjacency dictionaries for areal units, store trade
areas, service zones, and route cells.

## Geo-Causal Experiments

```python
GeoCausalPanel(
    rows,
    unit_col="unit_id",
    time_col="time",
    outcome_col="outcome",
    treatment_col="treatment",
    covariate_cols=None,
    latitude_col="latitude",
    longitude_col="longitude",
    region_col="region_id",
    spatial_weights=None,
)

SyntheticDIDEstimator(intervention_time="2026-03-08", seed=13)
GeoExperimentDesigner(intervention_time="2026-03-08", seed=13)
GeoLiftEstimator(intervention_time="2026-03-08", seed=13)
SpatialPlaceboTester(intervention_time="2026-03-08", seed=13)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `SyntheticDIDEstimator.fit(panel)` | `self` | Fits native synthetic DID over the pre/post split. |
| `SyntheticDIDEstimator.estimate_effect()` | `float` | Returns the average treatment effect estimate. |
| `SyntheticDIDEstimator.placebo_test(n=100)` | `list[float]` | Runs deterministic pseudo-treated placebo assignments. |
| `SyntheticDIDEstimator.summary()` | `dict` | Includes effect, weights, placebos, warnings, and assumptions. |
| `SyntheticDIDEstimator.plot(kind="placebo")` | matplotlib axes | Python-only helper; requires `matplotlib`. |
| `GeoExperimentDesigner.fit(panel).summary(candidate_count, placebo_n)` | `dict` | Chooses balanced candidate test geos and estimates detectable lift. |
| `SpatialPlaceboTester.fit(panel).summary()` | `dict` | Reports neighbor contamination, distances, and spatial exposure. |
| `InvariantRiskEncoder.fit_report(features, outcomes, regions, heldout_region=...)` | `dict` | Native-backed domain-shift representation diagnostic with supervised, domain-adversarial, invariant-risk, treatment-balance, and smoothness losses. It supplements, but does not replace, `SyntheticDIDEstimator` and `GeoExperimentDesigner`. |

Use these APIs for marketing lift, policy rollout, store openings, and network
changes. The summaries explicitly separate causal estimates from forecasts and
should be reported with the stated assumptions and spillover warnings.
Representation learning reports include an explicit warning that they do not
prove causal identification.

## Geographic Feature Encoders

```python
build_geo_sparse_sets({"pickup_zone": pickup_zone_ids})
build_zip_sparse_sets(origin_zip=origin_zip, destination_zip=destination_zip)

build_h3_sparse_sets({"pickup_h3": (pickup_lat, pickup_lng)}, resolution=9)
build_h3_route_sparse_sets(osrm_routes, name="route_h3", resolution=9)
encode_h3_route_cells(route, resolution=9)

build_s2_sparse_sets({"pickup_s2": (pickup_lat, pickup_lng)}, level=12)
build_s2_route_sparse_sets(valhalla_routes, name="route_s2", level=12)
encode_s2_route_cells(route, level=12)
```

H3 helpers require `h3`; S2 helpers require `s2sphere`.
Route encoders accept decoded route coordinate sequences, OSRM GeoJSON-style
route mappings, or Valhalla-style decoded shape mappings. Encoded polyline
strings raise `ValueError`; request decoded geometry from the routing engine
before fitting sparse route-cell features.

## Scalable GP Geostatistics

```python
NearestNeighborGPRegressor(
    kernel="exponential",
    range=1.0,
    sill=1.0,
    nugget=1e-6,
    n_neighbors=16,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    brute_force_threshold=2048,
    duplicate_tolerance=0.0,
)

SpatialGaussianProcessRegressor(...)
ResidualNNGPRegressor(base_estimator, gp=None)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, coords=coords)` | `self` | Coordinates must be finite two-column point coordinates. Duplicate coordinates within `duplicate_tolerance` raise an error. |
| `predict(X, coords=coords)` | `numpy.ndarray` | Returns local NNGP means. |
| `predict(X, coords=coords, return_std=True)` | `(mean, std)` | Standard deviations are derived from nonnegative prediction variances. |
| `predict(X, coords=coords, return_var=True)` | `(mean, variance)` | Returns variance directly. |
| `predict_interval(X, coords=coords, coverage=0.9)` | `(lower, upper)` | Uses common Gaussian z-scores without requiring SciPy. |
| `empirical_semivariogram(coords, values, ...)` | `list[dict]` | Returns lag bins with semivariance and pair counts. |
| `binned_variogram(coords, values, ...)` | `list[dict]` | Alias for the binned empirical semivariogram utility. |
| `fit_variogram_wls(bins, range_candidates=..., sill_candidates=...)` | `dict` | Weighted least-squares kernel/range/sill/nugget grid fit. |

Use `NearestNeighborGPRegressor` for point interpolation and uncertainty maps.
Use `ResidualNNGPRegressor` when a base estimator handles dense covariates and
the remaining residual field should be modeled spatially. The implementation is
CPU-only and does not require a GPU.

## `cartoboost.CartoBoostClassifier`

```python
CartoBoostClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    min_gain=1e-8,
    objective="auto",
    class_weight=None,
    split_policy="auto",
    leaf_predictor="constant",
    linear_leaf_features=None,
    fuzzy=False,
    fuzzy_bandwidth=0.0,
    fuzzy_kernel="linear",
    l2_regularization=1.0,
    constant_l2_regularization=0.0,
    random_state=None,
    n_threads=None,
)
```

`objective="auto"` selects binary logloss for two labels and multiclass
logloss for three or more labels. Python accepts arbitrary JSON-serializable
class labels, preserves their first-seen order in `classes_`, maps them to
training ids, and restores labels on `predict`. Use `class_weight="balanced"`
or a label-to-weight dict to weight gradients.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, feature_schema=None, sparse_sets=None)` | `self` | Fits binary or multiclass logloss. |
| `predict(X, sparse_sets=None)` | `numpy.ndarray` | Returns original class labels. |
| `predict_proba(X, sparse_sets=None)` | `numpy.ndarray` | Columns follow `classes_`. |
| `decision_function(X, sparse_sets=None)` | `numpy.ndarray` | Binary returns one raw margin per row; multiclass returns class margins. |
| `save(path)` | `None` | Writes CartoBoost classifier artifact plus Python class-label metadata. |
| `save_weights(path, format="auto")` | raises `NotImplementedError` | Classifier portable weights and ONNX export are intentionally unsupported. |
| `CartoBoostClassifier.load(path)` | estimator | Loads CartoBoost classifier artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

Use the same feature columns and split definitions as the baseline CartoBoost classifier.
Common labels include churn flag, high-delay bucket, cancellation risk class,
or demand-surge class.
Set `tensorboard_log_dir` to write native classifier training scalars.
Categorical columns follow the same mapping behavior as the CartoBoost
regressor and are saved with CartoBoost classifier class-label metadata.

## `cartoboost.CartoBoostRanker`

```python
CartoBoostRanker(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    min_gain=1e-8,
    objective="lambdarank",
    group_col=None,
    split_policy="auto",
    leaf_predictor="constant",
    linear_leaf_features=None,
    fuzzy=False,
    fuzzy_bandwidth=0.0,
    fuzzy_kernel="linear",
    l2_regularization=1.0,
    constant_l2_regularization=0.0,
    random_state=None,
    n_threads=None,
)
```

The CartoBoost ranker trains pairwise objectives over contiguous query groups.
Use `objective="pairwise_logit"` for unweighted pairwise logistic gradients or
`objective="lambdarank"` for NDCG-delta weighted gradients. Pass `groups` to
`fit` as group sizes whose positive entries sum to the row count, or as one
contiguous query id per row when the values do not form a valid size vector.
Set `group_col` to remove a query-id column from `X` and use those row-level
values for grouping. When `group_col` is used, the matching dense
`feature_schema` entry is removed before categorical encoding and model
training.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, groups=None, group_col=None, sample_weight=None, feature_schema=None, sparse_sets=None)` | `self` | Requires group sizes, row-level query ids, or `group_col`. |
| `predict(X, sparse_sets=None)` | `numpy.ndarray` | Returns one relevance score per row; accepts full frames with `group_col` or already-dropped feature matrices. |
| `score_groups(X, y, groups=None, group_col=None, sparse_sets=None)` | `dict` | Returns `ndcg`, `map`, and `mrr`. |
| `save(path)` | `None` | Writes CartoBoost ranker state plus Python grouping and categorical metadata. |
| `save_weights(path, format="auto")` | raises `NotImplementedError` | Ranker portable weights and ONNX export are intentionally unsupported. |
| `CartoBoostRanker.load(path)` | estimator | Loads CartoBoost ranker artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

Ranking labels are relevance scores within a group, not global regression
targets. Common examples include ranking candidate items, route alternatives,
or service actions within one query context.
Set `tensorboard_log_dir` to write native NDCG, MAP, and MRR training scalars.
Categorical ranker columns use train-side relevance labels for smoothed
target-stat encoding and persist their mappings in CartoBoost ranker artifacts.

## `cartoboost.forecasting`

Forecasting APIs validate timestamped inputs, produce deterministic forecast
tables, and provide leakage-safe evaluation for single-series and panel data.
Use these APIs when the question is future demand rather than
row-level fare or duration prediction.

`cartoboost.Prophet` is a Prophet-shaped compatibility façade for the
Rust-native piecewise model. It accepts pandas or Polars `ds`/`y` dataframes
and exposes `fit`, `predict`, `make_future_dataframe`, `add_seasonality`,
`add_regressor`, `add_country_holidays`, `setup_dataframe`, and
`predictive_samples`. Predictions use Prophet column names while retaining
CartoBoost component and interval diagnostics.

The audited Prophet 1.2.2 public method surface contains 33 methods; the
CartoBoost façade exposes all 33 names, including deterministic growth
initialization, Fourier, holiday-window, plotting, component, uncertainty,
preprocessing, and validation helpers. Run
`scripts/prophet_parity_audit.py` in both environments to compare the
machine-readable method and signature inventories. Stan/MCMC methods remain
intentionally unsupported; CartoBoost uses the Rust-native deterministic path.

Core schema:

| Entry point | Purpose |
| --- | --- |
| `ForecastFrame.from_pandas(df, timestamp_col, target_col, series_id_col=None, freq=None, ...)` | Validates and sorts single-series or panel history. |
| `ForecastResult.to_pandas()` | Returns stable forecast columns. |
| `ForecastResult.save_json(path)` / `ForecastResult.load_json(path)` | Round-trip forecast tables through JSON. |
| `PredictionInterval(level, lower, upper)` | Validates lower/upper interval bounds. |

`ForecastFrame.from_pandas(..., sample_weight_col="trip_count")` is the
opt-in path for duplicate rows at one timestamp. Duplicate series/timestamp
rows are collapsed before forecast validation: targets and numeric covariates
are weighted means, while the weight column is summed and kept as a historical
covariate.

`ForecastFrame.from_pandas(..., freq="D", allow_irregular=True)` keeps observed
timestamps as-is while recording a daily forecast cadence. This supports
Prophet-style irregular history for `PiecewiseLinearSeasonalForecaster` and
simple last-value/window baselines. Equal-step models still reject irregular
frames because their training rows, season lengths, or validation folds assume
regular spacing.

`ForecastFrame.from_pandas(..., allow_missing_targets=True)` allows `NaN` in
the target column but still rejects positive or negative infinity. Supported
models follow Prophet's missing-`y` behavior: fit on observed target rows,
preserve timestamps for horizon anchoring, and forecast after the latest input
timestamp. `PiecewiseLinearSeasonalForecaster` also keeps missing-target rows
in `history_components_frame()` with finite fitted values and null
actual/residual fields. `PiecewiseLinearSeasonalForecaster`,
`NaiveForecaster`, and non-seasonal window averages support missing targets;
regular statistical, seasonal, lag, direct, neural, intermittent-demand,
kriging, and auto models raise clear model-level errors.

`ForecastFrame.from_pandas(..., allow_missing_covariates=True)` allows `NaN` in
declared static, known-future, or historical covariates while still rejecting
infinity. This is available to every forecaster at frame construction time.
Models that ignore covariates can proceed; models that consume a missing
covariate raise a model-level error when fitting or predicting.

Forecasters:

| Entry point | Notes |
| --- | --- |
| `NaiveForecaster` | Repeats the last observed value; supports `allow_missing_targets=True` by skipping missing target rows during fit. |
| `SeasonalNaiveForecaster(season_length)` | Repeats the last seasonal cycle. |
| `ThetaForecaster(season_length=None, prediction_interval_levels=())` | Local theta method with optional seasonality and residual intervals. |
| `OptimizedThetaForecaster` | Deterministically selects theta/alpha from a validation grid. |
| `ETSForecaster` | Additive ETS with optional additive seasonality. |
| `AutoARIMAForecaster` | AutoARIMA over bounded ARIMA(p,d,q) candidates. |
| `LocalLevelKalmanForecaster` | Local-level Kalman model for noisy level-only series. |
| `KalmanForecaster` | Local-linear-trend Kalman model for noisy level and trend series. |
| `AutoLocalLevelKalmanForecaster` | Deterministic grid search over local-level process/observation variances; metadata includes `selected_params` and `validation_scores`. |
| `AutoKalmanForecaster` | Deterministic grid search over local-linear level/trend/observation variances; metadata includes `selected_params` and `validation_scores`. |
| `AutoStatsBank` | Validation bank over statistical forecasting candidates. |
| `CrostonForecaster` | Fixed Croston intermittent-demand forecaster for sparse non-negative series. |
| `SbaForecaster` | Fixed SBA intermittent-demand forecaster with Croston bias adjustment. |
| `TsbForecaster` | Fixed TSB intermittent-demand forecaster with separate demand and occurrence smoothing. |
| `KrigingForecaster` | Coordinate-aware panel forecaster using stable series coordinates and variogram controls. |
| `SpatialPiecewiseKrigingForecaster` | Piecewise seasonal CartoBoost base fused with cutoff-safe kriged regressors, residual kriging, or hybrid spatial correction; result JSON includes base mean, correction, variance, neighbors, components, and metadata. |
| `PiecewiseLinearSeasonalForecaster` | Piecewise linear seasonal local model with irregular-history fitting, Prophet-style missing-target fitting, explicit future timestamp prediction, linear, flat, or logistic growth, automatic or explicit changepoints, holiday tables and optional country holiday calendars normalized into event windows, Fourier seasonalities, conditional custom seasonalities, events, automatic extra-regressor standardization, per-component regularization, residual intervals, deterministic sampled trend uncertainty, external trend adjustments, residual shock propagation, fitted JSON round-trips, `components()` / `components_frame()` / `components_json()` forecast decomposition, and `history_components()` / `history_components_frame()` / `history_components_json()` fitted trend, movement, seasonality, event, and regressor diagnostics; interactive examples expose matching fitted artifact prediction and component helpers. |
| `CartoBoostLagForecaster` | Global recursive forecaster using leakage-safe lag, rolling, calendar, static, and known-future features with `CartoBoostRegressor`. |
| `NeuralPanelForecaster` | Neural panel forecaster with `n_lags`, `n_forecasts`, quantiles, trend, Fourier seasonality, event offsets, known-future regressors, lagged regressors, direct horizons, separate local/global/glocal seasonality, event, and regressor modes, median-first internal quantile residuals, CPU-default backend-dispatched dense prediction with optional explicit accelerators such as `"metal"`, `"rocm"`, or `"cuda"` on supported builds, and serializable metadata. |
| `LaneNeuralPanelForecaster` | Directional pair wrapper for `series_id="origin:destination"` panels; injects generated origin, destination, lane, and directional graph covariates into the panel model while keeping `A:B` distinct from `B:A`; `predict_for_lanes(horizon, series_ids)` applies fitted-lane fallback for explicit cold lane ids. |
| `AutoForecaster` | Guarded model selector over reusable internal forecasting candidates with validation metadata and fitted artifacts. |
| `NBeatsForecaster` | Deterministic N-BEATS style forecasting expert for regular forecast windows with CPU-default backend-dispatched dense prediction and optional explicit accelerators such as `"metal"`, `"rocm"`, or `"cuda"` on supported builds. |
| `NHiTSForecaster` | Deterministic N-HiTS style forecasting expert with pooled history windows and CPU-default backend-dispatched dense prediction and optional explicit accelerators such as `"metal"`, `"rocm"`, or `"cuda"` on supported builds. |
| `WeightedEnsembleForecaster` | Combines aligned component forecasts with fixed weights. |
| `BacktestWeightedEnsembleForecaster` | Reserved; raises clearly until backtest-weight learning is implemented. |

`PiecewiseLinearSeasonalForecaster` accepts `growth`, `component_mode`,
changepoint controls including `n_changepoints`, `changepoint_prior_scale`,
and explicit `changepoints` date lists, yearly/weekly/daily Fourier orders,
custom conditional seasonalities, event windows, `holidays`
tables, optional `add_country_holidays()` calendars via `holidays`,
additive or multiplicative regressor modes,
dynamic cap/floor regressors, prediction interval levels, quantile levels,
trend/coefficient uncertainty controls, `trend_adjustments`,
`trend_adjustments_by_series`, `residual_shock_window`,
`residual_shock_scale`, `residual_shock_decay`, and robust Huber fitting.
The default automatic changepoint count is 25. CartoBoost uses
`changepoint_range=1.0` by default so short-horizon lane
backtests can fit recent trend movement across the full training window. Set
`changepoint_range=0.8` for earlier changepoint placement.
Fitted models serialize with `to_json()` / `from_json()` and prediction results
preserve interval columns through JSON round-trips.

Evaluation and persistence:

| Entry point | Notes |
| --- | --- |
| `RollingOriginSplitter`, `ExpandingWindowSplitter`, `SlidingWindowSplitter` | Deterministic timestamp folds with `max(train) < min(validation)`. |
| `RollingOriginBacktester(horizon, min_train_size, step_size)` | Rust-backed rolling-origin evaluation; call `evaluate(model, frame)` and rows are aligned by `series_id`, `timestamp`, and `horizon`. |
| `ForecastMetricSet` | MAE, RMSE, MAPE, sMAPE, MASE, WAPE, bias, pinball loss, and interval metrics. |
| `ForecastRegistry` / `ForecastModelSpec` | Named model construction and optional dependency validation. |
| `ForecastArtifact` / `ForecastArtifactManifest` | JSON manifest plus CSV or Parquet forecast persistence. |
| `ForecastingConfig` | Strict TOML config parsing for forecast runs. |

Probabilistic and conformal layer:

| Entry point | Notes |
| --- | --- |
| `QuantileCartoBoostRegressor(quantiles=(0.1, 0.5, 0.9), **regressor_params)` | Fits one `CartoBoostRegressor(loss="quantile")` per level and returns a `DistributionalForecastResult` with non-crossing quantiles. |
| `ConformalIntervalRegressor(estimator, alpha=0.1)` | Wraps any estimator exposing `.fit/.predict`; fits only on train rows, calibrates on calibration residuals, and rejects holdout-leaking split order. |
| `SpatialConformalRegressor(estimator, alpha=0.1)` | Adds group-specific conformal widths for H3, S2, route, pickup zone, or spatial-block ids with global-width fallback for unseen groups. |
| `ForecastConformalCalibrator(alpha=0.1)` | Uses only residuals from cutoffs strictly before the forecast cutoff. |
| `DistributionalForecastResult` | Carries `mean`, `median`, `quantiles`, `std`, interval lower/upper bounds, and calibration metadata. |
| `pinball_loss`, `interval_coverage`, `mean_interval_width`, `crps_approximation`, `weighted_interval_score`, `pit_bins` | Distributional metrics backed by the native `cartoboost-prob` crate when the extension is available. |
| `weighted_conformal_residual_quantile`, `group_conformal_residual_quantiles`, `nearest_conformal_residual_quantiles` | Calibration primitives for weighted, group/H3/S2, spatial-block, and nearest-residual conformal workflows. |
| `benchmark_calibration_report_fields` | Emits coverage by horizon, coverage by spatial block, width by horizon, and residual Moran's I after calibration for benchmark artifacts. |

Do not report geo model quality without calibration and spatial residual
diagnostics. At minimum, report interval coverage, interval width, horizon or
spatial-block coverage, and residual Moran's I on holdout rows.

Unified model registry and geo selector:

| Entry point | Notes |
| --- | --- |
| `cartoboost.models.ModelRegistry.defaults()` | Supported registry across `forecasting`, `geo`, `graph`, `causal`, and `prob` namespaces with typed metadata. |
| `cartoboost.models.ModelSpec` / `ModelMetadata` | Constructor and metadata records for supported model families. |
| `cartoboost.models.DataContract` / `ModelEvidenceCard` | Supported evidence-contract types; no automatic geo selector or stack is shipped in v0.3. |
| `cartoboost.models.model_card(model)` | JSON-compatible params, lifecycle, and metadata summary for supported model families. |
| `cartoboost.causal` / `cartoboost.prob` | Supported aliases for geo-causal and probabilistic model namespaces. |
| `cartoboost.experimental` | Unstable research adapter namespace; adapters require explicit backends and are excluded from the stable registry. |

Representation primitives and selective state-space models are not distributed in
v0.3. The removed NumPy implementations have no import or compatibility namespace;
any future return requires a native binding and real-data evidence.

Inverted temporal transformer:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.InvertedTemporalTransformer` | Entity-token panel forecaster for synchronized wide panels with cross-entity attention, horizon-wise metrics, ablation report, and save/load parity. |
| `cartoboost.deep.InvertedEntityTransformer` | Alias for the entity-token inverted temporal transformer surface. |
| `cartoboost.deep.TemporalEntityTransformer(architecture="inverted_transformer")` | Routes the existing temporal entity surface to the inverted transformer implementation. |

Delay-aware graph transformer:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.DelayAwareGraphTransformer` / `PropagationDelayGraphForecaster` | Directed graph propagation forecaster with explicit per-edge delay priors, edge-delay sensitivity, save/load parity, and CPU/CUDA/ROCm/Metal/DirectML/WebGPU backend selection through the native accelerator contract. |
| `cartoboost.deep.DynamicAdjacencyTransformer` | Alias for the delay-aware graph transformer surface. |
| `cartoboost.deep.SpatioTemporalGraphForecaster(backbone="delay_aware_graph_transformer")` | Routes the generic graph sequence facade to the delay-aware graph implementation. |

Mixture-of-experts regime modeling:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.RegimeMoEForecaster` | Six-expert regime model for stable recurring, sparse cold-start, high-volume hub, volatile shock, long-distance pair, and low-signal fallback behavior; emits expert weights, expert predictions, combined predictions, router entropy, expert usage, and single-expert comparison metrics. |
| `cartoboost.deep.GeoTemporalMixtureOfExperts` / `PairRegimeRouter` / `EntityRegimeRouter` | Aliases for the regime MoE surface. |

Conditional flow uncertainty head:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.ConditionalFlowDistributionHead` | Native-backed residual distribution head that fits on model hidden state plus optional horizon, entity or pair, and graph context features; predicts deterministic samples, log likelihood, marginal quantiles, joint scenario paths, tail-risk metrics, and calibration metrics when actuals are supplied. |
| `cartoboost.deep.JointHorizonFlowHead` / `ResidualFlowCalibrator` | Aliases for the same conditional flow uncertainty surface. |
| `save(path)` / `ConditionalFlowDistributionHead.load(path)` | Persist and restore the fitted native JSON artifact with save/load prediction parity covered by the Python deep-model tests. |

Experimental diffusion scenario generation:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.GeoTemporalDiffusionScenarioModel` | Native-backed graph residual scenario generator for future regional fields, residual shock fields, graph-wide stress scenarios, candidate outcome distributions, or counterfactual scenario analysis. It returns scenario panels, scenario mean, scenario variance, spatial correlation, and point-forecast comparison metrics. |
| `cartoboost.deep.FlowScenarioGenerator` / `ConditionalResidualDiffusion` | Aliases for the same experimental scenario-generation surface. |
| `generate(point_forecast, edges)` | Requires a finite horizon-by-node point forecast panel and directed weighted edges. Metadata marks `capability_tier="experimental"` and excludes the output from primary benchmark evidence. |

Advanced experimental neural operators:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.GraphNeuralOperator` | Native-backed spatial field operator for horizon-by-node fields with coordinates, optional directed weighted graph edges, and optional exogenous fields. It returns `future_field`, `residual_field`, `uncertainty_field`, and advanced experimental metadata. |
| `cartoboost.deep.FourierGeoOperator` / `SpatioTemporalOperator` | Aliases for the same first-cut operator surface. |
| `GraphNeuralOperator.synthetic_benchmark()` | Runs the maintained smooth-field synthetic benchmark and reports operator RMSE, pointwise baseline RMSE, and improvement. |

Choice-set candidate competition:

| Entry point | Notes |
| --- | --- |
| `cartoboost.deep.ChoiceSetTransformer` | Native-backed candidate competition scorer. `score(candidates)` groups by `decision_id`, encodes candidate value, candidate features, context features, optional entity/pair embeddings, and existing utility/probability fields, then emits per-candidate utility and softmax choice probability. |
| `cartoboost.deep.UtilityNet` / `NestedChoiceHead` / `CounterfactualCandidateScorer` | Aliases for the same choice-set surface in this first cut. |
| `counterfactual_best(candidates)` / `calibration_report(candidates)` | Returns best candidates by decision group and Brier/ECE metrics when binary `chosen` labels are supplied. |

Optional foundation adapters:

| Entry point | Notes |
| --- | --- |
| `cartoboost.foundation.ChronosAdapter` / `TimesFMAdapter` / `MoiraiAdapter` / `TimeGPTAdapter` | Optional supported time-series foundation adapters for baselines, feature generation, cold-start experts, and benchmark comparators. |
| `cartoboost.foundation.TabPFNAdapter` / `TabPFNFeatureGenerator` / `PriorFittedBaseline` | Optional supported tabular foundation adapters with the same cache and dependency contract. |
| `FoundationForecastFeatures.benchmark_with_without_features(...)` | Compares current predictions with and without foundation features and reports RMSE delta. Adapter caches include external dependency name, version when installed, model id, model hash, input hash, output shape, and explicit AutoGeo enablement metadata. |

Python-owned JSON model artifacts include `artifact_type` and
`artifact_version` fields. CI runs `scripts/check_artifact_compatibility.py` to
verify nested version markers, save/load prediction drift, and explicit failure
for unsupported artifact versions.

`ForecastRegistry.defaults()` is an internal supported/demo registry. The stable
Python forecasting surface is intentionally limited to `NaiveForecaster`,
`SeasonalNaiveForecaster`, `CartoBoostLagForecaster`, `AutoForecaster`,
`ForecastFrame`, and `ForecastResult`; other registry entries are supported
implementations and are not part of the v0.3 source contract.

Plotting:

| Entry point | Notes |
| --- | --- |
| `cartoboost.plotting.plot` | Prophet-compatible forecast plot for `ds`/`yhat` forecast tables and Prophet-shaped local models. |
| `cartoboost.plotting.plot_backtest_metrics` | Rolling-origin or blocked-fold metric trajectories by model. |
| `cartoboost.plotting.plot_changepoint_effects` | Signed changepoint effect magnitudes. |
| `cartoboost.plotting.plot_components` | Prophet-compatible trend, holiday, seasonality, and regressor component panels. |
| `cartoboost.plotting.plot_cross_validation_metric` | Prophet-compatible horizon metric curve for cross-validation rows. |
| `cartoboost.plotting.plot_cutoff_predictions` | Cross-validation predictions grouped by cutoff. |
| `cartoboost.plotting.plot_predicted_actual` | Predicted-vs-actual scatter plot with a parity reference line. |
| `cartoboost.plotting.plot_residual_diagnostics` | Residual-vs-prediction and residual distribution diagnostics. |
| `cartoboost.plotting.plot_route_segments` | Static route-segment map with optional metric coloring. |
| `cartoboost.plotting.plot_metric_comparison` | Sorted bar chart for RMSE, MAE, WAPE, timing, or other metric rows. |
| `cartoboost.plotting.plot_forecast` | History, forecast, optional holdout actuals, and optional interval bands. |
| `cartoboost.plotting.plot_forecast_component` | Prophet-compatible single component plot. |
| `cartoboost.plotting.plot_forecast_components` | Trend, seasonal, event, or other component panels with optional changepoints. |
| `cartoboost.plotting.plot_horizon_metrics` | Forecast metric trajectories by horizon and model. |
| `cartoboost.plotting.plot_interval_calibration` | Nominal-vs-observed interval coverage with optional mean interval width. |
| `cartoboost.plotting.plot_plotly`, `plot_components_plotly`, `plot_forecast_component_plotly`, `plot_seasonality_plotly` | Prophet-compatible interactive Plotly utilities. |
| `cartoboost.plotting.plot_seasonality`, `plot_weekly`, `plot_yearly` | Prophet-compatible seasonality curves. |
| `cartoboost.plotting.plot_seasonality_curve` | Periodic component curve with optional uncertainty bands. |
| `cartoboost.plotting.plot_spatial_points` | Static latitude/longitude point map with optional metric coloring. |
| `cartoboost.plotting.save_figure` | Creates parent directories and writes a Matplotlib figure. |
| `cartoboost.plotting.seasonality_plot_df`, `set_y_as_percent`, `add_changepoints_to_plot`, `get_forecast_component_plotly_props`, `get_seasonality_plotly_props` | Prophet-compatible plotting helper utilities matching `prophet.plot` 1.2.2 public names. |
| `cartoboost.plotting.write_pydeck_point_map` | Interactive PyDeck point map written to HTML. |
| `cartoboost.plotting.write_pydeck_route_map` | Interactive PyDeck route arc map written to HTML. |
| `cartoboost.plotting.write_plot_report` | Writes a named bundle of provided diagnostics and returns output paths. |

See [Plotting](../plotting.md) for full examples. Install
`geopandas`, `matplotlib`, `pydeck`, and `shapely` when they are not already
available.

Sequence primitives:

| Entry point | Notes |
| --- | --- |
| `SequenceSeries`, `SequenceRow`, `ReferenceSignal` | Generic sequence and reference-axis containers for sequence utilities. |
| `SequenceStateSpaceConfig` | Process and observation noise configuration for sequence EKF/UKF/RTS routines. |
| `ReferencePathConfig` | Robust emission scale, Student-t degrees of freedom, transition penalty, and start-axis penalty for discrete reference-path inference. |
| `validate_sequence_frame` | Hard-fails on unordered positions, empty known prefixes, empty prediction suffixes, duplicate reference axes, and target leakage into prediction rows. |
| `forward_ekf`, `ukf_reference`, `rts_smoother`, `missing_target_continuation` | Generic state-space continuation over a reference signal. These do not replace the local-level or local-linear forecasting APIs. |
| `reference_path_viterbi`, `reference_path_posterior_mean` | Domain-neutral path inference over a reference axis. |
| `sequence_blend` | Fixed, validation-derived, or constrained nonnegative blending of aligned candidate sequence predictions. |
| `generate_group_oof_candidate_rows`, `validate_oof_meta_training`, `per_group_error_summary` | Group-level OOF candidate generation, meta-training leakage checks, and group RMSE/MAE summaries. |

For honest forecasting evidence, prefer `RollingOriginBacktester` or an
explicit future holdout over random row splits. Keep `series_id`, `timestamp`,
and `horizon` in the forecast table so CartoBoost and external tools can be
scored on the same lane/date rows.

## `cartoboost.NeuralEmbeddingRegressor`

```python
regressor = NeuralEmbeddingRegressor(
    dim=16,
    fallback="global_mean_vector",
    random_state=None,
    neural_transformer=None,
    use_residual=True,
    oof_folds=1,
    drop_id_column=True,
    id_column=None,
    support_prior_strength=1.0,
    base_model_kwargs=None,
    final_model_kwargs=None,
)
```

Optional neural-augmented estimator that appends ID embedding features to dense
features and trains a tabular model on the expanded matrix.

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, sparse_sets=None, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None, **fit_kwargs)` | `self` | Pass 1D IDs for one key or 2D IDs for multi-key embeddings. `fallback_ids` provides hierarchical fallback chains; `neighbor_ids` provides graph-aware fallback. |
| `predict(X, sparse_sets=None, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None)` | `numpy.ndarray` | Requires the same ID key count used during fit. |
| `transform(X, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None)` | `numpy.ndarray` | Returns dense matrix with neural columns appended. |
| `score(X, y, sparse_sets=None, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None)` | `float` | Computes mean absolute error on predictions. |
| `timings` | `dict[str, float]` | Fit timing components in milliseconds: `base_fit_ms`, `neural_fit_ms`, `final_fit_ms`. |

Set `oof_folds > 1` to train final-model embedding columns out of fold. Use
`support_prior_strength` to shrink rare IDs more strongly toward their prior.
Report neural results with both repeated-ID and cold-ID validation. A random
split gain is not evidence of cold location, lane, or route generalization.

## General Utilities

Utilities independent of the regressor and forecasting model APIs:

See [General Utilities](../general_utilities.md) for complete examples.
These helpers are useful for diagnostics and baselines, but quality claims
still need the same split, target transformation, and metric definitions as the
main model comparison.

| Entry point | Purpose |
| --- | --- |
| `cartoboost.utilities.naive_forecast(values, horizon)` and related `seasonal_naive_forecast`, `theta_forecast`, `optimized_theta_forecast`, `ets_forecast`, `arima_forecast`, `auto_arima_forecast` | Supported single-series forecasts for plain numeric sequences. |
| `cartoboost.local_level_kalman_filter(values, ..., horizon=0, interval_z=...)` | Local-level Kalman filtering for numeric sequences. Returns final level and variance, fixed-interval smoothed states, per-step estimates with fitted/residual/gain/likelihood diagnostics, residual summary metrics, optional flat forecast means, and optional forecast distributions with normal bounds. |
| `cartoboost.local_level_kalman_forecast(values, horizon, ...)` | Local-level Kalman forecast utility. |
| `cartoboost.utilities.kalman_filter(values, level_process_variance=..., trend_process_variance=..., observation_variance=..., horizon=0, interval_z=...)` | Supported local-linear Kalman filtering for numeric sequences. |
| `cartoboost.local_linear_trend_kalman_forecast(values, horizon, ...)` | Local-linear trend Kalman forecast utility. |
| `cartoboost.utilities.croston_forecast`, `sba_forecast`, `tsb_forecast` | Supported intermittent-demand utilities for non-negative numeric sequences. |
| `cartoboost.utilities.ordinary_kriging_predict(observations, targets, range=..., nugget=..., detailed=False)` | Supported ordinary kriging for observed `(x, y, value)` triples and target `(x, y)` coordinates. |
| `cartoboost.utilities.ordinary_kriging_leave_one_out(observations, ...)` | Supported leave-one-out kriging diagnostics for observed coordinates. |
| `cartoboost.utilities.empirical_variogram(observations, ...)` | Supported binned empirical semivariogram with lag ranges, mean lag distances, semivariances, and pair counts. |
| `cartoboost.utilities.fit_ordinary_kriging_variogram(observations, ...)` | Supported weighted least-squares variogram fitting over model/range/nugget/sill candidate grids. |
| `cartoboost.utilities.ordinary_kriging_leave_one_out_diagnostics(observations, ...)` | Supported leave-one-out predictions plus residual metrics such as bias, MAE, RMSE, standardized residuals, interval coverage, and average kriging variance. |
| `cartoboost.forecasting.sequence.*` | Sequence reference utilities for known-prefix continuation, reference path inference, leakage-safe OOF row generation and validation, group metrics, and aligned candidate blending. |

## Direct Graph Encoders

Neighborhood-based encoders for direct graph embeddings and optional downstream
feature workflows.

`Node2VecEncoder` is for transductive directed/weighted random-walk embeddings;
`GraphSageEncoder` is for homogeneous graphs; `HeteroGraphSageEncoder` is for
typed relations; `HinSageEncoder` is the typed-schema HinSAGE surface with
relation-aware sampling and link feature construction.
GraphSAGE-style encoders and standalone link predictors accept
`backend="cpu"` by default, `backend="auto"` as a CPU-resolving alias, or
available accelerators such as `"metal"`, `"rocm"`, or `"cuda"` for the shared dense forward and pair-score
kernels on supported builds where the corresponding native backend is compiled
in.

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(node_count, edges, edge_weights=None)` | `list[list[float]]` | Trains `Node2VecEncoder` on directed edges with optional non-negative weights. |
| `fit(node_count, edges, node_features)` | `list[list[float]]` | Trains GraphSAGE encoder weights on an edge list and returns node embeddings. |
| `fit(node_types, edges, node_features)` | `list[list[float]]` | Trains `HinSageEncoder` on typed nodes and `(source, target, relation)` edges validated against `edge_type_triples`. |
| `encode(node_features)` | `list[list[float]]` | Encodes features with learned weights for inference. |
| `link_embeddings(embeddings, pairs)` | `list[list[float]]` | Builds HinSAGE link-prediction features as `[source, target, abs_delta, product]`. |
| `loss_curve()` | `list[float]` | Per-epoch training loss history. |
| `save_artifact_json(path)` | `None` | Persists deterministic encoder artifact. |
| `to_artifact_json()` | `str` | Emits JSON artifact payload. |
| `load_artifact_json(path)` | `Node2VecEncoder` / `GraphSageEncoder` / `HeteroGraphSageEncoder` / `HinSageEncoder` | Loads serialized encoder state. |

## `cartoboost.graph` Feature Helpers

The `cartoboost.graph` package contains CartoBoost graph models, CartoBoost
link predictors, and graph-feature helpers. Use the direct models when the
graph itself is the thing you want to score; use `GraphFeatureTransformer`
only when you want dense and sparse graph inputs for another estimator.

For source-target modeling, build graph features from train-side relationships
when validation edges or timestamps must remain unseen. If validation edges
leak into topology construction, label the result as transductive rather than a
deployment holdout.

| Entry point | Purpose |
| --- | --- |
| `GraphFeatureConfig.from_config(cfg)` | Validates YAML-style graph config blocks with schema, directionality, metapaths, encoder settings, and outputs. |
| `GraphSchema`, `EdgeType`, `DirectionalityConfig` | Describe directed heterogeneous graph schemas and source-target requirements. |
| `DirectedMetaPath` | Validates typed node/relation/node metapaths against a `GraphSchema`. |
| `GraphFeatureTransformer.from_config(cfg)` | Fits node2vec, GraphSAGE, HeteroGraphSAGE, or typed-schema HinSAGE encoders and emits a `GraphFeatureBundle`. |
| `Node2VecFeatureEncoder.from_config(cfg)` | Configures node2vec with `dim`, `walk_length`, `walks_per_node`, `window_size`, `p`, `q`, and optional edge weights. |
| `HinSageFeatureEncoder.from_config(cfg)` | Configures HinSAGE with `node_type_count`, `edge_type_triples`, and optional per-relation `neighbor_samples`. |
| `GraphFeatureBundle` | Carries dense graph columns, optional sparse sets, feature names, node IDs, and provenance metadata. |
| `MetaPathWalkGenerator`, `TemporalWalkGenerator` | Generate constrained directed metapath walks and monotonic temporal walks. |
| `materialize_source_target_pair_nodes(edges)` | Creates stable `("od_pair", source, target)` nodes so `A -> B` and `B -> A` stay distinct. |
| `link_prediction_report(labels, scores, query_ids=None, k=10)` | Reports AUC/AP and optional top-k/MRR ranking metrics. |

Directional source-target features are opt-in through
`directionality.compute_asymmetry_features`. Supported outputs include
`graph_source_target_embedding`, `graph_target_source_embedding`,
`graph_forward_reverse_similarity_delta`, `graph_source_outbound_strength`,
`graph_target_inbound_strength`, `graph_flow_imbalance_ratio`,
`graph_directed_temporal_drift`, and generic flow metrics such as
`graph_source_target_affinity` and `graph_flow_asymmetry`.

## Standalone Graph And Neural Models

Use these models when graph or neural embeddings should score directly instead
of becoming generated feature columns.

CartoBoost regressors:

| Entry point | Fit signature | Notes |
| --- | --- | --- |
| `NeuralEmbeddingStandaloneRegressor` | `fit(ids, y, dense=None)` | Supervised ID embeddings with optional dense row features. |
| `Node2VecStandaloneRegressor` | `fit(node_count, edges, row_nodes, y, row_targets=None, dense=None, edge_weights=None)` | Graph-only random-walk embeddings plus row-level regression. |
| `GraphSageStandaloneRegressor` | `fit(node_features, edges, row_nodes, y, row_targets=None, dense=None)` | Homogeneous graph regression with node attributes. |
| `HeteroGraphSageStandaloneRegressor` | `fit(node_features, edges, row_nodes, y, row_targets=None, dense=None)` | Typed-edge regression without strict HinSAGE schema metadata. |
| `HinSageStandaloneRegressor` | `fit(node_features, node_types, edges, row_nodes, y, row_targets=None, dense=None)` | Typed-node and typed-relation graph regression. |

All standalone regressors expose `predict`, `score`, `save`, and `load`.

CartoBoost link predictors:

| Entry point | Fit signature | Notes |
| --- | --- | --- |
| `Node2VecLinkPredictor` | `fit(node_count, edges, edge_weights=None)` | Directed/weighted graph link scoring from random-walk embeddings. |
| `GraphSageLinkPredictor` | `fit(node_features, edges)` | Homogeneous graph link scoring. |
| `HeteroGraphSageLinkPredictor` | `fit(node_features, edges)` | Typed-edge link scoring. |
| `HinSageLinkPredictor` | `fit(node_features, node_types, edges)` | Typed-node and typed-relation link scoring. |

All CartoBoost link predictors expose `predict_scores`, `report`, `save`, and
`load`.

### Function: `cartoboost.benchmark_neural_vs_cartoboost`

```python
benchmark_neural_vs_cartoboost(
    X,
    y,
    ids,
    split_ratio=0.8,
    neural_kwargs=None,
    cartoboost_kwargs=None,
)
```

Returns:

- `structured_mae`
- `neural_mae` (reported as `hybrid_mae` in the current helper payload)
- `cartoboost_fit_ms`
- `cartoboost_predict_ms`
- `neural_fit_ms` (reported as `hybrid_fit_ms` in the current helper payload)
- `neural_predict_ms` (reported as `hybrid_predict_ms` in the current helper payload)

Use this helper for quick, deterministic smoke comparisons on a held-out split.
For publishable evidence, replace the helper's simple split with the blocked
or cold-ID split that matches the deployment question.

## `cartoboost.schema.FeatureSchema`

```python
FeatureSchema(dense, sparse_sets=None)
```

Helper for declaring numeric, periodic, and sparse-set feature roles.

| Method | Returns | Notes |
| --- | --- | --- |
| `to_dict()` | `dict` | Compact Python representation. |
| `to_json(dense_width, sparse_names)` | `str` | JSON-encoded schema payload. |

## SHAP Helpers

```python
cartoboost.explain.make_shap_explainer(model, background, **kwargs)
cartoboost.explain.explain_shap(model, X, background=..., **kwargs)
```

These functions are also available as estimator methods. See
[SHAP Support](../shap.md).

## Validation Manifests

Validation is native-backed and returns a typed manifest that records fold
indices and reproducibility metadata:

```python
import cartoboost
from cartoboost.geo import CoordinateMatrix
from cartoboost.validation import native_spatial_split

manifest = native_spatial_split(
    CoordinateMatrix(x_values, y_values, crs="EPSG:2263"),
    n_folds=5,
    dataset_fingerprint="sha256:...",
    coordinate_crs_note="EPSG:2263 projected taxi-zone centroids",
    model_version=cartoboost.__version__,
    dependency_versions={"cartoboost": cartoboost.__version__},
)
fold_id, train_idx, test_idx = manifest.folds()[0]
```

Use `native_buffered_spatial_split`, `native_grouped_split`,
`native_temporal_split`, and `native_spatial_temporal_split` for the other
stable policies. Store the manifest hash with every benchmark artifact; the
split is part of the experiment definition and must be identical for every
model in a comparison.

## Metric And Diagnostic Helpers

```python
cartoboost.metrics.logloss(y_true, y_proba)
cartoboost.metrics.roc_auc(y_true, y_score)
cartoboost.metrics.pr_auc(y_true, y_score)
cartoboost.metrics.brier_score(y_true, y_proba)
cartoboost.metrics.ece_calibration_error(y_true, y_proba, n_bins=10)
cartoboost.metrics.ndcg_at_k(relevance, scores, groups=None, k=None)
cartoboost.metrics.mean_average_precision(relevance, scores, groups=None, k=None)
cartoboost.metrics.mean_reciprocal_rank(relevance, scores, groups=None, k=None)
cartoboost.metrics.residual_morans_i(coordinates, residuals)
cartoboost.metrics.spatial_cv_gap(random_cv_score, spatial_cv_score)
```

Classification metrics are deterministic NumPy implementations for binary or
multiclass probability checks where applicable. Ranking metrics accept either
one global ranking, positive group sizes that sum to the row count, or
contiguous query ids when the values do not form a valid size vector.
`residual_morans_i` uses dense pairwise spatial weights and is intended for
validation samples.

## I/O

File-format helpers are not part of the stable 0.3 source API. Use Python's
standard `json` module or a dedicated dataframe/geospatial reader, then pass
validated NumPy or dataframe inputs to the stable estimators.

## Geo Encoding Helpers

```python
cartoboost.geo.clockwise_bearing_unit_vector((pickup_x, pickup_y), (dropoff_x, dropoff_y))
cartoboost.geo.initial_bearing_unit_vector_latlng(
    (pickup_latitude, pickup_longitude),
    (dropoff_latitude, dropoff_longitude),
)
cartoboost.geo.route_feature_vector((pickup_x, pickup_y), (dropoff_x, dropoff_y))
cartoboost.geo.radial_anchor_distances((pickup_x, pickup_y), anchors)
cartoboost.geo.rbf_anchor_features((pickup_x, pickup_y), anchors, length_scale=3.0)
cartoboost.geo.local_frame_features((pickup_x, pickup_y), origin=(0.0, 0.0), axis=(1.0, 1.0))
cartoboost.h3.build_h3_sparse_sets(
    {"pickup_h3": (pickup_latitude, pickup_longitude)},
    resolution=9,
    parent_resolutions=[5, 7],
)
cartoboost.h3.build_h3_route_sparse_sets(osrm_routes, name="route_h3", resolution=9)
cartoboost.s2.build_s2_sparse_sets(
    {"pickup_s2": (pickup_latitude, pickup_longitude)},
    level=12,
    parent_levels=[8, 10],
)
cartoboost.s2.build_s2_route_sparse_sets(valhalla_routes, name="route_s2", level=12)
```

Bearing helpers return continuous `(east, north)` unit-vector columns. Use the
planar helper for projected coordinates and the latitude/longitude helper for
great-circle initial bearings. Zero-distance pairs return `None`.
Route helpers add midpoint and direct distance columns. H3/S2 route encoders
turn decoded OSRM or Valhalla route geometries into variable-length sparse
route-cell rows. Radial and RBF helpers emit one column per explicit anchor.
Local-frame helpers emit `(along_axis, cross_axis)` for corridor-style features.

These helpers return `sparse_sets` dictionaries suitable for
`CartoBoostRegressor.fit(..., sparse_sets=...)`. H3 auto-encoding requires the
optional `h3` package; S2 auto-encoding requires `s2sphere`. Deterministic
normalization, coordinate/level validation, scaffold expansion, and sparse-row
assembly are handled by CartoBoost.
