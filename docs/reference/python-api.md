# Python API Reference

This page lists the public Python entry points used to fit, evaluate, explain,
and save CartoBoost regression, classification, ranking, forecasting,
standalone graph, and standalone neural models.

The API is organized around scientific model choice: fit the same train split
as the baselines, predict the same validation rows, compute the same metrics,
and keep artifacts that make the comparison reproducible.

## Model-Choice Map

| Need | Primary entry points | Evidence to collect |
| --- | --- | --- |
| Taxi fare or duration regression | `CartoBoostRegressor`, `FeatureSchema`, sparse zone sets | RMSE, MAE, R2 on random and spatial pickup-zone holdouts. |
| Taxi trip, route, or zone classification | `CartoBoostClassifier`, sparse zone sets | Logloss, ROC-AUC or PR-AUC, Brier score, and calibration checks on the same split as baselines. |
| Grouped route, customer, or zone ranking | `CartoBoostRanker`, grouped relevance labels | NDCG, MAP, MRR, and baseline ranking comparison by query group. |
| Pickup/dropoff demand forecasting | `ForecastFrame`, `CartoBoostLagForecaster`, splitters, backtester | Rolling-origin or out-of-time RMSE, MAE, WAPE, horizon metrics. |
| Spatial piecewise seasonal taxi forecasting | `SpatialPiecewiseKrigingForecaster` | Compare base piecewise seasonal, kriging, and fused rows under the same rolling-origin folds; inspect correction, variance, neighbors, and runtime metadata. |
| Directional neural lane forecasting | `NeuralPairwiseForecaster`, `LaneNeuralPairwiseForecaster` | Rolling-origin RMSE, MAE, WAPE, horizon metrics, quantile diagnostics, timing, and comparison against seasonal naive plus `CartoBoostLagForecaster`. |
| Repeated-ID residual signal | `NeuralEmbeddingRegressor`, `benchmark_neural_vs_cartoboost` | Repeated-ID and cold-ID splits, with out-of-fold embeddings when possible. |
| Pickup/dropoff topology | `cartoboost.graph`, standalone graph regressors, graph feature transformers | Same train-side graph construction for all rows, plus grouped or cold-source validation. |
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
    splitters=None,
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
)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, feature_schema=None, sparse_sets=None, eval_set=None)` | `self` | `eval_set` is accepted but currently ignored. |
| `predict(X, sparse_sets=None)` | `numpy.ndarray` | Requires matching dense width and sparse columns. |
| `predict_additive_values(X, sparse_sets=None)` | `numpy.ndarray` | Row sums equal `predict(X)`. |
| `make_shap_explainer(background, **kwargs)` | SHAP explainer | Requires optional SHAP dependency. |
| `explain_shap(X, background=..., **kwargs)` | `shap.Explanation` | Convenience SHAP entry point. |
| `save(path)` | `None` | Writes a model artifact. |
| `save_weights(path, format="auto")` | `None` | Writes JSON weights or supported ONNX. |
| `CartoBoostRegressor.load(path)` | estimator | Loads model artifacts. |
| `CartoBoostRegressor.load_weights(path)` | estimator | Loads weights artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

`X`, `y`, `sample_weight`, and sparse-set tables may be NumPy arrays or
dataframe-style objects. Install `cartoboost[duckdb]` to pass DuckDB relations
directly, or `cartoboost[polars]` for Polars inputs.

Dense inputs may include categorical columns. Pandas categorical,
string, or object columns are encoded during fit, and columns can be marked
explicitly with `FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL` in
`feature_schema`. The fitted artifact stores the category mapping so
`predict` and `load` use the same one-hot, subset partition, ordinal, or
smoothed target-stat encoding.

For benchmark comparisons, call `fit` only on the training indices from the
chosen split and call `predict` only on the matching validation indices. If
CartoBoost receives pickup/dropoff zone, hour, distance, or target-mean
features, provide comparable encoded columns to LightGBM, XGBoost, or other
baselines before interpreting a quality delta.

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
    splitters=None,
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
| `save(path)` | `None` | Writes classifier artifact plus Python class-label metadata. |
| `save_weights(path, format="auto")` | raises `NotImplementedError` | Classifier portable weights and ONNX export are intentionally unsupported. |
| `CartoBoostClassifier.load(path)` | estimator | Loads classifier artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

Use the same feature columns and split definitions as the baseline classifier.
For taxi classification, common labels include airport-trip flag, high-delay
bucket, cancellation risk class, or pickup-demand surge class.
Categorical columns follow the same mapping behavior as the regressor and are
saved with classifier class-label metadata.

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
    splitters=None,
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

The ranker trains pairwise objectives over contiguous query groups.
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
| `save(path)` | `None` | Writes ranker state plus Python grouping and categorical metadata. |
| `save_weights(path, format="auto")` | raises `NotImplementedError` | Ranker portable weights and ONNX export are intentionally unsupported. |
| `CartoBoostRanker.load(path)` | estimator | Loads ranker artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

Ranking labels are relevance scores within a group, not global regression
targets. For taxi workflows, examples include ranking candidate dropoff zones,
route alternatives, or service actions within one pickup/customer context.
Categorical ranker columns use train-side relevance labels for smoothed
target-stat encoding and persist their mappings in ranker artifacts.

## `cartoboost.forecasting`

Forecasting APIs validate timestamped inputs, produce deterministic forecast
tables, and provide leakage-safe evaluation for single-series and panel data.
Use these APIs when the question is future pickup/dropoff demand rather than
row-level fare or duration prediction.

Core schema:

| Entry point | Purpose |
| --- | --- |
| `ForecastFrame.from_pandas(df, timestamp_col, target_col, series_id_col=None, freq=None, ...)` | Validates and sorts single-series or panel history. |
| `ForecastResult.to_pandas()` | Returns stable forecast columns. |
| `ForecastResult.save_json(path)` / `ForecastResult.load_json(path)` | Round-trip forecast tables through JSON. |
| `PredictionInterval(level, lower, upper)` | Validates lower/upper interval bounds. |

`ForecastFrame.from_pandas(..., sample_weight_col="trip_count")` is the
opt-in path for duplicate taxi observations at one timestamp. Duplicate
series/timestamp rows are collapsed before forecast validation: targets and
numeric covariates are weighted means, while the weight column is summed and
kept as a historical covariate.

Forecasters:

| Entry point | Notes |
| --- | --- |
| `NaiveForecaster` | Repeats the last observed value. |
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
| `CrostonForecaster` | Fixed Croston intermittent-demand forecaster for sparse non-negative taxi demand. |
| `SbaForecaster` | Fixed SBA intermittent-demand forecaster with Croston bias adjustment. |
| `TsbForecaster` | Fixed TSB intermittent-demand forecaster with separate demand and occurrence smoothing. |
| `KrigingForecaster` | Coordinate-aware panel forecaster using stable series coordinates and variogram controls. |
| `SpatialPiecewiseKrigingForecaster` | Piecewise seasonal CartoBoost base fused with cutoff-safe kriged regressors, residual kriging, or hybrid spatial correction; result JSON includes base mean, correction, variance, neighbors, components, and metadata. |
| `PiecewiseLinearSeasonalForecaster` | Piecewise linear seasonal local model with linear, flat, or logistic growth, automatic or explicit changepoints, holiday tables and optional country holiday calendars normalized into event windows, Fourier seasonalities, conditional custom seasonalities, events, automatic extra-regressor standardization, per-component regularization, residual intervals, deterministic sampled trend uncertainty, external trend adjustments, residual shock propagation, fitted JSON round-trips, `components()` / `components_json()` forecast decomposition, and `history_components()` / `history_components_frame()` / `history_components_json()` fitted trend, movement, seasonality, event, and regressor diagnostics; interactive examples expose matching fitted artifact prediction and component helpers. |
| `CartoBoostLagForecaster` | Global recursive forecaster using leakage-safe lag, rolling, calendar, static, and known-future features with `CartoBoostRegressor`. |
| `NeuralPairwiseForecaster` | Rust-native neural panel forecaster with `n_lags`, `n_forecasts`, quantiles, trend, Fourier seasonality, event offsets, known-future regressors, lagged regressors, direct horizons, local/global component modes, and serializable metadata. |
| `LaneNeuralPairwiseForecaster` | Taxi lane wrapper for `series_id="origin:destination"` panels; records origin, destination, lane, directional graph-feature, and cold-lane fallback metadata while keeping `A:B` distinct from `B:A`; `predict_for_lanes(horizon, series_ids)` applies fitted-lane fallback for explicit cold lane ids. |
| `AutoForecaster` | Guarded model selector over reusable internal forecasting candidates with validation metadata and fitted artifacts. |
| `NBeatsForecaster` | Deterministic N-BEATS style forecasting expert for regular forecast windows. |
| `NHiTSForecaster` | Deterministic N-HiTS style forecasting expert with pooled history windows. |
| `WeightedEnsembleForecaster` | Combines aligned component forecasts with fixed weights. |
| `BacktestWeightedEnsembleForecaster` | Reserved; raises clearly until backtest-weight learning is implemented. |

`PiecewiseLinearSeasonalForecaster` accepts `growth`, `component_mode`,
changepoint controls including `n_changepoints`, `changepoint_prior_scale`,
and explicit `changepoints` date lists, yearly/weekly/daily Fourier orders,
custom conditional seasonalities, event windows, `holidays`
tables, optional `add_country_holidays()` calendars via `cartoboost[holidays]`,
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
| `RollingOriginBacktester(horizon, min_train_size, step_size)` | Fits a fresh model per fold and aligns rows by `series_id`, `timestamp`, and `horizon`. |
| `ForecastMetricSet` | MAE, RMSE, MAPE, sMAPE, MASE, WAPE, bias, pinball loss, and interval metrics. |
| `ForecastRegistry` / `ForecastModelSpec` | Named model construction and optional dependency validation. |
| `ForecastArtifact` / `ForecastArtifactManifest` | JSON manifest plus CSV or Parquet forecast persistence. |
| `ForecastingConfig` | Strict TOML config parsing for forecast runs. |

`ForecastRegistry.defaults()` contains only constructible public forecast
models with fit/predict behavior: `naive`, `seasonal_naive`,
`theta`, `optimized_theta`, `piecewise_linear_seasonal`, `ets`, `arima`,
`auto_arima`, `autostats_bank`, `croston`, `sba`, `tsb`, `kalman`,
`local_level_kalman`, `auto_kalman`, `auto_local_level_kalman`,
`cartoboost_lag`, and `auto_forecaster`.
Coordinate-specific models such as `KrigingForecaster` and reconciliation
helpers are constructed directly or through their dedicated config sections
rather than default registry entries.

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
| `cartoboost.plotting.plot_route_segments` | Static pickup/dropoff route segment map with optional metric coloring. |
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
`cartoboost[visualization]` when visualization dependencies are not already
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
split gain is not evidence of cold pickup-zone, dropoff-zone, lane, or route
generalization.

## General Utilities

Utilities independent of the regressor and forecasting model APIs:

See [General Utilities](../general_utilities.md) for complete examples.
These helpers are useful for diagnostics and baselines, but quality claims
still need the same split, target transformation, and metric definitions as the
main model comparison.

| Entry point | Purpose |
| --- | --- |
| `cartoboost.naive_forecast(values, horizon)` and related `seasonal_naive_forecast`, `theta_forecast`, `optimized_theta_forecast`, `ets_forecast`, `arima_forecast`, `auto_arima_forecast` | Single-series forecasts for plain numeric sequences. |
| `cartoboost.local_level_kalman_filter(values, ..., horizon=0, interval_z=...)` | Local-level Kalman filtering for numeric sequences. Returns final level and variance, fixed-interval smoothed states, per-step estimates with fitted/residual/gain/likelihood diagnostics, residual summary metrics, optional flat forecast means, and optional forecast distributions with normal bounds. |
| `cartoboost.local_level_kalman_forecast(values, horizon, ...)` | Local-level Kalman forecast utility. |
| `cartoboost.kalman_filter(values, level_process_variance=..., trend_process_variance=..., observation_variance=..., horizon=0, interval_z=...)` | Local-linear Kalman filtering for numeric sequences. Returns final state/covariance, fixed-interval smoothed states, per-step estimates with fitted/residual/covariance/gain/likelihood diagnostics, residual summary metrics, optional forecast means, and optional forecast distributions with normal bounds. |
| `cartoboost.local_linear_trend_kalman_forecast(values, horizon, ...)` | Local-linear trend Kalman forecast utility. |
| `cartoboost.croston_forecast`, `cartoboost.sba_forecast`, `cartoboost.tsb_forecast` | Intermittent-demand utilities for non-negative numeric sequences. |
| `cartoboost.ordinary_kriging_predict(observations, targets, range=..., nugget=..., detailed=False)` | Ordinary kriging for observed `(x, y, value)` triples and target `(x, y)` coordinates. Supports variogram, anisotropy, drift, and neighbor controls; detailed rows include variance and selected neighbor indices. |
| `cartoboost.ordinary_kriging_leave_one_out(observations, ...)` | Leave-one-out kriging diagnostics for observed coordinates. |
| `cartoboost.empirical_variogram(observations, ...)` | Binned empirical semivariogram with lag ranges, mean lag distances, semivariances, and pair counts. |
| `cartoboost.fit_ordinary_kriging_variogram(observations, ...)` | Weighted least-squares variogram fitting over model/range/nugget/sill candidate grids. |
| `cartoboost.ordinary_kriging_leave_one_out_diagnostics(observations, ...)` | Leave-one-out predictions plus residual metrics such as bias, MAE, RMSE, standardized residuals, interval coverage, and average kriging variance. |
| `cartoboost.forecasting.sequence.*` | Sequence reference utilities for known-prefix continuation, reference path inference, leakage-safe OOF row generation and validation, group metrics, and aligned candidate blending. |

## Direct Graph Encoders

Neighborhood-based encoders for direct graph embeddings and optional downstream
feature workflows.

`Node2VecEncoder` is for transductive directed/weighted random-walk embeddings;
`GraphSageEncoder` is for homogeneous graphs; `HeteroGraphSageEncoder` is for
typed relations; `HinSageEncoder` is the typed-schema HinSAGE surface with
relation-aware sampling and link feature construction.

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

The `cartoboost.graph` package contains standalone graph models, link
predictors, and graph-feature helpers. Use standalone classes for direct
modeling; use `GraphFeatureTransformer` only when you want dense and sparse
graph inputs for another estimator.

For taxi source-target modeling, build graph features from train-side
pickup/dropoff relationships when validation lanes or timestamps must remain
unseen. If validation edges leak into topology construction, label the result as
transductive rather than a deployment holdout.

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

Standalone regressors:

| Entry point | Fit signature | Notes |
| --- | --- | --- |
| `NeuralEmbeddingStandaloneRegressor` | `fit(ids, y, dense=None)` | Supervised ID embeddings with optional dense row features. |
| `Node2VecStandaloneRegressor` | `fit(node_count, edges, row_nodes, y, row_targets=None, dense=None, edge_weights=None)` | Graph-only random-walk embeddings plus row-level regression. |
| `GraphSageStandaloneRegressor` | `fit(node_features, edges, row_nodes, y, row_targets=None, dense=None)` | Homogeneous graph regression with node attributes. |
| `HeteroGraphSageStandaloneRegressor` | `fit(node_features, edges, row_nodes, y, row_targets=None, dense=None)` | Typed-edge regression without strict HinSAGE schema metadata. |
| `HinSageStandaloneRegressor` | `fit(node_features, node_types, edges, row_nodes, y, row_targets=None, dense=None)` | Typed-node and typed-relation graph regression. |

All standalone regressors expose `predict`, `score`, `save`, and `load`.

Standalone link predictors:

| Entry point | Fit signature | Notes |
| --- | --- | --- |
| `Node2VecLinkPredictor` | `fit(node_count, edges, edge_weights=None)` | Directed/weighted graph link scoring from random-walk embeddings. |
| `GraphSageLinkPredictor` | `fit(node_features, edges)` | Homogeneous graph link scoring. |
| `HeteroGraphSageLinkPredictor` | `fit(node_features, edges)` | Typed-edge link scoring. |
| `HinSageLinkPredictor` | `fit(node_features, node_types, edges)` | Typed-node and typed-relation link scoring. |

All standalone link predictors expose `predict_scores`, `report`, `save`, and
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

## `cartoboost.FeatureSchema`

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
cartoboost.make_shap_explainer(model, background, **kwargs)
cartoboost.explain_shap(model, X, background=..., **kwargs)
```

These functions are also available as estimator methods. See
[SHAP Support](../shap.md).

## Evaluation Helpers

```python
cartoboost.out_of_time_split(times, validation_fraction=0.2, gap=0)
cartoboost.spatial_blocked_cv(coordinates, n_splits=5)
cartoboost.spatial_buffered_cv(coordinates, n_splits=5, buffer_radius=...)
cartoboost.spatial_grouped_cv(coordinates, groups, n_splits=5, buffer_radius=0.0)
cartoboost.environmental_blocked_cv(environmental_features, n_splits=5)
cartoboost.temporal_blocked_cv(times, n_splits=5, gap=0)
cartoboost.grouped_blocked_cv(groups, n_splits=5)
```

`out_of_time_split` returns one `(train_idx, validation_idx)` pair for a future
holdout window. Use `validation_size=...` for an exact tail count or
`cutoff=...` for rows strictly after a time boundary. Use `gap=...` to remove
recent training rows immediately before the validation window.

Use these helpers to evaluate temporal-spatial generalization: future periods,
withheld locations, or held-out route groups often reveal failure modes that a
random split hides.

`spatial_buffered_cv` removes training rows within `buffer_radius` of each test
block. Use projected coordinates for positive buffers; latitude/longitude
degree buffers raise unless explicitly allowed. `spatial_grouped_cv` keeps
whole groups together and can add the same spatial buffer. `environmental_blocked_cv`
clusters environmental covariates with optional sklearn KMeans, or uses a
deterministic ordered fallback with `use_sklearn=False`.

Use the returned indices for every model in the comparison. The split is part
of the experiment definition and should be stored with benchmark artifacts or
reconstructed from a named command and seed.

## Metric And Diagnostic Helpers

```python
cartoboost.logloss(y_true, y_proba)
cartoboost.roc_auc(y_true, y_score)
cartoboost.pr_auc(y_true, y_score)
cartoboost.brier_score(y_true, y_proba)
cartoboost.ece_calibration_error(y_true, y_proba, n_bins=10)
cartoboost.ndcg_at_k(relevance, scores, groups=None, k=None)
cartoboost.mean_average_precision(relevance, scores, groups=None, k=None)
cartoboost.mean_reciprocal_rank(relevance, scores, groups=None, k=None)
cartoboost.residual_morans_i(coordinates, residuals)
cartoboost.spatial_cv_gap(random_cv_score, spatial_cv_score)
```

Classification metrics are deterministic NumPy implementations for binary or
multiclass probability checks where applicable. Ranking metrics accept either
one global ranking, positive group sizes that sum to the row count, or
contiguous query ids when the values do not form a valid size vector.
`residual_morans_i` uses dense pairwise spatial weights and is intended for
validation samples.

## I/O Helpers

```python
cartoboost.io.read_geojson(path)
```

Reads a GeoJSON file into a Python dictionary.

## Geo Encoding Helpers

```python
cartoboost.build_h3_sparse_sets(
    {"pickup_h3": (pickup_latitude, pickup_longitude)},
    resolution=9,
    parent_resolutions=[5, 7],
)
cartoboost.build_s2_sparse_sets(
    {"pickup_s2": (pickup_latitude, pickup_longitude)},
    level=12,
    parent_levels=[8, 10],
)
```

These helpers return `sparse_sets` dictionaries suitable for
`CartoBoostRegressor.fit(..., sparse_sets=...)`. H3 auto-encoding requires the
optional `h3` package; S2 auto-encoding requires `s2sphere`. Deterministic
normalization, coordinate/level validation, scaffold expansion, and sparse-row
assembly are handled by CartoBoost.
