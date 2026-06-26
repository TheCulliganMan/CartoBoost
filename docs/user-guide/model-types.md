# Choose A Model

Use this page as the user-guide router. CartoBoost has several first-class
model surfaces, and the right entry point depends on the scientific structure
in the data: row-level place/time effects, regular time series, shared panels,
direct graph structure, or learned ID embeddings.

Use the model whose assumptions match the unit being predicted. Then compare it
against simpler baselines under the same split before interpreting a gain.

## Start With The Scientific Unit

| Question | Use | Primary guide |
| --- | --- | --- |
| Does each row describe one taxi trip, route observation, zone-hour aggregate, or residual to regress? | `cartoboost.CartoBoostRegressor` | [Python API Reference](../reference/python-api.md) |
| Is the target a class label such as airport-trip flag, high-delay bucket, or route-risk class? | `cartoboost.CartoBoostClassifier` | [Python API Reference](../reference/python-api.md) |
| Are rows grouped by search, customer, pickup zone, lane, or route request and need within-group ordering? | `cartoboost.CartoBoostRanker` | [Python API Reference](../reference/python-api.md) |
| Are you choosing how place, time, sparse memberships, losses, fuzzy routing, or local residual trends enter that row-level model? | `CartoBoostRegressor` parameters | [Parameters](parameters.md) |
| Is the target one regular pickup-zone or lane series with its own history? | Local forecasters such as `SeasonalNaiveForecaster`, `ThetaForecaster`, `ETSForecaster`, `AutoARIMAForecaster`, or `KalmanForecaster` | [Forecasting Model Guides](forecasting-models/index.md) |
| Are many related pickup zones, dropoff zones, or route panels forecast from shared lag features? | `CartoBoostLagForecaster` | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) |
| Should nearby coordinates borrow signal for a forecast panel? | `KrigingForecaster` | [Kriging](forecasting-models/kriging.md) |
| Should temporal changepoints and seasonality be fused with cutoff-safe spatial kriging? | `SpatialPiecewiseKrigingForecaster` | [Spatial Piecewise Kriging](forecasting-models/spatial-piecewise-kriging.md) |
| Should a neural panel forecaster preserve directional pickup-dropoff lane identity? | `NeuralPanelForecaster` or `LaneNeuralPanelForecaster` | [Neural Panel](forecasting-models/neural-panel.md) |
| Should a deterministic neural forecasting expert learn from regular taxi-demand windows? | `NBeatsForecaster` or `NHiTSForecaster` | [Forecasting Model Guides](forecasting-models/index.md) |
| Do you need a fixed combination of fitted forecasters? | `WeightedEnsembleForecaster` | [Weighted Ensembles](forecasting-models/ensembles.md) |
| Are stable pickup zones, dropoff zones, or pairs themselves the learned artifact? | `NeuralEmbeddingStandaloneRegressor` | [Neural Features](../neural-features.md) |
| Is the relationship network the object being modeled? | Graph standalone regressors or link predictors | [Graph Features](../graph-features.md) |
| Do graph or neural embeddings only need to become columns for another estimator? | `GraphFeatureTransformer`, `NeuralEmbeddingFeatures`, or `NeuralEmbeddingRegressor` | [Graph Features](../graph-features.md), [Neural Features](../neural-features.md) |
| Do you need one-off forecast or spatial utilities? | Functions such as `theta_forecast`, `kalman_filter`, or `ordinary_kriging_predict` | [General Utilities](../general_utilities.md) |

## When CartoBoostRegressor Fits

`CartoBoostRegressor` is the main sklearn-style estimator for row-level taxi
regression. It is a good scientific choice when the target is plausibly shaped
by structured place/time effects rather than only by generic dense covariates.
Examples include fare, duration, demand, or residual models where pickup and
dropoff zones, route memberships, hour-of-day, local neighborhoods, or fuzzy
service boundaries should be part of the model rather than hidden in many
preprocessing columns.

Prefer it for experiments where you want to ask questions such as:

- Do pickup/dropoff effects persist after controlling for trip distance, hour,
  and day features?
- Are sparse zones, routes, H3/S2 cells, or service areas informative even when
  many memberships are rare?
- Does a smooth transition near a learned spatial boundary reduce localized
  residual artifacts?
- Does an outlier-resistant or quantile objective match the scientific target
  more closely than mean regression?
- Can the fitted artifact preserve the schema, splitters, loss, fuzzy settings,
  sparse-set requirements, and additive values needed for later interpretation?

Do not treat this as a broad claim about CartoBoost versus LightGBM, XGBoost,
or a simpler baseline. Use those models as serious comparisons under the same
train/test split and feature set. Select CartoBoost only when the structured
controls satisfy the specific holdout or diagnostic that matters for the study.

## Tabular And Spatial Regression

Start with dense numeric columns for the measured quantities: trip distance,
projected pickup/dropoff coordinates, pickup hour, day of week, route-level
aggregates, fare history, or duration history. Add sparse-set features when a
row belongs to pickup zones, dropoff zones, H3/S2 cells, service areas, route
memberships, or overlapping operational regions.

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Choose controls from the structure you want to test:

| Scientific need | Parameter family |
| --- | --- |
| Dense tabular baseline | `splitters=None`, `["auto"]`, `["axis"]`, or `["axis_histogram:<bins>"]` |
| Spatial boundaries in coordinates | `["axis", "diagonal_2d", "gaussian_2d"]` |
| Wraparound time effects | `["axis", "periodic:24"]` or another `periodic:<period>` |
| Sparse pickup/dropoff zones, routes, cells, or areas | `["axis", "sparse_set"]` plus `sparse_sets=` |
| Native categorical pickup/dropoff labels or service tiers | `FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL` in `feature_schema=` |
| Smooth changes near boundaries | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` |
| Outlier-resistant regression | `loss="mae"`, `loss="huber"`, or `loss="log_l2"` |
| Conditional intervals or asymmetric service targets | `loss="quantile"`, `quantile_alpha=...` |
| Local residual trend inside learned regions | `leaf_predictor="linear"`, `linear_leaf_features=[...]` |
| Domain monotonicity | `monotonic_constraints=[...]` |

See [Python API Reference](../reference/python-api.md), [Parameters](parameters.md),
[Feature Schema](../feature_schema.md), [Sparse Features](../sparse_features.md),
and [Temporal-Spatial Modeling](../spatial_modeling.md).

## Categorical Features

Regressor, classifier, and ranker inputs may include pandas categorical,
string, or object columns, or columns explicitly marked with
`FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL`. CartoBoost records a stable
category mapping in saved artifacts. Low-cardinality nominal columns become
numeric indicator columns, including deterministic subset partition indicators
where feasible; ordinal columns use a deterministic ordered mapping, and
high-cardinality nominal columns use smoothed target statistics with an explicit
unknown-category value.

```python
from cartoboost import CartoBoostRegressor, FeatureKind

schema = {"dense": [{"name": "PULocationID", "kind": FeatureKind.CATEGORICAL}]}
model = CartoBoostRegressor(splitters=["axis"])
model.fit(zone_features, fare, feature_schema=schema)
pred = model.predict(zone_features_holdout)
```

Keep categorical preprocessing inside the fitted CartoBoost artifact when
comparing against baselines: give the baseline an equivalent train-only
encoding and evaluate on the same split.

## Tabular And Spatial Classification

Use `CartoBoostClassifier` when each row has a discrete taxi-domain label and
the decision boundary may depend on pickup/dropoff coordinates, hour, route
memberships, or sparse zone signals. The classifier fits binary logistic loss
for two classes and multiclass logistic loss for three or more classes, with
sklearn-style label handling, `predict`, `predict_proba`, `decision_function`,
`class_weight`, and save/load label metadata.

```python
from cartoboost import CartoBoostClassifier

clf = CartoBoostClassifier(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
    class_weight="balanced",
)
clf.fit(X_train, airport_trip_flag)
prob_airport = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
```

Report classifier quality with logloss plus threshold-free metrics such as
ROC-AUC or PR-AUC when the positive class is rare. Compare against dummy and
standard tabular baselines on the same train/test split before interpreting a
CartoBoost gain.

## Grouped Ranking

Use `CartoBoostRanker` when rows are only comparable within a query group:
candidate dropoff zones for one pickup, route alternatives for one shipment,
or ranked taxi-zone actions for one planning context. The ranker uses pairwise
logistic or LambdaRank objectives and reports NDCG, MAP, and MRR from grouped
predictions.

```python
from cartoboost import CartoBoostRanker

ranker = CartoBoostRanker(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    splitters=["axis", "diagonal_2d", "gaussian_2d"],
    objective="lambdarank",
)
ranker.fit(X_train, relevance_train, groups=query_sizes_train)
scores = ranker.predict(X_test)
metrics = ranker.score_groups(X_test, relevance_test, groups=query_sizes_test)
```

Rows for each query must be contiguous. Pass `groups` as group sizes or
contiguous query ids, or set `group_col` when the query id is a column in `X`.

## Spatial Validation And Diagnostics

Use spatial validation when the claim is about generalizing to withheld zones,
route corridors, or environmental regimes rather than interpolating among
nearby training rows. `spatial_buffered_cv` holds out spatial blocks and removes
nearby training rows inside a buffer. `spatial_grouped_cv` combines whole-group
holdout with the same optional buffer, which is useful for pickup zones,
customer groups, or lane ids. `environmental_blocked_cv` clusters covariates
such as weather, demand regimes, or operational conditions.

```python
from cartoboost import residual_morans_i, spatial_buffered_cv, spatial_cv_gap

folds = list(
    spatial_buffered_cv(
        projected_pickup_xy,
        n_splits=5,
        buffer_radius=500.0,
        coordinate_units="meters",
    )
)
gap = spatial_cv_gap(random_cv_rmse, buffered_cv_rmse)
residual_i = residual_morans_i(projected_pickup_xy, y_test - pred_test)
```

Positive spatial buffers should use projected linear units. Latitude/longitude
degree buffers fail clearly unless the caller explicitly allows degree
distances.

## Browser Splitter Visualizer

Use the [Modeling Lab](../../modeling-lab) when you want to inspect a fitted
CartoBoost model in the browser before moving to a Python or CLI workflow. The
lab runs locally, loads bundled single-lane or varied-route yellow taxi samples,
and renders the fitted tree structure without sending data to a server.

The visualizer summarizes the boosted trees, split kinds, top splitter rules,
depth profile, and largest holdout residuals after fitting. This is the best
place to confirm whether axis, diagonal spatial, Gaussian spatial, periodic,
sparse-set, or fuzzy splitters are actually used on taxi pickup, dropoff, route,
fare, distance, duration, and demand features.

## Forecasting

Forecasting has two documentation layers:

| Layer | Covers | Start here |
| --- | --- | --- |
| Forecasting wrapper | `ForecastFrame`, dataframe conversion, rolling-origin backtesting, forecast metrics, artifacts, CLI workflows, and leakage checks | [Forecasting](../forecasting.md) |
| Forecasting model guides | Model-specific examples, interactive lab links, and tuning notes for forecasting classes | [Forecasting Model Guides](forecasting-models/index.md) |

Use `ForecastFrame.from_pandas` for production taxi demand or fare-duration
workflows because it validates timestamps, frequency, duplicate rows, target
values, panel ids, and covariate roles:

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
    known_future_covariates=["hour", "day_of_week"],
)
```

Choose the model guide by series structure:

| Series structure | Model guide |
| --- | --- |
| Last value or last season is the benchmark | [Naive And Seasonal Naive](forecasting-models/naive-seasonal.md) |
| Lightweight trend extrapolation | [Theta](forecasting-models/theta.md) |
| Additive level, trend, or seasonality | [ETS](forecasting-models/ets.md) |
| Autocorrelation and differencing | [ARIMA And AutoARIMA](forecasting-models/arima.md) |
| Noisy local level and trend | [Kalman](forecasting-models/kalman.md) |
| Sparse non-negative pickup demand with many true zero periods | `CrostonForecaster`, `SbaForecaster`, or `TsbForecaster` |
| Interpretable trend, changepoints, seasonality, events, and regressors | [Piecewise Linear Seasonal](forecasting-models/piecewise-linear-seasonal) |
| Prophet-compatible forecast plotting for Prophet-shaped outputs | [Plotting](../plotting.md) |
| Coordinate-aware panel interpolation | [Kriging](forecasting-models/kriging.md) |
| Piecewise temporal structure plus spatial residual or regressor kriging | [Spatial Piecewise Kriging](forecasting-models/spatial-piecewise-kriging.md) |
| Shared supervised lag model across many series | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) |
| Directional neural panel forecasting | [Neural Panel](forecasting-models/neural-panel.md) |
| Reusable statistical expert-bank selection | `AutoStatsBank` |
| Guarded default selector over forecast candidates | [AutoForecaster](forecasting-models/auto-forecaster.md) |
| Windowed neural extrapolation | `NBeatsForecaster` or `NHiTSForecaster` |
| Fixed-weight combinations of fitted forecasters | [Weighted Ensembles](forecasting-models/ensembles.md) |

## Graph And Neural Models

Graph and neural standalone models are direct APIs, not just feature builders
for `CartoBoostRegressor`.

Use `NeuralEmbeddingStandaloneRegressor` when the learned ID embedding is the
artifact to train, score, save, and serve. This works best when train and
prediction populations share stable IDs such as pickup zones, dropoff zones,
pickup-dropoff pairs, zone-hour buckets, or trip clusters. Under cold-zone or
cold-ID holdouts, report fallback behavior explicitly because unseen IDs cannot
recover learned ID-specific effects.

Use graph standalone regressors when relationships matter:

| Model | Use when |
| --- | --- |
| `Node2VecStandaloneRegressor` | Directed or weighted topology is useful and node attributes are not required. |
| `GraphSageStandaloneRegressor` | A homogeneous graph has node attributes such as airport flag, borough, or recent volume. |
| `HeteroGraphSageStandaloneRegressor` | Edges have relation IDs, but strict node-type schema validation is not required. |
| `HinSageStandaloneRegressor` | Nodes and relations are typed and source-target type constraints matter. |

Use graph or neural feature generators only when embeddings should become dense
columns for another model.

See [Graph Features](../graph-features.md) and [Neural Features](../neural-features.md).

## Validation Defaults

Whichever model family you choose, validate it against a serious baseline under
the same split:

| Model family | Minimum comparison |
| --- | --- |
| Tabular regression | Mean baseline plus LightGBM or XGBoost on the same features and split. |
| Forecasting | Naive or seasonal naive plus rolling-origin backtests. |
| Neural embeddings | A non-neural `CartoBoostRegressor` under random, temporal, and cold-ID splits where relevant. |
| Graph models | A tabular route model and a graph-free ID or zone baseline. |

For NYC taxi work, report the target, split, row count, features, RMSE, MAE,
R2 when applicable, train time, prediction time, and exact command or notebook
entry point used to produce the numbers.

## Recommended Reading Order

1. Read [Getting Started](../getting-started.md) for installation, the first
   model fit, and local validation commands.
2. Use this chooser to pick the model family.
3. For row-level regression, read the [Python API Reference](../reference/python-api.md), then
   [Parameters](parameters.md), then [Temporal-Spatial Modeling](../spatial_modeling.md)
   and the relevant feature pages.
4. For time-series work, read the [Forecasting](../forecasting.md) page
   when you need `ForecastFrame`, backtesting, forecast artifacts, or the CLI.
   Read [Forecasting Model Guides](forecasting-models/index.md) when you need examples for
   a specific model class.
5. For graph or neural work, start with the standalone model sections before
   using feature-generation helpers.
