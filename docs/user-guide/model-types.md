# Choose A CartoBoost Model

Use this page as the user-guide router. CartoBoost has several first-class
model surfaces, and the right entry point depends on the scientific structure
in the data: row-level place/time effects, regular time series, shared panels,
direct graph structure, or learned ID embeddings.

Use the model whose assumptions match the unit being predicted. Then compare it
against simpler baselines under the same split before interpreting a gain.

Use `cartoboost.preview.models.ModelRegistry` to inspect preview model metadata.
Automatic geo model selection is intentionally not shipped in v0.3; choose a
registered estimator explicitly and pair it with a native validation manifest.

## Start With The Scientific Unit

| Question | Use | Primary guide |
| --- | --- | --- |
| Does each row describe one observation with a numeric target or residual to regress? | `cartoboost.CartoBoostRegressor` | [CartoBoost Regressor](boosting-models/cartoboost-regressor.md) |
| Is the target a class label? | `cartoboost.CartoBoostClassifier` | [CartoBoost Classifier](boosting-models/cartoboost-classifier.md) |
| Are rows grouped and need within-group ordering? | `cartoboost.CartoBoostRanker` | [CartoBoost Ranker](boosting-models/cartoboost-ranker.md) |
| Are you choosing how place, time, sparse memberships, losses, fuzzy routing, or local residual trends enter that row-level model? | `CartoBoostRegressor` parameters | [Parameters](parameters.md) |
| Do you need an interpretable sparse-neighbor regression baseline for areal units, trade areas, zones, or route cells? | `SpatialLagRegressor`, `SpatialErrorRegressor`, `SpatialDurbinRegressor` | [Spatial Econometrics Models](spatial-econometrics.md) |
| Is the target one regular time series with its own history? | Local forecasters such as `SeasonalNaiveForecaster`, `ThetaForecaster`, `ETSForecaster`, `AutoARIMAForecaster`, or `KalmanForecaster` | [CartoBoost Forecasting Model Guides](forecasting-models/index.md) |
| Are many related series forecast from shared lag features? | `CartoBoostLagForecaster` | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) |
| Are road, lane, sensor, or zone-flow series connected by a directed graph? | `DCRNNForecaster` with `GraphTemporalFrame` | [Graph Spatiotemporal Forecasting](forecasting-models/graph-spatiotemporal.md) |
| Do directional markets need learned relationships, hierarchy fallback, shift explanations, and inspectable kernels? | `MarketStructureForecaster` with `MarketPanelFrame` | [Market Structure Forecasting](forecasting-models/graph-spatiotemporal.md) |
| Do regular forecast windows need a compact neural baseline? | `NBeatsForecaster` or `NHiTSForecaster` | [N-BEATS And N-HiTS](forecasting-models/beats-hits.md) |
| Should nearby coordinates borrow signal for a forecast panel? | `KrigingForecaster` | [Kriging](forecasting-models/kriging.md) |
| Do point observations need scalable GP interpolation with uncertainty? | `NearestNeighborGPRegressor` or `SpatialGaussianProcessRegressor` | [Scalable GP Geostatistics](geostatistics-models.md) |
| Should a base tabular model get a probabilistic spatial residual correction? | `ResidualNNGPRegressor` | [Scalable GP Geostatistics](geostatistics-models.md) |
| Should temporal changepoints and seasonality be fused with cutoff-safe spatial kriging? | `SpatialPiecewiseKrigingForecaster` | [Spatial Piecewise Kriging](forecasting-models/spatial-piecewise-kriging.md) |
| Are you estimating intervention lift or designing a geographic experiment rather than forecasting? | `SyntheticDIDEstimator`, `GeoLiftEstimator`, `GeoExperimentDesigner`, or `SpatialPlaceboTester` | [Geo-Causal Experiment Models](geo-causal-models.md) |
| Should a neural panel forecaster preserve directional entity identity? | `NeuralPanelForecaster` or `LaneNeuralPanelForecaster` | [Neural Panel](forecasting-models/neural-panel.md) |
| Do you need generic ordered-pair, response-curve, event, residual, graph-sequence, or constrained candidate models? | `cartoboost.preview.deep.*` | [Generic Deep Models](deep-models.md) |
| Do you need a fixed combination of fitted forecasters? | `WeightedEnsembleForecaster` | [Weighted Ensembles](forecasting-models/ensembles.md) |
| Are stable ids themselves the learned artifact? | `NeuralEmbeddingStandaloneRegressor` | [CartoBoost Neural Embedding Regressor](neural-models/cartoboost-neural-embedding.md) |
| Is the relationship network the object being modeled? | Graph models | [CartoBoost Graph Model Guides](graph-models/index.md) |
| Do graph or neural embeddings only need to become columns for another estimator? | `GraphFeatureTransformer`, `NeuralEmbeddingFeatures`, or `NeuralEmbeddingRegressor` | [Graph Features](../graph-features.md), [Neural Features](../neural-features.md) |
| Do you need one-off forecast or spatial utilities? | Functions such as `theta_forecast`, `kalman_filter`, or `ordinary_kriging_predict` | [General Utilities](../general_utilities.md) |

## Unified Model Registry

`cartoboost.preview.models.ModelRegistry.defaults()` describes preview model
surfaces under the namespaces `cartoboost.forecasting` and preview modules
such as `cartoboost.preview.geo`, `cartoboost.preview.graph`,
`cartoboost.preview.causal`, and `cartoboost.preview.prob`. Each spec carries
typed metadata: model name, namespace,
task types, capabilities, stability, artifact format, and optional dependency
notes.

```python
from cartoboost.preview.models import ModelRegistry

registry = ModelRegistry.defaults()
registry.names(namespace="geo")
```

The executable contract for this registry example is checked by
`scripts/check_docs_examples.py` in CI.

Select the estimator that matches the study design, then use
`cartoboost.validation` native split-manifest constructors to make the
leakage policy explicit. The v0.3 distribution does not include an automatic
geo selector or a model stack; any future return requires native selection
behavior and real-family evidence.

Experimental research adapters live under `cartoboost.preview.experimental`. They are
not registered as stable models and require an explicit backend before fitting.

## When CartoBoostRegressor Fits

`CartoBoostRegressor` is the main sklearn-style estimator for row-level
regression. It is a good scientific choice when the target is plausibly shaped
by structured place/time effects rather than only by generic dense covariates.
Examples include duration, fare, demand, or residual models where route
memberships, hour-of-day, local neighborhoods, or fuzzy service boundaries
should be part of the model rather than hidden in preprocessing columns.

Prefer it for experiments where you want to ask questions such as:

- Do structured place/time effects persist after controlling for distance,
  hour, and day features?
- Are sparse zones, routes, H3/S2 cells, or service areas informative even
  when many memberships are rare?
- Does a smooth transition near a learned spatial boundary reduce localized
  residual artifacts?
- Does an outlier-resistant or quantile objective match the scientific target
  more closely than mean regression?
- Can the fitted artifact preserve the schema, split policy, loss, fuzzy settings,
  sparse-set requirements, and additive values needed for later interpretation?

Do not treat this as a broad claim about CartoBoost versus LightGBM, XGBoost,
or a simpler baseline. Use those models as serious comparisons under the same
train/test split and feature set. Select CartoBoost only when the structured
controls satisfy the specific holdout or diagnostic that matters for the study.

## Tabular And Spatial Regression

Start with dense numeric columns for the measured quantities: distance,
projected coordinates, hour, day of week, route-level aggregates, fare
history, or duration history. Add sparse-set features when a row belongs to
zones, H3/S2 cells, service areas, route memberships, or overlapping
operational regions.

```python
from cartoboost import CartoBoostRegressor
from cartoboost.config import SplitPolicy

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    split_policy=SplitPolicy.STRUCTURED,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Choose controls from the structure you want to test:

| Scientific need | Parameter family |
| --- | --- |
| Dense tabular baseline | `SplitPolicy.AXIS_ONLY` |
| Spatial boundaries in coordinates | `SplitPolicy.STRUCTURED` plus spatial-pair schema entries |
| Wraparound time effects | `SplitPolicy.STRUCTURED` plus periodic schema entries |
| Sparse zones, routes, cells, or areas | `SplitPolicy.STRUCTURED` plus `sparse_sets=` |
| Native categorical labels or service tiers | `FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL` in `feature_schema=` |
| Smooth changes near boundaries | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` |
| Outlier-resistant regression | `loss="mae"`, `loss="huber"`, or `loss="log_l2"` |
| Conditional intervals or asymmetric service targets | `loss="quantile"`, `quantile_alpha=...` |
| Local residual trend inside learned regions | `leaf_predictor="linear"`, `linear_leaf_features=[...]` |
| Domain monotonicity | `monotonic_constraints=[...]` |

See [CartoBoost Regressor](boosting-models/cartoboost-regressor.md), [Python API Reference](../reference/python-api.md), [Parameters](parameters.md),
[Feature Schema](../feature_schema.md), [Sparse Features](../sparse_features.md),
and [Temporal-Spatial Modeling](../spatial_modeling.md).

## Categorical Features

CartoBoost regressor, classifier, and ranker inputs may include pandas categorical,
string, or object columns, or columns explicitly marked with
`FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL`. CartoBoost records a stable
category mapping in saved artifacts. Low-cardinality nominal columns become
numeric indicator columns, including deterministic subset partition indicators
where feasible; ordinal columns use a deterministic ordered mapping, and
high-cardinality nominal columns use smoothed target statistics with an explicit
unknown-category value.

```python
from cartoboost import CartoBoostRegressor
from cartoboost.preview import FeatureKind

schema = {"dense": [{"name": "location_id", "kind": FeatureKind.CATEGORICAL}]}
model = CartoBoostRegressor(split_policy="axis_only")
model.fit(zone_features, fare, feature_schema=schema)
pred = model.predict(zone_features_holdout)
```

Keep categorical preprocessing inside the fitted CartoBoost artifact when
comparing against baselines: give the baseline an equivalent train-only
encoding and evaluate on the same split.

## Tabular And Spatial Classification

Use `CartoBoostClassifier` when each row has a discrete label and the decision
boundary may depend on time, location, route memberships, or sparse signals.
The CartoBoost classifier fits binary logistic loss for two classes and
multiclass logistic loss for three or more classes, with sklearn-style label
handling, `predict`, `predict_proba`, `decision_function`, `class_weight`, and
save/load label metadata.

```python
from cartoboost import CartoBoostClassifier

clf = CartoBoostClassifier(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    min_samples_leaf=20,
    split_policy="structured",
    class_weight="balanced",
)
clf.fit(X_train, airport_trip_flag)
prob_airport = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
```

Report CartoBoost classifier quality with logloss plus threshold-free metrics
such as ROC-AUC or PR-AUC when the positive class is rare. Compare against
dummy and standard tabular baselines on the same train/test split before
interpreting a CartoBoost gain.

## Grouped Ranking

Use `CartoBoostRanker` when rows are only comparable within a query group. The
CartoBoost ranker uses pairwise logistic or LambdaRank objectives and reports
NDCG, MAP, and MRR from grouped predictions.

```python
from cartoboost import CartoBoostRanker

ranker = CartoBoostRanker(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    split_policy="structured",
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
nearby training rows. `spatial_buffered_cv` holds out spatial blocks and
removes nearby training rows inside a buffer. `spatial_grouped_cv` combines
whole-group holdout with the same optional buffer, which is useful for grouped
entities. `environmental_blocked_cv` clusters covariates such as weather,
demand regimes, or operational conditions.

Use the stable native manifest constructors in `cartoboost.validation` for
spatial and temporal split definitions. Preview diagnostics such as residual
Moran's I and the random-to-spatial score gap live under
`cartoboost.preview.metrics`; keep their output in the benchmark artifact
alongside the manifest hash.

Positive spatial buffers should use projected linear units. Latitude/longitude
degree buffers fail clearly unless the caller explicitly allows degree
distances.

## Browser Splitter Visualizer

Use the [Modeling Lab](../../modeling-lab) when you want to inspect a fitted
CartoBoost model in the browser before moving to a Python or CLI workflow. The
lab runs locally, loads bundled samples, and renders the fitted tree structure
without sending data to a server.

The visualizer summarizes the boosted trees, split kinds, top splitter rules,
depth profile, and largest holdout residuals after fitting. This is the best
place to confirm whether axis, diagonal spatial, Gaussian spatial, periodic,
sparse-set or fuzzy structured policy is actually used on your features.

## Forecasting

Forecasting has two documentation layers:

| Layer | Covers | Start here |
| --- | --- | --- |
| Forecasting wrapper | `ForecastFrame`, dataframe conversion, rolling-origin backtesting, forecast metrics, artifacts, CLI workflows, and leakage checks | [Forecasting](../forecasting.md) |
| CartoBoost forecasting model guides | Model-specific examples, interactive lab links, and tuning notes for forecasting classes | [CartoBoost Forecasting Model Guides](forecasting-models/index.md) |

Use `ForecastFrame.from_pandas` for production demand or time-series workflows
because it validates timestamps, frequency, duplicate rows, target values,
panel ids, and covariate roles:

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="entity_id",
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
| Sparse non-negative demand with many true zero periods | `CrostonForecaster`, `SbaForecaster`, or `TsbForecaster` |
| Interpretable trend, changepoints, seasonality, events, and regressors | [Piecewise Linear Seasonal](forecasting-models/piecewise-linear-seasonal) |
| Prophet-compatible forecast plotting for Prophet-shaped outputs | [Plotting](../plotting.md) |
| Coordinate-aware panel interpolation | [Kriging](forecasting-models/kriging.md) |
| Piecewise temporal structure plus spatial residual or regressor kriging | [Spatial Piecewise Kriging](forecasting-models/spatial-piecewise-kriging.md) |
| Shared supervised lag model across many series | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) |
| Directed graph sequence forecasting | [Graph Spatiotemporal Forecasting](forecasting-models/graph-spatiotemporal.md) |
| Directional neural panel forecasting | [Neural Panel](forecasting-models/neural-panel.md) |
| Reusable statistical expert-bank selection | `AutoStatsBank` |
| Guarded default selector over forecast candidates | [AutoForecaster](forecasting-models/auto-forecaster.md) |
| Fixed-weight combinations of fitted forecasters | [Weighted Ensembles](forecasting-models/ensembles.md) |

## Graph And Neural Models

Graph and neural standalone models are direct APIs, not just feature builders
for `CartoBoostRegressor`.

Use `NeuralEmbeddingStandaloneRegressor` when the learned ID embedding is the
artifact to train, score, save, and serve. This works best when train and
prediction populations share stable IDs such as zone IDs, route pairs,
zone-hour buckets, or trip clusters. Under cold-ID or cold-route holdouts,
report fallback behavior explicitly because unseen IDs cannot recover learned
ID-specific effects.

Use graph standalone regressors when relationships matter:

| Model | Use when |
| --- | --- |
| `Node2VecStandaloneRegressor` | Directed or weighted topology is useful and node attributes are not required. |
| `GraphSageStandaloneRegressor` | A homogeneous graph has node attributes such as airport flag, borough, or recent volume. |
| `HeteroGraphSageStandaloneRegressor` | Edges have relation IDs, but strict node-type schema validation is not required. |
| `HinSageStandaloneRegressor` | Nodes and relations are typed and source-target type constraints matter. |

Use graph or neural feature generators only when embeddings should become dense
columns for another model.

See [CartoBoost Graph Model Guides](graph-models/index.md), [Graph Features](../graph-features.md),
and [CartoBoost Neural Model Guides](neural-models/index.md).

## Validation Defaults

Whichever model family you choose, validate it against a serious baseline under
the same split:

| Model family | Minimum comparison |
| --- | --- |
| Tabular regression | Mean baseline plus LightGBM or XGBoost on the same features and split. |
| Forecasting | Naive or seasonal naive plus rolling-origin backtests. |
| Neural embeddings | A non-neural `CartoBoostRegressor` under random, temporal, and cold-ID splits where relevant. |
| Graph models | A tabular route model and a graph-free ID or zone baseline. |

For structured data work, report the target, split, row count, features, RMSE,
MAE, R2 when applicable, train time, prediction time, and exact command or
notebook entry point used to produce the numbers.

## Recommended Reading Order

1. Read [Getting Started](../getting-started.md) for installation, the first
   model fit, and local validation commands.
2. Use this chooser to pick the model family.
3. For row-level boosting, read the [CartoBoost Boosting Model Guides](boosting-models/index.md), then
   [Parameters](parameters.md), then [Temporal-Spatial Modeling](../spatial_modeling.md)
   and the relevant feature pages.
4. For time-series work, read the [Forecasting](../forecasting.md) page
   when you need `ForecastFrame`, backtesting, forecast artifacts, or the CLI.
   Read [CartoBoost Forecasting Model Guides](forecasting-models/index.md) when you need examples for
   a specific model class.
5. For graph work, start with the [CartoBoost Graph Model Guides](graph-models/index.md). For
   neural work, start with the [CartoBoost Neural Model Guides](neural-models/index.md) before
   using feature-generation helpers.
