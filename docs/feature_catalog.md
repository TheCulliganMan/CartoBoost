# Feature Catalog

This page summarizes CartoBoost feature surfaces and the modeling reasons to use
them. Detailed API contracts live in the linked topic pages and in the Python
API reference.

For taxi-trip science, choose features by the effect you need to measure:

- dense numeric features for scalar facts such as distance, fare components,
  coordinates, and recent counts;
- periodic features for cyclic time such as hour of day or day of week;
- sparse sets for rows that belong to several zones, cells, hierarchies, or
  route memberships;
- graph features when relationships among zones, routes, and time buckets are
  the signal;
- neural embeddings when stable high-cardinality IDs carry repeated residual
  behavior;
- SHAP explanations when you need an additive audit of how fitted features
  contributed to predictions.

## Regression Estimator

`cartoboost.CartoBoostRegressor` is the main sklearn-style estimator for
tabular, temporal, spatial, and sparse-set regression.

| Feature | Public surface | Modeling use |
| --- | --- | --- |
| L2 regression | `loss="l2"` or `"squared_error"` | Mean-oriented fare, duration, or demand targets. |
| L1 regression | `loss="l1"`, `"mae"`, or `"absolute_error"` | Robust median-like fits with constant leaves. |
| Huber regression | `loss="huber"`, `huber_delta=...` | Robust squared-error compromise with constant leaves. |
| Log-L2 regression | `loss="log_l2"`, `log_offset=1.0` | Positive skewed taxi targets such as fare or duration. |
| Quantile regression | `loss="quantile"` or `"pinball"`, `quantile_alpha=...`; Rust `QuantileRegressorSet` for p10-p90 bundles | Conditional quantiles for delay or fare-risk analysis. |
| Constant leaves | `leaf_predictor="constant"` | Default tree leaf behavior. |
| Linear leaves | `leaf_predictor="linear"`, `linear_leaf_features=[...]` | Local linear residual trends inside tree regions. |
| Sample weights | `fit(..., sample_weight=...)` | Weighted studies, rebalancing, or survey-style emphasis. |
| Monotonic and interaction constraints | `monotonic_constraints=[-1, 0, 1, ...]`; Rust `BoosterConfig.interaction_constraints` | Enforce known directional effects and allowed branch-level feature families in supported fits. |
| Additive values | `predict_additive_values(...)` | Per-tree additive contributions whose row sum matches prediction. |
| sklearn compatibility | `get_params`, `set_params`, `clone`, `Pipeline`, `GridSearchCV` | Standard estimator workflows. |
| Artifacts | `save`, `load`, `save_weights`, `load_weights` | Versioned model and weights artifacts. |
| ONNX export subset | `save_weights(path, format="onnx")` | Requires `cartoboost[onnx]`; dense axis-tree subset only. |

See [Python API Reference](reference/python-api.md),
[Parameters](user-guide/parameters.md), [Objectives](objectives.md),
[Constraints](constraints.md), and [Model Artifacts](model_artifact.md).

## Classification And Ranking Estimators

`cartoboost.CartoBoostClassifier` and `cartoboost.CartoBoostRanker` extend the
same tree modeling structure to discrete labels and grouped relevance labels.

| Feature | Public surface | Modeling use |
| --- | --- | --- |
| Binary classification | `CartoBoostClassifier(objective="binary_logloss")` | Airport-trip flags, high-delay labels, or route-risk labels. |
| Multiclass classification | `CartoBoostClassifier(objective="multiclass_logloss")` | Multi-bucket demand, delay, or service class labels. |
| Class weights | `class_weight="balanced"` or label-weight dict | Reweight rare classes in native gradient updates. |
| Probability outputs | `predict_proba`, `decision_function` | Calibration, threshold-free metrics, and ranking by class risk. |
| Grouped ranking | `CartoBoostRanker(objective="lambdarank")` | Candidate dropoff zones, route alternatives, or service actions within a query. |
| Pairwise ranking | `objective="pairwise_logit"` | Pairwise relevance optimization without NDCG weighting. |
| Ranking metrics | `score_groups`, `ndcg_at_k`, `mean_average_precision`, `mean_reciprocal_rank` | Query-group evaluation with NDCG, MAP, and MRR. |

See [Classification Quickstart](user-guide/classification-quickstart.md),
[Ranking Quickstart](user-guide/ranking-quickstart.md), and
[Python API Reference](reference/python-api.md).

## Native Categorical Features

Regressor, classifier, and ranker wrappers can encode categorical columns while
preserving the mapping in saved artifacts.

| Feature | Public surface | Modeling use |
| --- | --- | --- |
| Nominal categorical columns | pandas category/string/object or `FeatureKind.CATEGORICAL` | Pickup/dropoff zones, boroughs, service tiers, or route labels. |
| Ordinal categorical columns | `FeatureKind.ORDINAL` | Ordered service classes or ranked operational bins. |
| Low-cardinality encoding | Stable one-hot or subset partition indicators | Transparent treatment of small category sets. |
| High-cardinality encoding | Smoothed target-statistic column | Compact repeated-ID signal with an unknown-category value. |
| Artifact persistence | `save` / `load` wrapper artifacts | Prediction-time category mappings remain stable after loading. |

See [Categorical Features](user-guide/categorical-features.md) and
[Feature Schema](feature_schema.md).

## Splitters And Feature Semantics

Splitters tell the model what geometry or data shape is meaningful.

| Splitter | Public names | Modeling use |
| --- | --- | --- |
| Automatic | `auto` or `splitters=None` | Conservative default search. |
| Axis | `axis` | Standard one-feature thresholds. |
| Histogram axis | `axis_histogram`, `axis_hist`, `histogram`, `axis_histogram:<bins>` | Faster dense numeric threshold search. |
| Diagonal 2D | `diagonal_2d`, `diagonal2d` | Oblique route or coordinate boundaries. |
| Gaussian/radial 2D | `gaussian_2d`, `gaussian2d`, `radial` | Local hotspots, airports, depots, corridors, or neighborhood effects. |
| Periodic | `periodic_time`, `periodic_24`, `periodic:<period>` | Wraparound hour, weekday, or seasonal phase. |
| Sparse set | `sparse_set`, `sparse` | List-valued zone, route, grid, H3, or S2 memberships. |
| Fuzzy routing | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` | Fractional routing near split boundaries. |

`FeatureSchema` records dense numeric, categorical, ordinal, periodic,
sparse-set, H3 sparse-set, and S2 sparse-set roles so saved artifacts and
prediction inputs can be validated.

See [Feature Schema](feature_schema.md), [Sparse Features](sparse_features.md),
and [Spatial Modeling](spatial_modeling.md).

## Data Inputs And Optional Extras

Optional integrations stay optional. Helpers that require an optional package
raise a clear install error when the extra is missing.

| Capability | Public surface | Extra |
| --- | --- | --- |
| NumPy-style dense arrays | `fit(X, y)`, `predict(X)` | Core package. |
| pandas/dataframe-style inputs | Dataframe columns in estimator and forecasting helpers | Core package. |
| DuckDB relation inputs | Dense relation/query-result support | `cartoboost[duckdb]`. |
| Polars inputs | Dataframe support where documented | `cartoboost[polars]`. |
| H3 encoding | `latlng_to_h3_id`, `encode_h3_cells`, `build_h3_sparse_sets`, `h3_parent_id`, `normalize_h3_id` | `cartoboost[h3]`; validation, ID normalization, scaffold expansion, and row assembly. |
| S2 encoding | `latlng_to_s2_id`, `encode_s2_cells`, `build_s2_sparse_sets`, `s2_parent_id`, `normalize_s2_id` | `cartoboost[s2]`; validation, ID normalization, and row assembly. |
| Geographic sparse helpers | `build_geo_sparse_sets`, `build_zip_sparse_sets`, `coerce_geo_to_feature_id`, `coerce_zip_to_feature_id` | Core package. |
| SHAP explanations | `make_shap_explainer`, `explain_shap` | `cartoboost[explain]`. |
| Optuna workflows | Tuning examples/workflows | `cartoboost[optuna]`. |

See [Installation](installation.md), [Sparse Features](sparse_features.md), and
[SHAP Support](shap.md).

## Neural Embedding Models And Features

Neural embeddings are for stable, high-cardinality IDs that carry residual
signal, such as pickup zones, dropoff zones, OD pairs, or zone-hour buckets.
They are not the same claim as cold-zone generalization; report the split.

| Feature | Public surface | Notes |
| --- | --- | --- |
| Standalone supervised ID model | `NeuralEmbeddingStandaloneRegressor` | Direct regressor over learned ID embeddings plus optional dense features. |
| Hybrid neural features | `NeuralEmbeddingRegressor` | Learns ID vectors and appends them to a tabular model. |
| Neural feature blocks | `NeuralEmbeddingFeatures` | Deterministic feature-generation helper. |
| Fallback behavior | `ArtifactFallback` and fallback arguments | Handles unseen or rare IDs through configured fallback vectors/chains. |
| Benchmark helper | `benchmark_neural_vs_cartoboost` | Quick held-out comparison between structured and neural-enhanced models. |
| Artifacts | `save`, `load` on standalone models | Persist learned embedding model state. |

See [Neural Embedding Models And Features](neural-features.md).

## Graph Models And Features

Graph support is available both as direct standalone modeling and as optional
feature generation for another estimator. Use it when zone, route, or temporal
relationships are part of the model, especially directed pickup-dropoff effects.

| Feature | Public surface | Notes |
| --- | --- | --- |
| Node2Vec encoder | `Node2VecEncoder`, `Node2VecFeatureEncoder`, `Node2VecConfig`, `AliasSampler`, `RandomWalkGenerator`, `Node2VecTrainer`, `EdgeEmbeddingModel`, `EmbeddingFeatureTransformer` | Directed/weighted random-walk embeddings with p/q transition bias, alias sampling, deterministic Rayon-backed seeded walks, skip-gram training, and edge-row embedding features. |
| GraphSAGE encoder | `GraphSageEncoder`, `GraphSageFeatureEncoder`, `GraphSageConfig` | Homogeneous graph embeddings with node attributes. |
| HeteroGraphSAGE encoder | `HeteroGraphSageEncoder`, `HeteroGraphSageFeatureEncoder`, `HeteroGraphSageConfig` | Typed-edge graph embeddings. |
| HinSAGE encoder | `HinSageEncoder`, `HinSageFeatureEncoder`, `HinSageConfig` | Typed-node and typed-relation graph surface with schema validation. |
| Feature transformer | `GraphFeatureTransformer`, `GraphFeatureBundle` | Produces dense graph columns and optional sparse sets for another model. |
| Graph schemas | `GraphSchema`, `EdgeType`, `DirectionalityConfig`, `DirectedMetaPath`, `TemporalEdge` | Validates directed heterogeneous graph contracts. |
| Graph builders | `HomogeneousGraph`, `HeterogeneousGraph`, `SourceTargetPairNodes`, `materialize_source_target_pair_nodes` | Normalizes topology and preserves source-target pair identity. |
| Walk generators | `MetaPathWalkGenerator`, `TemporalWalkGenerator`, `SignedEdgeSampler` | Directed, temporal, and signed walk utilities. |
| Standalone graph regressors | `Node2VecStandaloneRegressor`, `GraphSageStandaloneRegressor`, `HeteroGraphSageStandaloneRegressor`, `HinSageStandaloneRegressor` | Direct graph regression without a boosted wrapper. |
| Standalone link predictors | `Node2VecLinkPredictor`, `GraphSageLinkPredictor`, `HeteroGraphSageLinkPredictor`, `HinSageLinkPredictor` | Link scoring plus reports. |
| Link metrics | `binary_auc`, `binary_average_precision`, `top_k_metrics`, `mean_reciprocal_rank`, `link_prediction_report` | Ranking and binary link-prediction diagnostics. |
| Directional features | `DirectionalFeature`, `DirectionalityConfig` | Preserves `source -> target` semantics and reverse-flow contrasts. |
| Graph regularization and rules | `CsrGraph`, `GraphLaplacian`, `GraphSmoother`, `GraphRegularizedBooster`, `GraphSplitRegularization`, `GraphLeafSmoothing`, `SymbolicRelationSet`, `RuleCompiler`, `MonotoneConstraintSet`, `InteractionConstraintSet` | Supplied sparse graphs, symbolic relation penalties, row-graph split scoring, graph-smoothed constant leaves, deterministic rule features, and split constraint checks for smoothing residuals, leaf values, and graph-aware predictions. |

See [Graph Models And Features](graph-features.md).

## General Utilities, Evaluation, And Forecasting

General utilities include single-series forecasts, Kalman filters,
intermittent-demand methods, sequence reference utilities, and ordinary
kriging. Kalman support includes frame-based local-level, local-linear, and
self-tuning forecasters plus diagnostic filter utilities. Sequence utilities
cover known-prefix continuation, reference-axis path inference, group-level OOF
candidate row generation, OOF leakage checks, per-group error summaries, and
aligned candidate blending. Evaluation helpers include out-of-time, temporal
blocked, spatial blocked, spatial buffered, environmental blocked, spatial
grouped, and grouped blocked splits; pinball loss; logloss; ROC-AUC; PR-AUC;
Brier score; ECE calibration error; NDCG; MAP; MRR; interval diagnostics;
CUSUM and Page-Hinkley regime signals; EWMA volatility; regime interval
policies; rolling median and MAD residuals; leakage-safe Kalman residual state
correction; residual Moran's I; spatial CV gap; jitter volatility; and
conformal residual helpers.

Forecasting classes validate data, train models, run rolling-origin backtests,
compute metrics, and preserve artifacts. The Prophet-style piecewise linear
seasonal model also supports external trend multipliers and recent residual
shock propagation for market-belief adjustments.

See [General Utilities](general_utilities.md), [Forecasting](forecasting.md),
and the [forecasting model guides](user-guide/forecasting-models/index.md).

## Command Line Interfaces

| Command group | Public surface | Notes |
| --- | --- | --- |
| Regression CLI | `cartoboost train`, `predict`, `eval`, `inspect` | Dense numeric CSV workflows. |
| Forecasting CLI | `cartoboost forecast fit`, `predict`, `backtest`, `compare` | Forecasting command scaffold and strict CSV validation. |

Use Python for sparse-set, graph-derived, neural embedding, and custom
forecasting workflows that need richer in-memory objects.

See [CLI Reference](reference/cli.md), [CLI User Guide](user-guide/cli.md), and
[Forecasting](forecasting.md).

## Quality And Benchmark Reporting

Benchmark claims should name the dataset, target, split, feature set, metric,
model settings, and whether data is synthetic, generated acceptance data, or
real benchmark data. For graph and neural features, keep standalone-model claims
separate from feature-generation claims.
