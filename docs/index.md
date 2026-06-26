# CartoBoost Documentation

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Release](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/release-version.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/release-version.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TheCulliganMan/CartoBoost/blob/main/LICENSE)

CartoBoost is a Python modeling toolkit for temporal-spatial
regression, classification, ranking, and forecasting problems where place and
time are part of the signal, not just columns in a table. It is designed for
work such as NYC taxi trip duration, fare amount, pickup-zone classification,
route ranking, pickup-zone demand, dropoff-zone demand, and
pickup-to-dropoff lane forecasting.

Use CartoBoost when your scientific question depends on structure that ordinary
tabular models only see after extensive manual feature engineering:

- time has cycles, such as hour-of-day or day-of-week demand;
- location has neighborhoods, corridors, hotspots, and boundaries;
- rows can belong to multiple places, such as pickup zones, dropoff zones, or
  route cells;
- direction matters, such as `PULocationID -> DOLocationID`;
- validation must respect time, space, groups, and leakage risks;
- claims need comparison against serious baselines such as LightGBM, XGBoost,
  local forecasting models, or external forecasting libraries.

CartoBoost is not a replacement for careful study design. It gives scientists a
set of model primitives, validation tools, artifacts, and benchmark workflows
that make temporal-spatial comparisons easier to run and easier to explain.

## Scientific Fit

CartoBoost is a good fit for taxi-domain studies such as:

- estimating fare or trip duration from pickup/dropoff zones, trip distance,
  pickup hour, weekday, and route context;
- modeling pickup-zone or dropoff-zone demand over time;
- forecasting daily demand for pickup/dropoff lanes such as JFK airport trips;
- measuring whether zone membership, route geometry, periodic hour behavior, or
  graph directionality adds signal beyond dense numeric features;
- producing reproducible model artifacts and benchmark reports from fixed train
  and validation splits.

It is less useful when the data have no meaningful place/time structure, when a
simple linear model already answers the scientific question, or when the study
cannot define a leakage-aware validation design.

## Modeling Ideas

Standard boosters are strong baselines for tabular regression. CartoBoost keeps
that workflow but adds splitters and feature contracts that match common
temporal-spatial structure:

- `periodic:24` keeps hour `23` and hour `0` adjacent instead of treating
  midnight as a hard break.
- `diagonal_2d` can represent oblique spatial boundaries that axis-only trees
  approximate indirectly.
- `gaussian_2d` can isolate local hotspots, depots, airports, or service areas.
- `sparse_set` can use list-valued taxi-zone or route-cell memberships without a
  wide one-hot matrix.
- fuzzy routing can reduce hard jumps around nearby places or times.
- graph encoders and graph regressors can preserve pickup/dropoff roles and
  source-target asymmetry.

These choices should be tested against simpler alternatives. In a taxi study,
that usually means holding the latest trips or demand windows out of training,
comparing the same split against LightGBM or XGBoost for regression, and using
rolling-origin backtests for forecasting.

## First Model

Code is intentionally secondary to the modeling decision: choose a target,
choose leakage-aware splits, define place/time features, then compare against a
baseline.

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=30,
    splitters=["axis", "periodic:24", "diagonal_2d", "gaussian_2d"],
)

model.fit(X_train, y_train)
predictions = model.predict(X_validation)
```

For NYC taxi data, `X_train` might contain trip distance, pickup-hour,
day-of-week, pickup coordinates, dropoff coordinates, `PULocationID`,
`DOLocationID`, airport-lane flags, or route-derived features. The target might
be transformed trip duration, fare amount, or future pickup-zone demand.

## Start Here

- [Installation](installation.md): PyPI installs, optional extras, source
  development, and troubleshooting.
- [Getting Started](getting-started.md): modeling-oriented first steps for taxi
  regression, forecasting, validation, baselines, and artifacts.
- [Choose A Model](user-guide/model-types.md): task-first router for the
  tabular regressor, local forecasting models, CartoBoost lag forecasting,
  kriging, spatial piecewise kriging, neural embedding models, graph models,
  and utilities.
- [Boosting Model Guides](user-guide/boosting-models/index.md): per-model
  guides for `CartoBoostRegressor`, `CartoBoostClassifier`, and
  `CartoBoostRanker`.
- [Classification Quickstart](user-guide/classification-quickstart.md):
  binary and multiclass logloss workflows with probability diagnostics.
- [Ranking Quickstart](user-guide/ranking-quickstart.md): grouped
  pairwise/LambdaRank workflows with NDCG, MAP, and MRR metrics.
- [Categorical Features](user-guide/categorical-features.md): native
  categorical and ordinal handling across regressor, classifier, and ranker.
- [Spatial CV Best Practices](user-guide/spatial-cv-best-practices.md):
  buffered, grouped, and environmental validation patterns.
- [Spatial Modeling](spatial_modeling.md): coordinate features, taxi-zone
  sparse sets, fuzzy routing, and blocked evaluation.
- [Forecasting](forecasting.md): `ForecastFrame`, rolling-origin backtests,
  leakage checks, CLI runs, portable forecast artifacts, and links to the
  individual forecasting model guides. It also maps the Rust-core quantile,
  residual-correction, regime-uncertainty, and probability-calibration
  primitives that support geotemporal forecasting workflows.
- [Benchmarks](benchmarks/index.md): reproducible comparison reports,
  acceptance checks, and claim limits.
- [Benchmark Methodology](benchmarks/methodology.md): evidence fields and v0.2
  acceptance gates for classification, ranking, categorical, spatial leakage,
  and regression-speed checks.

## Deeper References

- [Feature Catalog](feature_catalog.md): public modeling, forecasting, graph,
  neural, sparse, artifact, CLI, and benchmark features.
- [Python API](reference/python-api.md): sklearn-style estimator methods,
  forecasting classes, utility functions, save/load behavior, and explanation
  workflow.
- [Parameters](user-guide/parameters.md): estimator controls and supported
  splitters.
- [Graph Model Guides](user-guide/graph-models/index.md): standalone graph
  regressors, standalone link predictors, and graph feature workflows.
- [Graph Models And Features](graph-features.md): Node2Vec, GraphSAGE,
  HeteroGraphSAGE, HinSAGE, deterministic network embeddings, graph
  regularization, symbolic relation rules, split constraints, directed
  source-target features, metapaths, and graph feature bundles.
- [Neural Model Guides](user-guide/neural-models/index.md): standalone ID
  embedding regression and neural feature-generation workflows.
- [Neural Embedding Models And Features](neural-features.md): neural artifacts,
  fallback behavior, benchmark reporting, and failure modes.
- [Forecasting Model Guides](user-guide/forecasting-models/index.md): per-model examples
  for naive, seasonal naive, theta, ETS, ARIMA, AutoARIMA, Kalman, piecewise
  linear seasonal, kriging, spatial piecewise kriging, CartoBoost lag,
  AutoForecaster, and weighted ensembles.
- [Plotting](plotting.md): diagnostics, forecast figures, report bundles, and
  `prophet.plot.plot`-style rendering for Prophet-shaped outputs.

## What CartoBoost Supports

- L2 and quantile regression.
- L1, Huber, and log-L2 regression modes where supported by the current Rust
  backend.
- Constant and linear residual leaves.
- Axis, histogram-axis, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set,
  and fuzzy split behavior.
- Dense numeric arrays plus list-valued sparse-set columns in Python.
- Feature schemas for numeric, periodic, sparse-set, and contract validation.
- Versioned JSON model and weights artifacts.
- Forecasting workflows for pickup-zone, dropoff-zone, and lane-level demand,
  including deterministic forecast tables, leakage-safe rolling-origin
  evaluation, calibrated quantile intervals, residual-state correction,
  regime-aware uncertainty widening, calibrated probability events, and
  geographic-temporal benchmark fixtures.
- Optional SHAP explanations, Optuna tuning, Polars input support, and ONNX
  export for the supported dense axis-tree subset.
- Neural embedding features.
- node2vec, GraphSAGE, heterogeneous GraphSAGE, and typed-schema HinSAGE graph
  encoders plus standalone graph regressors and link predictors.

For a complete feature-by-feature map, see [Feature Catalog](feature_catalog.md).

## Install

```sh
uv add cartoboost
```

Optional extras are available for SHAP, Optuna, Polars, and ONNX:

```sh
uv add "cartoboost[explain,optuna,polars,onnx]"
```
