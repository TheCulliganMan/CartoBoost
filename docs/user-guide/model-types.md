---
sidebar_label: Choose a model
description: Match your prediction task and data structure to a CartoBoost model family.
---

# Choose a model

Start with the thing you need to predict. A trip's fare, next week's pickup
demand, and the effect of a geographic intervention require different model
assumptions and different holdouts.

Choose the smallest family that can represent the question. Add complexity
when a comparison on the same held-out observations supports it.

## Find your starting point

| Prediction task | Start here | A concrete example |
| --- | --- | --- |
| A numeric value for each row | [CartoBoost Regressor](boosting-models/cartoboost-regressor.md) | Estimate trip duration from distance, pickup hour, and location. |
| A class or class probability | [CartoBoost Classifier](boosting-models/cartoboost-classifier.md) | Classify an airport trip from the features available at dispatch. |
| An ordering within each group | [CartoBoost Ranker](boosting-models/cartoboost-ranker.md) | Rank candidate pickup zones for each dispatch decision. |
| Future values of a time series | [Forecasting models](forecasting-models/index.md) | Forecast hourly pickup counts by zone. |
| An outcome over connected entities | [Graph models](graph-models/index.md) | Learn from directed pickup-to-dropoff relationships. |
| Effects tied to repeated identifiers | [Neural embeddings](neural-models/index.md) | Learn a representation of recurring pickup and dropoff zones. |
| A pair, sequence, response curve, or outcome distribution | [Deep models](deep-models.md) | Model directional trips or a distribution of possible demand outcomes. |
| Values at unobserved coordinates | [Geostatistics](geostatistics-models.md) | Interpolate a spatial residual surface with uncertainty. |
| Dependence among neighboring areas | [Spatial econometrics](spatial-econometrics.md) | Estimate how zone-level demand relates to neighboring zones. |
| An intervention effect | [Geo-causal models](geo-causal-models.md) | Estimate a change in demand after a geographic intervention. |

For a working first model, follow [Your first model](../getting-started.md).
For an interactive introduction, open the [Modeling Lab](../../modeling-lab).
Browser examples illustrate behavior and input shape; benchmark reports provide
performance evidence.

## Row-level models: start with the target

The boosting estimators share structured feature controls. Select the estimator
from the target, then decide which structure belongs in the feature schema.

| Estimator | Target and output | Compare with |
| --- | --- | --- |
| [CartoBoostRegressor](boosting-models/cartoboost-regressor.md) | Numeric targets; one prediction per row. | A mean predictor and LightGBM or XGBoost. |
| [CartoBoostClassifier](boosting-models/cartoboost-classifier.md) | Binary or multiclass labels; classes and probabilities. | A dummy classifier and a tabular classifier using the same split. |
| [CartoBoostRanker](boosting-models/cartoboost-ranker.md) | Relevance within contiguous query groups; ranking scores. | A simple ranking rule and a grouped ranking baseline. |

For regression, compare RMSE, MAE, and R². For classification, include logloss
and ROC-AUC or PR-AUC as appropriate. For ranking, report grouped metrics such
as NDCG, MAP, or MRR. A row split that separates members of the same query does
not test ranking on unseen groups.

### Make structure explicit

| Signal in the data | Configuration | Guide |
| --- | --- | --- |
| Dense numerical effects | `split_policy="axis_only"` | [Parameters](parameters.md) |
| A daily or weekly cycle | `split_policy="structured"` with periodic schema entries | [Feature schema](../feature_schema.md) |
| Spatial boundaries | A spatial pair in the structured schema | [Spatial modeling](../spatial_modeling.md) |
| Multiple route cells or zone memberships | Named `sparse_sets` at fit and predict time | [Sparse features](../sparse_features.md) |
| `PULocationID`, `DOLocationID`, or ordered categories | Categorical or ordinal schema entries | [Categorical features](categorical-features.md) |
| Smooth transitions around learned boundaries | `fuzzy`, `fuzzy_bandwidth`, and `fuzzy_kernel` | [Parameters](parameters.md) |
| Outlier resistance or quantiles | A suitable `loss` and, for quantiles, `quantile_alpha` | [Objectives](../objectives.md) |
| Local trends within regions | `leaf_predictor="linear"` and selected linear features | [Parameters](parameters.md) |
| Required monotonic relationships | `monotonic_constraints` | [Constraints](../constraints.md) |

Fit encodings and learned features on training data only. Keep their mappings
with the fitted artifact, and give baseline models equivalent information.

## Forecasts: match the history and horizon

Use a `ForecastFrame` for regular time-indexed targets. Each series needs an
identity, a timestamp, and a target; covariates also need roles that reflect
whether they will be known at prediction time.

The [forecasting workflow](../forecasting.md) covers frame validation,
rolling-origin backtests, forecast metrics, and persistence. The model guides
below explain which assumptions distinguish each family.

| Series structure | Model guide | First comparison |
| --- | --- | --- |
| The last value or last season is a useful prediction | [Naive and seasonal naive](forecasting-models/naive-seasonal.md) | Use these as the baseline for more complex forecasts. |
| A lightweight trend model is sufficient | [Theta](forecasting-models/theta.md) | Seasonal naive. |
| Level, trend, and seasonality dominate | [ETS](forecasting-models/ets.md) | Seasonal naive and Theta. |
| Autocorrelation and differencing matter | [ARIMA and AutoARIMA](forecasting-models/arima.md) | ETS on the same origins. |
| A noisy local level or trend evolves over time | [Kalman](forecasting-models/kalman.md) | Naive and ETS. |
| Changepoints, events, and seasonal terms need interpretation | [Piecewise linear seasonal](forecasting-models/piecewise-linear-seasonal.md) | ETS and a lag model. |
| Many periods contain true zero demand | [Croston](forecasting-models/croston.md), [SBA](forecasting-models/sba.md), or [TSB](forecasting-models/tsb.md) | Naive and seasonal naive; distinguish missing values from zero. |
| Many series can share lagged features | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) | Local models and a conventional boosted lag model. |
| Coordinates should share information across series | [Kriging](forecasting-models/kriging.md) | A temporal model without spatial sharing. |
| Temporal trends and spatial residual structure both matter | [Spatial Piecewise Kriging](forecasting-models/spatial-piecewise-kriging.md) | The temporal model alone. |
| Regular windows support neural learning | [N-BEATS and N-HiTS](forecasting-models/beats-hits.md) | Local models and CartoBoost Lag. |
| Series have directional entity identities | [Neural Panel](forecasting-models/neural-panel.md) | A graph-free or embedding-free panel model. |
| Series are connected through observed or learned relationships | [Graph and market structure forecasting](forecasting-models/graph-spatiotemporal.md) | A temporal model with no graph. |
| Statistical candidates should be compared together | [AutoStatsBank](forecasting-models/auto-stats-bank.md) | Individual statistical candidates. |
| You need a selector over forecast candidates | [AutoForecaster](forecasting-models/auto-forecaster.md) | Its individual candidates on held-out origins. |
| A fixed combination is part of the design | [Weighted ensembles](forecasting-models/ensembles.md) | Each constituent forecast. |
| Predictions need calibrated intervals | [Probabilistic and conformal forecasting](forecasting-models/probabilistic-conformal.md) | Coverage and width on data excluded from calibration. |

Graph forecasting includes `DCRNNForecaster`, `GraphWaveNetForecaster`, and
`STAEformerForecaster`. The same guide describes the paper graph transformer
profiles and `MarketStructureForecaster` with its `MarketPanelFrame` input.
Select a profile from the data contract and validation task, not from the
architecture name alone.

## Graphs and embeddings: decide what you want to learn

Use a standalone graph or neural model when its prediction is the result you
want to serve. Use a feature encoder when its learned representation will
become input to another estimator.

| Information available | Start with |
| --- | --- |
| Directed or weighted graph topology | [Node2Vec](graph-models/cartoboost-node2vec.md) |
| A homogeneous graph with node attributes | [GraphSAGE](graph-models/cartoboost-graphsage.md) |
| Relation IDs without strict node-type constraints | [Heterogeneous GraphSAGE](graph-models/cartoboost-hetero-graphsage.md) |
| Typed nodes and relations with source–target constraints | [HinSAGE](graph-models/cartoboost-hinsage.md) |
| Stable repeated IDs and a standalone prediction target | [Neural embedding regressor](neural-models/cartoboost-neural-embedding.md) |
| Learned ID features for another estimator | [Neural embedding features](neural-models/cartoboost-neural-embedding-features.md) |
| Embeddings combined with a boosted regressor | [Neural embedding plus boosting](neural-models/cartoboost-neural-embedding-regressor.md) |

For a pickup/dropoff graph, construct relationships using only information
available before the evaluation cutoff. Test unseen-zone and unseen-route
behavior when deployment includes new entities. ID representations learned
from known zones do not establish performance on unknown zones.

See [Graph features](../graph-features.md) and [Neural features](../neural-features.md)
for the transformation contracts. Use the [deep-model chooser](deep-models.md)
for ordered-pair, sequence, event, response-curve, distribution, and constrained
decision models.

## Spatial estimates and intervention effects

A spatial predictor and a causal estimator answer different questions.
Interpolation asks what a value might be at a location. Causal estimation asks
what would have happened under a different intervention.

- [Spatial econometrics](spatial-econometrics.md) covers spatial lag, spatial
  error, and spatial Durbin regressions with explicit neighborhood structure.
- [Geostatistics](geostatistics-models.md) covers nearest-neighbor Gaussian
  processes, spatial Gaussian processes, and spatial residual corrections.
- [Geo-causal models](geo-causal-models.md) covers synthetic DID, geographic
  experiment design, placebo testing, and spillover diagnostics.

For one-off operations, the [utility reference](../general_utilities.md) covers
forecasting and spatial functions without a full estimator workflow.

## Use the holdout to make the final choice

| Deployment question | Evaluation design |
| --- | --- |
| Will the model work on later trips? | Hold out later observations and fit every learned transformation before the cutoff. |
| Will it work in unseen pickup zones? | Hold out spatial blocks or whole zones; add a buffer where neighboring observations would leak information. |
| Will it work on unseen routes or IDs? | Hold out complete groups and report behavior on unseen identities. |
| Will it forecast the required horizon? | Backtest multiple historical origins at that horizon. |

Use [spatial validation](spatial-cv-best-practices.md) and
[reproducible geo evaluation](geo-evaluation-standard.md) to define the split.
Spatial buffers should use projected linear units; degree-based distances
require an explicit opt-in. Keep sample size, features, model settings, and
training and prediction time alongside the metrics.

The [benchmark reports](../benchmarks/index.md) show results for specific
comparisons. They do not establish one winner for every dataset or task.

## Discover models from code

`cartoboost.models.ModelRegistry.defaults()` provides model metadata for tools
that need to enumerate names, tasks, capabilities, artifact formats, and
optional dependencies. Named modules such as `cartoboost.forecasting`,
`cartoboost.graph`, and `cartoboost.geostats` provide the corresponding APIs.

```python
from cartoboost.models import ModelRegistry

registry = ModelRegistry.defaults()
print(registry.names(namespace="geo"))
```

Automatic geo model selection is intentionally not shipped. Choose an estimator
explicitly and pair it with an appropriate validation split. Experimental
research adapters require an explicit backend and have the limitations
described in their guides.

{/* The registry example is exercised by scripts/check_docs_examples.py. */}
