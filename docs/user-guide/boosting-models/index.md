# Boosting Model Guides

These guides cover CartoBoost's sklearn-style boosted tree estimators for
row-level taxi modeling. Use this section when each row is a trip, route
observation, zone-hour aggregate, ranked candidate, or residual model rather
than a regular forecast series.

Use [Forecasting Model Guides](../forecasting-models/index.md) when the target
is future demand over a regular time index. Use
[Neural Model Guides](../neural-models/index.md) when the learned ID embedding
itself is the model under test.

## Pick A Guide

| Model guide | Best first use | Notes |
| --- | --- | --- |
| [CartoBoost Regressor](regressor.md) | Predict fare, duration, demand aggregates, or residuals from row-level features. | Main boosted tree estimator for numeric targets. |
| [CartoBoost Classifier](classifier.md) | Predict labels such as airport-trip flag, high-delay bucket, or route-risk class. | Supports binary and multiclass probability workflows. |
| [CartoBoost Ranker](ranker.md) | Order candidates within a pickup, lane, route request, or planning group. | Uses grouped pairwise or LambdaRank objectives. |

## Shared Inputs

Start with measured dense features such as trip distance, pickup hour,
day-of-week, projected pickup/dropoff coordinates, route history, fare history,
or duration history. Add structured feature contracts only when they match the
scientific question:

| Need | Guide |
| --- | --- |
| Estimator lifecycle, save/load, and sklearn-style methods | [Python Estimator](../python-estimator.md) |
| Splitter, loss, fuzzy routing, and leaf controls | [Parameters](../parameters.md) |
| Native categorical and ordinal columns | [Categorical Features](../categorical-features.md) |
| Spatially blocked validation | [Spatial CV Best Practices](../spatial-cv-best-practices.md) |
| Coordinates, route geometry, and fuzzy spatial behavior | [Spatial Modeling](../../spatial_modeling.md) |
| Feature schemas and validation contracts | [Feature Schema](../../feature_schema.md) |
| Pickup/dropoff zones, H3/S2 cells, and route memberships | [Sparse Features](../../sparse_features.md) |

## Validation

Boosting claims should compare against serious baselines on the same split and
feature set. For taxi regression that usually means a mean baseline plus
LightGBM or XGBoost. For classification, report logloss plus ROC-AUC or PR-AUC
when the positive class is rare. For ranking, report grouped metrics such as
NDCG, MAP, and MRR.

Record the target, split, row count, feature set, RMSE or task metric, train
time, prediction time, and the exact command or notebook entry point used to
produce the numbers.
