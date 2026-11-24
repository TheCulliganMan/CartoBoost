# Boosted Tree Models

Start with the target. A taxi trip can have a numeric duration, a class label,
or a relevance score within a dispatch query. CartoBoost provides a separate
estimator for each, with shared controls for time, location, and membership
structure.

## Choose Your Estimator

| What you need to predict | Model | Typical output |
| --- | --- | --- |
| Fare, duration, trip distance, or another numeric value | [CartoBoost Regressor](cartoboost-regressor.md) | One value per row |
| A binary or multiclass label | [CartoBoost Classifier](cartoboost-classifier.md) | Labels and class probabilities |
| Candidate order within a query | [CartoBoost Ranker](cartoboost-ranker.md) | A score for each candidate |

For future values on a regular time axis, use the
[forecasting guides](../forecasting-models/index.md). For learned pickup-zone
or route IDs, see [neural embeddings](../neural-models/index.md).

## When To Use Boosting

Boosting is a useful starting point when measured trip features explain the
target and their effects can be nonlinear. Begin with a dense tabular fit.
Declare additional structure when it represents something real: hour wraps at
24, coordinates form pairs, and a trip can belong to several route or zone sets.

| Add this structure | Read |
| --- | --- |
| Cyclic hours, coordinate pairs, or typed columns | [Feature Schema](../../feature_schema.md) |
| Native labels and ordered categories | [Categorical Features](../categorical-features.md) |
| Zones, routes, cells, or overlapping memberships | [Sparse Features](../../sparse_features.md) |
| Spatial boundaries and coordinate units | [Spatial Modeling](../../spatial_modeling.md) |
| Loss, tree size, fuzzy routing, and leaves | [Parameters](../parameters.md) |

## Fit, Compare, Keep The Evidence

Follow [Using the Python Estimator](../python-estimator.md) for table inputs,
sample weights, model inspection, and save/load. Compare CartoBoost with
LightGBM or XGBoost using the same features and holdout rows.

Use [spatial cross-validation](../spatial-cv-best-practices.md) when the model
must work in unseen pickup zones. Use a time holdout when it must predict later
trips. Report error and runtime together so additional structure has a clear
cost as well as a measured benefit.
