# CartoBoost Boosting Model Guides

Use these guides for row-level boosted tree models. If the target is a regular
time series, switch to the forecasting guides. If the learned ID representation
itself is the point of the experiment, switch to the neural guides.

## Use When

- You need a tree model for numeric, class, or grouped-ranking targets.
- The signal may depend on time, location, membership, or residual structure.
- You want one entry point for fit, predict, save, load, and parameter
  selection, then a narrower page for each estimator.

## Guides

- [CartoBoost Regressor](cartoboost-regressor.md): numeric targets such as duration,
  fare, demand, or residuals.
- [CartoBoost Classifier](cartoboost-classifier.md): binary or multiclass labels.
- [CartoBoost Ranker](cartoboost-ranker.md): grouped candidate ordering.

## Shared Setup

Start from dense measured features, then add specialized controls only when
they match the question:

- [Python Estimator](../python-estimator.md) for fit, predict, save, and load
  behavior.
- [Parameters](../parameters.md) for splitters, losses, fuzzy routing, and
  leaves.
- [Categorical Features](../categorical-features.md) for native categorical
  and ordinal columns.
- [Spatial CV Best Practices](../spatial-cv-best-practices.md) for blocked
  validation.
- [Spatial Modeling](../../spatial_modeling.md) for coordinates and spatial
  split behavior.
- [Feature Schema](../../feature_schema.md) for validation contracts.
- [Sparse Features](../../sparse_features.md) for zones, cells, and route
  memberships.

Compare against serious baselines on the same split, keep the feature access
equal, and record the exact command, dataset, split, and metric summary.
