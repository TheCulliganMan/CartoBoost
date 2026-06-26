# CartoBoost Regressor

Use `CartoBoostRegressor` when each row has a numeric taxi-domain target and
the signal may depend on structured place, time, route membership, or local
residual behavior. Examples include fare, trip duration, pickup-zone demand
aggregates, lane residuals, or model-stacking residuals.

Prefer it when you need to test questions such as:

- Do pickup/dropoff effects persist after controlling for trip distance, hour,
  and day features?
- Are sparse zones, routes, H3/S2 cells, or service areas informative even when
  many memberships are rare?
- Does a smooth transition near a learned spatial boundary reduce localized
  residual artifacts?
- Does an outlier-resistant or quantile objective match the target better than
  mean regression?

## Basic Fit

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

## Common Controls

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

## Read Next

See [Parameters](../parameters.md), [Feature Schema](../../feature_schema.md),
[Sparse Features](../../sparse_features.md), [Spatial Modeling](../../spatial_modeling.md),
and the [Python API Reference](../../reference/python-api.md).
