import {RegressionModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Regressor

Use `CartoBoostRegressor` for numeric row-level targets when the effect of
time, location, route membership, or other structure is part of the question.
Typical uses include duration, fare, demand, or residual modeling.

## Public Contract

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    split_policy="structured",
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## Browser WASM Example

<RegressionModelExample title="CartoBoost regressor browser model" mode="auto" loss="l2" />

## Use When

| Need | Better first choice |
| --- | --- |
| Numeric row-level prediction. | `CartoBoostRegressor` |
| Class probabilities or labels. | `CartoBoostClassifier` |
| Query-local ordering. | `CartoBoostRanker` |
| Time-indexed future values. | Forecasting models |

## Common Controls

| Scientific need | Parameter family |
| --- | --- |
| Dense tabular baseline | `split_policy="auto"` or `"axis_only"` |
| Declared spatial/periodic/sparse structure | `split_policy="structured"` plus `feature_schema=` |
| Sparse zones, routes, cells, or areas | `split_policy="structured"` plus `sparse_sets=` |
| Native categorical labels or ordered tiers | `FeatureKind.CATEGORICAL` or `FeatureKind.ORDINAL` in `feature_schema=` |
| Smooth changes near boundaries | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` |
| Outlier-resistant regression | `loss="mae"`, `loss="huber"`, or `loss="log_l2"` |
| Conditional intervals or asymmetric service targets | `loss="quantile"`, `quantile_alpha=...` |
| Local residual trend inside learned regions | `leaf_predictor="linear"`, `linear_leaf_features=[...]` |
| Domain monotonicity | `monotonic_constraints=[...]` |

Use [Parameters](../parameters.md), [Feature Schema](../../feature_schema.md),
[Sparse Features](../../sparse_features.md), and [Spatial Modeling](../../spatial_modeling.md)
for the contract details.

## Validation

Report RMSE, MAE, and task-specific business metrics on the same split as the
baselines. Use spatial, temporal, group, or cold-entity splits when those are
the claim being tested.
