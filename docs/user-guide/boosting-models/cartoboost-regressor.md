import {RegressionModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Regressor

Predict taxi fare, trip duration, or another numeric target with a boosted tree
model that can use cyclic time, coordinates, and zone memberships. Start with
`CartoBoostRegressor` when each row describes one trip or one aggregate and you
want to test whether that structure improves a tabular baseline.

## Python Example

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

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

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

## Limitations

- Structured splits help only when schema roles and units are correct.
- Tree predictions do not extrapolate smooth trends beyond the observed feature range reliably.
- High-cardinality identifiers need leakage-safe encoding or dedicated graph/neural handling.
- Compare training and inference cost with strong tabular baselines on the same data.
