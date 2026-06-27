# Temporal-Spatial Modeling

CartoBoost is built for regression problems where time, place, membership, or
local neighborhoods drive the target. Examples include demand by hour and zone,
adjustments by location, and operational metrics grouped by source/target
pairs.

## When It Is A Good Scientific Choice

Use `CartoBoostRegressor` when the modeling question depends on structured
place/time effects that should remain visible in the fitted workflow:

- Coordinates may define boundaries, corridors, or radial hotspots that are
  awkward to express with only axis-aligned cuts.
- Zone, route, H3/S2, grid, corridor, or service-area memberships may be
  sparse but scientifically meaningful.
- Hour-of-day, weekday, or seasonal phase may wrap around, so 23:00 and 00:00
  should be treated as neighbors rather than distant values.
- Service boundaries, geocoding noise, and route assignments may be fuzzy,
  making abrupt left/right split decisions undesirable.
- Heavy-tailed fare and duration targets may call for robust losses, and
  service-level studies may need conditional quantiles rather than only means.
- The saved artifact needs to preserve the schema, splitters, sparse-set
  requirements, loss, fuzzy settings, and additive values used for
  interpretation.

This does not replace serious baselines. XGBoost and LightGBM are excellent
tabular comparators. CartoBoost is useful when the feature engineering needed
for those models starts to hide the structure of the study, or when a
CartoBoost-specific control directly tests a scientific hypothesis.

## Feature Patterns

| Pattern | CartoBoost feature path | Why it helps |
| --- | --- | --- |
| Hour-of-day, weekday, seasonality | Dense periodic feature with `periodic:<period>` | Preserves wraparound adjacency. |
| Latitude/longitude or projected x/y | Dense numeric features with `diagonal_2d` or `gaussian_2d` | Learns spatial boundaries and neighborhoods without only stair-step axis cuts. |
| Zones, encoded H3 cells | `sparse_sets={...}` with `splitters=["sparse_set"]` | Uses list-valued memberships directly. |
| Smooth transitions near a boundary | `fuzzy=True` with `fuzzy_bandwidth` and optional `fuzzy_kernel` | Routes samples fractionally instead of forcing a hard left/right decision. |
| Local trend inside a region | `leaf_predictor="linear"` | Fits a ridge residual model inside leaves. |
| Heavy-tailed or asymmetric targets | `loss="mae"`, `loss="huber"`, `loss="log_l2"`, or `loss="quantile"` | Aligns the objective with the estimand. |

## Example

```python
from cartoboost import CartoBoostRegressor

schema = {
    "dense": [
        {"name": "pickup_x", "kind": "numeric"},
        {"name": "pickup_y", "kind": "numeric"},
        {"name": "hour", "kind": "periodic", "period": 24},
        {"name": "distance_m", "kind": "numeric"},
    ],
    "sparse_sets": [
        {"name": "taxi_zones", "kind": "sparse_set"},
    ],
}

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=30,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"],
    fuzzy=True,
    fuzzy_bandwidth=0.05,
    fuzzy_kernel="tricube",
)

model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"taxi_zones": taxi_zones_train},
    feature_schema=schema,
)
```

This specification says more than "fit a booster." It declares that pickup
coordinates, hour, distance, and taxi-zone memberships are part of the
scientific design, and it saves those roles with the fitted artifact when the
schema is provided.

## Encoded H3 Or Grid Cells

CartoBoost does not compute H3 cells from latitude and longitude. If you already
have H3, S2, grid, zone, or corridor IDs, encode them as non-negative integers
and pass them through a sparse-set column:

```python
model.fit(
    X_dense,
    y,
    sparse_sets={"pickup_h3": [[617700169957507071], [617700169957507583]]},
)
```

A sparse split routes left when a row contains one of the learned IDs and right
otherwise. Empty rows and unseen IDs route as no match. Under cold-cell,
cold-zone, or cold-route validation, report this behavior explicitly because
unseen IDs cannot recover learned ID-specific effects.

## Robust And Quantile Targets

Structured fare and duration data often have traffic disruptions, metering
differences, cancellations, and data-quality outliers. If the scientific
estimand is not the conditional mean, choose the loss accordingly:

| Target | Loss |
| --- | --- |
| Mean fare, duration, demand, or residual | `loss="l2"` or `loss="squared_error"` |
| Median-like or outlier-resistant residual | `loss="mae"` or `loss="absolute_error"` |
| Smooth robust objective with bounded outlier influence | `loss="huber"` |
| Positive skew with log-scale emphasis | `loss="log_l2"` |
| Conditional interval, lower-tail, or upper-tail service level | `loss="quantile"` with `quantile_alpha=...` |

`l1`, `huber`, `log_l2`, and quantile loss currently require
`leaf_predictor="constant"`.

## Artifact And Interpretation Workflow

For scientific work, keep the fitted model tied to the data contract that
produced it:

1. Use a feature schema when dense periodic roles or sparse-set columns matter.
2. Save the model JSON with `model.save(...)` so splitters, leaf predictor,
   fuzzy settings, loss, schema, and sparse-set requirements are preserved when
   available.
3. Use `predict_additive_values(X)` or the optional SHAP helpers to inspect
   which components move predictions for trips, zones, routes, or hours.
4. Report localized diagnostics by pickup zone, dropoff zone, route, hour, or
   spatial holdout group before making claims about place/time behavior.

## Evaluation

Temporal-spatial models should be tested with the kind of holdout they will
face in use:

- Use random holdouts to measure general regression quality.
- Use spatial holdouts to test new zones, cells, routes, or corridors.
- Use out-of-time validation to test later periods.
- Report residuals by pickup zone, dropoff zone, route, or hour to find
  localized failure modes.
- Compare against axis-only CartoBoost, XGBoost, or LightGBM baselines with the
  same train/test split and feature set.

CartoBoost is a candidate when these temporal-spatial holdouts show structure
that the model can express with fewer ad hoc preprocessing steps. Keep claims
tied to your data, features, split strategy, and metrics.
