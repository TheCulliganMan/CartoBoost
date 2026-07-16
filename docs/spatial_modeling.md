import {GeoFeatureExample} from '@site/src/components/ModelingLabClient';

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
- The saved artifact needs to preserve the schema, split policy, sparse-set
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
| Route bearing or directional drift | `clockwise_bearing_unit_vector` or `initial_bearing_unit_vector_latlng` | Encodes direction as continuous `(east, north)` columns so north and northwest are close without angle wraparound artifacts. |
| Hub, airport, depot, or CBD proximity | `radial_anchor_distances` or `rbf_anchor_features` | Emits distance or smooth radial-basis columns around explicit anchors. |
| Corridor-relative position | `local_frame_features` | Emits `along_axis` and `cross_axis` columns in a supplied local coordinate frame. |
| Latitude/longitude or projected x/y | Dense numeric features with `diagonal_2d` or `gaussian_2d` | Learns spatial boundaries and neighborhoods without only stair-step axis cuts. |
| Zones, encoded H3 cells | `sparse_sets={...}` with `split_policy="structured"` | Uses schema-declared list-valued memberships directly. |
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
    split_policy="structured",
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

## Bearing Unit Features

Use bearing unit vectors when the direction from origin to destination matters.
Do not feed raw degrees when wraparound continuity matters; 359 degrees and 0
degrees should be nearly identical, not far apart. CartoBoost exposes Rust-backed
helpers that return `(east, north)` unit components:

```python
from cartoboost.geo import (
    clockwise_bearing_unit_vector,
    initial_bearing_unit_vector_latlng,
)

pickup_xy = (0.0, 0.0)
dropoff_xy = (-3.0, 3.0)
bearing_east, bearing_north = clockwise_bearing_unit_vector(pickup_xy, dropoff_xy)

pickup_latlng = (40.7580, -73.9855)
dropoff_latlng = (40.7804, -73.9570)
route_east, route_north = initial_bearing_unit_vector_latlng(pickup_latlng, dropoff_latlng)
```

`clockwise_bearing_unit_vector` is for projected or planar `(x, y)` coordinates.
`initial_bearing_unit_vector_latlng` is for `(latitude, longitude)` degrees and
uses the great-circle initial bearing. Identical origin/destination points
return `None`; benchmark feature generation should keep or filter those rows
explicitly rather than silently replacing them.

Route, radial, and local-frame helpers build on the same Rust geo primitives:

```python
from cartoboost.geo import (
    local_frame_features,
    radial_anchor_distances,
    rbf_anchor_features,
    route_feature_vector,
)

mid_x, mid_y, distance, bearing_east, bearing_north = route_feature_vector(
    (0.0, 0.0),
    (3.0, 4.0),
)

anchors = [(0.0, 0.0), (4.0, -2.0), (-3.0, 3.0)]
distance_to_hubs = radial_anchor_distances((1.0, 2.0), anchors)
smooth_hub_features = rbf_anchor_features((1.0, 2.0), anchors, length_scale=3.0)
along_corridor, cross_corridor = local_frame_features(
    (2.0, 3.0),
    origin=(0.0, 0.0),
    axis=(1.0, 1.0),
)
```

<GeoFeatureExample title="Geo feature encoder browser examples" />

## Encoded Route And Grid Cells

CartoBoost can encode latitude/longitude points and decoded route geometries
into optional H3 or S2 sparse-set columns when the matching package is installed.
If you already have H3, S2, grid, zone, or corridor IDs, encode them as
non-negative integers and pass them through a sparse-set column:

```python
model.fit(
    X_dense,
    y,
    sparse_sets={"pickup_h3": [[617700169957507071], [617700169957507583]]},
)
```

For routes returned by OSRM or Valhalla, request decoded coordinates first. OSRM
GeoJSON route geometries use `[longitude, latitude]` coordinate pairs; direct
Python route sequences use `(latitude, longitude)` pairs.

```python
from cartoboost.h3 import build_h3_route_sparse_sets

route_sparse_sets = build_h3_route_sparse_sets(
    osrm_routes,
    name="route_h3",
    resolution=9,
    parent_resolutions=[5, 7],
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
2. Save the model JSON with `model.save(...)` so the split policy, leaf predictor,
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
