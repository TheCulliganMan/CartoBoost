# Spatial Econometrics Models

Classical spatial econometrics models are interpretable baselines for areal
units, store trade areas, service zones, and route cells. Use them when the
question is whether a sparse neighbor graph explains the target through a
lagged response, lagged errors, or lagged covariates.

These models are competitors to `CartoBoostRegressor`, not replacements. Fit
them on the same training rows, with the same measured covariates and the same
spatial weights, then compare residual diagnostics and holdout metrics.

## Models

| Model | Python class | Formula | Main diagnostic |
| --- | --- | --- | --- |
| Spatial lag | `SpatialLagRegressor` | `y = rho W y + X beta + e` | `rho` and residual Moran's I |
| Spatial error | `SpatialErrorRegressor` | `y = X beta + u`, `u = lambda W u + e` | `lambda` and residual Moran's I |
| Spatial Durbin | `SpatialDurbinRegressor` | `y = rho W y + X beta + W X theta + e` | direct, indirect, and total effects |
| Two-stage lag baseline | `SpatialTwoStageLeastSquares` | sparse lag baseline with the same public contract | `rho` and residual Moran's I |

All four models fit through Rust native code and use the shared geo-core sparse
`SpatialWeights` representation internally. Python only adapts arrays and
exposes the sklearn-style estimator surface.

## Spatial Weights

`SpatialWeights` accepts sparse COO inputs. Row standardization is enabled by
default, and isolated rows are recorded instead of silently removed.

```python
from cartoboost import SpatialWeights

taxi_zone_weights = SpatialWeights(
    4,
    4,
    rows=[0, 1, 1, 2, 2, 3],
    cols=[1, 0, 2, 1, 3, 2],
    values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

assert taxi_zone_weights.isolated_rows_ == []
```

For neighbor dictionaries, use `from_neighbors`:

```python
trade_area_weights = SpatialWeights.from_neighbors(
    {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }
)
```

Invalid weights fail before fitting. The matrix must be square, row counts must
match `X` and `y`, and weights, features, and targets must be finite.

## Areal Units Example

This example fits a spatial lag model for taxi-zone fare residuals after dense
trip features have been assembled.

```python
from cartoboost import SpatialLagRegressor

model = SpatialLagRegressor()
model.fit(
    X_train_zone_features,
    fare_residual_train,
    spatial_weights=taxi_zone_weights_train,
)

pred = model.predict(
    X_valid_zone_features,
    spatial_weights=taxi_zone_weights_valid,
)
summary = model.summary()

print(summary["diagnostics"]["rho"])
print(summary["diagnostics"]["residual_morans_i"])
```

Use this when the target for a pickup zone, dropoff zone, census area, or
territory may move with neighboring areas.

## Store Trade Areas

For store trade areas, build `W` from shared boundaries, drive-time overlap, or
nearest-neighbor catchments. Keep the covariates as measured store attributes:
local demand, inventory capacity, service hours, and distance to the nearest
hub.

```python
from cartoboost import SpatialErrorRegressor

store_error = SpatialErrorRegressor().fit(
    store_features_train,
    store_sales_residual_train,
    spatial_weights=store_trade_area_weights,
)

diagnostics = store_error.summary()["diagnostics"]
```

Choose the error model when the covariates explain the mean but residuals still
show spatial autocorrelation.

## Route Cells

For route cells, make each row a cell, segment, or origin-destination service
unit. Fit Durbin when neighboring cells' features are part of the hypothesis.

```python
from cartoboost import SpatialDurbinRegressor

durbin = SpatialDurbinRegressor().fit(
    route_cell_features_train,
    route_cell_duration_train,
    spatial_weights=route_cell_weights,
)

effects = durbin.summary()["diagnostics"]["total_effects"]
```

Report direct, indirect, and total effects with the feature order used in `X`.
Direct effects describe same-row covariates, indirect effects describe the
neighbor spillover component, and total effects combine them.

## CartoBoost Comparison

Run these models beside `CartoBoostRegressor` and residual kriging when the
study needs both predictive and interpretable spatial evidence.

```python
from cartoboost import CartoBoostRegressor, SpatialLagRegressor

cartoboost = CartoBoostRegressor(
    n_estimators=200,
    splitters=["axis", "diagonal_2d", "sparse_set"],
).fit(
    X_train,
    y_train,
    sparse_sets={"pickup_zone": pickup_zone_memberships_train},
)

lag = SpatialLagRegressor().fit(
    X_train_dense,
    y_train,
    spatial_weights=taxi_zone_weights_train,
)
```

Keep the comparison honest: same train/test split, same target, same feature
access, and clear reporting of whether the spatial weights were constructed
from training-side geography only.

When PySAL is installed, use it as an external audit on small fixtures:

```python
import libpysal
import spreg

pysal_w = libpysal.weights.W({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]})
pysal_w.transform = "r"
pysal_lag = spreg.ML_Lag(y_train.reshape(-1, 1), X_train_dense, w=pysal_w)
```

CartoBoost's spatial econometrics surface is intentionally sparse and
Rust-native; PySAL remains an optional comparison dependency rather than a core
runtime dependency.

## Model Lab And WASM Payload Shape

Model-lab and WASM callers should use the same sparse contract rather than
materializing dense `W` matrices. A complete request can be represented as:

```json
{
  "model": "spatial_durbin",
  "row_standardize": true,
  "weights": {
    "n_rows": 4,
    "n_cols": 4,
    "rows": [0, 1, 1, 2, 2, 3],
    "cols": [1, 0, 2, 1, 3, 2],
    "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  },
  "features": {
    "names": ["pickup_demand", "avg_trip_distance"],
    "rows": [[12.0, 1.6], [15.0, 1.9], [11.0, 2.2], [9.0, 2.4]]
  },
  "target": [18.0, 21.0, 19.0, 17.0]
}
```

The corresponding response should expose the fitted coefficients and
diagnostics:

```json
{
  "model": "SpatialDurbinRegressor",
  "coefficients": [0.31, 1.42],
  "diagnostics": {
    "rho": 0.18,
    "residual_morans_i": 0.03,
    "direct_effects": [0.31, 1.42],
    "indirect_effects": [0.09, 0.21],
    "total_effects": [0.40, 1.63],
    "isolated_rows": []
  }
}
```

Do not send dense adjacency matrices through model-lab or WASM APIs. Keep
weights sparse and fail the request when the graph shape, row count, or numeric
values are invalid.

## Persistence

Each estimator supports `save(path)` and classmethod `load(path)`.

```python
model.save("taxi-zone-spatial-lag.json")
restored = SpatialLagRegressor.load("taxi-zone-spatial-lag.json")
```

Predictions are stable after reload when the same `X` and compatible spatial
weights are supplied.
