# Spatial CV Best Practices

Use spatial validation when the claim is about generalizing to withheld places
or regimes. Random CV can overstate quality when nearby pickup/dropoff rows
share zone effects, road geometry, demand shocks, or weather conditions.

## Buffered Spatial Folds

`spatial_buffered_cv` holds out spatial blocks and removes training rows within
`buffer_radius` of each test block.

```python
from cartoboost import spatial_buffered_cv

folds = list(
    spatial_buffered_cv(
        projected_pickup_xy,
        n_splits=5,
        buffer_radius=500.0,
        coordinate_units="meters",
    )
)
```

Use projected coordinates for positive buffers. Latitude/longitude degree
buffers are ambiguous and raise unless `allow_degree_buffer=True`.

## Grouped Spatial Folds

Use `spatial_grouped_cv` when rows must be held out by group and also screened
for nearby leakage. This is useful for pickup zones, customers, lanes, or route
families.

```python
from cartoboost import spatial_grouped_cv

folds = list(
    spatial_grouped_cv(
        projected_pickup_xy,
        groups=pickup_zone_ids,
        n_splits=5,
        buffer_radius=500.0,
        coordinate_units="meters",
    )
)
```

Every test group is absent from training. If the buffer removes all training
rows for a fold, CartoBoost raises because the validation design is not usable
as stated.

## Environmental Blocks

`environmental_blocked_cv` clusters covariates such as weather, demand regime,
calendar pressure, or disruption indicators. It uses optional sklearn KMeans by
default and provides a deterministic ordered fallback with
`use_sklearn=False`.

```python
from cartoboost import environmental_blocked_cv

folds = list(
    environmental_blocked_cv(
        weather_and_demand_frame,
        feature_cols=["temperature", "rain_mm", "demand_lag_24h"],
        n_splits=5,
        random_state=42,
    )
)
```

## Diagnostics

Report the random-to-spatial gap and residual autocorrelation when spatial
claims matter.

```python
from cartoboost import residual_morans_i, spatial_cv_gap

gap = spatial_cv_gap(random_cv_rmse, buffered_cv_rmse)
moran_i = residual_morans_i(projected_pickup_xy_validation, residuals)
```

For loss metrics where lower is better, define the gap direction explicitly in
the benchmark table. The helper returns `random_cv_score - spatial_cv_score`.
