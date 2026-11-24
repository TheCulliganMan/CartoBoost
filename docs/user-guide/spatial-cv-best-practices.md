# Spatial Cross-Validation

Spatial validation holds out places rather than individual rows. Nearby taxi
trips can share zone
effects, traffic, weather, and route geometry, making a random row split easier
than deployment in an unfamiliar area.

Use `cartoboost.validation` to construct spatial, buffered, grouped, and
time-aware folds. A split manifest records the exact fold indices, coordinate
assumptions, and reproducibility hash so every candidate can use the same test.

## Buffered spatial folds

Use projected coordinates when training rows near a held-out block must be
removed from the training set:

```python
import cartoboost
from cartoboost.geo import CoordinateMatrix
from cartoboost.validation import native_buffered_spatial_split

manifest = native_buffered_spatial_split(
    CoordinateMatrix(pickup_x, pickup_y, crs="EPSG:2263"),
    n_folds=5,
    buffer_distance=500.0,
    dataset_fingerprint="sha256:...",
    coordinate_crs_note="EPSG:2263 projected pickup coordinates",
    model_version=cartoboost.__version__,
    dependency_versions={"cartoboost": cartoboost.__version__},
)
fold_id, train_idx, test_idx = manifest.folds()[0]
```

Latitude/longitude degree buffers are ambiguous. Project the coordinates first
and record the CRS in `coordinate_crs_note`.

## Grouped spatial folds

Use `native_grouped_split` when entire pickup zones, customers, lanes, or route
families must be absent from training. Use a buffered manifest as well when
nearby groups could leak signal through coordinates.

```python
import cartoboost
from cartoboost.validation import native_grouped_split

manifest = native_grouped_split(
    pickup_zone_ids,
    n_folds=5,
    dataset_fingerprint="sha256:...",
    coordinate_crs_note="not_applicable",
    model_version=cartoboost.__version__,
    dependency_versions={"cartoboost": cartoboost.__version__},
)
```

If a fold leaves no usable training rows, the native constructor raises rather
than weakening the requested validation design.

## Temporal and spatial-temporal folds

Use `native_temporal_split` for rolling-origin forecasts and
`native_spatial_temporal_split` when both location and time must be held out.
Keep the training cutoff strictly before the forecast horizon and store the
manifest hash alongside the forecast artifact.

## Diagnostics

Residual Moran's I and the random-to-spatial score gap are supported diagnostics:

```python
from cartoboost.metrics import residual_morans_i, spatial_cv_gap

gap = spatial_cv_gap(random_cv_rmse, buffered_cv_rmse)
residual_i = residual_morans_i(projected_pickup_xy_validation, residuals)
```

Use these diagnostics to explain a result, not to choose a model using holdout
labels. For a complete comparison, report the split hash, target transform,
feature access, baseline roster, RMSE/MAE, and fit/predict timing.
