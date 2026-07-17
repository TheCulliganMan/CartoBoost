import {GeostatisticsModelExample} from '@site/src/components/ModelingLabClient';

# Gaussian-Process Geostatistics

Import these models from `cartoboost.geostats`.

Use `NearestNeighborGPRegressor` when each observation has a point coordinate
and the target should be interpolated with uncertainty. It is a scalable
probabilistic spatial regressor: prediction uses local nearest-neighbor
Gaussian process conditionals instead of solving one global kriging system.

Use `ResidualNNGPRegressor` when a tabular model explains the main signal and a
spatial residual field remains. The base estimator fits `X`; CartoBoost then
fits an NNGP on `y - base.predict(X)` at the training coordinates and adds that
spatial correction at prediction time.

<GeostatisticsModelExample title="Nearest-neighbor GP browser interpolation" />

## Python API

```python
import numpy as np
from cartoboost.geostats import NearestNeighborGPRegressor

coords = np.array([
    [-73.9851, 40.7589],
    [-73.9772, 40.7527],
    [-73.9680, 40.7590],
    [-73.9969, 40.7420],
])
duration_residual = np.array([0.7, -0.2, -0.5, 0.9])

gp = NearestNeighborGPRegressor(
    kernel="matern_3_2",
    range=0.04,
    sill=1.0,
    nugget=1e-4,
    n_neighbors=8,
)
gp.fit(None, duration_residual, coords=coords)

mean, std = gp.predict(None, coords=np.array([[-73.981, 40.756]]), return_std=True)
lower, upper = gp.predict_interval(None, coords=np.array([[-73.981, 40.756]]), coverage=0.9)
```

`SpatialGaussianProcessRegressor` is the facade for the same scalable point GP
behavior. Prefer `NearestNeighborGPRegressor` when documenting NNGP-specific
settings.

## Residual Correction

```python
from cartoboost import CartoBoostRegressor
from cartoboost.geostats import NearestNeighborGPRegressor, ResidualNNGPRegressor

base = CartoBoostRegressor(
    n_estimators=120,
    split_policy="structured",
)
model = ResidualNNGPRegressor(
    base,
    gp=NearestNeighborGPRegressor(kernel="matern_5_2", range=0.05, n_neighbors=12),
)

model.fit(X_train, y_train, coords=pickup_coords_train)
prediction, std = model.predict(X_test, coords=pickup_coords_test, return_std=True)
```

This is useful for taxi duration, fare, or demand tasks where distance, hour,
zone, and route features explain most variation but localized pickup or
dropoff residuals remain. Validate the base model and the residual correction
on the same split before interpreting the correction as a spatial gain.

## Kernels And Parameters

| Parameter | Meaning |
| --- | --- |
| `kernel` | One of `exponential`, `squared_exponential`, `matern_3_2`, or `matern_5_2`. |
| `range` | Coordinate distance scale for covariance decay. |
| `sill` | Spatial covariance scale. |
| `nugget` | Independent observation noise and numerical regularization. |
| `n_neighbors` | Local conditioning set size for each prediction. |
| `anisotropy_angle_degrees`, `anisotropy_scaling` | Rotates and stretches the distance metric. |
| `duplicate_tolerance` | Coordinates within this distance are rejected at fit time. Aggregate or jitter duplicates explicitly. |

Duplicate coordinate handling is deliberately strict. If two training rows have
the same coordinate within `duplicate_tolerance`, fitting raises an error
instead of silently averaging, dropping, or jittering rows.

## Variogram Utilities

```python
from cartoboost.geostats import binned_variogram, fit_variogram_wls

bins = binned_variogram(pickup_coords_train, residuals, bin_count=12)
fit = fit_variogram_wls(
    bins,
    range_candidates=[0.02, 0.04, 0.08],
    sill_candidates=[0.5, 1.0, 2.0],
    nugget_candidates=[0.0, 1e-4, 1e-2],
)
```

`empirical_semivariogram` and `binned_variogram` return lag bins,
semivariance, and pair counts.
`fit_variogram_wls` uses pair-count weighted least squares over the supplied
kernel/range/sill/nugget candidate grid. Treat this as a parameter-estimation
utility, not a benchmark claim.

## Uncertainty Maps

`predict(..., return_std=True)` returns the local GP mean and standard
deviation. Prediction variance is nonnegative by construction and should be
lower near training coordinates than far from observed pickup/dropoff regions.
For maps, score a grid of projected coordinates, render the mean as the
surface, and render `std` or interval width as the uncertainty layer.

The implementation runs on CPU and does not require GPU support.
