import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Spatial Piecewise Kriging

`SpatialPiecewiseKrigingForecaster` is for panels that need both inspectable
temporal structure and spatial borrowing. It fits a piecewise linear seasonal
base, then uses ordinary kriging to add cutoff-safe spatial signal from stable
coordinates.

The interactive example below runs the same model when the sample table
has stable `longitude` and `latitude` columns.

## When To Use

Use this model when a panel has both of these mechanisms:

- each zone or lane has trend, changepoints, seasonality, events, or known
  future regressors that should remain inspectable;
- nearby coordinates have related residuals or known spatial covariates.

Good fits usually look like hourly demand by zone, store, or route panel where
each series has a stable coordinate such as a centroid or midpoint.

Prefer [Piecewise Linear Seasonal](piecewise-linear-seasonal.md) when geography
is not part of the claim. Prefer [Kriging](kriging.md) when the temporal base is
only a spatial panel interpolation problem. Prefer
[CartoBoost Lag](cartoboost-lag.md) or [AutoForecaster](auto-forecaster.md)
when the main signal is shared lag and calendar structure across many panels.

## Modes

The `mode` controls how spatial information enters the forecast:

| Mode | Use it when | Behavior |
| --- | --- | --- |
| `residual_kriging` | The temporal model is primary, but nearby series have correlated forecast errors. | Fits the piecewise base, computes cutoff-safe residuals by series, and kriges residual corrections for each horizon. |
| `kriged_regressors` | A known spatial covariate should be interpolated across target coordinates. | Kriges configured numeric spatial regressors and feeds them into the piecewise base as extra regressors. |
| `hybrid` | Both spatial regressors and residual geography matter. | Uses kriged regressors and residual corrections together. |

Missing coordinates are input errors by default. Set `allow_neighbor_fallback`
only when you explicitly accept falling back to a documented neighbor-based
behavior for incomplete coordinate coverage.

## Python Example

```python
from cartoboost.forecasting import ForecastFrame, SpatialPiecewiseKrigingForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="demand",
    series_id_col="zone_id",
    freq="h",
    known_future_covariates=["hour", "day_of_week"],
)

zone_centroids = {
    "132": (-73.7781, 40.6413),
    "161": (-73.9776, 40.7580),
    "236": (-73.9577, 40.7808),
}

model = SpatialPiecewiseKrigingForecaster(
    coordinates=zone_centroids,
    mode="hybrid",
    spatial_regressors=["airport_queue_pressure"],
    range=2.0,
    nugget=1.0e-6,
    max_neighbors=32,
    min_neighbors=4,
)
model.fit(frame)
forecast = model.predict(24)
```

The coordinate keys are string-matched to the frame series ids. Keep the same
rolling-origin split for every baseline when reporting a win.

## Diagnostics

The forecast JSON includes spatial and component details beside the final point
forecast:

| Column | Meaning |
| --- | --- |
| `prediction` | Final mean forecast after temporal and spatial terms. |
| `base_mean` | Piecewise linear seasonal forecast before the kriged correction. |
| `spatial_correction` | Residual kriging contribution added to the base forecast. |
| `kriging_variance` | Ordinary-kriging uncertainty for the spatial correction. |
| `selected_neighbors` | Neighbor series used by the kriging solve. |
| `component_decomposition` | Piecewise trend, seasonality, event, regressor, and related component payload. |
| `metadata` | Cutoff, mode, variogram, fallback, and runtime details. |

Use these fields before making quality claims. A lower aggregate RMSE is weaker
evidence if the correction is dominated by distant neighbors, large kriging
variance, or unstable coordinates.

## Interactive Example

<ForecastModelExample title="Spatial piecewise kriging coordinate-panel forecast" model="spatial_piecewise_kriging" sample="spatial" />

The embedded example uses stable `longitude` and `latitude` columns and shows
spatial diagnostic columns when the model returns them: base forecast, spatial
correction, kriging variance, and neighbor count.

The lab automatically avoids using coordinate and id columns as spatial
regressors. If it finds other numeric columns, it uses `hybrid`; otherwise it
uses `residual_kriging`.

For a quick interactive check, run the embedded forecast and inspect whether the
spatial correction is small, directional, or dominated by high kriging variance.

## Validation

The maintained synthetic panel diagnostic compares this model against
naive, seasonal naive, piecewise linear seasonal, and kriging under the same
rolling-origin split:

```bash
uv run --group dev python scripts/forecasting_benchmark.py \
  --days 180 \
  --horizon 7 \
  --folds 1 \
  --panel-series 6 \
  --output target/spatial_piecewise_kriging_benchmark.json
```

Use [Forecasting Benchmarks](../../benchmarks/forecasting.md) for the maintained
metric table and interpretation. Keep benchmark-specific labels in benchmark
artifacts; reusable model code and public APIs should describe the generic
spatial-temporal behavior.

## Limitations

- The temporal and spatial stages can each be misspecified; inspect both residuals.
- Coordinates, CRS units, and cutoff-safe neighbor data are required.
- Sparse panels may not support stable variogram or neighbor estimates.
- Retain spatial correction only when it improves an external holdout.
