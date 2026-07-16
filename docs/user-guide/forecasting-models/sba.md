import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# SBA Forecasting

`SbaForecaster` implements the Syntetos-Boylan approximation for intermittent
demand. It starts from Croston-style demand and interval smoothing, then applies
the standard bias adjustment to reduce Croston's upward bias on sparse series.

## Interactive Example

<ForecastModelExample title="SBA sparse-demand forecast" model="sba" />

## Python Example

```python
from cartoboost.forecasting import SbaForecaster

demand = [0, 0, 4, 0, 0, 7, 0, 3, 0, 0, 0, 5]

model = SbaForecaster(alpha=0.2)
model.fit(demand)
forecast = model.predict(6)
```

## Use When

Use SBA when Croston is a reasonable sparse-demand baseline but you want the
bias-adjusted level that usually forecasts below the unadjusted Croston value
for the same `alpha`.

## Fit Contract

| Requirement | Detail |
| --- | --- |
| Target values | Finite and non-negative. |
| Zero values | Treated as true no-demand periods. |
| Non-zero demand | At least one non-zero observation is required. |
| Smoothing | `alpha` must be greater than 0 and at most 1. |

## Validation

SBA is still a fixed local method. Validate it against Croston, TSB, seasonal
naive, and any broader selector under the same cutoff and horizon.

Report the zero fraction, demand-event count, MAE or RMSE, WAPE when defined,
and errors by horizon. Use identical smoothing-selection rules for Croston and SBA.

## Limitations

- SBA produces a flat horizon forecast and has no calendar or covariate model.
- Bias correction does not guarantee lower holdout error for every series.
- Results are unreliable when the history contains too few non-zero events.
