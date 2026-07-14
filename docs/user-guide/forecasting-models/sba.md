import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# SBA Forecasting

`SbaForecaster` implements the Syntetos-Boylan approximation for intermittent
demand. It starts from Croston-style demand and interval smoothing, then applies
the standard bias adjustment to reduce Croston's upward bias on sparse series.

## Browser WASM Example

<ForecastModelExample title="SBA sparse-demand forecast" model="sba" />

## Python Example

```python
from cartoboost.forecasting import SbaForecaster

demand = [0, 0, 4, 0, 0, 7, 0, 3, 0, 0, 0, 5]

model = SbaForecaster(alpha=0.2)
model.fit(demand)
forecast = model.predict(6)
```

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

SBA is still a fixed local method. Validate it against Croston, TSB, seasonal
naive, and any broader selector under the same cutoff and horizon.
