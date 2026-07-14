import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Croston Forecasting

`CrostonForecaster` is a fixed intermittent-demand method for sparse,
non-negative series where zero periods are true no-demand observations. It
separately smooths non-zero demand size and the interval between non-zero
events, then returns a flat forecast for the requested horizon.

## Browser WASM Example

<ForecastModelExample title="Croston sparse-demand forecast" model="croston" />

## Python Example

```python
from cartoboost.forecasting import CrostonForecaster

demand = [0, 0, 4, 0, 0, 7, 0, 3, 0, 0, 0, 5]

model = CrostonForecaster(alpha=0.2)
model.fit(demand)
forecast = model.predict(6)
```

Use Croston when you need the basic sparse-demand baseline before testing SBA,
TSB, seasonal naive, or a guarded selector such as `AutoForecaster`.

## Fit Contract

| Requirement | Detail |
| --- | --- |
| Target values | Finite and non-negative. |
| Zero values | Treated as real no-demand periods, not missing rows. |
| Non-zero demand | At least one non-zero observation is required. |
| Smoothing | `alpha` must be greater than 0 and at most 1. |

Croston is transparent, but it is not bias-adjusted. Compare it against
`SbaForecaster` and `TsbForecaster` on the same rolling-origin split before
claiming it is the best sparse-demand model for a panel.
