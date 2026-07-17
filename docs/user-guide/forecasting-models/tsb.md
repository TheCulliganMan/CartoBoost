import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# TSB Forecasting

`TsbForecaster` implements the Teunter-Syntetos-Babai intermittent-demand
method. It smooths demand size and demand occurrence probability separately,
which makes it useful when the chance of a non-zero event changes over time.

## Interactive Example

<ForecastModelExample title="TSB sparse-demand forecast" model="tsb" />

## Python Example

```python
from cartoboost.forecasting import TsbForecaster

demand = [0, 0, 4, 0, 0, 7, 0, 3, 0, 0, 0, 5]

model = TsbForecaster(alpha_demand=0.2, alpha_probability=0.1)
model.fit(demand)
forecast = model.predict(6)
```

## Use When

Use TSB when zeros are real no-demand periods and the event probability itself
is part of the signal, not just the spacing between past non-zero events.

## Fit Contract

| Requirement | Detail |
| --- | --- |
| Target values | Finite and non-negative. |
| Zero values | Treated as true no-demand periods. |
| Non-zero demand | At least one non-zero observation is required. |
| Demand smoothing | `alpha_demand` must be greater than 0 and at most 1. |
| Probability smoothing | `alpha_probability` must be greater than 0 and at most 1. |

## Validation

TSB can react differently from Croston and SBA when recent demand occurrence
changes. Report the zero fraction and compare all three methods on the same
rolling-origin split.

Tune both smoothing parameters using training-side rolling origins only. Report
errors by horizon and separately inspect periods after demand occurrence changes.

## Limitations

- TSB does not directly model seasonality, covariates, or related series.
- Two smoothing parameters increase selection risk on short histories.
- Zero values caused by missing data or supply constraints need preprocessing and a different interpretation.
