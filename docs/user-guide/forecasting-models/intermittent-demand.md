import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Intermittent Demand

`CrostonForecaster`, `SbaForecaster`, and `TsbForecaster` are fixed
intermittent-demand methods for sparse non-negative series. Use them when zero
values are real demand periods, not missing observations.

The Rust `IntermittentDemandForecaster` is the corresponding selector. It
compares an explicit zero baseline, Croston, SBA, TSB, and ADIDA when the ADIDA
bucket is supported by the training prefix.

## When To Use

- The target is non-negative.
- Many periods are true zeros.
- Demand size and demand occurrence are more useful than smooth trend.
- You need a transparent sparse-demand baseline before using a richer selector.

Do not use these methods for missing-row problems. Fill or validate the time
index first with `ForecastFrame`, then model true zero demand.

## Pick A Method

| Model | Use when |
| --- | --- |
| [`CrostonForecaster`](croston.md) | Demand is intermittent and you want the basic Croston decomposition. |
| [`SbaForecaster`](sba.md) | You want Croston-style smoothing with SBA bias adjustment. |
| [`TsbForecaster`](tsb.md) | Occurrence probability and demand size should be smoothed separately. |

The selector always keeps a real non-empty suffix out of fitting. An explicit
validation window that consumes the full history is rejected. All-zero series
use the zero-demand boundary model; they are not replaced by a positive demand
estimate.

ADIDA aggregates only complete buckets. Buckets are aligned from the most
recent observation backward, so an incomplete leading fragment is excluded
instead of being treated as a shorter, incomparable bucket. A history shorter
than one configured bucket fails explicitly.

## Public Contract

```python
from cartoboost.forecasting import CrostonForecaster, SbaForecaster, TsbForecaster

demand = [0, 0, 4, 0, 0, 7, 0, 3, 0, 0, 0, 5]

croston = CrostonForecaster(alpha=0.2).fit(demand)
sba = SbaForecaster(alpha=0.2).fit(demand)
tsb = TsbForecaster(alpha_demand=0.2, alpha_probability=0.1).fit(demand)

croston_forecast = croston.predict(6)
sba_forecast = sba.predict(6)
tsb_forecast = tsb.predict(6)
```

## Browser WASM Example

<ForecastModelExample title="Intermittent demand browser forecast" model="intermittent_demand" />

<ForecastModelExample title="Croston browser forecast" model="croston" />

<ForecastModelExample title="SBA browser forecast" model="sba" />

<ForecastModelExample title="TSB browser forecast" model="tsb" />

## Use When

| Situation | Better first choice |
| --- | --- |
| Many periods are true zero demand. | `CrostonForecaster`, `SbaForecaster`, or `TsbForecaster` |
| Rows are missing rather than true zero demand. | Fix the time index before modeling. |
| Demand is dense and seasonal. | `SeasonalNaiveForecaster`, `AutoStatsBank`, or `AutoForecaster` |
| Sparse demand should be selected automatically inside a broader panel roster. | `AutoForecaster` |

## Panel Fit

```python
from cartoboost.forecasting import ForecastFrame, TsbForecaster

frame = ForecastFrame.from_pandas(
    sparse_panel,
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="sku_id",
    freq="D",
)

model = TsbForecaster(alpha_demand=0.2, alpha_probability=0.1)
model.fit(frame)
forecast = model.predict(14)
```

## Validation

Compare intermittent-demand methods against naive, seasonal naive, and any
selector that includes intermittent candidates. Report zero fraction,
non-negative target validation, internal holdout length, ADIDA bucket size,
horizon metrics, and whether zeros represent true no-demand periods.
