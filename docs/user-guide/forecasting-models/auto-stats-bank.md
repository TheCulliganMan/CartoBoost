import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# AutoStatsBank

`AutoStatsBank` validates a fixed bank of local statistical forecasting
experts. Use it when you want deterministic model selection among reusable
classical candidates without moving to the broader `AutoForecaster` panel
selector.

## When To Use

- The forecast question is local to each series.
- You want validation-based selection across statistical experts.
- The model should stay deterministic and easy to audit.
- You need a stronger local baseline before trying lagged or neural panel
  models.

Use `AutoStatsBank` as a selection layer, not as proof that a complex model is
needed. Keep naive and seasonal naive in the comparison table.

## Public Contract

```python
from cartoboost.forecasting import AutoStatsBank, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_demand,
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
)

model = AutoStatsBank(
    season_length=24,
    validation_window=12,
)
model.fit(frame)
forecast = model.predict(12)
```

## Browser WASM Example

<ForecastModelExample title="AutoStatsBank browser forecast" model="autostats_bank" />

## Use When

| Situation | Better first choice |
| --- | --- |
| You need a transparent last-value or seasonal baseline. | `NaiveForecaster` or `SeasonalNaiveForecaster` |
| You want one selected local statistical expert. | `AutoStatsBank` |
| You need lag features shared across many panels. | `CartoBoostLagForecaster` |
| You need guarded panel selection across lag, direct, intermittent, and classical candidates. | `AutoForecaster` |

## Validation

`AutoStatsBank` always reserves a non-empty suffix and fits every eligible
expert on the earlier prefix. An explicit `validation_window` that leaves no
training history is rejected rather than capped. Seasonal and window experts
enter the roster only when the prefix satisfies their full configured history
requirements; explicitly configured experts that fail to fit or omit a
validation prediction produce an error instead of a partial score table.

Evaluate the selected bank under the same rolling-origin split as the
individual local forecasters. Report the selected candidate, internal
validation window, external test horizon, RMSE, MAE, WAPE, train time,
prediction time, and the simple baselines it had to beat.
