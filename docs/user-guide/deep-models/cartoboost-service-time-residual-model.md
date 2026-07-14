import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost ServiceTimeResidualModel

Use `ServiceTimeResidualModel` when you already have a required baseline
numeric estimate and want CartoBoost to learn the residual correction. The
final prediction is the baseline plus the learned residual.

## Public Contract

```python
from cartoboost.deep import ServiceTimeResidualModel

rows = [
    {
        "baseline_value": 12.0,
        "actual_value": 13.5,
        "features": [0.2, 1.0, 4.0],
    },
    {
        "baseline_value": 9.5,
        "actual_value": 8.9,
        "features": [0.1, 0.0, 3.0],
    },
]

model = ServiceTimeResidualModel()
model.fit(rows)
prediction = model.predict(rows, return_interval=True)
model.save("service-residual.json")
```

## Browser WASM Example

<DeepModelWasmExample model="ServiceTimeResidualModel" />

## When To Use

- A baseline estimate is required by the workflow.
- The model should correct residual error rather than replace the baseline.
- Features explain systematic bias around the baseline.
- Missing baseline values should be treated as data errors.

## Use When

| Need | Better first choice |
| --- | --- |
| Correct a known numeric baseline. | `ServiceTimeResidualModel` |
| Fit a numeric model from raw features only. | `CartoBoostRegressor` |
| Calibrate uncertainty around predictions. | Probabilistic and conformal models |
| Forecast future time-indexed values. | Forecasting model guides |

## Validation

Compare the corrected prediction against the baseline alone on the same split.
Report residual MAE/RMSE and final prediction MAE/RMSE so readers can see
whether the correction is useful or just adding variance.
