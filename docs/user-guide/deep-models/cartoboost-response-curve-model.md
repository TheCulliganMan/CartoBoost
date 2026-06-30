import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost ResponseCurveModel

Use `ResponseCurveModel` when rows contain context features, candidate values,
and observed responses, and you want to estimate how response changes as the
candidate value changes. This is useful for price, threshold, bid, dose, offer,
or service-level curves.

## Public Contract

```python
from cartoboost.deep import ResponseCurveFrame, ResponseCurveModel

frame = ResponseCurveFrame.from_pandas(
    candidates,
    feature_cols=["region_feature", "time_feature", "entity_feature"],
    candidate_value_col="candidate_value",
    response_col="response",
    group_col="decision_id",
    candidate_id_col="candidate_id",
)

model = ResponseCurveModel(
    response_type="binary",
    monotone="decreasing",
    calibration="isotonic",
    backend="cpu",
)
model.fit(frame)

curve = model.predict_curve(frame)
response = model.predict_response(frame)
best = model.best_candidate(frame)
```

## Browser WASM Example

<DeepModelWasmExample model="ResponseCurveModel" />

## When To Use

- Candidate value is a controlled input, not just another feature.
- You need a curve or best candidate, not only one prediction per row.
- The candidate effect should be monotone increasing or decreasing.
- Candidate groups represent decisions that should be compared together.

## Use When

| Need | Better first choice |
| --- | --- |
| Candidate response curves. | `ResponseCurveModel` |
| Calibrated binary probability without a candidate curve. | `EventOutcomeModel` |
| Select one feasible candidate from scored rows. | `ConstrainedDecisionOptimizer` |
| Generic row-level regression. | `CartoBoostRegressor` |

## Validation

Use grouped validation so candidates from the same decision do not leak across
train and holdout. Report response metrics and whether the selected candidate
improves the decision metric against simple candidate rules.
