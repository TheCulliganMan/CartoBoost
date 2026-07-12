import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost EventOutcomeModel

Use `EventOutcomeModel` when the target is a binary event and the output should
be a calibrated probability. It is for event risk, conversion, failure,
completion, or acceptance probability when calibration matters.

## Public Contract

```python
from cartoboost.preview.deep import EventOutcomeModel

model = EventOutcomeModel(calibration="temperature")
model.fit(features_train, event_train)

probability = model.predict_proba(features_holdout)
report = model.calibration_report(features_holdout, event_holdout)
model.save("event-outcome.json")
```

## Browser WASM Example

<DeepModelWasmExample model="EventOutcomeModel" />

## When To Use

- The target is binary.
- The probability value matters, not only the class label.
- You need calibration diagnostics such as Brier score.
- A downstream decision threshold will use the predicted probability.

## Use When

| Need | Better first choice |
| --- | --- |
| Calibrated binary event probability. | `EventOutcomeModel` |
| Multiclass labels or class probabilities. | `CartoBoostClassifier` |
| Candidate-specific response curves. | `ResponseCurveModel` |
| Conformal intervals around numeric predictions. | Probabilistic and conformal models |

## Validation

Report Brier score, log loss, ROC-AUC or PR-AUC when appropriate, and
calibration by probability bucket. Use a time, group, or entity split when
deployment will face new periods or entities.
