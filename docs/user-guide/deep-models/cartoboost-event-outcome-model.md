---
sidebar_label: Event outcome model
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Event Outcome Model

Estimate the probability of a binary event with `EventOutcomeModel`. Use it
for a taxi-trip outcome such as exceeding a duration threshold when the
probability itself drives a decision. The model exposes calibration reports
alongside predictions so you can check how often predicted events occur.

## Python Example

```python
from cartoboost.deep import EventOutcomeModel

model = EventOutcomeModel(calibration="temperature")
model.fit(features_train, event_train)

probability = model.predict_proba(features_holdout)
report = model.calibration_report(features_holdout, event_holdout)
model.save("event-outcome.json")
```

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

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

## Limitations

- Calibration can drift when the event rate changes.
- A good ranking metric does not guarantee calibrated probabilities.
- Binary outcomes do not represent competing risks or time-to-event behavior.
