import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost ConstrainedDecisionOptimizer

Use `ConstrainedDecisionOptimizer` when each decision group has several scored
candidates and the output must select one feasible candidate per group. It is
for decision selection after prediction, not for fitting the predictor itself.

## Python Example

```python
from cartoboost.deep import ConstrainedDecisionOptimizer

candidate_rows = [
    {
        "decision_id": "order-1",
        "candidate_id": "candidate-a",
        "expected_utility": 0.82,
        "response_probability": 0.76,
        "risk_score": 0.08,
    },
    {
        "decision_id": "order-1",
        "candidate_id": "candidate-b",
        "expected_utility": 0.91,
        "response_probability": 0.63,
        "risk_score": 0.05,
    },
]

optimizer = ConstrainedDecisionOptimizer(
    objective="risk_adjusted_utility",
    constraints={
        "min_response_probability": 0.7,
        "max_risk_score": 0.15,
    },
    fallback="raise",
)
choice = optimizer.select(candidate_rows)
```

## Browser WASM Example

<DeepModelWasmExample model="ConstrainedDecisionOptimizer" />

## When To Use

- Candidate rows are already scored by a model or rule.
- Constraints are hard requirements, not soft preferences.
- Each decision group should return one selected candidate.
- You need reason codes for selected or rejected candidates.

## Use When

| Need | Better first choice |
| --- | --- |
| Select one feasible candidate per decision group. | `ConstrainedDecisionOptimizer` |
| Learn candidate response curves. | `ResponseCurveModel` |
| Predict calibrated event probability for each row. | `EventOutcomeModel` |
| Rank all candidates without hard constraints. | `CartoBoostRanker` |

## Validation

Report constraint violation rate, selected utility, fallback rate, and the
baseline rule it replaces. Keep the scorer validation separate from optimizer
validation so model error and decision policy behavior are both visible.

## Limitations

- Optimization quality is bounded by scorer quality and candidate coverage.
- Infeasible groups require an explicit fallback policy.
- Offline utility may not capture operational side effects or delayed outcomes.
