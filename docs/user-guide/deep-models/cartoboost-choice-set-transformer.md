---
sidebar_label: Choice-set transformer
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Choice-Set Transformer

Score competing candidates together with `ChoiceSetTransformer`. Use it when
a taxi dispatch decision needs utilities and probabilities over its available
pickup choices, including an optional outside option.

The scorer combines candidate values, candidate and context features, optional
entity or pair embeddings, and existing utility or probability fields. Its
current architecture is utility scoring followed by softmax; the class name
does not imply candidate-to-candidate attention.

## Python Example

```python
from cartoboost.deep import ChoiceSetTransformer

report = ChoiceSetTransformer(
    temperature=0.85,
    monotone_candidate_value="decreasing",
    outside_option=True,
).score(candidate_rows)

probabilities = report["predictions"]
best_by_decision = report["counterfactual_best"]
```

`UtilityNet`, `NestedChoiceHead`, and `CounterfactualCandidateScorer` are
aliases for this surface.

## Use When

| Your task | Start with |
| --- | --- |
| Probabilities across competing pickup candidates. | `ChoiceSetTransformer` |
| Only the ordering within a query. | [CartoBoost Ranker](../boosting-models/cartoboost-ranker.md) |
| A feasible choice under hard constraints. | [ConstrainedDecisionOptimizer](cartoboost-constrained-decision-optimizer.md) |

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<DeepModelWasmExample model="ChoiceSetTransformer" />

## Validation

Use grouped validation by decision id. Report Brier score or ECE when chosen
labels exist, and compare the selected candidate against simple rule baselines.

## Limitations

- Candidate-set composition affects every predicted probability.
- Unobserved alternatives and biased choice sets can invalidate interpretation.
- The current utility-softmax architecture is not candidate-to-candidate attention.
