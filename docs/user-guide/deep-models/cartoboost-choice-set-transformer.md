import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost ChoiceSetTransformer

Use `ChoiceSetTransformer` when candidates compete within each decision set.
It scores candidate value, candidate features, context features, optional
entity or pair embeddings, and existing utility or probability fields, then
returns utilities and softmax choice probabilities.

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

Use this model when candidates compete within a decision set and probabilities
must sum within that set. Use `CartoBoostRanker` when only relative ordering is needed.

## Browser WASM Example

<DeepModelWasmExample model="ChoiceSetTransformer" />

## Validation

Use grouped validation by decision id. Report Brier score or ECE when chosen
labels exist, and compare the selected candidate against simple rule baselines.

## Limitations

- Candidate-set composition affects every predicted probability.
- Unobserved alternatives and biased choice sets can invalidate interpretation.
- The current utility-softmax architecture is not candidate-to-candidate attention.
