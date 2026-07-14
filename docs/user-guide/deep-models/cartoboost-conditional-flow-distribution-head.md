import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost ConditionalFlowDistributionHead

Use `ConditionalFlowDistributionHead` when a deep model needs joint residual
uncertainty rather than independent quantile bands. Fit it on hidden-state
features and residuals from the upstream model.

## Public Contract

```python
from cartoboost.deep import ConditionalFlowDistributionHead

head = ConditionalFlowDistributionHead(
    quantiles=(0.05, 0.5, 0.95),
    sample_count=64,
)
head.fit(
    residuals_train,
    model_hidden_state=hidden_train,
    horizon_embeddings=horizon_train,
    entity_or_pair_embeddings=entity_embedding_train,
)

prediction = head.predict(model_hidden_state=hidden_holdout, actual=residuals_holdout)
```

`JointHorizonFlowHead` and `ResidualFlowCalibrator` are aliases.

## Browser WASM Example

<DeepModelWasmExample model="ConditionalFlowDistributionHead" />

## Validation

Compare coverage, interval width, pinball loss, and tail calibration against
independent quantile, Gaussian residual, and conformal interval baselines.
