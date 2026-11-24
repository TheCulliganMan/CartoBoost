---
sidebar_label: Conditional residual distribution
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Conditional Residual Distribution

Generate residual scenarios conditioned on an upstream model's hidden state
with `ConditionalFlowDistributionHead`. This is useful when uncertainty across
forecast horizons matters to a decision, such as planning taxi supply across
the next several hours.

Fit the head on representative residuals and context. The implementation is a
conditional residual sampler, so evaluate its joint scenarios and tail behavior
separately from the point forecast.

## Python Example

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

## Use When

| Your task | Start with |
| --- | --- |
| Joint residual scenarios conditioned on model context. | `ConditionalFlowDistributionHead` |
| Marginal intervals from a separate calibration set. | [Probabilistic and conformal models](../forecasting-models/probabilistic-conformal.md) |
| Graph-wide stress scenarios around a point forecast. | [GeoTemporalDiffusionScenarioModel](cartoboost-geotemporal-diffusion-scenario-model.md) |

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<DeepModelWasmExample model="ConditionalFlowDistributionHead" />

## Validation

Compare coverage, interval width, pinball loss, and tail calibration against
independent quantile, Gaussian residual, and conformal interval baselines.

## Limitations

- Calibration depends on residuals representative of deployment.
- The current architecture is a conditional residual sampler, not an invertible flow.
- Joint scenarios add value only when cross-horizon dependence is validated.
