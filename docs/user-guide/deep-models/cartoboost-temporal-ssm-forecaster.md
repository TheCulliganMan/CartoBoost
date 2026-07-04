import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost TemporalSSMForecaster

Use `TemporalSSMForecaster` when a synchronized entity panel needs a
deterministic selective state-space backbone. The first implementation uses one
public architecture, `selective_ssm`, with CPU execution today and the same
surface reserved for future CUDA, ROCm, and MLX kernels.

## Public Contract

```python
from cartoboost.deep import EntityPanelFrame, TemporalSSMForecaster

frame = EntityPanelFrame(
    entity_ids=["PULocationID:161", "PULocationID:236"],
    timestamps=[0, 1, 2, 3, 4, 5],
    target=[[42, 35], [44, 36], [51, 40], [58, 46], [55, 45], [49, 43]],
    horizon=2,
    frequency="hourly",
)

model = TemporalSSMForecaster(lookback=4, horizon=2, state_dim=8)
model.fit(frame)
forecast = model.predict(2)
```

`EntityTemporalSSM`, `PairTemporalSSM`, and `GraphTemporalSSM` are aliases for
this first-cut selective SSM surface.

## Browser WASM Example

<DeepModelWasmExample model="TemporalSSMForecaster" />

## Validation

Compare against seasonal naive, `CartoBoostLagForecaster`, and a panel neural
forecast under the same rolling split. Report errors by horizon and entity.
