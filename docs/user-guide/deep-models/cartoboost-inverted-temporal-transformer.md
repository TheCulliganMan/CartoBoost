import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost InvertedTemporalTransformer

Use `InvertedTemporalTransformer` for wide synchronized panels where entities
are the attention tokens. This avoids treating every time step as an attention
token and reports cross-entity ablations so the graph-free entity interaction
claim can be checked.

## Public Contract

```python
from cartoboost.deep import EntityPanelFrame, InvertedTemporalTransformer

frame = EntityPanelFrame(
    entity_ids=["PULocationID:161", "PULocationID:236", "PULocationID:132"],
    timestamps=[0, 1, 2, 3, 4, 5],
    target=[
        [42, 35, 18],
        [44, 36, 19],
        [51, 40, 24],
        [58, 46, 31],
        [55, 45, 34],
        [49, 43, 30],
    ],
    horizon=2,
    frequency="hourly",
)

model = InvertedTemporalTransformer(lookback=4, horizon=2)
model.fit(frame)
forecast = model.predict(2)
```

`InvertedEntityTransformer` is an alias. The same implementation is reachable
through `TemporalEntityTransformer(architecture="inverted_transformer")`.

## Browser WASM Example

<DeepModelWasmExample model="InvertedTemporalTransformer" />

## Validation

Report horizon-wise error and an ablation that removes cross-entity attention.
Use this only when synchronized entities are the modeling unit.
