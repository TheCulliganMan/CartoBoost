import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost GraphNeuralOperator

Use `GraphNeuralOperator` for advanced experimental field-to-field mapping on
regional or gridded panels. It consumes field values, coordinates, graph edges,
and optional exogenous fields, then returns future, residual, and uncertainty
fields.

## Public Contract

```python
from cartoboost.preview.deep import GraphNeuralOperator

operator = GraphNeuralOperator(smoothing=0.25, coordinate_scale=0.1)
prediction = operator.predict(
    field_values=[[42, 35, 18], [44, 36, 19], [51, 40, 24]],
    coordinates=[[0.0, 0.0], [0.5, 0.4], [1.0, 0.1]],
    edges=[{"source": 0, "target": 1, "weight": 0.7}],
    exogenous_fields=[[0.1, 0.2, 0.0], [0.1, 0.3, 0.1], [0.2, 0.2, 0.1]],
)
benchmark = GraphNeuralOperator.synthetic_benchmark()
```

`FourierGeoOperator` and `SpatioTemporalOperator` are aliases.

## Browser WASM Example

<DeepModelWasmExample model="GraphNeuralOperator" />

## Validation

Report the maintained synthetic benchmark and compare against a pointwise MLP
proxy before making any field-transfer claim.
