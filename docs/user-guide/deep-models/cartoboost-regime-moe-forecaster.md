import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost RegimeMoEForecaster

Use `RegimeMoEForecaster` when one global model hides materially different
geo-temporal regimes. The public surface exposes six named experts: stable
recurring pattern, sparse cold-start, high-volume hub, volatile shock,
long-distance pair, and low-signal fallback.

## Python Example

```python
from cartoboost.deep import RegimeMoEForecaster

model = RegimeMoEForecaster()
model.fit(
    features_train,
    duration_train,
    entity_ids=pickup_zone_ids,
    time_features=hour_day_features,
    recent_volatility=rolling_duration_volatility,
    graph_centrality=pickup_graph_centrality,
)

parts = model.predict_components(features_holdout, entity_ids=pickup_zone_ids_holdout)
prediction = parts["combined_prediction"]
```

`GeoTemporalMixtureOfExperts`, `PairRegimeRouter`, and `EntityRegimeRouter`
are aliases for this first-cut MoE surface.

## Browser WASM Example

<DeepModelWasmExample model="RegimeMoEForecaster" />

## Use When

Use this model when distinct, repeatable regimes are plausible and expert
usage can be inspected. Prefer one model when the router collapses.

## Validation

Report router entropy, expert usage, combined RMSE, and a single-expert
comparison under the same split. Treat degenerate expert usage as a failed MoE
claim even if the aggregate error is acceptable.

## Limitations

- Mixtures add router instability and expert-identifiability risk.
- Aggregate error can hide unused or redundant experts.
- Regime interpretation requires stability across seeds and cutoffs.
