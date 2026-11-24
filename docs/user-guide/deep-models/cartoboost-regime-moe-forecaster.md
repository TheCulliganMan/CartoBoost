---
sidebar_label: Mixture-of-experts forecaster
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Regime Mixture-of-Experts Forecaster

Let different experts handle recurring patterns, sparse routes, busy hubs,
shocks, long-distance pairs, and low-signal observations with
`RegimeMoEForecaster`. Inspect both the combined prediction and expert usage:
a useful mixture needs distinct, repeatable behavior across its regimes.

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
are aliases for the same mixture-of-experts model. Training stores the reference
mean and first-feature median used by the experts, so a row has the same prediction
alone or in a batch. Supply recent residuals explicitly when using that router input;
training targets are never used as prediction-time features.

Artifacts preserve the fitted references and backend. Legacy artifacts load with
a warning and retain their original batch-dependent behavior; refit them to use
batch-independent predictions. `train_metrics` reports the fitted model's actual
training predictions. Use held-out data to compare forecast quality.

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<DeepModelWasmExample model="RegimeMoEForecaster" />

## Use When

| Your task | Start with |
| --- | --- |
| Distinct recurring regimes and inspectable expert usage. | `RegimeMoEForecaster` |
| A single model explains row-level variation. | [CartoBoost Regressor](../boosting-models/cartoboost-regressor.md) |
| Repeated ordered pickup/dropoff pairs. | [DirectionalPairForecaster](cartoboost-directional-pair-forecaster.md) |

## Validation

Report router entropy, expert usage, combined RMSE, and a single-expert
comparison under the same split. Treat degenerate expert usage as a failed MoE
claim even if the aggregate error is acceptable.

## Limitations

- Mixtures add router instability and expert-identifiability risk.
- Aggregate error can hide unused or redundant experts.
- Regime interpretation requires stability across seeds and cutoffs.
