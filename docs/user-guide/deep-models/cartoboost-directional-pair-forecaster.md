# CartoBoost DirectionalPairForecaster

Use `DirectionalPairForecaster` when each row is an ordered source-target pair
and reversing the pair changes the meaning. Good examples include origin-to-
destination flows, sender-to-receiver activity, upstream-to-downstream sensors,
or account-to-account interactions.

## Public Contract

```python
from cartoboost.deep import DirectionalPairForecaster, DirectionalPairFrame

frame = DirectionalPairFrame.from_pandas(
    pair_history,
    timestamp_col="timestamp",
    source_col="source_id",
    target_col="target_id",
    target_value_col="observed_value",
    numeric_covariates=["distance", "baseline_estimate", "hour"],
)

model = DirectionalPairForecaster(
    lookback=28,
    horizon=7,
    backbone="residual_mlp",
)
model.fit(frame)
prediction = model.predict(frame)
score = model.score(frame)
```

## When To Use

- Direction is part of the unit being modeled.
- Source and target ids repeat across rows.
- You have pair-level numeric covariates available at prediction time.
- You need a pair-specific model rather than a generic row-level regressor.

## Use When

| Need | Better first choice |
| --- | --- |
| Ordered source-target rows. | `DirectionalPairForecaster` |
| Directed graph sequence forecasting. | `SpatioTemporalGraphForecaster` |
| Directed graph embeddings for row models. | Graph model guides |
| Ordinary numeric row prediction. | `CartoBoostRegressor` |

## Validation

Report temporal splits and cold-pair splits separately. If the holdout contains
source-target pairs not seen during training, describe that as cold-pair
generalization rather than repeated-pair scoring.
