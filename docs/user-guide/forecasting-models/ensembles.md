# Weighted Ensembles

`WeightedEnsembleForecaster` combines native CartoBoost forecasting wrappers
with explicit fixed weights. The ensemble is intentionally simple: each member
is fitted, each member predicts the same horizon, and the final forecast is the
weighted average of aligned predictions.

## Try Candidate Members In The Modeling Lab

Open the browser-local roster comparison with the bundled taxi lane sample:
[Candidate roster comparison](/modeling-lab?sample=lane&model=seasonal_naive&run=compare).
The browser lab does not expose a fixed-weight ensemble as a separate model;
use it to compare candidate members such as seasonal naive, theta, ETS, ARIMA,
and Kalman on the same data before choosing ensemble weights in Python.

## When To Use

Use an ensemble when different models capture different parts of a taxi demand
pattern. For example, seasonal naive can preserve repeated hourly cycles, theta
can adapt to a changing level, and Kalman can update a noisy level/trend from
recent observations.

Do not use an ensemble just to add complexity. It should clear the best
individual member on a fixed validation split or encode a clear operating policy
such as "mostly seasonal, with a smaller trend correction."

## Scientific Role

A weighted ensemble is a fixed mixture of scientific hypotheses. Each component
should represent a different defensible mechanism, such as persistence,
seasonal repetition, smooth trend, state-space updating, spatial borrowing, or
shared supervised lag structure. The ensemble is useful only when those
mechanisms make complementary errors under the same validation design.

Choose it when validation shows that no single component dominates all horizons
or all taxi zones, and when the selected weights can be explained. The weights
are part of the model claim; they are not learned automatically by the native
ensemble wrapper.

## Assumptions And Failure Modes

The ensemble assumes component predictions are aligned to the same series ids,
timestamps, and horizons. It cannot create a signal that none of its members
learned. If every component misses a rush-hour disruption, averaging will miss
it too.

Failure modes include keeping a weak member because one split moved by chance,
changing component parameters while tuning weights, or reporting only ensemble
metrics without component metrics. Compare against the best individual member
and inspect horizon-specific errors before claiming that averaging adds
scientific value.

## Example

```python
from cartoboost.forecasting import (
    KalmanForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)

models = {
    "seasonal": SeasonalNaiveForecaster(season_length=24),
    "theta": ThetaForecaster(theta=2.0, alpha=0.2),
    "kalman": KalmanForecaster(),
}

ensemble = WeightedEnsembleForecaster(
    models=models,
    weights={
        "seasonal": 0.55,
        "theta": 0.30,
        "kalman": 0.15,
    },
    metadata={"purpose": "pickup-zone demand baseline"},
)

ensemble.fit(hourly_pickups)
forecast = ensemble.predict(12)
print(ensemble.get_metadata())
print(forecast.predictions())
```

Weights are normalized by the native ensemble, so `{1.0, 3.0}` becomes
`{0.25, 0.75}` in metadata.

## ForecastFrame Example

```python
from cartoboost.forecasting import ForecastFrame, KalmanForecaster
from cartoboost.forecasting import SeasonalNaiveForecaster, ThetaForecaster
from cartoboost.forecasting import WeightedEnsembleForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("PULocationID == 161"),
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    freq="h",
)

ensemble = WeightedEnsembleForecaster(
    models={
        "seasonal": SeasonalNaiveForecaster(season_length=24),
        "theta": ThetaForecaster(),
        "kalman": KalmanForecaster(),
    },
    weights={"seasonal": 0.6, "theta": 0.3, "kalman": 0.1},
)
ensemble.fit(frame)
forecast = ensemble.predict(24)
```

## Runnable Visual Example

Run the committed example to compare component forecasts against the weighted
blend for taxi airport lanes:

```bash
uv run python examples/forecasting/weighted_ensemble_visualization.py
```

It writes `target/examples/weighted_ensemble.png` and prints JSON with component
weights plus RMSE and MAE for each component and the ensemble. The example uses
synthetic hourly pickup counts and does not download data.

The core pattern is:

```python
from cartoboost.forecasting import (
    ForecastFrame,
    KalmanForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)

frame = ForecastFrame.from_pandas(
    train,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="lane_id",
    freq="h",
)

ensemble = WeightedEnsembleForecaster(
    models={
        "seasonal": SeasonalNaiveForecaster(season_length=24),
        "theta": ThetaForecaster(theta=2.0, alpha=0.25),
        "kalman": KalmanForecaster(),
    },
    weights={"seasonal": 0.55, "theta": 0.30, "kalman": 0.15},
)
ensemble.fit(frame)
forecast = ensemble.predict(12)
```

## Weight Interpretation

Weights should be chosen from validation results or a fixed policy. They are not
learned by `WeightedEnsembleForecaster`.

| Weight pattern | Interpretation | Risk |
| --- | --- | --- |
| High seasonal weight | Same-hour history is the main signal. | Underreacts to airport surges or disruptions. |
| High theta weight | Smooth trend and level adaptation matter. | Can miss sharp intra-day seasonality. |
| High Kalman weight | Recent level and trend should adapt smoothly. | Can underweight repeated hourly cycles. |
| Near-equal weights | Components have similar validation strength. | May hide that one member is consistently worse. |

Record the normalized weights with every benchmark result. A weighted ensemble
without its member list and weights is not reproducible.

## Choosing Weights

A pragmatic workflow is:

1. Score each component on the same rolling-origin splits.
2. Remove members that are consistently worse than a simple seasonal baseline.
3. Try a small weight grid such as seasonal-heavy, trend-heavy, and balanced.
4. Pick the simplest blend that clears the best individual member on average and
   does not fail badly on any taxi zone or lane segment.

Example grid:

```python
candidate_weights = [
    {"seasonal": 0.70, "theta": 0.20, "kalman": 0.10},
    {"seasonal": 0.55, "theta": 0.30, "kalman": 0.15},
    {"seasonal": 0.40, "theta": 0.45, "kalman": 0.15},
]
```

Keep the component model settings fixed while comparing weight grids. Changing
both the member parameters and the weights at the same time makes the result
hard to explain.

## Rules

| Rule | Details |
| --- | --- |
| At least one model is required | Empty `models` raises `ValueError`. |
| Weights must match model names exactly | Missing or extra names raise `ValueError`. |
| Components must be native wrappers | Arbitrary Python estimators are not accepted. |
| Prediction intervals are not supported yet | Interval arguments raise `NotImplementedError`. |

Supported native ensemble members currently include naive, seasonal naive,
theta, optimized theta, ETS, ARIMA, AutoARIMA, Kalman, and
`CartoBoostLagForecaster`.

## Visual Diagnostics

Plot the observed taxi series with every component and the ensemble. Useful
patterns:

| Visual pattern | Meaning | Typical next step |
| --- | --- | --- |
| Ensemble sits between two plausible members. | Blend is behaving as expected. | Validate the average RMSE or MAE. |
| Ensemble follows a visibly bad member. | That member has too much weight. | Lower the weight or remove the member. |
| All members miss the same rush-hour turn. | The ensemble cannot create a signal no member learned. | Add a model with the missing calendar, lag, or event behavior. |
| Ensemble is smoother but less accurate. | Averaging reduced variance but added bias. | Compare by horizon and by lane before keeping the blend. |

## Validation Notes

Report every component model and its weight. Compare the ensemble against the
best individual component, not only against the weakest baseline.

For taxi benchmarks, include:

- component model classes and parameters;
- normalized weights;
- split timestamps and horizon;
- per-component and ensemble RMSE and MAE;
- per-zone or per-lane error summaries when the frame is a panel;
- training and prediction time when making performance claims.

Use rolling-origin splits and keep the exact train/test rows fixed across all
members. If the ensemble only passes one horizon but misses most others,
document that horizon-specific behavior instead of presenting it as general.
