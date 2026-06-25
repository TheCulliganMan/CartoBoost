# Objectives

CartoBoost supports regression objectives for continuous temporal-spatial targets
such as demand, duration, fare, cost, or residual error.

Choose the objective by the scientific estimand. If the study asks for the
expected fare or duration, use an L2 point model and report RMSE/MAE/R2. If the
study asks for a service threshold, risk bound, or conservative planning value,
use quantile regression and report pinball loss plus interval diagnostics.

## L2 Regression

L2 squared error is the default objective:

```python
model = CartoBoostRegressor(loss="l2")
```

Use L2 when your primary score is RMSE, R2, or general point-prediction quality:
for example, expected log fare by pickup/dropoff zone and hour, or expected
duration after controlling for trip distance and airport-lane effects. It
supports the current splitters, sample weights, constant leaves, and linear
leaves.

## Quantile Regression

Quantile regression estimates a conditional quantile instead of the conditional
mean:

```python
model = CartoBoostRegressor(
    loss="quantile",
    quantile_alpha=0.9,
    leaf_predictor="constant",
)
```

Use quantile regression when upper or lower tails matter, such as high-delay
duration risk, high-cost fare estimates, or conservative pickup-demand
forecasts for staffing and dispatch.

Accepted names:

| Objective | Names |
| --- | --- |
| L2 squared error | `"l2"`, `"squared_error"` |
| Quantile pinball | `"quantile"`, `"pinball"` |

`quantile_alpha` must be in `(0, 1)`. Quantile loss currently requires
constant leaves.

The Rust core also includes reusable quantile math primitives for calibrated
forecasting workflows:

| Primitive | Purpose |
| --- | --- |
| `QuantileLoss` | Pinball loss for one quantile. |
| `HuberQuantileLoss` | Smooth pinball-style loss near zero residuals with asymmetric tails. |
| `CompositeQuantileLoss` | Aggregates multiple quantile losses for a shared forecast grid. |
| `DEFAULT_QUANTILE_LEVELS` | Standard p10, p25, p50, p75, and p90 forecast levels. |
| `QuantileRegressorSet` | Fits one Rust `Booster` per requested quantile and emits repaired non-crossing rows. |
| `interval_coverage` | Inclusive empirical interval coverage. |
| `mean_interval_width` | Mean upper-minus-lower interval width. |
| `crossing_rate` | Share of rows with decreasing quantile predictions. |
| `ConformalCalibrator` | Serializable split-conformal residual calibration for interval coverage. |

Use `repair_non_crossing_quantiles` before interval reporting when separately
trained quantile models produce crossed rows. `QuantileRegressorSetConfig`
defaults to the p10, p25, p50, p75, and p90 grid for native forecast bundles.

## Count Objective Helpers

The Rust core includes finite-difference-friendly helpers for count-style
objectives used by native modeling experiments and forecasting work. These
helpers expose objective value, gradient, and Hessian with respect to the raw
score. Count means use a log link, so `mean = exp(raw_score)` after internal
finite clipping.

| Helper | Target use |
| --- | --- |
| `PoissonObjective` | Pickup counts or trip counts where variance is close to the mean. |
| `NegativeBinomialObjective` | Overdispersed pickup/dropoff counts with positive dispersion. |
| `TweedieObjective` | Non-negative compound targets with mass near zero and positive continuous values. |
| `HurdleObjective` | Two-stage zero occurrence plus positive count or severity objective. |

The helpers validate finite non-negative targets and finite raw scores. Tweedie
power must be in `(1, 2)`, and negative-binomial dispersion must be positive.
They do not create benchmark claims by themselves; any public quality claim
still needs a real run with fixed data, settings, baselines, and recorded
metrics.

## Probability Calibration

CartoBoost's Rust core exposes reusable probability primitives for binary event
forecasting:

| Primitive | Purpose |
| --- | --- |
| `LogisticBoostingObjective` | Binary logistic objective alias for probability boosting. |
| `ProbabilityBooster` | Rust-native classifier alias for boosted probability models. |
| `SigmoidCalibrator` | Platt-style sigmoid calibration from raw scores or margins. |
| `TemperatureCalibrator` | Single-temperature scaling for overconfident or underconfident margins. |
| `IsotonicCalibrator` | Monotone nonparametric calibration using pool adjacent violators. |
| `CalibrationMetrics` | Brier score, log loss, ECE, and reliability-curve bucket data. |
| `CalibrationImprovement` | Before/after Brier, log loss, and ECE reductions for calibrated probabilities. |
| `success_within_threshold`, `event_within_horizon`, `failure_risk_event`, `escalation_risk_event` | Binary event-label helpers for calibrated event and risk forecasts. |

Event helpers convert deterministic outcomes into binary calibration targets for
success-within-threshold, event-within-horizon, failure-risk, and escalation-risk
workflows. Calibrators validate finite binary labels and keep emitted
probabilities in `[0, 1]`.

## Evaluation Guidance

- Use RMSE, MAE, and R2 for L2 point prediction.
- Use pinball loss for quantile models.
- Use Brier score, log loss, ECE, and reliability buckets for probability
  forecasts.
- Report the holdout type: random, temporal, spatial, grouped, or route-based.
- Compare objectives on the same split and feature set.
- For temporal-spatial work, include residual summaries by time bucket, zone, or
  taxi zone; aggregate metrics can hide localized failure modes.
