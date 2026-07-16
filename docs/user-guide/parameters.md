# Parameters

This page explains the training controls on `CartoBoostRegressor` and how they
change fitting behavior. Start with defaults, establish a validation split,
and tune only parameters that correspond to the structure in your data.

## Choose Parameters From The Data

Before tuning ranges, decide what claim the model needs to support. In
structured regression work, parameters should usually map to a modeling
question:

| Scientific question | Controls to consider |
| --- | --- |
| Is a dense tabular baseline enough for fare, duration, demand, or residual prediction? | `split_policy=SplitPolicy.AXIS_ONLY` |
| Are pickup/dropoff coordinates or projected x/y values defining spatial boundaries? | `SplitPolicy.STRUCTURED` plus declared spatial-pair schema entries |
| Does hour-of-day, weekday, or season wrap around? | `SplitPolicy.STRUCTURED` plus declared periodic schema entries |
| Are rare memberships, routes, cells, or service-area memberships part of the signal? | `SplitPolicy.STRUCTURED` plus declared sparse-set schema entries |
| Should nearby observations blend across an uncertain boundary? | `fuzzy=True`, `fuzzy_bandwidth`, `fuzzy_kernel` |
| Is the target about median-like behavior, outlier resistance, or asymmetric service risk? | `loss="mae"`, `loss="huber"`, `loss="log_l2"`, or `loss="quantile"` |
| Is a local trend still visible after the tree finds a region or time bucket? | `leaf_predictor="linear"`, `linear_leaf_features` |
| Does domain knowledge require monotone response to a dense feature? | `monotonic_constraints` |

Keep comparisons disciplined: change one family of modeling controls at a time
when possible, and compare against an axis-only CartoBoost baseline plus
LightGBM or XGBoost under the same split and feature set.

## Core Boosting

These parameters control model capacity and shrinkage. They are useful for
ordinary bias/variance tuning after the validation split is fixed.

| Parameter | Default | Notes |
| --- | --- | --- |
| `n_estimators` | `100` | Number of boosting rounds. Must be non-negative. |
| `learning_rate` | `0.05` | Shrinks each tree contribution. Must be finite and positive. |
| `max_depth` | `4` | Maximum tree depth. `0` produces a constant model. |
| `min_samples_leaf` | `20` | Minimum weighted row count per leaf candidate. |
| `min_gain` | `1e-8` | Minimum gain required to split. |
| `random_state` | `None` | Reserved for deterministic APIs; current training paths are deterministic. |
| `n_threads` | `None` | Number of native CPU threads; use it for reproducible scale measurements. |

## Loss

Choose the loss from the estimand. Mean regression is appropriate for many
fare or duration targets, but structured data often contains heavy tails,
dispatch exceptions, and localized service-level questions.

| Parameter | Default | Notes |
| --- | --- | --- |
| `loss` | `"l2"` | Accepts `"l2"`, `"squared_error"`, `"l1"`, `"mae"`, `"absolute_error"`, `"huber"`, `"log_l2"`, `"quantile"`, or `"pinball"`. |
| `quantile_alpha` | `0.5` | Required to be finite and in `(0, 1)` for quantile loss. |
| `huber_delta` | `1.0` | Positive clipping threshold for Huber loss. |
| `log_offset` | `1.0` | Positive offset for `log_l2`. |

`l1`, `huber`, `log_l2`, and quantile loss currently require
`leaf_predictor="constant"`.

## Split policy

`SplitPolicy` is the main CartoBoost structural modeling choice. `AUTO` lets
Rust choose the bounded dense path, `AXIS_ONLY` keeps the baseline exact, and
`STRUCTURED` derives candidates only from declared schema roles.

| Policy | Purpose |
| --- | --- |
| `SplitPolicy.AUTO` | Bounded native dense search selected from fit shape and objective. |
| `SplitPolicy.AXIS_ONLY` | Standard one-feature threshold baseline. |
| `SplitPolicy.STRUCTURED` | Native candidates derived only from periodic, spatial-pair, and sparse-set schema roles. |

Common temporal-spatial policies:

| Problem shape | Suggested policy |
| --- | --- |
| General tabular baseline | `SplitPolicy.AUTO` |
| Exact axis baseline | `SplitPolicy.AXIS_ONLY` |
| Dense location and time | `SplitPolicy.STRUCTURED` with declared roles |
| Route or cell membership | `SplitPolicy.STRUCTURED` with periodic and sparse roles |
| Location plus sparse memberships | `SplitPolicy.STRUCTURED` with spatial-pair and sparse roles |

## Leaves

| Parameter | Default | Notes |
| --- | --- | --- |
| `leaf_predictor` | `"constant"` | Accepts `"constant"` or `"linear"`. |
| `linear_leaf_features` | `None` | Python API currently expects stringified integer feature indices, such as `["0", "2"]`. |
| `l2_regularization` | `1.0` | Ridge penalty for linear leaves. |

Use linear leaves when the tree can find a region or time bucket but the
remaining residual trend inside that region is still approximately linear. For
example, a learned corridor may still have a distance or time-of-day trend
represented locally rather than globally.

## Fuzzy Routing

| Parameter | Default | Notes |
| --- | --- | --- |
| `fuzzy` | `False` | Enables fractional branch assignment during training and weighted prediction recursion. |
| `fuzzy_bandwidth` | `0.0` | Split transition bandwidth. Must be finite and non-negative. |
| `fuzzy_kernel` | `"linear"` | Transition shape. Accepts `"linear"`, `"gaussian"`, `"exponential"`, `"bisquare"`, `"epanechnikov"`, or `"tricube"`. |

Fuzzy routing is not compatible with monotonic constraints.

Use fuzzy routing for temporal-spatial features where nearby values should not
change abruptly at a learned boundary. This is especially relevant when zone
edges, corridor definitions, pickup coordinates, or service areas are noisy
measurements of a continuous process. Set `fuzzy_bandwidth` in the same units
as the feature values, such as projected coordinate units or hours. Use
`fuzzy_kernel="linear"` for simple piecewise interpolation, `"gaussian"` or
`"tricube"` for smoother transitions, and compact-support kernels like
`"bisquare"` or `"epanechnikov"` when you want the blend to drop off faster
near the edge of the band.

## Monotonic Constraints

`monotonic_constraints` is a list of `-1`, `0`, or `1` values with one entry per
dense feature:

- `1` requires predictions to be non-decreasing in that feature.
- `-1` requires predictions to be non-increasing in that feature.
- `0` leaves the feature unconstrained.

Current constraints require constant leaves, non-fuzzy training, and axis-style
split policy. Use `SplitPolicy.STRUCTURED` when the scientific design requires directional behavior,
such as non-decreasing fare with distance after accounting for the rest of the
feature set, and document that constraint in the model artifact or report.
