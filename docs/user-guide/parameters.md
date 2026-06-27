# Parameters

This page summarizes the public training controls exposed by
`CartoBoostRegressor`, with emphasis on choosing controls that match the
scientific structure of temporal-spatial regression.

## Choose Parameters From The Question

Before tuning ranges, decide what claim the model needs to support. In
structured regression work, parameters should usually map to a modeling
question:

| Scientific question | Controls to consider |
| --- | --- |
| Is a dense tabular baseline enough for fare, duration, demand, or residual prediction? | `splitters=None`, `["auto"]`, `["axis"]`, or `["axis_histogram:<bins>"]` |
| Are pickup/dropoff coordinates or projected x/y values defining spatial boundaries? | `diagonal_2d`, `gaussian_2d` |
| Does hour-of-day, weekday, or season wrap around? | `periodic:<period>` |
| Are rare memberships, routes, cells, or service-area memberships part of the signal? | `sparse_set` plus `sparse_sets=` |
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
| `n_threads` | `None` | Public parameter retained for compatibility; threaded training control is not currently exposed. |

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

## Splitters

The splitter list is the main CartoBoost modeling choice. By default,
`splitters=None` uses `auto`: exact `axis` on small or constrained fits, and a
fast histogram-axis search for larger dense L2 fits. Use `axis` explicitly when
you need exact threshold search, then add the splitters that match the
scientific structure in the rows.

| Name | Purpose |
| --- | --- |
| `auto` | Default policy that chooses exact axis or histogram-axis training from the fit shape and objective. |
| `axis` | Standard one-feature threshold splits. |
| `axis_histogram`, `axis_hist`, `histogram`, `axis_histogram:<bins>` | Fast axis-threshold search for dense numeric features. |
| `diagonal_2d`, `diagonal2d` | Oblique 2D boundaries for coordinates or projected x/y features. |
| `gaussian_2d`, `gaussian2d`, `radial` | Radial neighborhoods around local hotspots, depots, zones, or corridors. |
| `periodic_time`, `periodic_24`, `periodic:<period>` | Wraparound time features such as hour-of-day, weekday, or seasonal phase. |
| `sparse_set`, `sparse` | List-valued zone, grid, or encoded H3 memberships. |

Unknown splitter names raise `ValueError`.

Common temporal-spatial combinations:

| Problem shape | Suggested splitters |
| --- | --- |
| General tabular baseline | `None` or `["auto"]` |
| Exact axis baseline | `["axis"]` |
| Dense location and time | `["axis", "diagonal_2d", "gaussian_2d", "periodic:24"]` |
| Route or cell membership | `["axis", "periodic:24", "sparse_set"]` |
| Location plus sparse memberships | `["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"]` |

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
splitters. Use them when the scientific design requires directional behavior,
such as non-decreasing fare with distance after accounting for the rest of the
feature set, and document that constraint in the model artifact or report.
