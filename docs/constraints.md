# Constraints

Monotonic constraints force predictions to move in a declared direction as a
dense feature increases. They are useful when the direction is a modeling
requirement, such as price increasing with distance or demand decreasing with
travel time.

Interaction constraints restrict which feature combinations may appear together
along a tree branch. They are useful when a route, corridor, or lane forecast
should allow known families of effects while preventing unsupported cross-family
interactions.

Use constraints as scientific assumptions, not as tuning decoration. A
constraint should be defensible before training and should survive checks on
held-out taxi trips, route groups, and time blocks.

## Usage

`monotonic_constraints` has one entry per dense feature:

| Value | Meaning |
| --- | --- |
| `1` | Prediction must be non-decreasing as the feature increases. |
| `-1` | Prediction must be non-increasing as the feature increases. |
| `0` | Feature is unconstrained. |

```python
model = CartoBoostRegressor(
    split_policy="axis_only",
    monotonic_constraints=[1, 0, -1],
)
```

`BoosterConfig.interaction_constraints` accepts sorted, deduplicated
feature-index groups. During split search, the active branch features plus the
candidate split features must fit inside at least one group. Two-feature
spatial splitters are checked as a pair. Dense features use their matrix column
index; sparse-set columns are addressed after dense columns, so the first
sparse-set column is `dense_feature_count`.

For example, `vec![vec![0, 1], vec![2, 3]]` allows branch interactions within
features 0 and 1 or within features 2 and 3, but blocks a branch that mixes
feature 0 with feature 2.

Monotonic constraint support:

- Constant leaves.
- Non-fuzzy training.
- Axis-style splitters, including histogram-axis splitters.
- Dense features only.
- Regression.

Interaction constraints are enforced in the tree split search for axis,
histogram-axis, diagonal 2D, Gaussian 2D, periodic, dense sparse-set, and
list-valued sparse-set splitters. Groups must be sorted and in range; invalid
groups fail before training.

## Temporal-Spatial Guidance

Use constraints only when the direction is real, not just visually convenient.
Trip distance, elapsed time, toll amount, or known service-level features can
be good candidates. Latitude, longitude, zone ID, and similar location IDs
usually are not: their relationship to the target is often local,
discontinuous, or directional only within a specific market.

For temporal-spatial effects, prefer spatial splitters, periodic splitters,
sparse location features, blocked evaluation, and residual diagnostics unless a
monotonic rule is part of the problem definition.

## Validation

Check more than aggregate RMSE:

- Probe rows that differ only in the constrained feature.
- Check increasing and decreasing constraints separately.
- For interaction groups, inspect trained trees to confirm disallowed feature
  combinations do not appear on the same branch.
- Include tied or nearly tied feature values.
- Use spatial or temporal holdouts when the constrained feature is correlated
  with cartography, route, or time.
