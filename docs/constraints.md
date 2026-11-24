# Constraints

Encode a relationship your model must respect. A **monotonic constraint** controls
how predictions change as one dense feature increases. An **interaction
constraint** controls which features can appear together along a tree branch.

For taxi fare, a defensible rule might require predictions to increase with trip
distance while other inputs stay fixed. A route model might allow interactions
within location features and within time features, but keep those groups separate.
Choose the rule from the problem, then test it on held-out trips.

| You need to control | Use | Check after fitting |
| --- | --- | --- |
| The direction of a feature's effect | `monotonic_constraints` | Vary that feature alone and inspect predictions. |
| Feature combinations within a branch | `BoosterConfig.interaction_constraints` | Inspect branches for disallowed combinations. |

For the surrounding training workflow, see [Parameters](user-guide/parameters.md)
and [Spatial validation](user-guide/spatial-cv-best-practices.md).

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
