# Lane-Level Acceptance

Inspect how spatial, temporal, and sparse features work together on a controlled
taxi-lane dataset. The fixture contains 16 origin–destination lanes, each with
24 hourly observations, and exposes only observable features to the model.

These are deterministic synthetic checks. Use the [taxi zone result table](taxi-zone.md)
for measured errors and the [NYC taxi benchmarks](nyc-taxi.md) for real-trip
comparisons.

## Command

Install from PyPI for normal use:

```sh
uv add cartoboost
```

From a source checkout, rebuild the local native extension before regenerating
the maintained acceptance artifacts:

```sh
uv run --group dev maturin develop
uv run --group dev python scripts/run_lane_level_acceptance_metrics.py
```

## Dataset Shape

- 4 origin regions x 4 destination regions = 16 lanes.
- 24 hourly observations per lane.
- Observable columns: origin x/y, destination x/y, lane ID, hour, midpoint x/y,
  and distance.
- No hidden simulator metadata is passed into training.

## Outputs

Generated outputs live under `docs/assets/lane_level_tests/`:

- `acceptance_metrics.json`
- `acceptance_metrics.md`
- `route_midpoint_geometry.png`
- `hour_profile.png`
- `lane_heatmap.png`

Use these files to inspect route cartometry, hour effects, lane-level residuals,
and combined split behavior.

## What The Check Proves

The lane-level dataset is intended to show whether CartoBoost captures:

- Route-cell sparse-set encoding.
- Temporal profile behavior.
- Spatial route cartometry behavior.
- Combined split behavior when several feature families are present.

The fixture isolates these mechanisms by construction. It does not measure
accuracy on real trips or compare CartoBoost with external production baselines.
