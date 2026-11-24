# Taxi Zone Feature Checks

Sparse memberships recover the designed lane effect exactly, periodic splits
connect hours across midnight, and the combined model reduces synthetic holdout
RMSE from **12.68 to 4.86**. The table below separates each feature check.

This deterministic taxi-lane fixture measures whether the model can express
known patterns. The comparison is against axis-based feature variants, not
external modeling libraries. See [lane-level acceptance](lane-level.md) for the
dataset shape and [NYC taxi results](nyc-taxi.md) for real-data evidence.

## Reproduce

```sh
uv run python scripts/run_lane_level_acceptance_metrics.py \
  --output-dir docs/assets/lane_level_tests
```

Artifacts:

- `docs/assets/lane_level_tests/acceptance_metrics.json`
- `docs/assets/lane_level_tests/acceptance_metrics.md`
- `docs/assets/lane_level_tests/lane_heatmap.png`
- `docs/assets/lane_level_tests/hour_profile.png`
- `docs/assets/lane_level_tests/route_midpoint_geometry.png`

## Results

| Feature check | Main result | Outcome |
| --- | ---: | --- |
| Sparse lane membership | Sparse lane RMSE `0.0`; axis lane-ID RMSE `51.96` | Passed: exact sparse lane fit. |
| Hot lane isolation | Hot lane margin `240.0` | Passed: margin > 200. |
| Route midpoint cartometry | Gaussian midpoint RMSE `0.0`; axis midpoint RMSE `69.40` | Passed: Gaussian route RMSE < half axis baseline. |
| Periodic hour wraparound | Periodic hour RMSE `~0`; axis hour RMSE `31.58` | Passed: 23:00 and 01:00 edge gap `0.0`. |
| Combined lane/spatial/temporal | Holdout RMSE drops from `12.68` to `4.86` | Passed: full model / axis ratio `0.383`. |

## Plots

![Lane heatmap](../assets/lane_level_tests/lane_heatmap.png)

![Hour profile](../assets/lane_level_tests/hour_profile.png)

![Route midpoint geometry](../assets/lane_level_tests/route_midpoint_geometry.png)

## Interpretation

These results show that the implementation can represent each targeted feature
family on a controlled taxi-lane fixture. They are useful for diagnosing model
behavior before running a real-data benchmark.

Passing does not mean CartoBoost is more accurate than LightGBM, XGBoost, or a
forecasting library on real data.

