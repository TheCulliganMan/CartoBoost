# NYC Taxi Benchmarks

## Bottom Line

A full-year current-code run on real 2024 NYC TLC yellow taxi data compares one
primary `cartoboost` row with XGBoost, scikit-learn HistGradientBoosting,
RandomForest, ExtraTrees, Ridge, and a mean baseline under fixed settings.
CartoBoost has lower RMSE than the best finished baseline on duration random,
duration spatial holdout, and pickup-demand random. Ridge is slightly lower on
the current fare random and fare spatial-holdout rows, so the maintained
artifact supports a mixed result rather than a blanket win.

The pickup-demand spatial holdout is intentionally mean-only because the holdout
removes the zone history that learned models need. That split is recorded as a
design limitation, not as a model failure.

This is strong bounded evidence, not a universal claim: it uses one calendar
year, a 100,000-row sampled trip frame, fixed hyperparameters, and local
hardware timing.

## Data

| Field | Value |
| --- | --- |
| Source | NYC TLC trip records |
| Source URL | [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Taxi type | Yellow |
| Period | January 2024 through December 2024 |
| Sample size | 100,000 trip rows |
| Duration rows | 100,000 |
| Fare rows | 100,000 |
| Pickup-demand rows | 38,932 |
| Dataset hash | `7708e1f5350fce2c0de4c431df3d513ca372492888583968fdabeb8ed0b3a328` |
| Zone treatment | Train-only smoothed target-mean zone features for all eligible models |

Raw TLC files stay under `data/nyc_taxi/` and are not committed. The maintained
run used `--no-plots`, so missing local real inputs still hard-fail instead of
silently downgrading the benchmark.

## Reproduce

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-plots \
  --sample-size 100000 \
  --months 1,2,3,4,5,6,7,8,9,10,11,12 \
  --output-dir docs/assets/nyc_taxi_benchmarks \
  --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean \
  --n-estimators 48 \
  --cartoboost-n-estimators 48 \
  --tasks duration,fare,pickup_demand \
  --model-workers 1
```

Generated artifacts:

- `docs/assets/nyc_taxi_benchmarks/results.json`
- `docs/assets/nyc_taxi_benchmarks/results.jsonl`
- `docs/assets/nyc_taxi_benchmarks/results.md`

The JSON and Markdown artifacts record the runtime resource snapshot, baseline
dependency status, split manifest hashes, and output artifact sizes. The current
historical artifact includes a legacy split manifest hash for each task/split;
new runs of `scripts/run_nyc_taxi_quality_benchmarks.py` also persist
`train_index_sha256` and `test_index_sha256` per split. LightGBM and CatBoost
are part of the maintained roster and are expected to run on the same footing as
the other learned baselines in the validated environment.

| Baseline | Package | Import | Version | Importable | Required class | Class available |
| --- | --- | --- | --- | --- | --- | --- |
| sklearn | scikit-learn | sklearn | `1.9.0` | true |  |  |
| xgboost | xgboost | xgboost | `3.3.0` | true | XGBRegressor | true |
| lightgbm | lightgbm | lightgbm | `4.6.0` | true | LGBMRegressor | true |
| catboost | catboost | catboost | `1.2.10` | true | CatBoostRegressor | true |

## Comparison Summary

For each runnable learned-model split, the table compares the single primary
`cartoboost` row with the lowest-RMSE external baseline that finished under the
same task, split, sample, target transformation, and global settings.

| Task / split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Duration / random | 0.321631 | 0.037386 | HistGradientBoosting | 0.337427 | 0.039421 | -0.015796 | +0.021386 | CartoBoost lower RMSE |
| Duration / spatial holdout | 0.317218 | 0.037839 | HistGradientBoosting | 0.330881 | 0.039590 | -0.013663 | +0.021329 | CartoBoost lower RMSE |
| Fare / random | 0.174649 | 0.040675 | Ridge | 0.169843 | 0.038907 | +0.004807 | -0.006014 | Ridge lower RMSE |
| Fare / spatial holdout | 0.159090 | 0.039392 | Ridge | 0.158739 | 0.039154 | +0.000350 | -0.000692 | Ridge lower RMSE |
| Pickup demand / random | 0.598105 | 0.179037 | HistGradientBoosting | 0.631339 | 0.191566 | -0.033234 | +0.009805 | CartoBoost lower RMSE |

The pickup-demand spatial holdout skips learned models because held-out pickup
zones have no training-side demand history. Reporting learned-model scores there
would mostly measure fallback priors.

## What the Rows Mean

| Task / split | Prediction unit | Target | Validation question | Signal used by CartoBoost |
| --- | --- | --- | --- | --- |
| Duration / random | One completed taxi trip | Log trip duration in seconds | Can the model explain ordinary held-out trips drawn from the same year-wide trip distribution? | Trip distance, passenger count, hour/day periodicity, pickup/dropoff zones, and route geometry. |
| Duration / spatial holdout | One completed taxi trip from held-out pickup zones | Log trip duration in seconds | Does the duration structure transfer when pickup zones are held out? | Spatial splitters and route geometry rather than memorizing the validation rows. |
| Fare / random | One completed taxi trip | Log total fare amount | Can the model recover fare structure for ordinary held-out trips? | Distance, pickup/dropoff zones, hour/day effects, and cartometric route features. |
| Fare / spatial holdout | One completed taxi trip from held-out pickup zones | Log total fare amount | Does fare modeling generalize to zones not present in the training pickup set? | Route and zone geometry carry transferable fare signal beyond target-mean zone encodings. |
| Pickup demand / random | Pickup zone x hour x weekday bucket | Log pickup trip count | Can the model explain recurring zone-time demand for observed zones? | Graph-aware zone structure plus hour, weekday, and pickup-zone features. |

## Interpretation

- CartoBoost is the best finished model on the current duration and pickup
  demand comparable splits; Ridge is slightly lower on the current fare rows.
- The comparisons are fair at the benchmark level: same sample, same split
  rules, same fixed estimator budget, and the same feature-access policy.
- LightGBM and CatBoost are included in the maintained roster and finished the
  current full-year run under the same sample, split, and estimator budget as
  the other learned baselines.
- The pickup-demand cold-zone split is intentionally not scored with learned
  models because there is no training-side zone history to learn from.

## Current Artifacts

- [Results JSON](../assets/nyc_taxi_benchmarks/results.json)
- [Results JSONL](../assets/nyc_taxi_benchmarks/results.jsonl)
- [Results report](pathname:///docs/assets/nyc_taxi_benchmarks/results.md)
- [Asset README](pathname:///docs/assets/nyc_taxi_benchmarks/README.md)
