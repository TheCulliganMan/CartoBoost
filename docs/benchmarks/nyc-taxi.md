# NYC Taxi Benchmarks

## Bottom Line

The maintained JSON records a bounded real-data run on 2024 NYC TLC yellow taxi
data. It compares one primary `cartoboost` row with XGBoost, LightGBM, CatBoost,
and scikit-learn HistGradientBoosting under fixed settings. This current 50,000-
row, 24-tree qualification run includes random, spatial-holdout, and temporal
holdouts for duration and fare. Lane-demand evidence is reported separately in
the maintained forecasting artifact; neither artifact claims an AutoGeo selector
result because that selector is not shipped.

This is bounded evidence, not a universal claim: it uses four months, a
50,000-row sampled trip frame, fixed hyperparameters, and local hardware
timing.

## Data

| Field | Value |
| --- | --- |
| Source | NYC TLC trip records |
| Source URL | [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Taxi type | Yellow |
| Period | January 2024 through April 2024 |
| Sample size | 50,000 trip rows |
| Duration rows | 50,000 |
| Fare rows | 50,000 |
| Dataset hash | `see maintained results.json` |
| Zone treatment | Train-only smoothed target-mean zone features for all eligible models |

Raw TLC files stay under `data/nyc_taxi/` and are not committed. The maintained
run used `--no-plots`, so missing local real inputs still hard-fail instead of
silently downgrading the benchmark.

## Reproduce

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-plots \
  --sample-size 50000 \
  --months 1,2,3,4 \
  --output-dir docs/assets/nyc_taxi_benchmarks \
  --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting \
  --n-estimators 24 \
  --cartoboost-n-estimators 24 \
  --tasks duration,fare \
  --model-workers 1
```

Generated artifacts:

- `docs/assets/nyc_taxi_benchmarks/results.json`
- `docs/assets/nyc_taxi_benchmarks/results.jsonl`
- `docs/assets/nyc_taxi_benchmarks/results.md`

The JSON and Markdown artifacts record the runtime resource snapshot, baseline
dependency status, split manifest hashes, comparability audit, and output
artifact sizes. New runs of `scripts/run_nyc_taxi_quality_benchmarks.py`
persist `train_index_sha256` and `test_index_sha256` per split. LightGBM and
CatBoost are part of the maintained roster and run on the same footing as the
other learned baselines in the validated environment.

| Baseline | Package | Import | Version | Importable | Required class | Class available |
| --- | --- | --- | --- | --- | --- | --- |
| sklearn | scikit-learn | sklearn | `1.9.0` | true |  |  |
| xgboost | xgboost | xgboost | `3.3.0` | true | XGBRegressor | true |
| lightgbm | lightgbm | lightgbm | `4.6.0` | true | LGBMRegressor | true |
| catboost | catboost | catboost | `1.2.10` | true | CatBoostRegressor | true |

## Comparability Audit

| Check | Result |
| --- | --- |
| Same outer splits for requested models | true |
| Primary metric | RMSE |
| Selection mode | Fixed settings, no HPO |
| Selection uses outer test labels | false |
| Same feature-access policy | true |
| Train-only target encoding | true |
| Segment diagnostics used for selection | false |
| Completed external baselines | CatBoost, ExtraTrees, HistGradientBoosting, LightGBM, mean, RandomForest, Ridge, XGBoost |
| Skipped requested external baselines | none |
| CartoBoost/external comparison rows | 7 |

## Comparison Summary

For each runnable learned-model split, the table compares the single primary
`cartoboost` row with the lowest-RMSE external baseline that finished under the
same task, split, sample, target transformation, and global settings.

| Task / split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Duration / random | 0.325729 | 0.037797 | HistGradientBoosting | 0.341885 | 0.039919 | -0.016156 | +0.022112 | CartoBoost lower RMSE |
| Duration / spatial holdout | 0.328395 | 0.036862 | HistGradientBoosting | 0.352826 | 0.040023 | -0.024431 | +0.025444 | CartoBoost lower RMSE |
| Duration / out of time | 0.330825 | 0.038554 | HistGradientBoosting | 0.345772 | 0.040406 | -0.014947 | +0.020301 | CartoBoost lower RMSE |
| Fare / random | 0.172949 | 0.041365 | HistGradientBoosting | 0.179155 | 0.042961 | -0.006206 | +0.008249 | CartoBoost lower RMSE |
| Fare / spatial holdout | 0.274082 | 0.058724 | HistGradientBoosting | 0.241167 | 0.053757 | +0.032915 | -0.037310 | External lower RMSE |
| Fare / out of time | 0.179687 | 0.042392 | HistGradientBoosting | 0.186056 | 0.044033 | -0.006369 | +0.008575 | CartoBoost lower RMSE |


## What the Rows Mean

| Task / split | Prediction unit | Target | Validation question | Signal used by CartoBoost |
| --- | --- | --- | --- | --- |
| Duration / random | One completed taxi trip | Log trip duration in seconds | Can the model explain ordinary held-out trips drawn from the same year-wide trip distribution? | Trip distance, passenger count, hour/day periodicity, pickup/dropoff zones, and route geometry. |
| Duration / spatial holdout | One completed taxi trip from held-out pickup zones | Log trip duration in seconds | Does the duration structure transfer when pickup zones are held out? | Spatial splitters and route geometry rather than memorizing the validation rows. |
| Fare / random | One completed taxi trip | Log total fare amount | Can the model recover fare structure for ordinary held-out trips? | Distance, pickup/dropoff zones, hour/day effects, and cartometric route features. |
| Fare / spatial holdout | One completed taxi trip from held-out pickup zones | Log total fare amount | Does fare modeling generalize to zones not present in the training pickup set? | Route and zone geometry carry transferable fare signal beyond target-mean zone encodings. |
| Duration / out of time | One later-period taxi trip | Log trip duration in seconds | Does the model generalize forward to later pickup timestamps without timestamp overlap? | Earlier pickup/dropoff, trip, and time features only. |
| Fare / out of time | One later-period taxi trip | Log total fare amount | Does fare modeling generalize forward in time without timestamp overlap? | Earlier pickup/dropoff, trip, and time features only. |

## Interpretation

- This reduced run meets the documented comparison threshold on three of four
  duration/fare spatial and out-of-time
  comparisons. Fare spatial holdout remains the documented miss.
- The official AutoGeo admission audit counts zero real family wins because the
  selector is not shipped and this artifact is not a leakage-safe AutoGeo
  evidence package.
- The comparisons are fair at the benchmark level: same sample, same split
  rules, same fixed estimator budget, and the same feature-access policy.
- The pickup-demand spatial holdout skips learned models because held-out pickup
  zones have no training demand history; reporting a learned score there would
  collapse to a prior rather than test transferable structure.
- LightGBM and CatBoost are included in the maintained roster and finished the
  current run under the same sample, split, and estimator budget as the other
  learned baselines.

## Current Artifacts

- [Results JSON](../assets/nyc_taxi_benchmarks/results.json)
- [Results JSONL](../assets/nyc_taxi_benchmarks/results.jsonl)
- The generated results report is maintained at `docs/assets/nyc_taxi_benchmarks/results.md`.
- Asset metadata is maintained at `docs/assets/nyc_taxi_benchmarks/README.md`.
