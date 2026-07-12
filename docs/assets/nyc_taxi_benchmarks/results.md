# NYC Taxi Model Quality Benchmarks

## Research Question

On real NYC taxi data, do geographic and temporal feature families improve
prediction quality for trip duration, fare amount, and pickup-zone demand when
compared with strong gradient-boosted tabular baselines?

## Dataset

The benchmark uses NYC TLC taxi-derived records. The row-level tasks use trip
records with pickup/dropoff zone context, trip attributes, passenger count, and
time-of-day features. The demand task aggregates pickup activity by zone and
time bucket.

## Targets

Quality metrics are computed on transformed regression targets:

- Trip duration: log trip duration.
- Fare amount: log total amount.
- Pickup-zone demand: log pickup trip count for a zone-time bucket.

## Feature Sets

- Geographic features: pickup zone, dropoff zone, route geometry, and
  zone-level encodings.
- Temporal features: hour, weekday, and periodic time structure.
- Trip features: distance, passenger count, and related trip descriptors.
- Graph features for pickup demand: topology learned from observed pickup-zone
  relationships.

## Comparison Method

The primary `cartoboost` row is compared with the requested external
baselines that finish in the validated environment: XGBoost, LightGBM,
CatBoost, scikit-learn tree ensembles, Ridge, and a mean baseline under
the same task, split, target transformation, and global benchmark
settings.

- dataset source: nyc_tlc_trip_records
- source URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- dataset hash: 5110541e767f70aabdc8ee1eb158beaab347fb8d3fede1cd1a47e2637d3825f2
- sample size: 50000
- task rows: {'duration': 50000, 'fare': 50000}
- models requested: cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting
- baseline estimators: 24
- CartoBoost candidate estimators: 8
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean
- command arguments: `scripts/run_nyc_taxi_quality_benchmarks.py --no-plots --sample-size 50000 --months 1,2,3,4 --output-dir docs/assets/nyc_taxi_benchmarks --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting --n-estimators 24 --cartoboost-n-estimators 8 --tasks duration,fare --model-workers 1`

## Split Manifests

| Task | Split | Split kind | Split manifest hash | Train index hash | Test index hash | CRS note |
| --- | --- | --- | --- | --- | --- | --- |
| duration | random | seeded_row_shuffle | `sha256:c925c81ec67ed734883e447b0ed8cff585e6e3cf38e1e817725157ee58b9bf95` | `sha256:5656dfaa20d5c5d6eca74ec12ee802f587b040049ad44667e7d6e3b4a4254055` | `sha256:bee73ca172c57c21b478d46bb81c1524faff5ec9465e8c98e26c07687ed49f32` | No coordinate CRS applies to this random-row diagnostic split. |
| duration | spatial_holdout | group_spatial_cv | `sha256:69a310148327f58cf791c5df7765fa86ec8c8961aa88594fc63cf561ce37286a` | `sha256:f1ed6d3eeb63ae16c0f13fe796c8db8678cd29cf5fb07b258860f873653004df` | `sha256:2cc5ef67dbed99aff0b6de977e3d150e32de539c1b72494d42b1a71c708dbb61` | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |
| duration | out_of_time | chronological_timestamp_holdout | `sha256:c68ad145848792a952729e58fe59e75d38affd822e29a7fe460f0ed0a67ae25f` | `sha256:aa149280a2f2fc3b1d53da10fdd409517e69cd13c3cd215b1d548b09b4e74088` | `sha256:93189ab389199f948dbfce5cbac71d88ece6d66d64ab3da3e4a48e0ac0721098` | No coordinate CRS applies to this chronological timestamp split. |
| fare | random | seeded_row_shuffle | `sha256:ed57b397be0b8074fcec43717f8f1b4e040f671b61a31e64e38e58ab92c19d78` | `sha256:5656dfaa20d5c5d6eca74ec12ee802f587b040049ad44667e7d6e3b4a4254055` | `sha256:bee73ca172c57c21b478d46bb81c1524faff5ec9465e8c98e26c07687ed49f32` | No coordinate CRS applies to this random-row diagnostic split. |
| fare | spatial_holdout | group_spatial_cv | `sha256:27f918c994550d46489b2daa0df7ac5f2c3c4a99cb9b97675e3039fddc42c74a` | `sha256:f1ed6d3eeb63ae16c0f13fe796c8db8678cd29cf5fb07b258860f873653004df` | `sha256:2cc5ef67dbed99aff0b6de977e3d150e32de539c1b72494d42b1a71c708dbb61` | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |
| fare | out_of_time | chronological_timestamp_holdout | `sha256:35f1b4539497448393eb1e98daa23782145e79eac800f622f740048c8db4214e` | `sha256:aa149280a2f2fc3b1d53da10fdd409517e69cd13c3cd215b1d548b09b4e74088` | `sha256:93189ab389199f948dbfce5cbac71d88ece6d66d64ab3da3e4a48e0ac0721098` | No coordinate CRS applies to this chronological timestamp split. |

## Resource Usage

| Field | Value |
| --- | --- |
| cpu | `arm` |
| threads | `10` |
| os | `macOS-26.5.2-arm64-arm-64bit` |
| python | `3.12.13` |
| numpy | `1.26.4` |
| rustc | `rustc 1.94.0 (4a4ef493e 2026-03-02)` |

## Baseline Dependency Status

| Key | Package | Import | Version | Module importable | Required class | Required class available |
| --- | --- | --- | --- | ---: | --- | ---: |
| catboost | catboost | catboost | `1.2.10` | True | CatBoostRegressor | True |
| lightgbm | lightgbm | lightgbm | `4.6.0` | True | LGBMRegressor | True |
| sklearn | scikit-learn | sklearn | `1.9.0` | True |  |  |
| xgboost | xgboost | xgboost | `3.2.0` | True | XGBRegressor | True |

## Output Artifacts

| Artifact | Size bytes |
| --- | ---: |
| `results.json` | 198599 |
| `results.jsonl` | 32104 |
| `results.md` | 11461 |

## Comparability Audit

| Check | Value |
| --- | --- |
| Same outer splits for requested models | True |
| Primary metric | `rmse` |
| Selection mode | `fixed_settings_no_hpo` |
| Selection uses outer test labels | False |
| Same feature access policy | True |
| Train-only target encoding | True |
| Segment diagnostics used for selection | False |
| Completed external baselines | `catboost, hist_gradient_boosting, lightgbm, xgboost` |
| Skipped requested external baselines | `` |
| Completed CartoBoost-family rows | `cartoboost` |
| Skipped CartoBoost-family rows | `` |
| CartoBoost/external comparison rows | 6 |

## Selection and Leakage Policy

- global hyperparameters: fixed_before_holdout_scoring; no model family uses test labels for tuning
- primary cartoboost row: single configured cartoboost run; no internal candidate is selected on test metrics
- zone target encoding: fit on training rows for each outer split before transforming holdout rows
- graph feature gate: uses deterministic inner train/validation rows inside the training split only
- neural feature gate: uses deterministic inner train/validation rows inside the training split only
- segment diagnostics: computed after prediction and excluded from fitting, tuning, and model selection

## Problem Metrics

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.457439 | 0.360177 | 0.571030 | 0.054552 | 18.663864 | 0.008735 | 1144776.05 | n_estimators=8 |
| lightgbm | ok | 0.345548 | 0.266818 | 0.755219 | 0.040412 | 0.093518 | 0.001286 | 7778572.06 | n_estimators=24 |
| xgboost | ok | 0.345685 | 0.267435 | 0.755026 | 0.040506 | 0.047513 | 0.000531 | 18830937.54 | n_estimators=24 |
| catboost | ok | 0.361323 | 0.279924 | 0.732360 | 0.042397 | 0.121106 | 0.000664 | 15053620.88 |  |
| hist_gradient_boosting | ok | 0.341885 | 0.263563 | 0.760381 | 0.039919 | 0.153957 | 0.001714 | 5834874.23 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.524654 | 0.418250 | 0.579188 | 0.061638 | 16.921427 | 0.006318 | 1460550.19 | n_estimators=8 |
| lightgbm | ok | 0.361332 | 0.279944 | 0.800402 | 0.041256 | 0.068215 | 0.001087 | 8492998.97 | n_estimators=24 |
| xgboost | ok | 0.355842 | 0.274782 | 0.806422 | 0.040495 | 0.045317 | 0.000479 | 19281922.04 | n_estimators=24 |
| catboost | ok | 0.394509 | 0.307854 | 0.762066 | 0.045369 | 0.039848 | 0.000682 | 13522523.66 |  |
| hist_gradient_boosting | ok | 0.352826 | 0.271578 | 0.809689 | 0.040023 | 0.114401 | 0.001267 | 7282627.99 | n_estimators=24 |

### out_of_time

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.465097 | 0.367378 | 0.565770 | 0.055405 | 17.391089 | 0.014717 | 679501.73 | n_estimators=8 |
| lightgbm | ok | 0.349625 | 0.271534 | 0.754621 | 0.040950 | 0.084931 | 0.001194 | 8376373.91 | n_estimators=24 |
| xgboost | ok | 0.349224 | 0.271213 | 0.755184 | 0.040902 | 0.058870 | 0.000605 | 16526658.38 | n_estimators=24 |
| catboost | ok | 0.366737 | 0.285436 | 0.730013 | 0.043047 | 0.042033 | 0.000658 | 15208154.06 |  |
| hist_gradient_boosting | ok | 0.345772 | 0.267922 | 0.760000 | 0.040406 | 0.108944 | 0.002592 | 3857838.64 | n_estimators=24 |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.300779 | 0.232149 | 0.658512 | 0.072942 | 17.169359 | 0.007688 | 1300650.92 | n_estimators=8 |
| lightgbm | ok | 0.181542 | 0.139575 | 0.875595 | 0.043855 | 0.081103 | 0.001374 | 7278020.38 | n_estimators=24 |
| xgboost | ok | 0.181585 | 0.139417 | 0.875536 | 0.043805 | 0.041553 | 0.000491 | 20368341.00 | n_estimators=24 |
| catboost | ok | 0.192591 | 0.148521 | 0.859992 | 0.046666 | 0.036841 | 0.000582 | 17193208.97 |  |
| hist_gradient_boosting | ok | 0.179155 | 0.136731 | 0.878846 | 0.042961 | 0.116824 | 0.002613 | 3826346.61 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.459559 | 0.333773 | 0.535384 | 0.098828 | 23.878652 | 0.008414 | 1096743.52 | n_estimators=8 |
| lightgbm | ok | 0.260210 | 0.193420 | 0.851044 | 0.057270 | 0.095848 | 0.001340 | 6886777.90 | n_estimators=24 |
| xgboost | ok | 0.260712 | 0.194004 | 0.850468 | 0.057443 | 0.073861 | 0.001016 | 9079701.23 | n_estimators=24 |
| catboost | ok | 0.245934 | 0.187600 | 0.866940 | 0.055547 | 0.137234 | 0.001549 | 5957711.08 |  |
| hist_gradient_boosting | ok | 0.241167 | 0.181555 | 0.872048 | 0.053757 | 1.470009 | 0.011213 | 822948.90 | n_estimators=24 |

### out_of_time

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.307494 | 0.236401 | 0.651936 | 0.073948 | 23.971488 | 0.009548 | 1047298.63 | n_estimators=8 |
| lightgbm | ok | 0.188298 | 0.143095 | 0.869480 | 0.044761 | 0.093430 | 0.001754 | 5700035.63 | n_estimators=24 |
| xgboost | ok | 0.188598 | 0.143324 | 0.869064 | 0.044833 | 0.054218 | 0.000498 | 20071937.86 | n_estimators=24 |
| catboost | ok | 0.199495 | 0.152893 | 0.853496 | 0.047826 | 0.036966 | 0.000592 | 16880001.86 |  |
| hist_gradient_boosting | ok | 0.186056 | 0.140766 | 0.872569 | 0.044033 | 0.215904 | 0.003507 | 2851778.26 | n_estimators=24 |

