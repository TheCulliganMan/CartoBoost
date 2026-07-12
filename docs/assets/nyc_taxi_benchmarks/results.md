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
- CartoBoost candidate estimators: 24
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean
- command arguments: `scripts/run_nyc_taxi_quality_benchmarks.py --no-plots --sample-size 50000 --months 1,2,3,4 --output-dir docs/assets/nyc_taxi_benchmarks --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting --n-estimators 24 --cartoboost-n-estimators 24 --tasks duration,fare --model-workers 1`

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
| `results.json` | 198602 |
| `results.jsonl` | 32079 |
| `results.md` | 11462 |

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
| cartoboost | ok | 0.325729 | 0.249551 | 0.782493 | 0.037797 | 78.430027 | 0.031028 | 322288.67 | n_estimators=24 |
| lightgbm | ok | 0.345548 | 0.266818 | 0.755219 | 0.040412 | 0.116989 | 0.001451 | 6890018.12 | n_estimators=24 |
| xgboost | ok | 0.345685 | 0.267435 | 0.755026 | 0.040506 | 0.068901 | 0.000643 | 15564202.27 | n_estimators=24 |
| catboost | ok | 0.361323 | 0.279924 | 0.732360 | 0.042397 | 0.429706 | 0.007824 | 1278111.75 |  |
| hist_gradient_boosting | ok | 0.341885 | 0.263563 | 0.760381 | 0.039919 | 5.471722 | 0.076583 | 130577.50 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.328395 | 0.250132 | 0.835133 | 0.036862 | 85.535593 | 0.156089 | 59119.99 | n_estimators=24 |
| lightgbm | ok | 0.361332 | 0.279944 | 0.800402 | 0.041256 | 0.112302 | 0.002641 | 3494627.22 | n_estimators=24 |
| xgboost | ok | 0.355842 | 0.274782 | 0.806422 | 0.040495 | 0.096038 | 0.001031 | 8951983.54 | n_estimators=24 |
| catboost | ok | 0.394509 | 0.307854 | 0.762066 | 0.045369 | 0.328346 | 0.007525 | 1226373.41 |  |
| hist_gradient_boosting | ok | 0.352826 | 0.271578 | 0.809689 | 0.040023 | 0.513714 | 0.005017 | 1839407.09 | n_estimators=24 |

### out_of_time

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.330825 | 0.255646 | 0.780301 | 0.038554 | 73.030903 | 0.026602 | 375911.59 | n_estimators=24 |
| lightgbm | ok | 0.349625 | 0.271534 | 0.754621 | 0.040950 | 0.095843 | 0.001717 | 5825665.79 | n_estimators=24 |
| xgboost | ok | 0.349224 | 0.271213 | 0.755184 | 0.040902 | 0.049413 | 0.000551 | 18139207.42 | n_estimators=24 |
| catboost | ok | 0.366737 | 0.285436 | 0.730013 | 0.043047 | 0.049258 | 0.000781 | 12800000.05 |  |
| hist_gradient_boosting | ok | 0.345772 | 0.267922 | 0.760000 | 0.040406 | 0.242663 | 0.004724 | 2117074.20 | n_estimators=24 |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.172949 | 0.131650 | 0.887095 | 0.041365 | 84.188425 | 0.034870 | 286783.23 | n_estimators=24 |
| lightgbm | ok | 0.181542 | 0.139575 | 0.875595 | 0.043855 | 0.117026 | 0.002057 | 4860366.54 | n_estimators=24 |
| xgboost | ok | 0.181585 | 0.139417 | 0.875536 | 0.043805 | 0.067698 | 0.000736 | 13590815.74 | n_estimators=24 |
| catboost | ok | 0.192591 | 0.148521 | 0.859992 | 0.046666 | 0.055805 | 0.000830 | 12045783.64 |  |
| hist_gradient_boosting | ok | 0.179155 | 0.136731 | 0.878846 | 0.042961 | 0.271184 | 0.005650 | 1769806.88 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.274082 | 0.198329 | 0.834738 | 0.058724 | 109.662485 | 0.040003 | 230683.42 | n_estimators=24 |
| lightgbm | ok | 0.260210 | 0.193420 | 0.851044 | 0.057270 | 0.133054 | 0.002075 | 4447051.03 | n_estimators=24 |
| xgboost | ok | 0.260712 | 0.194004 | 0.850468 | 0.057443 | 0.073545 | 0.000759 | 12160105.63 | n_estimators=24 |
| catboost | ok | 0.245934 | 0.187600 | 0.866940 | 0.055547 | 0.055604 | 0.001007 | 9164608.45 |  |
| hist_gradient_boosting | ok | 0.241167 | 0.181555 | 0.872048 | 0.053757 | 0.274354 | 0.004519 | 2042214.17 | n_estimators=24 |

### out_of_time

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.179687 | 0.135522 | 0.881144 | 0.042392 | 78.614428 | 0.035962 | 278071.94 | n_estimators=24 |
| lightgbm | ok | 0.188298 | 0.143095 | 0.869480 | 0.044761 | 0.123748 | 0.002047 | 4884601.32 | n_estimators=24 |
| xgboost | ok | 0.188598 | 0.143324 | 0.869064 | 0.044833 | 0.071033 | 0.000818 | 12229932.44 | n_estimators=24 |
| catboost | ok | 0.199495 | 0.152893 | 0.853496 | 0.047826 | 0.057545 | 0.000911 | 10981975.33 |  |
| hist_gradient_boosting | ok | 0.186056 | 0.140766 | 0.872569 | 0.044033 | 0.393390 | 0.006203 | 1612199.06 | n_estimators=24 |

