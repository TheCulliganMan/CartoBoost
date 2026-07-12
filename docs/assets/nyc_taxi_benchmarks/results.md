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
- dataset hash: 8f6635f1c567a043982004c6bc26cf5be49a5f371d7e09f5e533bd7dcbb866aa
- sample size: 50000
- task rows: {'duration': 50000, 'fare': 50000, 'pickup_demand': 33705}
- models requested: cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting
- baseline estimators: 24
- CartoBoost candidate estimators: 8
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean
- command arguments: `scripts/run_nyc_taxi_quality_benchmarks.py --no-plots --sample-size 50000 --months 1,2,3,4 --output-dir docs/assets/nyc_taxi_benchmarks --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting --n-estimators 24 --cartoboost-n-estimators 8 --tasks duration,fare,pickup_demand --model-workers 1`

## Split Manifests

| Task | Split | Split kind | Split manifest hash | Train index hash | Test index hash | CRS note |
| --- | --- | --- | --- | --- | --- | --- |
| duration | random | seeded_row_shuffle | `sha256:cd433b64601c41a908a05052a7734f0eb9fa14da1292d5a8f0d0a86c7adc02a5` | `sha256:5656dfaa20d5c5d6eca74ec12ee802f587b040049ad44667e7d6e3b4a4254055` | `sha256:bee73ca172c57c21b478d46bb81c1524faff5ec9465e8c98e26c07687ed49f32` | No coordinate CRS applies to this random-row diagnostic split. |
| duration | spatial_holdout | group_spatial_cv | `sha256:c5514ace69e01872dc3f4b45a6614d18c5ba32bbb580b56ce2bd064ab208d691` | `sha256:f1ed6d3eeb63ae16c0f13fe796c8db8678cd29cf5fb07b258860f873653004df` | `sha256:2cc5ef67dbed99aff0b6de977e3d150e32de539c1b72494d42b1a71c708dbb61` | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |
| duration | out_of_time | chronological_timestamp_holdout | `sha256:fe30790ba2ceb9df23e5f63f56457de0005dc611060a08f57a68c1a990e066df` | `sha256:aa149280a2f2fc3b1d53da10fdd409517e69cd13c3cd215b1d548b09b4e74088` | `sha256:93189ab389199f948dbfce5cbac71d88ece6d66d64ab3da3e4a48e0ac0721098` | No coordinate CRS applies to this chronological timestamp split. |
| fare | random | seeded_row_shuffle | `sha256:4939270673000cf2459671dc078d7c5bd56d6ae4aa8dda0fef1fa502976b32b0` | `sha256:5656dfaa20d5c5d6eca74ec12ee802f587b040049ad44667e7d6e3b4a4254055` | `sha256:bee73ca172c57c21b478d46bb81c1524faff5ec9465e8c98e26c07687ed49f32` | No coordinate CRS applies to this random-row diagnostic split. |
| fare | spatial_holdout | group_spatial_cv | `sha256:eb6df77ba99a3fe44ea54b21fd972ea26d7c12f68ac9d120c6ecd14e8c99a6e0` | `sha256:f1ed6d3eeb63ae16c0f13fe796c8db8678cd29cf5fb07b258860f873653004df` | `sha256:2cc5ef67dbed99aff0b6de977e3d150e32de539c1b72494d42b1a71c708dbb61` | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |
| fare | out_of_time | chronological_timestamp_holdout | `sha256:9e1ebb96da97baa82c907f7a52fafdd533eedb30bb188115c7df2581ec65db50` | `sha256:aa149280a2f2fc3b1d53da10fdd409517e69cd13c3cd215b1d548b09b4e74088` | `sha256:93189ab389199f948dbfce5cbac71d88ece6d66d64ab3da3e4a48e0ac0721098` | No coordinate CRS applies to this chronological timestamp split. |
| pickup_demand | random | seeded_row_shuffle | `sha256:7da82ee4f5eff3154557591b9f2780437a88eef5da42913a1e6a79433c6a2ce5` | `sha256:17744927ddf9fee431e672bebf5802373bbfcb1f53456f02e64e600a50605f9f` | `sha256:2081f877d94a9fc8b2dcc9884f2b06d558847bae10cae56a8c62a02cd0e61b18` | No coordinate CRS applies to this random-row diagnostic split. |
| pickup_demand | spatial_holdout | group_spatial_cv | `sha256:0165869a86c2713c37b3ae036a953141b769f3b999e1d3cea9d06a37711420e5` | `sha256:79e7348566b046cdf1f763d896733217ac6502022b907ee39b2d60b114afa1c2` | `sha256:9d7df89aee4b9fc90356a787186ddf23ee8c60b5db6773b9159d46b32ade4860` | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |

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
| `results.json` | 237669 |
| `results.jsonl` | 37517 |
| `results.md` | 14303 |

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
| CartoBoost/external comparison rows | 7 |

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
| cartoboost | ok | 0.457439 | 0.360177 | 0.571030 | 0.054552 | 15.956272 | 0.006182 | 1617730.32 | n_estimators=8 |
| lightgbm | ok | 0.345548 | 0.266818 | 0.755219 | 0.040412 | 0.074415 | 0.001104 | 9057970.99 | n_estimators=24 |
| xgboost | ok | 0.345685 | 0.267435 | 0.755026 | 0.040506 | 0.051748 | 0.000515 | 19433210.97 | n_estimators=24 |
| catboost | ok | 0.361323 | 0.279924 | 0.732360 | 0.042397 | 0.131479 | 0.000554 | 18047805.05 |  |
| hist_gradient_boosting | ok | 0.341885 | 0.263563 | 0.760381 | 0.039919 | 0.123654 | 0.001397 | 7157555.69 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.524654 | 0.418250 | 0.579188 | 0.061638 | 16.310149 | 0.005395 | 1710512.29 | n_estimators=8 |
| lightgbm | ok | 0.361332 | 0.279944 | 0.800402 | 0.041256 | 0.092656 | 0.001443 | 6393902.64 | n_estimators=24 |
| xgboost | ok | 0.355842 | 0.274782 | 0.806422 | 0.040495 | 0.041156 | 0.000480 | 19216632.88 | n_estimators=24 |
| catboost | ok | 0.394509 | 0.307854 | 0.762066 | 0.045369 | 0.041878 | 0.000576 | 16028931.17 |  |
| hist_gradient_boosting | ok | 0.352826 | 0.271578 | 0.809689 | 0.040023 | 0.115724 | 0.001549 | 5957391.86 | n_estimators=24 |

### out_of_time

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.465097 | 0.367378 | 0.565770 | 0.055405 | 16.657409 | 0.006217 | 1608568.39 | n_estimators=8 |
| lightgbm | ok | 0.349625 | 0.271534 | 0.754621 | 0.040950 | 0.079459 | 0.001206 | 8294453.10 | n_estimators=24 |
| xgboost | ok | 0.349224 | 0.271213 | 0.755184 | 0.040902 | 0.041153 | 0.000450 | 22211955.13 | n_estimators=24 |
| catboost | ok | 0.366737 | 0.285436 | 0.730013 | 0.043047 | 0.035587 | 0.000641 | 15604665.05 |  |
| hist_gradient_boosting | ok | 0.345772 | 0.267922 | 0.760000 | 0.040406 | 0.096337 | 0.002283 | 4379881.10 | n_estimators=24 |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.300779 | 0.232149 | 0.658512 | 0.072942 | 17.892972 | 0.006071 | 1647073.36 | n_estimators=8 |
| lightgbm | ok | 0.181542 | 0.139575 | 0.875595 | 0.043855 | 0.076206 | 0.001365 | 7327574.85 | n_estimators=24 |
| xgboost | ok | 0.181585 | 0.139417 | 0.875536 | 0.043805 | 0.041222 | 0.000502 | 19921946.00 | n_estimators=24 |
| catboost | ok | 0.192591 | 0.148521 | 0.859992 | 0.046666 | 0.036825 | 0.000741 | 13492254.02 |  |
| hist_gradient_boosting | ok | 0.179155 | 0.136731 | 0.878846 | 0.042961 | 0.112331 | 0.002025 | 4939388.76 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.459559 | 0.333773 | 0.535384 | 0.098828 | 19.985161 | 0.007182 | 1284819.29 | n_estimators=8 |
| lightgbm | ok | 0.260210 | 0.193420 | 0.851044 | 0.057270 | 0.081404 | 0.001161 | 7951176.26 | n_estimators=24 |
| xgboost | ok | 0.260712 | 0.194004 | 0.850468 | 0.057443 | 0.043630 | 0.000464 | 19905820.34 | n_estimators=24 |
| catboost | ok | 0.245934 | 0.187600 | 0.866940 | 0.055547 | 0.037759 | 0.000628 | 14702063.46 |  |
| hist_gradient_boosting | ok | 0.241167 | 0.181555 | 0.872048 | 0.053757 | 0.104639 | 0.001641 | 5622115.59 | n_estimators=24 |

### out_of_time

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.307494 | 0.236401 | 0.651936 | 0.073948 | 21.218106 | 0.005916 | 1690450.46 | n_estimators=8 |
| lightgbm | ok | 0.188298 | 0.143095 | 0.869480 | 0.044761 | 0.078251 | 0.001237 | 8084074.35 | n_estimators=24 |
| xgboost | ok | 0.188598 | 0.143324 | 0.869064 | 0.044833 | 0.042426 | 0.000485 | 20602626.85 | n_estimators=24 |
| catboost | ok | 0.199495 | 0.152893 | 0.853496 | 0.047826 | 0.036635 | 0.000716 | 13965680.69 |  |
| hist_gradient_boosting | ok | 0.186056 | 0.140766 | 0.872569 | 0.044033 | 0.108240 | 0.001774 | 5636184.31 | n_estimators=24 |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 1.295407 | 1.054246 | 0.678170 | 0.326739 | 3.994039 | 0.003844 | 1753413.97 | n_estimators=8 |
| lightgbm | ok | 0.716880 | 0.576216 | 0.901438 | 0.178585 | 0.057416 | 0.000878 | 7674031.82 | n_estimators=24 |
| xgboost | ok | 0.717092 | 0.576275 | 0.901380 | 0.178603 | 0.032312 | 0.000426 | 15819301.86 | n_estimators=24 |
| catboost | ok | 0.792125 | 0.635068 | 0.879662 | 0.196825 | 0.025001 | 0.000439 | 15342247.56 |  |
| hist_gradient_boosting | ok | 0.699187 | 0.559237 | 0.906244 | 0.173322 | 0.083705 | 0.001529 | 4409845.45 | n_estimators=24 |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| lightgbm | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| xgboost | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| catboost | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| hist_gradient_boosting | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |

