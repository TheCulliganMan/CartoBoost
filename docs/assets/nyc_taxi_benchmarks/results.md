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
- dataset hash: 7708e1f5350fce2c0de4c431df3d513ca372492888583968fdabeb8ed0b3a328
- sample size: 100000
- task rows: {'duration': 100000, 'fare': 100000, 'pickup_demand': 38932}
- models requested: cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting, random_forest, extra_trees, ridge, mean
- baseline estimators: 48
- CartoBoost candidate estimators: 48
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean
- command arguments: `scripts/run_nyc_taxi_quality_benchmarks.py --no-plots --sample-size 100000 --months 1,2,3,4,5,6,7,8,9,10,11,12 --output-dir docs/assets/nyc_taxi_benchmarks --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean --n-estimators 48 --cartoboost-n-estimators 48 --tasks duration,fare,pickup_demand --model-workers 1`

## Resource Usage

| Field | Value |
| --- | --- |
| cpu | `arm` |
| threads | `10` |
| os | `macOS-26.5.1-arm64-arm-64bit-Mach-O` |
| python | `3.13.12` |
| numpy | `2.4.6` |
| rustc | `rustc 1.94.0 (4a4ef493e 2026-03-02)` |

## Baseline Dependency Status

| Key | Package | Import | Version | Module importable | Required class | Required class available |
| --- | --- | --- | --- | ---: | --- | ---: |
| catboost | catboost | catboost | `1.2.10` | True | CatBoostRegressor | True |
| lightgbm | lightgbm | lightgbm | `4.6.0` | True | LGBMRegressor | True |
| sklearn | scikit-learn | sklearn | `1.9.0` | True |  |  |
| xgboost | xgboost | xgboost | `3.3.0` | True | XGBRegressor | True |

## Output Artifacts

| Artifact | Size bytes |
| --- | ---: |
| `results.json` | 268044 |
| `results.jsonl` | 48828 |
| `results.md` | 17790 |

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
| Completed external baselines | `catboost, extra_trees, hist_gradient_boosting, lightgbm, mean, random_forest, ridge, xgboost` |
| Skipped requested external baselines | `` |
| Completed CartoBoost-family rows | `cartoboost` |
| Skipped CartoBoost-family rows | `` |
| CartoBoost/external comparison rows | 5 |

## Selection and Leakage Policy

- global hyperparameters: fixed_before_holdout_scoring; no model family uses test labels for tuning
- primary cartoboost row: single configured cartoboost run; no internal candidate is selected on test metrics
- zone target encoding: fit on training rows for each outer split before transforming holdout rows
- graph feature gate: uses deterministic inner train/validation rows inside the training split only
- neural feature gate: uses deterministic inner train/validation rows inside the training split only
- segment diagnostics: computed after prediction and excluded from fitting, tuning, and model selection

## CartoBoost vs External Baselines

For each runnable learned-model split, this table compares the single primary `cartoboost` row with the lowest-RMSE external baseline that finished under the same task, split, data sample, target transformation, and global benchmark settings.

| task | split | CartoBoost RMSE | CartoBoost WAPE | best external baseline | external RMSE | external WAPE | RMSE delta | R2 delta | result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| duration | random | 0.316428 | 0.036113 | hist_gradient_boosting | 0.328239 | 0.037575 | -0.011811 | 0.014469 | cartoboost_lower_rmse |
| duration | spatial_holdout | 0.327583 | 0.037662 | hist_gradient_boosting | 0.343622 | 0.039971 | -0.016039 | 0.021349 | cartoboost_lower_rmse |
| fare | random | 0.167022 | 0.037318 | hist_gradient_boosting | 0.172697 | 0.038730 | -0.005675 | 0.006727 | cartoboost_lower_rmse |
| fare | spatial_holdout | 0.183040 | 0.041493 | lightgbm | 0.184542 | 0.042480 | -0.001502 | 0.001887 | cartoboost_lower_rmse |
| pickup_demand | random | 0.529766 | 0.099725 | hist_gradient_boosting | 0.565712 | 0.106888 | -0.035945 | 0.006778 | cartoboost_lower_rmse |

### What Each Comparison Row Models

| task/split | prediction unit | target being modeled | validation question | modeling signal |
| --- | --- | --- | --- | --- |
| duration/random | one completed taxi trip | log trip duration in seconds | Can the model explain ordinary held-out trips drawn from the same month-wide trip distribution? | Base CartoBoost uses trip distance, passenger count, hour/weekday periodicity, pickup/dropoff zones, and route geometry. |
| duration/spatial_holdout | one completed taxi trip from held-out pickup zones | log trip duration in seconds | Does the trip-duration structure transfer when pickup zones are held out? | The gain comes from spatial splitters and route geometry rather than memorizing the exact validation rows. |
| fare/random | one completed taxi trip | log total fare amount | Can the model recover fare structure for ordinary held-out trips? | Distance, pickup/dropoff zones, hour/weekday effects, and cartometric route features align with how fares vary. |
| fare/spatial_holdout | one completed taxi trip from held-out pickup zones | log total fare amount | Does fare modeling generalize to zones not present in the training pickup set? | Route and zone geometry carry transferable fare signal beyond target-mean zone encodings. |
| pickup_demand/random | pickup zone x hour x weekday bucket | log pickup trip count | Can the model explain recurring zone-time demand for observed zones? | The node2vec row adds topology from observed pickup-zone relationships before modeling hour, weekday, and zone effects. |

### Interpretation Notes

- Fare and duration are primarily geotemporal row tasks. The base CartoBoost candidate uses native periodic hour/day splitters, diagonal and radial spatial splitters, and sparse-set taxi-zone membership. Those primitives let the model express pickup/dropoff geometry directly instead of asking an axis-only tabular baseline to approximate it through many rectangular cuts.
- Pickup demand is a zone-time graph problem. Graph rows are kept as diagnostics for topology-sensitive behavior, but the public comparison summary keeps `cartoboost` as the single product row.
- Graph and neural rows are not expected to improve every target. When the base geotemporal splitters already explain the signal, they match the base candidate and mainly add training cost. Their value is in workloads where ID residuals or source-target topology carry signal that ordinary dense columns do not expose.
- The pickup-demand cold-zone spatial holdout intentionally skips learned models. That split removes all zone demand history, so a quality comparison would collapse to priors rather than test model structure.

### Pickup-Zone Segment Diagnostics

These diagnostics are computed after prediction on each holdout split. They summarize pickup-zone error distribution and are not used for training, model selection, or tuning.

| task | split | model | pickup zones | zone rows min-max | zone RMSE p50 | zone RMSE p90 | worst zone RMSE |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| duration | random | cartoboost | 184 | 1-957 | 0.319366 | 0.597599 | 1.688514 |
| duration | random | hist_gradient_boosting | 184 | 1-957 | 0.334915 | 0.625150 | 1.689792 |
| duration | spatial_holdout | cartoboost | 46 | 1-3097 | 0.342112 | 0.599958 | 0.756333 |
| duration | spatial_holdout | hist_gradient_boosting | 46 | 1-3097 | 0.366946 | 0.618498 | 0.727659 |
| fare | random | cartoboost | 184 | 1-957 | 0.173822 | 0.371457 | 0.856359 |
| fare | random | hist_gradient_boosting | 184 | 1-957 | 0.181868 | 0.404962 | 0.915290 |
| fare | spatial_holdout | cartoboost | 46 | 1-3097 | 0.223480 | 0.396836 | 0.727161 |
| fare | spatial_holdout | lightgbm | 46 | 1-3097 | 0.235632 | 0.418852 | 0.770520 |
| pickup_demand | random | cartoboost | 257 | 1-46 | 0.470412 | 0.695883 | 3.232411 |
| pickup_demand | random | hist_gradient_boosting | 257 | 1-46 | 0.495826 | 0.727797 | 3.247891 |

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.316428 | 0.240482 | 0.809732 | 0.036113 | 26.370172 | 0.020450 | 978005.06 | n_estimators=48 |
| lightgbm | ok | 0.334510 | 0.255696 | 0.787366 | 0.038398 | 0.356458 | 0.012722 | 1572131.39 | n_estimators=48 |
| xgboost | ok | 0.333024 | 0.254586 | 0.789251 | 0.038231 | 1.044649 | 0.002995 | 6676773.05 | n_estimators=48 |
| catboost | ok | 0.348982 | 0.268441 | 0.768569 | 0.040312 | 1.252617 | 0.034256 | 583840.74 |  |
| hist_gradient_boosting | ok | 0.328239 | 0.250217 | 0.795263 | 0.037575 | 5.102143 | 0.097663 | 204785.93 | n_estimators=48 |
| random_forest | ok | 0.380746 | 0.294479 | 0.724522 | 0.044222 | 1.792315 | 0.016735 | 1195073.31 | n_estimators=48 |
| extra_trees | ok | 0.388196 | 0.302743 | 0.713638 | 0.045463 | 0.417526 | 0.032933 | 607292.07 | n_estimators=48 |
| ridge | ok | 0.389498 | 0.303845 | 0.711713 | 0.045628 | 0.009262 | 0.000351 | 57027343.07 |  |
| mean | ok | 0.725425 | 0.578521 | -0.000001 | 0.086876 | 0.000086 | 0.000016 | 1246724904.50 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.327583 | 0.251650 | 0.787195 | 0.037662 | 32.122466 | 0.021241 | 1106151.78 | n_estimators=48 |
| lightgbm | ok | 0.350911 | 0.273336 | 0.755808 | 0.040908 | 0.355463 | 0.008594 | 2734133.16 | n_estimators=48 |
| xgboost | ok | 0.352107 | 0.274287 | 0.754140 | 0.041050 | 0.198723 | 0.003227 | 7280878.71 | n_estimators=48 |
| catboost | ok | 0.357080 | 0.278693 | 0.747147 | 0.041710 | 0.208939 | 0.002579 | 9111684.39 |  |
| hist_gradient_boosting | ok | 0.343622 | 0.267074 | 0.765846 | 0.039971 | 1.279312 | 0.036958 | 635745.09 | n_estimators=48 |
| random_forest | ok | 0.383294 | 0.300408 | 0.708659 | 0.044960 | 1.494925 | 0.018143 | 1295062.76 | n_estimators=48 |
| extra_trees | ok | 0.387980 | 0.302913 | 0.701491 | 0.045334 | 0.400026 | 0.018050 | 1301753.51 | n_estimators=48 |
| ridge | ok | 0.396865 | 0.310395 | 0.687662 | 0.046454 | 0.006602 | 0.000326 | 72184324.30 |  |
| mean | ok | 0.710708 | 0.570193 | -0.001661 | 0.085336 | 0.000047 | 0.000012 | 1886006533.48 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.167022 | 0.119828 | 0.902671 | 0.037318 | 34.642536 | 0.012961 | 1543075.93 | n_estimators=48 |
| lightgbm | ok | 0.175487 | 0.126959 | 0.892555 | 0.039538 | 0.330189 | 0.008911 | 2244375.21 | n_estimators=48 |
| xgboost | ok | 0.175270 | 0.127023 | 0.892820 | 0.039558 | 0.309359 | 0.002871 | 6965304.20 | n_estimators=48 |
| catboost | ok | 0.184898 | 0.135921 | 0.880722 | 0.042329 | 0.179776 | 0.002433 | 8220726.55 |  |
| hist_gradient_boosting | ok | 0.172697 | 0.124364 | 0.895945 | 0.038730 | 2.796062 | 0.198745 | 100631.59 | n_estimators=48 |
| random_forest | ok | 0.199058 | 0.146641 | 0.861753 | 0.045668 | 3.033735 | 0.015076 | 1326630.14 | n_estimators=48 |
| extra_trees | ok | 0.199227 | 0.148851 | 0.861519 | 0.046356 | 0.634004 | 0.033235 | 601779.00 | n_estimators=48 |
| ridge | ok | 0.190073 | 0.139603 | 0.873952 | 0.043476 | 0.005763 | 0.000241 | 83131040.70 |  |
| mean | ok | 0.535368 | 0.411833 | -0.000001 | 0.128255 | 0.000108 | 0.000016 | 1243476345.11 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.183040 | 0.135017 | 0.885475 | 0.041493 | 42.293093 | 0.013867 | 1694346.67 | n_estimators=48 |
| lightgbm | ok | 0.184542 | 0.138228 | 0.883588 | 0.042480 | 0.299597 | 0.008506 | 2762299.08 | n_estimators=48 |
| xgboost | ok | 0.185297 | 0.138843 | 0.882634 | 0.042669 | 0.207475 | 0.003382 | 6948052.63 | n_estimators=48 |
| catboost | ok | 0.198256 | 0.149298 | 0.865644 | 0.045883 | 0.380581 | 0.003476 | 6759411.87 |  |
| hist_gradient_boosting | ok | 0.185264 | 0.138481 | 0.882676 | 0.042558 | 1.817673 | 0.037851 | 620742.27 | n_estimators=48 |
| random_forest | ok | 0.207419 | 0.156159 | 0.852938 | 0.047991 | 1.390105 | 0.017240 | 1362916.55 | n_estimators=48 |
| extra_trees | ok | 0.203642 | 0.154938 | 0.858245 | 0.047616 | 0.324294 | 0.017807 | 1319487.26 | n_estimators=48 |
| ridge | ok | 0.318001 | 0.220567 | 0.654330 | 0.067785 | 0.005280 | 0.000267 | 87876568.72 |  |
| mean | ok | 0.543820 | 0.421950 | -0.010920 | 0.129674 | 0.000047 | 0.000011 | 2152246877.38 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.529766 | 0.400057 | 0.951693 | 0.099725 | 4.269394 | 0.004508 | 1727311.52 | n_estimators=48 |
| lightgbm | ok | 0.595696 | 0.450784 | 0.938921 | 0.112371 | 0.168777 | 0.002265 | 3438160.55 | n_estimators=48 |
| xgboost | ok | 0.598845 | 0.453007 | 0.938274 | 0.112925 | 0.101551 | 0.001002 | 7771428.32 | n_estimators=48 |
| catboost | ok | 0.668529 | 0.512120 | 0.923073 | 0.127660 | 0.074800 | 0.000789 | 9874444.31 |  |
| hist_gradient_boosting | ok | 0.565712 | 0.428789 | 0.944915 | 0.106888 | 0.683416 | 0.011492 | 677490.27 | n_estimators=48 |
| random_forest | ok | 0.742919 | 0.569983 | 0.905000 | 0.142084 | 0.165537 | 0.017578 | 442937.94 | n_estimators=48 |
| extra_trees | ok | 0.840772 | 0.675830 | 0.878326 | 0.168469 | 0.085754 | 0.016485 | 472312.94 | n_estimators=48 |
| ridge | ok | 0.872242 | 0.688691 | 0.869048 | 0.171675 | 0.001717 | 0.000118 | 65750712.97 |  |
| mean | ok | 2.410684 | 1.948451 | -0.000276 | 0.485705 | 0.000028 | 0.000010 | 781879969.76 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| lightgbm | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| xgboost | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| catboost | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| hist_gradient_boosting | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| random_forest | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| extra_trees | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| ridge | skipped |  |  |  |  |  |  |  | learned models are not valid for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| mean | ok | 2.387509 | 1.973998 | -0.028338 | 0.538308 | 0.000026 | 0.000009 | 927995976.96 |  |

