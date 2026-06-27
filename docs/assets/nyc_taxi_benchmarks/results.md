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
| `plots/duration_random_cartoboost_graph_graphsage_predicted_actual.png` | 82615 |
| `plots/duration_random_cartoboost_graph_graphsage_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 84062 |
| `plots/duration_random_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_graph_hinsage_predicted_actual.png` | 82410 |
| `plots/duration_random_cartoboost_graph_hinsage_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_graph_node2vec_predicted_actual.png` | 82886 |
| `plots/duration_random_cartoboost_graph_node2vec_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_neural_predicted_actual.png` | 80798 |
| `plots/duration_random_cartoboost_neural_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_predicted_actual.png` | 80411 |
| `plots/duration_random_cartoboost_reference_predicted_actual.png` | 81222 |
| `plots/duration_random_cartoboost_reference_zone_residuals.png` | 33095 |
| `plots/duration_random_cartoboost_zone_residuals.png` | 33439 |
| `plots/duration_random_lightgbm_predicted_actual.png` | 79071 |
| `plots/duration_random_lightgbm_zone_residuals.png` | 33265 |
| `plots/duration_random_mean_predicted_actual.png` | 23830 |
| `plots/duration_random_mean_zone_residuals.png` | 31807 |
| `plots/duration_random_xgboost_predicted_actual.png` | 79883 |
| `plots/duration_random_xgboost_zone_residuals.png` | 33197 |
| `plots/duration_spatial_holdout_cartoboost_graph_graphsage_predicted_actual.png` | 85415 |
| `plots/duration_spatial_holdout_cartoboost_graph_graphsage_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 85772 |
| `plots/duration_spatial_holdout_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_graph_hinsage_predicted_actual.png` | 84897 |
| `plots/duration_spatial_holdout_cartoboost_graph_hinsage_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_graph_node2vec_predicted_actual.png` | 85343 |
| `plots/duration_spatial_holdout_cartoboost_graph_node2vec_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_neural_predicted_actual.png` | 83507 |
| `plots/duration_spatial_holdout_cartoboost_neural_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_predicted_actual.png` | 82818 |
| `plots/duration_spatial_holdout_cartoboost_reference_predicted_actual.png` | 84079 |
| `plots/duration_spatial_holdout_cartoboost_reference_zone_residuals.png` | 23245 |
| `plots/duration_spatial_holdout_cartoboost_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_lightgbm_predicted_actual.png` | 82625 |
| `plots/duration_spatial_holdout_lightgbm_zone_residuals.png` | 22766 |
| `plots/duration_spatial_holdout_mean_predicted_actual.png` | 25059 |
| `plots/duration_spatial_holdout_mean_zone_residuals.png` | 23994 |
| `plots/duration_spatial_holdout_xgboost_predicted_actual.png` | 82208 |
| `plots/duration_spatial_holdout_xgboost_zone_residuals.png` | 22787 |
| `plots/fare_random_cartoboost_graph_graphsage_predicted_actual.png` | 61148 |
| `plots/fare_random_cartoboost_graph_graphsage_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 62042 |
| `plots/fare_random_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_graph_hinsage_predicted_actual.png` | 60849 |
| `plots/fare_random_cartoboost_graph_hinsage_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_graph_node2vec_predicted_actual.png` | 61325 |
| `plots/fare_random_cartoboost_graph_node2vec_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_neural_predicted_actual.png` | 59314 |
| `plots/fare_random_cartoboost_neural_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_predicted_actual.png` | 58463 |
| `plots/fare_random_cartoboost_reference_predicted_actual.png` | 60547 |
| `plots/fare_random_cartoboost_reference_zone_residuals.png` | 31706 |
| `plots/fare_random_cartoboost_zone_residuals.png` | 35097 |
| `plots/fare_random_lightgbm_predicted_actual.png` | 58606 |
| `plots/fare_random_lightgbm_zone_residuals.png` | 35161 |
| `plots/fare_random_mean_predicted_actual.png` | 26760 |
| `plots/fare_random_mean_zone_residuals.png` | 34116 |
| `plots/fare_random_xgboost_predicted_actual.png` | 59208 |
| `plots/fare_random_xgboost_zone_residuals.png` | 35157 |
| `plots/fare_spatial_holdout_cartoboost_graph_graphsage_predicted_actual.png` | 70861 |
| `plots/fare_spatial_holdout_cartoboost_graph_graphsage_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 70792 |
| `plots/fare_spatial_holdout_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_graph_hinsage_predicted_actual.png` | 69929 |
| `plots/fare_spatial_holdout_cartoboost_graph_hinsage_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_graph_node2vec_predicted_actual.png` | 70415 |
| `plots/fare_spatial_holdout_cartoboost_graph_node2vec_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_neural_predicted_actual.png` | 68961 |
| `plots/fare_spatial_holdout_cartoboost_neural_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_predicted_actual.png` | 68242 |
| `plots/fare_spatial_holdout_cartoboost_reference_predicted_actual.png` | 69638 |
| `plots/fare_spatial_holdout_cartoboost_reference_zone_residuals.png` | 22282 |
| `plots/fare_spatial_holdout_cartoboost_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_lightgbm_predicted_actual.png` | 67968 |
| `plots/fare_spatial_holdout_lightgbm_zone_residuals.png` | 21768 |
| `plots/fare_spatial_holdout_mean_predicted_actual.png` | 27023 |
| `plots/fare_spatial_holdout_mean_zone_residuals.png` | 23871 |
| `plots/fare_spatial_holdout_xgboost_predicted_actual.png` | 67825 |
| `plots/fare_spatial_holdout_xgboost_zone_residuals.png` | 21757 |
| `plots/pickup_demand_random_cartoboost_graph_graphsage_predicted_actual.png` | 84853 |
| `plots/pickup_demand_random_cartoboost_graph_graphsage_zone_residuals.png` | 40190 |
| `plots/pickup_demand_random_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 84932 |
| `plots/pickup_demand_random_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 40231 |
| `plots/pickup_demand_random_cartoboost_graph_hinsage_predicted_actual.png` | 83890 |
| `plots/pickup_demand_random_cartoboost_graph_hinsage_zone_residuals.png` | 40231 |
| `plots/pickup_demand_random_cartoboost_graph_node2vec_predicted_actual.png` | 84344 |
| `plots/pickup_demand_random_cartoboost_graph_node2vec_zone_residuals.png` | 37132 |
| `plots/pickup_demand_random_cartoboost_neural_predicted_actual.png` | 85236 |
| `plots/pickup_demand_random_cartoboost_neural_zone_residuals.png` | 39983 |
| `plots/pickup_demand_random_cartoboost_predicted_actual.png` | 84343 |
| `plots/pickup_demand_random_cartoboost_reference_predicted_actual.png` | 91036 |
| `plots/pickup_demand_random_cartoboost_reference_zone_residuals.png` | 34979 |
| `plots/pickup_demand_random_cartoboost_zone_residuals.png` | 39983 |
| `plots/pickup_demand_random_lightgbm_predicted_actual.png` | 87813 |
| `plots/pickup_demand_random_lightgbm_zone_residuals.png` | 39292 |
| `plots/pickup_demand_random_mean_predicted_actual.png` | 26220 |
| `plots/pickup_demand_random_mean_zone_residuals.png` | 40035 |
| `plots/pickup_demand_random_xgboost_predicted_actual.png` | 88929 |
| `plots/pickup_demand_random_xgboost_zone_residuals.png` | 34913 |
| `plots/pickup_demand_spatial_holdout_cartoboost_predicted_actual.png` | 65851 |
| `plots/pickup_demand_spatial_holdout_cartoboost_reference_predicted_actual.png` | 64626 |
| `plots/pickup_demand_spatial_holdout_cartoboost_reference_zone_residuals.png` | 20819 |
| `plots/pickup_demand_spatial_holdout_cartoboost_zone_residuals.png` | 20721 |
| `plots/pickup_demand_spatial_holdout_lightgbm_predicted_actual.png` | 65465 |
| `plots/pickup_demand_spatial_holdout_lightgbm_zone_residuals.png` | 21110 |
| `plots/pickup_demand_spatial_holdout_mean_predicted_actual.png` | 27059 |
| `plots/pickup_demand_spatial_holdout_mean_zone_residuals.png` | 23382 |
| `plots/pickup_demand_spatial_holdout_xgboost_predicted_actual.png` | 64212 |
| `plots/pickup_demand_spatial_holdout_xgboost_zone_residuals.png` | 21078 |
| `prediction_throughput.png` | 119853 |
| `results.json` | 274027 |
| `results.jsonl` | 48819 |
| `results.md` | 25746 |
| `speed_summary.png` | 121081 |

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
| cartoboost | ok | 0.316428 | 0.240482 | 0.809732 | 0.036113 | 63.004261 | 0.018271 | 1094655.82 | n_estimators=48 |
| lightgbm | ok | 0.334510 | 0.255696 | 0.787366 | 0.038398 | 0.352005 | 0.006459 | 3096654.01 | n_estimators=48 |
| xgboost | ok | 0.333024 | 0.254586 | 0.789251 | 0.038231 | 0.242462 | 0.003048 | 6562308.31 | n_estimators=48 |
| catboost | ok | 0.348982 | 0.268441 | 0.768569 | 0.040312 | 0.396703 | 0.002386 | 8380912.13 |  |
| hist_gradient_boosting | ok | 0.328239 | 0.250217 | 0.795263 | 0.037575 | 3.263694 | 0.079356 | 252027.24 | n_estimators=48 |
| random_forest | ok | 0.380746 | 0.294479 | 0.724522 | 0.044222 | 4.043005 | 0.103886 | 192518.57 | n_estimators=48 |
| extra_trees | ok | 0.388196 | 0.302743 | 0.713638 | 0.045463 | 3.677489 | 0.177338 | 112778.88 | n_estimators=48 |
| ridge | ok | 0.389498 | 0.303845 | 0.711713 | 0.045628 | 0.064656 | 0.000666 | 30043161.03 |  |
| mean | ok | 0.725425 | 0.578521 | -0.000001 | 0.086876 | 0.000207 | 0.000035 | 577616328.85 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.327583 | 0.251650 | 0.787195 | 0.037662 | 70.771911 | 0.031613 | 743226.79 | n_estimators=48 |
| lightgbm | ok | 0.350911 | 0.273336 | 0.755808 | 0.040908 | 0.316237 | 0.009140 | 2570795.63 | n_estimators=48 |
| xgboost | ok | 0.352107 | 0.274287 | 0.754140 | 0.041050 | 0.281521 | 0.004034 | 5824552.52 | n_estimators=48 |
| catboost | ok | 0.357080 | 0.278693 | 0.747147 | 0.041710 | 0.378832 | 0.003813 | 6161874.98 |  |
| hist_gradient_boosting | ok | 0.343622 | 0.267074 | 0.765846 | 0.039971 | 6.462783 | 0.069389 | 338612.54 | n_estimators=48 |
| random_forest | ok | 0.383294 | 0.300408 | 0.708659 | 0.044960 | 4.354774 | 0.017045 | 1378488.97 | n_estimators=48 |
| extra_trees | ok | 0.387980 | 0.302913 | 0.701491 | 0.045334 | 0.313661 | 0.014353 | 1636990.63 | n_estimators=48 |
| ridge | ok | 0.396865 | 0.310395 | 0.687662 | 0.046454 | 0.015831 | 0.001623 | 14477268.31 |  |
| mean | ok | 0.710708 | 0.570193 | -0.001661 | 0.085336 | 0.000110 | 0.000027 | 874267844.20 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.167022 | 0.119828 | 0.902671 | 0.037318 | 71.240095 | 0.040704 | 491353.71 | n_estimators=48 |
| lightgbm | ok | 0.175487 | 0.126959 | 0.892555 | 0.039538 | 1.927366 | 0.011631 | 1719474.90 | n_estimators=48 |
| xgboost | ok | 0.175270 | 0.127023 | 0.892820 | 0.039558 | 0.516649 | 0.008636 | 2315786.41 | n_estimators=48 |
| catboost | ok | 0.184898 | 0.135921 | 0.880722 | 0.042329 | 0.201554 | 0.001984 | 10078526.98 |  |
| hist_gradient_boosting | ok | 0.172697 | 0.124364 | 0.895945 | 0.038730 | 17.938715 | 0.051563 | 387877.22 | n_estimators=48 |
| random_forest | ok | 0.199058 | 0.146641 | 0.861753 | 0.045668 | 1.342055 | 0.016327 | 1224933.50 | n_estimators=48 |
| extra_trees | ok | 0.199227 | 0.148851 | 0.861519 | 0.046356 | 0.650290 | 0.034204 | 584720.52 | n_estimators=48 |
| ridge | ok | 0.190073 | 0.139603 | 0.873952 | 0.043476 | 0.006688 | 0.000266 | 75164502.06 |  |
| mean | ok | 0.535368 | 0.411833 | -0.000001 | 0.128255 | 0.000082 | 0.000015 | 1375324755.55 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.183040 | 0.135017 | 0.885475 | 0.041493 | 59.724094 | 0.014588 | 1610615.81 | n_estimators=48 |
| lightgbm | ok | 0.184542 | 0.138228 | 0.883588 | 0.042480 | 0.370839 | 0.007541 | 3115801.42 | n_estimators=48 |
| xgboost | ok | 0.185297 | 0.138843 | 0.882634 | 0.042669 | 0.160501 | 0.002635 | 8916042.17 | n_estimators=48 |
| catboost | ok | 0.198256 | 0.149298 | 0.865644 | 0.045883 | 0.432566 | 0.003601 | 6524024.36 |  |
| hist_gradient_boosting | ok | 0.185264 | 0.138481 | 0.882676 | 0.042558 | 1.877323 | 0.026466 | 887788.93 | n_estimators=48 |
| random_forest | ok | 0.207419 | 0.156159 | 0.852938 | 0.047991 | 3.855059 | 0.096949 | 242354.23 | n_estimators=48 |
| extra_trees | ok | 0.203642 | 0.154938 | 0.858245 | 0.047616 | 0.550027 | 0.015924 | 1475466.14 | n_estimators=48 |
| ridge | ok | 0.318001 | 0.220567 | 0.654330 | 0.067785 | 0.004713 | 0.000218 | 107965566.83 |  |
| mean | ok | 0.543820 | 0.421950 | -0.010920 | 0.129674 | 0.000043 | 0.000010 | 2255552610.87 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.529766 | 0.400057 | 0.951693 | 0.099725 | 5.614285 | 0.009502 | 819377.72 | n_estimators=48 |
| lightgbm | ok | 0.595696 | 0.450784 | 0.938921 | 0.112371 | 0.290960 | 0.003566 | 2183245.72 | n_estimators=48 |
| xgboost | ok | 0.598845 | 0.453007 | 0.938274 | 0.112925 | 0.131560 | 0.001486 | 5238247.43 | n_estimators=48 |
| catboost | ok | 0.668529 | 0.512120 | 0.923073 | 0.127660 | 0.132846 | 0.001377 | 5655692.76 |  |
| hist_gradient_boosting | ok | 0.565712 | 0.428789 | 0.944915 | 0.106888 | 1.305205 | 0.044633 | 174446.06 | n_estimators=48 |
| random_forest | ok | 0.742919 | 0.569983 | 0.905000 | 0.142084 | 0.252850 | 0.014775 | 526977.19 | n_estimators=48 |
| extra_trees | ok | 0.840772 | 0.675830 | 0.878326 | 0.168469 | 0.089114 | 0.015572 | 499991.97 | n_estimators=48 |
| ridge | ok | 0.872242 | 0.688691 | 0.869048 | 0.171675 | 0.004282 | 0.000258 | 30129584.89 |  |
| mean | ok | 2.410684 | 1.948451 | -0.000276 | 0.485705 | 0.000042 | 0.000013 | 608658579.88 |  |

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
| mean | ok | 2.387509 | 1.973998 | -0.028338 | 0.538308 | 0.000064 | 0.000013 | 623516721.60 |  |

