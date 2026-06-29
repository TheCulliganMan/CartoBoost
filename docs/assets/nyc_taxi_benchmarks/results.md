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
- dataset hash: 741a94b7345cd469a8dc6261b116910f39131f6e1ca0e824dd319e53ef6bd8c8
- sample size: 30000
- task rows: {'duration': 30000, 'fare': 30000, 'pickup_demand': 24650}
- models requested: cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting, random_forest, extra_trees, ridge, mean
- baseline estimators: 24
- CartoBoost candidate estimators: 24
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean
- command arguments: `scripts/run_nyc_taxi_quality_benchmarks.py --output-dir docs/assets/nyc_taxi_benchmarks --no-download --no-plots --sample-size 30000 --tasks duration,fare,pickup_demand --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean --n-estimators 24 --cartoboost-n-estimators 24 --cartoboost-splitters axis_histogram:512,diagonal_2d,gaussian_2d,periodic:24,periodic:7,sparse_set --cartoboost-min-samples-leaf 20 --model-workers 1`

## Split Manifests

| Task | Split | Split kind | Split manifest hash | Rows | CRS note |
| --- | --- | --- | --- | ---: | --- |
| duration | random | seeded_row_shuffle | `sha256:37a8a1b8d95de0461bb4b226a2092804755744082d199dafa88183ef7abdb431` | 30000 | No coordinate CRS applies to this random-row diagnostic split. |
| duration | spatial_holdout | group_spatial_cv | `sha256:d29f8f08056097c37e031efa55bc076dc44e02f8f3301c49e5c88309af67cf2a` | 30000 | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |
| fare | random | seeded_row_shuffle | `sha256:be1874cbdc327ec965d7f6c8bef319e66d6d17d1b7a735885483dfe4ba004b6b` | 30000 | No coordinate CRS applies to this random-row diagnostic split. |
| fare | spatial_holdout | group_spatial_cv | `sha256:4c8fd1695bab05f1310616a99c3cd7e3f019e9d673c01ed2376bab0571ebf973` | 30000 | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |
| pickup_demand | random | seeded_row_shuffle | `sha256:eb1796d5ac7d353a949c32527fd64c6883b6766c321400f610418208289e76d4` | 24650 | No coordinate CRS applies to this random-row diagnostic split. |
| pickup_demand | spatial_holdout | group_spatial_cv | `sha256:35a6924ec908c8cc54ceb5ed31232c9c34352881ee422f1137523580550e93b4` | 24650 | NYC TLC pickup/dropoff zone identifiers are treated as spatial groups; distance-buffered claims require projected taxi zone geometry. |

Legacy note: this artifact predates persisted row-index hashes, so the JSON records a legacy manifest hash over split identity, row counts, held-out pickup zones, dataset hash, seed, CRS note, model version, and dependency versions. New runs from `scripts/run_nyc_taxi_quality_benchmarks.py` also record `train_index_sha256` and `test_index_sha256` per split.
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
| `results.json` | 269603 |
| `results.jsonl` | 48853 |
| `results.md` | 20720 |
| `speed_summary.png` | 121081 |

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
| cartoboost | ok | 0.321631 | 0.244797 | 0.787495 | 0.037386 | 12.596562 | 0.006319 | 949498.55 | n_estimators=24 |
| lightgbm | ok | 0.340456 | 0.260619 | 0.761891 | 0.039803 | 0.226092 | 0.002823 | 2125116.27 | n_estimators=24 |
| xgboost | ok | 0.340162 | 0.260461 | 0.762302 | 0.039779 | 0.135500 | 0.002383 | 2517395.14 | n_estimators=24 |
| catboost | ok | 0.358161 | 0.275918 | 0.736481 | 0.042139 | 0.366057 | 0.002001 | 2998563.75 |  |
| hist_gradient_boosting | ok | 0.337427 | 0.258119 | 0.766108 | 0.039421 | 1.626485 | 0.005606 | 1070297.67 | n_estimators=24 |
| random_forest | ok | 0.346062 | 0.266391 | 0.753984 | 0.040684 | 1.205274 | 0.021955 | 273282.64 | n_estimators=24 |
| extra_trees | ok | 0.360183 | 0.279085 | 0.733498 | 0.042623 | 0.156392 | 0.015439 | 388616.77 | n_estimators=24 |
| ridge | ok | 0.361002 | 0.277595 | 0.732285 | 0.042396 | 0.012554 | 0.000500 | 11998006.27 |  |
| mean | ok | 0.697752 | 0.557510 | -0.000130 | 0.085145 | 0.000124 | 0.000033 | 181356799.71 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.317218 | 0.244041 | 0.757616 | 0.037839 | 7.054656 | 0.009912 | 586030.16 | n_estimators=24 |
| lightgbm | ok | 0.334555 | 0.258528 | 0.730397 | 0.040085 | 0.526899 | 0.003132 | 1854922.61 | n_estimators=24 |
| xgboost | ok | 0.334581 | 0.258639 | 0.730355 | 0.040102 | 0.206545 | 0.002681 | 2166762.76 | n_estimators=24 |
| catboost | ok | 0.347276 | 0.268227 | 0.709504 | 0.041589 | 0.736538 | 0.004659 | 1246822.85 |  |
| hist_gradient_boosting | ok | 0.330881 | 0.255337 | 0.736287 | 0.039590 | 2.128190 | 0.010717 | 542014.93 | n_estimators=24 |
| random_forest | ok | 0.339897 | 0.264278 | 0.721719 | 0.040977 | 0.275638 | 0.018946 | 306614.32 | n_estimators=24 |
| extra_trees | ok | 0.353708 | 0.273389 | 0.698645 | 0.042389 | 0.074627 | 0.013747 | 422568.77 | n_estimators=24 |
| ridge | ok | 0.354533 | 0.274377 | 0.697238 | 0.042542 | 0.003670 | 0.000257 | 22643547.99 |  |
| mean | ok | 0.657256 | 0.519352 | -0.040539 | 0.080526 | 0.000047 | 0.000012 | 494381663.35 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.174649 | 0.128754 | 0.889225 | 0.040675 | 4.617731 | 0.002592 | 2315149.79 | n_estimators=24 |
| lightgbm | ok | 0.182229 | 0.136358 | 0.879402 | 0.043078 | 0.155377 | 0.002967 | 2022045.71 | n_estimators=24 |
| xgboost | ok | 0.182710 | 0.136996 | 0.878764 | 0.043279 | 0.074185 | 0.000789 | 7608979.84 | n_estimators=24 |
| catboost | ok | 0.196341 | 0.147281 | 0.859999 | 0.046529 | 0.060653 | 0.000909 | 6602178.27 |  |
| hist_gradient_boosting | ok | 0.180789 | 0.134599 | 0.881300 | 0.042522 | 0.520524 | 0.005613 | 1068931.28 | n_estimators=24 |
| random_forest | ok | 0.177191 | 0.129255 | 0.885978 | 0.040834 | 0.246418 | 0.016153 | 371459.53 | n_estimators=24 |
| extra_trees | ok | 0.181030 | 0.133740 | 0.880984 | 0.042251 | 0.166193 | 0.015330 | 391377.71 | n_estimators=24 |
| ridge | ok | 0.169843 | 0.123155 | 0.895239 | 0.038907 | 0.002752 | 0.000125 | 48128230.45 |  |
| mean | ok | 0.524746 | 0.399845 | -0.000010 | 0.126318 | 0.000040 | 0.000009 | 679269951.26 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.159090 | 0.120233 | 0.842719 | 0.039392 | 5.335479 | 0.003213 | 1808108.28 | n_estimators=24 |
| lightgbm | ok | 0.166558 | 0.127097 | 0.827605 | 0.041641 | 0.195030 | 0.003623 | 1603551.48 | n_estimators=24 |
| xgboost | ok | 0.167078 | 0.127587 | 0.826527 | 0.041801 | 0.074599 | 0.001055 | 5506813.76 | n_estimators=24 |
| catboost | ok | 0.176206 | 0.134761 | 0.807054 | 0.044152 | 0.078187 | 0.001158 | 5014783.38 |  |
| hist_gradient_boosting | ok | 0.163738 | 0.124610 | 0.833393 | 0.040826 | 1.312745 | 0.043642 | 133104.84 | n_estimators=24 |
| random_forest | ok | 0.166072 | 0.124522 | 0.828609 | 0.040797 | 0.309378 | 0.012883 | 450901.35 | n_estimators=24 |
| extra_trees | ok | 0.170498 | 0.129226 | 0.819354 | 0.042338 | 0.058362 | 0.015070 | 385476.33 | n_estimators=24 |
| ridge | ok | 0.158739 | 0.119507 | 0.843411 | 0.039154 | 0.002610 | 0.000133 | 43814075.11 |  |
| mean | ok | 0.425543 | 0.344144 | -0.125328 | 0.112751 | 0.000043 | 0.000010 | 603533346.78 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.598105 | 0.481325 | 0.914156 | 0.179037 | 2.057228 | 0.003394 | 1452580.89 | n_estimators=24 |
| lightgbm | ok | 0.647961 | 0.521602 | 0.899248 | 0.194018 | 0.388376 | 0.003982 | 1238097.15 | n_estimators=24 |
| xgboost | ok | 0.647344 | 0.521361 | 0.899440 | 0.193929 | 0.060264 | 0.000969 | 5087498.42 | n_estimators=24 |
| catboost | ok | 0.705192 | 0.567393 | 0.880664 | 0.211051 | 0.058541 | 0.000837 | 5891547.35 |  |
| hist_gradient_boosting | ok | 0.631339 | 0.515010 | 0.904351 | 0.191566 | 0.803042 | 0.005419 | 909775.88 | n_estimators=24 |
| random_forest | ok | 0.645452 | 0.486077 | 0.900027 | 0.180804 | 0.086077 | 0.014975 | 329213.53 | n_estimators=24 |
| extra_trees | ok | 0.697081 | 0.534807 | 0.883394 | 0.198930 | 0.032248 | 0.014625 | 337102.66 | n_estimators=24 |
| ridge | ok | 0.800820 | 0.606266 | 0.846105 | 0.225510 | 0.001248 | 0.000093 | 53177763.11 |  |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.651973 | 0.000021 | 0.000008 | 636128966.21 |  |

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
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.659779 | 0.000019 | 0.000007 | 688969813.15 |  |

