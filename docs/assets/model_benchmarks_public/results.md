# Model Benchmark Suite

This generated report compares the primary CartoBoost regressor against optional external baselines on deterministic public tabular workloads and embedded graph diagnostics.

## Command

`PYTHONPATH=python uv run --group dev --group bench python scripts/run_model_benchmark_suite.py ...`

Command arguments:

`scripts/run_model_benchmark_suite.py --output-dir docs/assets/model_benchmarks_public --datasets diabetes,california_housing,karate --n-rows 5000 --models mean,cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,node2vec_regressor,graphsage_regressor --n-estimators 24 --graph-dim 4 --graph-epochs 2 --selection-mode validation_search --validation-trials 3 --repeat-seeds 42,43,44 --no-plots`

## Configuration

- Seed: `42`
- Datasets requested: `diabetes, california_housing, karate`
- Rows per workload: `5000`
- Train fraction: `0.8`
- Selection mode: `validation_search`
- Validation trials per tunable model: `3`
- Models requested: `mean, cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting, random_forest, extra_trees, ridge, node2vec_regressor, graphsage_regressor`

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
| `results.json` | 329988 |
| `results.jsonl` | 137248 |
| `results.md` | 18616 |

## Comparability Audit

| Check | Value |
| --- | --- |
| Same outer splits for requested models | True |
| Primary metric | `rmse` |
| Selection metric | `rmse` |
| Selection uses outer test labels | False |
| Equal tunable trial budget | True |
| Tunable trial count | `3` |
| Completed external baselines | `catboost, extra_trees, hist_gradient_boosting, lightgbm, mean, random_forest, ridge, xgboost` |
| Skipped requested external baselines | `` |
| Completed CartoBoost-family rows | `cartoboost, graphsage_regressor, node2vec_regressor` |
| CartoBoost/external comparison rows | 4 |

## Selection and Leakage Policy

- global hyperparameters: every tunable model family uses the same inner-validation budget before holdout scoring; no model family uses test labels for tuning
- primary cartoboost row: single selected cartoboost run retrained after inner validation; no internal candidate is selected on test metrics
- neural feature gate: uses deterministic inner train/validation rows inside the training split only
- graph feature gate: uses deterministic inner train/validation rows inside the training split only
- external baseline selection: best external baseline is selected only for reporting after each model is scored
- diagnostic rows: graph, neural, and link-prediction rows are diagnostics and are not substitutes for the primary cartoboost comparison row

## Split Definitions

| Split | Kind | Train fraction | Purpose |
| --- | --- | --- | --- |
| random | seeded_row_shuffle | configured_by_--train-frac | interpolation across rows drawn from the same workload distribution |
| group_holdout | seeded_group_holdout | configured_by_--train-frac_over_unique_groups | cold-group generalization for workloads with repeated IDs or graph sources |

## Dataset Sources

| Workload | Source | Rows | Features | SHA-256 fingerprint |
| --- | --- | ---: | ---: | --- |
| diabetes | sklearn.datasets.load_diabetes bundled public regression dataset. | 442 | 10 | `d0e115e7bf84c3d7f4c1b43e7e1cb0bf35cd01ad1e0fd239320748b66f1f3888` |
| california_housing | sklearn.datasets.fetch_california_housing deterministic 5000-row seed-42 sample from the 20,640-row public California housing dataset. | 5000 | 8 | `d0f75cd29b2fa35166c72d168c78cd2f206ab5b1c2d6a29e38437c55d3fa77ad` |
| karate | Embedded Zachary karate club edge list and post-split labels from the benchmark harness constants. | 78 | 5 | `069058a0030b0e4859fbfb8254bc70c9f73eceb83c0fad5e2f1eba22352a6824` |

## Results

## CartoBoost vs External Baselines

For each regression split, this table compares the single primary `cartoboost` row with the lowest-RMSE external baseline that finished under the same data split and global benchmark settings.

| Workload | Split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| diabetes | random | 54.3467 | 0.2875 | ridge | 51.5180 | 0.2657 | 2.8287 | -0.0470 | external_lower_or_tied_rmse |
| california_housing | random | 0.6301 | 0.2264 | hist_gradient_boosting | 0.5958 | 0.2138 | 0.0342 | -0.0307 | external_lower_or_tied_rmse |
| karate | random | 0.2665 | 0.2210 | xgboost | 0.0488 | 0.0409 | 0.2177 | -1.1713 | external_lower_or_tied_rmse |
| karate | group_holdout | 0.2661 | 0.1695 | xgboost | 0.2584 | 0.1157 | 0.0077 | -0.0393 | external_lower_or_tied_rmse |

### Interpretation Notes

- Dense public or synthetic workloads are baseline sanity checks for ordinary tabular regression behavior without graph or neural inputs.
- Neural workloads, when requested, show the difference between repeated-ID and cold-ID claims. Neural and graph rows are diagnostics and are not used as substitutes for the primary `cartoboost` comparison row.
- The graph workload separates two surfaces. Augmented CartoBoost uses graph features as extra columns for the booster, while standalone GraphSAGE-style regressors and link predictors can score graph tasks without a boosted wrapper. The link-predictor rows report AUC/AP because they are ranking candidate source-target edges, not predicting the regression target.
- External baseline rows use the same train/test split and global benchmark settings; no test labels are used for model selection.

## Repeated External Baseline Comparison

Repeated rows use the same model roster, validation-search budget, and split policy with different deterministic seeds. Negative RMSE and WAPE deltas favor CartoBoost; positive R2 deltas favor CartoBoost.

| Workload | Split | Seeds | Best external baseline counts | RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | R2 delta mean | R2 delta 95% CI | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| california_housing | random | 42, 43, 44 | hist_gradient_boosting: 3 | 0.030574 | 0.026971 to 0.034176 | 0.012214 | -0.029786 | -0.030655 to -0.028918 | external_lower_rmse |
| diabetes | random | 42, 43, 44 | hist_gradient_boosting: 1, ridge: 2 | 2.249828 | 0.676595 to 3.823060 | 0.016519 | -0.039117 | -0.065315 to -0.012918 | external_lower_rmse |
| karate | group_holdout | 42, 43, 44 | extra_trees: 1, xgboost: 2 | 0.112862 | -0.066401 to 0.292125 | 0.091720 | nan | nan to nan | mixed_interval |
| karate | random | 42, 43, 44 | extra_trees: 1, xgboost: 2 | 0.100129 | -0.017642 to 0.217899 | 0.097043 | -0.526215 | -1.172317 to 0.119886 | mixed_interval |

## Validation Search Selections

The table records the inner-validation winner for each tunable model. Final holdout metrics above are computed only after retraining the selected configuration on the full outer training split.

| Workload | Split | Model | Selected trial | Validation RMSE | Inner train rows | Inner validation rows | Selected config |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| diabetes | random | cartoboost | 2 | 59.452637 | 283 | 70 | `{"learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 18}` |
| diabetes | random | lightgbm | 3 | 59.035589 | 283 | 70 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| diabetes | random | xgboost | 3 | 59.057725 | 283 | 70 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| diabetes | random | catboost | 3 | 59.273588 | 283 | 70 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| diabetes | random | hist_gradient_boosting | 3 | 60.028953 | 283 | 70 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| diabetes | random | random_forest | 1 | 59.847678 | 283 | 70 | `{"max_depth": 4, "min_samples_leaf": 2, "n_estimators": 24}` |
| diabetes | random | extra_trees | 3 | 57.392940 | 283 | 70 | `{"max_depth": 5, "min_samples_leaf": 4, "n_estimators": 30}` |
| diabetes | random | ridge | 1 | 56.288485 | 283 | 70 | `{"ridge_alpha": 0.1}` |
| california_housing | random | cartoboost | 1 | 0.644431 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "n_estimators": 24}` |
| california_housing | random | lightgbm | 1 | 0.641852 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | xgboost | 1 | 0.649380 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | catboost | 1 | 0.699424 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | hist_gradient_boosting | 1 | 0.628613 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | random_forest | 3 | 0.670049 | 3200 | 800 | `{"max_depth": 5, "min_samples_leaf": 4, "n_estimators": 30}` |
| california_housing | random | extra_trees | 3 | 0.738621 | 3200 | 800 | `{"max_depth": 5, "min_samples_leaf": 4, "n_estimators": 30}` |
| california_housing | random | ridge | 3 | 0.700590 | 3200 | 800 | `{"ridge_alpha": 10.0}` |
| karate | random | cartoboost | 1 | 0.251990 | 50 | 12 | `{"learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "n_estimators": 24}` |
| karate | random | lightgbm | 1 | 0.282165 | 50 | 12 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | random | xgboost | 2 | 0.281668 | 50 | 12 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| karate | random | catboost | 3 | 0.274774 | 50 | 12 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| karate | random | hist_gradient_boosting | 1 | 0.282902 | 50 | 12 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | random | random_forest | 2 | 0.285531 | 50 | 12 | `{"max_depth": 3, "min_samples_leaf": 1, "n_estimators": 18}` |
| karate | random | extra_trees | 2 | 0.245033 | 50 | 12 | `{"max_depth": 3, "min_samples_leaf": 1, "n_estimators": 18}` |
| karate | random | ridge | 3 | 0.272913 | 50 | 12 | `{"ridge_alpha": 10.0}` |
| karate | random | node2vec_regressor | 1 | 0.282902 | 50 | 12 | `{"graph_dim": 4, "graph_epochs": 2, "learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "n_estimators": 24}` |
| karate | random | graphsage_regressor | 1 | 0.282902 | 50 | 12 | `{"graph_dim": 4, "graph_epochs": 2, "learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "n_estimators": 24}` |
| karate | group_holdout | cartoboost | 2 | 0.344617 | 42 | 10 | `{"learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 18}` |
| karate | group_holdout | lightgbm | 3 | 0.348115 | 42 | 10 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| karate | group_holdout | xgboost | 1 | 0.076476 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | group_holdout | catboost | 3 | 0.270421 | 42 | 10 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| karate | group_holdout | hist_gradient_boosting | 3 | 0.348115 | 42 | 10 | `{"learning_rate": 0.064, "max_depth": 3, "n_estimators": 30}` |
| karate | group_holdout | random_forest | 1 | 0.242643 | 42 | 10 | `{"max_depth": 4, "min_samples_leaf": 2, "n_estimators": 24}` |
| karate | group_holdout | extra_trees | 1 | 0.188111 | 42 | 10 | `{"max_depth": 4, "min_samples_leaf": 2, "n_estimators": 24}` |
| karate | group_holdout | ridge | 1 | 0.233270 | 42 | 10 | `{"ridge_alpha": 0.1}` |
| karate | group_holdout | node2vec_regressor | 2 | 0.353552 | 42 | 10 | `{"graph_dim": 2, "graph_epochs": 1, "learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 18}` |
| karate | group_holdout | graphsage_regressor | 2 | 0.324110 | 42 | 10 | `{"graph_dim": 2, "graph_epochs": 1, "learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 18}` |

### sklearn diabetes

Frozen public scikit-learn diabetes regression workload with 442 rows, 10 numeric features, and disease-progression target.

#### random

Train rows: `353`; test rows: `89`.
Train index SHA-256: `c283a0fd7785bad10b7846411aa608d952724a4167e18f73ef589a64308a908a`; test index SHA-256: `a821cc67f044dc1884b43cca8b91624118475c7270b7c3a61b8296d5b772d654`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 66.0195 | 80.1589 | -0.0087 | 0.4177 | 0.0000 | 21576050 |
| cartoboost | ok | 45.4421 | 54.3467 | 0.5364 | 0.2875 | 0.0183 | 924272 |
| lightgbm | ok | 45.3479 | 53.9218 | 0.5436 | 0.2869 | 0.0223 | 383966 |
| xgboost | ok | 46.1395 | 55.0509 | 0.5243 | 0.2919 | 0.0220 | 351605 |
| catboost | ok | 47.8578 | 56.9310 | 0.4912 | 0.3028 | 0.0036 | 596315 |
| hist_gradient_boosting | ok | 44.1108 | 52.8130 | 0.5621 | 0.2791 | 0.1255 | 22649 |
| random_forest | ok | 44.5594 | 54.5394 | 0.5331 | 0.2819 | 0.0177 | 5504 |
| extra_trees | ok | 45.6382 | 54.9881 | 0.5253 | 0.2887 | 0.0135 | 5608 |
| ridge | ok | 42.0061 | 51.5180 | 0.5834 | 0.2657 | 0.0002 | 3060208 |
| node2vec_regressor | skipped: all validation-search candidates failed: workload has no graph topology |  |  |  |  |  |  |
| graphsage_regressor | skipped: all validation-search candidates failed: workload has no graph topology |  |  |  |  |  |  |

### California housing

Public scikit-learn California housing regression workload with eight numeric census-block features and median house value target.

#### random

Train rows: `4000`; test rows: `1000`.
Train index SHA-256: `c852c7fee0c7fe3a52c7053306e50afd8983882fa30ceb44aa4ffb9ea599f668`; test index SHA-256: `f9a7e8c6c3732700e2e7f6d168a253178d2dae578216a06e4d988d05bce6ead4`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.9268 | 1.1702 | -0.0002 | 0.4457 | 0.0000 | 406835966 |
| cartoboost | ok | 0.4707 | 0.6301 | 0.7100 | 0.2264 | 0.0879 | 2657807 |
| lightgbm | ok | 0.4729 | 0.6337 | 0.7067 | 0.2274 | 0.0520 | 2180231 |
| xgboost | ok | 0.4727 | 0.6304 | 0.7098 | 0.2273 | 0.0351 | 2060972 |
| catboost | ok | 0.5274 | 0.6999 | 0.6422 | 0.2537 | 0.0127 | 3298512 |
| hist_gradient_boosting | ok | 0.4446 | 0.5958 | 0.7407 | 0.2138 | 0.5014 | 141429 |
| random_forest | ok | 0.4638 | 0.6353 | 0.7052 | 0.2230 | 0.0598 | 66648 |
| extra_trees | ok | 0.5687 | 0.7694 | 0.5677 | 0.2735 | 0.0387 | 47920 |
| ridge | ok | 0.5250 | 0.7151 | 0.6265 | 0.2525 | 0.0028 | 7407408 |
| node2vec_regressor | skipped: all validation-search candidates failed: workload has no graph topology |  |  |  |  |  |  |
| graphsage_regressor | skipped: all validation-search candidates failed: workload has no graph topology |  |  |  |  |  |  |

### Zachary karate club

Frozen public 78-edge Zachary karate club graph workload. Rows are observed edges; the regression target is whether the two endpoints share the same post-split club label.

#### random

Train rows: `62`; test rows: `16`.
Train index SHA-256: `1367cc4e05f89b99b101292f4105c89818e39baaec774ed5d7c5a522158c6832`; test index SHA-256: `cae5be975a23027d748116778eae74072f5097fc8a61653bf0bd63217e68f326`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.2036 | 0.2614 | -0.1666 | 0.2172 | 0.0000 | 2370373 |
| cartoboost | ok | 0.2072 | 0.2665 | -0.2120 | 0.2210 | 0.0055 | 51023 |
| lightgbm | ok | 0.1993 | 0.2557 | -0.1158 | 0.2125 | 0.0165 | 21220 |
| xgboost | ok | 0.0383 | 0.0488 | 0.9593 | 0.0409 | 0.0476 | 20640 |
| catboost | ok | 0.1403 | 0.1788 | 0.4546 | 0.1496 | 0.0034 | 39332 |
| hist_gradient_boosting | ok | 0.2023 | 0.2547 | -0.1069 | 0.2158 | 0.0513 | 5227 |
| random_forest | ok | 0.0719 | 0.1039 | 0.8156 | 0.0767 | 0.0190 | 924 |
| extra_trees | ok | 0.1304 | 0.1607 | 0.5594 | 0.1390 | 0.0191 | 954 |
| ridge | ok | 0.1728 | 0.2281 | 0.1121 | 0.1843 | 0.0003 | 294253 |
| node2vec_regressor | ok | 0.1976 | 0.2658 | -0.2062 | 0.2107 | 0.0047 | 81754 |
| graphsage_regressor | ok | 0.2023 | 0.2547 | -0.1069 | 0.2158 | 0.0044 | 51759 |

#### group_holdout

Train rows: `52`; test rows: `26`.
Train index SHA-256: `dddac4b5f12ac598f17c4aaea291a98aba3f738d686b682c8cf642a6f3ba1c3a`; test index SHA-256: `b10d872fdad08114783ad1e005843d965e6ef28616371c8395487296ed69d123`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.2337 | 0.3218 | -0.0145 | 0.2642 | 0.0000 | 3948976 |
| cartoboost | ok | 0.1499 | 0.2661 | 0.3065 | 0.1695 | 0.0067 | 225108 |
| lightgbm | ok | 0.2318 | 0.3148 | 0.0288 | 0.2620 | 0.0114 | 47280 |
| xgboost | ok | 0.1024 | 0.2584 | 0.3458 | 0.1157 | 0.0371 | 46647 |
| catboost | ok | 0.1587 | 0.2726 | 0.2719 | 0.1794 | 0.0027 | 93905 |
| hist_gradient_boosting | ok | 0.2318 | 0.3148 | 0.0288 | 0.2620 | 0.0954 | 1819 |
| random_forest | ok | 0.1351 | 0.3075 | 0.0738 | 0.1528 | 0.0354 | 1485 |
| extra_trees | ok | 0.1351 | 0.2720 | 0.2752 | 0.1527 | 0.0167 | 1577 |
| ridge | ok | 0.2015 | 0.2988 | 0.1252 | 0.2278 | 0.0003 | 523487 |
| node2vec_regressor | ok | 0.1572 | 0.2447 | 0.4132 | 0.1777 | 0.0060 | 204390 |
| graphsage_regressor | ok | 0.1619 | 0.2472 | 0.4012 | 0.1830 | 0.0044 | 80986 |

## Interpretation Notes

- Dense workloads check numeric behavior without ID or graph augmentation.
- Neural workloads, when requested, include repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on augmented source-target rows.
