# Repeated NYC Taxi Benchmark

This report reruns the maintained NYC taxi benchmark and summarizes quality confidence intervals, paired baseline deltas, and speed ratios.

- runs: 3
- seeds: 11, 29, 47
- command arguments: `scripts/run_repeated_nyc_taxi_benchmarks.py --runs 3 --run-dir target/repeated-nyc-expanded-real-30k-maintained --summary-json docs/assets/nyc_taxi_benchmarks/repeated_results.json --summary-md docs/assets/nyc_taxi_benchmarks/repeated_results.md --no-download --no-plots --sample-size 30000 --tasks duration,fare,pickup_demand --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean --n-estimators 24 --cartoboost-n-estimators 24 --cartoboost-splitters axis_histogram:512,diagonal_2d,gaussian_2d,periodic:24,periodic:7,sparse_set --cartoboost-min-samples-leaf 20 --model-workers 1`
- run artifacts: `target/repeated-nyc-expanded-real-30k-maintained`
- models: cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting, random_forest, extra_trees, ridge, mean
- sample size: 30000
- baseline estimators: 24; CartoBoost candidate estimators: 24
- baseline max depth: 4; CartoBoost candidate max depth: 5
- CartoBoost splitters: axis_histogram:512,diagonal_2d,gaussian_2d,periodic:24,periodic:7,sparse_set; XGBoost tree_method: hist
- zone treatment: target_mean
- primary comparison uses one `cartoboost` row against the lowest-RMSE external baseline that finished in each run.

## Output Artifacts

| Artifact | Size bytes |
| --- | ---: |
| `docs/assets/nyc_taxi_benchmarks/repeated_results.json` | 76148 |
| `docs/assets/nyc_taxi_benchmarks/repeated_results.md` | 9681 |

## Comparability Audit

| Check | Value |
| --- | --- |
| Run count | 3 |
| Same outer splits in every child run | True |
| Any outer test-label selection | False |
| Same feature-access policy in every child run | True |
| Completed external baselines in all runs | `extra_trees, hist_gradient_boosting, mean, random_forest, ridge, xgboost` |
| Skipped requested external baselines in any run | `catboost, lightgbm` |
| Completed CartoBoost-family rows in all runs | `cartoboost` |
| CartoBoost/external comparison rows | 15 |

## Quality Summary

| task/split | model | RMSE mean | RMSE 95% CI | MAE mean | R2 mean | RMSE wins/ties | train median sec | predict rows/sec median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| duration/random | cartoboost | 0.321169 | 0.315977 to 0.326361 | 0.244571 | 0.790778 | 3 | 44.448889 | 466801.98 |
| duration/random | extra_trees | 0.355254 | 0.351205 to 0.359303 | 0.273684 | 0.743997 | 0 | 0.157635 | 2067035.32 |
| duration/random | hist_gradient_boosting | 0.335674 | 0.329368 to 0.341980 | 0.256638 | 0.771468 | 0 | 0.130217 | 4918032.88 |
| duration/random | mean | 0.702196 | 0.693223 to 0.711170 | 0.558627 | -0.000116 | 0 | 0.000012 | 1220262274.38 |
| duration/random | random_forest | 0.344734 | 0.339538 to 0.349930 | 0.265978 | 0.758919 | 0 | 0.905740 | 1498189.44 |
| duration/random | ridge | 0.358300 | 0.353960 to 0.362640 | 0.277120 | 0.739565 | 0 | 0.001505 | 92249493.56 |
| duration/random | xgboost | 0.338896 | 0.333660 to 0.344133 | 0.259956 | 0.767046 | 0 | 0.077567 | 5453514.10 |
| duration/spatial_holdout | cartoboost | 0.330828 | 0.317818 to 0.343837 | 0.251259 | 0.778013 | 3 | 52.045215 | 352728.77 |
| duration/spatial_holdout | extra_trees | 0.363845 | 0.347164 to 0.380525 | 0.282059 | 0.732548 | 0 | 0.177073 | 1601542.12 |
| duration/spatial_holdout | hist_gradient_boosting | 0.345947 | 0.328191 to 0.363703 | 0.264843 | 0.757785 | 0 | 0.268409 | 1479954.69 |
| duration/spatial_holdout | mean | 0.734758 | 0.605524 to 0.863993 | 0.590460 | -0.071484 | 0 | 0.000018 | 774111752.59 |
| duration/spatial_holdout | random_forest | 0.353296 | 0.336236 to 0.370356 | 0.272879 | 0.748137 | 0 | 0.949863 | 1655866.60 |
| duration/spatial_holdout | ridge | 0.398195 | 0.348953 to 0.447437 | 0.307200 | 0.683925 | 0 | 0.001408 | 70062634.95 |
| duration/spatial_holdout | xgboost | 0.347894 | 0.332701 to 0.363088 | 0.267233 | 0.754578 | 0 | 0.078276 | 4886230.90 |
| fare/random | cartoboost | 0.169674 | 0.166922 to 0.172426 | 0.127762 | 0.895443 | 0 | 58.297641 | 424375.66 |
| fare/random | extra_trees | 0.177615 | 0.172826 to 0.182405 | 0.133990 | 0.885482 | 0 | 0.212340 | 1488679.70 |
| fare/random | hist_gradient_boosting | 0.175451 | 0.172927 to 0.177974 | 0.132820 | 0.888205 | 0 | 0.318159 | 1648087.67 |
| fare/random | mean | 0.524851 | 0.516132 to 0.533569 | 0.397940 | -0.000081 | 0 | 0.000016 | 804501866.13 |
| fare/random | random_forest | 0.171635 | 0.168119 to 0.175151 | 0.127950 | 0.892988 | 0 | 1.441391 | 1311153.01 |
| fare/random | ridge | 0.165326 | 0.160693 to 0.169959 | 0.122431 | 0.900682 | 3 | 0.001574 | 69297696.72 |
| fare/random | xgboost | 0.177478 | 0.175181 to 0.179775 | 0.135116 | 0.885617 | 0 | 0.078768 | 4911321.81 |
| fare/spatial_holdout | cartoboost | 0.211883 | 0.136772 to 0.286995 | 0.157818 | 0.845188 | 2 | 46.997481 | 559154.67 |
| fare/spatial_holdout | extra_trees | 0.220673 | 0.154391 to 0.286955 | 0.167598 | 0.829599 | 0 | 0.166332 | 1999011.43 |
| fare/spatial_holdout | hist_gradient_boosting | 0.215329 | 0.142671 to 0.287988 | 0.161813 | 0.839431 | 0 | 0.156637 | 2336062.98 |
| fare/spatial_holdout | mean | 0.578982 | 0.341519 to 0.816445 | 0.451545 | -0.141054 | 0 | 0.000013 | 1154461118.82 |
| fare/spatial_holdout | random_forest | 0.192034 | 0.167719 to 0.216349 | 0.144994 | 0.862210 | 1 | 0.956461 | 1699562.78 |
| fare/spatial_holdout | ridge | 0.193039 | 0.164341 to 0.221737 | 0.146432 | 0.862026 | 0 | 0.001261 | 85148603.08 |
| fare/spatial_holdout | xgboost | 0.218679 | 0.143636 to 0.293722 | 0.164756 | 0.834661 | 0 | 0.060339 | 6942318.84 |
| pickup_demand/random | cartoboost | 0.592596 | 0.586702 to 0.598490 | 0.479194 | 0.914713 | 3 | 11.265360 | 654859.36 |
| pickup_demand/random | extra_trees | 0.702811 | 0.697877 to 0.707746 | 0.536577 | 0.880023 | 0 | 0.132030 | 1048480.63 |
| pickup_demand/random | hist_gradient_boosting | 0.626449 | 0.624499 to 0.628400 | 0.510477 | 0.904689 | 0 | 0.154979 | 1547678.04 |
| pickup_demand/random | mean | 2.029528 | 2.011892 to 2.047163 | 1.747732 | -0.000289 | 0 | 0.000011 | 827464219.13 |
| pickup_demand/random | random_forest | 0.632404 | 0.623780 to 0.641028 | 0.477597 | 0.902838 | 0 | 0.316850 | 1220561.39 |
| pickup_demand/random | ridge | 0.801130 | 0.798478 to 0.803782 | 0.608198 | 0.844127 | 0 | 0.001297 | 58719162.65 |
| pickup_demand/random | xgboost | 0.641861 | 0.640825 to 0.642898 | 0.517665 | 0.899940 | 0 | 0.022340 | 7279439.16 |
| pickup_demand/spatial_holdout | mean | 2.088607 | 2.088607 to 2.088607 | 1.807484 | -0.002958 | 3 | 0.000008 | 978860542.13 |

## Primary CartoBoost vs Best External Baseline

Negative RMSE and WAPE deltas favor CartoBoost. Positive R2 deltas favor CartoBoost. The external model count records which baseline was lowest-RMSE for that split across runs.

| task/split | runs | best external model counts | RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | R2 delta mean | R2 delta 95% CI |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| duration/random | 3 | hist_gradient_boosting: 3 | -0.014505 | -0.016121 to -0.012639 | -0.001843 | 0.019311 | 0.017008 to 0.021278 |
| duration/spatial_holdout | 3 | hist_gradient_boosting: 1, random_forest: 1, xgboost: 1 | -0.014260 | -0.017035 to -0.010027 | -0.002072 | 0.019047 | 0.015947 to 0.022415 |
| fare/random | 3 | ridge: 3 | 0.004348 | 0.001972 to 0.005972 | 0.001684 | -0.005239 | -0.007013 to -0.002473 |
| fare/spatial_holdout | 3 | hist_gradient_boosting: 1, random_forest: 1, ridge: 1 | 0.021544 | -0.005706 to 0.073071 | 0.003694 | -0.020020 | -0.074373 to 0.009312 |
| pickup_demand/random | 3 | hist_gradient_boosting: 2, random_forest: 1 | -0.033195 | -0.036340 to -0.029866 | -0.007023 | 0.009827 | 0.008833 to 0.010901 |

## Paired Baseline Deltas

Negative RMSE deltas favor the CartoBoost-family row. Positive R2 deltas favor it.

| task/split | comparison | RMSE delta mean | RMSE delta 95% CI | R2 delta mean | R2 delta 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| duration/random | cartoboost_vs_xgboost | -0.017728 | -0.017933 to -0.017341 | 0.023732 | 0.023503 to 0.023959 |
| duration/spatial_holdout | cartoboost_vs_xgboost | -0.017066 | -0.019040 to -0.015125 | 0.023435 | 0.018780 to 0.027282 |
| fare/random | cartoboost_vs_xgboost | -0.007804 | -0.008306 to -0.007008 | 0.009826 | 0.009018 to 0.010290 |
| fare/spatial_holdout | cartoboost_vs_xgboost | -0.006796 | -0.008254 to -0.005579 | 0.010527 | 0.007727 to 0.013564 |
| pickup_demand/random | cartoboost_vs_xgboost | -0.049266 | -0.052336 to -0.043205 | 0.014773 | 0.012918 to 0.015881 |

## Speed Ratios

| task/split | train ratio vs XGBoost median | train ratio min-max | predict rps ratio vs XGBoost median | predict rps ratio min-max | RMSE delta vs Carto ref | R2 delta vs Carto ref | RMSE delta vs XGB | R2 delta vs XGB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| duration/random | 669.27x | 548.03x-763.86x | 0.076x | 0.059x-0.094x | n/a | n/a | -0.017908 | 0.023735 | miss |
| duration/spatial_holdout | 739.54x | 565.03x-765.41x | 0.072x | 0.068x-0.091x | n/a | n/a | -0.017035 | 0.024244 | miss |
| fare/random | 740.12x | 527.49x-775.26x | 0.082x | 0.044x-0.086x | n/a | n/a | -0.008097 | 0.010171 | miss |
| fare/spatial_holdout | 733.67x | 538.24x-972.28x | 0.081x | 0.079x-0.100x | n/a | n/a | -0.006554 | 0.010292 | miss |
| pickup_demand/random | 491.03x | 411.46x-504.28x | 0.089x | 0.083x-0.094x | n/a | n/a | -0.052255 | 0.015520 | miss |
