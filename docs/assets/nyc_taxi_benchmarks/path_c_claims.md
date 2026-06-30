# NYC Taxi Path C Claims

Path C is bounded real-data evidence on NYC TLC 2024 yellow taxi tasks. It proves implementation behavior and benchmark discipline on this dataset, not universal market superiority.

| Claim | Task | Split | Falsifier | RMSE | Baseline RMSE | Delta | Pass |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| directional_structure | duration | spatial_holdout | unordered_pair_baseline | 0.452740 | 0.547270 | 0.094530 | True |
| directional_structure | duration | spatial_holdout | source_target_additive_baseline | 0.452740 | 0.677731 | 0.224991 | True |
| spatial_transfer | duration | spatial_holdout | target_encoded_zone_only_baseline | 0.452740 | 0.493316 | 0.040576 | True |
| spatial_transfer | duration | spatial_holdout | mean_baseline | 0.452740 | 0.710708 | 0.257968 | True |
| residual_correction | duration | spatial_holdout | raw_baseline | 0.365945 | 0.412204 | 0.046259 | True |
| residual_correction | duration | spatial_holdout | global_residual_mean | 0.365945 | 0.412204 | 0.046259 | True |
| residual_correction | duration | spatial_holdout | linear_residual_model | 0.365945 | 0.411375 | 0.045430 | True |
| directional_structure | fare | spatial_holdout | unordered_pair_baseline | 0.310279 | 0.343386 | 0.033107 | True |
| directional_structure | fare | spatial_holdout | source_target_additive_baseline | 0.310279 | 0.529702 | 0.219423 | True |
| spatial_transfer | fare | spatial_holdout | target_encoded_zone_only_baseline | 0.310279 | 0.326899 | 0.016619 | True |
| spatial_transfer | fare | spatial_holdout | mean_baseline | 0.310279 | 0.543820 | 0.233541 | True |
| residual_correction | fare | spatial_holdout | raw_baseline | 0.185682 | 0.201508 | 0.015826 | True |
| residual_correction | fare | spatial_holdout | global_residual_mean | 0.185682 | 0.201508 | 0.015826 | True |
| residual_correction | fare | spatial_holdout | linear_residual_model | 0.185682 | 0.197879 | 0.012197 | True |
| temporal_structure | pickup_demand | rolling_origin_zone_time | seasonal_naive | 1.084924 | 2.439408 | 1.354484 | True |
| temporal_structure | pickup_demand | rolling_origin_zone_time | trailing_mean | 1.084924 | 0.794309 | -0.290615 | False |
| temporal_structure | pickup_demand | rolling_origin_zone_time | pooled_ridge | 1.084924 | 2.411636 | 1.326712 | True |
| known_future_sensitivity | pickup_demand | rolling_origin_zone_time | future_known_covariates_ablated | 1.084924 | 1.203282 | 0.118358 | True |
