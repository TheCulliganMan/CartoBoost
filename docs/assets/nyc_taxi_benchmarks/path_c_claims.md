# NYC Taxi Path C Claims

Path C is bounded real-data evidence on NYC TLC 2024 yellow taxi tasks. It proves implementation behavior and benchmark discipline on this dataset, not universal market superiority.

| Claim | Task | Split | Falsifier | Unit | RMSE | Baseline RMSE | Improvement | Threshold | Pass |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| directional_structure | duration | spatial_holdout | unordered_pair_baseline | pickup_zone_to_dropoff_zone_by_hour_day_bucket | 0.452740 | 0.547270 | 17.27% | 2.00% | True |
| directional_structure | duration | spatial_holdout | source_target_additive_baseline | pickup_zone_to_dropoff_zone_by_hour_day_bucket | 0.452740 | 0.677731 | 33.20% | 2.00% | True |
| spatial_transfer | duration | spatial_holdout | target_encoded_zone_only_baseline | completed_trip | 0.420272 | 0.493316 | 14.81% | 1.00% | True |
| spatial_transfer | duration | spatial_holdout | mean_baseline | completed_trip | 0.420272 | 0.710708 | 40.87% | 1.00% | True |
| residual_correction | duration | spatial_holdout | raw_baseline | completed_trip | 0.365945 | 0.412204 | 11.22% | 1.00% | True |
| residual_correction | duration | spatial_holdout | global_residual_mean | completed_trip | 0.365945 | 0.412204 | 11.22% | 1.00% | True |
| residual_correction | duration | spatial_holdout | linear_residual_model | completed_trip | 0.365945 | 0.411375 | 11.04% | 1.00% | True |
| directional_structure | fare | spatial_holdout | unordered_pair_baseline | pickup_zone_to_dropoff_zone_by_hour_day_bucket | 0.310279 | 0.343386 | 9.64% | 2.00% | True |
| directional_structure | fare | spatial_holdout | source_target_additive_baseline | pickup_zone_to_dropoff_zone_by_hour_day_bucket | 0.310279 | 0.529702 | 41.42% | 2.00% | True |
| spatial_transfer | fare | spatial_holdout | target_encoded_zone_only_baseline | completed_trip | 0.281317 | 0.326899 | 13.94% | 1.00% | True |
| spatial_transfer | fare | spatial_holdout | mean_baseline | completed_trip | 0.281317 | 0.543820 | 48.27% | 1.00% | True |
| residual_correction | fare | spatial_holdout | raw_baseline | completed_trip | 0.185682 | 0.201508 | 7.85% | 1.00% | True |
| residual_correction | fare | spatial_holdout | global_residual_mean | completed_trip | 0.185682 | 0.201508 | 7.85% | 1.00% | True |
| residual_correction | fare | spatial_holdout | linear_residual_model | completed_trip | 0.185682 | 0.197879 | 6.16% | 1.00% | True |
| temporal_structure | pickup_demand | rolling_origin_zone_time | seasonal_naive | pickup_zone_hour | 0.406106 | 0.492814 | 17.59% | 2.00% | True |
| temporal_structure | pickup_demand | rolling_origin_zone_time | trailing_mean | pickup_zone_hour | 0.406106 | 0.803782 | 49.48% | 2.00% | True |
| temporal_structure | pickup_demand | rolling_origin_zone_time | pooled_ridge | pickup_zone_hour | 0.406106 | 1.593807 | 74.52% | 2.00% | True |
| known_future_sensitivity | pickup_demand | rolling_origin_zone_time | future_known_covariates_ablated | pickup_zone_hour | 0.406106 | 0.828720 | 51.00% | 1.00% | True |
