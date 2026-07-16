#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::parse_forecast_timestamp;

    fn ts(day: u32) -> NaiveDateTime {
        parse_forecast_timestamp(&format!("2026-01-{day:02}")).expect("timestamp")
    }

    #[test]
    fn delta_and_trend_features_use_only_prior_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
                ForecastRow::single(ts(3), 15.0),
                ForecastRow::single(ts(4), 19.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![3],
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![50],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let rows = builder.transform_frame(&frame).expect("features");
        let last = rows.last().expect("last row");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "target_roll_mean_2".to_string(),
                "target_roll_std_3".to_string(),
                "target_ewm_alpha_050".to_string(),
                "target_delta_lag_2".to_string(),
                "target_roll_trend_3".to_string(),
            ]
        );
        assert_eq!(last.timestamp, ts(4));
        assert_eq!(last.features[0], 15.0);
        assert_eq!(last.features[1], 13.5);
        assert!((last.features[2] - 2.0548046676563256).abs() < 1e-12);
        assert_eq!(last.features[3], 13.0);
        assert_eq!(last.features[4], 5.0);
        assert_eq!(last.features[5], 2.5);

        let next = builder
            .transform_next(&rows[0].series_id, frame.rows(), ts(5))
            .expect("next features");
        assert_eq!(next[3], 16.0);
    }

    #[test]
    fn partial_rolling_mean_uses_available_prior_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 16.0),
                ForecastRow::single(ts(3), 22.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: vec![5],
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &["target_partial_roll_mean_5".to_string()]
        );
        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].timestamp, ts(2));
        assert_eq!(rows[0].features, vec![10.0]);
        assert_eq!(rows[1].timestamp, ts(3));
        assert_eq!(rows[1].features, vec![13.0]);

        let next = builder
            .transform_next(&rows[0].series_id, frame.rows(), ts(4))
            .expect("next features");
        assert_eq!(next, vec![16.0]);
    }

    #[test]
    fn trend_and_ewm_feature_windows_are_validated() {
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![0],
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: vec![0],
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![0],
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: vec![0],
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: vec![0],
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: vec![1],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![0],
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![101],
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .is_err());
    }

    #[test]
    fn sorted_prior_next_features_match_public_sorted_transform() {
        let history = vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 12.0),
            ForecastRow::single(ts(3), 16.0),
            ForecastRow::single(ts(4), 20.0),
        ];
        let unsorted_history = vec![
            history[2].clone(),
            history[0].clone(),
            history[3].clone(),
            history[1].clone(),
        ];
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: vec![50],
            calendar_features: vec![CalendarFeature::ElapsedIndex],
            difference_lags: vec![2],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let public = builder
            .transform_next(&history[0].series_id, &unsorted_history, ts(5))
            .expect("public transform");
        let sorted = builder
            .transform_next_sorted_prior(&history[0].series_id, &history, ts(5))
            .expect("sorted transform");

        assert_eq!(sorted, public);
        assert!(builder
            .transform_next_sorted_prior(&history[0].series_id, &unsorted_history, ts(5))
            .is_err());
    }

    #[test]
    fn sorted_prior_next_features_use_known_future_covariates() {
        let mut rows = Vec::new();
        for day in 1..=4 {
            let mut covariates = BTreeMap::new();
            covariates.insert("promo".to_string(), f64::from(day));
            rows.push(ForecastRow::with_covariates(
                "store_item",
                ts(day),
                f64::from(day * 10),
                covariates,
            ));
        }
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["promo".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");
        let mut future_covariates = BTreeMap::new();
        future_covariates.insert("promo".to_string(), 99.0);

        let stale = builder
            .transform_next_sorted_prior("store_item", &rows, ts(5))
            .expect("stale features");
        let known = builder
            .transform_next_sorted_prior_with_covariates(
                "store_item",
                &rows,
                ts(5),
                Some(&future_covariates),
            )
            .expect("known future features");

        assert_eq!(
            builder.feature_names(),
            &["target_lag_1", "covariate_promo"]
        );
        assert_eq!(stale, vec![40.0, 4.0]);
        assert_eq!(known, vec![40.0, 99.0]);
    }

    #[test]
    fn cached_training_features_match_position_builder() {
        let mut rows = Vec::new();
        for day in 1..=8 {
            let mut covariates = BTreeMap::new();
            covariates.insert("distance_miles".to_string(), f64::from(day) * 1.5);
            rows.push(ForecastRow::with_covariates(
                "lane_a",
                ts(day),
                f64::from(day * day + 3),
                covariates,
            ));
        }
        let frame = ForecastFrame::new(rows.clone(), crate::forecasting::ForecastFrequency::Daily)
            .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1, 2, 4],
            rolling_mean_windows: vec![2, 4],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![3],
            rolling_min_windows: vec![3],
            rolling_max_windows: vec![3],
            ewm_alpha_percents: vec![25, 90],
            calendar_features: vec![
                CalendarFeature::DayOfYear,
                CalendarFeature::ElapsedIndex,
                CalendarFeature::ElapsedPhase(14),
            ],
            difference_lags: vec![2],
            rolling_trend_windows: vec![4],
            covariate_features: vec!["distance_miles".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let cached_rows = builder.transform_frame(&frame).expect("cached");
        let manual_rows = (0..rows.len())
            .filter_map(|idx| {
                builder
                    .features_for_position(&rows, idx)
                    .expect("manual features")
                    .map(|features| LagFeatureRow {
                        series_id: "lane_a".to_string(),
                        timestamp: rows[idx].timestamp,
                        target: rows[idx].target,
                        features,
                    })
            })
            .collect::<Vec<_>>();

        assert_eq!(cached_rows.len(), manual_rows.len());
        for (cached, manual) in cached_rows.iter().zip(manual_rows.iter()) {
            assert_eq!(cached.series_id, manual.series_id);
            assert_eq!(cached.timestamp, manual.timestamp);
            assert_eq!(cached.target, manual.target);
            assert_eq!(cached.features.len(), manual.features.len());
            for (left, right) in cached.features.iter().zip(manual.features.iter()) {
                assert!((left - right).abs() < 1e-10, "{left} != {right}");
            }
        }
    }

    #[test]
    fn covariate_features_use_current_row_for_training_and_latest_for_prediction() {
        let mut first_covariates = BTreeMap::new();
        first_covariates.insert("distance_miles".to_string(), 2.5);
        let mut second_covariates = BTreeMap::new();
        second_covariates.insert("distance_miles".to_string(), 2.5);
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::with_covariates("lane_a", ts(1), 10.0, first_covariates),
                ForecastRow::with_covariates("lane_a", ts(2), 12.0, second_covariates),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![
                CalendarFeature::DayOfYear,
                CalendarFeature::ElapsedIndex,
                CalendarFeature::ElapsedPhase(14),
            ],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["distance_miles".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day_of_year".to_string(),
                "calendar_elapsed_index".to_string(),
                "calendar_elapsed_phase".to_string(),
                "covariate_distance_miles".to_string(),
            ]
        );
        assert_eq!(rows[0].features, vec![10.0, 2.0, 1.0, 1.0, 2.5]);

        let next = builder
            .transform_next("lane_a", frame.rows(), ts(3))
            .expect("next");
        assert_eq!(next, vec![12.0, 3.0, 2.0, 2.0, 2.5]);
    }

    #[test]
    fn elapsed_phase_calendar_feature_uses_configured_period() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("lane_a", ts(1), 10.0),
                ForecastRow::new("lane_a", ts(2), 12.0),
                ForecastRow::new("lane_a", ts(3), 14.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::ElapsedPhase(7)],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_elapsed_phase".to_string(),
            ]
        );
        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(rows[0].features, vec![10.0, 1.0]);
        let next = builder
            .transform_next("lane_a", frame.rows(), ts(4))
            .expect("next");
        assert_eq!(next, vec![14.0, 3.0]);

        let invalid = LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::ElapsedPhase(1)],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        });
        assert!(invalid.is_err());
    }

    #[test]
    fn covariate_calendar_interactions_are_leakage_safe() {
        let mut first_covariates = BTreeMap::new();
        first_covariates.insert("airport_lane".to_string(), 1.0);
        let mut second_covariates = BTreeMap::new();
        second_covariates.insert("airport_lane".to_string(), 1.0);
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::with_covariates("lane_a", ts(1), 10.0, first_covariates),
                ForecastRow::with_covariates("lane_a", ts(2), 12.0, second_covariates),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::Day, CalendarFeature::ElapsedPhase(14)],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["airport_lane".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: true,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day".to_string(),
                "calendar_elapsed_phase".to_string(),
                "covariate_airport_lane".to_string(),
                "covariate_airport_lane_x_calendar_day".to_string(),
                "covariate_airport_lane_x_calendar_elapsed_phase".to_string(),
            ]
        );
        let rows = builder.transform_frame(&frame).expect("features");
        assert_eq!(rows[0].features, vec![10.0, 2.0, 1.0, 1.0, 2.0, 1.0]);

        let next = builder
            .transform_next("lane_a", frame.rows(), ts(3))
            .expect("next");
        assert_eq!(next, vec![12.0, 3.0, 2.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn covariate_indicators_encode_low_cardinality_context() {
        let mut first_covariates = BTreeMap::new();
        first_covariates.insert("pickup_borough_code".to_string(), 3.0);
        let mut second_covariates = BTreeMap::new();
        second_covariates.insert("pickup_borough_code".to_string(), 3.0);
        let history = vec![
            ForecastRow::with_covariates("lane_a", ts(1), 10.0, first_covariates),
            ForecastRow::with_covariates("lane_a", ts(2), 12.0, second_covariates),
        ];
        let mut indicator_values = BTreeMap::new();
        indicator_values.insert("pickup_borough_code".to_string(), vec![1.0, 3.0]);
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![CalendarFeature::Day],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: indicator_values,
            covariate_calendar_interactions: true,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day".to_string(),
                "covariate_pickup_borough_code_is_1".to_string(),
                "covariate_pickup_borough_code_is_3".to_string(),
                "covariate_pickup_borough_code_is_1_x_calendar_day".to_string(),
                "covariate_pickup_borough_code_is_3_x_calendar_day".to_string(),
            ]
        );
        let features = builder
            .features_for_position(&history, 1)
            .expect("features")
            .expect("enough history");
        assert_eq!(features, vec![10.0, 2.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn calendar_fourier_features_encode_cycles_without_extra_history() {
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![
                CalendarFeature::DayOfWeekSin,
                CalendarFeature::DayOfWeekCos,
                CalendarFeature::MonthSin,
                CalendarFeature::MonthCos,
                CalendarFeature::DaySin,
                CalendarFeature::DayCos,
                CalendarFeature::MonthStart,
                CalendarFeature::MonthMiddle,
                CalendarFeature::MonthEnd,
            ],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "calendar_day_of_week_sin".to_string(),
                "calendar_day_of_week_cos".to_string(),
                "calendar_month_sin".to_string(),
                "calendar_month_cos".to_string(),
                "calendar_day_sin".to_string(),
                "calendar_day_cos".to_string(),
                "calendar_month_start".to_string(),
                "calendar_month_middle".to_string(),
                "calendar_month_end".to_string(),
            ]
        );
        let monday_in_january = parse_forecast_timestamp("2026-01-05").expect("timestamp");
        let history = [ForecastRow::single(ts(4), 10.0)];
        let features = builder
            .transform_next("__single__", &history, monday_in_january)
            .expect("features");
        assert!((features[0] - 0.0).abs() < 1e-12);
        assert!((features[1] - 1.0).abs() < 1e-12);
        assert!((features[2] - 0.0).abs() < 1e-12);
        assert!((features[3] - 1.0).abs() < 1e-12);
        assert!((features[4] - cyclic_sin(4.0, 31.0)).abs() < 1e-12);
        assert!((features[5] - cyclic_cos(4.0, 31.0)).abs() < 1e-12);
        assert_eq!(features[6], 0.0);
        assert_eq!(features[7], 0.0);
        assert_eq!(features[8], 0.0);

        let month_end = parse_forecast_timestamp("2026-01-31").expect("timestamp");
        let month_end_features = builder
            .transform_next("__single__", &history, month_end)
            .expect("features");
        assert_eq!(month_end_features[6], 0.0);
        assert_eq!(month_end_features[7], 0.0);
        assert_eq!(month_end_features[8], 1.0);
    }

    #[test]
    fn calendar_event_flags_expand_covariate_interactions() {
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: vec![
                CalendarFeature::DayOfWeek,
                CalendarFeature::MonthStart,
                CalendarFeature::MonthEnd,
            ],
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: vec!["airport_lane".to_string()],
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: true,
        })
        .expect("builder");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "calendar_day_of_week".to_string(),
                "calendar_month_start".to_string(),
                "calendar_month_end".to_string(),
                "covariate_airport_lane".to_string(),
                "covariate_airport_lane_x_calendar_day_of_week".to_string(),
                "covariate_airport_lane_x_calendar_month_start".to_string(),
                "covariate_airport_lane_x_calendar_month_end".to_string(),
            ]
        );
    }
}
