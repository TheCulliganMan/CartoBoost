#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_filter_tracks_linear_trend_and_forecasts() {
        let config = LocalLinearKalmanConfig::new(0.01, 0.001, 0.1).expect("config");
        let result = fit_local_linear_kalman(&[12.0, 14.0, 16.0, 18.0], config).expect("filter");
        let forecast = local_linear_kalman_forecast(result.final_state, 2).expect("forecast");

        assert_eq!(result.estimates.len(), 3);
        assert_eq!(result.smoothed_states.len(), 4);
        assert!(result.final_state.trend > 0.0);
        assert!(result.final_covariance[0][0] > 0.0);
        assert!(result.estimates.last().unwrap().innovation_variance > 0.0);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.residual_summary.fitted_count, 3);
        assert!(result.residual_summary.rmse >= 0.0);
        assert!(forecast[1] > forecast[0]);

        let distribution = local_linear_kalman_forecast_distribution(
            result.final_state,
            result.final_covariance,
            config,
            2,
            1.96,
        )
        .expect("forecast distribution");
        assert_eq!(distribution.len(), 2);
        assert_eq!(distribution[1].mean, forecast[1]);
        assert!(distribution[1].lower < distribution[1].mean);
        assert!(distribution[1].upper > distribution[1].mean);
    }

    #[test]
    fn kalman_filter_rejects_bad_inputs() {
        let config = LocalLinearKalmanConfig::new(0.01, 0.001, 0.1).expect("config");

        assert!(fit_local_linear_kalman(&[1.0], config).is_err());
        assert!(fit_local_linear_kalman(&[1.0, f64::NAN], config).is_err());
        assert!(LocalLinearKalmanConfig::new(0.0, 0.001, 0.1).is_err());
        assert!(local_linear_kalman_forecast(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0
            },
            0
        )
        .is_err());
        assert!(local_linear_kalman_forecast(
            LocalLinearKalmanState {
                level: f64::NAN,
                trend: 0.0,
            },
            1,
        )
        .is_err());
        let invalid_config = LocalLinearKalmanConfig {
            level_process_variance: -1.0,
            trend_process_variance: 0.001,
            observation_variance: 0.1,
        };
        assert!(fit_local_linear_kalman(&[1.0, 2.0], invalid_config).is_err());
        assert!(local_linear_kalman_forecast_distribution(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0,
            },
            [[1.0, 0.0], [0.0, 1.0]],
            invalid_config,
            1,
            1.96,
        )
        .is_err());
        assert!(local_linear_kalman_forecast_distribution(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0,
            },
            [[1.0, 2.0], [0.0, 1.0]],
            config,
            1,
            1.96,
        )
        .is_err());
    }

    #[test]
    fn local_linear_kalman_filters_second_observation_once() {
        let config = LocalLinearKalmanConfig::new(1.0, 1.0, 1.0).expect("config");

        let result = fit_local_linear_kalman(&[0.0, 10.0], config).expect("filter");
        let estimate = result.estimates.first().expect("second observation update");

        assert_eq!(estimate.prior_level, 0.0);
        assert_eq!(estimate.prior_trend, 0.0);
        assert_eq!(estimate.innovation, 10.0);
        assert!((result.final_state.level - 7.5).abs() < 1.0e-12);
        assert!((result.final_state.trend - 2.5).abs() < 1.0e-12);
        assert!((result.final_covariance[0][0] - 0.75).abs() < 1.0e-12);
        assert!((result.final_covariance[0][1] - 0.25).abs() < 1.0e-12);
        assert!((result.final_covariance[1][1] - 1.75).abs() < 1.0e-12);
    }

    #[test]
    fn local_linear_kalman_smoothing_is_scale_invariant() {
        let values = [0.0, 10.0, 0.0, 10.0];
        let base = fit_local_linear_kalman(
            &values,
            LocalLinearKalmanConfig::new(0.1, 0.01, 1.0).expect("base config"),
        )
        .expect("base filter");
        let scaled = fit_local_linear_kalman(
            &values,
            LocalLinearKalmanConfig::new(1.0e-7, 1.0e-8, 1.0e-6).expect("scaled config"),
        )
        .expect("scaled filter");

        for (base, scaled) in base.smoothed_states.iter().zip(&scaled.smoothed_states) {
            assert!((base.level - scaled.level).abs() < 1.0e-8);
            assert!((base.trend - scaled.trend).abs() < 1.0e-8);
        }
    }

    #[test]
    fn local_linear_kalman_forecast_variance_propagates_process_noise_exactly() {
        let config = LocalLinearKalmanConfig::new(2.0, 3.0, 5.0).expect("config");
        let distribution = local_linear_kalman_forecast_distribution(
            LocalLinearKalmanState {
                level: 0.0,
                trend: 0.0,
            },
            [[0.0, 0.0], [0.0, 0.0]],
            config,
            3,
            1.96,
        )
        .expect("forecast distribution");

        assert_eq!(distribution[0].variance, 7.0);
        assert_eq!(distribution[1].variance, 12.0);
        assert_eq!(distribution[2].variance, 26.0);
    }

    #[test]
    fn local_level_kalman_forecasts_flat_level() {
        let config = LocalLevelKalmanConfig::new(0.01, 0.1).expect("config");
        let result = fit_local_level_kalman(&[12.0, 13.0, 13.5], config).expect("filter");
        let forecast = local_level_kalman_forecast(result.final_level, 3).expect("forecast");

        assert_eq!(result.estimates.len(), 2);
        assert_eq!(result.smoothed_states.len(), 3);
        assert!(result.final_variance > 0.0);
        assert!(result.estimates.last().unwrap().gain > 0.0);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.residual_summary.fitted_count, 2);
        assert_eq!(forecast, vec![result.final_level; 3]);

        let distribution = local_level_kalman_forecast_distribution(
            result.final_level,
            result.final_variance,
            config,
            3,
            1.96,
        )
        .expect("forecast distribution");
        assert_eq!(distribution.len(), 3);
        assert_eq!(distribution[0].mean, result.final_level);
        assert!(distribution[0].variance > result.final_variance);
    }

    #[test]
    fn intermittent_demand_methods_are_positive_and_bias_adjusted() {
        let values = [0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0];
        let croston =
            intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Croston)
                .expect("croston");
        let sba = intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Sba)
            .expect("sba");
        let tsb = intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Tsb)
            .expect("tsb");

        assert_eq!(croston.len(), 2);
        assert!(croston[0] > 0.0);
        assert!(sba[0] < croston[0]);
        assert!(tsb[0] > 0.0);
    }

    #[test]
    fn intermittent_demand_rejects_invalid_inputs() {
        assert!(intermittent_demand_forecast(
            &[0.0, -1.0],
            1,
            0.1,
            0.1,
            IntermittentDemandMethod::Croston,
        )
        .is_err());
        assert!(intermittent_demand_forecast(
            &[0.0, 0.0],
            1,
            0.1,
            0.1,
            IntermittentDemandMethod::Tsb,
        )
        .is_err());
        assert!(intermittent_demand_forecast(
            &[1.0],
            0,
            0.1,
            0.1,
            IntermittentDemandMethod::Croston,
        )
        .is_err());
    }

    #[test]
    fn ordinary_kriging_returns_exact_known_coordinate_with_tiny_nugget() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(1.0, 1.0e-9).expect("config");

        let prediction =
            ordinary_kriging_predict(&observations, (0.0, 0.0), config).expect("kriging");

        assert!((prediction.mean - 12.0).abs() < 1.0e-4);
        assert_eq!(prediction.weights.len(), 2);
    }

    #[test]
    fn bounded_ordinary_kriging_matches_reference_equations() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 1.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 3.0,
            },
            KrigingObservation {
                x: 0.0,
                y: 1.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 1.0,
                value: 5.0,
            },
        ];
        let cases = [
            (
                KrigingVariogramModel::Exponential,
                2.058_716_343_389_453_8,
                0.916_622_073_252_181_6,
                [
                    0.406_016_481_098_025_74,
                    0.305_494_690_594_402_15,
                    0.235_409_450_343_212_84,
                    0.053_079_377_964_359_19,
                ],
            ),
            (
                KrigingVariogramModel::Gaussian,
                1.771_379_181_116_378_6,
                0.405_754_171_709_639_1,
                [
                    0.433_091_393_303_780_1,
                    0.374_703_977_216_747_64,
                    0.248_949_097_078_335_18,
                    -0.056_744_467_598_862_924,
                ],
            ),
            (
                KrigingVariogramModel::Spherical,
                2.085_389_982_235_605,
                1.517_206_693_789_695_7,
                [
                    0.437_944_953_865_678_67,
                    0.290_440_969_324_087,
                    0.193_982_754_551_168_7,
                    0.077_631_322_259_065_66,
                ],
            ),
        ];

        for (model, expected_mean, expected_variance, expected_weights) in cases {
            let config = OrdinaryKrigingConfig::new(1.2, 0.15)
                .expect("config")
                .with_sill(1.7)
                .expect("sill")
                .with_variogram_model(model);
            let prediction = ordinary_kriging_predict(&observations, (0.4, 0.3), config)
                .expect("reference prediction");

            assert!((prediction.mean - expected_mean).abs() < 1.0e-12);
            assert!((prediction.variance - expected_variance).abs() < 1.0e-12);
            for (actual, expected) in prediction.weights.iter().zip(expected_weights) {
                assert!((actual - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn linear_variogram_is_unbounded_and_has_known_midpoint_solution() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 0.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 2.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(2.0, 0.0)
            .expect("config")
            .with_sill(4.0)
            .expect("sill")
            .with_variogram_model(KrigingVariogramModel::Linear);

        assert_eq!(theoretical_semivariogram(3.0, config), 6.0);
        let prediction =
            ordinary_kriging_predict(&observations, (1.0, 0.0), config).expect("prediction");
        assert!((prediction.mean - 1.0).abs() < 1.0e-12);
        assert!((prediction.variance - 2.0).abs() < 1.0e-12);
        assert!((prediction.weights[0] - 0.5).abs() < 1.0e-12);
        assert!((prediction.weights[1] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn ordinary_kriging_supports_variogram_neighbors_and_variance() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 0.0,
                value: 50.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6)
            .expect("config")
            .with_variogram_model(KrigingVariogramModel::Spherical)
            .with_neighbor_limits(Some(2), 2, None)
            .expect("neighbors");

        let prediction =
            ordinary_kriging_predict(&observations, (10.0, 0.0), config).expect("kriging");

        assert!(prediction.variance >= 0.0);
        assert_eq!(prediction.weights.len(), 2);
        assert_eq!(prediction.neighbor_indices.len(), 2);
    }

    #[test]
    fn ordinary_kriging_system_matches_predict_many() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 5.0,
                value: 55.0,
            },
        ];
        let targets = vec![(0.0, 0.0), (5.0, 2.0), (20.0, 5.0)];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6)
            .expect("config")
            .with_variogram_model(KrigingVariogramModel::Gaussian);

        let system = OrdinaryKrigingSystem::new(&observations, config).expect("system");
        let cached = system.predict_many(&targets).expect("cached");
        let direct = targets
            .iter()
            .map(|target| ordinary_kriging_predict_unchecked(&observations, *target, config))
            .collect::<Result<Vec<_>>>()
            .expect("direct");

        assert_eq!(system.observation_count(), observations.len());
        assert_eq!(system.drift_terms(), 1);
        assert_eq!(cached.len(), direct.len());
        for (cached, direct) in cached.iter().zip(direct.iter()) {
            assert!((cached.mean - direct.mean).abs() < 1.0e-8);
            assert!((cached.variance - direct.variance).abs() < 1.0e-8);
            assert_eq!(cached.neighbor_indices, vec![0, 1, 2]);
        }
    }

    #[test]
    fn ordinary_kriging_system_rejects_local_neighbor_config() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6)
            .expect("config")
            .with_neighbor_limits(Some(1), 1, None)
            .expect("neighbors");

        assert!(OrdinaryKrigingSystem::new(&observations, config).is_err());
    }

    #[test]
    fn ordinary_kriging_leave_one_out_returns_all_observations() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 0.0,
                value: 50.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6).expect("config");

        let diagnostics = ordinary_kriging_leave_one_out(&observations, config).expect("loo");

        assert_eq!(diagnostics.len(), observations.len());
        assert!(diagnostics
            .iter()
            .all(|prediction| prediction.variance >= 0.0));
    }

    #[test]
    fn ordinary_kriging_leave_one_out_reports_original_neighbor_indices() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 0.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 4.0,
                y: 0.0,
                value: 4.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(2.0, 1.0e-6)
            .expect("config")
            .with_neighbor_limits(Some(1), 1, None)
            .expect("neighbors");

        let predictions = ordinary_kriging_leave_one_out(&observations, config).expect("LOO");

        assert_eq!(predictions[0].neighbor_indices, vec![1]);
        assert_eq!(predictions[1].neighbor_indices, vec![0]);
        assert_eq!(predictions[2].neighbor_indices, vec![1]);
    }

    #[test]
    fn empirical_variogram_bins_coordinate_pairs() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 10.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 16.0,
            },
        ];

        let bins = empirical_variogram(&observations, 2, None, 0.0, 1.0).expect("variogram");

        assert!(!bins.is_empty());
        assert_eq!(bins.iter().map(|bin| bin.pair_count).sum::<usize>(), 3);
        assert!(bins.iter().all(|bin| bin.semivariance >= 0.0));
    }

    #[test]
    fn empirical_variogram_keeps_collocated_pairs_for_nugget_evidence() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 0.0,
            },
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 1.0,
            },
        ];

        let bins = empirical_variogram(&observations, 2, None, 0.0, 1.0)
            .expect("variogram with collocated observations");

        assert_eq!(bins.iter().map(|bin| bin.pair_count).sum::<usize>(), 3);
        assert!(bins
            .iter()
            .any(|bin| bin.lag_min == 0.0 && bin.semivariance >= 2.0));
    }

    #[test]
    fn variogram_fit_selects_candidate_config() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 10.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 2.0,
                y: 0.0,
                value: 16.0,
            },
            KrigingObservation {
                x: 3.0,
                y: 0.0,
                value: 20.0,
            },
        ];

        let fit = fit_ordinary_kriging_variogram(
            &observations,
            &[
                KrigingVariogramModel::Exponential,
                KrigingVariogramModel::Spherical,
            ],
            &[1.0, 2.0],
            &[0.0, 0.1],
            &[1.0, 5.0],
            3,
            0.0,
            1.0,
        )
        .expect("fit");

        assert!(fit.weighted_sse.is_finite());
        assert!([1.0, 2.0].contains(&fit.config.range));
        assert!([1.0, 5.0].contains(&fit.config.sill));
    }

    #[test]
    fn kriging_leave_one_out_diagnostics_summarize_residuals() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
            KrigingObservation {
                x: 20.0,
                y: 0.0,
                value: 50.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(5.0, 1.0e-6).expect("config");

        let (predictions, diagnostics) =
            ordinary_kriging_leave_one_out_diagnostics(&observations, config).expect("diagnostics");

        assert_eq!(predictions.len(), observations.len());
        assert_eq!(diagnostics.observation_count, observations.len());
        assert!(diagnostics.rmse >= 0.0);
        assert!((0.0..=1.0).contains(&diagnostics.interval_coverage_95));
    }

    #[test]
    fn universal_kriging_linear_drift_reproduces_plane() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 2.0,
            },
            KrigingObservation {
                x: 1.0,
                y: 0.0,
                value: 5.0,
            },
            KrigingObservation {
                x: 0.0,
                y: 1.0,
                value: 7.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(10.0, 1.0e-9)
            .expect("config")
            .with_drift(KrigingDrift::Linear);

        let prediction =
            ordinary_kriging_predict(&observations, (0.5, 0.5), config).expect("kriging");

        assert!((prediction.mean - 6.0).abs() < 1.0e-6);
    }

    #[test]
    fn ordinary_kriging_rejects_bad_inputs() {
        let config = OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("config");

        assert!(ordinary_kriging_predict(&[], (0.0, 0.0), config).is_err());
        assert!(OrdinaryKrigingConfig::new(0.0, 1.0e-6).is_err());
        assert!(OrdinaryKrigingConfig::new(1.0, -1.0).is_err());
        let invalid_public_config = OrdinaryKrigingConfig {
            range: 1.0,
            nugget: 0.0,
            sill: f64::NAN,
            variogram_model: KrigingVariogramModel::Exponential,
            drift: KrigingDrift::Ordinary,
            anisotropy_angle_degrees: 0.0,
            anisotropy_scaling: 1.0,
            max_neighbors: None,
            min_neighbors: 1,
            max_distance: None,
        };
        assert!(ordinary_kriging_predict(
            &[KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 1.0,
            }],
            (0.0, 0.0),
            invalid_public_config,
        )
        .is_err());
        let impossible_linear_neighbors = OrdinaryKrigingConfig::new(1.0, 0.0)
            .expect("config")
            .with_neighbor_limits(Some(2), 1, None)
            .expect("ordinary neighbor config")
            .with_drift(KrigingDrift::Linear);
        assert!(impossible_linear_neighbors.validate().is_err());
        assert!(ordinary_kriging_predict(
            &[KrigingObservation {
                x: 0.0,
                y: f64::NAN,
                value: 1.0,
            }],
            (0.0, 0.0),
            config,
        )
        .is_err());
    }
}
