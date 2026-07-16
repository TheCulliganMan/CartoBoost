#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_have_unit_sill_at_zero_distance() {
        for kernel in [
            CovarianceKernel::Exponential,
            CovarianceKernel::SquaredExponential,
            CovarianceKernel::Matern32,
            CovarianceKernel::Matern52,
        ] {
            let config = NngpConfig {
                kernel,
                sill: 2.5,
                ..NngpConfig::default()
            };
            assert!((covariance([0.0, 0.0], [0.0, 0.0], config) - 2.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn synthetic_field_recovery_and_variance_shape() {
        let coords = (0..40)
            .map(|i| {
                let x = i as f64 / 4.0;
                [x, (x * 0.7).sin()]
            })
            .collect::<Vec<_>>();
        let y = coords
            .iter()
            .map(|coord| (coord[0] * 0.4).sin() + 0.5 * coord[1])
            .collect::<Vec<_>>();
        let config = NngpConfig {
            kernel: CovarianceKernel::Matern32,
            range: 2.0,
            sill: 1.0,
            nugget: 1.0e-6,
            n_neighbors: 12,
            ..NngpConfig::default()
        };
        let mut model = NearestNeighborGPRegressor::new(config).expect("model");
        model.fit(&coords, &y).expect("fit");
        let train_pred = model.predict(&[coords[10]]).expect("train pred").remove(0);
        let far_pred = model.predict(&[[50.0, 50.0]]).expect("far pred").remove(0);
        assert!((train_pred.mean - y[10]).abs() < 1.0e-3);
        assert!(train_pred.variance >= 0.0);
        assert!(far_pred.variance >= train_pred.variance);
    }

    #[test]
    fn duplicate_coordinate_policy_is_explicit() {
        let mut model = NearestNeighborGPRegressor::new(NngpConfig::default()).expect("model");
        let err = model.fit(&[[0.0, 0.0], [0.0, 0.0]], &[1.0, 2.0]);
        assert!(err.is_err());
    }

    #[test]
    fn kd_tree_neighbors_match_brute_force_ordering() {
        let coords = (0..80)
            .map(|idx| {
                let x = (idx % 11) as f64 * 0.13 + (idx / 11) as f64 * 0.001;
                let y = (idx / 11) as f64 * 0.17 + (idx % 7) as f64 * 0.002;
                [x, y]
            })
            .collect::<Vec<_>>();
        let y = coords
            .iter()
            .map(|coord| coord[0].sin() + coord[1].cos())
            .collect::<Vec<_>>();
        let config = NngpConfig {
            n_neighbors: 9,
            brute_force_threshold: 8,
            anisotropy: Anisotropy {
                angle_degrees: 25.0,
                scaling: 1.7,
            },
            ..NngpConfig::default()
        };
        let mut model = NearestNeighborGPRegressor::new(config).expect("model");
        model.fit(&coords, &y).expect("fit");
        assert_eq!(model.neighbor_index_kind(), "kd_tree");
        let target = [0.57, 0.43];
        let transformed_coords = coords
            .iter()
            .map(|coord| transform_point(*coord, config))
            .collect::<Vec<_>>();
        let expected =
            deterministic_neighbors(&transformed_coords, transform_point(target, config), 9);
        let prediction = model.predict(&[target]).expect("predict").remove(0);
        assert_eq!(prediction.neighbor_indices, expected);
    }

    #[test]
    fn variogram_fit_selects_candidate() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let y = [0.0, 1.0, 1.5, 1.75];
        let bins =
            empirical_semivariogram(&coords, &y, 3, None, Anisotropy::default()).expect("bins");
        let fit = fit_variogram_wls(
            &bins,
            &[CovarianceKernel::Exponential, CovarianceKernel::Matern32],
            &[1.0, 2.0],
            &[0.5, 1.0],
            &[0.0, 0.05],
        )
        .expect("fit");
        assert!(fit.weighted_sse.is_finite());
    }

    #[test]
    fn config_rejects_non_finite_anisotropy_angle() {
        let result = NngpConfig {
            anisotropy: Anisotropy {
                angle_degrees: f64::NAN,
                scaling: 1.0,
            },
            ..NngpConfig::default()
        }
        .validate();

        assert!(matches!(result, Err(GeostatsError::InvalidInput(_))));
    }

    #[test]
    fn empirical_variogram_rejects_invalid_numeric_inputs() {
        let valid_coords = [[0.0, 0.0], [1.0, 0.0]];
        let valid_values = [1.0, 2.0];

        assert!(empirical_semivariogram(
            &[[f64::NAN, 0.0], [1.0, 0.0]],
            &valid_values,
            2,
            None,
            Anisotropy::default(),
        )
        .is_err());
        assert!(empirical_semivariogram(
            &valid_coords,
            &[1.0, f64::INFINITY],
            2,
            None,
            Anisotropy::default(),
        )
        .is_err());
        assert!(empirical_semivariogram(
            &valid_coords,
            &valid_values,
            2,
            Some(-1.0),
            Anisotropy::default(),
        )
        .is_err());
        assert!(empirical_semivariogram(
            &valid_coords,
            &valid_values,
            2,
            None,
            Anisotropy {
                angle_degrees: f64::NAN,
                scaling: 1.0,
            },
        )
        .is_err());
        assert!(empirical_semivariogram(
            &[[f64::MAX, 0.0], [-f64::MAX, 0.0]],
            &valid_values,
            2,
            None,
            Anisotropy::default(),
        )
        .is_err());
    }

    #[test]
    fn empirical_variogram_keeps_collocated_pairs_as_nugget_evidence() {
        let bins = empirical_semivariogram(
            &[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            &[0.0, 2.0, 1.0],
            2,
            None,
            Anisotropy::default(),
        )
        .expect("variogram");

        assert_eq!(bins.iter().map(|bin| bin.pair_count).sum::<usize>(), 3);
        assert!(bins
            .iter()
            .any(|bin| bin.lag_start == 0.0 && bin.semivariance >= 2.0));
    }

    #[test]
    fn variogram_fit_rejects_invalid_bins_candidates_and_objectives() {
        let valid_bin = EmpiricalVariogramBin {
            lag_start: 0.0,
            lag_end: 1.0,
            lag_center: 0.5,
            semivariance: 1.0,
            pair_count: 2,
        };
        let fit =
            |bins: &[EmpiricalVariogramBin], ranges: &[f64], sills: &[f64], nuggets: &[f64]| {
                fit_variogram_wls(
                    bins,
                    &[CovarianceKernel::Exponential],
                    ranges,
                    sills,
                    nuggets,
                )
            };

        assert!(fit(
            &[EmpiricalVariogramBin {
                semivariance: f64::NAN,
                ..valid_bin.clone()
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
        assert!(fit(
            &[EmpiricalVariogramBin {
                pair_count: 0,
                ..valid_bin.clone()
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
        assert!(fit(
            &[EmpiricalVariogramBin {
                lag_center: 2.0,
                ..valid_bin.clone()
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
        assert!(fit(
            std::slice::from_ref(&valid_bin),
            &[f64::NAN],
            &[1.0],
            &[0.0]
        )
        .is_err());
        assert!(fit(std::slice::from_ref(&valid_bin), &[1.0], &[0.0], &[0.0]).is_err());
        assert!(fit(std::slice::from_ref(&valid_bin), &[1.0], &[1.0], &[-1.0]).is_err());
        assert!(fit(
            &[EmpiricalVariogramBin {
                semivariance: f64::MAX,
                ..valid_bin
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
    }

    #[test]
    fn variance_validation_only_clamps_roundoff_scale_negatives() {
        assert_eq!(
            checked_nonnegative(-1.0e-12, 1.0, "variance").expect("tiny roundoff"),
            0.0
        );
        assert!(checked_nonnegative(-1.0e-4, 1.0, "variance").is_err());
        assert!(checked_nonnegative(f64::NAN, 1.0, "variance").is_err());
    }
}
