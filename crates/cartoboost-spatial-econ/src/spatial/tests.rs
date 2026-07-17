#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostics(
        n_samples: usize,
        n_features: usize,
        rho: Option<f64>,
        lambda: Option<f64>,
    ) -> SpatialDiagnostics {
        SpatialDiagnostics {
            residual_morans_i: 0.0,
            log_likelihood: None,
            aic: None,
            bic: None,
            rho,
            lambda,
            sigma2: 1.0,
            n_samples,
            n_features,
            isolated_rows: Vec::new(),
            direct_effects: None,
            indirect_effects: None,
            total_effects: None,
        }
    }

    fn chain_weights() -> SpatialWeights {
        spatial_weights_from_coo(
            4,
            4,
            vec![0, 1, 1, 2, 2, 3],
            vec![1, 0, 2, 1, 3, 2],
            vec![1.0; 6],
            true,
        )
        .expect("weights")
    }

    fn ring_weights(n: usize) -> SpatialWeights {
        let mut rows = Vec::with_capacity(2 * n);
        let mut cols = Vec::with_capacity(2 * n);
        for row in 0..n {
            rows.extend([row, row]);
            cols.extend([(row + n - 1) % n, (row + 1) % n]);
        }
        spatial_weights_from_coo(n, n, rows, cols, vec![1.0; 2 * n], true).expect("ring weights")
    }

    fn fixture_x() -> Vec<Vec<f64>> {
        [0.0, 1.0, 4.0, 2.0, 7.0, 3.0, 9.0, 5.0, 11.0, 6.0, 10.0, 8.0]
            .into_iter()
            .map(|value| vec![value])
            .collect()
    }

    fn fixture_innovations() -> Vec<f64> {
        vec![
            0.30, -0.20, 0.10, -0.35, 0.25, 0.05, -0.15, 0.40, -0.25, 0.15, -0.05, -0.10,
        ]
    }

    fn spatial_lag_target(
        x: &[Vec<f64>],
        weights: &SpatialWeights,
        rho: f64,
        beta: f64,
        theta: f64,
    ) -> Vec<f64> {
        let wx = sparse_matrix_lag(weights, x).expect("WX");
        let innovations = fixture_innovations();
        let structural_mean: Vec<f64> = x
            .iter()
            .zip(wx)
            .zip(innovations)
            .map(|((row, lagged), innovation)| 1.5 + beta * row[0] + theta * lagged[0] + innovation)
            .collect();
        solve_spatial_lag_mean(structural_mean, rho, weights).expect("known SAR target")
    }

    #[test]
    fn spatial_lag_fits_known_toy_system() {
        let weights = ring_weights(12);
        let x = fixture_x();
        let y = spatial_lag_target(&x, &weights, 0.35, 1.2, 0.0);
        let model =
            SpatialRegressionModel::fit(SpatialModelKind::SpatialLag, x.clone(), y, &weights)
                .expect("fit");
        let pred = model.predict(x, &weights).expect("predict");
        assert_eq!(pred.len(), 12);
        assert!(model.diagnostics().rho.is_some());
        assert!(model.diagnostics().log_likelihood.is_some());
        assert!(model.diagnostics().direct_effects.is_some());
        assert!(model.diagnostics().residual_morans_i.is_finite());
        let rho = model.diagnostics().rho.expect("rho");
        let expected_likelihood = gaussian_log_likelihood(
            &model.residuals,
            spatial_log_abs_determinant(rho, &weights).expect("Jacobian"),
        )
        .expect("likelihood");
        assert!(
            (model.diagnostics().log_likelihood.expect("likelihood") - expected_likelihood).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn spatial_error_reports_lambda() {
        let weights = ring_weights(12);
        let x = fixture_x();
        let disturbances = solve_spatial_lag_mean(fixture_innovations(), 0.4, &weights)
            .expect("known SEM disturbances");
        let y: Vec<f64> = x
            .iter()
            .zip(disturbances)
            .map(|(row, disturbance)| 2.0 + 1.1 * row[0] + disturbance)
            .collect();
        let model = SpatialRegressionModel::fit(SpatialModelKind::SpatialError, x, y, &weights)
            .expect("fit");
        assert!(model.diagnostics().lambda.is_some());
        assert!(model.diagnostics().log_likelihood.is_some());
    }

    #[test]
    fn spatial_two_stage_least_squares_reports_rho() {
        let weights = chain_weights();
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.6, 4.4, 5.8];
        let model = SpatialRegressionModel::fit(
            SpatialModelKind::SpatialTwoStageLeastSquares,
            x.clone(),
            y,
            &weights,
        )
        .expect("fit");
        assert!(model.diagnostics().rho.is_some());
        assert!(model.diagnostics().log_likelihood.is_none());
        assert!(model.diagnostics().aic.is_none());
        assert!(model.diagnostics().bic.is_none());
        assert_eq!(model.predict(x, &weights).unwrap().len(), 4);
    }

    #[test]
    fn durbin_reports_effects_and_roundtrips() {
        let weights = ring_weights(12);
        let x = fixture_x();
        let y = spatial_lag_target(&x, &weights, 0.25, 1.1, 0.4);
        let model =
            SpatialRegressionModel::fit(SpatialModelKind::SpatialDurbin, x.clone(), y, &weights)
                .expect("fit");
        assert!(model.diagnostics().total_effects.is_some());
        let path = std::env::temp_dir().join("cartoboost-spatial-econ-test.json");
        model.save(&path).expect("save");
        let loaded = SpatialRegressionModel::load(&path).expect("load");
        let before = model.predict(x.clone(), &weights).unwrap();
        let after = loaded.predict(x, &weights).unwrap();
        assert!(before
            .iter()
            .zip(after)
            .all(|(left, right)| (left - right).abs() < 1.0e-12));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn durbin_effects_use_exact_spatial_multiplier() {
        let weights = spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        let (direct, indirect, total) =
            effects(Some(0.25), &[2.0], &[0.5], &weights).expect("effects");
        assert!((direct.unwrap()[0] - 2.266_666_666_666_666_6).abs() < 1.0e-12);
        assert!((total.unwrap()[0] - 3.333_333_333_333_333_5).abs() < 1.0e-12);
        assert!((indirect.unwrap()[0] - 1.066_666_666_666_666_9).abs() < 1.0e-12);
    }

    #[test]
    fn spatial_lag_prediction_solves_reduced_form_mean() {
        let weights = spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        let model = SpatialRegressionModel {
            kind: SpatialModelKind::SpatialLag,
            intercept: 1.0,
            coefficients: vec![2.0],
            durbin_coefficients: Vec::new(),
            rho: Some(0.25),
            lambda: None,
            fitted_values: Vec::new(),
            residuals: Vec::new(),
            diagnostics: diagnostics(2, 1, Some(0.25), None),
        };
        let prediction = model
            .predict(vec![vec![0.0], vec![1.0]], &weights)
            .expect("reduced-form prediction");
        let denominator = 1.0 - 0.25_f64.powi(2);
        assert!((prediction[0] - (1.0 + 0.25 * 3.0) / denominator).abs() < 1.0e-12);
        assert!((prediction[1] - (3.0 + 0.25) / denominator).abs() < 1.0e-12);
    }

    #[test]
    fn spatial_error_prediction_does_not_reuse_training_innovations() {
        let weights = spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        let model = SpatialRegressionModel {
            kind: SpatialModelKind::SpatialError,
            intercept: 1.0,
            coefficients: vec![2.0],
            durbin_coefficients: Vec::new(),
            rho: None,
            lambda: Some(0.75),
            fitted_values: vec![101.0, -97.0],
            residuals: vec![100.0, -100.0],
            diagnostics: diagnostics(2, 1, None, Some(0.75)),
        };
        assert_eq!(
            model
                .predict(vec![vec![0.0], vec![1.0]], &weights)
                .expect("SEM mean"),
            vec![1.0, 3.0]
        );
    }

    #[test]
    fn singular_design_is_not_hidden_by_ridge_regularization() {
        let weights = ring_weights(6);
        let x = vec![vec![1.0, 2.0]; 6];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.5, -0.5];
        assert!(matches!(
            SpatialRegressionModel::fit(SpatialModelKind::Ols, x, y, &weights),
            Err(SpatialEconError::SingularSystem)
        ));
    }

    #[test]
    fn saturated_durbin_fit_fails_clearly() {
        let error = SpatialRegressionModel::fit(
            SpatialModelKind::SpatialDurbin,
            vec![vec![1.0], vec![2.0], vec![4.0], vec![8.0]],
            vec![2.0, 3.0, 6.0, 10.0],
            &chain_weights(),
        )
        .expect_err("saturated likelihood must fail");
        assert!(error.to_string().contains("more observations"));
    }

    #[test]
    fn invalid_weights_fail_clearly() {
        let err = spatial_weights_from_coo(2, 3, vec![0], vec![1], vec![1.0], false)
            .expect_err("must fail");
        assert!(err.to_string().contains("square"));

        let err = spatial_weights_from_coo(2, 2, vec![0], vec![0], vec![1.0], false)
            .expect_err("self weights must fail");
        assert!(err.to_string().contains("zero diagonal"));
    }

    #[test]
    fn isolated_nodes_are_recorded() {
        let weights = spatial_weights_from_coo(3, 3, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        assert_eq!(weights.isolated_nodes(), vec![2]);
    }
}
