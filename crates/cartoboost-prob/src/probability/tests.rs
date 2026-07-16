#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_conformal_covers_synthetic_holdout_without_holdout_training() {
        let calibration_prediction = vec![10.0; 20];
        let calibration_actual = (0..20)
            .map(|idx| 10.0 + if idx % 2 == 0 { 1.0 } else { -1.0 } * (idx % 5) as f64)
            .collect::<Vec<_>>();
        let order = SplitOrder {
            train_end_exclusive: 10,
            calibration_start: 10,
            calibration_end_exclusive: 30,
            test_start: 30,
        };
        let q = split_conformal_residual_quantile(
            &calibration_actual,
            &calibration_prediction,
            0.1,
            order,
        )
        .unwrap();
        let lower = vec![10.0 - q; 10];
        let upper = vec![10.0 + q; 10];
        let actual = vec![9.0, 11.0, 10.0, 8.0, 12.0, 10.5, 9.5, 11.5, 8.5, 12.5];
        assert!(interval_coverage(&actual, &lower, &upper).unwrap() >= 0.9);
    }

    #[test]
    fn rolling_origin_uses_only_past_cutoff_residuals() {
        let actual = vec![10.0, 11.0, 14.0, 50.0];
        let prediction = vec![10.0, 10.0, 10.0, 10.0];
        let qs = rolling_origin_conformal_residual_quantiles(&actual, &prediction, &[2, 3], 0.1)
            .unwrap();
        assert_eq!(qs, vec![1.0, 4.0]);
    }

    #[test]
    fn distributional_metrics_validate_and_score() {
        let actual = vec![1.0, 2.0];
        let quantiles = vec![0.1, 0.5, 0.9];
        let predictions = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]];
        assert!(crps_approximation(&actual, &quantiles, &predictions).unwrap() >= 0.0);
        let pits = pit_bins(&actual, &quantiles, &predictions, 5).unwrap();
        assert_eq!(pits.counts.iter().sum::<usize>(), 2);
        let wis = weighted_interval_score(
            &actual,
            &[1.0, 2.0],
            &[(0.2, vec![0.0, 1.0], vec![2.0, 3.0])],
        )
        .unwrap();
        assert!(wis >= 0.0);
    }

    #[test]
    fn conditional_flow_head_emits_joint_distribution_outputs_and_metrics() {
        let hidden = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.5],
            vec![2.0, 0.0],
            vec![3.0, 1.0],
        ];
        let residuals = vec![-0.5, 0.1, 0.4, 0.9];
        let artifact =
            conditional_flow_fit_json(&hidden, &residuals, &[0.05, 0.5, 0.95], 8).unwrap();
        let output_json =
            conditional_flow_predict_json(&artifact, &hidden, Some(&residuals)).unwrap();
        let output: FlowPrediction = serde_json::from_str(&output_json).unwrap();

        assert_eq!(output.samples.len(), hidden.len());
        assert_eq!(output.samples[0].len(), 8);
        assert_eq!(output.marginal_quantiles[0].len(), 3);
        assert_eq!(output.joint_scenario_paths.len(), 8);
        assert_eq!(output.log_likelihood.len(), hidden.len());
        assert!(output
            .tail_risk_metrics
            .contains_key("expected_shortfall_low"));
        assert!(output.metrics.contains_key("crps"));
        assert!(output.metrics.contains_key("pinball_median"));
        assert!(output.metrics.contains_key("interval_coverage"));
        assert!(output.metrics.contains_key("joint_path_calibration"));
        assert!(output.metrics.contains_key("tail_event_calibration"));
    }

    #[test]
    fn diffusion_scenario_generator_reports_shape_variance_and_spatial_correlation() {
        let point_forecast = vec![
            vec![10.0, 12.0, 13.0],
            vec![11.0, 12.5, 14.0],
            vec![12.0, 13.0, 15.0],
        ];
        let edges = vec![
            DiffusionEdge {
                source: 0,
                target: 1,
                weight: 1.0,
            },
            DiffusionEdge {
                source: 1,
                target: 2,
                weight: 0.7,
            },
        ];
        let output_json =
            diffusion_scenario_generate_json(&point_forecast, &edges, 6, 2, 0.4).unwrap();
        let output: DiffusionScenarioPrediction = serde_json::from_str(&output_json).unwrap();

        assert_eq!(output.scenarios.len(), 6);
        assert_eq!(output.scenarios[0].len(), point_forecast.len());
        assert_eq!(output.scenarios[0][0].len(), point_forecast[0].len());
        assert_eq!(output.scenario_mean.len(), point_forecast.len());
        assert_eq!(output.scenario_variance[0].len(), point_forecast[0].len());
        assert!(output.spatial_correlation.is_finite());
        assert!(output
            .point_forecast_comparison
            .contains_key("mean_absolute_delta"));
        assert_eq!(
            output.metadata.get("capability_tier").map(String::as_str),
            Some("experimental")
        );
        assert_eq!(
            output.metadata.get("auto_geo_enabled").map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn benchmark_report_fields_group_coverage_by_horizon_and_block() {
        let report = benchmark_calibration_report_fields(
            &[10.0, 12.0, 20.0, 25.0],
            &[9.0, 11.0, 15.0, 24.0],
            &[11.0, 13.0, 22.0, 24.5],
            &[1, 1, 2, 2],
            &[
                "pickup_142".into(),
                "pickup_142".into(),
                "pickup_236".into(),
                "pickup_236".into(),
            ],
            Some(0.05),
        )
        .unwrap();
        assert_eq!(report.coverage_by_horizon[&1], 1.0);
        assert_eq!(report.coverage_by_spatial_block["pickup_236"], 0.5);
        assert_eq!(report.residual_morans_i_after_calibration, Some(0.05));
    }

    #[test]
    fn nearest_calibration_residuals_use_local_neighbors() {
        let q = nearest_calibration_residual_quantiles(
            &[10.0, 20.0, 100.0],
            &[9.0, 18.0, 90.0],
            &[0.0, 1.0, 100.0],
            &[0.0, 1.0, 100.0],
            &[0.1, 99.0],
            &[0.1, 99.0],
            1,
            0.1,
            SplitOrder {
                train_end_exclusive: 1,
                calibration_start: 1,
                calibration_end_exclusive: 4,
                test_start: 4,
            },
        )
        .unwrap();
        assert_eq!(q, vec![1.0, 10.0]);
    }
}
