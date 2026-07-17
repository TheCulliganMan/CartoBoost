#[cfg(test)]
mod tests {
    use super::*;

    fn known_effect_panel(effect: f64) -> GeoCausalPanel {
        let mut rows = Vec::new();
        for unit in ["treated", "control_a", "control_b", "control_c"] {
            for time in 0..8 {
                let post = time >= 4;
                let base = 10.0 + time as f64;
                let unit_shift = match unit {
                    "treated" | "control_a" => 1.0,
                    "control_b" => 0.8,
                    _ => 1.2,
                };
                rows.push(GeoCausalRow {
                    unit_id: unit.to_string(),
                    time: format!("2026-01-{time:02}"),
                    outcome: base
                        + unit_shift
                        + if unit == "treated" && post {
                            effect
                        } else {
                            0.0
                        },
                    treatment: unit == "treated" && post,
                    covariates: BTreeMap::new(),
                    latitude: Some(40.0 + if unit == "treated" { 0.0 } else { 1.0 }),
                    longitude: Some(-73.0),
                    region_id: Some(unit.to_string()),
                });
            }
        }
        GeoCausalPanel::new(
            rows,
            vec![SpatialWeight {
                from_unit: "treated".to_string(),
                to_unit: "control_a".to_string(),
                weight: 1.0,
            }],
        )
        .unwrap()
    }

    #[test]
    fn synthetic_did_recovers_known_effect() {
        let panel = known_effect_panel(5.0);
        let mut estimator = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: "2026-01-04".to_string(),
            seed: 7,
        });
        estimator.fit(panel).unwrap();
        assert!((estimator.estimate_effect().unwrap().effect - 5.0).abs() < 0.25);
    }

    #[test]
    fn zero_effect_placebo_is_centered_near_zero_and_deterministic() {
        let panel = known_effect_panel(0.0);
        let mut a = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: "2026-01-04".to_string(),
            seed: 42,
        });
        a.fit(panel.clone()).unwrap();
        let first = a.placebo_test(4).unwrap();
        let mut b = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: "2026-01-04".to_string(),
            seed: 42,
        });
        b.fit(panel).unwrap();
        assert_eq!(first, b.placebo_test(4).unwrap());
        assert!(mean(first).abs() < 0.5);
    }

    #[test]
    fn spillover_warning_fires_for_adjacent_units() {
        let diagnostics = spillover_diagnostics(&known_effect_panel(1.0));
        assert!(!diagnostics.warnings.is_empty());
        assert_eq!(diagnostics.adjacent_treated_control_pairs.len(), 1);
    }

    #[test]
    fn causal_representation_improves_heldout_region_and_warns_on_identification() {
        let mut features = Vec::new();
        let mut outcomes = Vec::new();
        let mut regions = Vec::new();
        for (region, shift) in [("a", 0.0), ("b", 3.0), ("c", -4.0)] {
            for idx in 0..8 {
                let stable = idx as f64 / 4.0;
                features.push(vec![stable + shift, stable * 0.5 + shift]);
                outcomes.push(2.0 + 1.5 * stable);
                regions.push(region.to_string());
            }
        }
        let report = causal_representation_report(&features, &outcomes, &regions, "c").unwrap();

        assert!(report.invariant_rmse < report.raw_rmse);
        assert!(report.improvement > 0.0);
        assert!(report.losses.contains_key("domain_adversarial_loss"));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("does not prove causal identification")));
        assert_eq!(
            report.metadata.get("supplements").map(String::as_str),
            Some("SyntheticDIDEstimator,GeoExperimentDesigner")
        );
    }
}
