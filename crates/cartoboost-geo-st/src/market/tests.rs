#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> MarketPanelFrame {
        let primary = (0..21)
            .map(|day| vec![10.0 + day as f64 * 0.1, 12.0 + day as f64 * 0.1, 8.0])
            .collect();
        let secondary = (0..21)
            .map(|day| vec![20.0 + day as f64, 18.0 + day as f64, 5.0])
            .collect();
        MarketPanelFrame::new(
            vec!["a:b".into(), "a:c".into(), "b:a".into()],
            (0..21).collect(),
            vec!["benchmark".into(), "volume".into()],
            primary,
            secondary,
            vec!["a".into(), "a".into(), "b".into()],
            vec!["b".into(), "c".into(), "a".into()],
            vec![vec![]; 3],
            vec![[0.0; 4]; 3],
            vec![vec![]; 21],
            None,
            vec![],
            vec![],
            2,
            "daily".into(),
        )
        .unwrap()
    }
    #[test]
    fn learns_sparse_directional_relationships() {
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&frame()).unwrap();
        let edges = model.relationships().unwrap();
        assert_eq!(model.neural_embeddings.len(), 3);
        assert_eq!(model.neural_embeddings[0].len(), 16);
        assert!(edges.len() <= 8 * 3);
        assert!(edges
            .iter()
            .any(|edge| edge.kinds.contains(&RelationshipKind::SharedOrigin)));
        assert!(edges
            .iter()
            .any(|edge| edge.kinds.contains(&RelationshipKind::ReverseLane)));
    }

    #[test]
    fn top_k_is_enforced_per_source_lane() {
        let mut model = MarketStructureForecaster::new(MarketStructureConfig {
            top_k: 1,
            ..MarketStructureConfig::default()
        })
        .unwrap();
        model.fit(&frame()).unwrap();
        let edges = model.relationships().unwrap();
        for lane in ["a:b", "a:c", "b:a"] {
            assert!(
                edges
                    .iter()
                    .filter(|edge| edge.source_lane_id == lane)
                    .count()
                    <= 1
            );
        }
    }

    #[test]
    fn rejects_missing_geography_and_unavailable_label_cutoff() {
        let mut invalid_geo = frame();
        invalid_geo.coordinates[0][0] = f64::NAN;
        assert!(invalid_geo.validate().is_err());

        let mut invalid_label = frame();
        invalid_label.expert_labels.push(ExpertEventLabel {
            lane_id: "a:b".into(),
            timestamp: 999,
            shift: MarketShiftKind::Market,
            version: "review-1".into(),
        });
        assert!(invalid_label.validate().is_err());
    }
    #[test]
    fn predicts_and_explains() {
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&frame()).unwrap();
        assert_eq!(model.predict(2, None).unwrap().len(), 6);
        assert_eq!(model.nowcast().unwrap().len(), 3);
        let weekly = model.weekly_rollups(2, None).unwrap();
        assert_eq!(weekly.len(), 3);
        assert_eq!(weekly[0].days, 2);
        let explorer = model.explorer_payload(2).unwrap();
        assert_eq!(explorer["lanes"].as_array().unwrap().len(), 3);
        assert_eq!(explorer["forecasts"].as_array().unwrap().len(), 6);
        assert_eq!(explorer["explanations"].as_array().unwrap().len(), 3);
        assert!(explorer["kernels"].is_array());
    }
    #[test]
    fn rejects_unknown_expert_lane() {
        let mut input = frame();
        input.expert_priors.push(ExpertRelationshipPrior {
            version: "1".into(),
            source_lane_id: "missing".into(),
            target_lane_id: "a:b".into(),
            allowed: true,
            weight: 1.0,
        });
        assert!(input.validate().is_err());
    }

    #[test]
    fn expert_ban_and_artifact_round_trip_are_preserved() {
        let mut input = frame();
        input.expert_priors.push(ExpertRelationshipPrior {
            version: "review-1".into(),
            source_lane_id: "a:b".into(),
            target_lane_id: "a:c".into(),
            allowed: false,
            weight: 0.0,
        });
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert!(!model
            .relationships()
            .unwrap()
            .iter()
            .any(|edge| edge.source_lane_id == "a:b" && edge.target_lane_id == "a:c"));
        let restored =
            MarketStructureForecaster::from_json_string(&model.to_json_string().unwrap()).unwrap();
        assert_eq!(
            model.predict(1, None).unwrap(),
            restored.predict(1, None).unwrap()
        );
    }

    #[test]
    fn recorded_expert_label_is_preserved_without_overriding_model_assessment() {
        let mut input = frame();
        input.expert_labels.push(ExpertEventLabel {
            lane_id: "a:b".into(),
            timestamp: 20,
            shift: MarketShiftKind::LocalOrMix,
            version: "review-1".into(),
        });
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        let explanation = model.nowcast().unwrap().remove(0);
        assert_eq!(explanation.shift, MarketShiftKind::NoShift);
        assert_eq!(
            explanation.expert_label,
            Some(ExpertEventLabel {
                lane_id: "a:b".into(),
                timestamp: 20,
                shift: MarketShiftKind::LocalOrMix,
                version: "review-1".into(),
            })
        );
    }

    #[test]
    fn shared_shift_is_market_but_isolated_mix_shock_stays_local() {
        let mut input = frame();
        for time in 0..input.primary.len() {
            input.primary[time][0] = 10.0;
            input.primary[time][1] = 12.0;
        }
        let last = input.primary.len() - 1;
        input.primary[last][0] = 30.0;
        input.primary[last][1] = 36.0;
        let mut model = MarketStructureForecaster::new(MarketStructureConfig {
            graph_strength: 0.8,
            local_strength: 0.1,
            ..MarketStructureConfig::default()
        })
        .unwrap();
        model.fit(&input).unwrap();
        let shared = model.nowcast().unwrap();
        assert_eq!(shared[0].shift, MarketShiftKind::Market);

        input.primary[last][1] = 12.0;
        input.mix = Some(
            (0..input.primary.len())
                .map(|time| {
                    vec![
                        vec![if time == last { 1.0 } else { 0.0 }],
                        vec![0.0],
                        vec![0.0],
                    ]
                })
                .collect(),
        );
        let mut local = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        local.fit(&input).unwrap();
        let isolated = local.nowcast().unwrap();
        assert_eq!(isolated[0].shift, MarketShiftKind::LocalOrMix);
        assert_eq!(isolated[1].shift, MarketShiftKind::NoShift);
    }

    #[test]
    fn known_future_calendar_is_required_and_changes_forecast_path() {
        let mut input = frame();
        input.calendar = (0..input.timestamps.len())
            .map(|index| vec![if index % 2 == 0 { 1.0 } else { 0.0 }])
            .collect();
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert!(model.predict(1, None).is_err());
        assert!(model.explorer_payload(1).is_ok());
        let inactive = model.predict(1, Some(&[vec![0.0]])).unwrap();
        let active = model.predict(1, Some(&[vec![1.0]])).unwrap();
        assert_ne!(inactive[0].primary, active[0].primary);
    }

    #[test]
    fn interval_calibration_uses_a_train_only_trailing_origin() {
        let mut input = frame();
        for (time, row) in input.primary.iter_mut().enumerate() {
            row[0] += time as f64 * 0.05;
        }
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert!(model.interval_calibration_multiplier.is_finite());
        assert!(model.interval_calibration_multiplier >= 1.0);
        let prediction = model.predict(1, None).unwrap();
        assert!(prediction[0].primary_lower < prediction[0].primary_upper);
        assert!(model.joint_heads.is_some());
        assert_eq!(
            model.joint_heads.as_ref().unwrap().primary_quantiles.len(),
            model.config.quantile_levels.len()
        );
    }

    #[test]
    fn joint_heads_reject_invalid_quantile_levels() {
        assert!(MarketStructureForecaster::new(MarketStructureConfig {
            quantile_levels: vec![0.5, 0.1],
            ..MarketStructureConfig::default()
        })
        .is_err());
    }

    #[test]
    fn unobserved_lane_uses_explicit_hierarchy_without_a_filled_observation() {
        let mut input = frame();
        for row in &mut input.primary {
            row[2] = f64::NAN;
        }
        input.hierarchy_groups[2] = vec!["parent:a".into()];
        input.hierarchy_groups[0] = vec!["parent:a".into()];
        let mut model = MarketStructureForecaster::new(MarketStructureConfig::default()).unwrap();
        model.fit(&input).unwrap();
        assert_eq!(model.predict(1, None).unwrap().len(), 3);
        let explanation = model.nowcast().unwrap();
        assert_eq!(explanation[2].observed_primary, None);
        assert_eq!(explanation[2].support, MarketSupportKind::Hierarchy);
        assert_eq!(explanation[2].shift, MarketShiftKind::NoShift);
    }
}
