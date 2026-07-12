use cartoboost_core::forecasting::{
    calendar_profile_candidate_prediction, candidate_complexity_rank,
    forecast_magnitude_guard_allows, include_autostats_candidate, lag_origin_consistency_guard,
    native_auto_raw_candidate_is_confident, relative_loss_displacement_allowed, requires_lag_spine,
    seasonal_naive_candidate_prediction, selectable_candidate_names, shared_candidate_names,
    stable_magnitude_candidate_choice, trend_candidate_prediction, validation_ensemble_weights,
    validation_unavailable_candidate_choice, AutoForecastConfig, AutoForecastModel,
    CandidateSelectionPolicy, ExpertScore, ForecastEnsemble, ForecastFrame, ForecastFrequency,
    ForecastMetricSet, ForecastObjective, ForecastRow, Forecaster, LagFeatureConfig,
    NaiveForecaster, RuleBasedGating, RuleBasedGatingGuardrails, SeasonalNaiveForecaster,
    ValidationScoreTable,
};
use chrono::NaiveDate;
use std::collections::BTreeMap;

#[test]
fn rule_based_gating_converts_validation_errors_to_weights() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("naive", "rmse", 4.0),
        ExpertScore::global("seasonal", "rmse", 2.0),
    ])
    .expect("score table");
    let gating = RuleBasedGating::new("rmse", table).expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(weights.len(), 2);
    assert!((weights["naive"] - (1.0 / 3.0)).abs() < 1e-12);
    assert!((weights["seasonal"] - (2.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn forecast_magnitude_guard_rejects_implausible_candidate_scale() {
    assert!(forecast_magnitude_guard_allows(999.0, 2.0).expect("guard"));
    assert!(!forecast_magnitude_guard_allows(2.0e9, 20.0).expect("guard"));
}

#[test]
fn lag_spine_guard_keeps_low_frequency_but_allows_high_frequency_validation() {
    assert!(requires_lag_spine("low_frequency_competition", 1, 14));
    assert!(requires_lag_spine("low_frequency_competition", 12, 18));
    assert!(!requires_lag_spine("low_frequency_competition", 24, 48));
    assert!(!requires_lag_spine("classical_competition_full", 12, 18));
}

#[test]
fn native_candidate_primitives_match_benchmark_formulas() {
    let values = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];

    assert_eq!(
        seasonal_naive_candidate_prediction(&values, 4).expect("seasonal"),
        18.0
    );
    assert_eq!(
        trend_candidate_prediction(&values, 2, 4, "half_drift").expect("trend"),
        26.0
    );
    assert_eq!(
        trend_candidate_prediction(&values, 2, 4, "seasonal_cycle_drift_050").expect("trend"),
        20.0
    );
    assert_eq!(
        calendar_profile_candidate_prediction(
            &values,
            &[1, 2, 3, 4, 1, 2, 3, 4],
            2,
            "day_of_month",
            None,
        )
        .expect("calendar"),
        16.0
    );
    assert_eq!(
        calendar_profile_candidate_prediction(
            &values,
            &[1, 2, 3, 4, 1, 2, 3, 4],
            9,
            "elapsed_phase",
            Some(4),
        )
        .expect("calendar"),
        14.0
    );
}

#[test]
fn native_validation_ensemble_weights_use_best_four_inverse_square_losses() {
    let weights = validation_ensemble_weights(&BTreeMap::from([
        ("b".to_string(), 2.0),
        ("a".to_string(), 1.0),
        ("c".to_string(), 4.0),
        ("d".to_string(), 8.0),
        ("e".to_string(), 16.0),
    ]))
    .expect("weights");

    assert!(!weights.contains_key("e"));
    assert!((weights.values().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(weights["a"] > weights["b"]);
}

#[test]
fn native_validation_ensemble_weights_handle_zero_and_extreme_losses() {
    let perfect = validation_ensemble_weights(&BTreeMap::from([
        ("perfect_a".to_string(), 0.0),
        ("perfect_b".to_string(), 0.0),
        ("other".to_string(), 1.0),
    ]))
    .expect("perfect weights");
    assert_eq!(
        perfect,
        BTreeMap::from([
            ("perfect_a".to_string(), 0.5),
            ("perfect_b".to_string(), 0.5),
        ])
    );

    let extreme = validation_ensemble_weights(&BTreeMap::from([
        ("a".to_string(), 1.0e300),
        ("b".to_string(), 2.0e300),
    ]))
    .expect("extreme finite scores");
    assert!((extreme.values().sum::<f64>() - 1.0).abs() < 1.0e-12);
    assert!(extreme["a"] > extreme["b"]);
}

#[test]
fn native_validation_ensemble_weights_reject_invalid_scores() {
    assert!(validation_ensemble_weights(&BTreeMap::new()).is_err());
    assert!(
        validation_ensemble_weights(&BTreeMap::from([("bad".to_string(), f64::NAN,)])).is_err()
    );
}

#[test]
fn native_candidate_roster_policy_matches_forecast_benchmark_rosters() {
    let shared = shared_candidate_names();
    assert_eq!(shared[0], "shared_seasonal_base");
    assert!(shared.contains(&"shared_calendar_elapsed_phase".to_string()));

    let classical = selectable_candidate_names("cartoboost_auto_forecast", "classical_competition");
    assert!(classical.contains(&"cartoboost_lag".to_string()));
    assert!(classical.contains(&"cartoboost_autostats_bank".to_string()));

    let hierarchical =
        selectable_candidate_names("cartoboost_auto_forecast", "hierarchical_reconciliation");
    assert!(hierarchical.contains(&"shared_elapsed_phase_total_reconciled_035".to_string()));
    assert!(hierarchical.contains(&"shared_reconciled_autostats_blend".to_string()));
    assert!(!hierarchical.contains(&"shared_state_reconciled_auto".to_string()));

    let rank_portfolio = selectable_candidate_names("cartoboost_auto_forecast", "rank_portfolio");
    assert!(rank_portfolio.contains(&"shared_elapsed_phase_rank_blend".to_string()));
    assert!(rank_portfolio.contains(&"shared_market_neutral_zero".to_string()));
    assert!(
        !selectable_candidate_names("cartoboost_lag", "rank_portfolio")
            .contains(&"shared_elapsed_phase_rank_blend".to_string())
    );
}

#[test]
fn native_autostats_and_raw_auto_policy_are_rust_owned() {
    assert!(include_autostats_candidate(
        "classical_competition_full",
        12,
        18
    ));
    assert!(include_autostats_candidate(
        "classical_competition_full",
        4,
        8
    ));
    assert!(include_autostats_candidate("classical_competition", 12, 18));
    assert!(!include_autostats_candidate(
        "classical_competition",
        24,
        48
    ));
    assert!(include_autostats_candidate(
        "hierarchical_reconciliation",
        1,
        28
    ));

    assert!(native_auto_raw_candidate_is_confident(
        Some("cartoboost_raw"),
        Some(0.50)
    ));
    assert!(!native_auto_raw_candidate_is_confident(
        Some("cartoboost_raw"),
        Some(0.49)
    ));
    assert!(!native_auto_raw_candidate_is_confident(
        Some("cartoboost_residual_blend"),
        Some(0.90)
    ));
}

#[test]
fn relative_loss_displacement_requires_material_gain() {
    assert!(
        !relative_loss_displacement_allowed(0.2000000000, 0.1999999995, 0.005).expect("tiny gain")
    );
    assert!(
        relative_loss_displacement_allowed(0.2000000000, 0.1980000000, 0.005)
            .expect("material gain")
    );
    assert!(
        relative_loss_displacement_allowed(-0.004, -0.028, 0.005).expect("signed objective gain")
    );
    assert!(!relative_loss_displacement_allowed(-0.028, -0.004, 0.005)
        .expect("signed objective regression"));
    assert!(relative_loss_displacement_allowed(0.2, f64::NAN, 0.005).is_err());
    assert!(relative_loss_displacement_allowed(0.2, 0.1, 1.5).is_err());
}

#[test]
fn lowest_finite_candidate_policy_accepts_negative_losses() {
    let policy = CandidateSelectionPolicy::new("rank_portfolio", Some(1)).expect("policy");
    let selected = policy
        .select(&BTreeMap::from([
            ("cartoboost_auto_forecast".to_string(), -0.012),
            ("cartoboost_lag".to_string(), -0.017),
            ("shared_seasonal_drift".to_string(), 0.006),
        ]))
        .expect("selection");

    assert_eq!(selected.candidate, "cartoboost_lag");
}

#[test]
fn native_origin_consistency_guard_reports_lag_fallback_diagnostics() {
    let guarded = lag_origin_consistency_guard(
        "cartoboost_auto_forecast",
        "synthetic",
        &[1.0, 1.0, 1.0],
        &[0.8, 1.1, 0.7],
    )
    .expect("guard")
    .expect("diagnostic");

    assert_eq!(guarded["candidate"], "cartoboost_auto_forecast");
    assert_eq!(
        guarded["reason"],
        "candidate_lost_at_least_one_inner_origin_to_lag"
    );
    assert_eq!(guarded["origin_count"], 3);
    assert_eq!(guarded["losing_origin_count"], 1);
    assert!((guarded["min_relative_gain_vs_lag"].as_f64().unwrap() + 0.1).abs() < 1.0e-12);

    assert!(lag_origin_consistency_guard(
        "cartoboost_auto_forecast",
        "synthetic",
        &[1.0, 1.0, 1.0],
        &[0.8, 0.9, 0.7],
    )
    .expect("guard")
    .is_none());
    assert!(lag_origin_consistency_guard(
        "cartoboost_auto_forecast",
        "classical_competition",
        &[1.0, 1.0, 1.0],
        &[0.8, 1.1, 0.7],
    )
    .expect("guard")
    .is_none());
}

#[test]
fn stable_magnitude_candidate_choice_filters_implausible_forecast_scales() {
    let selected = stable_magnitude_candidate_choice(
        "shared_elapsed_phase_total_reconciled_050",
        &BTreeMap::from([
            (
                "shared_elapsed_phase_total_reconciled_050".to_string(),
                0.60,
            ),
            ("shared_calendar_elapsed_phase".to_string(), 0.70),
            ("cartoboost_lag".to_string(), 0.80),
        ]),
        &BTreeMap::from([
            (
                "shared_elapsed_phase_total_reconciled_050".to_string(),
                2.0e9,
            ),
            ("shared_calendar_elapsed_phase".to_string(), 4.0),
            ("cartoboost_lag".to_string(), 2.0),
        ]),
        10.0,
        Some(2),
    )
    .expect("stable choice");

    assert_eq!(selected, "shared_calendar_elapsed_phase");
}

#[test]
fn rmse_wape_objective_blends_normalized_rmse_and_wape() {
    let objective = ForecastObjective::parse("rmse_wape").expect("objective");
    let metrics = ForecastMetricSet {
        mae: 2.0,
        rmse: 4.0,
        normalized_rmse: 0.25,
        wape: 0.15,
        smape: 0.1,
        bias: 0.0,
        mase: None,
    };

    assert_eq!(objective.as_str(), "rmse_wape");
    assert!((objective.metric_value(&metrics) - 0.2).abs() < 1e-12);
}

#[test]
fn rule_based_gating_uses_series_scores_for_panel_frames() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::for_series("naive", "PU1->DO2", "mae", 1.0),
        ExpertScore::for_series("seasonal", "PU1->DO2", "mae", 3.0),
        ExpertScore::for_series("naive", "PU3->DO4", "mae", 3.0),
        ExpertScore::for_series("seasonal", "PU3->DO4", "mae", 1.0),
    ])
    .expect("score table");
    let gating = RuleBasedGating::new("mae", table).expect("gating");
    let frame = taxi_panel_frame();

    let weights = gating.weights_for_frame(&frame).expect("weights");

    assert!((weights["naive"] - 0.5).abs() < 1e-12);
    assert!((weights["seasonal"] - 0.5).abs() < 1e-12);
}

#[test]
fn rule_based_gating_can_use_hard_winner_when_validation_gap_is_decisive() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("cartoboost_lag", "scaled_error", 0.50),
        ExpertScore::global("classical_bank", "scaled_error", 0.80),
        ExpertScore::global("direct_tree", "scaled_error", 0.90),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "scaled_error",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(3),
            hard_winner_relative_gain: Some(0.05),
            weight_bounds: Some((0.15, 0.85)),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(
        weights,
        std::collections::BTreeMap::from([("cartoboost_lag".to_string(), 1.0)])
    );
}

#[test]
fn rule_based_gating_protects_baseline_until_displacement_gain_is_met() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("cartoboost_lag", "rps", 0.257),
        ExpertScore::global("point_auto", "rps", 0.252),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rps",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(2),
            weight_bounds: Some((0.15, 0.85)),
            baseline: Some("cartoboost_lag".to_string()),
            baseline_displacement_gain: Some(0.03),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(
        weights,
        std::collections::BTreeMap::from([("cartoboost_lag".to_string(), 1.0)])
    );
}

#[test]
fn rule_based_gating_protects_baseline_for_series_scores() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("cartoboost_lag", "rmse_wape", 1.00),
        ExpertScore::global("direct_tree", "rmse_wape", 1.20),
        ExpertScore::for_series("cartoboost_lag", "PU1->DO2", "rmse_wape", 1.00),
        ExpertScore::for_series("direct_tree", "PU1->DO2", "rmse_wape", 0.98),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rmse_wape",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(2),
            weight_bounds: Some((0.15, 0.85)),
            baseline: Some("cartoboost_lag".to_string()),
            baseline_displacement_gain: Some(0.03),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating
        .weights_for(Some("PU1->DO2"), None)
        .expect("series weights");

    assert_eq!(
        weights,
        std::collections::BTreeMap::from([("cartoboost_lag".to_string(), 1.0)])
    );
}

#[test]
fn rule_based_gating_protects_baseline_for_horizon_scores() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("cartoboost_lag", "rmse_wape", 1.00),
        ExpertScore::global("seasonal_delta_lag", "rmse_wape", 1.20),
        ExpertScore::for_horizon("cartoboost_lag", 2, "rmse_wape", 1.00),
        ExpertScore::for_horizon("seasonal_delta_lag", 2, "rmse_wape", 0.98),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rmse_wape",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(2),
            weight_bounds: Some((0.15, 0.85)),
            baseline: Some("cartoboost_lag".to_string()),
            baseline_displacement_gain: Some(0.03),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, Some(2)).expect("horizon weights");

    assert_eq!(
        weights,
        std::collections::BTreeMap::from([("cartoboost_lag".to_string(), 1.0)])
    );
}

#[test]
fn rule_based_gating_bounds_close_race_blends() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("lag_plus", "rmse", 1.00),
        ExpertScore::global("direct_tree", "rmse", 1.01),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rmse",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(2),
            hard_winner_relative_gain: Some(0.05),
            weight_bounds: Some((0.15, 0.85)),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(weights.len(), 2);
    assert!(weights
        .values()
        .all(|weight| (0.15..=0.85).contains(weight)));
    assert!((weights.values().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn rule_based_gating_projection_keeps_all_weights_within_bounds() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("best", "rmse", 0.01),
        ExpertScore::global("second", "rmse", 1.0),
        ExpertScore::global("third", "rmse", 1.0),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rmse",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(3),
            weight_bounds: Some((0.2, 0.8)),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert!((weights.values().sum::<f64>() - 1.0).abs() < 1.0e-12);
    assert!(weights.values().all(|weight| (0.2..=0.8).contains(weight)));
    assert!((weights["best"] - 0.6).abs() < 1.0e-12);
    assert!((weights["second"] - 0.2).abs() < 1.0e-12);
    assert!((weights["third"] - 0.2).abs() < 1.0e-12);
}

#[test]
fn candidate_selection_policy_prefers_simpler_close_candidate() {
    let policy = CandidateSelectionPolicy::new("synthetic", None).expect("policy");
    let selected = policy
        .select(&BTreeMap::from([
            ("cartoboost_auto_forecast".to_string(), 1.00),
            ("shared_seasonal_base".to_string(), 1.03),
            ("shared_calendar_dom".to_string(), 0.99),
        ]))
        .expect("selection");

    assert_eq!(selected.candidate, "shared_seasonal_base");
    assert!(
        candidate_complexity_rank("cartoboost_lag")
            < candidate_complexity_rank("cartoboost_auto_forecast")
    );
}

#[test]
fn candidate_selection_policy_uses_lowest_finite_for_classical_competitions() {
    for profile in ["classical_competition_full", "classical_competition"] {
        let policy = CandidateSelectionPolicy::new(profile, None).expect("policy");
        let selected = policy
            .select(&BTreeMap::from([
                ("cartoboost_autostats_bank".to_string(), 1.244),
                ("shared_seasonal_base".to_string(), 1.190),
                ("cartoboost_lag".to_string(), 1.751),
            ]))
            .expect("selection");

        assert_eq!(selected.candidate, "shared_seasonal_base");
    }
}

#[test]
fn classical_full_profile_does_not_apply_lag_origin_consistency_guard() {
    let guard = lag_origin_consistency_guard(
        "shared_seasonal_base",
        "classical_competition_full",
        &[1.0, 1.0],
        &[0.5, 1.1],
    )
    .expect("guard");

    assert!(guard.is_none());
}

#[test]
fn validation_unavailable_fallback_keeps_lag_for_classical_auto() {
    let available = vec![
        "cartoboost_lag".to_string(),
        "cartoboost_auto_forecast".to_string(),
    ];

    let selected = validation_unavailable_candidate_choice(
        "cartoboost_auto_forecast",
        "classical_competition_full",
        &available,
    )
    .expect("selection");
    assert_eq!(selected, "cartoboost_lag");

    let robust =
        validation_unavailable_candidate_choice("cartoboost_auto_forecast", "robust", &available)
            .expect("selection");
    assert_eq!(robust, "cartoboost_auto_forecast");
}

#[test]
fn candidate_selection_policy_applies_hierarchical_reconciled_and_lag_guards() {
    let policy =
        CandidateSelectionPolicy::new("hierarchical_reconciliation", None).expect("policy");
    let selected = policy
        .select(&BTreeMap::from([
            ("shared_calendar_elapsed_phase".to_string(), 0.7025),
            (
                "shared_elapsed_phase_total_reconciled_020".to_string(),
                0.7194,
            ),
            (
                "shared_elapsed_phase_total_reconciled_035".to_string(),
                0.7142,
            ),
            (
                "shared_elapsed_phase_total_reconciled_050".to_string(),
                0.7096,
            ),
            ("shared_reconciled_autostats_blend".to_string(), 0.7091),
            (
                "shared_point_autostats_elapsed_phase_blend".to_string(),
                0.7187,
            ),
        ]))
        .expect("selection");
    assert_eq!(
        selected.candidate,
        "shared_elapsed_phase_total_reconciled_050"
    );

    let low_support_policy =
        CandidateSelectionPolicy::new("hierarchical_reconciliation", Some(1)).expect("policy");
    let guarded = low_support_policy
        .select(&BTreeMap::from([
            ("cartoboost_lag".to_string(), 3.704_550_573_895_694_3),
            (
                "shared_point_autostats_elapsed_phase_blend".to_string(),
                3.640_384_191_677_357,
            ),
        ]))
        .expect("selection");
    assert_eq!(guarded.candidate, "cartoboost_lag");
}

#[test]
fn candidate_selection_policy_uses_lowest_finite_for_rank_portfolio_profile() {
    let policy = CandidateSelectionPolicy::new("rank_portfolio", None).expect("policy");
    let selected = policy
        .select(&BTreeMap::from([
            (
                "cartoboost_auto_forecast".to_string(),
                0.199_663_314_769_405_5,
            ),
            (
                "shared_calendar_elapsed_phase".to_string(),
                0.199_663_314_308_199_97,
            ),
            ("cartoboost_lag".to_string(), 0.199_663_314_439_439_74),
            (
                "shared_market_neutral_zero".to_string(),
                0.199_663_314_262_017_22,
            ),
        ]))
        .expect("selection");

    assert_eq!(selected.candidate, "shared_market_neutral_zero");
}

#[test]
fn forecast_ensemble_alias_keeps_weighted_deterministic_predictions() {
    let mut ensemble = ForecastEnsemble::new(vec![
        ("naive".to_string(), Box::new(NaiveForecaster::new()), 1.0),
        (
            "seasonal".to_string(),
            Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
            3.0,
        ),
    ])
    .expect("ensemble");
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 20.0),
            ForecastRow::single(ts(3), 30.0),
            ForecastRow::single(ts(4), 40.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("frame");

    ensemble.fit(&frame).expect("fit");
    let predictions = ensemble.predict(2).expect("predict");
    let means = predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect::<Vec<_>>();

    assert_eq!(means, vec![32.5, 40.0]);
}

#[test]
fn auto_forecast_model_fits_guarded_hybrid_and_emits_weights() {
    let frame = ForecastFrame::new(
        (1..=30)
            .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        validation_window: Some(2),
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("forecast");
    let metadata = model.metadata();

    assert_eq!(forecast.predictions().len(), 2);
    assert_eq!(metadata["model"], "cartoboost_auto_forecast");
    assert_eq!(metadata["objective"], "rmse");
    assert_eq!(metadata["nonnegative_output"], true);
    assert!(!metadata["weights"].as_object().expect("weights").is_empty());
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean >= 0.0));
    let global_score_count = metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .filter(|score| {
            score["metric"] == "rmse" && score["series_id"].is_null() && score["horizon"].is_null()
        })
        .count();
    assert!(global_score_count >= metadata["weights"].as_object().expect("weights").len());
    assert!(metadata["effective_lag_config"]["lags"]
        .as_array()
        .expect("effective lags")
        .iter()
        .any(|value| value.as_u64() == Some(7)));
    assert!(metadata["horizon_weights"]
        .as_object()
        .expect("horizon weights")
        .contains_key("1"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["metric"] == "rmse" && score["horizon"] == 1));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "cartoboost_lag"));
    assert!(!metadata["members"].as_object().expect("members").is_empty());
}

#[test]
fn auto_forecast_model_considers_recency_weighted_lag_for_level_shifts() {
    let frame = ForecastFrame::new(
        (1..=30)
            .map(|day| {
                let target = if day <= 20 {
                    20.0 + f64::from(day % 7)
                } else {
                    55.0 + f64::from(day % 7)
                };
                ForecastRow::single(ts(day), target)
            })
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2, 7],
            rolling_mean_windows: vec![2, 7],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![1, 7],
            rolling_trend_windows: vec![3],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        validation_window: Some(4),
        validation_origin_count: 1,
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "recency_weighted_lag"));
}

#[test]
fn auto_forecast_model_skips_log1p_candidate_for_negative_targets() {
    let frame = ForecastFrame::new(
        (1..=18)
            .map(|day| ForecastRow::single(ts(day), f64::from(day) - 10.0))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        validation_window: Some(2),
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert_eq!(metadata["nonnegative_output"], false);
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .all(|score| score["expert"] != "log1p_scaled_lag"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .all(|score| score["expert"] != "intermittent_demand"));
}

#[test]
fn auto_forecast_model_includes_intermittent_candidate_for_sparse_demand() {
    let values = [
        0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 12.0, 0.0, 0.0,
        13.0,
    ];
    let frame = ForecastFrame::new(
        values
            .iter()
            .enumerate()
            .map(|(index, value)| ForecastRow::single(ts(index as u32 + 1), *value))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        validation_window: Some(3),
        objective: ForecastObjective::Wape,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert_eq!(metadata["nonnegative_output"], true);
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "intermittent_demand"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .all(|score| score["expert"] != "cartoboost_direct"
            && score["expert"] != "cartoboost_rectified_recursive"));
}

#[test]
fn auto_forecast_model_uses_series_weights_for_panels() {
    let mut rows = Vec::new();
    for day in 1..=18 {
        rows.push(ForecastRow::new(
            "pickup_zone_1",
            ts(day),
            10.0 + f64::from(day),
        ));
        rows.push(ForecastRow::new(
            "pickup_zone_2",
            ts(day),
            60.0 - f64::from(day * 2),
        ));
    }
    let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        validation_window: Some(2),
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("forecast");
    let metadata = model.metadata();
    let series_weights = metadata["series_weights"]
        .as_object()
        .expect("series weights");

    assert_eq!(forecast.predictions().len(), 4);
    assert!(series_weights.contains_key("pickup_zone_1"));
    assert!(series_weights.contains_key("pickup_zone_2"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["series_id"] == "pickup_zone_1"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["series_id"] == "pickup_zone_2"));
}

#[test]
fn auto_forecast_model_skips_series_weights_when_validation_support_is_thin() {
    let mut rows = Vec::new();
    for day in 1..=18 {
        rows.push(ForecastRow::new(
            "pickup_zone_1",
            ts(day),
            10.0 + f64::from(day),
        ));
        rows.push(ForecastRow::new(
            "pickup_zone_2",
            ts(day),
            60.0 - f64::from(day * 2),
        ));
    }
    let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        validation_window: Some(2),
        validation_origin_count: 1,
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert_eq!(metadata["series_weight_min_validation_points"], 4);
    assert!(metadata["series_weights"]
        .as_object()
        .expect("series weights")
        .is_empty());
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["series_id"] == "pickup_zone_1"));
}

#[test]
fn auto_forecast_model_prunes_unsupported_lag_before_validation() {
    let frame = ForecastFrame::new(
        (1..=8)
            .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![28],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
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
        },
        validation_window: Some(2),
        season_length: 2,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    let weights = metadata["weights"].as_object().expect("weights");
    assert!(!weights.is_empty());
    let effective_lags = metadata["effective_lag_config"]["lags"]
        .as_array()
        .expect("effective lags");
    assert!(effective_lags.contains(&serde_json::json!(1)));
    assert!(!effective_lags.contains(&serde_json::json!(28)));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("validation scores")
        .iter()
        .any(|score| score["expert"] == "cartoboost_lag"));
    let forecast = model.predict(1).expect("forecast");
    assert_eq!(forecast.predictions().len(), 1);
    assert!(forecast.predictions()[0].mean.is_finite());
}

fn taxi_panel_frame() -> ForecastFrame {
    ForecastFrame::new(
        vec![
            ForecastRow::new("PU1->DO2", ts(1), 10.0),
            ForecastRow::new("PU1->DO2", ts(2), 12.0),
            ForecastRow::new("PU1->DO2", ts(3), 14.0),
            ForecastRow::new("PU3->DO4", ts(1), 30.0),
            ForecastRow::new("PU3->DO4", ts(2), 28.0),
            ForecastRow::new("PU3->DO4", ts(3), 26.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("frame")
}

fn ts(day: u32) -> chrono::NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, day)
        .expect("date")
        .and_hms_opt(0, 0, 0)
        .expect("time")
}
