use cartoboost_core::forecasting::{
    proportional_total_reconciliation, HierarchySpec, Reconciler, ReconciliationMethod,
    TemporalAggregation, TemporalHierarchy,
};
use cartoboost_core::metrics::{
    aggregate_equal_level_wrmsse, ordered_nonnegative_weights, wrmsse, WrmsseSeries,
};

fn taxi_hierarchy() -> HierarchySpec {
    HierarchySpec::from_edges(vec![
        ("all_taxi", None),
        ("PU1", Some("all_taxi")),
        ("PU2", Some("all_taxi")),
        ("PU1_DO1", Some("PU1")),
        ("PU1_DO2", Some("PU1")),
        ("PU2_DO1", Some("PU2")),
    ])
    .expect("valid hierarchy")
}

#[test]
fn sparse_hierarchy_aggregates_bottom_taxi_series() {
    let hierarchy = taxi_hierarchy();
    let aggregated = hierarchy
        .aggregate_bottom_values(&[10.0, 15.0, 20.0])
        .expect("aggregate");

    assert_eq!(
        hierarchy.bottom_node_ids(),
        vec!["PU1_DO1", "PU1_DO2", "PU2_DO1"]
    );
    assert_eq!(aggregated, vec![45.0, 25.0, 20.0, 10.0, 15.0, 20.0]);
    assert!(hierarchy
        .is_coherent(&aggregated, 1e-12)
        .expect("coherence check"));
}

#[test]
fn bottom_up_and_top_down_reconciliation_return_coherent_forecasts() {
    let hierarchy = taxi_hierarchy();
    let base = vec![
        vec![100.0, 120.0],
        vec![70.0, 80.0],
        vec![40.0, 45.0],
        vec![20.0, 25.0],
        vec![30.0, 35.0],
        vec![50.0, 55.0],
    ];

    let bottom_up = Reconciler::new(hierarchy.clone(), ReconciliationMethod::BottomUp)
        .reconcile(&base)
        .expect("bottom-up");
    assert_eq!(bottom_up[0], vec![100.0, 115.0]);
    assert_eq!(bottom_up[1], vec![50.0, 60.0]);
    assert_eq!(bottom_up[2], vec![50.0, 55.0]);

    let top_down = Reconciler::new(hierarchy.clone(), ReconciliationMethod::TopDown)
        .reconcile(&base)
        .expect("top-down");
    assert_eq!(top_down[0], vec![100.0, 120.0]);
    assert!(hierarchy
        .is_coherent(
            &top_down.iter().map(|row| row[0]).collect::<Vec<_>>(),
            1e-10
        )
        .expect("coherence"));
    assert_eq!(top_down[3][0], 20.0);
    assert_eq!(top_down[4][0], 30.0);
    assert_eq!(top_down[5][0], 50.0);
}

#[test]
fn middle_out_uses_requested_level_and_ols_projects_to_coherent_space() {
    let hierarchy = taxi_hierarchy();
    let base = vec![
        vec![90.0],
        vec![60.0],
        vec![50.0],
        vec![20.0],
        vec![40.0],
        vec![55.0],
    ];

    let middle = Reconciler::new(
        hierarchy.clone(),
        ReconciliationMethod::MiddleOut { level: 1 },
    )
    .reconcile(&base)
    .expect("middle-out");
    assert_eq!(middle[0][0], 110.0);
    assert_eq!(middle[1][0], 60.0);
    assert_eq!(middle[2][0], 50.0);
    assert_eq!(middle[3][0], 20.0);
    assert_eq!(middle[4][0], 40.0);
    assert_eq!(middle[5][0], 50.0);

    let ols = Reconciler::new(hierarchy.clone(), ReconciliationMethod::Ols)
        .reconcile(&base)
        .expect("ols");
    let first_horizon = ols.iter().map(|row| row[0]).collect::<Vec<_>>();
    assert!(hierarchy
        .is_coherent(&first_horizon, 1e-10)
        .expect("coherence"));
    assert!(ols[0][0] > 90.0);
    assert!(ols[0][0] < 110.0);
}

#[test]
fn wls_and_mint_shrink_accept_small_reference_inputs() {
    let hierarchy = taxi_hierarchy();
    let base = vec![
        vec![90.0],
        vec![60.0],
        vec![50.0],
        vec![20.0],
        vec![40.0],
        vec![55.0],
    ];

    let wls = Reconciler::new(
        hierarchy.clone(),
        ReconciliationMethod::Wls {
            variances: vec![4.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        },
    )
    .reconcile(&base)
    .expect("wls");
    assert!(hierarchy
        .is_coherent(&wls.iter().map(|row| row[0]).collect::<Vec<_>>(), 1e-10)
        .expect("coherence"));

    let residuals = vec![
        vec![2.0, -2.0, 1.0],
        vec![1.0, -1.0, 0.5],
        vec![1.5, -1.5, 0.25],
        vec![0.5, -0.5, 0.25],
        vec![0.7, -0.7, 0.25],
        vec![0.9, -0.9, 0.25],
    ];
    let mint = Reconciler::new(
        hierarchy.clone(),
        ReconciliationMethod::MinTShrink {
            residuals,
            shrinkage: 0.25,
        },
    )
    .reconcile(&base)
    .expect("mint shrink");
    assert!(hierarchy
        .is_coherent(&mint.iter().map(|row| row[0]).collect::<Vec<_>>(), 1e-10)
        .expect("coherence"));
}

#[test]
fn temporal_hierarchy_aggregates_non_overlapping_windows() {
    let hierarchy = TemporalHierarchy::new(
        "hour",
        vec![
            TemporalAggregation::new("six_hour", 6).expect("six hour"),
            TemporalAggregation::new("day", 24).expect("day"),
        ],
    )
    .expect("temporal hierarchy");

    let values = (1..=24).map(f64::from).collect::<Vec<_>>();
    let levels = hierarchy.aggregate_series(&values).expect("aggregate");

    assert_eq!(levels[0].values.len(), 24);
    assert_eq!(levels[1].values, vec![21.0, 57.0, 93.0, 129.0]);
    assert_eq!(levels[2].values, vec![300.0]);
}

#[test]
fn wrmsse_matches_manual_m5_style_reference() {
    let rows = vec![
        WrmsseSeries::new(
            "PULocationID=1",
            vec![10.0, 12.0, 14.0, 16.0],
            vec![18.0, 20.0],
            vec![17.0, 23.0],
            2.0,
        ),
        WrmsseSeries::new(
            "PULocationID=2",
            vec![20.0, 21.0, 23.0, 26.0],
            vec![30.0, 34.0],
            vec![32.0, 31.0],
            1.0,
        ),
    ];

    let score = wrmsse(&rows, 1).expect("wrmsse");

    assert!((score.series[0].scale - 4.0).abs() < 1e-12);
    assert!((score.series[1].scale - (14.0 / 3.0)).abs() < 1e-12);
    assert!((score.series[0].rmsse - (10.0_f64 / 8.0).sqrt()).abs() < 1e-12);
    assert!((score.series[1].rmsse - (13.0_f64 / (2.0 * 14.0 / 3.0)).sqrt()).abs() < 1e-12);
    assert!((score.score - 1.138_753_888_734_651_6).abs() < 1e-12);
}

#[test]
fn aggregate_equal_level_wrmsse_aggregates_hierarchy_levels() {
    let score = aggregate_equal_level_wrmsse(&[
        ("total".to_string(), 0.6),
        ("state".to_string(), 0.9),
        ("item_store".to_string(), 1.2),
    ])
    .expect("m5 aggregate wrmsse");

    assert!((score.score - 0.9).abs() < 1e-12);
    assert_eq!(score.levels.len(), 3);
    assert_eq!(score.levels[0].level, "total");
    assert!((score.levels[0].level_weight - (1.0 / 3.0)).abs() < 1e-12);
    assert!((score.levels[2].contribution - 0.4).abs() < 1e-12);
}

#[test]
fn ordered_nonnegative_weights_clip_missing_and_fallback_to_unit() {
    let weighted = ordered_nonnegative_weights(
        &["a".to_string(), "b".to_string(), "c".to_string()],
        &[("a".to_string(), -2.0), ("b".to_string(), 4.0)],
    )
    .expect("weights");

    assert_eq!(
        weighted,
        vec![
            ("a".to_string(), 0.0),
            ("b".to_string(), 4.0),
            ("c".to_string(), 0.0),
        ]
    );

    let fallback = ordered_nonnegative_weights(
        &["a".to_string(), "b".to_string()],
        &[("a".to_string(), -1.0)],
    )
    .expect("fallback");
    assert_eq!(
        fallback,
        vec![("a".to_string(), 1.0), ("b".to_string(), 1.0)]
    );
}

#[test]
fn proportional_total_reconciliation_blends_toward_target_total() {
    let reconciled =
        proportional_total_reconciliation(&[2.0, 3.0, 5.0], 20.0, 0.5).expect("reconcile");

    assert_eq!(reconciled, vec![3.0, 4.5, 7.5]);

    let error = proportional_total_reconciliation(&[0.0, 0.0], 10.0, 0.5)
        .expect_err("zero allocation basis must fail");
    assert!(error.to_string().contains("sums to zero"));

    let stable =
        proportional_total_reconciliation(&[-99.0, 100.0], 100.0, 0.5).expect("signed base");
    assert_eq!(stable, vec![-49.5, 100.0]);
}

#[test]
fn top_down_reconciliation_rejects_undefined_allocation_proportions() {
    let hierarchy = taxi_hierarchy();
    let mut base = vec![vec![100.0]; hierarchy.node_count()];
    for (node_idx, values) in base.iter_mut().enumerate() {
        if hierarchy.bottom_position_for_node(node_idx).is_some() {
            values[0] = 0.0;
        }
    }

    let error = Reconciler::new(hierarchy, ReconciliationMethod::TopDown)
        .reconcile(&base)
        .expect_err("zero bottom allocation basis must fail");
    assert!(error.to_string().contains("positive finite sum"));
}
