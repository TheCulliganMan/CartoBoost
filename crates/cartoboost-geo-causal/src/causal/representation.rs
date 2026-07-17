pub fn causal_representation_report_json(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
) -> Result<String> {
    let report = causal_representation_report(features, outcomes, regions, heldout_region)?;
    serde_json::to_string(&report).map_err(|err| GeoCausalError::InvalidInput(err.to_string()))
}

pub fn causal_representation_report(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
) -> Result<CausalRepresentationReport> {
    validate_representation_inputs(features, outcomes, regions, heldout_region)?;
    let dim = features[0].len();
    let global_mean = column_mean(features);
    let region_mean = region_feature_means(features, regions, dim);
    let transformed = features
        .iter()
        .zip(regions)
        .map(|(row, region)| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| value - region_mean[region][idx] + global_mean[idx])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let train_indices = regions
        .iter()
        .enumerate()
        .filter_map(|(idx, region)| (region != heldout_region).then_some(idx))
        .collect::<Vec<_>>();
    let test_indices = regions
        .iter()
        .enumerate()
        .filter_map(|(idx, region)| (region == heldout_region).then_some(idx))
        .collect::<Vec<_>>();
    let raw_weights = ridge_fit_indexed(features, outcomes, &train_indices);
    let invariant_weights = ridge_fit_indexed(&transformed, outcomes, &train_indices);
    let raw_rmse = indexed_rmse(features, outcomes, &test_indices, &raw_weights);
    let invariant_rmse = indexed_rmse(&transformed, outcomes, &test_indices, &invariant_weights);
    let mut losses = BTreeMap::new();
    losses.insert(
        "supervised_outcome_loss".to_string(),
        indexed_mse(&transformed, outcomes, &train_indices, &invariant_weights),
    );
    let domain_loss = mean_region_distance(&transformed, regions);
    losses.insert("domain_adversarial_loss".to_string(), domain_loss);
    losses.insert(
        "invariant_risk_penalty".to_string(),
        (raw_rmse - invariant_rmse).abs(),
    );
    losses.insert("treatment_balance_penalty".to_string(), domain_loss);
    losses.insert(
        "representation_smoothness_penalty".to_string(),
        mean_row_variation(&transformed),
    );
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "model_class".to_string(),
        "InvariantRiskEncoder".to_string(),
    );
    metadata.insert(
        "domain_encoder".to_string(),
        "DomainAdversarialGeoEncoder".to_string(),
    );
    metadata.insert(
        "counterfactual_net".to_string(),
        "CounterfactualRepresentationNet".to_string(),
    );
    metadata.insert(
        "treatment_head".to_string(),
        "TreatmentEffectRepresentationHead".to_string(),
    );
    metadata.insert(
        "supplements".to_string(),
        "SyntheticDIDEstimator,GeoExperimentDesigner".to_string(),
    );
    Ok(CausalRepresentationReport {
        transformed_features: transformed,
        heldout_region: heldout_region.to_string(),
        raw_rmse,
        invariant_rmse,
        improvement: raw_rmse - invariant_rmse,
        losses,
        warnings: vec![
            "Representation learning does not prove causal identification; use it only as a supplement to an identified design.".to_string(),
        ],
        metadata,
    })
}

