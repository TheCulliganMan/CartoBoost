use super::{FeatureKind, FeatureSchema};

#[test]
fn unnamed_numeric_schema_is_deterministic() {
    let schema = FeatureSchema::unnamed_numeric(3);

    assert_eq!(
        schema.names,
        vec![
            "feature_0".to_string(),
            "feature_1".to_string(),
            "feature_2".to_string()
        ]
    );
    assert_eq!(schema.kinds, vec![FeatureKind::Numeric; 3]);
    assert_eq!(schema.len(), 3);
}

#[test]
fn schema_validation_rejects_mismatched_fields() {
    let schema = FeatureSchema {
        names: vec!["x".to_string()],
        kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
    };

    assert!(schema.validate().is_err());
}

#[test]
fn schema_validation_rejects_empty_and_duplicate_names() {
    let empty_name = FeatureSchema {
        names: vec!["".to_string()],
        kinds: vec![FeatureKind::Numeric],
    };
    assert!(empty_name.validate().is_err());

    let duplicate_name = FeatureSchema {
        names: vec!["x".to_string(), "x".to_string()],
        kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
    };
    assert!(duplicate_name.validate().is_err());
}

#[test]
fn schema_validation_rejects_zero_period() {
    let schema = FeatureSchema {
        names: vec!["hour".to_string()],
        kinds: vec![FeatureKind::Periodic { period: 0 }],
    };

    assert!(schema.validate().is_err());
}

#[test]
fn schema_validation_accepts_spatial_features() {
    let schema = FeatureSchema {
        names: vec!["pickup_x".to_string(), "pickup_y".to_string()],
        kinds: vec![FeatureKind::Spatial, FeatureKind::Spatial],
    };

    assert!(schema.validate().is_ok());
}

#[test]
fn schema_validation_accepts_categorical_and_ordinal_features() {
    let schema = FeatureSchema {
        names: vec!["borough".to_string(), "service_tier".to_string()],
        kinds: vec![FeatureKind::Categorical, FeatureKind::Ordinal],
    };

    assert!(schema.validate().is_ok());
}

#[test]
fn wide_schema_reports_first_duplicate_in_input_order() {
    let mut schema = FeatureSchema::unnamed_numeric(4_096);
    assert!(schema.validate().is_ok());
    schema.names[2_048] = schema.names[17].clone();
    schema.names[2_049] = schema.names[3].clone();
    assert!(schema
        .validate()
        .unwrap_err()
        .to_string()
        .contains("'feature_17'"));
}
