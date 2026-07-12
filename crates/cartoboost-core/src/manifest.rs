//! Rust-owned public capability manifest for the v0.3 stable boundary.

use serde_json::Value;

/// Canonical model metadata consumed by Python, WASM, and release tooling.
///
/// The manifest deliberately describes only generic capabilities. Preview
/// implementations can be added here without making them part of the stable
/// contract, but they must carry an explicit tier and evidence level.
pub const MODEL_MANIFEST_JSON: &str = r#"[
  {"key":"models.cartoboost_regressor","tier":"stable","backend":"rust_native","task":"regression","artifact_version":2,"dependencies":[],"evidence_level":"real_data"},
  {"key":"models.cartoboost_classifier","tier":"stable","backend":"rust_native","task":"classification","artifact_version":2,"dependencies":[],"evidence_level":"real_data"},
  {"key":"models.cartoboost_ranker","tier":"stable","backend":"rust_native","task":"ranking","artifact_version":2,"dependencies":[],"evidence_level":"real_data"},
  {"key":"forecasting.auto_forecaster","tier":"stable","backend":"rust_native","task":"forecasting","artifact_version":2,"dependencies":[],"evidence_level":"real_data"},
  {"key":"forecasting.cartoboost_lag","tier":"stable","backend":"rust_native","task":"forecasting","artifact_version":2,"dependencies":[],"evidence_level":"real_data"},
  {"key":"graph.dcrnn","tier":"experimental","backend":"python_orchestration","task":"forecasting","artifact_version":2,"dependencies":["torch"],"evidence_level":"experimental_only"},
  {"key":"geo.nngp","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"},
  {"key":"geo.residual_nngp","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"},
  {"key":"geo.spatial_lag","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":["libpysal","spreg"],"evidence_level":"synthetic"},
  {"key":"geo.spatial_error","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":["libpysal","spreg"],"evidence_level":"synthetic"},
  {"key":"geo.spatial_durbin","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":["libpysal","spreg"],"evidence_level":"synthetic"},
  {"key":"causal.synthetic_did","tier":"preview","backend":"python_orchestration","task":"causal_panel","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"},
  {"key":"causal.geo_lift_design","tier":"preview","backend":"python_orchestration","task":"causal_panel","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"},
  {"key":"causal.spatial_placebo","tier":"preview","backend":"python_orchestration","task":"causal_panel","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"},
  {"key":"prob.conformal_interval","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"},
  {"key":"prob.spatial_conformal","tier":"preview","backend":"python_orchestration","task":"regression","artifact_version":2,"dependencies":[],"evidence_level":"synthetic"}
]"#;

/// Return the validated canonical manifest as JSON.
pub fn model_manifest_json() -> &'static str {
    // Keep malformed edits from crossing the native boundary.
    debug_assert!(serde_json::from_str::<Value>(MODEL_MANIFEST_JSON).is_ok());
    MODEL_MANIFEST_JSON
}

#[cfg(test)]
mod tests {
    use super::model_manifest_json;
    use serde_json::Value;

    #[test]
    fn manifest_contains_exact_stable_surface() {
        let entries: Vec<Value> = serde_json::from_str(model_manifest_json()).unwrap();
        let stable: Vec<&str> = entries
            .iter()
            .filter(|entry| entry["tier"] == "stable")
            .map(|entry| entry["key"].as_str().unwrap())
            .collect();
        assert_eq!(stable.len(), 5);
        assert!(stable.contains(&"models.cartoboost_regressor"));
        assert!(stable.contains(&"forecasting.auto_forecaster"));
        assert_eq!(entries.len(), 16);
        assert!(entries.iter().any(|entry| entry["tier"] == "preview"));
        assert!(entries.iter().any(|entry| entry["tier"] == "experimental"));
    }
}
