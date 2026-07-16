#[cfg(test)]
mod tests {
    use super::*;

    fn meta(row_count: usize) -> GeoFrameMeta {
        GeoFrameMeta::new(
            "sha256:test".to_string(),
            "EPSG:2263".to_string(),
            "0.2.32".to_string(),
            BTreeMap::from([("cartoboost".to_string(), "0.2.32".to_string())]),
            Some(42),
            row_count,
            None,
        )
        .unwrap()
    }

    #[test]
    fn containers_round_trip() {
        let coords = CoordinateMatrix::new(
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            Some("EPSG:4326".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(
            CoordinateMatrix::from_json_str(&coords.to_json_string().unwrap()).unwrap(),
            coords
        );
        let time = TimeIndex::new(
            vec!["2024-01-01".to_string(), "2024-01-02".to_string()],
            Some("D".to_string()),
            None,
        )
        .unwrap();
        let panel =
            PanelIndex::new(vec!["zone_a".to_string(), "zone_a".to_string()], Some(time)).unwrap();
        assert_eq!(
            PanelIndex::from_json_str(&panel.to_json_string().unwrap()).unwrap(),
            panel
        );
    }

    #[test]
    fn spatial_weights_csr_behaviors() {
        let weights = SpatialWeights::from_edges(3, vec![(0, 1, 2.0), (1, 0, 2.0)], false).unwrap();
        assert!(weights.is_symmetric(0.0));
        assert_eq!(weights.isolated_nodes(), vec![2]);
        let normalized = weights.row_normalize();
        assert_eq!(normalized.data, vec![1.0, 1.0]);
        assert_eq!(
            SpatialWeights::from_json_str(&normalized.to_json_string().unwrap()).unwrap(),
            normalized
        );
    }

    #[test]
    fn shared_distance_helpers_are_stable() {
        assert_eq!(squared_euclidean_distance([0.0, 0.0], [3.0, 4.0]), 25.0);
        assert_eq!(euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0);
        assert_eq!(
            clockwise_bearing_unit_vector([0.0, 0.0], [0.0, 5.0]),
            Some([0.0, 1.0])
        );
        assert_eq!(
            clockwise_bearing_unit_vector([0.0, 0.0], [5.0, 0.0]),
            Some([1.0, 0.0])
        );
        assert_eq!(clockwise_bearing_unit_vector([1.0, 1.0], [1.0, 1.0]), None);
        assert_eq!(
            route_feature_vector([0.0, 0.0], [3.0, 4.0]),
            Some([1.5, 2.0, 5.0, 0.6, 0.8])
        );
        assert_eq!(
            radial_anchor_distances([3.0, 4.0], &[[0.0, 0.0], [3.0, 0.0]]),
            vec![5.0, 4.0]
        );
        assert_eq!(
            rbf_anchor_features([0.0, 0.0], &[[0.0, 0.0], [1.0, 0.0]], 1.0).unwrap(),
            vec![1.0, (-0.5_f64).exp()]
        );
        assert!(rbf_anchor_features([0.0, 0.0], &[[0.0, 0.0]], 0.0).is_err());
        assert_eq!(
            local_frame_features([2.0, 3.0], [1.0, 1.0], [0.0, 1.0]),
            Some([2.0, -1.0])
        );
        assert_eq!(
            transform_anisotropic_point([3.0, 4.0], 0.0, 2.0),
            [3.0, 2.0]
        );
        assert!(
            (anisotropic_euclidean_distance([0.0, 0.0], [3.0, 4.0], 0.0, 2.0) - 13.0_f64.sqrt())
                .abs()
                < 1.0e-12
        );
        let nyc_to_london_km = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((nyc_to_london_km - 5_570.0).abs() < 20.0);
        let north = initial_bearing_unit_vector_latlng(40.0, -73.0, 41.0, -73.0).unwrap();
        assert!(north[0].abs() < 1.0e-12);
        assert!((north[1] - 1.0).abs() < 1.0e-12);
        let northwest = initial_bearing_unit_vector_latlng(40.0, -73.0, 41.0, -74.0).unwrap();
        assert!(northwest[0] < 0.0);
        assert!(northwest[1] > 0.0);
        assert!((northwest[0] * northwest[0] + northwest[1] * northwest[1] - 1.0).abs() < 1.0e-12);
        assert!(
            (haversine_distance_meters(40.7128, -74.0060, 51.5074, -0.1278)
                - nyc_to_london_km * 1_000.0)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn split_manifest_is_deterministic() {
        let coords =
            CoordinateMatrix::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.0; 4], None, None).unwrap();
        let a = spatial_block_cv(&coords, 2, meta(4), "spatial_block".to_string()).unwrap();
        let b = spatial_block_cv(&coords, 2, meta(4), "spatial_block".to_string()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.hash().unwrap(), b.hash().unwrap());
        assert!(a.hash().unwrap().starts_with("sha256:"));
    }
}
