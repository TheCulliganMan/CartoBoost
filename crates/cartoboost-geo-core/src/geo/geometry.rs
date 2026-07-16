pub fn squared_euclidean_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    dx * dx + dy * dy
}

pub fn euclidean_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    squared_euclidean_distance(left, right).sqrt()
}

pub fn clockwise_bearing_unit_vector(origin: [f64; 2], destination: [f64; 2]) -> Option<[f64; 2]> {
    let dx = destination[0] - origin[0];
    let dy = destination[1] - origin[1];
    let distance = (dx * dx + dy * dy).sqrt();
    if distance == 0.0 || !distance.is_finite() {
        return None;
    }
    Some([dx / distance, dy / distance])
}

pub fn route_feature_vector(origin: [f64; 2], destination: [f64; 2]) -> Option<[f64; 5]> {
    let bearing = clockwise_bearing_unit_vector(origin, destination)?;
    Some([
        0.5 * (origin[0] + destination[0]),
        0.5 * (origin[1] + destination[1]),
        euclidean_distance(origin, destination),
        bearing[0],
        bearing[1],
    ])
}

pub fn radial_anchor_distances(point: [f64; 2], anchors: &[[f64; 2]]) -> Vec<f64> {
    anchors
        .iter()
        .map(|anchor| euclidean_distance(point, *anchor))
        .collect()
}

pub fn rbf_anchor_features(
    point: [f64; 2],
    anchors: &[[f64; 2]],
    length_scale: f64,
) -> Result<Vec<f64>> {
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(GeoCoreError::InvalidInput(
            "length_scale must be finite and positive".to_string(),
        ));
    }
    Ok(anchors
        .iter()
        .map(|anchor| {
            let distance_sq = squared_euclidean_distance(point, *anchor);
            (-0.5 * distance_sq / (length_scale * length_scale)).exp()
        })
        .collect())
}

pub fn local_frame_features(point: [f64; 2], origin: [f64; 2], axis: [f64; 2]) -> Option<[f64; 2]> {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    let east = axis[0] / norm;
    let north = axis[1] / norm;
    let dx = point[0] - origin[0];
    let dy = point[1] - origin[1];
    let along = dx * east + dy * north;
    let cross = -dx * north + dy * east;
    Some([along, cross])
}

pub fn initial_bearing_unit_vector_latlng(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> Option<[f64; 2]> {
    let lat1 = origin_latitude.to_radians();
    let lat2 = destination_latitude.to_radians();
    let dlon = (destination_longitude - origin_longitude).to_radians();
    let east = dlon.sin() * lat2.cos();
    let north = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    let norm = (east * east + north * north).sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some([east / norm, north / norm])
}

pub fn anisotropic_euclidean_distance(
    left: [f64; 2],
    right: [f64; 2],
    angle_degrees: f64,
    scaling: f64,
) -> f64 {
    let left = transform_anisotropic_point(left, angle_degrees, scaling);
    let right = transform_anisotropic_point(right, angle_degrees, scaling);
    euclidean_distance(left, right)
}

pub fn transform_anisotropic_point(point: [f64; 2], angle_degrees: f64, scaling: f64) -> [f64; 2] {
    let angle = angle_degrees.to_radians();
    let cos = angle.cos();
    let sin = angle.sin();
    let x = point[0] * cos + point[1] * sin;
    let y = (-point[0] * sin + point[1] * cos) / scaling;
    [x, y]
}

pub fn haversine_distance_km(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> f64 {
    let lat1 = origin_latitude.to_radians();
    let lon1 = origin_longitude.to_radians();
    let lat2 = destination_latitude.to_radians();
    let lon2 = destination_longitude.to_radians();
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * a.sqrt().asin()
}

pub fn haversine_distance_meters(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> f64 {
    haversine_distance_km(
        origin_latitude,
        origin_longitude,
        destination_latitude,
        destination_longitude,
    ) * 1_000.0
}

fn stable_hex64(bytes: &[u8]) -> String {
    let mut text = String::new();
    for salt in 0_u64..4 {
        let mut hash = 0xcbf29ce484222325_u64 ^ salt;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        text.push_str(&format!("{hash:016x}"));
    }
    text
}

