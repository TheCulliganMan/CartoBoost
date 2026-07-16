fn validate_numeric_series(values: &[f64], label: &str) -> Result<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "{label} at index {idx} must be finite"
            )));
        }
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be in (0, 1]"
        )));
    }
    Ok(())
}

fn croston_level(values: &[f64], alpha: f64) -> Result<f64> {
    let first_nonzero = values
        .iter()
        .position(|value| *value > 0.0)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "intermittent demand forecast requires at least one non-zero observation"
                    .to_string(),
            )
        })?;
    let mut demand = values[first_nonzero];
    let mut interval = (first_nonzero + 1) as f64;
    let mut elapsed = 0usize;
    for value in values.iter().skip(first_nonzero + 1) {
        elapsed += 1;
        if *value > 0.0 {
            demand += alpha * (*value - demand);
            interval += alpha * (elapsed as f64 - interval);
            elapsed = 0;
        }
    }
    if interval <= 0.0 || !interval.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand interval estimate is invalid".to_string(),
        ));
    }
    Ok(demand / interval)
}

fn tsb_level(values: &[f64], alpha: f64, beta: f64) -> Result<f64> {
    let first_nonzero = values.iter().find(|value| **value > 0.0).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "TSB forecast requires at least one non-zero observation".to_string(),
        )
    })?;
    let mut demand = *first_nonzero;
    let mut probability = 0.0;
    for value in values {
        let occurrence = if *value > 0.0 { 1.0 } else { 0.0 };
        probability += beta * (occurrence - probability);
        if *value > 0.0 {
            demand += alpha * (*value - demand);
        }
    }
    Ok(probability * demand)
}

