impl MarketPanelFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lane_ids: Vec<String>,
        timestamps: Vec<i64>,
        target_names: Vec<String>,
        primary: Vec<Vec<f64>>,
        secondary: Vec<Vec<f64>>,
        origin_ids: Vec<String>,
        destination_ids: Vec<String>,
        hierarchy_groups: Vec<Vec<String>>,
        coordinates: Vec<[f64; 4]>,
        calendar: Vec<Vec<f64>>,
        mix: Option<Vec<Vec<Vec<f64>>>>,
        expert_priors: Vec<ExpertRelationshipPrior>,
        expert_labels: Vec<ExpertEventLabel>,
        horizon: usize,
        frequency: String,
    ) -> Result<Self> {
        let frame = Self {
            lane_ids,
            timestamps,
            target_names,
            primary,
            secondary,
            origin_ids,
            destination_ids,
            hierarchy_groups,
            coordinates,
            calendar,
            mix,
            expert_priors,
            expert_labels,
            horizon,
            frequency,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<()> {
        let lanes = self.lane_ids.len();
        let times = self.timestamps.len();
        if lanes == 0 || times <= self.horizon || self.horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "market frame requires lanes and more rows than a positive horizon".to_string(),
            ));
        }
        if self.target_names.len() != 2
            || self.target_names.iter().any(String::is_empty)
            || self.target_names[0] == self.target_names[1]
        {
            return Err(GeoStError::InvalidFrame(
                "market frame requires two distinct nonempty target names".to_string(),
            ));
        }
        if self.primary.len() != times
            || self.secondary.len() != times
            || self.calendar.len() != times
            || self.origin_ids.len() != lanes
            || self.destination_ids.len() != lanes
            || self.hierarchy_groups.len() != lanes
            || self.coordinates.len() != lanes
        {
            return Err(GeoStError::InvalidFrame(
                "market frame dimensions do not agree".to_string(),
            ));
        }
        if self.timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(GeoStError::InvalidFrame(
                "market timestamps must be strictly increasing".to_string(),
            ));
        }
        let calendar_width = self.calendar.first().map_or(0, Vec::len);
        for row in &self.calendar {
            if row.len() != calendar_width || row.iter().any(|x| !x.is_finite()) {
                return Err(GeoStError::InvalidFrame(
                    "calendar features must be finite and rectangular".to_string(),
                ));
            }
        }
        for (primary_row, secondary_row) in self.primary.iter().zip(&self.secondary) {
            if primary_row.len() != lanes
                || secondary_row.len() != lanes
                || primary_row
                    .iter()
                    .any(|x| x.is_infinite() || (!x.is_nan() && *x <= 0.0))
                || secondary_row.iter().any(|x| !x.is_finite() || *x < 0.0)
            {
                return Err(GeoStError::InvalidFrame("primary observations must be positive or NaN and secondary target nonnegative with shape [time, lane]".to_string()));
            }
        }
        for lane in 0..lanes {
            let observed = self
                .primary
                .iter()
                .filter(|row| !row[lane].is_nan())
                .count();
            if observed == 0 && self.hierarchy_groups[lane].is_empty() {
                return Err(GeoStError::InvalidFrame(format!(
                    "lane '{}' has no observed primary values and no hierarchy group for partial pooling",
                    self.lane_ids[lane]
                )));
            }
        }
        if self
            .coordinates
            .iter()
            .any(|point| point.iter().any(|x| !x.is_finite()))
        {
            return Err(GeoStError::InvalidFrame(
                "lane coordinates must be finite".to_string(),
            ));
        }
        if let Some(mix) = &self.mix {
            if mix.len() != times || mix.iter().any(|row| row.len() != lanes) {
                return Err(GeoStError::InvalidFrame(
                    "mix must have shape [time, lane, feature]".to_string(),
                ));
            }
            let width = mix.first().and_then(|row| row.first()).map_or(0, Vec::len);
            if width == 0
                || mix
                    .iter()
                    .flatten()
                    .any(|row| row.len() != width || row.iter().any(|x| !x.is_finite()))
            {
                return Err(GeoStError::InvalidFrame(
                    "mix features must be nonempty, finite, and rectangular".to_string(),
                ));
            }
        }
        let known: BTreeSet<_> = self.lane_ids.iter().collect();
        if self.expert_priors.iter().any(|prior| {
            !known.contains(&prior.source_lane_id)
                || !known.contains(&prior.target_lane_id)
                || !prior.weight.is_finite()
                || prior.weight < 0.0
                || prior.version.is_empty()
        }) {
            return Err(GeoStError::InvalidFrame("expert priors must reference known lanes with a nonnegative finite weight and version".to_string()));
        }
        if self.expert_labels.iter().any(|label| {
            !known.contains(&label.lane_id)
                || !self.timestamps.contains(&label.timestamp)
                || label.version.is_empty()
        }) {
            return Err(GeoStError::InvalidFrame(
                "expert labels must reference a known lane, observed timestamp, and version"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

