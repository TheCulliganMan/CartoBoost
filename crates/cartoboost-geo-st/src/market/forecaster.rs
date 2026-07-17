#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketStructureForecaster {
    pub config: MarketStructureConfig,
    lane_ids: Vec<String>,
    #[serde(default)]
    origin_ids: Vec<String>,
    #[serde(default)]
    destination_ids: Vec<String>,
    #[serde(default)]
    coordinates: Vec<[f64; 4]>,
    hierarchy_groups: Vec<Vec<String>>,
    timestamps: Vec<i64>,
    frequency: String,
    relationships: Vec<Vec<MarketRelationship>>,
    target_names: Vec<String>,
    primary_means: Vec<f64>,
    secondary_means: Vec<f64>,
    primary_scales: Vec<f64>,
    /// Per-lane empirical 80% residual radii fitted only on observed history.
    primary_interval_radii: Vec<f64>,
    interval_calibration_multiplier: f64,
    secondary_scales: Vec<f64>,
    weekly_primary: Vec<Vec<f64>>,
    weekly_secondary: Vec<Vec<f64>>,
    primary_calendar_weights: Vec<f64>,
    secondary_calendar_weights: Vec<f64>,
    primary_history: Vec<Vec<f64>>,
    primary_observed: Vec<Vec<bool>>,
    secondary_history: Vec<Vec<f64>>,
    calendar_width: usize,
    last_calendar: Vec<f64>,
    mix_coefficients: Vec<f64>,
    cross_target_couplings: Vec<f64>,
    last_mix: Option<Vec<f64>>,
    /// Trainable GraphSAGE kernel embeddings of candidate lane relationships.
    neural_embeddings: Vec<Vec<f32>>,
    /// Frozen GraphSAGE encoder plus jointly-trained robust output adapters.
    /// The point adapters use Huber gradients; each quantile adapter uses its
    /// own pinball gradient on the same graph-aware lane state.
    joint_heads: Option<JointMarketHeads>,
    expert_shift_calibration: Option<ExpertShiftCalibration>,
    expert_labels: Vec<ExpertEventLabel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct JointMarketHeads {
    primary_huber: Vec<f64>,
    secondary_huber: Vec<f64>,
    primary_quantiles: Vec<Vec<f64>>,
    secondary_quantiles: Vec<Vec<f64>>,
}

/// Train-only class centroids for optional reviewer-labelled event calibration.
/// Recorded labels remain exposed independently on explanations; these values
/// only calibrate the model's classification boundary for later assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ExpertShiftCalibration {
    market: Option<[f64; 3]>,
    local_or_mix: Option<[f64; 3]>,
    no_shift: Option<[f64; 3]>,
}

impl MarketStructureForecaster {
    pub fn new(config: MarketStructureConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            lane_ids: Vec::new(),
            origin_ids: Vec::new(),
            destination_ids: Vec::new(),
            coordinates: Vec::new(),
            hierarchy_groups: Vec::new(),
            timestamps: Vec::new(),
            frequency: String::new(),
            relationships: Vec::new(),
            target_names: Vec::new(),
            primary_means: Vec::new(),
            secondary_means: Vec::new(),
            primary_scales: Vec::new(),
            primary_interval_radii: Vec::new(),
            interval_calibration_multiplier: 1.0,
            secondary_scales: Vec::new(),
            weekly_primary: Vec::new(),
            weekly_secondary: Vec::new(),
            primary_calendar_weights: Vec::new(),
            secondary_calendar_weights: Vec::new(),
            primary_history: Vec::new(),
            primary_observed: Vec::new(),
            secondary_history: Vec::new(),
            calendar_width: 0,
            last_calendar: Vec::new(),
            mix_coefficients: Vec::new(),
            cross_target_couplings: Vec::new(),
            last_mix: None,
            neural_embeddings: Vec::new(),
            joint_heads: None,
            expert_shift_calibration: None,
            expert_labels: Vec::new(),
        })
    }

    pub fn fit(&mut self, frame: &MarketPanelFrame) -> Result<()> {
        frame.validate()?;
        self.config.validate()?;
        let lanes = frame.lane_ids.len();
        let (log_primary, primary_observed) = log_primary_with_missing(&frame.primary);
        let log_secondary = matrix_map(&frame.secondary, |x| (x + 1.0).ln());
        self.primary_means = primary_means_with_hierarchy(frame, &log_primary, &primary_observed)?;
        self.secondary_means = column_means(&log_secondary);
        let primary_residuals =
            centered_masked(&log_primary, &self.primary_means, &primary_observed);
        self.primary_scales = primary_scales_with_hierarchy(
            frame,
            &log_primary,
            &primary_observed,
            &self.primary_means,
        )?;
        self.primary_interval_radii =
            primary_interval_radii(&primary_residuals, &primary_observed, &self.primary_scales);
        self.interval_calibration_multiplier = if self.config.calibrate_intervals {
            interval_calibration_multiplier(frame, &self.config)?
        } else {
            1.0
        };
        self.secondary_scales = column_scales(&centered(&log_secondary, &self.secondary_means));
        self.weekly_primary =
            weekly_effects_masked(&primary_residuals, &primary_observed, &frame.timestamps);
        self.weekly_secondary = weekly_effects(
            &centered(&log_secondary, &self.secondary_means),
            &frame.timestamps,
        );
        self.primary_calendar_weights =
            calendar_weights_masked(&primary_residuals, &primary_observed, &frame.calendar);
        self.secondary_calendar_weights = calendar_weights(
            &centered(&log_secondary, &self.secondary_means),
            &frame.calendar,
        );
        self.mix_coefficients = mix_coefficients(&primary_residuals, frame.mix.as_ref());
        self.cross_target_couplings = cross_target_couplings(
            &primary_residuals,
            &centered(&log_secondary, &self.secondary_means),
        );
        let provisional = learn_relationships(
            frame,
            &primary_residuals,
            &primary_observed,
            self.config.top_k,
            self.config.correlation_floor,
            None,
        )?;
        self.neural_embeddings = if self.config.neural_epochs == 0 {
            Vec::new()
        } else {
            fit_graph_kernel(
                frame,
                &self.primary_means,
                &self.secondary_means,
                &provisional,
                self.config.neural_hidden_dim,
                self.config.neural_epochs,
            )?
        };
        self.relationships = learn_relationships(
            frame,
            &primary_residuals,
            &primary_observed,
            self.config.top_k,
            self.config.correlation_floor,
            (!self.neural_embeddings.is_empty()).then_some(&self.neural_embeddings),
        )?;
        self.joint_heads = Some(fit_joint_heads(
            frame,
            &log_primary,
            &primary_observed,
            &log_secondary,
            &self.primary_means,
            &self.secondary_means,
            &self.weekly_primary,
            &self.weekly_secondary,
            &self.primary_calendar_weights,
            &self.secondary_calendar_weights,
            &self.cross_target_couplings,
            &self.relationships,
            &self.neural_embeddings,
            &self.config,
        )?);
        self.expert_shift_calibration = fit_expert_shift_calibration(
            frame,
            &log_primary,
            &primary_observed,
            &self.primary_means,
            &self.primary_scales,
            &self.weekly_primary,
            &self.primary_calendar_weights,
            &self.mix_coefficients,
            &self.relationships,
            &self.config,
        );
        if self.relationships.len() != lanes {
            return Err(GeoStError::InvalidFrame(
                "failed to learn a relationship set for every lane".to_string(),
            ));
        }
        self.lane_ids = frame.lane_ids.clone();
        self.origin_ids = frame.origin_ids.clone();
        self.destination_ids = frame.destination_ids.clone();
        self.coordinates = frame.coordinates.clone();
        self.hierarchy_groups = frame.hierarchy_groups.clone();
        self.timestamps = frame.timestamps.clone();
        self.frequency = frame.frequency.clone();
        self.target_names = frame.target_names.clone();
        self.primary_history = log_primary;
        self.primary_observed = primary_observed;
        self.secondary_history = log_secondary;
        self.calendar_width = frame.calendar.first().map_or(0, Vec::len);
        self.last_calendar = frame.calendar.last().cloned().unwrap_or_default();
        self.last_mix = frame
            .mix
            .as_ref()
            .and_then(|rows| rows.last())
            .map(|lanes| lanes.iter().map(|features| features[0]).collect());
        self.expert_labels = frame.expert_labels.clone();
        Ok(())
    }

    pub fn predict(
        &self,
        horizon: usize,
        future_calendar: Option<&[Vec<f64>]>,
    ) -> Result<Vec<MarketPrediction>> {
        if self.lane_ids.is_empty() {
            return Err(GeoStError::NotFit);
        }
        if horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "forecast horizon must be positive".to_string(),
            ));
        }
        if let Some(calendar) = future_calendar {
            if calendar.len() < horizon
                || calendar.iter().take(horizon).any(|row| {
                    row.len() != self.calendar_width || row.iter().any(|x| !x.is_finite())
                })
            {
                return Err(GeoStError::InvalidFrame(
                    "future calendar must provide finite features for every forecast step"
                        .to_string(),
                ));
            }
        } else if self.calendar_width > 0 {
            return Err(GeoStError::InvalidFrame(
                "future calendar is required because calendar features were fitted".to_string(),
            ));
        }
        let mut primary = last_observed_by_lane(
            &self.primary_history,
            &self.primary_observed,
            &self.primary_means,
        );
        let mut secondary = self
            .secondary_history
            .last()
            .cloned()
            .ok_or(GeoStError::NotFit)?;
        let last_timestamp = *self.timestamps.last().ok_or(GeoStError::NotFit)?;
        let mut result = Vec::with_capacity(horizon * self.lane_ids.len());
        for step in 1..=horizon {
            let timestamp = last_timestamp + step as i64;
            let mut next_primary = vec![0.0; self.lane_ids.len()];
            let mut next_secondary = vec![0.0; self.lane_ids.len()];
            for lane in 0..self.lane_ids.len() {
                let seasonal_primary =
                    self.weekly_primary[(timestamp.rem_euclid(7)) as usize][lane];
                let seasonal_secondary =
                    self.weekly_secondary[(timestamp.rem_euclid(7)) as usize][lane];
                let peer_primary = peer_value(
                    lane,
                    &primary,
                    &self.primary_means,
                    &self.relationships,
                    &self.lane_ids,
                    timestamp,
                );
                let peer_secondary = peer_value(
                    lane,
                    &secondary,
                    &self.secondary_means,
                    &self.relationships,
                    &self.lane_ids,
                    timestamp,
                );
                let calendar = future_calendar
                    .map(|rows| rows[step - 1].as_slice())
                    .unwrap_or(&[]);
                next_primary[lane] = self.primary_means[lane]
                    + seasonal_primary
                    + self.config.local_strength * (primary[lane] - self.primary_means[lane])
                    + self.config.graph_strength * peer_primary
                    + calendar_effect(&self.primary_calendar_weights, calendar);
                next_secondary[lane] = self.secondary_means[lane]
                    + seasonal_secondary
                    + self.config.local_strength * (secondary[lane] - self.secondary_means[lane])
                    + self.config.graph_strength * peer_secondary
                    + calendar_effect(&self.secondary_calendar_weights, calendar);
                // Preserve the primary smoother as the benchmark path. The
                // supporting target consumes its co-movement with the primary
                // target, but cannot push the primary benchmark away from its
                // direct graph and temporal evidence.
                next_secondary[lane] +=
                    self.cross_target_couplings[lane] * (primary[lane] - self.primary_means[lane]);
                if let Some(heads) = &self.joint_heads {
                    let features = forecast_head_features(
                        lane,
                        &primary,
                        &secondary,
                        &self.primary_means,
                        &self.secondary_means,
                        &self.relationships,
                        &self.lane_ids,
                        timestamp,
                        calendar,
                        self.last_mix.as_deref(),
                        &self.neural_embeddings,
                    );
                    // The robust adapters predict residual corrections to the
                    // decomposed graph path, never a competing absolute level.
                    next_primary[lane] += dot(&heads.primary_huber, &features);
                    next_secondary[lane] += dot(&heads.secondary_huber, &features);
                }
                let primary_value = next_primary[lane].exp();
                let spread =
                    self.primary_interval_radii[lane] * self.interval_calibration_multiplier;
                let (lower, upper) = self.joint_heads.as_ref().map_or_else(
                    || (next_primary[lane] - spread, next_primary[lane] + spread),
                    |heads| {
                        let features = forecast_head_features(
                            lane,
                            &primary,
                            &secondary,
                            &self.primary_means,
                            &self.secondary_means,
                            &self.relationships,
                            &self.lane_ids,
                            timestamp,
                            calendar,
                            self.last_mix.as_deref(),
                            &self.neural_embeddings,
                        );
                        let values = heads
                            .primary_quantiles
                            .iter()
                            .map(|head| next_primary[lane] + dot(head, &features))
                            .collect::<Vec<_>>();
                        quantile_interval(
                            &self.config.quantile_levels,
                            &values,
                            next_primary[lane],
                            spread,
                        )
                    },
                );
                result.push(MarketPrediction {
                    lane_id: self.lane_ids[lane].clone(),
                    timestamp,
                    horizon: step,
                    primary: primary_value,
                    primary_lower: lower.exp(),
                    primary_upper: upper.exp(),
                    secondary: (next_secondary[lane].exp() - 1.0).max(0.0),
                });
            }
            primary = next_primary;
            secondary = next_secondary;
        }
        Ok(result)
    }

    pub fn nowcast(&self) -> Result<Vec<MarketExplanation>> {
        if self.lane_ids.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let primary = last_observed_by_lane(
            &self.primary_history,
            &self.primary_observed,
            &self.primary_means,
        );
        // Remove the fitted lane-local mix contribution before passing a lane
        // state through the market graph. A known composition event therefore
        // remains local rather than becoming evidence for a neighbor alert.
        let primary_without_mix = primary
            .iter()
            .enumerate()
            .map(|(lane, value)| {
                *value
                    - self
                        .last_mix
                        .as_ref()
                        .map_or(0.0, |mix| self.mix_coefficients[lane] * mix[lane])
            })
            .collect::<Vec<_>>();
        let active_mix = self
            .last_mix
            .as_ref()
            .is_some_and(|mix| mix.iter().any(|value| value.abs() > 1e-12));
        let mut rows = Vec::with_capacity(self.lane_ids.len());
        for lane in 0..self.lane_ids.len() {
            let observed_index = last_observed_index(&self.primary_observed, lane);
            let timestamp = observed_index
                .map(|index| self.timestamps[index])
                .or_else(|| self.timestamps.last().copied())
                .ok_or(GeoStError::NotFit)?;
            let seasonal = self.weekly_primary[(timestamp.rem_euclid(7)) as usize][lane];
            let calendar = calendar_effect(&self.primary_calendar_weights, &self.last_calendar);
            let peer = peer_value(
                lane,
                &primary_without_mix,
                &self.primary_means,
                &self.relationships,
                &self.lane_ids,
                timestamp,
            );
            let market_log =
                self.primary_means[lane] + seasonal + calendar + self.config.graph_strength * peer;
            let observed = primary[lane];
            let local = observed - market_log;
            let mix_component = self
                .last_mix
                .as_ref()
                .map_or(0.0, |mix| self.mix_coefficients[lane] * mix[lane]);
            let unexplained_local = local - mix_component;
            let scale = self.primary_scales[lane].max(1e-8);
            let observed_delta =
                last_observed_delta(&self.primary_history, &self.primary_observed, lane);
            let heuristic_shift = if observed_index.is_none()
                || observed_index.is_some_and(|index| index + 1 != self.primary_history.len())
                || observed_delta.abs() / scale < 1.0
            {
                // A connected lane can move while this lane does not. Avoid
                // turning that relationship into a red-herring local alert.
                MarketShiftKind::NoShift
            } else if peer.abs() / scale >= self.config.shift_zscore
                && unexplained_local.abs() / scale < self.config.shift_zscore
                // A current caller-supplied mix event contaminates the graph
                // snapshot. Defer market classification until a clean cutoff
                // rather than propagating that local evidence to neighbors.
                && !active_mix
            {
                MarketShiftKind::Market
            } else if local.abs() / scale >= self.config.shift_zscore
                || mix_component.abs() / scale >= self.config.shift_zscore
            {
                MarketShiftKind::LocalOrMix
            } else {
                MarketShiftKind::NoShift
            };
            let shift = self
                .expert_shift_calibration
                .as_ref()
                .and_then(|calibration| {
                    calibrated_shift(
                        calibration,
                        [
                            peer.abs() / scale,
                            local.abs() / scale,
                            mix_component.abs() / scale,
                        ],
                    )
                })
                .unwrap_or(heuristic_shift);
            let label = self
                .expert_labels
                .iter()
                .find(|label| label.lane_id == self.lane_ids[lane] && label.timestamp == timestamp)
                .cloned();
            rows.push(MarketExplanation {
                lane_id: self.lane_ids[lane].clone(),
                timestamp,
                observed_primary: observed_index.map(|_| observed.exp()),
                support: if observed_index.is_some() {
                    MarketSupportKind::Lane
                } else {
                    MarketSupportKind::Hierarchy
                },
                smoothed_primary: market_log.exp(),
                market_component: (self.primary_means[lane] + seasonal + calendar).exp(),
                local_or_mix_component: local,
                seasonal_component: seasonal + calendar,
                residual_component: unexplained_local,
                uncertainty: scale,
                shift,
                top_relationships: self.relationships[lane].clone(),
                expert_label: label,
            });
        }
        Ok(rows)
    }

    pub fn weekly_rollups(
        &self,
        horizon: usize,
        future_calendar: Option<&[Vec<f64>]>,
    ) -> Result<Vec<WeeklyMarketPrediction>> {
        let daily = self.predict(horizon, future_calendar)?;
        let mut grouped = BTreeMap::<(String, i64), Vec<MarketPrediction>>::new();
        for row in daily {
            let week_start_timestamp = row.timestamp - row.timestamp.rem_euclid(7);
            grouped
                .entry((row.lane_id.clone(), week_start_timestamp))
                .or_default()
                .push(row);
        }
        Ok(grouped
            .into_iter()
            .map(|((lane_id, week_start_timestamp), rows)| {
                let days = rows.len();
                WeeklyMarketPrediction {
                    lane_id,
                    week_start_timestamp,
                    days,
                    primary: rows.iter().map(|row| row.primary).sum::<f64>() / days as f64,
                    primary_lower: rows.iter().map(|row| row.primary_lower).sum::<f64>()
                        / days as f64,
                    primary_upper: rows.iter().map(|row| row.primary_upper).sum::<f64>()
                        / days as f64,
                    secondary: rows.iter().map(|row| row.secondary).sum(),
                }
            })
            .collect())
    }

    pub fn relationships(&self) -> Result<Vec<MarketRelationship>> {
        if self.lane_ids.is_empty() {
            Err(GeoStError::NotFit)
        } else {
            Ok(self.relationships.iter().flatten().cloned().collect())
        }
    }

    /// A portable analyst payload for Python notebooks and browser/WASM views.
    /// It intentionally exposes model evidence instead of rendering policy.
    pub fn explorer_payload(&self, horizon: usize) -> Result<serde_json::Value> {
        if self.lane_ids.is_empty() {
            return Err(GeoStError::NotFit);
        }
        let lanes = self
            .lane_ids
            .iter()
            .enumerate()
            .map(|(index, lane_id)| {
                let coordinate = self.coordinates.get(index).copied().unwrap_or([0.0; 4]);
                json!({
                    "lane_id": lane_id,
                    "origin_id": self.origin_ids.get(index),
                    "destination_id": self.destination_ids.get(index),
                    "origin_x": coordinate[0],
                    "origin_y": coordinate[1],
                    "destination_x": coordinate[2],
                    "destination_y": coordinate[3],
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "lanes": lanes,
            // Explorer callers request an immediately inspectable current-state
            // view. Reuse the final known calendar state here; explicit
            // `predict` calls still require caller-supplied future calendar
            // values when calendar features were fitted.
            "forecasts": self.predict(
                horizon,
                (!self.last_calendar.is_empty()).then(|| vec![self.last_calendar.clone(); horizon]).as_deref(),
            )?,
            "explanations": self.nowcast()?,
            "kernels": self.relationships()?,
            "target_names": self.target_names,
        }))
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_json_string()?).map_err(GeoStError::from)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_string(&fs::read_to_string(path).map_err(GeoStError::from)?)
    }
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(GeoStError::from)
    }
    pub fn from_json_string(value: &str) -> Result<Self> {
        serde_json::from_str(value).map_err(GeoStError::from)
    }
}

