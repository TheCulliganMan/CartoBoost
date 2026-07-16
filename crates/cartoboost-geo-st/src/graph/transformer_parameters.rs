#[derive(Clone, Copy)]
struct GraphParameterLayout {
    input: usize,
    time2vec_frequency: usize,
    time2vec_phase: usize,
    in_degree_embedding: usize,
    out_degree_embedding: usize,
    temporal_q: usize,
    temporal_k: usize,
    temporal_v: usize,
    spatial_q: usize,
    spatial_k: usize,
    spatial_v: usize,
    shortest_path_bias: usize,
    router: usize,
    spatial_router: usize,
    spatial_expert_heads: usize,
    expert_heads: usize,
    recurrence: usize,
    lsttn_dilated_convolution: usize,
    lsttn_short_wave: usize,
    stgformer_pointwise: usize,
    lsttn_adaptive_source: usize,
    lsttn_adaptive_target: usize,
    lsttn_weekly_adaptive_source: usize,
    lsttn_weekly_adaptive_target: usize,
    lsttn_short_adaptive_source: usize,
    lsttn_short_adaptive_target: usize,
    lsttn_periodic_projection: usize,
    lsttn_fusion: usize,
    graphon_nodes: usize,
    graphon_time: usize,
    output: usize,
    pretrain_mask_token: usize,
    pretrain_position: usize,
    pretrain_decoder: usize,
    lsttn_patch_embedding: usize,
    lsttn_transformer_ffn: usize,
    lsttn_transformer_norm: usize,
    lsttn_transformer_out: usize,
    lsttn_encoder_decoder: usize,
    lsttn_decoder_q: usize,
    lsttn_decoder_k: usize,
    lsttn_decoder_v: usize,
    lsttn_decoder_out: usize,
    lsttn_decoder_ffn: usize,
    lsttn_decoder_norm: usize,
    total: usize,
}

impl GraphParameterLayout {
    fn new(
        nodes: usize,
        hidden: usize,
        horizons: usize,
        experts: usize,
        graph_order: usize,
        periodicity: usize,
        context_window: usize,
    ) -> Self {
        let input = 0;
        // Seven numeric inputs (local signal, graph signal, learned Time2Vec,
        // daily and weekly phase, and independent in/out-degree encodings),
        // followed by a learned bias per hidden channel.
        let time2vec_frequency = input + hidden * 8;
        let time2vec_phase = time2vec_frequency + hidden;
        let in_degree_embedding = time2vec_phase + hidden;
        let out_degree_embedding = in_degree_embedding + (nodes + 1) * hidden;
        let temporal_q = out_degree_embedding + (nodes + 1) * hidden;
        // `tape_linear` stores a full matrix plus one bias per output.  Keep
        // every Q/K/V range disjoint: sharing a trailing bias with the next
        // projection would silently couple attention parameters.
        let projection = hidden * (hidden + 1);
        // STGormer uses three temporal and three spatial transformer blocks.
        // Reserve one independent Q/K/V projection per block; the other
        // profiles use the first stage of these generic attention ranges.
        let transformer_blocks = 4;
        let temporal_k = temporal_q + transformer_blocks * projection;
        let temporal_v = temporal_k + transformer_blocks * projection;
        let spatial_q = temporal_v + transformer_blocks * projection;
        let spatial_k = spatial_q + transformer_blocks * projection;
        let spatial_v = spatial_k + transformer_blocks * projection;
        let shortest_path_bias = spatial_v + transformer_blocks * projection;
        let router = shortest_path_bias + nodes + 1;
        // STGormer keeps independent temporal and spatial routers and
        // expert FNNs.  `router`/`expert_heads` name the temporal path for
        // backwards readability; the adjacent ranges are the spatial path.
        let spatial_router = router + experts * (hidden + 1);
        let spatial_expert_heads = spatial_router + experts * (hidden + 1);
        let expert_heads = spatial_expert_heads + experts * horizons * (hidden + 1);
        // Long/short fusion, recurrent gates, and high-order propagation
        // gates share this learned block; each profile uses the appropriate
        // subset in its forward graph.
        let recurrence = expert_heads + experts * horizons * (hidden + 1);
        let lsttn_dilated_convolution = recurrence + (graph_order + 6) * hidden;
        // Two gated temporal-convolution blocks for LSTTN's short-term
        // Graph WaveNet branch.  Each owns a filter and gate convolution,
        // their biases, and a post-adaptive-graph channel projection.
        let lsttn_short_wave = lsttn_dilated_convolution + 4 * (3 * hidden * hidden + hidden);
        let lsttn_short_layer = 12 * hidden * hidden + 6 * hidden;
        let stgformer_pointwise = lsttn_short_wave
            + 2 * hidden
            + hidden
            + 8 * lsttn_short_layer
            + 2 * (hidden * hidden + hidden);
        let lsttn_adaptive_source = stgformer_pointwise + graph_order * hidden * (hidden + 1);
        let lsttn_adaptive_target = lsttn_adaptive_source + nodes * 10;
        let lsttn_weekly_adaptive_source = lsttn_adaptive_target + nodes * 10;
        let lsttn_weekly_adaptive_target = lsttn_weekly_adaptive_source + nodes * 10;
        let lsttn_short_adaptive_source = lsttn_weekly_adaptive_target + nodes * 10;
        let lsttn_short_adaptive_target = lsttn_short_adaptive_source + nodes * 10;
        let lsttn_periodic_projection = lsttn_short_adaptive_target + nodes * 10;
        // LSTTN has three explicit MLP stages: long-trend/day/week fusion,
        // a second trend-seasonality projection, and short/long fusion.
        let lsttn_fusion = lsttn_periodic_projection + 2 * (7 * hidden * hidden + hidden);
        let graphon_nodes = lsttn_fusion + (6 * hidden * hidden + 3 * hidden);
        let graphon_time = graphon_nodes + experts * nodes;
        let output = graphon_time + experts * hidden;
        let pretrain_mask_token = output + horizons * (hidden + 1);
        // Allocate an independent learned position for every patch in the
        // configured long context. This keeps long-horizon positions distinct
        // instead of aliasing later weeks onto an old fixed-size table.
        let patch_width = (periodicity / 24).max(1);
        let pretrain_positions = context_window.div_ceil(patch_width).max(1);
        let pretrain_position = pretrain_mask_token + hidden;
        let pretrain_decoder = pretrain_position + pretrain_positions * hidden;
        let lsttn_patch_embedding = pretrain_decoder + patch_width * (hidden + 1);
        let lsttn_transformer_ffn = lsttn_patch_embedding + patch_width * hidden + hidden;
        let transformer_ffn = 8 * hidden * hidden + 5 * hidden;
        let lsttn_transformer_norm = lsttn_transformer_ffn + transformer_blocks * transformer_ffn;
        let lsttn_transformer_out = lsttn_transformer_norm + transformer_blocks * 4 * hidden;
        let lsttn_encoder_decoder = lsttn_transformer_out + transformer_blocks * projection;
        let lsttn_decoder_q = lsttn_encoder_decoder + projection;
        let lsttn_decoder_k = lsttn_decoder_q + projection;
        let lsttn_decoder_v = lsttn_decoder_k + projection;
        let lsttn_decoder_out = lsttn_decoder_v + projection;
        let lsttn_decoder_ffn = lsttn_decoder_out + projection;
        let lsttn_decoder_norm = lsttn_decoder_ffn + transformer_ffn;
        let total = lsttn_decoder_norm + 4 * hidden;
        Self {
            input,
            time2vec_frequency,
            time2vec_phase,
            in_degree_embedding,
            out_degree_embedding,
            temporal_q,
            temporal_k,
            temporal_v,
            spatial_q,
            spatial_k,
            spatial_v,
            shortest_path_bias,
            router,
            spatial_router,
            spatial_expert_heads,
            expert_heads,
            recurrence,
            lsttn_dilated_convolution,
            lsttn_short_wave,
            stgformer_pointwise,
            lsttn_adaptive_source,
            lsttn_adaptive_target,
            lsttn_weekly_adaptive_source,
            lsttn_weekly_adaptive_target,
            lsttn_short_adaptive_source,
            lsttn_short_adaptive_target,
            lsttn_periodic_projection,
            lsttn_fusion,
            graphon_nodes,
            graphon_time,
            output,
            pretrain_mask_token,
            pretrain_position,
            pretrain_decoder,
            lsttn_patch_embedding,
            lsttn_transformer_ffn,
            lsttn_transformer_norm,
            lsttn_transformer_out,
            lsttn_encoder_decoder,
            lsttn_decoder_q,
            lsttn_decoder_k,
            lsttn_decoder_v,
            lsttn_decoder_out,
            lsttn_decoder_ffn,
            lsttn_decoder_norm,
            total,
        }
    }
}

impl TrainableGraphTransformerState {
    #[allow(clippy::too_many_arguments)]
    fn initialized(
        nodes: usize,
        hidden: usize,
        attention_heads: usize,
        periodicity: usize,
        recent_window: usize,
        context_window: usize,
        horizons: usize,
        experts: usize,
        graph_order: usize,
        seed: u64,
    ) -> Self {
        let layout = GraphParameterLayout::new(
            nodes,
            hidden,
            horizons,
            experts,
            graph_order,
            periodicity,
            context_window,
        );
        let mut state = seed;
        let mut parameters = (0..layout.total)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) - 0.5) * 0.08
            })
            .collect::<Vec<_>>();
        for layer in 0..4 {
            let norm = layout.lsttn_transformer_norm + layer * 4 * hidden;
            parameters[norm..norm + hidden].fill(1.0);
            parameters[norm + 2 * hidden..norm + 3 * hidden].fill(1.0);
        }
        parameters[layout.lsttn_decoder_norm..layout.lsttn_decoder_norm + hidden].fill(1.0);
        parameters[layout.lsttn_decoder_norm + 2 * hidden..layout.lsttn_decoder_norm + 3 * hidden]
            .fill(1.0);
        let short_layer_width = 12 * hidden * hidden + 6 * hidden;
        let short_layers = layout.lsttn_short_wave + 2 * hidden + hidden;
        for layer in 0..8 {
            let gamma =
                short_layers + layer * short_layer_width + 12 * hidden * hidden + 4 * hidden;
            parameters[gamma..gamma + hidden].fill(1.0);
        }
        Self {
            first_moment: vec![0.0; layout.total],
            second_moment: vec![0.0; layout.total],
            parameters,
            steps: 0,
            nodes,
            hidden,
            attention_heads,
            periodicity,
            recent_window,
            context_window,
            horizons,
            experts,
            graph_order,
            periodic_short_lag: 0,
            target_scale: 1.0,
            normalized_zero: 0.0,
        }
    }

    fn layout(&self) -> GraphParameterLayout {
        GraphParameterLayout::new(
            self.nodes,
            self.hidden,
            self.horizons,
            self.experts,
            self.graph_order,
            self.periodicity,
            self.context_window,
        )
    }

    fn frozen_lsttn_patch_representations(
        &self,
        window: &[Vec<f64>],
        _adjacency: &CsrAdjacency,
        _phase_offset: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let layout = self.layout();
        let patch_width = (self.periodicity / 24).max(1);
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / self.hidden;
        let patch_count = window.len() / patch_width;
        let projection = self.hidden * (self.hidden + 1);
        let ffn_width = 8 * self.hidden * self.hidden + 5 * self.hidden;
        let by_node = (0..self.nodes)
            .into_par_iter()
            .map(|node| {
                let mut sequence = (0..patch_count)
                    .map(|patch| {
                        (0..self.hidden)
                            .map(|channel| {
                                let mut value = self.parameters[layout.lsttn_patch_embedding
                                    + patch_width * self.hidden
                                    + channel];
                                for offset in 0..patch_width {
                                    value += self.parameters[layout.lsttn_patch_embedding
                                        + offset * self.hidden
                                        + channel]
                                        * window[patch * patch_width + offset][node];
                                }
                                (value
                                    + self.parameters[layout.pretrain_position
                                        + (patch % position_count) * self.hidden
                                        + channel])
                                    * (self.hidden as f64).sqrt()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for layer in 0..4 {
                    sequence = numeric_transformer_encoder_layer(
                        &self.parameters,
                        &sequence,
                        layout.temporal_q + layer * projection,
                        layout.temporal_k + layer * projection,
                        layout.temporal_v + layer * projection,
                        layout.lsttn_transformer_out + layer * projection,
                        layout.lsttn_transformer_ffn + layer * ffn_width,
                        layout.lsttn_transformer_norm + layer * 4 * self.hidden,
                        self.hidden,
                        self.attention_heads,
                    );
                }
                sequence
            })
            .collect::<Vec<_>>();
        (0..patch_count)
            .map(|patch| {
                (0..self.nodes)
                    .map(|node| {
                        by_node[node][patch]
                            .iter()
                            .map(|value| *value as f32)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn adamw_step(&mut self, gradients: &[f64], learning_rate: f64, weight_decay: f64) {
        self.steps += 1;
        let step = self.steps as f64;
        for (index, gradient) in gradients.iter().copied().enumerate() {
            let gradient = gradient + weight_decay * self.parameters[index];
            self.first_moment[index] = 0.9 * self.first_moment[index] + 0.1 * gradient;
            self.second_moment[index] =
                0.999 * self.second_moment[index] + 0.001 * gradient * gradient;
            let corrected_first = self.first_moment[index] / (1.0 - 0.9_f64.powf(step));
            let corrected_second = self.second_moment[index] / (1.0 - 0.999_f64.powf(step));
            self.parameters[index] -=
                learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8);
        }
    }

    /// The paper pretrains MST and freezes it before fitting LSTTN.  These
    /// ranges own the patch projection, learned patch positions, and the
    /// temporal Q/K/V projections used to contextualize patch embeddings.
    fn freeze_lsttn_transformer_gradients(
        &self,
        layout: GraphParameterLayout,
        gradients: &mut [f64],
    ) {
        gradients[layout.input..layout.spatial_q].fill(0.0);
        gradients[layout.pretrain_mask_token..layout.total].fill(0.0);
    }

    /// Build one paper-style LSTTN training example without mutating model
    /// state.  This is deliberately separate from Adam so a 32-window batch
    /// can evaluate on Rayon workers and reduce its gradients deterministically
    /// before taking one optimizer step, just like the reference mini-batch
    /// trainer.
    fn lsttn_example_loss_and_gradients(
        &self,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        targets: &[Vec<f64>],
        phase_offset: usize,
        frozen_patches: Option<&[Vec<Vec<f32>>]>,
        time_features: Option<&[Vec<f64>]>,
    ) -> (f64, Vec<f64>) {
        let owned_frozen = frozen_patches
            .is_none()
            .then(|| self.frozen_lsttn_patch_representations(window, adjacency, phase_offset));
        let frozen_patches = frozen_patches.or(owned_frozen.as_deref());
        let (tape, outputs, _, _) = self.forward(GraphForwardContext {
            profile: &GraphTransformerProfile::LongShortFusion,
            window,
            adjacency,
            excluded_expert: None,
            phase_offset,
            long_context_is_pooled: false,
            lsttn_frozen_patches: frozen_patches,
            lsttn_time_features: time_features,
            deferred: false,
            training: true,
        });
        let mut loss = tape.constant(0.0);
        let valid = targets
            .iter()
            .flatten()
            .filter(|target| (**target - self.normalized_zero).abs() > 1e-12)
            .count();
        let scale = tape.constant(1.0 / valid.max(1) as f64);
        for node in 0..self.nodes {
            for horizon in 0..self.horizons {
                if (targets[horizon][node] - self.normalized_zero).abs() <= 1e-12 {
                    continue;
                }
                let residual = tape.add(
                    outputs[node][horizon],
                    tape.constant(-targets[horizon][node]),
                );
                let residual = tape.mul(residual, tape.constant(self.target_scale));
                let mae = tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)));
                loss = tape.add(loss, tape.mul(mae, scale));
            }
        }
        (tape.value(loss), tape.backward(loss, self.parameters.len()))
    }

    #[allow(clippy::too_many_arguments)]
    fn train_example_with_context(
        &mut self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        targets: &[Vec<f64>],
        excluded_expert: Option<usize>,
        learning_rate: f64,
        weight_decay: f64,
        phase_offset: usize,
        long_context_is_pooled: bool,
        backend: Option<&BackendSelection>,
    ) -> Result<f64> {
        // LSTTN freezes its pretrained masked-subseries Transformer during
        // supervised fitting.  Keep this path on the native tape so the
        // frozen ranges are enforced identically on every supported host.
        let accelerated = *profile != GraphTransformerProfile::LongShortFusion
            && backend.is_some_and(|selection| selection.selected != "cpu");
        let frozen_lsttn = (*profile == GraphTransformerProfile::LongShortFusion)
            .then(|| self.frozen_lsttn_patch_representations(window, adjacency, phase_offset));
        let (tape, outputs, router_weights, _) = self.forward(GraphForwardContext {
            profile,
            window,
            adjacency,
            excluded_expert,
            phase_offset,
            long_context_is_pooled,
            lsttn_frozen_patches: frozen_lsttn.as_deref(),
            lsttn_time_features: None,
            deferred: accelerated,
            training: true,
        });
        let mut loss = tape.constant(0.0);
        let valid = targets
            .iter()
            .flatten()
            .filter(|target| {
                *profile != GraphTransformerProfile::LongShortFusion
                    || (**target - self.normalized_zero).abs() > 1e-12
            })
            .count();
        let scale = tape.constant(1.0 / valid.max(1) as f64);
        for node in 0..self.nodes {
            for horizon in 0..self.horizons {
                if *profile == GraphTransformerProfile::LongShortFusion
                    && (targets[horizon][node] - self.normalized_zero).abs() <= 1e-12
                {
                    continue;
                }
                let residual = tape.add(
                    outputs[node][horizon],
                    tape.constant(-targets[horizon][node]),
                );
                let point_loss = if *profile == GraphTransformerProfile::LongShortFusion {
                    let residual = tape.mul(residual, tape.constant(self.target_scale));
                    // The reference LSTTN trains its forecast stage with
                    // masked MAE.  The infinitesimal smoothing keeps the
                    // derivative defined at zero for the native tape while
                    // preserving the MAE value at reporting precision.
                    tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)))
                } else {
                    tape.mul(residual, residual)
                };
                loss = tape.add(loss, tape.mul(point_loss, scale));
            }
        }
        if *profile == GraphTransformerProfile::HeterogeneousMoE && !router_weights.is_empty() {
            // STGormer's auxiliary router objective penalizes concentrated
            // expert probability mass.  The mean routing probability per
            // expert is differentiable, so this keeps experts available
            // rather than allowing one feed-forward path to collapse.
            let count = tape.constant(router_weights.len() as f64);
            let expert_count = tape.constant(self.experts as f64);
            let coefficient = tape.constant(0.01);
            for expert in 0..self.experts {
                let mass = router_weights
                    .iter()
                    .fold(tape.constant(0.0), |sum, weights| {
                        tape.add(sum, weights[expert])
                    });
                let mean_mass = tape.div(mass, count);
                loss = tape.add(
                    loss,
                    tape.mul(
                        coefficient,
                        tape.mul(expert_count, tape.mul(mean_mass, mean_mass)),
                    ),
                );
            }
        }
        if accelerated {
            let next_step = self.steps + 1;
            let value = tape.accelerated_train_step(
                backend.expect("non-CPU backend is present"),
                loss,
                &mut self.parameters,
                &mut self.first_moment,
                &mut self.second_moment,
                next_step,
                learning_rate,
                weight_decay,
            )?;
            self.steps = next_step;
            Ok(value)
        } else {
            let value = tape.value(loss);
            let mut gradients = tape.backward(loss, self.parameters.len());
            if *profile == GraphTransformerProfile::LongShortFusion {
                self.freeze_lsttn_transformer_gradients(self.layout(), &mut gradients);
            }
            self.adamw_step(&gradients, learning_rate, weight_decay);
            Ok(value)
        }
    }

    /// LSTTN's self-supervised stage: encode the unmasked equal-length
    /// subseries, insert learned mask tokens at the withheld positions, and
    /// decode only those patches.  The shared input/Q/K/V projections are the
    /// same ones used by the forecasting path, so pretraining transfers a
    /// contextual long-history representation instead of training a detached
    /// auxiliary model.
    #[allow(clippy::needless_range_loop)]
    fn train_masked_subseries_reconstruction(
        &mut self,
        window: &[Vec<f64>],
        learning_rate: f64,
        weight_decay: f64,
        backend: Option<&BackendSelection>,
    ) -> Result<f64> {
        let patch_width = (self.periodicity / 24).max(1);
        if !window.len().is_multiple_of(patch_width) {
            return Err(GeoStError::InvalidFrame(format!(
                "LSTTN long-history window length {} must be divisible by patch width {}",
                window.len(),
                patch_width
            )));
        }
        let patches = window.len() / patch_width;
        if patches < 2 {
            return Err(GeoStError::InvalidFrame(
                "LSTTN masked-subseries pretraining requires at least two patches".to_string(),
            ));
        }
        let masked = masked_patch_indices(patches, self.steps);
        let visible = (0..patches)
            .filter(|patch| !masked.contains(patch))
            .collect::<Vec<_>>();
        let layout = self.layout();
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / self.hidden;
        let mut total_loss = 0.0;
        let mut gradients = vec![0.0; self.parameters.len()];
        let valid_reconstruction_values = masked
            .iter()
            .flat_map(|patch| {
                (0..self.nodes).flat_map(move |node| {
                    (0..patch_width).map(move |offset| window[patch * patch_width + offset][node])
                })
            })
            .filter(|target| (*target - self.normalized_zero).abs() > 1e-12)
            .count();
        let accelerated = backend.is_some_and(|selection| selection.selected != "cpu");
        // Self-attention over the visible long-history patches is independent
        // per H3 node.  Keep a bounded tape for both CPU and accelerator
        // execution, while accumulating the exact full-batch gradient before
        // taking the optimizer step.  This prevents a city-scale panel from
        // multiplying `nodes × patches × hidden` tape state at once.
        const CPU_PRETRAIN_NODE_BATCH: usize = 16;
        const ACCELERATOR_PRETRAIN_NODE_BATCH: usize = 32;
        let node_batch_size = if accelerated {
            ACCELERATOR_PRETRAIN_NODE_BATCH
        } else {
            CPU_PRETRAIN_NODE_BATCH
        }
        .min(self.nodes)
        .max(1);
        // One tape now contains every randomly masked patch for a bounded
        // node batch.  Constructing a tape per patch used to repeat the
        // visible-context encoder and issue hundreds of tiny accelerator
        // launches for a long daily history.  Keeping the node dimension
        // bounded preserves the city-scale memory ceiling while making each
        // reconstruction pass a single batched objective.
        for node_start in (0..self.nodes).step_by(node_batch_size) {
            let node_end = (node_start + node_batch_size).min(self.nodes);
            let tape = if accelerated {
                AutodiffTape::deferred()
            } else {
                AutodiffTape::new()
            };
            let parameter_nodes = self
                .parameters
                .iter()
                .enumerate()
                .map(|(index, value)| tape.parameter(index, *value))
                .collect::<Vec<_>>();
            let parameter = |index: usize| parameter_nodes[index];
            let mut loss = tape.constant(0.0);
            let scale = tape.constant(1.0 / valid_reconstruction_values.max(1) as f64);
            for node in node_start..node_end {
                let mut encoded = visible
                    .iter()
                    .map(|patch| {
                        (0..self.hidden)
                            .map(|channel| {
                                let mut value = parameter(
                                    layout.lsttn_patch_embedding
                                        + patch_width * self.hidden
                                        + channel,
                                );
                                for offset in 0..patch_width {
                                    value = tape.add(
                                        value,
                                        tape.mul(
                                            parameter(
                                                layout.lsttn_patch_embedding
                                                    + offset * self.hidden
                                                    + channel,
                                            ),
                                            tape.constant(
                                                window[patch * patch_width + offset][node],
                                            ),
                                        ),
                                    );
                                }
                                tape_deterministic_dropout(
                                    &tape,
                                    tape.mul(
                                        tape.add(
                                            value,
                                            parameter(
                                                layout.pretrain_position
                                                    + (patch % position_count) * self.hidden
                                                    + channel,
                                            ),
                                        ),
                                        tape.constant((self.hidden as f64).sqrt()),
                                    ),
                                    self.steps ^ node as u64,
                                    patch * self.hidden + channel,
                                    true,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let projection = self.hidden * (self.hidden + 1);
                let ffn_width = 8 * self.hidden * self.hidden + 5 * self.hidden;
                for layer in 0..4 {
                    encoded = tape_transformer_encoder_layer(
                        &tape,
                        &parameter_nodes,
                        &encoded,
                        layout.temporal_q + layer * projection,
                        layout.temporal_k + layer * projection,
                        layout.temporal_v + layer * projection,
                        layout.lsttn_transformer_out + layer * projection,
                        layout.lsttn_transformer_ffn + layer * ffn_width,
                        layout.lsttn_transformer_norm + layer * 4 * self.hidden,
                        self.hidden,
                        self.attention_heads,
                        self.steps ^ ((node as u64) << 16) ^ layer as u64,
                        true,
                    );
                }
                let decoder_scale = tape.constant((self.hidden as f64).sqrt());
                let mut decoder_input = encoded
                    .iter()
                    .map(|token| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.lsttn_encoder_decoder,
                            token,
                            self.hidden,
                            self.hidden,
                        )
                        .into_iter()
                        .map(|value| tape.mul(value, decoder_scale))
                        .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                decoder_input.extend(masked.iter().map(|patch| {
                    (0..self.hidden)
                        .map(|channel| {
                            tape.mul(
                                tape_deterministic_dropout(
                                    &tape,
                                    tape.add(
                                        parameter(layout.pretrain_mask_token + channel),
                                        parameter(
                                            layout.pretrain_position
                                                + (patch % position_count) * self.hidden
                                                + channel,
                                        ),
                                    ),
                                    self.steps ^ ((node as u64) << 32),
                                    patch * self.hidden + channel,
                                    true,
                                ),
                                decoder_scale,
                            )
                        })
                        .collect::<Vec<_>>()
                }));
                let decoded = tape_transformer_encoder_layer(
                    &tape,
                    &parameter_nodes,
                    &decoder_input,
                    layout.lsttn_decoder_q,
                    layout.lsttn_decoder_k,
                    layout.lsttn_decoder_v,
                    layout.lsttn_decoder_out,
                    layout.lsttn_decoder_ffn,
                    layout.lsttn_decoder_norm,
                    self.hidden,
                    self.attention_heads,
                    self.steps ^ ((node as u64) << 48),
                    true,
                );
                for (masked_index, patch) in masked.iter().enumerate() {
                    let context = &decoded[visible.len() + masked_index];
                    for offset in 0..patch_width {
                        let target = window[patch * patch_width + offset][node];
                        if (target - self.normalized_zero).abs() <= 1e-12 {
                            continue;
                        }
                        let mut prediction =
                            parameter(layout.pretrain_decoder + patch_width * self.hidden + offset);
                        for (channel, context_value) in context.iter().enumerate() {
                            prediction = tape.add(
                                prediction,
                                tape.mul(
                                    parameter(
                                        layout.pretrain_decoder + offset * self.hidden + channel,
                                    ),
                                    *context_value,
                                ),
                            );
                        }
                        let residual = tape.add(prediction, tape.constant(-target));
                        let residual = tape.mul(residual, tape.constant(self.target_scale));
                        let mae =
                            tape.sqrt(tape.add(tape.mul(residual, residual), tape.constant(1e-12)));
                        loss = tape.add(loss, tape.mul(scale, mae));
                    }
                }
            }
            if accelerated {
                let next_step = self.steps + 1;
                total_loss += tape.accelerated_train_step(
                    backend.expect("non-CPU backend is present"),
                    loss,
                    &mut self.parameters,
                    &mut self.first_moment,
                    &mut self.second_moment,
                    next_step,
                    learning_rate,
                    weight_decay,
                )?;
                self.steps = next_step;
            } else {
                total_loss += tape.value(loss);
                for (total, gradient) in gradients
                    .iter_mut()
                    .zip(tape.backward(loss, self.parameters.len()))
                {
                    *total += gradient;
                }
            }
        }
        if !accelerated {
            self.adamw_step(&gradients, learning_rate, weight_decay);
        }
        Ok(total_loss)
    }

    #[cfg(test)]
    fn predict_window(
        &self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
    ) -> Vec<Vec<f64>> {
        self.predict_window_with_context(profile, window, adjacency, 0, false, None, None)
            .expect("CPU graph transformer prediction")
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_window_with_context(
        &self,
        profile: &GraphTransformerProfile,
        window: &[Vec<f64>],
        adjacency: &CsrAdjacency,
        phase_offset: usize,
        long_context_is_pooled: bool,
        backend: Option<&BackendSelection>,
        time_features: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<f64>>> {
        let accelerated = *profile == GraphTransformerProfile::LongShortFusion
            && backend.is_some_and(|selection| selection.selected != "cpu");
        let frozen_lsttn = (*profile == GraphTransformerProfile::LongShortFusion)
            .then(|| self.frozen_lsttn_patch_representations(window, adjacency, phase_offset));
        let (tape, outputs, _, _) = self.forward(GraphForwardContext {
            profile,
            window,
            adjacency,
            excluded_expert: None,
            phase_offset,
            long_context_is_pooled,
            lsttn_frozen_patches: frozen_lsttn.as_deref(),
            lsttn_time_features: time_features,
            deferred: accelerated,
            training: false,
        });
        if accelerated {
            let selection = backend.expect("non-CPU backend is present");
            let values = tape.accelerated_values(selection)?;
            return Ok((0..self.horizons)
                .map(|horizon| {
                    (0..self.nodes)
                        .map(|node| values[outputs[node][horizon]] as f64)
                        .collect()
                })
                .collect());
        }
        Ok((0..self.horizons)
            .map(|horizon| {
                (0..self.nodes)
                    .map(|node| tape.value(outputs[node][horizon]))
                    .collect()
            })
            .collect())
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(&self, context: GraphForwardContext<'_>) -> GraphForwardOutput {
        let GraphForwardContext {
            profile,
            window,
            adjacency,
            excluded_expert,
            phase_offset,
            long_context_is_pooled,
            lsttn_frozen_patches,
            lsttn_time_features,
            deferred,
            training,
        } = context;
        let layout = self.layout();
        let tape = if deferred {
            AutodiffTape::deferred()
        } else {
            AutodiffTape::new()
        };
        let parameter_nodes = self
            .parameters
            .iter()
            .enumerate()
            .map(|(index, value)| tape.parameter(index, *value))
            .collect::<Vec<_>>();
        let parameter = |_tape: &AutodiffTape, index: usize| parameter_nodes[index];
        let nodes = self.nodes;
        let hidden = self.hidden;
        let times = window.len();
        let native_patch_width = (self.periodicity / 24).max(1);
        let time_scale = if long_context_is_pooled {
            native_patch_width
        } else {
            1
        };
        let effective_periodicity = (self.periodicity / time_scale).max(1);
        let effective_short_periodicity = (if self.periodic_short_lag == 0 {
            self.periodicity
        } else {
            self.periodic_short_lag
        } / time_scale)
            .max(1);
        // LSTTN's periodic graph convolution uses both directed structural
        // diffusions as well as its learned adaptive diffusion.  Preserve a
        // normalized reverse graph rather than treating the supplied road
        // graph as undirected.
        let reverse_adjacency = adjacency.transpose(nodes).row_normalized();
        let adaptive_adjacency = adjacency.with_adaptive_self_candidates(nodes);
        let observed_values = window
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| tape.constant(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let adjacency_weights = adjacency
            .data
            .iter()
            .map(|weight| tape.constant(*weight))
            .collect::<Vec<_>>();
        let reverse_adjacency_weights = reverse_adjacency
            .data
            .iter()
            .map(|weight| tape.constant(*weight))
            .collect::<Vec<_>>();
        let mut required_embedding_times = vec![true; times];
        if *profile == GraphTransformerProfile::LongShortFusion && lsttn_frozen_patches.is_some() {
            required_embedding_times.fill(false);
        }
        let mut graph_values = vec![vec![0usize; nodes]; times];
        for time in 0..times {
            if !required_embedding_times[time] {
                continue;
            }
            for target in 0..nodes {
                graph_values[time][target] = (adjacency.indptr[target]
                    ..adjacency.indptr[target + 1])
                    .fold(tape.constant(0.0), |sum, edge| {
                        tape.add(
                            sum,
                            tape.mul(
                                adjacency_weights[edge],
                                observed_values[time][adjacency.indices[edge]],
                            ),
                        )
                    });
            }
        }
        let degrees = graph_in_degrees(adjacency, nodes);
        let out_degrees = graph_out_degrees(adjacency, nodes);
        let positions = (0..times)
            .map(|time| tape.constant((time + 1) as f64 / times as f64))
            .collect::<Vec<_>>();
        let periodic_features = (0..times)
            .map(|time| {
                [
                    tape.constant(periodic_phase(
                        phase_offset + time + 1,
                        effective_short_periodicity,
                    )),
                    tape.constant(periodic_phase(
                        phase_offset + time + 1,
                        effective_periodicity,
                    )),
                ]
            })
            .collect::<Vec<_>>();
        let degree_features = (0..nodes)
            .map(|node| {
                [
                    tape.constant(degrees[node] / nodes.max(1) as f64),
                    tape.constant(out_degrees[node] / nodes.max(1) as f64),
                ]
            })
            .collect::<Vec<_>>();
        let mut embedding = vec![vec![vec![0usize; hidden]; nodes]; times];
        for time in 0..times {
            if !required_embedding_times[time] {
                continue;
            }
            let position = positions[time];
            for node in 0..nodes {
                for channel in 0..hidden {
                    let time2vec = tape.sin(tape.add(
                        tape.mul(
                            parameter(&tape, layout.time2vec_frequency + channel),
                            position,
                        ),
                        parameter(&tape, layout.time2vec_phase + channel),
                    ));
                    let inputs = [
                        observed_values[time][node],
                        graph_values[time][node],
                        time2vec,
                        periodic_features[time][0],
                        periodic_features[time][1],
                        degree_features[node][0],
                        degree_features[node][1],
                    ];
                    let mut value = parameter(&tape, layout.input + 7 * hidden + channel);
                    for (input, input_value) in inputs.iter().enumerate() {
                        value = tape.add(
                            value,
                            tape.mul(
                                parameter(&tape, layout.input + input * hidden + channel),
                                *input_value,
                            ),
                        );
                    }
                    // STGormer uses independent learned embeddings indexed by
                    // in- and out-degree, rather than treating centrality as
                    // a single scalar feature.
                    let in_degree = degrees[node].round().clamp(0.0, nodes as f64) as usize;
                    let out_degree = out_degrees[node].round().clamp(0.0, nodes as f64) as usize;
                    value = tape.add(
                        value,
                        parameter(
                            &tape,
                            layout.in_degree_embedding + in_degree * hidden + channel,
                        ),
                    );
                    value = tape.add(
                        value,
                        parameter(
                            &tape,
                            layout.out_degree_embedding + out_degree * hidden + channel,
                        ),
                    );
                    embedding[time][node][channel] = tape.tanh(value);
                }
            }
        }

        let mut temporal = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::LongShortFusion {
            // LSTTN keeps the latest native-resolution embedding available to
            // its profile path. Its long branch builds learned
            // patch states and one-query contextual attention below;
            // materializing generic all-pairs attention here would make the
            // long-context profile quadratic in history length.
            temporal.clone_from(&embedding[times - 1]);
        } else {
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.temporal_q,
                    &embedding[times - 1][node],
                    hidden,
                    hidden,
                );
                let mut keys = Vec::with_capacity(times);
                let mut values = Vec::with_capacity(times);
                for time in 0..times {
                    let key = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_k,
                        &embedding[time][node],
                        hidden,
                        hidden,
                    );
                    let value = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_v,
                        &embedding[time][node],
                        hidden,
                        hidden,
                    );
                    keys.push(key);
                    values.push(value);
                }
                for head in 0..self.attention_heads {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    if *profile == GraphTransformerProfile::EfficientHighOrder {
                        // The official STGformer implementation L2-normalizes
                        // Q and K, then applies the efficient-attention
                        // rearrangement: Q(K^T V) + N V over Q(sum K) + N.
                        // This is not a generic kernel/feature-map attention.
                        temporal[node][start..end].copy_from_slice(&tape_stgformer_fast_attention(
                            &tape,
                            &query[start..end],
                            &tape_stgformer_attention_summary(
                                &tape,
                                &keys
                                    .iter()
                                    .map(|key| key[start..end].to_vec())
                                    .collect::<Vec<_>>(),
                                &values
                                    .iter()
                                    .map(|value| value[start..end].to_vec())
                                    .collect::<Vec<_>>(),
                            ),
                            &values[times - 1][start..end],
                        ));
                    } else {
                        let scores = keys
                            .iter()
                            .map(|key| tape_dot(&tape, &query[start..end], &key[start..end]))
                            .collect::<Vec<_>>();
                        let weights = tape_softmax(&tape, &scores);
                        let head_values = values
                            .iter()
                            .map(|value| value[start..end].to_vec())
                            .collect::<Vec<_>>();
                        temporal[node][start..end].copy_from_slice(&tape_weighted_sum(
                            &tape,
                            &weights,
                            &head_values,
                            end - start,
                        ));
                    }
                }
            }
        }

        // LongShortFusion owns its sparse forward, backward, and adaptive
        // graph diffusions below. It does not use Graphormer shortest-path
        // attention, so allocating an all-pairs distance matrix would be
        // quadratic and unnecessary for global graphs.
        let distances = (*profile != GraphTransformerProfile::LongShortFusion)
            .then(|| graph_distances(adjacency, nodes));
        let mut spatial = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::LongShortFusion {
            // LSTTN's periodic branch owns its forward, backward, and
            // adaptive graph diffusions, so no generic spatial-attention
            // state is needed before long/short fusion.
        } else if *profile == GraphTransformerProfile::EfficientHighOrder {
            // STGformer uses one QKV projection for its spatial and temporal
            // paths.  The efficient K^T V statistic is shared by all query
            // nodes within each head.
            let keys = (0..nodes)
                .map(|node| {
                    tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_k,
                        &temporal[node],
                        hidden,
                        hidden,
                    )
                })
                .collect::<Vec<_>>();
            let values = (0..nodes)
                .map(|node| {
                    tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.temporal_v,
                        &temporal[node],
                        hidden,
                        hidden,
                    )
                })
                .collect::<Vec<_>>();
            let summaries = (0..self.attention_heads)
                .map(|head| {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    tape_stgformer_attention_summary(
                        &tape,
                        &keys
                            .iter()
                            .map(|key| key[start..end].to_vec())
                            .collect::<Vec<_>>(),
                        &values
                            .iter()
                            .map(|value| value[start..end].to_vec())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.temporal_q,
                    &temporal[node],
                    hidden,
                    hidden,
                );
                for head in 0..self.attention_heads {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    spatial[node][start..end].copy_from_slice(&tape_stgformer_fast_attention(
                        &tape,
                        &query[start..end],
                        &summaries[head],
                        &values[node][start..end],
                    ));
                }
            }
        } else {
            let distances = distances
                .as_ref()
                .expect("non-LSTTN spatial attention requires graph distances");
            for node in 0..nodes {
                let query = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.spatial_q,
                    &temporal[node],
                    hidden,
                    hidden,
                );
                let mut keys = Vec::with_capacity(nodes);
                let mut values = Vec::with_capacity(nodes);
                for other in 0..nodes {
                    let key = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.spatial_k,
                        &temporal[other],
                        hidden,
                        hidden,
                    );
                    let value = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.spatial_v,
                        &temporal[other],
                        hidden,
                        hidden,
                    );
                    let distance = distances[node][other].min(nodes) as f64;
                    // Graphormer-style learnable scalar embedding for each
                    // shortest-path distance, including the disconnected cap.
                    let bias = parameter(&tape, layout.shortest_path_bias + distance as usize);
                    keys.push((key, bias));
                    values.push(value);
                }
                for head in 0..self.attention_heads {
                    let start = head * hidden / self.attention_heads;
                    let end = (head + 1) * hidden / self.attention_heads;
                    let scores = keys
                        .iter()
                        .map(|(key, bias)| {
                            tape.add(tape_dot(&tape, &query[start..end], &key[start..end]), *bias)
                        })
                        .collect::<Vec<_>>();
                    let weights = tape_softmax(&tape, &scores);
                    let head_values = values
                        .iter()
                        .map(|value| value[start..end].to_vec())
                        .collect::<Vec<_>>();
                    spatial[node][start..end].copy_from_slice(&tape_weighted_sum(
                        &tape,
                        &weights,
                        &head_values,
                        end - start,
                    ));
                }
            }
        }

        if *profile == GraphTransformerProfile::HeterogeneousMoE {
            let distances = distances
                .as_ref()
                .expect("heterogeneous graph attention requires graph distances");
            // STGormer stacks three causal temporal-attention and spatial-
            // attention blocks.  Each stage owns an independent Q/K/V set;
            // spatial output becomes the representation consumed by the next
            // temporal block, preserving both axes at every depth.
            let projection = hidden * (hidden + 1);
            let mut states = embedding.clone();
            let mut final_temporal = vec![vec![tape.constant(0.0); hidden]; nodes];
            let mut final_spatial = vec![vec![tape.constant(0.0); hidden]; nodes];
            for block in 0..3 {
                let temporal_q = layout.temporal_q + block * projection;
                let temporal_k = layout.temporal_k + block * projection;
                let temporal_v = layout.temporal_v + block * projection;
                let spatial_q = layout.spatial_q + block * projection;
                let spatial_k = layout.spatial_k + block * projection;
                let spatial_v = layout.spatial_v + block * projection;
                let mut block_temporal = vec![vec![vec![tape.constant(0.0); hidden]; nodes]; times];
                for time in 0..times {
                    for node in 0..nodes {
                        let query = tape_linear(
                            &tape,
                            &parameter_nodes,
                            temporal_q,
                            &states[time][node],
                            hidden,
                            hidden,
                        );
                        let keys = (0..=time)
                            .map(|past| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    temporal_k,
                                    &states[past][node],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        let values = (0..=time)
                            .map(|past| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    temporal_v,
                                    &states[past][node],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        for head in 0..self.attention_heads {
                            let start = head * hidden / self.attention_heads;
                            let end = (head + 1) * hidden / self.attention_heads;
                            let scores = keys
                                .iter()
                                .map(|key| tape_dot(&tape, &query[start..end], &key[start..end]))
                                .collect::<Vec<_>>();
                            let weights = tape_softmax(&tape, &scores);
                            let values = values
                                .iter()
                                .map(|value| value[start..end].to_vec())
                                .collect::<Vec<_>>();
                            block_temporal[time][node][start..end].copy_from_slice(
                                &tape_weighted_sum(&tape, &weights, &values, end - start),
                            );
                        }
                    }
                }
                let mut block_spatial = vec![vec![vec![tape.constant(0.0); hidden]; nodes]; times];
                for time in 0..times {
                    for node in 0..nodes {
                        let query = tape_linear(
                            &tape,
                            &parameter_nodes,
                            spatial_q,
                            &block_temporal[time][node],
                            hidden,
                            hidden,
                        );
                        let keys = (0..nodes)
                            .map(|other| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    spatial_k,
                                    &block_temporal[time][other],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        let values = (0..nodes)
                            .map(|other| {
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    spatial_v,
                                    &block_temporal[time][other],
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        for head in 0..self.attention_heads {
                            let start = head * hidden / self.attention_heads;
                            let end = (head + 1) * hidden / self.attention_heads;
                            let scores = (0..nodes)
                                .map(|other| {
                                    let distance = distances[node][other].min(nodes);
                                    tape.add(
                                        tape_dot(
                                            &tape,
                                            &query[start..end],
                                            &keys[other][start..end],
                                        ),
                                        parameter(&tape, layout.shortest_path_bias + distance),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let weights = tape_softmax(&tape, &scores);
                            let values = values
                                .iter()
                                .map(|value| value[start..end].to_vec())
                                .collect::<Vec<_>>();
                            block_spatial[time][node][start..end].copy_from_slice(
                                &tape_weighted_sum(&tape, &weights, &values, end - start),
                            );
                        }
                    }
                }
                states = (0..times)
                    .map(|time| {
                        (0..nodes)
                            .map(|node| {
                                tape_add_vectors(
                                    &tape,
                                    &block_temporal[time][node],
                                    &block_spatial[time][node],
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                final_temporal = block_temporal[times - 1].clone();
                final_spatial = block_spatial[times - 1].clone();
            }
            temporal = final_temporal;
            spatial = final_spatial;
        }

        // The gated graph-temporal profile keeps an explicit normalized graph
        // convolution alongside spatial attention.  The convolution starts
        // from causal temporal states and passes through a learned projection
        // before the GRU-style gates consume it.
        let mut graph_convolution = vec![vec![tape.constant(0.0); hidden]; nodes];
        if *profile == GraphTransformerProfile::GatedGraphTemporal {
            for node in 0..nodes {
                let mut aggregated = vec![tape.constant(0.0); hidden];
                for edge in adjacency.indptr[node]..adjacency.indptr[node + 1] {
                    let neighbor = adjacency.indices[edge];
                    for channel in 0..hidden {
                        aggregated[channel] = tape.add(
                            aggregated[channel],
                            tape.mul(adjacency_weights[edge], temporal[neighbor][channel]),
                        );
                    }
                }
                graph_convolution[node] = tape_linear(
                    &tape,
                    &parameter_nodes,
                    layout.spatial_v,
                    &aggregated,
                    hidden,
                    hidden,
                );
            }
        }

        // STGformer retains every graph propagation order, applies the same
        // efficient QKV attention block to each order, and recursively
        // interacts the resulting attention with a learned pointwise map of
        // the prior order.  `temporal` is fused into each order's input so the
        // efficient spatial operation receives a spatiotemporal state.
        let mut stgformer_representation = vec![vec![0usize; hidden]; nodes];
        if *profile == GraphTransformerProfile::EfficientHighOrder {
            let mut propagated = embedding[times - 1].clone();
            let mut previous = propagated.clone();
            stgformer_representation = propagated.clone();
            for order in 0..self.graph_order {
                let order_input = (0..nodes)
                    .map(|node| tape_add_vectors(&tape, &propagated[node], &temporal[node]))
                    .collect::<Vec<_>>();
                let queries = order_input
                    .iter()
                    .map(|input| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.temporal_q,
                            input,
                            hidden,
                            hidden,
                        )
                    })
                    .collect::<Vec<_>>();
                let keys = order_input
                    .iter()
                    .map(|input| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.temporal_k,
                            input,
                            hidden,
                            hidden,
                        )
                    })
                    .collect::<Vec<_>>();
                let values = order_input
                    .iter()
                    .map(|input| {
                        tape_linear(
                            &tape,
                            &parameter_nodes,
                            layout.temporal_v,
                            input,
                            hidden,
                            hidden,
                        )
                    })
                    .collect::<Vec<_>>();
                let summaries = (0..self.attention_heads)
                    .map(|head| {
                        let start = head * hidden / self.attention_heads;
                        let end = (head + 1) * hidden / self.attention_heads;
                        tape_stgformer_attention_summary(
                            &tape,
                            &keys
                                .iter()
                                .map(|key| key[start..end].to_vec())
                                .collect::<Vec<_>>(),
                            &values
                                .iter()
                                .map(|value| value[start..end].to_vec())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut attention = vec![vec![tape.constant(0.0); hidden]; nodes];
                for node in 0..nodes {
                    for head in 0..self.attention_heads {
                        let start = head * hidden / self.attention_heads;
                        let end = (head + 1) * hidden / self.attention_heads;
                        attention[node][start..end].copy_from_slice(
                            &tape_stgformer_fast_attention(
                                &tape,
                                &queries[node][start..end],
                                &summaries[head],
                                &values[node][start..end],
                            ),
                        );
                    }
                }
                let scale = tape.constant(match order {
                    0 => 1.0,
                    1 => 0.01,
                    _ => 0.001,
                });
                for node in 0..nodes {
                    let pointwise = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.stgformer_pointwise + order * hidden * (hidden + 1),
                        &previous[node],
                        hidden,
                        hidden,
                    );
                    for channel in 0..hidden {
                        stgformer_representation[node][channel] = tape.add(
                            stgformer_representation[node][channel],
                            tape.mul(
                                scale,
                                tape.mul(attention[node][channel], pointwise[channel]),
                            ),
                        );
                    }
                }
                previous = attention;
                let mut next = vec![vec![tape.constant(0.0); hidden]; nodes];
                for target in 0..nodes {
                    for edge in adjacency.indptr[target]..adjacency.indptr[target + 1] {
                        let source = adjacency.indices[edge];
                        for channel in 0..hidden {
                            next[target][channel] = tape.add(
                                next[target][channel],
                                tape.mul(adjacency_weights[edge], propagated[source][channel]),
                            );
                        }
                    }
                }
                propagated = next;
            }
        }

        // Periodic patch embeddings are shared by every destination node in
        // LSTTN's graph diffusion.  Building them inside the node loop would
        // duplicate the same tape subgraph `nodes` times, which turns one
        // periodic feature extraction into quadratic memory use on METR-LA.
        let lsttn_period_embeddings = if *profile == GraphTransformerProfile::LongShortFusion {
            let patch_width = if long_context_is_pooled {
                1
            } else {
                native_patch_width
            };
            let patch_count = lsttn_frozen_patches
                .map_or_else(|| embedding.chunks(patch_width).len(), <[_]>::len);
            [effective_short_periodicity, effective_periodicity]
                .into_iter()
                .filter_map(|period| {
                    let period_patches = (period / patch_width).max(1);
                    (patch_count > period_patches).then(|| {
                        let patch_index = patch_count - period_patches - 1;
                        if let Some(cached) = lsttn_frozen_patches {
                            cached[patch_index]
                                .iter()
                                .map(|node_values| {
                                    node_values
                                        .iter()
                                        .map(|value| tape.constant(*value as f64))
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        } else {
                            let patch_start = patch_index * patch_width;
                            let patch =
                                &embedding[patch_start..(patch_start + patch_width).min(times)];
                            (0..nodes)
                                .map(|period_node| {
                                    (0..hidden)
                                        .map(|channel| {
                                            let sum = patch
                                                .iter()
                                                .fold(tape.constant(0.0), |sum, row| {
                                                    tape.add(sum, row[period_node][channel])
                                                });
                                            tape.div(sum, tape.constant(patch.len() as f64))
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let lsttn_periodic_features = if *profile == GraphTransformerProfile::LongShortFusion {
            lsttn_period_embeddings
                .iter()
                .enumerate()
                .map(|(period, values)| {
                    let (adaptive_source, adaptive_target) = if period == 0 {
                        (layout.lsttn_adaptive_source, layout.lsttn_adaptive_target)
                    } else {
                        (
                            layout.lsttn_weekly_adaptive_source,
                            layout.lsttn_weekly_adaptive_target,
                        )
                    };
                    // Adaptive adjacency is learned on the directed CSR
                    // support, not over every possible node pair. The old
                    // dense [nodes, nodes] softmax made a 36k-node LSTTN
                    // infeasible before either CPU or CUDA could reach the
                    // actual diffusion kernels. Each output row retains a
                    // row-normalized learned distribution over its existing
                    // directed neighbors plus one sparse self candidate.
                    let adaptive_weights = (0..nodes)
                        .flat_map(|target| {
                            let range = adaptive_adjacency.indptr[target]
                                ..adaptive_adjacency.indptr[target + 1];
                            let logits = range
                                .map(|edge| {
                                    let source = adaptive_adjacency.indices[edge];
                                    let score = (0..10).fold(tape.constant(0.0), |sum, latent| {
                                        tape.add(
                                            sum,
                                            tape.mul(
                                                parameter(
                                                    &tape,
                                                    adaptive_source + target * 10 + latent,
                                                ),
                                                parameter(
                                                    &tape,
                                                    adaptive_target + source * 10 + latent,
                                                ),
                                            ),
                                        )
                                    });
                                    tape.max(score, tape.constant(0.0))
                                })
                                .collect::<Vec<_>>();
                            if logits.is_empty() {
                                Vec::new().into_iter()
                            } else {
                                tape_softmax(&tape, &logits).into_iter()
                            }
                        })
                        .collect::<Vec<_>>();
                    let forward_one =
                        tape_csr_diffuse(&tape, adjacency, &adjacency_weights, values, hidden);
                    let forward_two = tape_csr_diffuse(
                        &tape,
                        adjacency,
                        &adjacency_weights,
                        &forward_one,
                        hidden,
                    );
                    let backward_one = tape_csr_diffuse(
                        &tape,
                        &reverse_adjacency,
                        &reverse_adjacency_weights,
                        values,
                        hidden,
                    );
                    let backward_two = tape_csr_diffuse(
                        &tape,
                        &reverse_adjacency,
                        &reverse_adjacency_weights,
                        &backward_one,
                        hidden,
                    );
                    let adaptive_one = tape_csr_diffuse(
                        &tape,
                        &adaptive_adjacency,
                        &adaptive_weights,
                        values,
                        hidden,
                    );
                    let adaptive_two = tape_csr_diffuse(
                        &tape,
                        &adaptive_adjacency,
                        &adaptive_weights,
                        &adaptive_one,
                        hidden,
                    );
                    (0..nodes)
                        .map(|node| {
                            let mut concatenated = values[node].clone();
                            concatenated.extend(&forward_one[node]);
                            concatenated.extend(&forward_two[node]);
                            concatenated.extend(&backward_one[node]);
                            concatenated.extend(&backward_two[node]);
                            concatenated.extend(&adaptive_one[node]);
                            concatenated.extend(&adaptive_two[node]);
                            tape_linear(
                                &tape,
                                &parameter_nodes,
                                layout.lsttn_periodic_projection
                                    + period * (7 * hidden * hidden + hidden),
                                &concatenated,
                                7 * hidden,
                                hidden,
                            )
                            .into_iter()
                            .enumerate()
                            .map(|(channel, value)| {
                                tape_deterministic_dropout_rate(
                                    &tape,
                                    value,
                                    self.steps ^ ((period as u64) << 40),
                                    node * hidden + channel,
                                    training,
                                    0.3,
                                )
                            })
                            .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut graphon_expert_states =
            vec![vec![vec![tape.constant(0.0); hidden]; self.experts]; nodes];
        let mut lsttn_short_sequence = None;
        let mut representation = vec![vec![0usize; hidden]; nodes];
        for node in 0..nodes {
            representation[node] = match profile {
                GraphTransformerProfile::HeterogeneousMoE => {
                    tape_add_vectors(&tape, &temporal[node], &spatial[node])
                }
                GraphTransformerProfile::EfficientHighOrder => {
                    stgformer_representation[node].clone()
                }
                GraphTransformerProfile::LongShortFusion => {
                    // Four stacked three-tap dilated convolutions (dilations
                    // 1, 2, 4, and 8) provide LSTTN's exponentially growing
                    // long-term receptive field.  Each tap mixes hidden
                    // channels through its own learned convolution kernel.
                    // The convolution consumes the masked-subseries encoder's
                    // patch representations, not raw timestamps; this is the
                    // level at which LSTTN extracts long trend and periodic
                    // features.
                    let patch_width = if long_context_is_pooled {
                        1
                    } else {
                        native_patch_width
                    };
                    let position_count =
                        (layout.pretrain_decoder - layout.pretrain_position) / hidden;
                    let subseries = if let Some(cached) = lsttn_frozen_patches {
                        cached
                            .iter()
                            .map(|patch| {
                                patch[node]
                                    .iter()
                                    .map(|value| tape.constant(*value as f64))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    } else {
                        embedding
                            .chunks(patch_width)
                            .enumerate()
                            .map(|(patch_index, patch)| {
                                (0..hidden)
                                    .map(|channel| {
                                        let sum =
                                            patch.iter().fold(tape.constant(0.0), |sum, row| {
                                                tape.add(sum, row[node][channel])
                                            });
                                        let pooled =
                                            tape.div(sum, tape.constant(patch.len() as f64));
                                        tape.tanh(tape.add(
                                            pooled,
                                            parameter(
                                                &tape,
                                                layout.pretrain_position
                                                    + (patch_index % position_count) * hidden
                                                    + channel,
                                            ),
                                        ))
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    };
                    let mut long_sequence = subseries.clone();
                    for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate() {
                        let layer_offset = layout.lsttn_dilated_convolution
                            + layer * (3 * hidden * hidden + hidden);
                        let convolution_times = long_sequence.len().div_ceil(2);
                        let mut convolved =
                            vec![vec![tape.constant(0.0); hidden]; convolution_times];
                        for output_time in 0..convolution_times {
                            for output_channel in 0..hidden {
                                let mut value = parameter(
                                    &tape,
                                    layer_offset + 3 * hidden * hidden + output_channel,
                                );
                                for tap in 0..3 {
                                    let centered = output_time * 2;
                                    let source_time = match tap {
                                        0 => centered.checked_sub(dilation),
                                        1 => Some(centered),
                                        _ => centered
                                            .checked_add(dilation)
                                            .filter(|time| *time < long_sequence.len()),
                                    };
                                    if let Some(source_time) = source_time {
                                        for input_channel in 0..hidden {
                                            value = tape.add(
                                                value,
                                                tape.mul(
                                                    parameter(
                                                        &tape,
                                                        layer_offset
                                                            + tap * hidden * hidden
                                                            + input_channel * hidden
                                                            + output_channel,
                                                    ),
                                                    long_sequence[source_time][input_channel],
                                                ),
                                            );
                                        }
                                    }
                                }
                                convolved[output_time][output_channel] = tape_gelu(&tape, value);
                            }
                        }
                        let pooled_times = convolved.len().div_ceil(2);
                        long_sequence = (0..pooled_times)
                            .map(|output_time| {
                                (0..hidden)
                                    .map(|channel| {
                                        let center = output_time * 2;
                                        [center.checked_sub(1), Some(center), Some(center + 1)]
                                            .into_iter()
                                            .flatten()
                                            .filter(|time| *time < convolved.len())
                                            .map(|time| convolved[time][channel])
                                            .reduce(|left, right| tape.max(left, right))
                                            .expect("nonempty LSTTN max-pool window")
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                    }
                    let long = long_sequence[long_sequence.len() - 1].clone();
                    let periodic_components = lsttn_periodic_features
                        .iter()
                        .map(|period| period[node].clone())
                        .collect::<Vec<_>>();
                    // The short branch is a Graph WaveNet-style stack rather
                    // than a reuse of the generic transformer attention.  A
                    // causal gated temporal convolution learns local traffic
                    // changes, then an input-conditioned adaptive adjacency
                    // propagates those changes across nodes.  Keeping this
                    // separate from the long dilation stack makes the
                    // long/short fusion an actual architectural distinction.
                    if lsttn_short_sequence.is_none() {
                        let short_start = times.saturating_sub(self.recent_window);
                        let start_projection = layout.lsttn_short_wave;
                        // The reference Graph WaveNet consumes the first two
                        // traffic-frame channels: the normalized signal and
                        // normalized time-of-day.  Left padding brings a
                        // 12-step short history to its 13-step receptive field.
                        let receptive_field = 13usize;
                        let raw_short_len = times - short_start;
                        let padded_len = (raw_short_len + 1).max(receptive_field);
                        let left_padding = padded_len - raw_short_len;
                        let mut short_sequence =
                            vec![vec![vec![tape.constant(0.0); hidden]; nodes]; padded_len];
                        for (local_time, absolute_time) in (short_start..times).enumerate() {
                            for current_node in 0..nodes {
                                let time_of_day = lsttn_time_features
                                    .map(|features| features[absolute_time][current_node])
                                    .unwrap_or_else(|| {
                                        ((phase_offset + absolute_time)
                                            % effective_periodicity.max(1))
                                            as f64
                                            / effective_periodicity.max(1) as f64
                                    });
                                let inputs = [
                                    observed_values[absolute_time][current_node],
                                    tape.constant(time_of_day),
                                ];
                                short_sequence[left_padding + local_time][current_node] =
                                    tape_linear(
                                        &tape,
                                        &parameter_nodes,
                                        start_projection,
                                        &inputs,
                                        2,
                                        hidden,
                                    );
                            }
                        }
                        let mut skip = vec![
                            vec![vec![tape.constant(0.0); hidden]; nodes];
                            short_sequence.len()
                        ];
                        let short_adaptive = (0..nodes)
                            .flat_map(|target| {
                                let logits = (adaptive_adjacency.indptr[target]
                                    ..adaptive_adjacency.indptr[target + 1])
                                    .map(|edge| {
                                        let source = adaptive_adjacency.indices[edge];
                                        let score =
                                            (0..10).fold(tape.constant(0.0), |sum, latent| {
                                                tape.add(
                                                    sum,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            layout.lsttn_short_adaptive_source
                                                                + target * 10
                                                                + latent,
                                                        ),
                                                        parameter(
                                                            &tape,
                                                            layout.lsttn_short_adaptive_target
                                                                + source * 10
                                                                + latent,
                                                        ),
                                                    ),
                                                )
                                            });
                                        tape.max(score, tape.constant(0.0))
                                    })
                                    .collect::<Vec<_>>();
                                if logits.is_empty() {
                                    Vec::new().into_iter()
                                } else {
                                    tape_softmax(&tape, &logits).into_iter()
                                }
                            })
                            .collect::<Vec<_>>();
                        let layer_width = 12 * hidden * hidden + 6 * hidden;
                        let layers_start = start_projection + 2 * hidden + hidden;
                        for (layer, dilation) in
                            [1usize, 2, 1, 2, 1, 2, 1, 2].into_iter().enumerate()
                        {
                            let layer_offset = layers_start + layer * layer_width;
                            let filter_offset = layer_offset;
                            let gate_offset = filter_offset + 2 * hidden * hidden;
                            let filter_bias = gate_offset + 2 * hidden * hidden;
                            let gate_bias = filter_bias + hidden;
                            let graph_projection = gate_bias + hidden;
                            let skip_projection = graph_projection + 7 * hidden * hidden + hidden;
                            let norm = skip_projection + hidden * hidden + hidden;
                            let output_times = short_sequence.len() - dilation;
                            let mut gated =
                                vec![vec![vec![tape.constant(0.0); hidden]; nodes]; output_times];
                            let skip_offset = skip.len() - output_times;
                            let mut cropped_skip =
                                vec![vec![vec![tape.constant(0.0); hidden]; nodes]; output_times];
                            for time in 0..output_times {
                                for current_node in 0..nodes {
                                    for output_channel in 0..hidden {
                                        let mut filter =
                                            parameter(&tape, filter_bias + output_channel);
                                        let mut gate = parameter(&tape, gate_bias + output_channel);
                                        for tap in 0..2 {
                                            let source_time = time + tap * dilation;
                                            for input_channel in 0..hidden {
                                                let source = short_sequence[source_time]
                                                    [current_node][input_channel];
                                                filter = tape.add(
                                                    filter,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            filter_offset
                                                                + tap * hidden * hidden
                                                                + input_channel * hidden
                                                                + output_channel,
                                                        ),
                                                        source,
                                                    ),
                                                );
                                                gate = tape.add(
                                                    gate,
                                                    tape.mul(
                                                        parameter(
                                                            &tape,
                                                            gate_offset
                                                                + tap * hidden * hidden
                                                                + input_channel * hidden
                                                                + output_channel,
                                                        ),
                                                        source,
                                                    ),
                                                );
                                            }
                                        }
                                        gated[time][current_node][output_channel] =
                                            tape.mul(tape.tanh(filter), tape.sigmoid(gate));
                                    }
                                    let projected_skip = tape_linear(
                                        &tape,
                                        &parameter_nodes,
                                        skip_projection,
                                        &gated[time][current_node],
                                        hidden,
                                        hidden,
                                    );
                                    for channel in 0..hidden {
                                        cropped_skip[time][current_node][channel] = tape.add(
                                            skip[skip_offset + time][current_node][channel],
                                            projected_skip[channel],
                                        );
                                    }
                                }
                            }
                            skip = cropped_skip;
                            let mut next = Vec::with_capacity(output_times);
                            for time in 0..output_times {
                                let forward_one = tape_csr_diffuse(
                                    &tape,
                                    adjacency,
                                    &adjacency_weights,
                                    &gated[time],
                                    hidden,
                                );
                                let forward_two = tape_csr_diffuse(
                                    &tape,
                                    adjacency,
                                    &adjacency_weights,
                                    &forward_one,
                                    hidden,
                                );
                                let backward_one = tape_csr_diffuse(
                                    &tape,
                                    &reverse_adjacency,
                                    &reverse_adjacency_weights,
                                    &gated[time],
                                    hidden,
                                );
                                let backward_two = tape_csr_diffuse(
                                    &tape,
                                    &reverse_adjacency,
                                    &reverse_adjacency_weights,
                                    &backward_one,
                                    hidden,
                                );
                                let adaptive_one = tape_csr_diffuse(
                                    &tape,
                                    &adaptive_adjacency,
                                    &short_adaptive,
                                    &gated[time],
                                    hidden,
                                );
                                let adaptive_two = tape_csr_diffuse(
                                    &tape,
                                    &adaptive_adjacency,
                                    &short_adaptive,
                                    &adaptive_one,
                                    hidden,
                                );
                                next.push(
                                    (0..nodes)
                                        .map(|current_node| {
                                            let mut features = gated[time][current_node].clone();
                                            features.extend(&forward_one[current_node]);
                                            features.extend(&forward_two[current_node]);
                                            features.extend(&backward_one[current_node]);
                                            features.extend(&backward_two[current_node]);
                                            features.extend(&adaptive_one[current_node]);
                                            features.extend(&adaptive_two[current_node]);
                                            let graph = tape_linear(
                                                &tape,
                                                &parameter_nodes,
                                                graph_projection,
                                                &features,
                                                7 * hidden,
                                                hidden,
                                            );
                                            graph
                                                .iter()
                                                .zip(&short_sequence[time + dilation][current_node])
                                                .enumerate()
                                                .map(|(channel, (value, residual))| {
                                                    let value = tape_deterministic_dropout_rate(
                                                        &tape,
                                                        *value,
                                                        self.steps ^ ((layer as u64) << 48),
                                                        (time * nodes + current_node) * hidden
                                                            + channel,
                                                        training,
                                                        0.3,
                                                    );
                                                    tape.add(value, *residual)
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .collect::<Vec<_>>(),
                                );
                            }
                            for channel in 0..hidden {
                                let count = (next.len() * nodes) as f64;
                                let mean = next.iter().flatten().fold(
                                    tape.constant(0.0),
                                    |sum, values| {
                                        tape.add(
                                            sum,
                                            tape.mul(values[channel], tape.constant(1.0 / count)),
                                        )
                                    },
                                );
                                let variance = next.iter().flatten().fold(
                                    tape.constant(0.0),
                                    |sum, values| {
                                        let centered = tape.add(
                                            values[channel],
                                            tape.mul(mean, tape.constant(-1.0)),
                                        );
                                        tape.add(
                                            sum,
                                            tape.mul(
                                                tape.mul(centered, centered),
                                                tape.constant(1.0 / count),
                                            ),
                                        )
                                    },
                                );
                                let denominator =
                                    tape.sqrt(tape.add(variance, tape.constant(1e-5)));
                                for time in &mut next {
                                    for values in time {
                                        values[channel] = tape.add(
                                            tape.mul(
                                                tape.div(
                                                    tape.add(
                                                        values[channel],
                                                        tape.mul(mean, tape.constant(-1.0)),
                                                    ),
                                                    denominator,
                                                ),
                                                parameter(&tape, norm + channel),
                                            ),
                                            parameter(&tape, norm + hidden + channel),
                                        );
                                    }
                                }
                            }
                            short_sequence = next;
                        }
                        let end_one = layers_start + 8 * layer_width;
                        let end_two = end_one + hidden * hidden + hidden;
                        let final_short = (0..nodes)
                            .map(|current_node| {
                                let activated = skip[skip.len() - 1][current_node]
                                    .iter()
                                    .map(|value| tape.max(*value, tape.constant(0.0)))
                                    .collect::<Vec<_>>();
                                let hidden_values = tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    end_one,
                                    &activated,
                                    hidden,
                                    hidden,
                                )
                                .into_iter()
                                .map(|value| tape.max(value, tape.constant(0.0)))
                                .collect::<Vec<_>>();
                                tape_linear(
                                    &tape,
                                    &parameter_nodes,
                                    end_two,
                                    &hidden_values,
                                    hidden,
                                    hidden,
                                )
                            })
                            .collect::<Vec<_>>();
                        lsttn_short_sequence = Some(vec![final_short]);
                    }
                    let short_sequence = lsttn_short_sequence
                        .as_ref()
                        .expect("LSTTN short branch is initialized");
                    // The paper concatenates the long-trend, weekly, and
                    // daily graph features, sends them through a two-layer
                    // trend-seasonality MLP, then concatenates that result
                    // with the Graph WaveNet short-term state for a final
                    // MLP.  A learned gate is not equivalent to this path.
                    let zero = tape.constant(0.0);
                    let daily = periodic_components
                        .first()
                        .cloned()
                        .unwrap_or_else(|| vec![zero; hidden]);
                    let weekly = periodic_components
                        .get(1)
                        .cloned()
                        .unwrap_or_else(|| vec![zero; hidden]);
                    let mut trend_seasonality_input = long;
                    trend_seasonality_input.extend(daily);
                    trend_seasonality_input.extend(weekly);
                    let first_trend_seasonality = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.lsttn_fusion,
                        &trend_seasonality_input,
                        3 * hidden,
                        hidden,
                    )
                    .into_iter()
                    .map(|value| tape.max(value, zero))
                    .collect::<Vec<_>>();
                    let trend_seasonality = tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.lsttn_fusion + (3 * hidden + 1) * hidden,
                        &first_trend_seasonality,
                        hidden,
                        hidden,
                    );
                    let mut final_input = short_sequence[short_sequence.len() - 1][node].clone();
                    final_input.extend(trend_seasonality);
                    tape_linear(
                        &tape,
                        &parameter_nodes,
                        layout.lsttn_fusion + (4 * hidden * hidden + 2 * hidden),
                        &final_input,
                        2 * hidden,
                        hidden,
                    )
                    .into_iter()
                    .map(|value| tape.max(value, zero))
                    .collect()
                }
                GraphTransformerProfile::GatedGraphTemporal => {
                    let mut gated = vec![0usize; hidden];
                    for channel in 0..hidden {
                        let reset = tape.sigmoid(tape.add(
                            temporal[node][channel],
                            tape.mul(
                                parameter(&tape, layout.recurrence + 1 + channel),
                                graph_convolution[node][channel],
                            ),
                        ));
                        let update = tape.sigmoid(tape.add(
                            graph_convolution[node][channel],
                            parameter(&tape, layout.recurrence + 1 + hidden + channel),
                        ));
                        let candidate = tape.tanh(tape.add(
                            temporal[node][channel],
                            tape.mul(reset, graph_convolution[node][channel]),
                        ));
                        gated[channel] = tape.add(
                            tape.mul(update, temporal[node][channel]),
                            tape.mul(
                                tape.add(tape.constant(1.0), tape.mul(tape.constant(-1.0), update)),
                                candidate,
                            ),
                        );
                    }
                    gated
                }
                GraphTransformerProfile::SpatialShiftGraphonMoE => {
                    for expert in 0..self.experts {
                        let source_embedding =
                            parameter(&tape, layout.graphon_nodes + expert * nodes + node);
                        for other in 0..nodes {
                            let target_embedding =
                                parameter(&tape, layout.graphon_nodes + expert * nodes + other);
                            for channel in 0..hidden {
                                let temporal_embedding = parameter(
                                    &tape,
                                    layout.graphon_time + expert * hidden + channel,
                                );
                                let edge_logit = tape.add(
                                    tape.add(
                                        tape.mul(source_embedding, target_embedding),
                                        temporal_embedding,
                                    ),
                                    temporal[node][channel],
                                );
                                // A binary Gumbel-Softmax relaxation samples
                                // a graph from each expert graphon while
                                // retaining a pathwise gradient.  `steps` is
                                // advanced by every optimizer update, so
                                // training sees fresh samples; a fitted model
                                // uses its final serialized step and therefore
                                // makes deterministic predictions.
                                let sample_noise = tape.constant(graphon_gumbel_logistic_noise(
                                    self.steps, expert, node, other, channel,
                                ));
                                let edge_probability =
                                    tape.sigmoid(tape.mul(
                                        tape.constant(2.0),
                                        tape.add(edge_logit, sample_noise),
                                    ));
                                graphon_expert_states[node][expert][channel] = tape.add(
                                    graphon_expert_states[node][expert][channel],
                                    tape.mul(edge_probability, spatial[other][channel]),
                                );
                            }
                        }
                    }
                    // Each expert remains separate through its router and
                    // forecast head; `representation` is only a placeholder
                    // for the common branch below.
                    temporal[node].clone()
                }
            };
        }
        let mut outputs = vec![vec![0usize; self.horizons]; nodes];
        let mut router_weights = Vec::with_capacity(nodes);
        for node in 0..nodes {
            let routed = matches!(
                profile,
                GraphTransformerProfile::HeterogeneousMoE
                    | GraphTransformerProfile::SpatialShiftGraphonMoE
            );
            if routed {
                if *profile == GraphTransformerProfile::HeterogeneousMoE {
                    // STGormer routes the temporal and spatial transformer
                    // outputs independently.  They must not share either a
                    // gate or expert FNN: those are the mechanisms that let
                    // a road/time regime select different specialists along
                    // the two axes.
                    let temporal_logits = (0..self.experts)
                        .map(|expert| {
                            let mut logit =
                                parameter(&tape, layout.router + expert * (hidden + 1) + hidden);
                            for channel in 0..hidden {
                                logit = tape.add(
                                    logit,
                                    tape.mul(
                                        parameter(
                                            &tape,
                                            layout.router + expert * (hidden + 1) + channel,
                                        ),
                                        temporal[node][channel],
                                    ),
                                );
                            }
                            logit
                        })
                        .collect::<Vec<_>>();
                    let spatial_logits = (0..self.experts)
                        .map(|expert| {
                            let mut logit = parameter(
                                &tape,
                                layout.spatial_router + expert * (hidden + 1) + hidden,
                            );
                            for channel in 0..hidden {
                                logit = tape.add(
                                    logit,
                                    tape.mul(
                                        parameter(
                                            &tape,
                                            layout.spatial_router + expert * (hidden + 1) + channel,
                                        ),
                                        spatial[node][channel],
                                    ),
                                );
                            }
                            logit
                        })
                        .collect::<Vec<_>>();
                    let temporal_weights = tape_softmax(&tape, &temporal_logits);
                    let spatial_weights = tape_softmax(&tape, &spatial_logits);
                    router_weights.push(temporal_weights.clone());
                    router_weights.push(spatial_weights.clone());
                    for horizon in 0..self.horizons {
                        let mut temporal_result = tape.constant(0.0);
                        let mut spatial_result = tape.constant(0.0);
                        for expert in 0..self.experts {
                            let temporal_offset = layout.expert_heads
                                + expert * self.horizons * (hidden + 1)
                                + horizon * (hidden + 1);
                            let spatial_offset = layout.spatial_expert_heads
                                + expert * self.horizons * (hidden + 1)
                                + horizon * (hidden + 1);
                            let mut temporal_head = parameter(&tape, temporal_offset + hidden);
                            let mut spatial_head = parameter(&tape, spatial_offset + hidden);
                            for channel in 0..hidden {
                                temporal_head = tape.add(
                                    temporal_head,
                                    tape.mul(
                                        parameter(&tape, temporal_offset + channel),
                                        temporal[node][channel],
                                    ),
                                );
                                spatial_head = tape.add(
                                    spatial_head,
                                    tape.mul(
                                        parameter(&tape, spatial_offset + channel),
                                        spatial[node][channel],
                                    ),
                                );
                            }
                            temporal_result = tape.add(
                                temporal_result,
                                tape.mul(temporal_weights[expert], temporal_head),
                            );
                            spatial_result = tape.add(
                                spatial_result,
                                tape.mul(spatial_weights[expert], spatial_head),
                            );
                        }
                        outputs[node][horizon] = tape.mul(
                            tape.constant(0.5),
                            tape.add(temporal_result, spatial_result),
                        );
                    }
                } else {
                    let mut logits = Vec::with_capacity(self.experts);
                    for expert in 0..self.experts {
                        let expert_representation =
                            if *profile == GraphTransformerProfile::SpatialShiftGraphonMoE {
                                let graphon_state = if excluded_expert.is_some() {
                                    tape_detach_vectors(&tape, &graphon_expert_states[node][expert])
                                } else {
                                    graphon_expert_states[node][expert].clone()
                                };
                                tape_add_vectors(&tape, &temporal[node], &graphon_state)
                            } else {
                                representation[node].clone()
                            };
                        let mut logit =
                            parameter(&tape, layout.router + expert * (hidden + 1) + hidden);
                        for channel in 0..hidden {
                            logit = tape.add(
                                logit,
                                tape.mul(
                                    parameter(
                                        &tape,
                                        layout.router + expert * (hidden + 1) + channel,
                                    ),
                                    expert_representation[channel],
                                ),
                            );
                        }
                        if excluded_expert == Some(expert) {
                            logit = tape.add(logit, tape.constant(-30.0));
                        }
                        logits.push(logit);
                    }
                    let weights = tape_softmax(&tape, &logits);
                    for horizon in 0..self.horizons {
                        let mut result = tape.constant(0.0);
                        for expert in 0..self.experts {
                            let expert_representation = if *profile
                                == GraphTransformerProfile::SpatialShiftGraphonMoE
                            {
                                let graphon_state = if excluded_expert.is_some() {
                                    tape_detach_vectors(&tape, &graphon_expert_states[node][expert])
                                } else {
                                    graphon_expert_states[node][expert].clone()
                                };
                                tape_add_vectors(&tape, &temporal[node], &graphon_state)
                            } else {
                                representation[node].clone()
                            };
                            let offset = layout.expert_heads
                                + expert * self.horizons * (hidden + 1)
                                + horizon * (hidden + 1);
                            let mut head = parameter(&tape, offset + hidden);
                            for channel in 0..hidden {
                                head = tape.add(
                                    head,
                                    tape.mul(
                                        parameter(&tape, offset + channel),
                                        expert_representation[channel],
                                    ),
                                );
                            }
                            result = tape.add(result, tape.mul(weights[expert], head));
                        }
                        outputs[node][horizon] = result;
                    }
                }
            } else {
                for horizon in 0..self.horizons {
                    let offset = layout.output + horizon * (hidden + 1);
                    let mut head = parameter(&tape, offset + hidden);
                    for channel in 0..hidden {
                        head = tape.add(
                            head,
                            tape.mul(
                                parameter(&tape, offset + channel),
                                representation[node][channel],
                            ),
                        );
                    }
                    outputs[node][horizon] = head;
                }
            }
        }
        (tape, outputs, router_weights, representation)
    }
}

