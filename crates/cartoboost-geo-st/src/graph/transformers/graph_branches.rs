#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaLsttnTensorExecutor {
    /// Materializes the learned directed adaptive CSR weights on-device. The
    /// row softmax writes directly into the graph-weight slot consumed by the
    /// two-hop diffusion stages, keeping the O(edges) adaptive graph sparse.
    fn adaptive_weights(&mut self, source_offset: usize, target_offset: usize) -> Result<()> {
        self.arena
            .csr_adaptive_logits_parameter_slice_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::PARAMETERS,
                source_offset,
                target_offset,
                Self::ADAPTIVE_LOGITS,
                self.nodes,
                self.adaptive_edges,
                10,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .csr_row_softmax_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_LOGITS,
                Self::ADAPTIVE_WEIGHTS,
                self.nodes,
                self.adaptive_edges,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Executes the four-stage exponentially dilated long-trend stack from
    /// the frozen patch encoder. The final tensor is `[batch, 1, tile_nodes,
    /// hidden]` for the normal LSTTN context sizes and remains resident for
    /// the daily/weekly/short fusion stage.
    fn long_branch(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
    ) -> Result<usize> {
        self.arena
            .transpose_node_time_f32(
                Self::ATTENTION_SEQUENCE,
                Self::LONG_TEMPORAL,
                batches,
                tile_nodes,
                patches,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut input = Self::LONG_TEMPORAL;
        let mut output = Self::LONG_STAGE_A;
        let mut times = patches;
        for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate() {
            let offset = layout.lsttn_dilated_convolution + layer * (3 * hidden * hidden + hidden);
            self.arena
                .lsttn_long_conv_pool_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    offset,
                    offset + 3 * hidden * hidden,
                    output,
                    batches,
                    times,
                    tile_nodes,
                    hidden,
                    dilation,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            times = times.div_ceil(2).div_ceil(2);
            input = output;
            output = if output == Self::LONG_STAGE_A {
                Self::LONG_STAGE_B
            } else {
                Self::LONG_STAGE_A
            };
        }
        debug_assert_eq!(times, 1);
        Ok(input)
    }

    fn short_input_projection(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        lookback: usize,
        tile_nodes: usize,
        channels: usize,
        recent_window: usize,
        hidden: usize,
        phase_offset: usize,
        periodicity: usize,
    ) -> Result<usize> {
        self.arena
            .lsttn_short_input_projection_parameter_slice_f32(
                Self::SUPERVISED_INPUT,
                Self::PARAMETERS,
                layout.lsttn_short_wave,
                layout.lsttn_short_wave + 2 * hidden,
                Self::SHORT_INPUT,
                batches,
                lookback,
                tile_nodes,
                channels,
                recent_window,
                hidden,
                phase_offset,
                periodicity,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn short_wave_layer_forward(
        &mut self,
        input: usize,
        skip: usize,
        next_skip: usize,
        layer: usize,
        dilation: usize,
        layout: GraphParameterLayout,
        batches: usize,
        times: usize,
        hidden: usize,
        training: bool,
        step: u64,
    ) -> Result<usize> {
        let nodes = self.nodes;
        let layer_width = 12 * hidden * hidden + 6 * hidden;
        let layers_start = layout.lsttn_short_wave + 3 * hidden;
        let base = layers_start + layer * layer_width;
        let filter = base;
        let gate = filter + 2 * hidden * hidden;
        let filter_bias = gate + 2 * hidden * hidden;
        let gate_bias = filter_bias + hidden;
        let graph_projection = gate_bias + hidden;
        let skip_projection = graph_projection + 7 * hidden * hidden + hidden;
        let norm = skip_projection + hidden * hidden + hidden;
        let out_times = times - dilation;
        let seq_batches = batches * out_times;
        let seq_len = seq_batches * nodes * hidden;
        self.arena
            .causal_conv2_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                filter,
                filter_bias,
                Self::SHORT_FILTER,
                batches,
                times,
                nodes,
                hidden,
                hidden,
                dilation,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .causal_conv2_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                gate,
                gate_bias,
                Self::SHORT_GATE,
                batches,
                times,
                nodes,
                hidden,
                hidden,
                dilation,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .gated_tanh_sigmoid_f32(
                Self::SHORT_FILTER,
                Self::SHORT_GATE,
                Self::SHORT_GATED,
                seq_len,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_GATED,
            Self::SHORT_SKIP_PROJECTION,
            skip_projection,
            seq_batches * nodes,
            hidden,
            hidden,
        )?;
        self.arena
            .add_tail_time_f32(
                skip,
                Self::SHORT_SKIP_PROJECTION,
                next_skip,
                batches,
                times,
                out_times,
                nodes,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_GATED,
                Self::SHORT_FORWARD_ONE,
                seq_batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_FORWARD_TWO,
                seq_batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_GATED,
                Self::SHORT_BACKWARD_ONE,
                seq_batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_BACKWARD_TWO,
                seq_batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_GATED,
                Self::SHORT_ADAPTIVE_ONE,
                seq_batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_ADAPTIVE_TWO,
                seq_batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_GATED,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_CONCAT_A,
                seq_batches * nodes,
                hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_A,
                Self::SHORT_FORWARD_TWO,
                Self::SHORT_CONCAT_B,
                seq_batches * nodes,
                2 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_B,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_CONCAT_C,
                seq_batches * nodes,
                3 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_C,
                Self::SHORT_BACKWARD_TWO,
                Self::SHORT_CONCAT_D,
                seq_batches * nodes,
                4 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_D,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_CONCAT_E,
                seq_batches * nodes,
                5 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_E,
                Self::SHORT_ADAPTIVE_TWO,
                Self::SHORT_CONCAT_F,
                seq_batches * nodes,
                6 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_CONCAT_F,
            Self::SHORT_GRAPH,
            graph_projection,
            seq_batches * nodes,
            7 * hidden,
            hidden,
        )?;
        self.arena
            .add_tail_time_f32(
                input,
                Self::SHORT_GRAPH,
                Self::SHORT_RESIDUAL,
                batches,
                times,
                out_times,
                nodes,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .deterministic_dropout_f32(
                Self::SHORT_RESIDUAL,
                Self::SHORT_INPUT,
                seq_len,
                step ^ ((layer as u64) << 48),
                0,
                training,
                0.3,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .batch_norm_channels_parameter_slice_f32(
                Self::SHORT_INPUT,
                Self::PARAMETERS,
                norm,
                norm + hidden,
                Self::SHORT_BATCH_STATS,
                Self::SHORT_NORMALIZED,
                batches,
                out_times,
                nodes,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        Ok(out_times)
    }

    /// Full-node Graph WaveNet executor. CSR diffusion is global by design;
    /// callers must not use this after a node-tile upload.
    #[allow(clippy::too_many_arguments)]
    fn short_branch(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        lookback: usize,
        channels: usize,
        recent_window: usize,
        hidden: usize,
        phase_offset: usize,
        periodicity: usize,
        training: bool,
        step: u64,
    ) -> Result<usize> {
        let nodes = self.nodes;
        let mut times = self.short_input_projection(
            layout,
            batches,
            lookback,
            nodes,
            channels,
            recent_window,
            hidden,
            phase_offset,
            periodicity,
        )?;
        self.adaptive_weights(
            layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_target,
        )?;
        self.arena
            .fill_f32(Self::SHORT_SKIP_A, batches * times * nodes * hidden, 0.0)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let mut input = Self::SHORT_INPUT;
        let mut skip = Self::SHORT_SKIP_A;
        let layer_width = 12 * hidden * hidden + 6 * hidden;
        let layers_start = layout.lsttn_short_wave + 3 * hidden;
        for (layer, dilation) in [1usize, 2, 1, 2, 1, 2, 1, 2].into_iter().enumerate() {
            let base = layers_start + layer * layer_width;
            let filter = base;
            let gate = filter + 2 * hidden * hidden;
            let filter_bias = gate + 2 * hidden * hidden;
            let gate_bias = filter_bias + hidden;
            let graph_projection = gate_bias + hidden;
            let skip_projection = graph_projection + 7 * hidden * hidden + hidden;
            let norm = skip_projection + hidden * hidden + hidden;
            let out_times = times - dilation;
            let seq_batches = batches * out_times;
            let seq_len = seq_batches * nodes * hidden;
            self.arena
                .causal_conv2_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    filter,
                    filter_bias,
                    Self::SHORT_FILTER,
                    batches,
                    times,
                    nodes,
                    hidden,
                    hidden,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .causal_conv2_parameter_slice_f32(
                    input,
                    Self::PARAMETERS,
                    gate,
                    gate_bias,
                    Self::SHORT_GATE,
                    batches,
                    times,
                    nodes,
                    hidden,
                    hidden,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .gated_tanh_sigmoid_f32(
                    Self::SHORT_FILTER,
                    Self::SHORT_GATE,
                    Self::SHORT_GATED,
                    seq_len,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.affine_projection(
                Self::SHORT_GATED,
                Self::SHORT_SKIP_PROJECTION,
                skip_projection,
                seq_batches * nodes,
                hidden,
                hidden,
            )?;
            let next_skip = if skip == Self::SHORT_SKIP_A {
                Self::SHORT_SKIP_B
            } else {
                Self::SHORT_SKIP_A
            };
            self.arena
                .add_tail_time_f32(
                    skip,
                    Self::SHORT_SKIP_PROJECTION,
                    next_skip,
                    batches,
                    times,
                    out_times,
                    nodes,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_FORWARD_ONE,
                    seq_batches,
                    nodes,
                    hidden,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_FORWARD_ONE,
                    Self::SHORT_FORWARD_TWO,
                    seq_batches,
                    nodes,
                    hidden,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_BACKWARD_ONE,
                    seq_batches,
                    nodes,
                    hidden,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_BACKWARD_ONE,
                    Self::SHORT_BACKWARD_TWO,
                    seq_batches,
                    nodes,
                    hidden,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_ADAPTIVE_ONE,
                    seq_batches,
                    nodes,
                    hidden,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_ADAPTIVE_ONE,
                    Self::SHORT_ADAPTIVE_TWO,
                    seq_batches,
                    nodes,
                    hidden,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_GATED,
                    Self::SHORT_FORWARD_ONE,
                    Self::SHORT_CONCAT_A,
                    seq_batches * nodes,
                    hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_A,
                    Self::SHORT_FORWARD_TWO,
                    Self::SHORT_CONCAT_B,
                    seq_batches * nodes,
                    2 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_B,
                    Self::SHORT_BACKWARD_ONE,
                    Self::SHORT_CONCAT_C,
                    seq_batches * nodes,
                    3 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_C,
                    Self::SHORT_BACKWARD_TWO,
                    Self::SHORT_CONCAT_D,
                    seq_batches * nodes,
                    4 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_D,
                    Self::SHORT_ADAPTIVE_ONE,
                    Self::SHORT_CONCAT_E,
                    seq_batches * nodes,
                    5 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .concat_channels_f32(
                    Self::SHORT_CONCAT_E,
                    Self::SHORT_ADAPTIVE_TWO,
                    Self::SHORT_CONCAT_F,
                    seq_batches * nodes,
                    6 * hidden,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.affine_projection(
                Self::SHORT_CONCAT_F,
                Self::SHORT_GRAPH,
                graph_projection,
                seq_batches * nodes,
                7 * hidden,
                hidden,
            )?;
            self.arena
                .add_tail_time_f32(
                    input,
                    Self::SHORT_GRAPH,
                    Self::SHORT_RESIDUAL,
                    batches,
                    times,
                    out_times,
                    nodes,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .deterministic_dropout_f32(
                    Self::SHORT_RESIDUAL,
                    Self::SHORT_INPUT,
                    seq_len,
                    step ^ ((layer as u64) << 48),
                    0,
                    training,
                    0.3,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .batch_norm_channels_parameter_slice_f32(
                    Self::SHORT_INPUT,
                    Self::PARAMETERS,
                    norm,
                    norm + hidden,
                    Self::SHORT_BATCH_STATS,
                    Self::SHORT_NORMALIZED,
                    batches,
                    out_times,
                    nodes,
                    hidden,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            input = Self::SHORT_NORMALIZED;
            skip = next_skip;
            times = out_times;
        }
        let end_one = layers_start + 8 * layer_width;
        let end_two = end_one + hidden * hidden + hidden;
        let rows = batches * times * nodes;
        self.arena
            .relu_f32(skip, Self::SHORT_FILTER, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_FILTER,
            Self::SHORT_GATE,
            end_one,
            rows,
            hidden,
            hidden,
        )?;
        self.arena
            .relu_f32(Self::SHORT_GATE, Self::SHORT_GATED, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_GATED,
            Self::SHORT_SKIP_PROJECTION,
            end_two,
            rows,
            hidden,
            hidden,
        )?;
        Ok(Self::SHORT_SKIP_PROJECTION)
    }

    fn periodic_feature(
        &mut self,
        layout: GraphParameterLayout,
        output: usize,
        batches: usize,
        patches: usize,
        hidden: usize,
        lag_patches: usize,
        seasonal: bool,
    ) -> Result<()> {
        if lag_patches >= patches {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN periodic lag exceeds frozen patch context".to_string(),
            ));
        }
        let nodes = self.nodes;
        self.arena
            .select_node_major_time_f32(
                Self::ATTENTION_SEQUENCE,
                Self::SHORT_FILTER,
                batches,
                nodes,
                patches,
                hidden,
                patches - lag_patches - 1,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let (source, target, projection) = if seasonal {
            (
                layout.lsttn_weekly_adaptive_source,
                layout.lsttn_weekly_adaptive_target,
                layout.lsttn_periodic_projection + (7 * hidden * hidden + hidden),
            )
        } else {
            (
                layout.lsttn_adaptive_source,
                layout.lsttn_adaptive_target,
                layout.lsttn_periodic_projection,
            )
        };
        self.adaptive_weights(source, target)?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FILTER,
                Self::SHORT_FORWARD_ONE,
                batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_FORWARD_TWO,
                batches,
                nodes,
                hidden,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::SHORT_BACKWARD_ONE,
                batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_BACKWARD_TWO,
                batches,
                nodes,
                hidden,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::SHORT_ADAPTIVE_ONE,
                batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_ADAPTIVE_TWO,
                batches,
                nodes,
                hidden,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_FILTER,
                Self::SHORT_FORWARD_ONE,
                Self::SHORT_CONCAT_A,
                batches * nodes,
                hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_A,
                Self::SHORT_FORWARD_TWO,
                Self::SHORT_CONCAT_B,
                batches * nodes,
                2 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_B,
                Self::SHORT_BACKWARD_ONE,
                Self::SHORT_CONCAT_C,
                batches * nodes,
                3 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_C,
                Self::SHORT_BACKWARD_TWO,
                Self::SHORT_CONCAT_D,
                batches * nodes,
                4 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_D,
                Self::SHORT_ADAPTIVE_ONE,
                Self::SHORT_CONCAT_E,
                batches * nodes,
                5 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::SHORT_CONCAT_E,
                Self::SHORT_ADAPTIVE_TWO,
                Self::SHORT_CONCAT_F,
                batches * nodes,
                6 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let saved_input = if seasonal {
            Self::PERIODIC_SEASONAL_INPUT
        } else {
            Self::PERIODIC_SHORT_INPUT
        };
        self.arena
            .deterministic_dropout_f32(
                Self::SHORT_CONCAT_F,
                saved_input,
                batches * nodes * 7 * hidden,
                0,
                0,
                false,
                0.0,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            saved_input,
            output,
            projection,
            batches * nodes,
            7 * hidden,
            hidden,
        )
    }

}
