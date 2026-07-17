#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaLsttnTensorExecutor {
    fn recompute_long_input_for_layer(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        hidden: usize,
        layer_limit: usize,
    ) -> Result<(usize, usize)> {
        self.arena
            .transpose_node_time_f32(
                Self::ATTENTION_SEQUENCE,
                Self::LONG_TEMPORAL,
                batches,
                self.nodes,
                patches,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut input = Self::LONG_TEMPORAL;
        let mut output = Self::LONG_STAGE_A;
        let mut times = patches;
        for (layer, dilation) in [1usize, 2, 4, 8].into_iter().enumerate().take(layer_limit) {
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
                    self.nodes,
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
        Ok((input, times))
    }

    fn long_branch_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
    ) -> Result<()> {
        let layout = state.layout();
        let patch_width = (state.periodicity / 24).max(1);
        let patches = state.context_window / patch_width;
        let hidden = state.hidden;
        let mut output_gradient = Self::LONG_GRADIENT;
        for layer in (0..4).rev() {
            let (input, times) =
                self.recompute_long_input_for_layer(layout, batches, patches, hidden, layer)?;
            let input_gradient = if output_gradient == Self::LONG_BACKWARD_A {
                Self::LONG_BACKWARD_B
            } else {
                Self::LONG_BACKWARD_A
            };
            let offset = layout.lsttn_dilated_convolution + layer * (3 * hidden * hidden + hidden);
            let dilation = [1usize, 2, 4, 8][layer];
            self.arena
                .lsttn_long_conv_pool_parameter_slice_backward_f32(
                    input,
                    Self::PARAMETERS,
                    offset,
                    offset + 3 * hidden * hidden,
                    output_gradient,
                    input_gradient,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    times,
                    self.nodes,
                    hidden,
                    dilation,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            output_gradient = input_gradient;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn recompute_short_prefix(
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
        layer_limit: usize,
    ) -> Result<(usize, usize, usize)> {
        let mut times = self.short_input_projection(
            layout,
            batches,
            lookback,
            self.nodes,
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
            .fill_f32(
                Self::SHORT_SKIP_A,
                batches * times * self.nodes * hidden,
                0.0,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let mut input = Self::SHORT_INPUT;
        let mut skip = Self::SHORT_SKIP_A;
        for (layer, dilation) in [1usize, 2, 1, 2, 1, 2, 1, 2]
            .into_iter()
            .enumerate()
            .take(layer_limit)
        {
            let next_skip = if skip == Self::SHORT_SKIP_A {
                Self::SHORT_SKIP_B
            } else {
                Self::SHORT_SKIP_A
            };
            times = self.short_wave_layer_forward(
                input, skip, next_skip, layer, dilation, layout, batches, times, hidden, training,
                step,
            )?;
            input = Self::SHORT_NORMALIZED;
            skip = next_skip;
        }
        Ok((input, skip, times))
    }

    #[allow(clippy::too_many_arguments)]
    fn short_branch_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        channels: usize,
        phase_offset: usize,
        training: bool,
    ) -> Result<()> {
        let layout = state.layout();
        let h = state.hidden;
        let nodes = self.nodes;
        let layer_width = 12 * h * h + 6 * h;
        let layers_start = layout.lsttn_short_wave + 3 * h;
        let (mut input, mut skip, mut times) = self.recompute_short_prefix(
            layout,
            batches,
            state.context_window,
            channels,
            state.recent_window,
            h,
            phase_offset,
            state.periodicity,
            training,
            state.steps,
            8,
        )?;
        let rows = batches * times * nodes;
        let end_one = layers_start + 8 * layer_width;
        let end_two = end_one + h * h + h;
        self.arena
            .relu_f32(skip, Self::SHORT_FILTER, rows * h)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(Self::SHORT_FILTER, Self::SHORT_GATE, end_one, rows, h, h)?;
        self.arena
            .relu_f32(Self::SHORT_GATE, Self::SHORT_GATED, rows * h)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::SHORT_GATED,
            Self::SHORT_SKIP_PROJECTION,
            end_two,
            rows,
            h,
            h,
        )?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::SHORT_GATED,
                Self::PARAMETERS,
                end_two,
                end_two + h * h,
                Self::SHORT_GRADIENT,
                Self::SHORT_REVERSE_GATED_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .relu_backward_f32(
                Self::SHORT_GATE,
                Self::SHORT_REVERSE_GATED_GRADIENT,
                Self::SHORT_REVERSE_GRAPH_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::SHORT_FILTER,
                Self::PARAMETERS,
                end_one,
                end_one + h * h,
                Self::SHORT_REVERSE_GRAPH_GRADIENT,
                Self::SHORT_REVERSE_SKIP_GRADIENT_A,
                Self::PARAMETER_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .relu_backward_f32(
                skip,
                Self::SHORT_REVERSE_SKIP_GRADIENT_A,
                Self::SHORT_REVERSE_SKIP_GRADIENT_B,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        let mut skip_gradient = Self::SHORT_REVERSE_SKIP_GRADIENT_B;
        let mut output_gradient = Self::SHORT_REVERSE_INPUT_GRADIENT_A;
        self.arena
            .fill_f32(output_gradient, batches * times * nodes * h, 0.0)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        for layer in (0..8).rev() {
            let dilation = [1usize, 2, 1, 2, 1, 2, 1, 2][layer];
            let (layer_input, layer_skip, layer_times) = self.recompute_short_prefix(
                layout,
                batches,
                state.context_window,
                channels,
                state.recent_window,
                h,
                phase_offset,
                state.periodicity,
                training,
                state.steps,
                layer,
            )?;
            let layer_next_skip = if layer_skip == Self::SHORT_SKIP_A {
                Self::SHORT_SKIP_B
            } else {
                Self::SHORT_SKIP_A
            };
            let out_times = self.short_wave_layer_forward(
                layer_input,
                layer_skip,
                layer_next_skip,
                layer,
                dilation,
                layout,
                batches,
                layer_times,
                h,
                training,
                state.steps,
            )?;
            debug_assert_eq!(out_times, times);
            let base = layers_start + layer * layer_width;
            let filter = base;
            let gate = filter + 2 * h * h;
            let filter_bias = gate + 2 * h * h;
            let gate_bias = filter_bias + h;
            let graph_projection = gate_bias + h;
            let skip_projection = graph_projection + 7 * h * h + h;
            let norm = skip_projection + h * h + h;
            let seq_rows = batches * out_times * nodes;
            self.arena
                .batch_norm_channels_parameter_slice_backward_f32(
                    Self::SHORT_INPUT,
                    Self::PARAMETERS,
                    norm,
                    norm + h,
                    Self::SHORT_BATCH_STATS,
                    output_gradient,
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    out_times,
                    nodes,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .deterministic_dropout_f32(
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_REVERSE_SPLIT_B,
                    batches * out_times * nodes * h,
                    state.steps ^ ((layer as u64) << 48),
                    0,
                    training,
                    0.3,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_tail_time_backward_f32(
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    Self::SHORT_REVERSE_GRAPH_GRADIENT,
                    batches,
                    layer_times,
                    out_times,
                    nodes,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .affine_backward_parameter_slice_f32(
                    Self::SHORT_CONCAT_F,
                    Self::PARAMETERS,
                    graph_projection,
                    graph_projection + 7 * h * h,
                    Self::SHORT_REVERSE_GRAPH_GRADIENT,
                    Self::SHORT_REVERSE_CONCAT_GRADIENT,
                    Self::PARAMETER_GRADIENT,
                    seq_rows,
                    7 * h,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_CONCAT_GRADIENT,
                    Self::SHORT_GRAPH_GRAD_IDENTITY,
                    Self::SHORT_REVERSE_SPLIT_A,
                    seq_rows,
                    h,
                    6 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_GRAPH_GRAD_FORWARD_ONE,
                    Self::SHORT_REVERSE_SPLIT_C,
                    seq_rows,
                    h,
                    5 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::SHORT_GRAPH_GRAD_FORWARD_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    seq_rows,
                    h,
                    4 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_GRAPH_GRAD_REVERSE_ONE,
                    Self::SHORT_REVERSE_SPLIT_A,
                    seq_rows,
                    h,
                    3 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_GRAPH_GRAD_REVERSE_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    seq_rows,
                    h,
                    2 * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .split_channels_f32(
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_ONE,
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_TWO,
                    seq_rows,
                    h,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .deterministic_dropout_f32(
                    Self::SHORT_GRAPH_GRAD_IDENTITY,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                    0,
                    0,
                    false,
                    0.0,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_FORWARD_ONE,
                    Self::SHORT_GRAPH_GRAD_FORWARD_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_GRAPH_GRAD_FORWARD_ONE,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_SPLIT_B,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::FORWARD_INDPTR,
                    Self::FORWARD_INDICES,
                    Self::FORWARD_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.forward_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_BACKWARD_ONE,
                    Self::SHORT_GRAPH_GRAD_REVERSE_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_GRAPH_GRAD_REVERSE_ONE,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_SPLIT_F,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::REVERSE_INDPTR,
                    Self::REVERSE_INDICES,
                    Self::REVERSE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_REVERSE_SPLIT_F,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.reverse_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_ADAPTIVE_ONE,
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_TWO,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_GRAPH_GRAD_ADAPTIVE_ONE,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_SPLIT_A,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_diffuse_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_GATED,
                    Self::SHORT_REVERSE_SPLIT_A,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    batches * out_times,
                    nodes,
                    h,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_row_softmax_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_WEIGHTS,
                    Self::SHORT_REVERSE_EDGE_GRADIENT,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    nodes,
                    self.adaptive_edges,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .csr_adaptive_logits_parameter_slice_backward_f32(
                    Self::ADAPTIVE_INDPTR,
                    Self::ADAPTIVE_INDICES,
                    Self::PARAMETERS,
                    layout.lsttn_short_adaptive_source,
                    layout.lsttn_short_adaptive_target,
                    Self::SHORT_REVERSE_LOGIT_GRADIENT,
                    Self::PARAMETER_GRADIENT,
                    nodes,
                    self.adaptive_edges,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_E,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .affine_backward_parameter_slice_f32(
                    Self::SHORT_GATED,
                    Self::PARAMETERS,
                    skip_projection,
                    skip_projection + h * h,
                    skip_gradient,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::PARAMETER_GRADIENT,
                    seq_rows,
                    h,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_tail_time_backward_f32(
                    skip_gradient,
                    Self::SHORT_REVERSE_SKIP_GRADIENT_A,
                    Self::SHORT_REVERSE_SPLIT_C,
                    batches,
                    layer_times,
                    out_times,
                    nodes,
                    h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .gated_tanh_sigmoid_backward_f32(
                    Self::SHORT_FILTER,
                    Self::SHORT_GATE,
                    Self::SHORT_REVERSE_GATED_GRADIENT,
                    Self::SHORT_REVERSE_FILTER_GRADIENT,
                    Self::SHORT_REVERSE_GATE_GRADIENT,
                    seq_rows * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .causal_conv2_parameter_slice_backward_f32(
                    layer_input,
                    Self::PARAMETERS,
                    filter,
                    filter_bias,
                    Self::SHORT_REVERSE_FILTER_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    layer_times,
                    nodes,
                    h,
                    h,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .causal_conv2_parameter_slice_backward_f32(
                    layer_input,
                    Self::PARAMETERS,
                    gate,
                    gate_bias,
                    Self::SHORT_REVERSE_GATE_GRADIENT,
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::PARAMETER_GRADIENT,
                    batches,
                    layer_times,
                    nodes,
                    h,
                    h,
                    dilation,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    Self::SHORT_REVERSE_SPLIT_B,
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    batches * layer_times * nodes * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            self.arena
                .add_f32(
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    Self::SHORT_REVERSE_SPLIT_C,
                    Self::SHORT_REVERSE_INPUT_GRADIENT_B,
                    batches * layer_times * nodes * h,
                )
                .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
            output_gradient = Self::SHORT_REVERSE_INPUT_GRADIENT_B;
            skip_gradient = Self::SHORT_REVERSE_SKIP_GRADIENT_A;
            input = layer_input;
            skip = layer_skip;
            times = layer_times;
        }
        let _ = (input, skip, times);
        Ok(())
    }

}
