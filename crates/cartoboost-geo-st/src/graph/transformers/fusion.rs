#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaLsttnTensorExecutor {
    fn fuse_and_direct_output(
        &mut self,
        layout: GraphParameterLayout,
        long: usize,
        short: usize,
        batches: usize,
        hidden: usize,
        horizons: usize,
    ) -> Result<usize> {
        let rows = batches * self.nodes;
        self.arena
            .concat_channels_f32(
                long,
                Self::PERIODIC_SHORT,
                Self::FUSION_A,
                rows,
                hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .concat_channels_f32(
                Self::FUSION_A,
                Self::PERIODIC_SEASONAL,
                Self::FUSION_B,
                rows,
                2 * hidden,
                hidden,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_B,
            Self::FUSION_C,
            layout.lsttn_fusion,
            rows,
            3 * hidden,
            hidden,
        )?;
        self.arena
            .relu_f32(Self::FUSION_C, Self::FUSION_D, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_D,
            Self::FUSION_C,
            layout.lsttn_fusion + (3 * hidden + 1) * hidden,
            rows,
            hidden,
            hidden,
        )?;
        self.arena
            .concat_channels_f32(short, Self::FUSION_C, Self::FUSION_A, rows, hidden, hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_A,
            Self::FUSION_OUTPUT,
            layout.lsttn_fusion + (4 * hidden * hidden + 2 * hidden),
            rows,
            2 * hidden,
            hidden,
        )?;
        self.arena
            .relu_f32(Self::FUSION_OUTPUT, Self::FUSION_OUTPUT, rows * hidden)
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.affine_projection(
            Self::FUSION_OUTPUT,
            Self::DIRECT_NODE_MAJOR,
            layout.output,
            rows,
            hidden,
            horizons,
        )?;
        self.arena
            .node_major_horizons_to_output_f32(
                Self::DIRECT_NODE_MAJOR,
                Self::FUSED_DIRECT_OUTPUT,
                batches,
                self.nodes,
                horizons,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        Ok(Self::FUSED_DIRECT_OUTPUT)
    }

    fn supervised_forward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        channels: usize,
        phase_offset: usize,
        training: bool,
    ) -> Result<usize> {
        if batches == 0 || batches > 32 {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN batches must be 1..=32".to_string(),
            ));
        }
        let layout = state.layout();
        let patch_width = (state.periodicity / 24).max(1);
        let patches = state.context_window / patch_width;
        let hidden = state.hidden;
        self.patch_embedding(
            layout,
            batches,
            state.context_window,
            self.nodes,
            channels,
            patch_width,
            hidden,
        )?;
        self.add_patch_positions(layout, batches, patches, self.nodes, hidden)?;
        self.patch_attention_layout(batches, patches, self.nodes, hidden)?;
        self.frozen_encoder(
            layout,
            batches,
            patches,
            self.nodes,
            hidden,
            state.attention_heads,
        )?;
        let long = self.long_branch(layout, batches, patches, self.nodes, hidden)?;
        let short = self.short_branch(
            layout,
            batches,
            state.context_window,
            channels,
            state.recent_window,
            hidden,
            phase_offset,
            state.periodicity,
            training,
            state.steps,
        )?;
        let short_lag = (if state.periodic_short_lag == 0 {
            state.periodicity
        } else {
            state.periodic_short_lag
        } / patch_width)
            .max(1);
        let seasonal_lag = (state.periodicity / patch_width).max(1);
        self.periodic_feature(
            layout,
            Self::PERIODIC_SHORT,
            batches,
            patches,
            hidden,
            short_lag,
            false,
        )?;
        self.periodic_feature(
            layout,
            Self::PERIODIC_SEASONAL,
            batches,
            patches,
            hidden,
            seasonal_lag,
            true,
        )?;
        self.fuse_and_direct_output(layout, long, short, batches, hidden, state.horizons)
    }

    fn direct_head_loss_and_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
    ) -> Result<()> {
        let layout = state.layout();
        let len = batches * state.horizons * self.nodes;
        self.begin_supervised_gradient()?;
        self.arena
            .masked_inverse_scale_mae_loss_backward_f32(
                Self::FUSED_DIRECT_OUTPUT,
                Self::SUPERVISED_TARGET,
                Self::DIRECT_OUTPUT_GRADIENT,
                Self::SUPERVISED_LOSS,
                len,
                state.normalized_zero as f32,
                state.target_scale as f32,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .output_to_node_major_horizons_f32(
                Self::DIRECT_OUTPUT_GRADIENT,
                Self::DIRECT_NODE_MAJOR_GRADIENT,
                batches,
                self.nodes,
                state.horizons,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_OUTPUT,
                Self::PARAMETERS,
                layout.output,
                layout.output + state.hidden * state.horizons,
                Self::DIRECT_NODE_MAJOR_GRADIENT,
                Self::FUSION_REPRESENTATION_GRADIENT,
                Self::PARAMETER_GRADIENT,
                batches * self.nodes,
                state.hidden,
                state.horizons,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        Ok(())
    }

    fn supervised_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        channels: usize,
        phase_offset: usize,
        training: bool,
    ) -> Result<()> {
        self.direct_head_loss_and_backward(state, batches)?;
        self.fusion_backward(state, batches)?;
        self.short_branch_backward(state, batches, channels, phase_offset, training)?;
        self.long_branch_backward(state, batches)?;
        self.periodic_projection_backward(state, batches, false)?;
        self.periodic_projection_backward(state, batches, true)?;
        let patch_width = (state.periodicity / 24).max(1);
        let short_lag = (if state.periodic_short_lag == 0 {
            state.periodicity
        } else {
            state.periodic_short_lag
        } / patch_width)
            .max(1);
        let seasonal_lag = (state.periodicity / patch_width).max(1);
        self.periodic_graph_backward(state, batches, short_lag, false)?;
        self.periodic_graph_backward(state, batches, seasonal_lag, true)
    }

    fn fusion_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
    ) -> Result<()> {
        let layout = state.layout();
        let rows = batches * self.nodes;
        let h = state.hidden;
        self.arena
            .relu_backward_f32(
                Self::FUSION_OUTPUT,
                Self::FUSION_REPRESENTATION_GRADIENT,
                Self::FUSION_RELU_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_A,
                Self::PARAMETERS,
                layout.lsttn_fusion + (4 * h * h + 2 * h),
                layout.lsttn_fusion + (4 * h * h + 2 * h) + 2 * h * h,
                Self::FUSION_RELU_GRADIENT,
                Self::FUSION_CONCAT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                2 * h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::FUSION_CONCAT_GRADIENT,
                Self::SHORT_GRADIENT,
                Self::TREND_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_D,
                Self::PARAMETERS,
                layout.lsttn_fusion + (3 * h + 1) * h,
                layout.lsttn_fusion + (3 * h + 1) * h + h * h,
                Self::TREND_GRADIENT,
                Self::FUSION_SECOND_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .relu_backward_f32(
                Self::FUSION_D,
                Self::FUSION_SECOND_GRADIENT,
                Self::FUSION_FIRST_RELU_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                Self::FUSION_B,
                Self::PARAMETERS,
                layout.lsttn_fusion,
                layout.lsttn_fusion + 3 * h * h,
                Self::FUSION_FIRST_RELU_GRADIENT,
                Self::FUSION_INPUT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                rows,
                3 * h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::FUSION_INPUT_GRADIENT,
                Self::LONG_GRADIENT,
                Self::PERIODIC_PAIR_GRADIENT,
                rows,
                h,
                2 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_PAIR_GRADIENT,
                Self::PERIODIC_SHORT_GRADIENT,
                Self::PERIODIC_SEASONAL_GRADIENT,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))
    }

    fn periodic_projection_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        seasonal: bool,
    ) -> Result<()> {
        let layout = state.layout();
        let h = state.hidden;
        let projection =
            layout.lsttn_periodic_projection + if seasonal { 7 * h * h + h } else { 0 };
        let (input, output_gradient, input_gradient) = if seasonal {
            (
                Self::PERIODIC_SEASONAL_INPUT,
                Self::PERIODIC_SEASONAL_GRADIENT,
                Self::PERIODIC_SEASONAL_INPUT_GRADIENT,
            )
        } else {
            (
                Self::PERIODIC_SHORT_INPUT,
                Self::PERIODIC_SHORT_GRADIENT,
                Self::PERIODIC_SHORT_INPUT_GRADIENT,
            )
        };
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                projection,
                projection + 7 * h * h,
                output_gradient,
                input_gradient,
                Self::PARAMETER_GRADIENT,
                batches * self.nodes,
                7 * h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))
    }

    fn periodic_graph_backward(
        &mut self,
        state: &TrainableGraphTransformerState,
        batches: usize,
        lag_patches: usize,
        seasonal: bool,
    ) -> Result<()> {
        let layout = state.layout();
        let h = state.hidden;
        let patches = state.context_window / (state.periodicity / 24).max(1);
        let rows = batches * self.nodes;
        let (source, target, output, input_gradient) = if seasonal {
            (
                layout.lsttn_weekly_adaptive_source,
                layout.lsttn_weekly_adaptive_target,
                Self::PERIODIC_SEASONAL,
                Self::PERIODIC_SEASONAL_INPUT_GRADIENT,
            )
        } else {
            (
                layout.lsttn_adaptive_source,
                layout.lsttn_adaptive_target,
                Self::PERIODIC_SHORT,
                Self::PERIODIC_SHORT_INPUT_GRADIENT,
            )
        };
        self.periodic_feature(layout, output, batches, patches, h, lag_patches, seasonal)?;
        self.arena
            .split_channels_f32(
                input_gradient,
                Self::PERIODIC_GRAD_IDENTITY,
                Self::PERIODIC_TEMP_A,
                rows,
                h,
                6 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_GRAD_FORWARD_ONE,
                Self::PERIODIC_TEMP_B,
                rows,
                h,
                5 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_B,
                Self::PERIODIC_GRAD_FORWARD_TWO,
                Self::PERIODIC_TEMP_A,
                rows,
                h,
                4 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_GRAD_REVERSE_ONE,
                Self::PERIODIC_TEMP_B,
                rows,
                h,
                3 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_B,
                Self::PERIODIC_GRAD_REVERSE_TWO,
                Self::PERIODIC_TEMP_A,
                rows,
                h,
                2 * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .split_channels_f32(
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_GRAD_ADAPTIVE_ONE,
                Self::PERIODIC_GRAD_ADAPTIVE_TWO,
                rows,
                h,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .csr_diffuse_backward_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FORWARD_ONE,
                Self::PERIODIC_GRAD_FORWARD_TWO,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_FORWARD_ONE,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_C,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_backward_f32(
                Self::FORWARD_INDPTR,
                Self::FORWARD_INDICES,
                Self::FORWARD_WEIGHTS,
                Self::SHORT_FILTER,
                Self::PERIODIC_TEMP_C,
                Self::PERIODIC_FORWARD_BASE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.forward_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .csr_diffuse_backward_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_BACKWARD_ONE,
                Self::PERIODIC_GRAD_REVERSE_TWO,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_REVERSE_ONE,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_C,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_backward_f32(
                Self::REVERSE_INDPTR,
                Self::REVERSE_INDICES,
                Self::REVERSE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::PERIODIC_TEMP_C,
                Self::PERIODIC_REVERSE_BASE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.reverse_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .csr_diffuse_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_ADAPTIVE_ONE,
                Self::PERIODIC_GRAD_ADAPTIVE_TWO,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                batches,
                self.nodes,
                h,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_ADAPTIVE_ONE,
                Self::PERIODIC_TEMP_A,
                Self::PERIODIC_TEMP_C,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_diffuse_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::ADAPTIVE_WEIGHTS,
                Self::SHORT_FILTER,
                Self::PERIODIC_TEMP_C,
                Self::PERIODIC_ADAPTIVE_BASE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                batches,
                self.nodes,
                h,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                Self::PERIODIC_TEMP_B,
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_row_softmax_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_WEIGHTS,
                Self::PERIODIC_ADAPTIVE_EDGE_GRADIENT,
                Self::PERIODIC_ADAPTIVE_LOGIT_GRADIENT,
                self.nodes,
                self.adaptive_edges,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .csr_adaptive_logits_parameter_slice_backward_f32(
                Self::ADAPTIVE_INDPTR,
                Self::ADAPTIVE_INDICES,
                Self::PARAMETERS,
                source,
                target,
                Self::PERIODIC_ADAPTIVE_LOGIT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                self.nodes,
                self.adaptive_edges,
                h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;

        self.arena
            .add_f32(
                Self::PERIODIC_GRAD_IDENTITY,
                Self::PERIODIC_FORWARD_BASE_GRADIENT,
                Self::PERIODIC_BASE_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_BASE_GRADIENT,
                Self::PERIODIC_REVERSE_BASE_GRADIENT,
                Self::PERIODIC_BASE_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))?;
        self.arena
            .add_f32(
                Self::PERIODIC_BASE_GRADIENT,
                Self::PERIODIC_ADAPTIVE_BASE_GRADIENT,
                Self::PERIODIC_BASE_GRADIENT,
                rows * h,
            )
            .map_err(|e| GeoStError::InvalidBackend(e.to_string()))
    }

}
