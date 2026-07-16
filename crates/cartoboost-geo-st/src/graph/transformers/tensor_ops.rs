#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaLsttnTensorExecutor {
    fn affine_projection(
        &mut self,
        input_slot: usize,
        output_slot: usize,
        parameter_offset: usize,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        self.arena
            .affine_parameter_slice_f32(
                input_slot,
                Self::PARAMETERS,
                parameter_offset,
                parameter_offset + input_width * output_width,
                output_slot,
                rows,
                input_width,
                output_width,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn patch_embedding(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        times: usize,
        tile_nodes: usize,
        channels: usize,
        patch_width: usize,
        hidden: usize,
    ) -> Result<()> {
        self.arena
            .patch_embedding_f32(
                Self::SUPERVISED_INPUT,
                Self::PARAMETERS,
                layout.lsttn_patch_embedding,
                layout.lsttn_patch_embedding + patch_width * hidden,
                Self::PATCH_EMBEDDING,
                batches,
                times,
                tile_nodes,
                channels,
                patch_width,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn add_patch_positions(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
    ) -> Result<()> {
        self.arena
            .add_patch_positions_f32(
                Self::PATCH_EMBEDDING,
                Self::PARAMETERS,
                layout.pretrain_position,
                Self::PATCH_WITH_POSITION,
                batches,
                patches,
                tile_nodes,
                hidden,
                (hidden as f32).sqrt(),
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn patch_attention_layout(
        &mut self,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
    ) -> Result<()> {
        self.arena
            .patches_to_attention_sequences_f32(
                Self::PATCH_WITH_POSITION,
                Self::ATTENTION_SEQUENCE,
                batches,
                patches,
                tile_nodes,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn encoder_attention(
        &mut self,
        layout: GraphParameterLayout,
        layer: usize,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        let projection = hidden * (hidden + 1);
        let rows = batches * patches * tile_nodes;
        self.affine_projection(
            Self::ATTENTION_SEQUENCE,
            Self::ENCODER_Q,
            layout.temporal_q + layer * projection,
            rows,
            hidden,
            hidden,
        )?;
        self.affine_projection(
            Self::ATTENTION_SEQUENCE,
            Self::ENCODER_K,
            layout.temporal_k + layer * projection,
            rows,
            hidden,
            hidden,
        )?;
        self.affine_projection(
            Self::ATTENTION_SEQUENCE,
            Self::ENCODER_V,
            layout.temporal_v + layer * projection,
            rows,
            hidden,
            hidden,
        )?;
        self.arena
            .attention_f32(
                Self::ENCODER_Q,
                Self::ENCODER_K,
                Self::ENCODER_V,
                Self::ENCODER_ATTENTION,
                batches * tile_nodes,
                patches,
                heads,
                hidden / heads,
                false,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Runs all four frozen MST encoder layers on contiguous device tensors.
    /// The input and final representation are both `[batch * nodes, patches,
    /// hidden]`; no activation crosses PCIe between blocks.  These parameters
    /// are frozen during supervised fitting, but the device computation is
    /// still the representation consumed by the CUDA LSTTN branches.
    fn frozen_encoder(
        &mut self,
        layout: GraphParameterLayout,
        batches: usize,
        patches: usize,
        tile_nodes: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        if hidden == 0 || heads == 0 || hidden % heads != 0 {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN attention hidden size must divide evenly across heads".to_string(),
            ));
        }
        let rows = batches * patches * tile_nodes;
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;
        for layer in 0..4 {
            self.encoder_attention(layout, layer, batches, patches, tile_nodes, hidden, heads)?;
            self.affine_projection(
                Self::ENCODER_ATTENTION,
                Self::ENCODER_PROJECTED,
                layout.lsttn_transformer_out + layer * projection,
                rows,
                hidden,
                hidden,
            )?;
            self.arena
                .add_f32(
                    Self::ATTENTION_SEQUENCE,
                    Self::ENCODER_PROJECTED,
                    Self::ENCODER_RESIDUAL,
                    rows * hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            let norm = layout.lsttn_transformer_norm + layer * 4 * hidden;
            self.arena
                .layer_norm_parameter_slice_f32(
                    Self::ENCODER_RESIDUAL,
                    Self::PARAMETERS,
                    norm,
                    norm + hidden,
                    Self::ENCODER_NORMALIZED,
                    rows,
                    hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.affine_projection(
                Self::ENCODER_NORMALIZED,
                Self::ENCODER_FFN_EXPANDED,
                layout.lsttn_transformer_ffn + layer * ffn_width,
                rows,
                hidden,
                4 * hidden,
            )?;
            self.arena
                .relu_f32(
                    Self::ENCODER_FFN_EXPANDED,
                    Self::ENCODER_FFN_ACTIVATED,
                    rows * 4 * hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.affine_projection(
                Self::ENCODER_FFN_ACTIVATED,
                Self::ENCODER_FFN_CONTRACTED,
                layout.lsttn_transformer_ffn + layer * ffn_width + (hidden + 1) * 4 * hidden,
                rows,
                4 * hidden,
                hidden,
            )?;
            self.arena
                .add_f32(
                    Self::ENCODER_NORMALIZED,
                    Self::ENCODER_FFN_CONTRACTED,
                    Self::ENCODER_FFN_RESIDUAL,
                    rows * hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
            self.arena
                .layer_norm_parameter_slice_f32(
                    Self::ENCODER_FFN_RESIDUAL,
                    Self::PARAMETERS,
                    norm + 2 * hidden,
                    norm + 3 * hidden,
                    Self::ATTENTION_SEQUENCE,
                    rows,
                    hidden,
                )
                .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_forward_slots(
        &mut self,
        input: usize,
        output: usize,
        q: usize,
        k: usize,
        v: usize,
        attention: usize,
        projected: usize,
        residual: usize,
        normalized: usize,
        ffn_expanded: usize,
        ffn_activated: usize,
        ffn_contracted: usize,
        ffn_residual: usize,
        q_offset: usize,
        k_offset: usize,
        v_offset: usize,
        out_offset: usize,
        ffn_offset: usize,
        norm_offset: usize,
        sequences: usize,
        tokens: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        let rows = sequences * tokens;
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;
        self.affine_projection(input, q, q_offset, rows, hidden, hidden)?;
        self.affine_projection(input, k, k_offset, rows, hidden, hidden)?;
        self.affine_projection(input, v, v_offset, rows, hidden, hidden)?;
        self.arena
            .attention_f32(
                q,
                k,
                v,
                attention,
                sequences,
                tokens,
                heads,
                hidden / heads,
                false,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.affine_projection(attention, projected, out_offset, rows, hidden, hidden)?;
        self.arena
            .add_f32(input, projected, residual, rows * hidden)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .layer_norm_parameter_slice_f32(
                residual,
                Self::PARAMETERS,
                norm_offset,
                norm_offset + hidden,
                normalized,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.affine_projection(
            normalized,
            ffn_expanded,
            ffn_offset,
            rows,
            hidden,
            4 * hidden,
        )?;
        self.arena
            .relu_f32(ffn_expanded, ffn_activated, rows * 4 * hidden)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.affine_projection(
            ffn_activated,
            ffn_contracted,
            ffn_offset + (hidden + 1) * 4 * hidden,
            rows,
            4 * hidden,
            hidden,
        )?;
        self.arena
            .add_f32(normalized, ffn_contracted, ffn_residual, rows * hidden)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .layer_norm_parameter_slice_f32(
                ffn_residual,
                Self::PARAMETERS,
                norm_offset + 2 * hidden,
                norm_offset + 3 * hidden,
                output,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let _ = (projection, ffn_width);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_backward_slots(
        &mut self,
        input: usize,
        output_gradient: usize,
        input_gradient: usize,
        q: usize,
        k: usize,
        v: usize,
        attention: usize,
        projected: usize,
        residual: usize,
        normalized: usize,
        ffn_expanded: usize,
        ffn_activated: usize,
        ffn_contracted: usize,
        ffn_residual: usize,
        q_offset: usize,
        k_offset: usize,
        v_offset: usize,
        out_offset: usize,
        ffn_offset: usize,
        norm_offset: usize,
        sequences: usize,
        tokens: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<()> {
        let rows = sequences * tokens;
        self.transformer_layer_forward_slots(
            input,
            Self::PRETRAIN_TEMP_F,
            q,
            k,
            v,
            attention,
            projected,
            residual,
            normalized,
            ffn_expanded,
            ffn_activated,
            ffn_contracted,
            ffn_residual,
            q_offset,
            k_offset,
            v_offset,
            out_offset,
            ffn_offset,
            norm_offset,
            sequences,
            tokens,
            hidden,
            heads,
        )?;
        self.arena
            .layer_norm_parameter_slice_backward_f32(
                ffn_residual,
                Self::PARAMETERS,
                norm_offset + 2 * hidden,
                norm_offset + 3 * hidden,
                output_gradient,
                Self::PRETRAIN_TEMP_A,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                ffn_activated,
                Self::PARAMETERS,
                ffn_offset + (hidden + 1) * 4 * hidden,
                ffn_offset + (hidden + 1) * 4 * hidden + 4 * hidden * hidden,
                Self::PRETRAIN_TEMP_A,
                Self::PRETRAIN_TEMP_B,
                Self::PARAMETER_GRADIENT,
                rows,
                4 * hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .relu_backward_f32(
                ffn_expanded,
                Self::PRETRAIN_TEMP_B,
                Self::PRETRAIN_TEMP_C,
                rows * 4 * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                normalized,
                Self::PARAMETERS,
                ffn_offset,
                ffn_offset + hidden * 4 * hidden,
                Self::PRETRAIN_TEMP_C,
                Self::PRETRAIN_TEMP_B,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                4 * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                Self::PRETRAIN_TEMP_A,
                Self::PRETRAIN_TEMP_B,
                Self::PRETRAIN_TEMP_C,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .layer_norm_parameter_slice_backward_f32(
                residual,
                Self::PARAMETERS,
                norm_offset,
                norm_offset + hidden,
                Self::PRETRAIN_TEMP_C,
                Self::PRETRAIN_TEMP_A,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                attention,
                Self::PARAMETERS,
                out_offset,
                out_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_A,
                Self::PRETRAIN_TEMP_B,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .attention_backward_f32(
                q,
                k,
                v,
                Self::PRETRAIN_TEMP_B,
                Self::PRETRAIN_TEMP_C,
                Self::PRETRAIN_TEMP_D,
                Self::PRETRAIN_TEMP_E,
                sequences,
                tokens,
                heads,
                hidden / heads,
                false,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                q_offset,
                q_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_C,
                input_gradient,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                k_offset,
                k_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_D,
                Self::PRETRAIN_TEMP_C,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                input_gradient,
                Self::PRETRAIN_TEMP_C,
                input_gradient,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                input,
                Self::PARAMETERS,
                v_offset,
                v_offset + hidden * hidden,
                Self::PRETRAIN_TEMP_E,
                Self::PRETRAIN_TEMP_C,
                Self::PARAMETER_GRADIENT,
                rows,
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                input_gradient,
                Self::PRETRAIN_TEMP_C,
                input_gradient,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_f32(
                input_gradient,
                Self::PRETRAIN_TEMP_A,
                input_gradient,
                rows * hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

}
