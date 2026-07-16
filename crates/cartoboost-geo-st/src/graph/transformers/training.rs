#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaLsttnTensorExecutor {
    /// Starts a deterministic reduced-batch gradient accumulation and applies
    /// its single AdamW update. Every caller contributes into the same
    /// resident vector between these calls; checkpoints later copy only the
    /// portable parameter/moment vectors back to Rust.
    fn begin_supervised_gradient(&mut self) -> Result<()> {
        let len = self
            .arena
            .capacity_f32(Self::PARAMETERS)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .fill_f32(Self::PARAMETER_GRADIENT, len, 0.0)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn adamw_supervised_step(
        &mut self,
        step: u64,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let len = self
            .arena
            .capacity_f32(Self::PARAMETERS)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .clip_gradient_l2_f32(Self::PARAMETER_GRADIENT, Self::GRADIENT_NORM, len, 5.0)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .adamw_step_f32(
                Self::PARAMETERS,
                Self::FIRST_MOMENT,
                Self::SECOND_MOMENT,
                Self::PARAMETER_GRADIENT,
                len,
                step,
                learning_rate as f32,
                weight_decay as f32,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    /// Packs the real supervised contract without per-node host dispatch:
    /// inputs are row-major `[batch, lookback, nodes, channels]`, targets
    /// are `[batch, horizon, nodes]`. `windows` and `targets` retain the
    /// native graph-frame row representation at the API boundary only.
    fn upload_supervised_batch(
        &mut self,
        windows: &[&[Vec<f64>]],
        targets: &[&[Vec<f64>]],
        channels: usize,
    ) -> Result<()> {
        self.upload_supervised_node_tile(windows, targets, channels, 0..self.nodes)
    }

    /// Physical node-tile upload for the global logical batch.  Rows remain
    /// ordered as the original global node axis; the range is merely a
    /// checkpointing boundary and never changes graph/node identity.
    fn upload_supervised_node_tile(
        &mut self,
        windows: &[&[Vec<f64>]],
        targets: &[&[Vec<f64>]],
        channels: usize,
        node_range: std::ops::Range<usize>,
    ) -> Result<()> {
        if windows.is_empty()
            || windows.len() > 32
            || windows.len() != targets.len()
            || channels == 0
            || node_range.start >= node_range.end
            || node_range.end > self.nodes
        {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN supervised batches require 1..=32 matching windows and targets"
                    .to_string(),
            ));
        }
        let lookback = windows[0].len();
        let horizon = targets[0].len();
        if lookback == 0 || horizon == 0 {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN supervised windows and targets must be non-empty".to_string(),
            ));
        }
        let nodes = windows[0].first().map_or(0, |row| row.len() / channels);
        if nodes != self.nodes {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN input channels do not match the resident graph nodes".to_string(),
            ));
        }
        let finite_f32 = |value: f64| -> Result<f32> {
            let value = value as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(GeoStError::InvalidFrame(
                    "CUDA LSTTN batch contains a non-finite f32 value".to_string(),
                ))
            }
        };
        let tile_nodes = node_range.len();
        let mut input = Vec::with_capacity(windows.len() * lookback * tile_nodes * channels);
        let mut target = Vec::with_capacity(targets.len() * horizon * tile_nodes);
        for (window, target_rows) in windows.iter().zip(targets) {
            if window.len() != lookback
                || target_rows.len() != horizon
                || window.iter().any(|row| row.len() != nodes * channels)
                || target_rows.iter().any(|row| row.len() != nodes)
            {
                return Err(GeoStError::InvalidFrame(
                    "CUDA LSTTN batch has inconsistent tensor rows".to_string(),
                ));
            }
            for row in window.iter() {
                for node in node_range.clone() {
                    for channel in 0..channels {
                        input.push(finite_f32(row[node * channels + channel])?);
                    }
                }
            }
            for row in target_rows.iter() {
                for node in node_range.clone() {
                    target.push(finite_f32(row[node])?);
                }
            }
        }
        self.arena
            .upload_f32(Self::SUPERVISED_INPUT, &input)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .upload_f32(Self::SUPERVISED_TARGET, &target)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .reserve_f32(Self::DIRECT_OUTPUT, target.len())
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn upload_pretraining_window(&mut self, window: &[Vec<f64>], channels: usize) -> Result<()> {
        if window.is_empty() || channels == 0 || window.iter().any(|row| row.len() != self.nodes) {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN pretraining window must match the resident graph".to_string(),
            ));
        }
        let mut input = Vec::with_capacity(window.len() * self.nodes * channels);
        for row in window {
            for value in row {
                let value = *value as f32;
                if !value.is_finite() {
                    return Err(GeoStError::InvalidFrame(
                        "CUDA LSTTN pretraining window contains a non-finite f32 value".to_string(),
                    ));
                }
                input.push(value);
            }
        }
        self.arena
            .upload_f32(Self::SUPERVISED_INPUT, &input)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn pretraining_encoder_prefix(
        &mut self,
        layout: GraphParameterLayout,
        layers: usize,
        visible: usize,
        hidden: usize,
        heads: usize,
    ) -> Result<usize> {
        if layers == 0 {
            return Ok(Self::PRETRAIN_VISIBLE_TOKENS);
        }
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;
        let mut input = Self::PRETRAIN_VISIBLE_TOKENS;
        let mut output = Self::PRETRAIN_LAYER_A;
        for layer in 0..layers {
            self.transformer_layer_forward_slots(
                input,
                output,
                Self::ENCODER_Q,
                Self::ENCODER_K,
                Self::ENCODER_V,
                Self::ENCODER_ATTENTION,
                Self::ENCODER_PROJECTED,
                Self::ENCODER_RESIDUAL,
                Self::ENCODER_NORMALIZED,
                Self::ENCODER_FFN_EXPANDED,
                Self::ENCODER_FFN_ACTIVATED,
                Self::ENCODER_FFN_CONTRACTED,
                Self::ENCODER_FFN_RESIDUAL,
                layout.temporal_q + layer * projection,
                layout.temporal_k + layer * projection,
                layout.temporal_v + layer * projection,
                layout.lsttn_transformer_out + layer * projection,
                layout.lsttn_transformer_ffn + layer * ffn_width,
                layout.lsttn_transformer_norm + layer * 4 * hidden,
                self.nodes,
                visible,
                hidden,
                heads,
            )?;
            input = output;
            output = if output == Self::PRETRAIN_LAYER_A {
                Self::PRETRAIN_LAYER_B
            } else {
                Self::PRETRAIN_LAYER_A
            };
        }
        Ok(input)
    }

    fn cuda_train_masked_subseries_reconstruction(
        &mut self,
        state: &mut TrainableGraphTransformerState,
        window: &[Vec<f64>],
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<f64> {
        let patch_width = (state.periodicity / 24).max(1);
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
        let masked = masked_patch_indices(patches, state.steps);
        let visible = (0..patches)
            .filter(|patch| !masked.contains(patch))
            .collect::<Vec<_>>();
        let masked_u32 = masked
            .iter()
            .map(|patch| u32::try_from(*patch))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                GeoStError::InvalidFrame("CUDA LSTTN patch index exceeds u32".to_string())
            })?;
        let visible_u32 = visible
            .iter()
            .map(|patch| u32::try_from(*patch))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                GeoStError::InvalidFrame("CUDA LSTTN patch index exceeds u32".to_string())
            })?;
        let layout = state.layout();
        let hidden = state.hidden;
        let heads = state.attention_heads;
        let position_count = (layout.pretrain_decoder - layout.pretrain_position) / hidden;
        let scale = (hidden as f32).sqrt();
        let projection = hidden * (hidden + 1);
        let ffn_width = 8 * hidden * hidden + 5 * hidden;

        self.upload_pretraining_window(window, 1)?;
        self.arena
            .upload_u32(Self::VISIBLE_PATCH_INDICES, &visible_u32)
            .and_then(|_| {
                self.arena
                    .upload_u32(Self::MASKED_PATCH_INDICES, &masked_u32)
            })
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.begin_supervised_gradient()?;
        self.patch_embedding(layout, 1, window.len(), self.nodes, 1, patch_width, hidden)?;
        self.add_patch_positions(layout, 1, patches, self.nodes, hidden)?;
        self.patch_attention_layout(1, patches, self.nodes, hidden)?;
        self.arena
            .gather_patch_tokens_f32(
                Self::ATTENTION_SEQUENCE,
                Self::VISIBLE_PATCH_INDICES,
                Self::PRETRAIN_VISIBLE_TOKENS,
                1,
                self.nodes,
                patches,
                visible.len(),
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let encoded = self.pretraining_encoder_prefix(layout, 4, visible.len(), hidden, heads)?;
        self.affine_projection(
            encoded,
            Self::PRETRAIN_LAYER_GRAD_A,
            layout.lsttn_encoder_decoder,
            self.nodes * visible.len(),
            hidden,
            hidden,
        )?;
        self.arena
            .assemble_masked_decoder_tokens_f32(
                Self::PRETRAIN_LAYER_GRAD_A,
                Self::MASKED_PATCH_INDICES,
                Self::PARAMETERS,
                layout.pretrain_mask_token,
                layout.pretrain_position,
                Self::PRETRAIN_DECODER_INPUT,
                1,
                self.nodes,
                visible.len(),
                masked.len(),
                hidden,
                position_count,
                scale,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.transformer_layer_forward_slots(
            Self::PRETRAIN_DECODER_INPUT,
            Self::PRETRAIN_DECODER_OUTPUT,
            Self::PRETRAIN_DECODER_Q,
            Self::PRETRAIN_DECODER_K,
            Self::PRETRAIN_DECODER_V,
            Self::PRETRAIN_DECODER_ATTENTION,
            Self::PRETRAIN_DECODER_PROJECTED,
            Self::PRETRAIN_DECODER_RESIDUAL,
            Self::PRETRAIN_DECODER_NORMALIZED,
            Self::PRETRAIN_DECODER_FFN_EXPANDED,
            Self::PRETRAIN_DECODER_FFN_ACTIVATED,
            Self::PRETRAIN_DECODER_FFN_CONTRACTED,
            Self::PRETRAIN_DECODER_FFN_RESIDUAL,
            layout.lsttn_decoder_q,
            layout.lsttn_decoder_k,
            layout.lsttn_decoder_v,
            layout.lsttn_decoder_out,
            layout.lsttn_decoder_ffn,
            layout.lsttn_decoder_norm,
            self.nodes,
            visible.len() + masked.len(),
            hidden,
            heads,
        )?;
        self.arena
            .masked_patch_reconstruction_loss_backward_f32(
                Self::PRETRAIN_DECODER_OUTPUT,
                Self::SUPERVISED_INPUT,
                Self::MASKED_PATCH_INDICES,
                Self::PARAMETERS,
                layout.pretrain_decoder,
                layout.pretrain_decoder + patch_width * hidden,
                Self::PRETRAIN_CONTEXT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                Self::PRETRAIN_LOSS,
                1,
                window.len(),
                self.nodes,
                1,
                visible.len(),
                masked.len(),
                patch_width,
                hidden,
                state.normalized_zero as f32,
                state.target_scale as f32,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.transformer_layer_backward_slots(
            Self::PRETRAIN_DECODER_INPUT,
            Self::PRETRAIN_CONTEXT_GRADIENT,
            Self::PRETRAIN_DECODER_INPUT_GRADIENT,
            Self::PRETRAIN_DECODER_Q,
            Self::PRETRAIN_DECODER_K,
            Self::PRETRAIN_DECODER_V,
            Self::PRETRAIN_DECODER_ATTENTION,
            Self::PRETRAIN_DECODER_PROJECTED,
            Self::PRETRAIN_DECODER_RESIDUAL,
            Self::PRETRAIN_DECODER_NORMALIZED,
            Self::PRETRAIN_DECODER_FFN_EXPANDED,
            Self::PRETRAIN_DECODER_FFN_ACTIVATED,
            Self::PRETRAIN_DECODER_FFN_CONTRACTED,
            Self::PRETRAIN_DECODER_FFN_RESIDUAL,
            layout.lsttn_decoder_q,
            layout.lsttn_decoder_k,
            layout.lsttn_decoder_v,
            layout.lsttn_decoder_out,
            layout.lsttn_decoder_ffn,
            layout.lsttn_decoder_norm,
            self.nodes,
            visible.len() + masked.len(),
            hidden,
            heads,
        )?;
        self.arena
            .assemble_masked_decoder_tokens_backward_f32(
                Self::PRETRAIN_DECODER_INPUT_GRADIENT,
                Self::MASKED_PATCH_INDICES,
                Self::PARAMETER_GRADIENT,
                layout.pretrain_mask_token,
                layout.pretrain_position,
                Self::PRETRAIN_VISIBLE_GRADIENT,
                1,
                self.nodes,
                visible.len(),
                masked.len(),
                hidden,
                position_count,
                scale,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .affine_backward_parameter_slice_f32(
                encoded,
                Self::PARAMETERS,
                layout.lsttn_encoder_decoder,
                layout.lsttn_encoder_decoder + hidden * hidden,
                Self::PRETRAIN_VISIBLE_GRADIENT,
                Self::PRETRAIN_ENCODER_OUTPUT_GRADIENT,
                Self::PARAMETER_GRADIENT,
                self.nodes * visible.len(),
                hidden,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let mut output_gradient = Self::PRETRAIN_ENCODER_OUTPUT_GRADIENT;
        for layer in (0..4).rev() {
            let input =
                self.pretraining_encoder_prefix(layout, layer, visible.len(), hidden, heads)?;
            let input_gradient = if output_gradient == Self::PRETRAIN_LAYER_GRAD_A {
                Self::PRETRAIN_LAYER_GRAD_B
            } else {
                Self::PRETRAIN_LAYER_GRAD_A
            };
            self.transformer_layer_backward_slots(
                input,
                output_gradient,
                input_gradient,
                Self::ENCODER_Q,
                Self::ENCODER_K,
                Self::ENCODER_V,
                Self::ENCODER_ATTENTION,
                Self::ENCODER_PROJECTED,
                Self::ENCODER_RESIDUAL,
                Self::ENCODER_NORMALIZED,
                Self::ENCODER_FFN_EXPANDED,
                Self::ENCODER_FFN_ACTIVATED,
                Self::ENCODER_FFN_CONTRACTED,
                Self::ENCODER_FFN_RESIDUAL,
                layout.temporal_q + layer * projection,
                layout.temporal_k + layer * projection,
                layout.temporal_v + layer * projection,
                layout.lsttn_transformer_out + layer * projection,
                layout.lsttn_transformer_ffn + layer * ffn_width,
                layout.lsttn_transformer_norm + layer * 4 * hidden,
                self.nodes,
                visible.len(),
                hidden,
                heads,
            )?;
            output_gradient = input_gradient;
        }
        self.arena
            .gather_patch_tokens_backward_f32(
                output_gradient,
                Self::VISIBLE_PATCH_INDICES,
                Self::PRETRAIN_SEQUENCE_GRADIENT,
                1,
                self.nodes,
                patches,
                visible.len(),
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .attention_sequences_to_patches_f32(
                Self::PRETRAIN_SEQUENCE_GRADIENT,
                Self::PRETRAIN_PATCH_LAYOUT_GRADIENT,
                1,
                patches,
                self.nodes,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .add_patch_positions_backward_f32(
                Self::PRETRAIN_PATCH_LAYOUT_GRADIENT,
                Self::PRETRAIN_PATCH_EMBEDDING_GRADIENT,
                Self::PARAMETER_GRADIENT,
                layout.pretrain_position,
                1,
                patches,
                self.nodes,
                hidden,
                scale,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .patch_embedding_parameter_slice_backward_f32(
                Self::SUPERVISED_INPUT,
                Self::PRETRAIN_PATCH_EMBEDDING_GRADIENT,
                Self::PARAMETER_GRADIENT,
                layout.lsttn_patch_embedding,
                layout.lsttn_patch_embedding + patch_width * hidden,
                1,
                window.len(),
                self.nodes,
                1,
                patch_width,
                hidden,
            )
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        let next_step = state.steps + 1;
        self.adamw_supervised_step(next_step, learning_rate, weight_decay)?;
        state.steps = next_step;
        let mut loss = [0.0_f32; 2];
        self.arena
            .download_f32(Self::PRETRAIN_LOSS, &mut loss)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.synchronize_portable_state(state)?;
        Ok(f64::from(loss[0]))
    }

}
