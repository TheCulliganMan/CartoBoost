#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaLsttnTensorExecutor {
    /// Copies only portable f32 optimizer/model state back into the native
    /// checkpoint representation. CUDA buffers and contexts remain owned by
    /// the executor and are recreated from this state on resume.
    fn synchronize_portable_state(&self, state: &mut TrainableGraphTransformerState) -> Result<()> {
        if state.parameters.len() != state.first_moment.len()
            || state.parameters.len() != state.second_moment.len()
        {
            return Err(GeoStError::InvalidFrame(
                "CUDA LSTTN checkpoint state has inconsistent optimizer lengths".to_string(),
            ));
        }
        let len = state.parameters.len();
        let mut parameters = vec![0.0_f32; len];
        let mut first = vec![0.0_f32; len];
        let mut second = vec![0.0_f32; len];
        self.arena
            .download_f32(Self::PARAMETERS, &mut parameters)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .download_f32(Self::FIRST_MOMENT, &mut first)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        self.arena
            .download_f32(Self::SECOND_MOMENT, &mut second)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        state.parameters = parameters.into_iter().map(f64::from).collect();
        state.first_moment = first.into_iter().map(f64::from).collect();
        state.second_moment = second.into_iter().map(f64::from).collect();
        Ok(())
    }

    /// LSTTN pretrains its masked-subseries transformer, then keeps that
    /// encoder frozen during direct forecasting.  The CUDA executor owns the
    /// gradient buffer, so enforce the same parameter contract there before
    /// AdamW rather than allowing a CUDA-only fine-tuning variant.
    fn freeze_pretrained_transformer_gradients(
        &mut self,
        state: &TrainableGraphTransformerState,
    ) -> Result<()> {
        let layout = state.layout();
        let mut gradients = vec![0.0_f32; state.parameters.len()];
        self.arena
            .download_f32(Self::PARAMETER_GRADIENT, &mut gradients)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        gradients[layout.input..layout.spatial_q].fill(0.0);
        gradients[layout.pretrain_mask_token..layout.total].fill(0.0);
        self.arena
            .upload_f32(Self::PARAMETER_GRADIENT, &gradients)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))
    }

    fn mean_supervised_loss(&self) -> Result<f64> {
        // The CUDA loss kernel reduces all examples, horizons, and nodes into
        // one MAE scalar (plus an internal valid-value count).
        let mut losses = [0.0_f32; 2];
        self.arena
            .download_f32(Self::SUPERVISED_LOSS, &mut losses)
            .map_err(|error| GeoStError::InvalidBackend(error.to_string()))?;
        Ok(f64::from(losses[0]))
    }
}

