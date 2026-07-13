pub mod dataloader;
pub mod nbeats;
pub mod neural_panel;
pub mod nhits;
pub mod scaler;

pub use dataloader::{ForecastWindow, WindowDataset};
pub use nbeats::{NBeatsConfig, NBeatsForecaster};
pub use neural_panel::{
    fit_dense_regressor, ComponentMode, DenseRegressorConfig, LaneNeuralPanelConfig,
    LaneNeuralPanelForecaster, MlpState, NeuralPanelConfig, NeuralPanelForecaster, NeuralPanelLoss,
    NeuralPanelMode, NeuralPanelWindow, NeuralPanelWindowDataset, TrendMode,
};
pub use nhits::{NHiTSConfig, NHiTSForecaster};
pub use scaler::StandardScaler;

fn validate_window_config(
    input_size: usize,
    hidden_size: usize,
    epochs: usize,
) -> crate::Result<()> {
    if input_size == 0 {
        return Err(crate::NeuralError::InvalidArgument(
            "input_size must be positive".to_string(),
        ));
    }
    if hidden_size == 0 {
        return Err(crate::NeuralError::InvalidArgument(
            "hidden_size must be positive".to_string(),
        ));
    }
    if epochs == 0 {
        return Err(crate::NeuralError::InvalidArgument(
            "epochs must be positive".to_string(),
        ));
    }
    Ok(())
}
