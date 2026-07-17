pub mod backend;
#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
mod cuda_oxide_backend;
#[cfg(all(feature = "directml", target_os = "windows"))]
mod directml_backend;
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
mod hip_backend;
mod metal_backend;
#[cfg(feature = "webgpu")]
mod webgpu_backend;

#[derive(Debug, thiserror::Error)]
pub enum AcceleratorError {
    #[error("{0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, AcceleratorError>;

pub use backend::*;
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub use webgpu_backend::{
    adamw_f32_async as webgpu_adamw_f32_async,
    affine_scores_f32_async as webgpu_affine_scores_f32_async,
    csr_diffusion_backward_f32_async as webgpu_csr_diffusion_backward_f32_async,
    csr_diffusion_f32_async as webgpu_csr_diffusion_f32_async,
    csr_row_softmax_backward_f32_async as webgpu_csr_row_softmax_backward_f32_async,
    csr_row_softmax_f32_async as webgpu_csr_row_softmax_f32_async,
    dense_layer_f32_async as webgpu_dense_layer_f32_async,
    dispatch_report_async as webgpu_dispatch_report_async,
    layer_norm_f32_async as webgpu_layer_norm_f32_async,
    pair_sigmoid_scores_f32_async as webgpu_pair_sigmoid_scores_f32_async,
    pairwise_squared_distances_f32_async as webgpu_pairwise_squared_distances_f32_async,
    scalar_graph_f32_async as webgpu_scalar_graph_f32_async,
    scalar_graph_train_step_f32_async as webgpu_scalar_graph_train_step_f32_async,
    train_tanh_mlp_f32_async as webgpu_train_tanh_mlp_f32_async,
};
