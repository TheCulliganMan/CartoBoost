use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
use std::sync::OnceLock;
use std::time::Instant;

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
pub use crate::cuda_oxide_backend::{CudaCsrDiffusionWorkspace, CudaTensorArena};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    Auto,
    Cpu,
    Cuda,
    Directml,
    Rocm,
    Metal,
    Webgpu,
}

impl ComputeBackend {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "directml" | "dml" => Ok(Self::Directml),
            "hip" | "rocm" => Ok(Self::Rocm),
            "metal" => Ok(Self::Metal),
            "webgpu" => Ok(Self::Webgpu),
            other => Err(NeuralError::InvalidArgument(format!(
                "unknown compute backend {other:?}; expected auto, cpu, cuda, directml, hip/rocm, metal, or webgpu"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Directml => "directml",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Webgpu => "webgpu",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendSelection {
    pub requested: String,
    pub selected: String,
    pub available: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BackendDispatchReport {
    pub requested: String,
    pub selected: String,
    pub operation: String,
    pub len: usize,
    pub checksum: f64,
    pub expected_checksum: f64,
    pub elapsed_ms: f64,
    pub accelerated: bool,
}

/// Gradients for a batched CSR diffusion. `input_grad` has the same layout as
/// the input (`[batch, nodes, channels]`) and `edge_grad` follows CSR edge
/// order. Keeping these buffers explicit is the tensor backend's alternative
/// to building one scalar-autodiff node per multiply/add.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrDiffusionBackward {
    pub input_grad: Vec<f32>,
    pub edge_grad: Vec<f32>,
}

impl Default for BackendSelection {
    fn default() -> Self {
        Self {
            requested: "auto".to_string(),
            selected: "cpu".to_string(),
            available: available_backends(),
        }
    }
}

pub fn select_backend(requested: Option<&str>) -> Result<BackendSelection> {
    let requested_backend = ComputeBackend::parse(requested)?;
    let available = available_backends();
    let selected = match requested_backend {
        ComputeBackend::Auto => "cpu".to_string(),
        ComputeBackend::Cpu => "cpu".to_string(),
        ComputeBackend::Cuda
        | ComputeBackend::Directml
        | ComputeBackend::Rocm
        | ComputeBackend::Metal
        | ComputeBackend::Webgpu => {
            let name = requested_backend.as_str();
            if !available.iter().any(|candidate| candidate == name) {
                return Err(NeuralError::InvalidArgument(format!(
                    "requested compute backend {name:?} is not available in this build; available backends: {}",
                    available.join(", ")
                )));
            }
            name.to_string()
        }
    };
    Ok(BackendSelection {
        requested: requested_backend.as_str().to_string(),
        selected,
        available,
    })
}

#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
static ROCM_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_available() -> bool {
    crate::cuda_oxide_backend::is_available()
}

#[cfg(all(
    feature = "cuda",
    not(all(feature = "cuda-oxide", target_os = "linux")),
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_available() -> bool {
    false
}

#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
fn rocm_available() -> bool {
    *ROCM_AVAILABLE.get_or_init(crate::hip_backend::is_available)
}

pub fn available_backends() -> Vec<String> {
    let mut backends = vec!["cpu".to_string()];
    if crate::metal_backend::is_available() {
        backends.push("metal".to_string());
    }
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    if cuda_available() {
        backends.push("cuda".to_string());
    }
    #[cfg(all(feature = "directml", target_os = "windows"))]
    if crate::directml_backend::is_available() {
        backends.push("directml".to_string());
    }
    #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
    if rocm_available() {
        backends.push("rocm".to_string());
    }
    #[cfg(feature = "webgpu")]
    if crate::webgpu_backend::is_available() {
        backends.push("webgpu".to_string());
    }
    backends
}

pub fn backend_dispatch_report(
    requested: Option<&str>,
    len: usize,
) -> Result<BackendDispatchReport> {
    let selection = select_backend(requested)?;
    let len = len.max(1);
    match selection.selected.as_str() {
        "cpu" => cpu_vector_add_report(selection, len),
        "cuda" => cuda_vector_add_report(selection, len),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::vector_add_report(
            selection,
            len,
            expected_vector_add_checksum(len),
        ),
        "metal" => crate::metal_backend::metal_vector_add_report(selection, len),
        "rocm" => rocm_vector_add_report(selection, len),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::vector_add_report(selection, len),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} is selectable but does not have a verified dispatch kernel yet"
        ))),
    }
}

pub fn backend_affine_scores(
    selection: &BackendSelection,
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    validate_affine_inputs(features, means, weights, intercepts)?;
    match selection.selected.as_str() {
        "cpu" => cpu_affine_scores(features, means, weights, intercepts),
        "cuda" => cuda_affine_scores(features, means, weights, intercepts),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::affine_scores(features, means, weights, intercepts),
        "metal" => crate::metal_backend::metal_affine_scores(features, means, weights, intercepts),
        "rocm" => rocm_affine_scores(features, means, weights, intercepts),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::affine_scores(features, means, weights, intercepts),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} is selectable but does not have a verified affine scoring kernel yet"
        ))),
    }
}

/// Applies a directed CSR matrix to a contiguous batch of node features.
/// Input and output are row-major `[batch, nodes, channels]`. Empty rows are
/// deliberately zero, which gives isolated nodes a stable, explicit result.
pub fn backend_csr_diffusion_f32(
    selection: &BackendSelection,
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    validate_csr_diffusion_inputs(indptr, indices, weights, channels, values)?;
    match selection.selected.as_str() {
        "cpu" => cpu_csr_diffusion_f32(indptr, indices, weights, channels, values),
        "cuda" => cuda_csr_diffusion_f32(indptr, indices, weights, channels, values),
        #[cfg(all(
            feature = "metal",
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )
        ))]
        "metal" => crate::metal_backend::csr_diffusion(indptr, indices, weights, channels, values),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => {
            crate::directml_backend::csr_diffusion(indptr, indices, weights, channels, values)
        }
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::csr_diffusion(indptr, indices, weights, channels, values),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => {
            crate::webgpu_backend::csr_diffusion(indptr, indices, weights, channels, values)
        }
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide the CUDA LSTTN CSR diffusion kernel"
        ))),
    }
}

/// Backpropagates through `backend_csr_diffusion_f32`. The edge gradient is
/// useful for both learned edge weights and adaptive adjacency softmax
/// backward; callers apply the softmax Jacobian on the same CSR edge order.
pub fn backend_csr_diffusion_backward_f32(
    selection: &BackendSelection,
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    validate_csr_diffusion_inputs(indptr, indices, weights, channels, values)?;
    if output_grad.len() != values.len() || output_grad.iter().any(|value| !value.is_finite()) {
        return Err(NeuralError::InvalidArgument(
            "CSR diffusion output gradient must be finite and match input shape".to_string(),
        ));
    }
    match selection.selected.as_str() {
        "cpu" => {
            cpu_csr_diffusion_backward_f32(indptr, indices, weights, channels, values, output_grad)
        }
        "cuda" => {
            cuda_csr_diffusion_backward_f32(indptr, indices, weights, channels, values, output_grad)
        }
        #[cfg(all(
            feature = "metal",
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )
        ))]
        "metal" => crate::metal_backend::csr_diffusion_backward(
            indptr,
            indices,
            weights,
            channels,
            values,
            output_grad,
        ),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::csr_diffusion_backward(
            indptr,
            indices,
            weights,
            channels,
            values,
            output_grad,
        ),
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::csr_diffusion_backward(
            indptr,
            indices,
            weights,
            channels,
            values,
            output_grad,
        ),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::csr_diffusion_backward(
            indptr,
            indices,
            weights,
            channels,
            values,
            output_grad,
        ),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide the CUDA LSTTN CSR diffusion backward kernel"
        ))),
    }
}

/// Row-wise softmax over CSR edges. Each row is normalized independently and
/// an empty row contributes no values, preserving sparse isolated-node
/// semantics without a synthetic dense row.
pub fn backend_csr_row_softmax_f32(
    selection: &BackendSelection,
    indptr: &[u32],
    logits: &[f32],
) -> Result<Vec<f32>> {
    validate_csr_row_inputs(indptr, logits)?;
    match selection.selected.as_str() {
        "cpu" => cpu_csr_row_softmax_f32(indptr, logits),
        "cuda" => cuda_csr_row_softmax_f32(indptr, logits),
        #[cfg(all(
            feature = "metal",
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )
        ))]
        "metal" => crate::metal_backend::csr_row_softmax(indptr, logits),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::csr_row_softmax(indptr, logits),
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::csr_row_softmax(indptr, logits),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::csr_row_softmax(indptr, logits),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide the CUDA LSTTN CSR row-softmax kernel"
        ))),
    }
}

/// Backpropagates a CSR row-softmax. `output_grad` is the gradient with
/// respect to the normalized edge weights and the return is the gradient with
/// respect to the corresponding row logits.
pub fn backend_csr_row_softmax_backward_f32(
    selection: &BackendSelection,
    indptr: &[u32],
    weights: &[f32],
    output_grad: &[f32],
) -> Result<Vec<f32>> {
    validate_csr_row_inputs(indptr, weights)?;
    if output_grad.len() != weights.len() || output_grad.iter().any(|value| !value.is_finite()) {
        return Err(NeuralError::InvalidArgument(
            "CSR row-softmax output gradient must be finite and match edge shape".to_string(),
        ));
    }
    match selection.selected.as_str() {
        "cpu" => cpu_csr_row_softmax_backward_f32(indptr, weights, output_grad),
        "cuda" => cuda_csr_row_softmax_backward_f32(indptr, weights, output_grad),
        #[cfg(all(
            feature = "metal",
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )
        ))]
        "metal" => crate::metal_backend::csr_row_softmax_backward(indptr, weights, output_grad),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => {
            crate::directml_backend::csr_row_softmax_backward(indptr, weights, output_grad)
        }
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::csr_row_softmax_backward(indptr, weights, output_grad),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::csr_row_softmax_backward(indptr, weights, output_grad),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide the CUDA LSTTN CSR row-softmax backward kernel"
        ))),
    }
}

/// Applies one decoupled AdamW update in-place. Parameter and moment buffers
/// are contiguous tensor storage so a CUDA batch needs no scalar autograd
/// graph or host-side per-parameter update loop.
#[allow(clippy::too_many_arguments)]
pub fn backend_adamw_step_f32(
    selection: &BackendSelection,
    parameters: &mut [f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    gradients: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    if parameters.is_empty()
        || first_moment.len() != parameters.len()
        || second_moment.len() != parameters.len()
        || gradients.len() != parameters.len()
        || step == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || !weight_decay.is_finite()
        || parameters
            .iter()
            .chain(first_moment.iter())
            .chain(second_moment.iter())
            .chain(gradients)
            .any(|v| !v.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "invalid AdamW tensor state".to_string(),
        ));
    }
    match selection.selected.as_str() {
        "cpu" => cpu_adamw_step_f32(
            parameters,
            first_moment,
            second_moment,
            gradients,
            step,
            learning_rate,
            weight_decay,
        ),
        "cuda" => cuda_adamw_step_f32(
            parameters,
            first_moment,
            second_moment,
            gradients,
            step,
            learning_rate,
            weight_decay,
        ),
        #[cfg(all(
            feature = "metal",
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )
        ))]
        "metal" => crate::metal_backend::adamw(
            parameters,
            first_moment,
            second_moment,
            gradients,
            step,
            learning_rate,
            weight_decay,
        ),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::adamw(
            parameters,
            first_moment,
            second_moment,
            gradients,
            step,
            learning_rate,
            weight_decay,
        ),
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::adamw(
            parameters,
            first_moment,
            second_moment,
            gradients,
            step,
            learning_rate,
            weight_decay,
        ),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::adamw(
            parameters,
            first_moment,
            second_moment,
            gradients,
            step,
            learning_rate,
            weight_decay,
        ),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide the CUDA LSTTN AdamW kernel"
        ))),
    }
}

/// Applies affine layer normalization to contiguous `[rows, width]` tensors.
/// This is the normalization used around LSTTN Transformer attention and FFN
/// residuals; epsilon is fixed to the model's numerically stable `1e-5`.
pub fn backend_layer_norm_f32(
    selection: &BackendSelection,
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    if rows == 0
        || width == 0
        || values.len() != rows * width
        || gamma.len() != width
        || beta.len() != width
        || values
            .iter()
            .chain(gamma)
            .chain(beta)
            .any(|v| !v.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "invalid layer-normalization tensor shape or values".to_string(),
        ));
    }
    match selection.selected.as_str() {
        "cpu" => cpu_layer_norm_f32(values, rows, width, gamma, beta),
        "cuda" => cuda_layer_norm_f32(values, rows, width, gamma, beta),
        #[cfg(all(
            feature = "metal",
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "tvos",
                target_os = "visionos"
            )
        ))]
        "metal" => crate::metal_backend::layer_norm(values, rows, width, gamma, beta),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::layer_norm(values, rows, width, gamma, beta),
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::layer_norm(values, rows, width, gamma, beta),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::layer_norm(values, rows, width, gamma, beta),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide the CUDA LSTTN layer-normalization kernel"
        ))),
    }
}

/// Computes LSTTN's zero-masked inverse-scale MAE and its prediction gradient
/// for contiguous forecast tensors. Targets equal to `masked_zero` do not
/// contribute to either value or gradient.
pub fn masked_inverse_scale_mae_f32(
    predictions: &[f32],
    targets: &[f32],
    masked_zero: f32,
    target_scale: f32,
) -> Result<(f32, Vec<f32>)> {
    if predictions.len() != targets.len()
        || predictions.is_empty()
        || !masked_zero.is_finite()
        || !target_scale.is_finite()
        || target_scale <= 0.0
        || predictions.iter().chain(targets).any(|v| !v.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "invalid masked MAE tensors".to_string(),
        ));
    }
    let valid = targets
        .iter()
        .filter(|target| (**target - masked_zero).abs() > 1.0e-12)
        .count();
    let mut gradient = vec![0.0; predictions.len()];
    if valid == 0 {
        return Ok((0.0, gradient));
    }
    let scale = target_scale / valid as f32;
    let mut loss = 0.0;
    for index in 0..predictions.len() {
        if (targets[index] - masked_zero).abs() > 1.0e-12 {
            let residual = predictions[index] - targets[index];
            loss += residual.abs() * scale;
            gradient[index] = if residual > 0.0 {
                scale
            } else if residual < 0.0 {
                -scale
            } else {
                0.0
            };
        }
    }
    Ok((loss, gradient))
}

fn cpu_layer_norm_f32(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        let input = &values[row * width..(row + 1) * width];
        let mean = input.iter().sum::<f32>() / width as f32;
        let variance = input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        for col in 0..width {
            output[row * width + col] =
                (input[col] - mean) / (variance + 1.0e-5).sqrt() * gamma[col] + beta[col];
        }
    }
    Ok(output)
}

fn cpu_adamw_step_f32(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    gradients: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    let correction_first = 1.0 - 0.9_f32.powi(step as i32);
    let correction_second = 1.0 - 0.999_f32.powi(step as i32);
    for index in 0..parameters.len() {
        let gradient = gradients[index] + weight_decay * parameters[index];
        first[index] = 0.9 * first[index] + 0.1 * gradient;
        second[index] = 0.999 * second[index] + 0.001 * gradient * gradient;
        parameters[index] -= learning_rate * (first[index] / correction_first)
            / ((second[index] / correction_second).sqrt() + 1.0e-8);
    }
    Ok(())
}

fn validate_csr_row_inputs(indptr: &[u32], values: &[f32]) -> Result<()> {
    if indptr.len() < 2
        || indptr[0] != 0
        || indptr.last().copied() != Some(values.len() as u32)
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "invalid CSR row tensor inputs".to_string(),
        ));
    }
    Ok(())
}

fn cpu_csr_row_softmax_f32(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    let mut weights = vec![0.0; logits.len()];
    for row in 0..indptr.len() - 1 {
        let range = indptr[row] as usize..indptr[row + 1] as usize;
        if range.is_empty() {
            continue;
        }
        let maximum = logits[range.clone()]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = range
            .clone()
            .map(|edge| (logits[edge] - maximum).exp())
            .sum::<f32>();
        for edge in range {
            weights[edge] = (logits[edge] - maximum).exp() / denominator;
        }
    }
    Ok(weights)
}

fn cpu_csr_row_softmax_backward_f32(
    indptr: &[u32],
    weights: &[f32],
    output_grad: &[f32],
) -> Result<Vec<f32>> {
    let mut logits_grad = vec![0.0; weights.len()];
    for row in 0..indptr.len() - 1 {
        let range = indptr[row] as usize..indptr[row + 1] as usize;
        let dot = range
            .clone()
            .map(|edge| weights[edge] * output_grad[edge])
            .sum::<f32>();
        for edge in range {
            logits_grad[edge] = weights[edge] * (output_grad[edge] - dot);
        }
    }
    Ok(logits_grad)
}

fn validate_csr_diffusion_inputs(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<()> {
    if indptr.len() < 2
        || indptr[0] != 0
        || indptr.last().copied() != Some(indices.len() as u32)
        || indices.len() != weights.len()
        || channels == 0
        || !values.len().is_multiple_of((indptr.len() - 1) * channels)
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
        || indices
            .iter()
            .any(|index| *index as usize >= indptr.len() - 1)
        || weights
            .iter()
            .chain(values.iter())
            .any(|value| !value.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "invalid contiguous CSR diffusion tensor inputs".to_string(),
        ));
    }
    Ok(())
}

fn cpu_csr_diffusion_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    let nodes = indptr.len() - 1;
    let mut output = vec![0.0; values.len()];
    for batch in 0..values.len() / (nodes * channels) {
        for row in 0..nodes {
            for edge in indptr[row] as usize..indptr[row + 1] as usize {
                let source = indices[edge] as usize;
                for channel in 0..channels {
                    output[(batch * nodes + row) * channels + channel] +=
                        weights[edge] * values[(batch * nodes + source) * channels + channel];
                }
            }
        }
    }
    Ok(output)
}

fn cpu_csr_diffusion_backward_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    let nodes = indptr.len() - 1;
    let mut input_grad = vec![0.0; values.len()];
    let mut edge_grad = vec![0.0; weights.len()];
    for batch in 0..values.len() / (nodes * channels) {
        for row in 0..nodes {
            for edge in indptr[row] as usize..indptr[row + 1] as usize {
                let source = indices[edge] as usize;
                for channel in 0..channels {
                    let gradient = output_grad[(batch * nodes + row) * channels + channel];
                    input_grad[(batch * nodes + source) * channels + channel] +=
                        weights[edge] * gradient;
                    edge_grad[edge] +=
                        gradient * values[(batch * nodes + source) * channels + channel];
                }
            }
        }
    }
    Ok(CsrDiffusionBackward {
        input_grad,
        edge_grad,
    })
}

/// Evaluates a topologically ordered scalar computation graph on the selected
/// accelerator. Leaf nodes use `initial_values`; every other node is described
/// by an opcode and up to two earlier node indices. This lets model crates keep
/// graph construction and validation in Rust while executing the complete
/// numeric inference graph on the device.
pub fn backend_scalar_graph_f32(
    selection: &BackendSelection,
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    validate_scalar_graph_inputs(initial_values, opcodes, left, right)?;
    match selection.selected.as_str() {
        "metal" => crate::metal_backend::with_metal_autoreleasepool(|| {
            crate::metal_backend::metal_scalar_graph_f32(initial_values, opcodes, left, right)
        }),
        #[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
        "cuda" => cuda_scalar_graph_f32(initial_values, opcodes, left, right),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::scalar_graph(initial_values, opcodes, left, right),
        #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
        "rocm" => crate::hip_backend::scalar_graph(initial_values, opcodes, left, right),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::scalar_graph(initial_values, opcodes, left, right),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide a complete scalar-graph inference kernel"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backend_scalar_graph_train_step_f32(
    selection: &BackendSelection,
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
    parameter_ids: &[u32],
    loss: usize,
    parameters: &mut [f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<f32> {
    validate_scalar_graph_inputs(initial_values, opcodes, left, right)?;
    if parameter_ids.len() != opcodes.len()
        || loss >= opcodes.len()
        || parameters.is_empty()
        || first_moment.len() != parameters.len()
        || second_moment.len() != parameters.len()
        || step == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || !weight_decay.is_finite()
    {
        return Err(NeuralError::InvalidArgument(
            "scalar-graph training state or optimizer configuration is invalid".to_string(),
        ));
    }
    if parameter_ids
        .iter()
        .enumerate()
        .any(|(index, parameter)| opcodes[index] == 1 && (*parameter as usize) >= parameters.len())
        || parameters
            .iter()
            .chain(first_moment.iter())
            .chain(second_moment.iter())
            .any(|value| !value.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "scalar-graph training parameters must be finite and correctly indexed".to_string(),
        ));
    }
    match selection.selected.as_str() {
        "metal" => crate::metal_backend::with_metal_autoreleasepool(|| {
            crate::metal_backend::metal_scalar_graph_train_step_f32(
                initial_values,
                opcodes,
                left,
                right,
                parameter_ids,
                loss,
                parameters,
                first_moment,
                second_moment,
                step,
                learning_rate,
                weight_decay,
            )
        }),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::scalar_graph_train_step(
            initial_values,
            opcodes,
            left,
            right,
            parameter_ids,
            loss,
            parameters,
            first_moment,
            second_moment,
            step,
            learning_rate,
            weight_decay,
        ),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide complete scalar-graph training"
        ))),
    }
}

fn validate_scalar_graph_inputs(
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<()> {
    let len = initial_values.len();
    if len == 0 || opcodes.len() != len || left.len() != len || right.len() != len {
        return Err(NeuralError::InvalidArgument(
            "scalar graph arrays must be non-empty and have identical lengths".to_string(),
        ));
    }
    for index in 0..len {
        let opcode = opcodes[index];
        if opcode > 11 {
            return Err(NeuralError::InvalidArgument(
                "scalar graph contains an unknown opcode".to_string(),
            ));
        }
        if opcode >= 2 && left[index] as usize >= index {
            return Err(NeuralError::InvalidArgument(
                "scalar graph unary dependency must precede its output".to_string(),
            ));
        }
        if matches!(opcode, 2 | 3 | 4 | 10) && right[index] as usize >= index {
            return Err(NeuralError::InvalidArgument(
                "scalar graph binary dependency must precede its output".to_string(),
            ));
        }
        if opcode <= 1 && !initial_values[index].is_finite() {
            return Err(NeuralError::InvalidArgument(
                "scalar graph leaves must be finite".to_string(),
            ));
        }
    }
    Ok(())
}

pub fn backend_dense_layer_f32(
    selection: &BackendSelection,
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    validate_dense_layer_inputs(features, weights, biases)?;
    match selection.selected.as_str() {
        "cpu" => cpu_dense_layer_f32(features, weights, biases),
        "cuda" => cuda_dense_layer_f32(features, weights, biases),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::dense_layer(features, weights, biases),
        "metal" => crate::metal_backend::metal_dense_layer_f32(features, weights, biases),
        "rocm" => rocm_dense_layer_f32(features, weights, biases),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::dense_layer_f32(features, weights, biases),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} is selectable but does not have a verified dense layer kernel yet"
        ))),
    }
}

pub fn backend_pair_sigmoid_scores_f32(
    selection: &BackendSelection,
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    validate_pair_sigmoid_inputs(embeddings, pairs)?;
    match selection.selected.as_str() {
        "cpu" => cpu_pair_sigmoid_scores_f32(embeddings, pairs),
        "cuda" => cuda_pair_sigmoid_scores_f32(embeddings, pairs),
        #[cfg(all(feature = "directml", target_os = "windows"))]
        "directml" => crate::directml_backend::pair_sigmoid_scores(embeddings, pairs),
        "metal" => crate::metal_backend::metal_pair_sigmoid_scores_f32(embeddings, pairs),
        "rocm" => rocm_pair_sigmoid_scores_f32(embeddings, pairs),
        #[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
        "webgpu" => crate::webgpu_backend::pair_sigmoid_scores_f32(embeddings, pairs),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} is selectable but does not have a verified pair scoring kernel yet"
        ))),
    }
}

/// Trains the single-hidden-layer tanh MLP used by the N-BEATS and N-HiTS
/// experts.  Accelerator implementations keep the complete parameter vector
/// resident on the selected device for the training loop; CPU callers retain
/// the model's existing deterministic SGD implementation.
pub fn backend_train_tanh_mlp_f32(
    selection: &BackendSelection,
    inputs: &[Vec<f32>],
    targets: &[f32],
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    parameters: &mut [f32],
) -> Result<()> {
    validate_tanh_mlp_training_inputs(
        inputs,
        targets,
        hidden_size,
        epochs,
        learning_rate,
        parameters,
    )?;
    match selection.selected.as_str() {
        "metal" => crate::metal_backend::metal_train_tanh_mlp_f32(
            inputs,
            targets,
            hidden_size,
            epochs,
            learning_rate,
            parameters,
        ),
        "cuda" => cuda_train_tanh_mlp_f32(
            inputs,
            targets,
            hidden_size,
            epochs,
            learning_rate,
            parameters,
        ),
        "rocm" => rocm_train_tanh_mlp_f32(
            inputs,
            targets,
            hidden_size,
            epochs,
            learning_rate,
            parameters,
        ),
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide an accelerated tanh-MLP training kernel"
        ))),
    }
}

fn validate_tanh_mlp_training_inputs(
    inputs: &[Vec<f32>],
    targets: &[f32],
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    parameters: &[f32],
) -> Result<()> {
    if inputs.is_empty() || inputs.len() != targets.len() {
        return Err(NeuralError::InvalidArgument(
            "tanh-MLP training inputs and targets must be non-empty and aligned".to_string(),
        ));
    }
    let input_size = inputs[0].len();
    if input_size == 0
        || hidden_size == 0
        || epochs == 0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err(NeuralError::InvalidArgument(
            "tanh-MLP input width, hidden width, epochs, and learning rate must be positive"
                .to_string(),
        ));
    }
    if inputs
        .iter()
        .any(|row| row.len() != input_size || row.iter().any(|value| !value.is_finite()))
        || targets.iter().any(|value| !value.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "tanh-MLP training inputs and targets must be finite and rectangular".to_string(),
        ));
    }
    let expected = hidden_size * input_size + hidden_size + hidden_size + 1;
    if parameters.len() != expected || parameters.iter().any(|value| !value.is_finite()) {
        return Err(NeuralError::InvalidArgument(
            "tanh-MLP parameter vector has an invalid shape or non-finite value".to_string(),
        ));
    }
    Ok(())
}

fn validate_pair_sigmoid_inputs(embeddings: &[Vec<f32>], pairs: &[(usize, usize)]) -> Result<()> {
    if embeddings.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "pair scoring embeddings cannot be empty".to_string(),
        ));
    }
    if pairs.is_empty() {
        return Ok(());
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(NeuralError::InvalidArgument(
            "pair scoring embeddings must have nonzero width".to_string(),
        ));
    }
    if embeddings
        .iter()
        .any(|row| row.len() != dim || row.iter().any(|value| !value.is_finite()))
    {
        return Err(NeuralError::InvalidArgument(
            "pair scoring embeddings must be finite and rectangular".to_string(),
        ));
    }
    if pairs
        .iter()
        .any(|&(source, target)| source >= embeddings.len() || target >= embeddings.len())
    {
        return Err(NeuralError::InvalidArgument(
            "pair scoring node ids must be within the embedding table".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_dense_layer_inputs(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<()> {
    if features.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "dense layer features cannot be empty".to_string(),
        ));
    }
    let cols = features[0].len();
    if cols == 0 || biases.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "dense layer features and biases must be non-empty".to_string(),
        ));
    }
    if weights.len() != cols * biases.len() {
        return Err(NeuralError::InvalidArgument(
            "dense layer weights length must equal input width times output width".to_string(),
        ));
    }
    if features
        .iter()
        .any(|row| row.len() != cols || row.iter().any(|value| !value.is_finite()))
        || weights.iter().any(|value| !value.is_finite())
        || biases.iter().any(|value| !value.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "dense layer inputs must be finite and rectangular".to_string(),
        ));
    }
    Ok(())
}

fn validate_affine_inputs(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<()> {
    if features.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "affine score features cannot be empty".to_string(),
        ));
    }
    if means.len() != weights.len() || means.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "affine score means and weights must have the same nonzero width".to_string(),
        ));
    }
    if intercepts.len() != features.len() {
        return Err(NeuralError::InvalidArgument(
            "affine score intercepts length must match row count".to_string(),
        ));
    }
    if means
        .iter()
        .chain(weights)
        .chain(intercepts)
        .any(|value| !value.is_finite())
    {
        return Err(NeuralError::InvalidArgument(
            "affine score parameters must be finite".to_string(),
        ));
    }
    if features
        .iter()
        .any(|row| row.len() != weights.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(NeuralError::InvalidArgument(
            "affine score feature rows must be finite and match parameter width".to_string(),
        ));
    }
    Ok(())
}

fn cpu_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    Ok(features
        .iter()
        .zip(intercepts)
        .map(|(row, &intercept)| {
            intercept
                + row
                    .iter()
                    .zip(means)
                    .zip(weights)
                    .map(|((&x, &mean), &weight)| (x - mean) * weight)
                    .sum::<f64>()
        })
        .collect())
}

fn cpu_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let out_dim = biases.len();
    Ok(features
        .iter()
        .map(|row| {
            (0..out_dim)
                .map(|out| {
                    let mut value = biases[out];
                    for (input, &feature) in row.iter().enumerate() {
                        value += feature * weights[input * out_dim + out];
                    }
                    value
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

fn cpu_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Ok(pairs
        .iter()
        .map(|&(source, target)| {
            let score = embeddings[source]
                .iter()
                .zip(&embeddings[target])
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>();
            1.0 / (1.0 + (-score).exp())
        })
        .collect())
}

fn cpu_vector_add_report(selection: BackendSelection, len: usize) -> Result<BackendDispatchReport> {
    let start = Instant::now();
    let checksum = (0..len)
        .map(|idx| {
            let x = idx as f32 * 0.5;
            let y = idx as f32 * 1.5;
            (x + y) as f64
        })
        .sum::<f64>();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(BackendDispatchReport {
        requested: selection.requested,
        selected: selection.selected,
        operation: "vector_add_f32".to_string(),
        len,
        checksum,
        expected_checksum: expected_vector_add_checksum(len),
        elapsed_ms,
        accelerated: false,
    })
}

pub(crate) fn expected_vector_add_checksum(len: usize) -> f64 {
    (len as f64) * ((len - 1) as f64)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_vector_add_report(
    selection: BackendSelection,
    len: usize,
) -> Result<BackendDispatchReport> {
    crate::cuda_oxide_backend::vector_add_report(selection, len, expected_vector_add_checksum(len))
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    crate::cuda_oxide_backend::affine_scores(features, means, weights, intercepts)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    crate::cuda_oxide_backend::dense_layer(features, weights, biases)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    crate::cuda_oxide_backend::pair_sigmoid_scores(embeddings, pairs)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_csr_diffusion_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    crate::cuda_oxide_backend::csr_diffusion(indptr, indices, weights, channels, values)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_csr_diffusion_backward_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    crate::cuda_oxide_backend::csr_diffusion_backward(
        indptr,
        indices,
        weights,
        channels,
        values,
        output_grad,
    )
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_csr_row_softmax_f32(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    crate::cuda_oxide_backend::csr_row_softmax(indptr, logits)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_csr_row_softmax_backward_f32(
    indptr: &[u32],
    weights: &[f32],
    output_grad: &[f32],
) -> Result<Vec<f32>> {
    crate::cuda_oxide_backend::csr_row_softmax_backward(indptr, weights, output_grad)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_adamw_step_f32(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    gradients: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    crate::cuda_oxide_backend::adamw(
        parameters,
        first,
        second,
        gradients,
        step,
        learning_rate,
        weight_decay,
    )
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_layer_norm_f32(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    crate::cuda_oxide_backend::layer_norm(values, rows, width, gamma, beta)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_scalar_graph_f32(
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    crate::cuda_oxide_backend::scalar_graph(initial_values, opcodes, left, right)
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_train_tanh_mlp_f32(
    _inputs: &[Vec<f32>],
    _targets: &[f32],
    _hidden_size: usize,
    _epochs: usize,
    _learning_rate: f32,
    _parameters: &mut [f32],
) -> Result<()> {
    Err(NeuralError::InvalidArgument(
        "CUDA Oxide tanh-MLP training kernel is not implemented".to_string(),
    ))
}

#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_unavailable<T>() -> Result<T> {
    Err(NeuralError::InvalidArgument(
        "CUDA Oxide is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_vector_add_report(_: BackendSelection, _: usize) -> Result<BackendDispatchReport> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_affine_scores(_: &[Vec<f64>], _: &[f64], _: &[f64], _: &[f64]) -> Result<Vec<f64>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_dense_layer_f32(_: &[Vec<f32>], _: &[f32], _: &[f32]) -> Result<Vec<Vec<f32>>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_pair_sigmoid_scores_f32(_: &[Vec<f32>], _: &[(usize, usize)]) -> Result<Vec<f64>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_csr_diffusion_f32(
    _: &[u32],
    _: &[u32],
    _: &[f32],
    _: usize,
    _: &[f32],
) -> Result<Vec<f32>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_csr_diffusion_backward_f32(
    _: &[u32],
    _: &[u32],
    _: &[f32],
    _: usize,
    _: &[f32],
    _: &[f32],
) -> Result<CsrDiffusionBackward> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_csr_row_softmax_f32(_: &[u32], _: &[f32]) -> Result<Vec<f32>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_csr_row_softmax_backward_f32(_: &[u32], _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_adamw_step_f32(
    _: &mut [f32],
    _: &mut [f32],
    _: &mut [f32],
    _: &[f32],
    _: u64,
    _: f32,
    _: f32,
) -> Result<()> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_layer_norm_f32(_: &[f32], _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
    cuda_unavailable()
}
#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_train_tanh_mlp_f32(
    _: &[Vec<f32>],
    _: &[f32],
    _: usize,
    _: usize,
    _: f32,
    _: &mut [f32],
) -> Result<()> {
    cuda_unavailable()
}

#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
fn rocm_vector_add_report(
    selection: BackendSelection,
    len: usize,
) -> Result<BackendDispatchReport> {
    crate::hip_backend::vector_add_report(selection, len, expected_vector_add_checksum(len))
}
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
fn rocm_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    crate::hip_backend::affine_scores(features, means, weights, intercepts)
}
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
fn rocm_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    crate::hip_backend::dense_layer(features, weights, biases)
}
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
fn rocm_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    crate::hip_backend::pair_sigmoid_scores(embeddings, pairs)
}
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
fn rocm_train_tanh_mlp_f32(
    _inputs: &[Vec<f32>],
    _targets: &[f32],
    _hidden_size: usize,
    _epochs: usize,
    _learning_rate: f32,
    _parameters: &mut [f32],
) -> Result<()> {
    Err(NeuralError::InvalidArgument(
        "ROCm runtime kernels are unsupported; use the pure Rust or CUDA Oxide backend".to_string(),
    ))
}
#[cfg(not(all(feature = "rocm", any(target_os = "linux", target_os = "windows"))))]
fn rocm_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "ROCm dispatch is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", any(target_os = "linux", target_os = "windows"))))]
fn rocm_affine_scores(
    _features: &[Vec<f64>],
    _means: &[f64],
    _weights: &[f64],
    _intercepts: &[f64],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "ROCm affine scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", any(target_os = "linux", target_os = "windows"))))]
fn rocm_dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "ROCm dense layer scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", any(target_os = "linux", target_os = "windows"))))]
fn rocm_pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "ROCm pair scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", any(target_os = "linux", target_os = "windows"))))]
fn rocm_train_tanh_mlp_f32(
    _inputs: &[Vec<f32>],
    _targets: &[f32],
    _hidden_size: usize,
    _epochs: usize,
    _learning_rate: f32,
    _parameters: &mut [f32],
) -> Result<()> {
    Err(NeuralError::InvalidArgument(
        "ROCm tanh-MLP training is not available in this build".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_backend_selects_cpu_even_when_accelerators_are_available() {
        let selection = select_backend(Some("auto")).unwrap();
        assert_eq!(selection.requested, "auto");
        assert_eq!(selection.selected, "cpu");

        let default_selection = select_backend(None).unwrap();
        assert_eq!(default_selection.requested, "auto");
        assert_eq!(default_selection.selected, "cpu");
    }

    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn webgpu_backend_requires_a_compiled_and_available_adapter() {
        let err = select_backend(Some("webgpu")).unwrap_err();
        assert!(err
            .to_string()
            .contains("requested compute backend \"webgpu\" is not available"));
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_backend_is_selectable_exactly_when_an_adapter_is_available() {
        let available = available_backends()
            .iter()
            .any(|backend| backend == "webgpu");
        match select_backend(Some("webgpu")) {
            Ok(selection) => {
                assert!(available);
                assert_eq!(selection.requested, "webgpu");
                assert_eq!(selection.selected, "webgpu");
            }
            Err(error) => {
                assert!(!available);
                assert!(error.to_string().contains("not available"));
            }
        }
    }

    #[test]
    fn cpu_dispatch_report_matches_expected_checksum() {
        let report = backend_dispatch_report(Some("cpu"), 16).unwrap();
        assert_eq!(report.selected, "cpu");
        assert!(!report.accelerated);
        assert!((report.checksum - report.expected_checksum).abs() < 1.0e-6);
    }

    #[test]
    fn cpu_affine_scores_match_manual_formula() {
        let selection = select_backend(Some("cpu")).unwrap();
        let features = vec![vec![2.0, 4.0], vec![1.0, -1.0]];
        let means = vec![1.0, 2.0];
        let weights = vec![0.5, -2.0];
        let intercepts = vec![3.0, 4.0];
        let scores =
            backend_affine_scores(&selection, &features, &means, &weights, &intercepts).unwrap();
        assert_eq!(scores, vec![-0.5, 10.0]);
    }

    #[test]
    fn cpu_dense_layer_f32_matches_manual_formula() {
        let selection = select_backend(Some("cpu")).unwrap();
        let features = vec![vec![1.0, 2.0], vec![-1.0, 0.5]];
        let weights = vec![0.5, -1.0, 2.0, 0.25];
        let biases = vec![0.1, -0.2];
        let scores = backend_dense_layer_f32(&selection, &features, &weights, &biases).unwrap();
        let expected = [vec![4.6, -0.7], vec![0.6, 0.925]];
        for (row, expected_row) in scores.iter().zip(expected) {
            for (actual, expected) in row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn cpu_pair_sigmoid_scores_f32_matches_manual_formula() {
        let selection = select_backend(Some("cpu")).unwrap();
        let embeddings = vec![vec![1.0, 2.0], vec![0.5, -1.0], vec![2.0, 0.25]];
        let pairs = vec![(0, 1), (1, 2)];
        let scores = backend_pair_sigmoid_scores_f32(&selection, &embeddings, &pairs).unwrap();
        let expected = [
            1.0 / (1.0 + (-(-1.5_f64)).exp()),
            1.0 / (1.0 + (-0.75_f64).exp()),
        ];
        for (actual, expected) in scores.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_dispatch_report_runs_vector_add_kernel() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "webgpu")
        {
            return;
        }
        let report = backend_dispatch_report(Some("webgpu"), 64).unwrap();
        assert_eq!(report.selected, "webgpu");
        assert!(report.accelerated);
        assert!((report.checksum - report.expected_checksum).abs() < 1.0e-3);
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_affine_scores_match_cpu_scores() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "webgpu")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let webgpu = select_backend(Some("webgpu")).unwrap();
        let features = (0..128)
            .map(|row| {
                (0..16)
                    .map(|col| row as f64 * 0.125 + col as f64 * 0.25)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let means = (0..16).map(|idx| idx as f64 * 0.1).collect::<Vec<_>>();
        let weights = (0..16)
            .map(|idx| (idx as f64 + 1.0) * 0.01)
            .collect::<Vec<_>>();
        let intercepts = (0..128).map(|idx| idx as f64 * 0.001).collect::<Vec<_>>();
        let cpu_scores =
            backend_affine_scores(&cpu, &features, &means, &weights, &intercepts).unwrap();
        let webgpu_scores =
            backend_affine_scores(&webgpu, &features, &means, &weights, &intercepts).unwrap();
        for (left, right) in cpu_scores.iter().zip(&webgpu_scores) {
            assert!((left - right).abs() < 1.0e-3);
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_dense_layer_f32_matches_cpu_scores() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "webgpu")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let webgpu = select_backend(Some("webgpu")).unwrap();
        let features = (0..64)
            .map(|row| {
                (0..12)
                    .map(|col| row as f32 * 0.03125 + col as f32 * 0.125)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = (0..(12 * 7))
            .map(|idx| (idx as f32 + 1.0) * 0.0025)
            .collect::<Vec<_>>();
        let biases = (0..7).map(|idx| idx as f32 * 0.01).collect::<Vec<_>>();
        let cpu_scores = backend_dense_layer_f32(&cpu, &features, &weights, &biases).unwrap();
        let webgpu_scores = backend_dense_layer_f32(&webgpu, &features, &weights, &biases).unwrap();
        for (cpu_row, webgpu_row) in cpu_scores.iter().zip(&webgpu_scores) {
            for (left, right) in cpu_row.iter().zip(webgpu_row) {
                assert!((left - right).abs() < 1.0e-3);
            }
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_pair_sigmoid_scores_f32_matches_cpu_scores() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "webgpu")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let webgpu = select_backend(Some("webgpu")).unwrap();
        let embeddings = (0..32)
            .map(|row| {
                (0..10)
                    .map(|col| row as f32 * 0.015625 + col as f32 * 0.0625)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pairs = (0..64)
            .map(|idx| (idx % 32, (idx * 7 + 3) % 32))
            .collect::<Vec<_>>();
        let cpu_scores = backend_pair_sigmoid_scores_f32(&cpu, &embeddings, &pairs).unwrap();
        let webgpu_scores = backend_pair_sigmoid_scores_f32(&webgpu, &embeddings, &pairs).unwrap();
        for (left, right) in cpu_scores.iter().zip(&webgpu_scores) {
            assert!((left - right).abs() < 1.0e-3);
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_lsttn_primitives_match_cpu_backend() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "webgpu")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let webgpu = select_backend(Some("webgpu")).unwrap();
        let indptr = [0, 2, 3, 3];
        let indices = [0, 1, 2];
        let edge_weights = [0.25, 0.75, 1.0];
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output_grad = [0.5, -1.0, 1.5, 2.0, -0.25, 0.75];

        assert_eq!(
            backend_csr_diffusion_f32(&cpu, &indptr, &indices, &edge_weights, 2, &values,).unwrap(),
            backend_csr_diffusion_f32(&webgpu, &indptr, &indices, &edge_weights, 2, &values,)
                .unwrap(),
        );
        assert_eq!(
            backend_csr_diffusion_backward_f32(
                &cpu,
                &indptr,
                &indices,
                &edge_weights,
                2,
                &values,
                &output_grad,
            )
            .unwrap(),
            backend_csr_diffusion_backward_f32(
                &webgpu,
                &indptr,
                &indices,
                &edge_weights,
                2,
                &values,
                &output_grad,
            )
            .unwrap(),
        );

        let logits = [0.2, -0.4, 1.1];
        let cpu_softmax = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        let webgpu_softmax = backend_csr_row_softmax_f32(&webgpu, &indptr, &logits).unwrap();
        assert_eq!(cpu_softmax, webgpu_softmax);
        assert_eq!(
            backend_csr_row_softmax_backward_f32(&cpu, &indptr, &cpu_softmax, &output_grad[..3],)
                .unwrap(),
            backend_csr_row_softmax_backward_f32(
                &webgpu,
                &indptr,
                &webgpu_softmax,
                &output_grad[..3],
            )
            .unwrap(),
        );

        let gamma = [1.0, 0.75, 1.25];
        let beta = [0.1, -0.2, 0.3];
        assert_close(
            &backend_layer_norm_f32(&cpu, &values, 2, 3, &gamma, &beta).unwrap(),
            &backend_layer_norm_f32(&webgpu, &values, 2, 3, &gamma, &beta).unwrap(),
            2.0e-6,
        );

        let mut cpu_parameters = vec![0.5, -1.0, 2.0];
        let mut cpu_first = vec![0.0; 3];
        let mut cpu_second = vec![0.0; 3];
        let mut webgpu_parameters = cpu_parameters.clone();
        let mut webgpu_first = cpu_first.clone();
        let mut webgpu_second = cpu_second.clone();
        let gradients = [0.25, -0.5, 0.75];
        backend_adamw_step_f32(
            &cpu,
            &mut cpu_parameters,
            &mut cpu_first,
            &mut cpu_second,
            &gradients,
            1,
            0.01,
            0.001,
        )
        .unwrap();
        backend_adamw_step_f32(
            &webgpu,
            &mut webgpu_parameters,
            &mut webgpu_first,
            &mut webgpu_second,
            &gradients,
            1,
            0.01,
            0.001,
        )
        .unwrap();
        assert_close(&cpu_parameters, &webgpu_parameters, 1.0e-6);
        assert_close(&cpu_first, &webgpu_first, 1.0e-6);
        assert_close(&cpu_second, &webgpu_second, 1.0e-6);

        let graph_values = backend_scalar_graph_f32(
            &webgpu,
            &[2.0, 3.0, 0.0],
            &[0, 0, 3],
            &[0, 0, 0],
            &[0, 0, 1],
        )
        .unwrap();
        assert_eq!(graph_values[2], 6.0);
        let mut graph_parameters = [2.0];
        let mut graph_first = [0.0];
        let mut graph_second = [0.0];
        let graph_loss = backend_scalar_graph_train_step_f32(
            &webgpu,
            &[0.0, 0.0],
            &[1, 3],
            &[0, 0],
            &[0, 0],
            &[0, u32::MAX],
            1,
            &mut graph_parameters,
            &mut graph_first,
            &mut graph_second,
            1,
            0.01,
            0.0,
        )
        .unwrap();
        assert_eq!(graph_loss, 4.0);
        assert!(graph_parameters[0] < 2.0);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_dispatch_report_runs_vector_add_kernel() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let report = backend_dispatch_report(Some("cuda"), 64).unwrap();
        assert_eq!(report.selected, "cuda");
        assert!(report.accelerated);
        assert!((report.checksum - report.expected_checksum).abs() < 1.0e-3);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_affine_scores_match_cpu_scores() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let features = (0..128)
            .map(|row| {
                (0..16)
                    .map(|col| row as f64 * 0.125 + col as f64 * 0.25)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let means = (0..16).map(|idx| idx as f64 * 0.1).collect::<Vec<_>>();
        let weights = (0..16)
            .map(|idx| (idx as f64 + 1.0) * 0.01)
            .collect::<Vec<_>>();
        let intercepts = (0..128).map(|idx| idx as f64 * 0.001).collect::<Vec<_>>();
        let cpu_scores =
            backend_affine_scores(&cpu, &features, &means, &weights, &intercepts).unwrap();
        let cuda_scores =
            backend_affine_scores(&cuda, &features, &means, &weights, &intercepts).unwrap();
        for (left, right) in cpu_scores.iter().zip(&cuda_scores) {
            assert!((left - right).abs() < 1.0e-3);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_dense_layer_f32_matches_cpu_scores() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let features = (0..64)
            .map(|row| {
                (0..12)
                    .map(|col| row as f32 * 0.03125 + col as f32 * 0.125)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = (0..(12 * 7))
            .map(|idx| (idx as f32 + 1.0) * 0.0025)
            .collect::<Vec<_>>();
        let biases = (0..7).map(|idx| idx as f32 * 0.01).collect::<Vec<_>>();
        let cpu_scores = backend_dense_layer_f32(&cpu, &features, &weights, &biases).unwrap();
        let cuda_scores = backend_dense_layer_f32(&cuda, &features, &weights, &biases).unwrap();
        for (cpu_row, cuda_row) in cpu_scores.iter().zip(&cuda_scores) {
            for (left, right) in cpu_row.iter().zip(cuda_row) {
                assert!((left - right).abs() < 1.0e-3);
            }
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_pair_sigmoid_scores_f32_matches_cpu_scores() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let embeddings = (0..32)
            .map(|row| {
                (0..10)
                    .map(|col| row as f32 * 0.015625 + col as f32 * 0.0625)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pairs = (0..64)
            .map(|idx| (idx % 32, (idx * 7 + 3) % 32))
            .collect::<Vec<_>>();
        let cpu_scores = backend_pair_sigmoid_scores_f32(&cpu, &embeddings, &pairs).unwrap();
        let cuda_scores = backend_pair_sigmoid_scores_f32(&cuda, &embeddings, &pairs).unwrap();
        for (left, right) in cpu_scores.iter().zip(&cuda_scores) {
            assert!((left - right).abs() < 1.0e-3);
        }
    }

    #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn rocm_dispatch_report_runs_vector_add_kernel() {
        if !available_backends().iter().any(|backend| backend == "rocm") {
            return;
        }
        let report = backend_dispatch_report(Some("rocm"), 64).unwrap();
        assert_eq!(report.selected, "rocm");
        assert!(report.accelerated);
        assert!((report.checksum - report.expected_checksum).abs() < 1.0e-3);
    }

    #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn rocm_affine_scores_match_cpu_scores() {
        if !available_backends().iter().any(|backend| backend == "rocm") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let rocm = select_backend(Some("rocm")).unwrap();
        let features = (0..128)
            .map(|row| {
                (0..16)
                    .map(|col| row as f64 * 0.125 + col as f64 * 0.25)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let means = (0..16).map(|idx| idx as f64 * 0.1).collect::<Vec<_>>();
        let weights = (0..16)
            .map(|idx| (idx as f64 + 1.0) * 0.01)
            .collect::<Vec<_>>();
        let intercepts = (0..128).map(|idx| idx as f64 * 0.001).collect::<Vec<_>>();
        let cpu_scores =
            backend_affine_scores(&cpu, &features, &means, &weights, &intercepts).unwrap();
        let rocm_scores =
            backend_affine_scores(&rocm, &features, &means, &weights, &intercepts).unwrap();
        for (left, right) in cpu_scores.iter().zip(&rocm_scores) {
            assert!((left - right).abs() < 1.0e-3);
        }
    }

    #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn rocm_dense_layer_f32_matches_cpu_scores() {
        if !available_backends().iter().any(|backend| backend == "rocm") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let rocm = select_backend(Some("rocm")).unwrap();
        let features = (0..64)
            .map(|row| {
                (0..12)
                    .map(|col| row as f32 * 0.03125 + col as f32 * 0.125)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = (0..(12 * 7))
            .map(|idx| (idx as f32 + 1.0) * 0.0025)
            .collect::<Vec<_>>();
        let biases = (0..7).map(|idx| idx as f32 * 0.01).collect::<Vec<_>>();
        let cpu_scores = backend_dense_layer_f32(&cpu, &features, &weights, &biases).unwrap();
        let rocm_scores = backend_dense_layer_f32(&rocm, &features, &weights, &biases).unwrap();
        for (cpu_row, rocm_row) in cpu_scores.iter().zip(&rocm_scores) {
            for (left, right) in cpu_row.iter().zip(rocm_row) {
                assert!((left - right).abs() < 1.0e-3);
            }
        }
    }

    #[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn rocm_pair_sigmoid_scores_f32_matches_cpu_scores() {
        if !available_backends().iter().any(|backend| backend == "rocm") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let rocm = select_backend(Some("rocm")).unwrap();
        let embeddings = (0..32)
            .map(|row| {
                (0..10)
                    .map(|col| row as f32 * 0.015625 + col as f32 * 0.0625)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pairs = (0..64)
            .map(|idx| (idx % 32, (idx * 7 + 3) % 32))
            .collect::<Vec<_>>();
        let cpu_scores = backend_pair_sigmoid_scores_f32(&cpu, &embeddings, &pairs).unwrap();
        let rocm_scores = backend_pair_sigmoid_scores_f32(&rocm, &embeddings, &pairs).unwrap();
        for (left, right) in cpu_scores.iter().zip(&rocm_scores) {
            assert!((left - right).abs() < 1.0e-3);
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn webgpu_tensor_kernels_match_cuda_cpu_contracts() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "webgpu")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let webgpu = select_backend(Some("webgpu")).unwrap();
        let indptr = [0, 2, 3];
        let indices = [0, 1, 1];
        let weights = [0.25, 0.75, 1.0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let output_grad = [0.5, -0.25, 1.5, 2.0];

        assert_close(
            &backend_csr_diffusion_f32(&cpu, &indptr, &indices, &weights, 2, &values).unwrap(),
            &backend_csr_diffusion_f32(&webgpu, &indptr, &indices, &weights, 2, &values).unwrap(),
            1.0e-5,
        );
        let cpu_backward = backend_csr_diffusion_backward_f32(
            &cpu,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        let gpu_backward = backend_csr_diffusion_backward_f32(
            &webgpu,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        assert_close(&cpu_backward.input_grad, &gpu_backward.input_grad, 1.0e-5);
        assert_close(&cpu_backward.edge_grad, &gpu_backward.edge_grad, 1.0e-5);

        let logits = [0.5, -1.0, 2.0];
        let cpu_softmax = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        let gpu_softmax = backend_csr_row_softmax_f32(&webgpu, &indptr, &logits).unwrap();
        assert_close(&cpu_softmax, &gpu_softmax, 1.0e-5);
        let gradient = [0.3, -0.2, 0.7];
        assert_close(
            &backend_csr_row_softmax_backward_f32(&cpu, &indptr, &cpu_softmax, &gradient).unwrap(),
            &backend_csr_row_softmax_backward_f32(&webgpu, &indptr, &gpu_softmax, &gradient)
                .unwrap(),
            1.0e-5,
        );

        let mut cpu_parameters = [1.0, -2.0, 0.5];
        let mut gpu_parameters = cpu_parameters;
        let mut cpu_first = [0.0; 3];
        let mut gpu_first = cpu_first;
        let mut cpu_second = [0.0; 3];
        let mut gpu_second = cpu_second;
        let gradients = [0.1, -0.25, 0.75];
        backend_adamw_step_f32(
            &cpu,
            &mut cpu_parameters,
            &mut cpu_first,
            &mut cpu_second,
            &gradients,
            1,
            0.001,
            0.01,
        )
        .unwrap();
        backend_adamw_step_f32(
            &webgpu,
            &mut gpu_parameters,
            &mut gpu_first,
            &mut gpu_second,
            &gradients,
            1,
            0.001,
            0.01,
        )
        .unwrap();
        assert_close(&cpu_parameters, &gpu_parameters, 1.0e-5);
        assert_close(&cpu_first, &gpu_first, 1.0e-6);
        assert_close(&cpu_second, &gpu_second, 1.0e-6);

        let gamma = [1.0, 0.5];
        let beta = [0.1, -0.2];
        assert_close(
            &backend_layer_norm_f32(&cpu, &values, 2, 2, &gamma, &beta).unwrap(),
            &backend_layer_norm_f32(&webgpu, &values, 2, 2, &gamma, &beta).unwrap(),
            1.0e-5,
        );
    }

    #[cfg(feature = "webgpu")]
    fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert!(
                (expected - actual).abs() <= tolerance,
                "{expected} != {actual}"
            );
        }
    }

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn metal_dispatch_report_runs_vector_add_kernel() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let report = backend_dispatch_report(Some("metal"), 64).unwrap();
        assert_eq!(report.selected, "metal");
        assert!(report.accelerated);
        assert!((report.checksum - report.expected_checksum).abs() < 1.0e-3);
    }

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn metal_affine_scores_match_cpu_scores() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let metal = select_backend(Some("metal")).unwrap();
        let features = (0..128)
            .map(|row| {
                (0..16)
                    .map(|col| row as f64 * 0.125 + col as f64 * 0.25)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let means = (0..16).map(|idx| idx as f64 * 0.1).collect::<Vec<_>>();
        let weights = (0..16)
            .map(|idx| (idx as f64 + 1.0) * 0.01)
            .collect::<Vec<_>>();
        let intercepts = (0..128).map(|idx| idx as f64 * 0.001).collect::<Vec<_>>();
        let cpu_scores =
            backend_affine_scores(&cpu, &features, &means, &weights, &intercepts).unwrap();
        let metal_scores =
            backend_affine_scores(&metal, &features, &means, &weights, &intercepts).unwrap();
        for (left, right) in cpu_scores.iter().zip(&metal_scores) {
            assert!((left - right).abs() < 1.0e-4);
        }
    }

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn metal_dense_layer_f32_matches_cpu_scores() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let metal = select_backend(Some("metal")).unwrap();
        let features = (0..64)
            .map(|row| {
                (0..12)
                    .map(|col| row as f32 * 0.03125 + col as f32 * 0.125)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = (0..(12 * 7))
            .map(|idx| (idx as f32 + 1.0) * 0.0025)
            .collect::<Vec<_>>();
        let biases = (0..7).map(|idx| idx as f32 * 0.01).collect::<Vec<_>>();
        let cpu_scores = backend_dense_layer_f32(&cpu, &features, &weights, &biases).unwrap();
        let metal_scores = backend_dense_layer_f32(&metal, &features, &weights, &biases).unwrap();
        for (cpu_row, metal_row) in cpu_scores.iter().zip(&metal_scores) {
            for (left, right) in cpu_row.iter().zip(metal_row) {
                assert!((left - right).abs() < 1.0e-4);
            }
        }
    }

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn metal_pair_sigmoid_scores_f32_matches_cpu_scores() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let metal = select_backend(Some("metal")).unwrap();
        let embeddings = (0..32)
            .map(|row| {
                (0..10)
                    .map(|col| row as f32 * 0.015625 + col as f32 * 0.0625)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pairs = (0..64)
            .map(|idx| (idx % 32, (idx * 7 + 3) % 32))
            .collect::<Vec<_>>();
        let cpu_scores = backend_pair_sigmoid_scores_f32(&cpu, &embeddings, &pairs).unwrap();
        let metal_scores = backend_pair_sigmoid_scores_f32(&metal, &embeddings, &pairs).unwrap();
        for (left, right) in cpu_scores.iter().zip(&metal_scores) {
            assert!((left - right).abs() < 1.0e-5);
        }
    }

    #[test]
    fn csr_diffusion_preserves_batch_layout_and_isolated_rows() {
        let cpu = select_backend(Some("cpu")).unwrap();
        // Row 2 is intentionally isolated.
        let indptr = [0, 2, 3, 3];
        let indices = [0, 1, 2];
        let weights = [0.25, 0.75, 1.0];
        let values = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, // batch 1
        ];
        let output =
            backend_csr_diffusion_f32(&cpu, &indptr, &indices, &weights, 2, &values).unwrap();
        assert_eq!(
            output,
            vec![2.5, 3.5, 5.0, 6.0, 0.0, 0.0, 5.0, 7.0, 10.0, 12.0, 0.0, 0.0]
        );
    }

    #[test]
    fn csr_diffusion_backward_matches_central_difference() {
        let cpu = select_backend(Some("cpu")).unwrap();
        let indptr = [0, 2, 3];
        let indices = [0, 1, 0];
        let weights = [0.25, 0.75, -0.5];
        let values = [1.0, -2.0, 3.0, 4.0];
        let output_grad = [0.5, -1.0, 2.0, 0.25];
        let backward = backend_csr_diffusion_backward_f32(
            &cpu,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        let objective = |input: &[f32], edge_weights: &[f32]| -> f32 {
            backend_csr_diffusion_f32(&cpu, &indptr, &indices, edge_weights, 2, input)
                .unwrap()
                .iter()
                .zip(output_grad)
                .map(|(value, gradient)| value * gradient)
                .sum()
        };
        let epsilon = 1.0e-3;
        for index in 0..values.len() {
            let mut plus = values;
            let mut minus = values;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical =
                (objective(&plus, &weights) - objective(&minus, &weights)) / (2.0 * epsilon);
            assert!((numerical - backward.input_grad[index]).abs() < 2.0e-3);
        }
        for index in 0..weights.len() {
            let mut plus = weights;
            let mut minus = weights;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical =
                (objective(&values, &plus) - objective(&values, &minus)) / (2.0 * epsilon);
            assert!((numerical - backward.edge_grad[index]).abs() < 2.0e-3);
        }
    }

    #[test]
    fn csr_row_softmax_backward_matches_central_difference() {
        let cpu = select_backend(Some("cpu")).unwrap();
        let indptr = [0, 2, 2, 5];
        let logits = [0.25, -0.5, 1.0, 0.0, -0.75];
        let output_grad = [0.5, -1.0, 0.25, 2.0, -0.5];
        let weights = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        assert!((weights[0] + weights[1] - 1.0).abs() < 1.0e-6);
        assert!((weights[2] + weights[3] + weights[4] - 1.0).abs() < 1.0e-6);
        let backward =
            backend_csr_row_softmax_backward_f32(&cpu, &indptr, &weights, &output_grad).unwrap();
        let objective = |input: &[f32]| -> f32 {
            backend_csr_row_softmax_f32(&cpu, &indptr, input)
                .unwrap()
                .iter()
                .zip(output_grad)
                .map(|(weight, gradient)| weight * gradient)
                .sum()
        };
        let epsilon = 1.0e-3;
        for index in 0..logits.len() {
            let mut plus = logits;
            let mut minus = logits;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (objective(&plus) - objective(&minus)) / (2.0 * epsilon);
            assert!((numerical - backward[index]).abs() < 2.0e-3);
        }
    }

    #[test]
    fn adamw_updates_contiguous_cpu_tensors() {
        let cpu = select_backend(Some("cpu")).unwrap();
        let mut parameters = vec![1.0, -2.0, 0.5];
        let mut first = vec![0.0; 3];
        let mut second = vec![0.0; 3];
        backend_adamw_step_f32(
            &cpu,
            &mut parameters,
            &mut first,
            &mut second,
            &[0.5, -0.25, 1.0],
            1,
            0.01,
            0.001,
        )
        .unwrap();
        assert!(parameters
            .iter()
            .zip([1.0, -2.0, 0.5])
            .any(|(actual, initial)| actual != &initial));
        assert!(
            first.iter().all(|value| value.is_finite()) && second.iter().all(|value| *value >= 0.0)
        );
    }

    #[test]
    fn masked_inverse_scale_mae_omits_zero_targets() {
        let (loss, gradient) =
            masked_inverse_scale_mae_f32(&[2.0, 99.0, -1.0], &[1.0, 0.0, 1.0], 0.0, 2.0).unwrap();
        assert!((loss - 3.0).abs() < 1.0e-6);
        assert_eq!(gradient, vec![1.0, 0.0, -1.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_csr_diffusion_and_backward_match_cpu() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let indptr = [0, 2, 3, 3];
        let indices = [0, 1, 2];
        let weights = [0.25, 0.75, -0.5];
        let values = [
            1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        ];
        let output_grad = [0.5; 12];
        let cpu_output =
            backend_csr_diffusion_f32(&cpu, &indptr, &indices, &weights, 2, &values).unwrap();
        let cuda_output =
            backend_csr_diffusion_f32(&cuda, &indptr, &indices, &weights, 2, &values).unwrap();
        let cpu_backward = backend_csr_diffusion_backward_f32(
            &cpu,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        let cuda_backward = backend_csr_diffusion_backward_f32(
            &cuda,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        for (left, right) in cpu_output.iter().zip(cuda_output) {
            assert!((left - right).abs() < 1.0e-4);
        }
        for (left, right) in cpu_backward.input_grad.iter().zip(cuda_backward.input_grad) {
            assert!((left - right).abs() < 1.0e-4);
        }
        for (left, right) in cpu_backward.edge_grad.iter().zip(cuda_backward.edge_grad) {
            assert!((left - right).abs() < 1.0e-4);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_reuses_warmed_batch_slots() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(5).unwrap();
        let input = vec![0.25_f32; 32];
        let activations = vec![-0.5_f32; 64];
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &activations).unwrap();
        let warmed_allocations = arena.allocation_count();
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &activations).unwrap();
        assert_eq!(arena.allocation_count(), warmed_allocations);
        assert_eq!(arena.capacity_f32(0).unwrap(), input.len());
        assert_eq!(arena.capacity_f32(1).unwrap(), activations.len());
        let mut round_trip = vec![0.0_f32; input.len()];
        arena.download_f32(0, &mut round_trip).unwrap();
        assert_eq!(round_trip, input);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_composes_affine_residual_and_gelu_on_device() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // x is [2, 3], w is [3, 2]. The residual deliberately aliases a
        // separate slot so this verifies a normal Transformer-style dataflow
        // rather than a host round trip between kernels.
        let x = [1.0_f32, -2.0, 0.5, 0.25, 1.0, -1.0];
        let w = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5];
        let b = [0.25_f32, -0.75];
        let residual = [0.1_f32, -0.2, 0.3, 0.4];
        let mut expected = Vec::new();
        for row in x.chunks_exact(3) {
            for out in 0..2 {
                let affine = b[out] + (0..3).map(|col| row[col] * w[col * 2 + out]).sum::<f32>();
                let value = affine + residual[expected.len()];
                expected.push(
                    0.5 * value
                        * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh()),
                );
            }
        }
        let mut arena = CudaTensorArena::new(6).unwrap();
        arena.upload_f32(0, &x).unwrap();
        arena.upload_f32(1, &w).unwrap();
        arena.upload_f32(2, &b).unwrap();
        arena.upload_f32(3, &residual).unwrap();
        arena.affine_f32(0, 1, 2, 4, 2, 3, 2).unwrap();
        arena.add_f32(4, 3, 5, 4).unwrap();
        // Reuse the affine activation slot after its final consumer. This is
        // the arena lifetime pattern used by the batch executor.
        arena.gelu_f32(5, 4, 4).unwrap();
        arena.synchronize().unwrap();
        let allocations = arena.allocation_count();
        let mut actual = [0.0_f32; 4];
        arena.download_f32(4, &mut actual).unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() < 2.0e-5, "{left} != {right}");
        }
        arena.affine_f32(0, 1, 2, 4, 2, 3, 2).unwrap();
        arena.add_f32(4, 3, 5, 4).unwrap();
        arena.gelu_f32(5, 4, 4).unwrap();
        assert_eq!(arena.allocation_count(), allocations);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_affine_reads_global_parameter_slice() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let input = [1.0_f32, 2.0, -1.0, 0.5]; // [2, 2]
                                               // Prefix/suffix values prove the projection reads offsets rather than
                                               // uploading a copied matrix/bias tensor.
        let parameters = [99.0_f32, 0.5, -1.0, 2.0, 0.25, 0.1, -0.2, -77.0];
        let mut arena = CudaTensorArena::new(3).unwrap();
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &parameters).unwrap();
        arena
            .affine_parameter_slice_f32(0, 1, 1, 5, 2, 2, 2, 2)
            .unwrap();
        arena.synchronize().unwrap();
        let mut output = [0.0_f32; 4];
        arena.download_f32(2, &mut output).unwrap();
        for (actual, expected) in output.iter().zip([4.6, -0.7, 0.6, 0.925]) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_patch_embedding_uses_contiguous_batch_layout() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // [batch=1, time=4, nodes=2, channels=1], width=2, hidden=2.
        let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        // Prefix then patch weights [[2, -1], [0.5, 1.5]], bias [0.1, -0.2].
        let parameters = [99.0_f32, 2.0, -1.0, 0.5, 1.5, 0.1, -0.2];
        let mut arena = CudaTensorArena::new(3).unwrap();
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &parameters).unwrap();
        arena
            .patch_embedding_f32(0, 1, 1, 5, 2, 1, 4, 2, 1, 2, 2)
            .unwrap();
        arena.synchronize().unwrap();
        let mut output = [0.0_f32; 8];
        arena.download_f32(2, &mut output).unwrap();
        let expected = [
            3.1, 1.8, 30.1, 19.8, // patch 0, nodes 0/1
            8.1, 2.8, 80.1, 29.8, // patch 1, nodes 0/1
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(2, &[1.0_f32; 8]).unwrap();
        arena.fill_f32(1, parameters.len(), 0.0).unwrap();
        arena
            .patch_embedding_parameter_slice_backward_f32(0, 2, 1, 1, 5, 1, 4, 2, 1, 2, 2)
            .unwrap();
        let mut gradients = [0.0_f32; 7];
        arena.download_f32(1, &mut gradients).unwrap();
        assert_eq!(gradients, [0.0, 44.0, 44.0, 66.0, 66.0, 4.0, 4.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_patch_position_backward_and_mask_tokens_are_resident() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(6).unwrap();
        arena.upload_f32(0, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        arena
            .upload_f32(1, &[0.5_f32, 1.0, 10.0, 20.0, 30.0, 40.0])
            .unwrap();
        arena
            .add_patch_positions_f32(0, 1, 2, 2, 1, 2, 1, 2, 2.0)
            .unwrap();
        let mut positioned = [0.0_f32; 4];
        arena.download_f32(2, &mut positioned).unwrap();
        assert_eq!(positioned, [22.0, 44.0, 66.0, 88.0]);
        arena.upload_f32(3, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        arena.fill_f32(4, 6, 0.0).unwrap();
        arena
            .add_patch_positions_backward_f32(3, 5, 4, 2, 1, 2, 1, 2, 2.0)
            .unwrap();
        let mut input_gradient = [0.0_f32; 4];
        let mut parameter_gradient = [0.0_f32; 6];
        arena.download_f32(5, &mut input_gradient).unwrap();
        arena.download_f32(4, &mut parameter_gradient).unwrap();
        assert_eq!(input_gradient, [2.0, 4.0, 6.0, 8.0]);
        assert_eq!(parameter_gradient, [0.0, 0.0, 2.0, 4.0, 6.0, 8.0]);

        arena.upload_f32(0, &[7.0_f32, 8.0]).unwrap();
        arena.upload_u32(0, &[1]).unwrap();
        arena
            .assemble_masked_decoder_tokens_f32(0, 0, 1, 0, 2, 2, 1, 1, 1, 1, 2, 2, 3.0)
            .unwrap();
        let mut decoder = [0.0_f32; 4];
        arena.download_f32(2, &mut decoder).unwrap();
        assert_eq!(decoder, [21.0, 24.0, 91.5, 123.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_masked_patch_reconstruction_backward_matches_cpu() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(7).unwrap();
        // decoded layout: [batch=1, nodes=1, visible+masked=2, hidden=2].
        arena.upload_f32(0, &[99.0_f32, 99.0, 2.0, 3.0]).unwrap();
        // target layout: [batch=1, time=2, nodes=1, channels=1].
        arena.upload_f32(1, &[8.0_f32, 10.0]).unwrap();
        arena.upload_u32(0, &[0]).unwrap();
        // decoder weights [patch_width=2, hidden=2], then decoder bias [2].
        arena
            .upload_f32(2, &[1.0_f32, 1.0, 2.0, 0.5, 0.0, 1.0])
            .unwrap();
        arena.fill_f32(3, 6, 0.0).unwrap();
        arena
            .masked_patch_reconstruction_loss_backward_f32(
                0, 1, 0, 2, 0, 4, 4, 3, 5, 1, 2, 1, 1, 1, 1, 2, 2, 0.0, 1.0,
            )
            .unwrap();
        arena.synchronize().unwrap();

        let mut loss = [0.0_f32; 2];
        let mut context_gradient = [0.0_f32; 4];
        let mut parameter_gradient = [0.0_f32; 6];
        arena.download_f32(5, &mut loss).unwrap();
        arena.download_f32(4, &mut context_gradient).unwrap();
        arena.download_f32(3, &mut parameter_gradient).unwrap();

        assert!((loss[0] - 3.25).abs() < 1.0e-5, "{loss:?}");
        assert_eq!(loss[1], 2.0);
        assert_eq!(context_gradient[0], 0.0);
        assert_eq!(context_gradient[1], 0.0);
        assert!((context_gradient[2] + 1.5).abs() < 1.0e-5);
        assert!((context_gradient[3] + 0.75).abs() < 1.0e-5);
        let expected = [-1.0_f32, -1.5, -1.0, -1.5, -0.5, -0.5];
        for (actual, expected) in parameter_gradient.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_affine_backward_matches_cpu_reduction() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let x = [1.0_f32, 2.0, -1.0, 0.5, -0.25, 3.0]; // [2, 3]
        let w = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.5]; // [3, 2]
        let dy = [0.25_f32, -0.5, 2.0, 1.0]; // [2, 2]
        let mut expected_dx = [0.0_f32; 6];
        let mut expected_dw = [0.0_f32; 6];
        let mut expected_db = [0.0_f32; 2];
        for row in 0..2 {
            for input in 0..3 {
                expected_dx[row * 3 + input] = (0..2)
                    .map(|out| dy[row * 2 + out] * w[input * 2 + out])
                    .sum();
            }
        }
        for input in 0..3 {
            for out in 0..2 {
                expected_dw[input * 2 + out] = (0..2)
                    .map(|row| x[row * 3 + input] * dy[row * 2 + out])
                    .sum();
            }
        }
        for out in 0..2 {
            expected_db[out] = dy[out] + dy[2 + out];
        }
        let mut arena = CudaTensorArena::new(6).unwrap();
        arena.upload_f32(0, &x).unwrap();
        arena.upload_f32(1, &w).unwrap();
        arena.upload_f32(2, &dy).unwrap();
        arena
            .affine_backward_f32(0, 1, 2, 3, 4, 5, 2, 3, 2)
            .unwrap();
        arena.synchronize().unwrap();
        let mut dx = [0.0_f32; 6];
        let mut dw = [0.0_f32; 6];
        let mut db = [0.0_f32; 2];
        arena.download_f32(3, &mut dx).unwrap();
        arena.download_f32(4, &mut dw).unwrap();
        arena.download_f32(5, &mut db).unwrap();
        for (actual, expected) in dx
            .iter()
            .zip(expected_dx)
            .chain(dw.iter().zip(expected_dw))
            .chain(db.iter().zip(expected_db))
        {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_keeps_csr_graph_and_activations_resident() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // rows: 0 <- {0, 1}; 1 <- {2}; 2 <- {}. Two one-channel batches.
        let indptr = [0_u32, 2, 3, 3];
        let indices = [0_u32, 1, 2];
        let weights = [0.25_f32, 0.75, -0.5];
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut arena = CudaTensorArena::new(3).unwrap();
        arena.upload_u32(0, &indptr).unwrap();
        arena.upload_u32(1, &indices).unwrap();
        arena.upload_f32(0, &weights).unwrap();
        arena.upload_f32(1, &values).unwrap();
        let warm_allocations = arena.allocation_count();
        arena.csr_diffuse_f32(0, 1, 0, 1, 2, 2, 3, 1, 3).unwrap();
        arena.synchronize().unwrap();
        let mut actual = [0.0_f32; 6];
        arena.download_f32(2, &mut actual).unwrap();
        assert_eq!(actual, [1.75, -1.5, 0.0, 4.75, -3.0, 0.0]);
        arena.csr_diffuse_f32(0, 1, 0, 1, 2, 2, 3, 1, 3).unwrap();
        assert_eq!(arena.allocation_count(), warm_allocations + 1);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_adaptive_csr_softmax_backward_matches_cpu() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let indptr = [0_u32, 2, 2, 5];
        let logits = [0.25_f32, -0.5, 1.0, 0.0, -0.75];
        let output_gradient = [0.5_f32, -1.0, 0.25, 2.0, -0.5];
        let cpu = select_backend(Some("cpu")).unwrap();
        let expected_weights = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        let expected_gradient = backend_csr_row_softmax_backward_f32(
            &cpu,
            &indptr,
            &expected_weights,
            &output_gradient,
        )
        .unwrap();
        let mut arena = CudaTensorArena::new(4).unwrap();
        arena.upload_u32(0, &indptr).unwrap();
        arena.upload_f32(0, &logits).unwrap();
        arena.upload_f32(2, &output_gradient).unwrap();
        arena.csr_row_softmax_f32(0, 0, 1, 3, 5).unwrap();
        arena
            .csr_row_softmax_backward_f32(0, 1, 2, 3, 3, 5)
            .unwrap();
        arena.synchronize().unwrap();
        let mut actual_weights = [0.0_f32; 5];
        let mut actual_gradient = [0.0_f32; 5];
        arena.download_f32(1, &mut actual_weights).unwrap();
        arena.download_f32(3, &mut actual_gradient).unwrap();
        for (actual, expected) in actual_weights
            .iter()
            .zip(expected_weights)
            .chain(actual_gradient.iter().zip(expected_gradient))
        {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_layer_norm_forward_and_backward_match_finite_difference() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let values = [1.0_f32, -2.0, 3.0, 4.0, -1.0, 0.5];
        let gamma = [1.0_f32, 0.5, -0.25];
        let beta = [0.1_f32, -0.2, 0.3];
        let dy = [0.5_f32, -1.0, 0.25, 2.0, -0.5, 0.75];
        let cpu = select_backend(Some("cpu")).unwrap();
        let expected = backend_layer_norm_f32(&cpu, &values, 2, 3, &gamma, &beta).unwrap();
        let mut arena = CudaTensorArena::new(8).unwrap();
        arena.upload_f32(0, &values).unwrap();
        arena.upload_f32(1, &gamma).unwrap();
        arena.upload_f32(2, &beta).unwrap();
        arena.upload_f32(4, &dy).unwrap();
        arena.layer_norm_f32(0, 1, 2, 3, 2, 3).unwrap();
        arena
            .layer_norm_backward_f32(0, 1, 4, 5, 6, 7, 2, 3)
            .unwrap();
        arena.synchronize().unwrap();
        let mut actual = [0.0_f32; 6];
        let mut dx = [0.0_f32; 6];
        let mut dg = [0.0_f32; 3];
        let mut db = [0.0_f32; 3];
        arena.download_f32(3, &mut actual).unwrap();
        arena.download_f32(5, &mut dx).unwrap();
        arena.download_f32(6, &mut dg).unwrap();
        arena.download_f32(7, &mut db).unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() < 1.0e-5);
        }
        let objective = |x: &[f32], g: &[f32], b: &[f32]| -> f32 {
            backend_layer_norm_f32(&cpu, x, 2, 3, g, b)
                .unwrap()
                .iter()
                .zip(dy)
                .map(|(a, d)| a * d)
                .sum()
        };
        let eps = 1.0e-3;
        for i in 0..values.len() {
            let mut plus = values;
            let mut minus = values;
            plus[i] += eps;
            minus[i] -= eps;
            let numeric =
                (objective(&plus, &gamma, &beta) - objective(&minus, &gamma, &beta)) / (2.0 * eps);
            assert!(
                (dx[i] - numeric).abs() < 3.0e-3,
                "dx {i}: {} != {numeric}",
                dx[i]
            );
        }
        for i in 0..gamma.len() {
            let mut plus = gamma;
            let mut minus = gamma;
            plus[i] += eps;
            minus[i] -= eps;
            let numeric = (objective(&values, &plus, &beta) - objective(&values, &minus, &beta))
                / (2.0 * eps);
            assert!((dg[i] - numeric).abs() < 3.0e-3);
            let mut plus_b = beta;
            let mut minus_b = beta;
            plus_b[i] += eps;
            minus_b[i] -= eps;
            let numeric_b = (objective(&values, &gamma, &plus_b)
                - objective(&values, &gamma, &minus_b))
                / (2.0 * eps);
            assert!((db[i] - numeric_b).abs() < 3.0e-3);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_attention_matches_reference_and_is_causal() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // [one sequence, three tokens, one head, two channels]
        let q = [1.0_f32, 0.0, 0.5, 1.0, -1.0, 0.25];
        let k = [0.25_f32, 1.0, 1.0, -0.5, 0.5, 0.75];
        let v = [2.0_f32, -1.0, 0.5, 1.5, -2.0, 0.25];
        let mut expected = [0.0_f32; 6];
        let scale = (2.0_f32).sqrt().recip();
        for token in 0..3 {
            let mut scores = (0..=token)
                .map(|key| {
                    (0..2)
                        .map(|d| q[token * 2 + d] * k[key * 2 + d])
                        .sum::<f32>()
                        * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for score in &mut scores {
                *score = (*score - maximum).exp();
            }
            let total: f32 = scores.iter().sum();
            for d in 0..2 {
                expected[token * 2 + d] = (0..=token)
                    .map(|key| scores[key] / total * v[key * 2 + d])
                    .sum();
            }
        }
        let mut arena = CudaTensorArena::new(4).unwrap();
        arena.upload_f32(0, &q).unwrap();
        arena.upload_f32(1, &k).unwrap();
        arena.upload_f32(2, &v).unwrap();
        arena.attention_f32(0, 1, 2, 3, 1, 3, 1, 2, true).unwrap();
        arena.synchronize().unwrap();
        let mut actual = [0.0_f32; 6];
        arena.download_f32(3, &mut actual).unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() < 1.0e-5, "{left} != {right}");
        }
        assert_eq!(&actual[..2], &v[..2]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_attention_backward_matches_finite_difference() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let q = [1.0_f32, 0.0, 0.5, 1.0, -1.0, 0.25];
        let k = [0.25_f32, 1.0, 1.0, -0.5, 0.5, 0.75];
        let v = [2.0_f32, -1.0, 0.5, 1.5, -2.0, 0.25];
        let dy = [0.5_f32, -1.0, 0.25, 2.0, -0.5, 0.75];
        let objective = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            let mut total = 0.0;
            let scale = (2.0_f32).sqrt().recip();
            for token in 0..3 {
                let mut scores = (0..=token)
                    .map(|key| {
                        (0..2)
                            .map(|d| q[token * 2 + d] * k[key * 2 + d])
                            .sum::<f32>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for score in &mut scores {
                    *score = (*score - max).exp();
                }
                let norm: f32 = scores.iter().sum();
                for d in 0..2 {
                    let y = (0..=token)
                        .map(|key| scores[key] / norm * v[key * 2 + d])
                        .sum::<f32>();
                    total += y * dy[token * 2 + d];
                }
            }
            total
        };
        let mut arena = CudaTensorArena::new(7).unwrap();
        arena.upload_f32(0, &q).unwrap();
        arena.upload_f32(1, &k).unwrap();
        arena.upload_f32(2, &v).unwrap();
        arena.upload_f32(3, &dy).unwrap();
        arena
            .attention_backward_f32(0, 1, 2, 3, 4, 5, 6, 1, 3, 1, 2, true)
            .unwrap();
        arena.synchronize().unwrap();
        let mut dq = [0.0_f32; 6];
        let mut dk = [0.0_f32; 6];
        let mut dv = [0.0_f32; 6];
        arena.download_f32(4, &mut dq).unwrap();
        arena.download_f32(5, &mut dk).unwrap();
        arena.download_f32(6, &mut dv).unwrap();
        let eps = 1.0e-3;
        for i in 0..6 {
            let mut plus = q;
            let mut minus = q;
            plus[i] += eps;
            minus[i] -= eps;
            let numerical = (objective(&plus, &k, &v) - objective(&minus, &k, &v)) / (2.0 * eps);
            assert!(
                (dq[i] - numerical).abs() < 4.0e-3,
                "q {i}: {} != {numerical}",
                dq[i]
            );
            let mut plus = k;
            let mut minus = k;
            plus[i] += eps;
            minus[i] -= eps;
            let numerical = (objective(&q, &plus, &v) - objective(&q, &minus, &v)) / (2.0 * eps);
            assert!(
                (dk[i] - numerical).abs() < 4.0e-3,
                "k {i}: {} != {numerical}",
                dk[i]
            );
            let mut plus = v;
            let mut minus = v;
            plus[i] += eps;
            minus[i] -= eps;
            let numerical = (objective(&q, &k, &plus) - objective(&q, &k, &minus)) / (2.0 * eps);
            assert!(
                (dv[i] - numerical).abs() < 4.0e-3,
                "v {i}: {} != {numerical}",
                dv[i]
            );
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_adamw_keeps_optimizer_state_on_device() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut expected_p = vec![1.0_f32, -2.0, 0.5];
        let mut expected_m = vec![0.0; 3];
        let mut expected_v = vec![0.0; 3];
        let gradients = [0.5_f32, -0.25, 1.0];
        let cpu = select_backend(Some("cpu")).unwrap();
        backend_adamw_step_f32(
            &cpu,
            &mut expected_p,
            &mut expected_m,
            &mut expected_v,
            &gradients,
            1,
            0.01,
            0.001,
        )
        .unwrap();
        let mut arena = CudaTensorArena::new(4).unwrap();
        arena.upload_f32(0, &[1.0_f32, -2.0, 0.5]).unwrap();
        arena.upload_f32(1, &[0.0_f32; 3]).unwrap();
        arena.upload_f32(2, &[0.0_f32; 3]).unwrap();
        arena.upload_f32(3, &gradients).unwrap();
        arena.adamw_step_f32(0, 1, 2, 3, 3, 1, 0.01, 0.001).unwrap();
        arena.synchronize().unwrap();
        let mut p = [0.0_f32; 3];
        let mut m = [0.0_f32; 3];
        let mut v = [0.0_f32; 3];
        arena.download_f32(0, &mut p).unwrap();
        arena.download_f32(1, &mut m).unwrap();
        arena.download_f32(2, &mut v).unwrap();
        for (actual, expected) in p
            .iter()
            .zip(expected_p)
            .chain(m.iter().zip(expected_m))
            .chain(v.iter().zip(expected_v))
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_clips_gradient_norm_without_host_round_trip() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let gradients = [3.0_f32, 4.0, 12.0];
        let mut arena = CudaTensorArena::new(2).unwrap();
        arena.upload_f32(0, &gradients).unwrap();
        arena
            .clip_gradient_l2_f32(0, 1, gradients.len(), 3.0)
            .unwrap();
        arena.synchronize().unwrap();
        let mut actual = [0.0_f32; 3];
        arena.download_f32(0, &mut actual).unwrap();
        let norm = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 3.0).abs() < 1.0e-5);
        for (value, expected) in actual.iter().zip([9.0 / 13.0, 12.0 / 13.0, 36.0 / 13.0]) {
            assert!((value - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_causal_convolution_matches_reference() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // [batch=1, time=4, nodes=2, channels=1]
        let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let weights = [2.0_f32, -0.5];
        let bias = [1.0_f32];
        let mut arena = CudaTensorArena::new(4).unwrap();
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &weights).unwrap();
        arena.upload_f32(2, &bias).unwrap();
        arena
            .causal_conv2_f32(0, 1, 2, 3, 1, 4, 2, 1, 1, 1)
            .unwrap();
        arena.synchronize().unwrap();
        let mut actual = [0.0_f32; 6];
        arena.download_f32(3, &mut actual).unwrap();
        assert_eq!(actual, [2.0, 11.0, 3.5, 26.0, 5.0, 41.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_causal_convolution_reads_resident_parameter_slice() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        // Deliberately keep unrelated values around the convolution slice.
        let parameters = [9.0_f32, 2.0, -0.5, 1.0, -7.0];
        let mut arena = CudaTensorArena::new(5).unwrap();
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &parameters).unwrap();
        arena
            .causal_conv2_parameter_slice_f32(0, 1, 1, 3, 2, 1, 4, 2, 1, 1, 1)
            .unwrap();
        arena.synchronize().unwrap();
        let mut actual = [0.0_f32; 6];
        arena.download_f32(2, &mut actual).unwrap();
        assert_eq!(actual, [2.0, 11.0, 3.5, 26.0, 5.0, 41.0]);
        arena.upload_f32(3, &[1.0_f32; 6]).unwrap();
        arena.fill_f32(4, parameters.len(), 0.0).unwrap();
        arena
            .causal_conv2_parameter_slice_backward_f32(0, 1, 1, 3, 3, 2, 4, 1, 4, 2, 1, 1, 1)
            .unwrap();
        let mut input_gradient = [0.0_f32; 8];
        let mut parameter_gradient = [0.0_f32; 5];
        arena.download_f32(2, &mut input_gradient).unwrap();
        arena.download_f32(4, &mut parameter_gradient).unwrap();
        assert_eq!(input_gradient, [2.0, 2.0, 1.5, 1.5, 1.5, 1.5, -0.5, -0.5]);
        assert_eq!(parameter_gradient, [0.0, 66.0, 99.0, 6.0, 0.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_lsttn_long_conv_pool_matches_reference() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // One channel: convolution centers are 0 and 2, then the pool keeps
        // the maximum GELU response. The parameter vector is a true slice.
        let mut arena = CudaTensorArena::new(6).unwrap();
        arena.upload_f32(0, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        arena.upload_f32(1, &[1.0_f32, 1.0, 1.0, 0.0]).unwrap();
        arena
            .lsttn_long_conv_pool_parameter_slice_f32(0, 1, 0, 3, 2, 1, 4, 1, 1, 1)
            .unwrap();
        let mut actual = [0.0_f32; 1];
        arena.download_f32(2, &mut actual).unwrap();
        let gelu = |value: f32| {
            0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
        };
        assert!((actual[0] - gelu(9.0)).abs() < 1e-5);
        arena.upload_f32(3, &[1.0_f32]).unwrap();
        arena.fill_f32(4, 4, 0.0).unwrap();
        arena
            .lsttn_long_conv_pool_parameter_slice_backward_f32(0, 1, 0, 3, 3, 5, 4, 1, 4, 1, 1, 1)
            .unwrap();
        let mut input_gradient = [0.0_f32; 4];
        let mut parameter_gradient = [0.0_f32; 4];
        arena.download_f32(5, &mut input_gradient).unwrap();
        arena.download_f32(4, &mut parameter_gradient).unwrap();
        let gelu_grad = |value: f32| {
            let c = 0.797_884_6;
            let u = c * (value + 0.044_715 * value.powi(3));
            let t = u.tanh();
            0.5 * (1.0 + t) + 0.5 * value * (1.0 - t * t) * c * (1.0 + 0.134_145 * value * value)
        };
        let g = gelu_grad(9.0);
        assert!((input_gradient[0]).abs() < 1.0e-5);
        for value in &input_gradient[1..] {
            assert!((*value - g).abs() < 1.0e-4);
        }
        assert!((parameter_gradient[0] - 2.0 * g).abs() < 1.0e-4);
        assert!((parameter_gradient[1] - 3.0 * g).abs() < 1.0e-4);
        assert!((parameter_gradient[2] - 4.0 * g).abs() < 1.0e-4);
        assert!((parameter_gradient[3] - g).abs() < 1.0e-4);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_lsttn_short_input_projection_pads_and_adds_time() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(3).unwrap();
        arena.upload_f32(0, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        // signal weight, time weight, bias
        arena.upload_f32(1, &[1.0_f32, 2.0, 0.0]).unwrap();
        let padded = arena
            .lsttn_short_input_projection_parameter_slice_f32(0, 1, 0, 2, 2, 1, 4, 1, 1, 2, 1, 0, 4)
            .unwrap();
        assert_eq!(padded, 13);
        let mut actual = vec![0.0_f32; 13];
        arena.download_f32(2, &mut actual).unwrap();
        assert!(actual[..11].iter().all(|value| *value == 0.0));
        assert!((actual[11] - 4.0).abs() < 1e-6);
        assert!((actual[12] - 5.5).abs() < 1e-6);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_adaptive_csr_logits_use_resident_node_embeddings() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // CSR rows: 0 -> {0,1}; 1 -> {0}.  Two unrelated sentinels ensure
        // the parameter offsets, rather than a copied host slice, are used.
        let parameters = [99.0_f32, 1.0, 2.0, 3.0, 4.0, -8.0, 5.0, 6.0, 7.0, 8.0];
        let mut arena = CudaTensorArena::new(4).unwrap();
        arena.upload_u32(0, &[0, 2, 3]).unwrap();
        arena.upload_u32(1, &[0, 1, 0]).unwrap();
        arena.upload_f32(0, &parameters).unwrap();
        arena
            .csr_adaptive_logits_parameter_slice_f32(0, 1, 0, 1, 6, 1, 2, 3, 2)
            .unwrap();
        arena.synchronize().unwrap();
        let mut logits = [0.0_f32; 3];
        arena.download_f32(1, &mut logits).unwrap();
        assert_eq!(logits, [17.0, 23.0, 39.0]);
        arena.csr_row_softmax_f32(0, 1, 2, 2, 3).unwrap();
        let mut weights = [0.0_f32; 3];
        arena.download_f32(2, &mut weights).unwrap();
        assert!((weights[0] + weights[1] - 1.0).abs() < 1.0e-6);
        assert!((weights[2] - 1.0).abs() < 1.0e-6);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_graph_wavenet_gate_matches_reference() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let filter = [-2.0_f32, 0.0, 1.5];
        let gate = [1.0_f32, -1.0, 0.5];
        let expected = filter
            .iter()
            .zip(gate)
            .map(|(f, g)| f.tanh() / (1.0 + (-g).exp()))
            .collect::<Vec<_>>();
        let mut arena = CudaTensorArena::new(5).unwrap();
        arena.upload_f32(0, &filter).unwrap();
        arena.upload_f32(1, &gate).unwrap();
        arena.gated_tanh_sigmoid_f32(0, 1, 2, filter.len()).unwrap();
        let mut actual = [0.0_f32; 3];
        arena.download_f32(2, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        arena.upload_f32(3, &[1.0_f32, 0.5, -0.25]).unwrap();
        arena
            .gated_tanh_sigmoid_backward_f32(0, 1, 3, 2, 4, filter.len())
            .unwrap();
        let mut filter_gradient = [0.0_f32; 3];
        let mut gate_gradient = [0.0_f32; 3];
        arena.download_f32(2, &mut filter_gradient).unwrap();
        arena.download_f32(4, &mut gate_gradient).unwrap();
        for i in 0..3 {
            let t = filter[i].tanh();
            let s = 1.0 / (1.0 + (-gate[i]).exp());
            let dy = [1.0_f32, 0.5, -0.25][i];
            assert!((filter_gradient[i] - dy * (1.0 - t * t) * s).abs() < 1.0e-6);
            assert!((gate_gradient[i] - dy * t * s * (1.0 - s)).abs() < 1.0e-6);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_csr_diffusion_backward_is_deterministic_and_sparse() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // row 0 <- {0,1}; row 1 <- {1}
        let mut arena = CudaTensorArena::new(6).unwrap();
        arena.upload_u32(0, &[0, 2, 3]).unwrap();
        arena.upload_u32(1, &[0, 1, 1]).unwrap();
        arena.upload_f32(0, &[2.0_f32, 3.0, 4.0]).unwrap();
        arena.upload_f32(1, &[11.0_f32, 13.0]).unwrap();
        arena.upload_f32(2, &[5.0_f32, 7.0]).unwrap();
        arena
            .csr_diffuse_backward_f32(0, 1, 0, 1, 2, 3, 4, 1, 2, 1, 3)
            .unwrap();
        let mut input_gradient = [0.0_f32; 2];
        let mut edge_gradient = [0.0_f32; 3];
        arena.download_f32(3, &mut input_gradient).unwrap();
        arena.download_f32(4, &mut edge_gradient).unwrap();
        assert_eq!(input_gradient, [10.0, 43.0]);
        assert_eq!(edge_gradient, [55.0, 65.0, 91.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_concatenates_channel_axis() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(5).unwrap();
        arena.upload_f32(0, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap(); // [2,2]
        arena
            .upload_f32(1, &[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0])
            .unwrap(); // [2,3]
        arena.concat_channels_f32(0, 1, 2, 2, 2, 3).unwrap();
        let mut actual = [0.0_f32; 10];
        arena.download_f32(2, &mut actual).unwrap();
        assert_eq!(
            actual,
            [1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0]
        );
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_accumulates_parameter_gradient_slice() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(2).unwrap();
        arena.upload_f32(0, &[1.0_f32, -2.0, 3.0]).unwrap();
        arena.upload_f32(1, &[10.0_f32; 6]).unwrap();
        arena.accumulate_parameter_slice_f32(0, 1, 2, 3).unwrap();
        let mut actual = [0.0_f32; 6];
        arena.download_f32(1, &mut actual).unwrap();
        assert_eq!(actual, [10.0, 10.0, 11.0, 8.0, 13.0, 10.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_transposes_node_and_time_axes() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // [batch=1, nodes=2, times=3, channels=1]
        let mut arena = CudaTensorArena::new(2).unwrap();
        arena
            .upload_f32(0, &[0.0_f32, 1.0, 2.0, 10.0, 11.0, 12.0])
            .unwrap();
        arena.transpose_node_time_f32(0, 1, 1, 2, 3, 1).unwrap();
        let mut actual = [0.0_f32; 6];
        arena.download_f32(1, &mut actual).unwrap();
        assert_eq!(actual, [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_gathers_visible_patch_tokens_for_pretraining() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // [batch=1, nodes=2, patches=4, hidden=1], gather patches 2 and 0.
        let values = [0.0_f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let mut arena = CudaTensorArena::new(3).unwrap();
        arena.upload_f32(0, &values).unwrap();
        arena.upload_u32(0, &[2, 0]).unwrap();
        arena
            .gather_patch_tokens_f32(0, 0, 1, 1, 2, 4, 2, 1)
            .unwrap();
        let mut actual = [0.0_f32; 4];
        arena.download_f32(1, &mut actual).unwrap();
        assert_eq!(actual, [2.0, 0.0, 12.0, 10.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_adds_causal_tail_per_batch() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        // [batch=2,time=3,nodes=1,channels=1], right has two times.
        let mut arena = CudaTensorArena::new(5).unwrap();
        arena
            .upload_f32(0, &[1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0])
            .unwrap();
        arena
            .upload_f32(1, &[100.0_f32, 200.0, 300.0, 400.0])
            .unwrap();
        arena.add_tail_time_f32(0, 1, 2, 2, 3, 2, 1, 1).unwrap();
        let mut actual = [0.0_f32; 4];
        arena.download_f32(2, &mut actual).unwrap();
        assert_eq!(actual, [102.0, 203.0, 320.0, 430.0]);
        arena
            .add_tail_time_backward_f32(2, 3, 4, 2, 3, 2, 1, 1)
            .unwrap();
        let mut left_gradient = [0.0_f32; 6];
        let mut right_gradient = [0.0_f32; 4];
        arena.download_f32(3, &mut left_gradient).unwrap();
        arena.download_f32(4, &mut right_gradient).unwrap();
        assert_eq!(left_gradient, [0.0, 102.0, 203.0, 0.0, 320.0, 430.0]);
        assert_eq!(right_gradient, actual);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_deterministic_dropout_repeats_native_mask() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let values = [1.0_f32; 32];
        let seed = 0x1234_5678_9abc_def0_u64;
        let base = 19usize;
        let mut expected = [0.0_f32; 32];
        for (index, value) in expected.iter_mut().enumerate() {
            let mut state = seed ^ ((base + index) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = if state % 10_000 < 3_000 {
                0.0
            } else {
                1.0 / 0.7
            };
        }
        let mut arena = CudaTensorArena::new(2).unwrap();
        arena.upload_f32(0, &values).unwrap();
        arena
            .deterministic_dropout_f32(0, 1, values.len(), seed, base, true, 0.3)
            .unwrap();
        let mut actual = [0.0_f32; 32];
        arena.download_f32(1, &mut actual).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_batch_norms_channels_across_time_and_nodes() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(6).unwrap();
        arena.upload_f32(0, &[1.0_f32, 10.0, 3.0, 14.0]).unwrap();
        // gamma[2], beta[2]
        arena.upload_f32(1, &[1.0_f32, 1.0, 0.0, 0.0]).unwrap();
        arena
            .batch_norm_channels_parameter_slice_f32(0, 1, 0, 2, 2, 3, 1, 2, 1, 2)
            .unwrap();
        let mut actual = [0.0_f32; 4];
        arena.download_f32(3, &mut actual).unwrap();
        for (actual, expected) in actual.into_iter().zip([-1.0_f32, -1.0, 1.0, 1.0]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        arena.upload_f32(4, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        arena.fill_f32(5, 4, 0.0).unwrap();
        arena
            .batch_norm_channels_parameter_slice_backward_f32(0, 1, 0, 2, 2, 4, 3, 5, 1, 2, 1, 2)
            .unwrap();
        let mut input_gradient = [0.0_f32; 4];
        let mut parameter_gradient = [0.0_f32; 4];
        arena.download_f32(3, &mut input_gradient).unwrap();
        arena.download_f32(5, &mut parameter_gradient).unwrap();
        assert!(input_gradient.iter().all(|value| value.abs() < 1.0e-4));
        assert!((parameter_gradient[0] - 2.0).abs() < 1.0e-4);
        assert!((parameter_gradient[1] - 2.0).abs() < 1.0e-4);
        assert_eq!(&parameter_gradient[2..], &[4.0, 6.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_masked_inverse_scale_mae_has_zero_masked_gradients() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(4).unwrap();
        arena.upload_f32(0, &[2.0_f32, 3.0, 4.0]).unwrap();
        arena.upload_f32(1, &[1.0_f32, 0.0, 5.0]).unwrap();
        arena
            .masked_inverse_scale_mae_loss_backward_f32(0, 1, 2, 3, 3, 0.0, 2.0)
            .unwrap();
        let mut gradient = [0.0_f32; 3];
        let mut loss = [0.0_f32; 2];
        arena.download_f32(2, &mut gradient).unwrap();
        arena.download_f32(3, &mut loss).unwrap();
        assert_eq!(gradient, [1.0, 0.0, -1.0]);
        assert!((loss[0] - 2.0).abs() < 1e-6);
        assert_eq!(loss[1], 2.0);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_affine_parameter_slice_backward_accumulates_global_gradient() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(5).unwrap();
        arena.upload_f32(0, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        arena
            .upload_f32(1, &[9.0_f32, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
            .unwrap();
        arena.upload_f32(2, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        arena.upload_f32(3, &[0.0_f32; 7]).unwrap();
        arena
            .affine_backward_parameter_slice_f32(0, 1, 1, 5, 2, 4, 3, 2, 2, 2)
            .unwrap();
        let mut dx = [0.0_f32; 4];
        let mut dp = [0.0_f32; 7];
        arena.download_f32(4, &mut dx).unwrap();
        arena.download_f32(3, &mut dp).unwrap();
        assert_eq!(dx, [5.0, 11.0, 11.0, 25.0]);
        assert_eq!(dp, [0.0, 10.0, 14.0, 14.0, 20.0, 4.0, 6.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_relu_backward_masks_nonpositive_inputs() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let mut arena = CudaTensorArena::new(3).unwrap();
        arena.upload_f32(0, &[-2.0_f32, 0.0, 3.0]).unwrap();
        arena.upload_f32(1, &[4.0_f32, -5.0, 6.0]).unwrap();
        arena.relu_backward_f32(0, 1, 2, 3).unwrap();
        let mut gradient = [0.0_f32; 3];
        arena.download_f32(2, &mut gradient).unwrap();
        assert_eq!(gradient, [0.0, 0.0, 6.0]);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_tensor_arena_causal_convolution_backward_matches_finite_difference() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let weights = [2.0_f32, -0.5];
        let output_gradient = [0.5_f32, -1.0, 0.25, 2.0, -0.5, 0.75];
        let mut arena = CudaTensorArena::new(7).unwrap();
        arena.upload_f32(0, &input).unwrap();
        arena.upload_f32(1, &weights).unwrap();
        arena.upload_f32(2, &output_gradient).unwrap();
        arena
            .causal_conv2_backward_f32(0, 1, 2, 3, 4, 5, 1, 4, 2, 1, 1, 1)
            .unwrap();
        arena.synchronize().unwrap();
        let mut dx = [0.0_f32; 8];
        let mut dw = [0.0_f32; 2];
        let mut db = [0.0_f32; 1];
        arena.download_f32(3, &mut dx).unwrap();
        arena.download_f32(4, &mut dw).unwrap();
        arena.download_f32(5, &mut db).unwrap();
        let objective = |input: &[f32], weights: &[f32], bias: f32| {
            let mut total = 0.0;
            for time in 0..3 {
                for node in 0..2 {
                    let output = bias
                        + input[time * 2 + node] * weights[0]
                        + input[(time + 1) * 2 + node] * weights[1];
                    total += output * output_gradient[time * 2 + node];
                }
            }
            total
        };
        let epsilon = 1.0e-3;
        for index in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus, &weights, 1.0) - objective(&minus, &weights, 1.0))
                / (2.0 * epsilon);
            assert!(
                (dx[index] - numeric).abs() < 6.0e-3,
                "dx {index}: {} != {numeric}",
                dx[index]
            );
        }
        for index in 0..weights.len() {
            let mut plus = weights;
            let mut minus = weights;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric =
                (objective(&input, &plus, 1.0) - objective(&input, &minus, 1.0)) / (2.0 * epsilon);
            assert!((dw[index] - numeric).abs() < 6.0e-3);
        }
        let numeric = (objective(&input, &weights, 1.0 + epsilon)
            - objective(&input, &weights, 1.0 - epsilon))
            / (2.0 * epsilon);
        assert!((db[0] - numeric).abs() < 6.0e-3);
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_csr_row_softmax_and_backward_match_cpu() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let indptr = [0, 2, 2, 5];
        let logits = [0.25, -0.5, 1.0, 0.0, -0.75];
        let output_grad = [0.5, -1.0, 0.25, 2.0, -0.5];
        let cpu_weights = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        let cuda_weights = backend_csr_row_softmax_f32(&cuda, &indptr, &logits).unwrap();
        let cpu_backward =
            backend_csr_row_softmax_backward_f32(&cpu, &indptr, &cpu_weights, &output_grad)
                .unwrap();
        let cuda_backward =
            backend_csr_row_softmax_backward_f32(&cuda, &indptr, &cuda_weights, &output_grad)
                .unwrap();
        for (left, right) in cpu_weights.iter().zip(cuda_weights) {
            assert!((left - right).abs() < 1.0e-5);
        }
        for (left, right) in cpu_backward.iter().zip(cuda_backward) {
            assert!((left - right).abs() < 1.0e-5);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_adamw_matches_cpu() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let mut cpu_p = vec![1.0, -2.0, 0.5];
        let mut cpu_m = vec![0.0; 3];
        let mut cpu_v = vec![0.0; 3];
        let mut cuda_p = cpu_p.clone();
        let mut cuda_m = cpu_m.clone();
        let mut cuda_v = cpu_v.clone();
        let gradients = [0.5, -0.25, 1.0];
        backend_adamw_step_f32(
            &cpu, &mut cpu_p, &mut cpu_m, &mut cpu_v, &gradients, 1, 0.01, 0.001,
        )
        .unwrap();
        backend_adamw_step_f32(
            &cuda,
            &mut cuda_p,
            &mut cuda_m,
            &mut cuda_v,
            &gradients,
            1,
            0.01,
            0.001,
        )
        .unwrap();
        for (left, right) in cpu_p
            .iter()
            .zip(cuda_p)
            .chain(cpu_m.iter().zip(cuda_m))
            .chain(cpu_v.iter().zip(cuda_v))
        {
            assert!((left - right).abs() < 1.0e-5);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_layer_norm_matches_cpu() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let cuda = select_backend(Some("cuda")).unwrap();
        let values = [1.0, -2.0, 3.0, 4.0, -1.0, 0.5];
        let gamma = [1.0, 0.5, -0.25];
        let beta = [0.1, -0.2, 0.3];
        let expected = backend_layer_norm_f32(&cpu, &values, 2, 3, &gamma, &beta).unwrap();
        let actual = backend_layer_norm_f32(&cuda, &values, 2, 3, &gamma, &beta).unwrap();
        for (left, right) in expected.iter().zip(actual) {
            assert!((left - right).abs() < 1.0e-5);
        }
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_csr_workspace_reuses_same_shape_allocations() {
        if !available_backends().iter().any(|backend| backend == "cuda") {
            return;
        }
        let indptr = [0, 2, 3, 3];
        let indices = [0, 1, 2];
        let weights = [0.25, 0.75, -0.5];
        let values = [
            1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0,
        ];
        let mut workspace = CudaCsrDiffusionWorkspace::new(&indptr, &indices, &weights).unwrap();
        let first = workspace.diffuse(2, &values).unwrap();
        let allocations = workspace.allocation_count();
        let second = workspace.diffuse(2, &values).unwrap();
        assert_eq!(workspace.allocation_count(), allocations);
        assert_eq!(workspace.value_capacity(), values.len());
        assert_eq!(first, second);
    }

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn metal_csr_diffusion_and_backward_match_cpu() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let metal = select_backend(Some("metal")).unwrap();
        let indptr = [0, 2, 3];
        let indices = [0, 1, 1];
        let weights = [0.25, 0.75, -0.5];
        let values = [1.0, -2.0, 3.0, 4.0];
        let output_grad = [0.5, -0.25, 1.0, 2.0];
        let expected =
            backend_csr_diffusion_f32(&cpu, &indptr, &indices, &weights, 2, &values).unwrap();
        let actual =
            backend_csr_diffusion_f32(&metal, &indptr, &indices, &weights, 2, &values).unwrap();
        let expected_backward = backend_csr_diffusion_backward_f32(
            &cpu,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        let actual_backward = backend_csr_diffusion_backward_f32(
            &metal,
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &output_grad,
        )
        .unwrap();
        for (left, right) in expected
            .iter()
            .zip(actual)
            .chain(
                expected_backward
                    .input_grad
                    .iter()
                    .zip(actual_backward.input_grad),
            )
            .chain(
                expected_backward
                    .edge_grad
                    .iter()
                    .zip(actual_backward.edge_grad),
            )
        {
            assert!((left - right).abs() < 1.0e-5);
        }
    }

    #[cfg(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    ))]
    #[test]
    fn metal_softmax_adamw_and_layer_norm_match_cpu() {
        if !available_backends()
            .iter()
            .any(|backend| backend == "metal")
        {
            return;
        }
        let cpu = select_backend(Some("cpu")).unwrap();
        let metal = select_backend(Some("metal")).unwrap();
        let indptr = [0, 2, 4];
        let logits = [0.5, -1.0, 2.0, 0.25];
        let output_grad = [1.0, -0.5, 0.25, 2.0];
        let cpu_weights = backend_csr_row_softmax_f32(&cpu, &indptr, &logits).unwrap();
        let metal_weights = backend_csr_row_softmax_f32(&metal, &indptr, &logits).unwrap();
        let cpu_grad =
            backend_csr_row_softmax_backward_f32(&cpu, &indptr, &cpu_weights, &output_grad)
                .unwrap();
        let metal_grad =
            backend_csr_row_softmax_backward_f32(&metal, &indptr, &metal_weights, &output_grad)
                .unwrap();
        for (left, right) in cpu_weights
            .iter()
            .zip(metal_weights)
            .chain(cpu_grad.iter().zip(metal_grad))
        {
            assert!((left - right).abs() < 1.0e-5);
        }

        let mut cpu_p = vec![1.0, -2.0, 0.5];
        let mut cpu_m = vec![0.0; 3];
        let mut cpu_v = vec![0.0; 3];
        let mut metal_p = cpu_p.clone();
        let mut metal_m = cpu_m.clone();
        let mut metal_v = cpu_v.clone();
        let gradients = [0.5, -0.25, 1.0];
        backend_adamw_step_f32(
            &cpu, &mut cpu_p, &mut cpu_m, &mut cpu_v, &gradients, 1, 0.01, 0.001,
        )
        .unwrap();
        backend_adamw_step_f32(
            &metal,
            &mut metal_p,
            &mut metal_m,
            &mut metal_v,
            &gradients,
            1,
            0.01,
            0.001,
        )
        .unwrap();
        for (left, right) in cpu_p
            .iter()
            .zip(metal_p)
            .chain(cpu_m.iter().zip(metal_m))
            .chain(cpu_v.iter().zip(metal_v))
        {
            assert!((left - right).abs() < 1.0e-5);
        }

        let values = [1.0, -2.0, 3.0, 4.0, -1.0, 0.5];
        let gamma = [1.0, 0.5, -0.25];
        let beta = [0.1, -0.2, 0.3];
        let expected = backend_layer_norm_f32(&cpu, &values, 2, 3, &gamma, &beta).unwrap();
        let actual = backend_layer_norm_f32(&metal, &values, 2, 3, &gamma, &beta).unwrap();
        for (left, right) in expected.iter().zip(actual) {
            assert!((left - right).abs() < 1.0e-5);
        }
    }
}
