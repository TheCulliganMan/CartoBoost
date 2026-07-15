use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "rocm", target_os = "linux"))]
use std::ffi::{c_char, c_void, CString};
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use std::future::Future;
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use std::sync::mpsc;
#[cfg(any(
    all(feature = "webgpu", not(target_arch = "wasm32")),
    all(feature = "rocm", target_os = "linux")
))]
use std::sync::OnceLock;
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Instant;

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
pub use crate::cuda_oxide_backend::{CudaCsrDiffusionWorkspace, CudaTensorArena};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    Auto,
    Cpu,
    Cuda,
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
            "rocm" => Ok(Self::Rocm),
            "metal" => Ok(Self::Metal),
            "webgpu" => Ok(Self::Webgpu),
            other => Err(NeuralError::InvalidArgument(format!(
                "unknown compute backend {other:?}; expected auto, cpu, cuda, rocm, metal, or webgpu"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
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

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
static WEBGPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(all(feature = "rocm", target_os = "linux"))]
static ROCM_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_available() -> bool {
    *WEBGPU_AVAILABLE.get_or_init(|| webgpu_request_device().is_ok())
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
fn webgpu_available() -> bool {
    // Browser adapter discovery is asynchronous.  Synchronous model APIs must
    // not spin-wait on a JavaScript promise; use the async Wasm exports below.
    false
}

#[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
fn cuda_available() -> bool {
    crate::cuda_oxide_backend::is_available()
}

#[cfg(not(all(feature = "cuda-oxide", target_os = "linux")))]
fn cuda_available() -> bool {
    false
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_available() -> bool {
    *ROCM_AVAILABLE.get_or_init(rocm_probe)
}

pub fn available_backends() -> Vec<String> {
    let mut backends = vec!["cpu".to_string()];
    if cfg!(all(
        feature = "metal",
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos"
        )
    )) {
        backends.push("metal".to_string());
    }
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    if cuda_available() {
        backends.push("cuda".to_string());
    }
    #[cfg(all(feature = "rocm", target_os = "linux"))]
    if rocm_available() {
        backends.push("rocm".to_string());
    }
    #[cfg(feature = "webgpu")]
    if webgpu_available() {
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
        "metal" => metal_vector_add_report(selection, len),
        "rocm" => rocm_vector_add_report(selection, len),
        "webgpu" => webgpu_vector_add_report(selection, len),
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
        "metal" => metal_affine_scores(features, means, weights, intercepts),
        "rocm" => rocm_affine_scores(features, means, weights, intercepts),
        "webgpu" => webgpu_affine_scores(features, means, weights, intercepts),
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
        "metal" => with_metal_autoreleasepool(|| {
            metal_scalar_graph_f32(initial_values, opcodes, left, right)
        }),
        #[cfg(all(feature = "cuda-oxide", target_os = "linux"))]
        "cuda" => cuda_scalar_graph_f32(initial_values, opcodes, left, right),
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
        "metal" => with_metal_autoreleasepool(|| {
            metal_scalar_graph_train_step_f32(
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
        other => Err(NeuralError::InvalidArgument(format!(
            "backend {other:?} does not provide complete scalar-graph training"
        ))),
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
fn with_metal_autoreleasepool<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    objc::rc::autoreleasepool(operation)
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn with_metal_autoreleasepool<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    operation()
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

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
fn metal_scalar_graph_f32(
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    use metal::{
        CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;

    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void evaluate_scalar_graph_f32(
            device float* values [[buffer(0)]],
            const device uchar* opcodes [[buffer(1)]],
            const device uint* left [[buffer(2)]],
            const device uint* right [[buffer(3)]],
            const device uint* schedule [[buffer(4)]],
            constant uint& schedule_offset [[buffer(5)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint output = schedule[schedule_offset + id];
            uchar opcode = opcodes[output];
            float a = values[left[output]];
            float b = values[right[output]];
            switch (opcode) {
                case 2: values[output] = a + b; break;
                case 3: values[output] = a * b; break;
                case 4: values[output] = a / max(b, 1.0e-12f); break;
                case 5: values[output] = tanh(a); break;
                case 6: values[output] = exp(a); break;
                case 7: values[output] = sqrt(max(a, 1.0e-12f)); break;
                case 8: values[output] = sin(a); break;
                case 9: values[output] = 1.0f / (1.0f + exp(-a)); break;
                case 10: values[output] = max(a, b); break;
                case 11: values[output] = a; break;
                default: break;
            }
        }
    "#;

    struct MetalScalarContext {
        device: metal::Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
        buffers: Vec<Option<metal::Buffer>>,
    }
    impl MetalScalarContext {
        fn new() -> Result<Self> {
            let device = Device::system_default().ok_or_else(|| {
                NeuralError::InvalidArgument("no Metal device is available".to_string())
            })?;
            let library = device
                .new_library_with_source(SOURCE, &CompileOptions::new())
                .map_err(|error| {
                    NeuralError::InvalidArgument(format!(
                        "failed to compile Metal scalar-graph kernel: {error}"
                    ))
                })?;
            let function = library
                .get_function("evaluate_scalar_graph_f32", None)
                .map_err(|error| NeuralError::InvalidArgument(error.to_string()))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| NeuralError::InvalidArgument(error.to_string()))?;
            let queue = device.new_command_queue();
            Ok(Self {
                device,
                queue,
                pipeline,
                buffers: (0..6).map(|_| None).collect(),
            })
        }
    }
    fn upload<T>(context: &mut MetalScalarContext, slot: usize, values: &[T]) -> metal::Buffer {
        let bytes = std::mem::size_of_val(values).max(1) as u64;
        if context.buffers[slot]
            .as_ref()
            .is_none_or(|buffer| buffer.length() < bytes)
        {
            context.buffers[slot] = Some(
                context
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            );
        }
        let buffer = context.buffers[slot]
            .as_ref()
            .expect("reusable Metal scalar buffer")
            .clone();
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                std::mem::size_of_val(values),
            );
        }
        buffer
    }
    thread_local! {
        static METAL_SCALAR_CONTEXT: RefCell<Option<MetalScalarContext>> = const { RefCell::new(None) };
    }

    let mut levels = vec![0usize; opcodes.len()];
    let mut by_level = vec![Vec::<u32>::new()];
    for index in 0..opcodes.len() {
        if opcodes[index] <= 1 {
            continue;
        }
        let mut level = levels[left[index] as usize] + 1;
        if matches!(opcodes[index], 2 | 3 | 4 | 10) {
            level = level.max(levels[right[index] as usize] + 1);
        }
        levels[index] = level;
        if by_level.len() <= level {
            by_level.resize_with(level + 1, Vec::new);
        }
        by_level[level].push(index as u32);
    }
    let mut schedule = Vec::new();
    let mut offsets = Vec::with_capacity(by_level.len());
    for nodes in &by_level {
        offsets.push(schedule.len() as u32);
        schedule.extend_from_slice(nodes);
    }

    METAL_SCALAR_CONTEXT.with(|cell| {
        let mut maybe_context = cell.borrow_mut();
        if maybe_context.is_none() {
            *maybe_context = Some(MetalScalarContext::new()?);
        }
        let context = maybe_context
            .as_mut()
            .expect("initialized Metal scalar context");
        let values_buffer = upload(context, 0, initial_values);
        let opcode_buffer = upload(context, 1, opcodes);
        let left_buffer = upload(context, 2, left);
        let right_buffer = upload(context, 3, right);
        let schedule_buffer = upload(context, 4, &schedule);
        let offset_buffer = upload(context, 5, &offsets);
        let command = context.queue.new_command_buffer();
        for (level, nodes) in by_level.iter().enumerate().skip(1) {
            if nodes.is_empty() {
                continue;
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&context.pipeline);
            encoder.set_buffer(0, Some(&values_buffer), 0);
            encoder.set_buffer(1, Some(&opcode_buffer), 0);
            encoder.set_buffer(2, Some(&left_buffer), 0);
            encoder.set_buffer(3, Some(&right_buffer), 0);
            encoder.set_buffer(4, Some(&schedule_buffer), 0);
            encoder.set_buffer(
                5,
                Some(&offset_buffer),
                (level * std::mem::size_of::<u32>()) as u64,
            );
            let width = context
                .pipeline
                .thread_execution_width()
                .max(1)
                .min(nodes.len() as u64);
            encoder.dispatch_threads(
                MTLSize::new(nodes.len() as u64, 1, 1),
                MTLSize::new(width, 1, 1),
            );
            encoder.end_encoding();
        }
        command.commit();
        command.wait_until_completed();
        let output = unsafe {
            std::slice::from_raw_parts(values_buffer.contents().cast::<f32>(), initial_values.len())
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(NeuralError::InvalidArgument(
                "Metal scalar-graph inference produced a non-finite value".to_string(),
            ));
        }
        Ok(output.to_vec())
    })
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
#[allow(clippy::too_many_arguments)]
fn metal_scalar_graph_train_step_f32(
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
    use metal::{
        CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;

    let mut levels = vec![0usize; opcodes.len()];
    let mut by_level = vec![Vec::<u32>::new()];
    for index in 0..opcodes.len() {
        if opcodes[index] <= 1 {
            continue;
        }
        let mut level = levels[left[index] as usize] + 1;
        if matches!(opcodes[index], 2 | 3 | 4 | 10) {
            level = level.max(levels[right[index] as usize] + 1);
        }
        levels[index] = level;
        if by_level.len() <= level {
            by_level.resize_with(level + 1, Vec::new);
        }
        by_level[level].push(index as u32);
    }
    let mut schedule = Vec::new();
    let mut offsets = Vec::with_capacity(by_level.len());
    for nodes in &by_level {
        offsets.push(schedule.len() as u32);
        schedule.extend_from_slice(nodes);
    }
    let mut gradients = vec![0.0f32; opcodes.len()];
    gradients[loss] = 1.0;
    let parameter_gradients = vec![0.0f32; parameters.len()];

    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void evaluate_scalar_graph_f32(
            device float* values [[buffer(0)]],
            const device uchar* opcodes [[buffer(1)]],
            const device uint* left [[buffer(2)]],
            const device uint* right [[buffer(3)]],
            const device uint* schedule [[buffer(4)]],
            constant uint& schedule_offset [[buffer(5)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint output = schedule[schedule_offset + id];
            uchar opcode = opcodes[output];
            float a = values[left[output]];
            float b = values[right[output]];
            switch (opcode) {
                case 2: values[output] = a + b; break;
                case 3: values[output] = a * b; break;
                case 4: values[output] = a / max(b, 1.0e-12f); break;
                case 5: values[output] = tanh(a); break;
                case 6: values[output] = exp(a); break;
                case 7: values[output] = sqrt(max(a, 1.0e-12f)); break;
                case 8: values[output] = sin(a); break;
                case 9: values[output] = 1.0f / (1.0f + exp(-a)); break;
                case 10: values[output] = max(a, b); break;
                case 11: values[output] = a; break;
                default: break;
            }
        }

        kernel void backward_scalar_graph_f32(
            const device float* values [[buffer(0)]],
            const device uchar* opcodes [[buffer(1)]],
            const device uint* left [[buffer(2)]],
            const device uint* right [[buffer(3)]],
            const device uint* schedule [[buffer(4)]],
            constant uint& schedule_offset [[buffer(5)]],
            device atomic_float* gradients [[buffer(6)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint node = schedule[schedule_offset + id];
            float gradient = atomic_load_explicit(&gradients[node], memory_order_relaxed);
            uint lhs = left[node];
            uint rhs = right[node];
            switch (opcodes[node]) {
                case 2:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient, memory_order_relaxed);
                    atomic_fetch_add_explicit(&gradients[rhs], gradient, memory_order_relaxed);
                    break;
                case 3:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient * values[rhs], memory_order_relaxed);
                    atomic_fetch_add_explicit(&gradients[rhs], gradient * values[lhs], memory_order_relaxed);
                    break;
                case 4: {
                    float denominator = max(values[rhs], 1.0e-12f);
                    atomic_fetch_add_explicit(&gradients[lhs], gradient / denominator, memory_order_relaxed);
                    atomic_fetch_add_explicit(&gradients[rhs], -gradient * values[lhs] / (denominator * denominator), memory_order_relaxed);
                    break;
                }
                case 5:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient * (1.0f - values[node] * values[node]), memory_order_relaxed);
                    break;
                case 6:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient * values[node], memory_order_relaxed);
                    break;
                case 7:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient / (2.0f * max(values[node], 1.0e-12f)), memory_order_relaxed);
                    break;
                case 8:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient * cos(values[lhs]), memory_order_relaxed);
                    break;
                case 9:
                    atomic_fetch_add_explicit(&gradients[lhs], gradient * values[node] * (1.0f - values[node]), memory_order_relaxed);
                    break;
                case 10:
                    atomic_fetch_add_explicit(
                        &gradients[values[lhs] >= values[rhs] ? lhs : rhs],
                        gradient,
                        memory_order_relaxed
                    );
                    break;
                default: break;
            }
        }

        kernel void gather_parameter_gradients_f32(
            const device uchar* opcodes [[buffer(0)]],
            const device uint* parameter_ids [[buffer(1)]],
            const device atomic_float* gradients [[buffer(2)]],
            device atomic_float* parameter_gradients [[buffer(3)]],
            uint node [[thread_position_in_grid]]
        ) {
            if (opcodes[node] == 1) {
                float gradient = atomic_load_explicit(&gradients[node], memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &parameter_gradients[parameter_ids[node]],
                    gradient,
                    memory_order_relaxed
                );
            }
        }

        kernel void adamw_scalar_graph_f32(
            device float* parameters [[buffer(0)]],
            device float* first [[buffer(1)]],
            device float* second [[buffer(2)]],
            const device atomic_float* gradients [[buffer(3)]],
            constant float& learning_rate [[buffer(4)]],
            constant float& weight_decay [[buffer(5)]],
            constant float& first_correction [[buffer(6)]],
            constant float& second_correction [[buffer(7)]],
            uint parameter [[thread_position_in_grid]]
        ) {
            float gradient = atomic_load_explicit(&gradients[parameter], memory_order_relaxed)
                + weight_decay * parameters[parameter];
            float m = 0.9f * first[parameter] + 0.1f * gradient;
            float v = 0.999f * second[parameter] + 0.001f * gradient * gradient;
            first[parameter] = m;
            second[parameter] = v;
            parameters[parameter] -= learning_rate
                * (m / first_correction)
                / (sqrt(v / second_correction) + 1.0e-8f);
        }
    "#;

    struct MetalScalarTrainingContext {
        device: metal::Device,
        queue: CommandQueue,
        forward: ComputePipelineState,
        backward: ComputePipelineState,
        gather: ComputePipelineState,
        adam: ComputePipelineState,
        buffers: Vec<Option<metal::Buffer>>,
    }
    impl MetalScalarTrainingContext {
        fn new() -> Result<Self> {
            let device = Device::system_default().ok_or_else(|| {
                NeuralError::InvalidArgument("no Metal device is available".to_string())
            })?;
            let library = device
                .new_library_with_source(SOURCE, &CompileOptions::new())
                .map_err(|error| {
                    NeuralError::InvalidArgument(format!(
                        "failed to compile Metal scalar-graph training kernels: {error}"
                    ))
                })?;
            let pipeline = |name| -> Result<ComputePipelineState> {
                let function = library
                    .get_function(name, None)
                    .map_err(|error| NeuralError::InvalidArgument(error.to_string()))?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|error| NeuralError::InvalidArgument(error.to_string()))
            };
            let forward = pipeline("evaluate_scalar_graph_f32")?;
            let backward = pipeline("backward_scalar_graph_f32")?;
            let gather = pipeline("gather_parameter_gradients_f32")?;
            let adam = pipeline("adamw_scalar_graph_f32")?;
            let queue = device.new_command_queue();
            Ok(Self {
                device,
                queue,
                forward,
                backward,
                gather,
                adam,
                buffers: (0..16).map(|_| None).collect(),
            })
        }
    }
    fn upload_training<T>(
        context: &mut MetalScalarTrainingContext,
        slot: usize,
        values: &[T],
    ) -> metal::Buffer {
        let bytes = std::mem::size_of_val(values).max(1) as u64;
        if context.buffers[slot]
            .as_ref()
            .is_none_or(|buffer| buffer.length() < bytes)
        {
            context.buffers[slot] = Some(
                context
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            );
        }
        let buffer = context.buffers[slot]
            .as_ref()
            .expect("reusable Metal training buffer")
            .clone();
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                std::mem::size_of_val(values),
            );
        }
        buffer
    }
    thread_local! {
        static METAL_SCALAR_TRAINING_CONTEXT: RefCell<Option<MetalScalarTrainingContext>> = const { RefCell::new(None) };
    }

    METAL_SCALAR_TRAINING_CONTEXT.with(|cell| {
        let mut maybe_context = cell.borrow_mut();
        if maybe_context.is_none() {
            *maybe_context = Some(MetalScalarTrainingContext::new()?);
        }
        let context = maybe_context
            .as_mut()
            .expect("initialized Metal training context");
        let values_buffer = upload_training(context, 0, initial_values);
        let opcode_buffer = upload_training(context, 1, opcodes);
        let left_buffer = upload_training(context, 2, left);
        let right_buffer = upload_training(context, 3, right);
        let schedule_buffer = upload_training(context, 4, &schedule);
        let offset_buffer = upload_training(context, 5, &offsets);
        let gradient_buffer = upload_training(context, 6, &gradients);
        let parameter_id_buffer = upload_training(context, 7, parameter_ids);
        let parameter_gradient_buffer = upload_training(context, 8, &parameter_gradients);
        let parameter_buffer = upload_training(context, 9, parameters);
        let first_buffer = upload_training(context, 10, first_moment);
        let second_buffer = upload_training(context, 11, second_moment);
        let learning_buffer = upload_training(context, 12, std::slice::from_ref(&learning_rate));
        let weight_decay_buffer = upload_training(context, 13, std::slice::from_ref(&weight_decay));
        let first_correction = 1.0f32 - 0.9f32.powf(step as f32);
        let second_correction = 1.0f32 - 0.999f32.powf(step as f32);
        let first_correction_buffer =
            upload_training(context, 14, std::slice::from_ref(&first_correction));
        let second_correction_buffer =
            upload_training(context, 15, std::slice::from_ref(&second_correction));
        let command = context.queue.new_command_buffer();
        for (level, nodes) in by_level.iter().enumerate().skip(1) {
            if nodes.is_empty() {
                continue;
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&context.forward);
            encoder.set_buffer(0, Some(&values_buffer), 0);
            encoder.set_buffer(1, Some(&opcode_buffer), 0);
            encoder.set_buffer(2, Some(&left_buffer), 0);
            encoder.set_buffer(3, Some(&right_buffer), 0);
            encoder.set_buffer(4, Some(&schedule_buffer), 0);
            encoder.set_buffer(
                5,
                Some(&offset_buffer),
                (level * std::mem::size_of::<u32>()) as u64,
            );
            let width = context
                .forward
                .thread_execution_width()
                .max(1)
                .min(nodes.len() as u64);
            encoder.dispatch_threads(
                MTLSize::new(nodes.len() as u64, 1, 1),
                MTLSize::new(width, 1, 1),
            );
            encoder.end_encoding();
        }
        for level in (1..by_level.len()).rev() {
            let nodes = &by_level[level];
            if nodes.is_empty() {
                continue;
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&context.backward);
            encoder.set_buffer(0, Some(&values_buffer), 0);
            encoder.set_buffer(1, Some(&opcode_buffer), 0);
            encoder.set_buffer(2, Some(&left_buffer), 0);
            encoder.set_buffer(3, Some(&right_buffer), 0);
            encoder.set_buffer(4, Some(&schedule_buffer), 0);
            encoder.set_buffer(
                5,
                Some(&offset_buffer),
                (level * std::mem::size_of::<u32>()) as u64,
            );
            encoder.set_buffer(6, Some(&gradient_buffer), 0);
            let width = context
                .backward
                .thread_execution_width()
                .max(1)
                .min(nodes.len() as u64);
            encoder.dispatch_threads(
                MTLSize::new(nodes.len() as u64, 1, 1),
                MTLSize::new(width, 1, 1),
            );
            encoder.end_encoding();
        }
        let gather_encoder = command.new_compute_command_encoder();
        gather_encoder.set_compute_pipeline_state(&context.gather);
        gather_encoder.set_buffer(0, Some(&opcode_buffer), 0);
        gather_encoder.set_buffer(1, Some(&parameter_id_buffer), 0);
        gather_encoder.set_buffer(2, Some(&gradient_buffer), 0);
        gather_encoder.set_buffer(3, Some(&parameter_gradient_buffer), 0);
        let gather_width = context
            .gather
            .thread_execution_width()
            .max(1)
            .min(opcodes.len() as u64);
        gather_encoder.dispatch_threads(
            MTLSize::new(opcodes.len() as u64, 1, 1),
            MTLSize::new(gather_width, 1, 1),
        );
        gather_encoder.end_encoding();
        let adam_encoder = command.new_compute_command_encoder();
        adam_encoder.set_compute_pipeline_state(&context.adam);
        adam_encoder.set_buffer(0, Some(&parameter_buffer), 0);
        adam_encoder.set_buffer(1, Some(&first_buffer), 0);
        adam_encoder.set_buffer(2, Some(&second_buffer), 0);
        adam_encoder.set_buffer(3, Some(&parameter_gradient_buffer), 0);
        adam_encoder.set_buffer(4, Some(&learning_buffer), 0);
        adam_encoder.set_buffer(5, Some(&weight_decay_buffer), 0);
        adam_encoder.set_buffer(6, Some(&first_correction_buffer), 0);
        adam_encoder.set_buffer(7, Some(&second_correction_buffer), 0);
        let adam_width = context
            .adam
            .thread_execution_width()
            .max(1)
            .min(parameters.len() as u64);
        adam_encoder.dispatch_threads(
            MTLSize::new(parameters.len() as u64, 1, 1),
            MTLSize::new(adam_width, 1, 1),
        );
        adam_encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let loss_value = unsafe { *values_buffer.contents().cast::<f32>().add(loss) };
        unsafe {
            parameters.copy_from_slice(std::slice::from_raw_parts(
                parameter_buffer.contents().cast::<f32>(),
                parameters.len(),
            ));
            first_moment.copy_from_slice(std::slice::from_raw_parts(
                first_buffer.contents().cast::<f32>(),
                first_moment.len(),
            ));
            second_moment.copy_from_slice(std::slice::from_raw_parts(
                second_buffer.contents().cast::<f32>(),
                second_moment.len(),
            ));
        }
        if parameters
            .iter()
            .chain(first_moment.iter())
            .chain(second_moment.iter())
            .any(|value| !value.is_finite())
        {
            return Err(NeuralError::InvalidArgument(
                "Metal scalar-graph optimizer produced non-finite state".to_string(),
            ));
        }
        Ok(loss_value)
    })
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn metal_scalar_graph_f32(
    _initial_values: &[f32],
    _opcodes: &[u8],
    _left: &[u32],
    _right: &[u32],
) -> Result<Vec<f32>> {
    Err(NeuralError::InvalidArgument(
        "Metal scalar-graph inference is not available in this build".to_string(),
    ))
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
#[allow(clippy::too_many_arguments)]
fn metal_scalar_graph_train_step_f32(
    _initial_values: &[f32],
    _opcodes: &[u8],
    _left: &[u32],
    _right: &[u32],
    _parameter_ids: &[u32],
    _loss: usize,
    _parameters: &mut [f32],
    _first_moment: &mut [f32],
    _second_moment: &mut [f32],
    _step: u64,
    _learning_rate: f32,
    _weight_decay: f32,
) -> Result<f32> {
    Err(NeuralError::InvalidArgument(
        "Metal scalar-graph training is not available in this build".to_string(),
    ))
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
        "metal" => metal_dense_layer_f32(features, weights, biases),
        "rocm" => rocm_dense_layer_f32(features, weights, biases),
        "webgpu" => webgpu_dense_layer_f32(features, weights, biases),
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
        "metal" => metal_pair_sigmoid_scores_f32(embeddings, pairs),
        "rocm" => rocm_pair_sigmoid_scores_f32(embeddings, pairs),
        "webgpu" => webgpu_pair_sigmoid_scores_f32(embeddings, pairs),
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
        "metal" => metal_train_tanh_mlp_f32(
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

fn validate_dense_layer_inputs(
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

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
fn metal_vector_add_report(
    selection: BackendSelection,
    len: usize,
) -> Result<BackendDispatchReport> {
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add_f32(
            const device float* left [[buffer(0)]],
            const device float* right [[buffer(1)]],
            device float* output [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            output[id] = left[id] + right[id];
        }
    "#;

    let left = (0..len).map(|idx| idx as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|idx| idx as f32 * 1.5).collect::<Vec<_>>();
    let byte_len = (len * std::mem::size_of::<f32>()) as u64;
    let start = Instant::now();
    let device = Device::system_default()
        .ok_or_else(|| NeuralError::InvalidArgument("no Metal device is available".to_string()))?;
    let library = device
        .new_library_with_source(SOURCE, &CompileOptions::new())
        .map_err(|err| {
            NeuralError::InvalidArgument(format!("failed to compile Metal kernel: {err}"))
        })?;
    let kernel = library
        .get_function("vector_add_f32", None)
        .map_err(|err| {
            NeuralError::InvalidArgument(format!("failed to load Metal kernel: {err}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|err| {
            NeuralError::InvalidArgument(format!("failed to create Metal pipeline: {err}"))
        })?;
    let command_queue = device.new_command_queue();
    let left_buffer = device.new_buffer_with_data(
        left.as_ptr().cast(),
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let right_buffer = device.new_buffer_with_data(
        right.as_ptr().cast(),
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&left_buffer), 0);
    encoder.set_buffer(1, Some(&right_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    let threads = pipeline.thread_execution_width().max(1).min(len as u64);
    encoder.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let output = unsafe { std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), len) };
    let checksum = output.iter().map(|value| *value as f64).sum::<f64>();
    Ok(BackendDispatchReport {
        requested: selection.requested,
        selected: selection.selected,
        operation: "vector_add_f32".to_string(),
        len,
        checksum,
        expected_checksum: expected_vector_add_checksum(len),
        elapsed_ms,
        accelerated: true,
    })
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn metal_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "Metal dispatch is not available in this build".to_string(),
    ))
}

fn expected_vector_add_checksum(len: usize) -> f64 {
    (len as f64) * ((len - 1) as f64)
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
fn metal_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    use metal::{
        CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;

    const AFFINE_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void affine_scores_f32(
            const device float* features [[buffer(0)]],
            const device float* means [[buffer(1)]],
            const device float* weights [[buffer(2)]],
            const device float* intercepts [[buffer(3)]],
            device float* output [[buffer(4)]],
            constant uint& cols [[buffer(5)]],
            uint row [[thread_position_in_grid]]
        ) {
            float score = intercepts[row];
            uint offset = row * cols;
            for (uint col = 0; col < cols; col++) {
                score += (features[offset + col] - means[col]) * weights[col];
            }
            output[row] = score;
        }
    "#;

    struct MetalAffineContext {
        device: metal::Device,
        command_queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    impl MetalAffineContext {
        fn new() -> Result<Self> {
            let device = Device::system_default().ok_or_else(|| {
                NeuralError::InvalidArgument("no Metal device is available".to_string())
            })?;
            let library = device
                .new_library_with_source(AFFINE_SOURCE, &CompileOptions::new())
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to compile Metal affine kernel: {err}"
                    ))
                })?;
            let kernel = library
                .get_function("affine_scores_f32", None)
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to load Metal affine kernel: {err}"
                    ))
                })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to create Metal affine pipeline: {err}"
                    ))
                })?;
            let command_queue = device.new_command_queue();
            Ok(Self {
                device,
                command_queue,
                pipeline,
            })
        }
    }

    thread_local! {
        static METAL_AFFINE_CONTEXT: RefCell<Option<MetalAffineContext>> = const { RefCell::new(None) };
    }

    let rows = features.len();
    let cols = weights.len();
    let flat_features = features
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as f32))
        .collect::<Vec<_>>();
    let means = means.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let weights = weights
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let intercepts = intercepts
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let cols_param = cols as u32;
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    METAL_AFFINE_CONTEXT.with(|cell| {
        let mut maybe_context = cell.borrow_mut();
        if maybe_context.is_none() {
            *maybe_context = Some(MetalAffineContext::new()?);
        }
        let context = maybe_context
            .as_ref()
            .expect("initialized Metal affine context");
        let feature_buffer = context.device.new_buffer_with_data(
            flat_features.as_ptr().cast(),
            flat_features.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let mean_buffer = context.device.new_buffer_with_data(
            means.as_ptr().cast(),
            means.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = context.device.new_buffer_with_data(
            weights.as_ptr().cast(),
            weights.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let intercept_buffer = context.device.new_buffer_with_data(
            intercepts.as_ptr().cast(),
            intercepts.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = context.device.new_buffer(
            rows as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let cols_buffer = context.device.new_buffer_with_data(
            (&cols_param as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&context.pipeline);
        encoder.set_buffer(0, Some(&feature_buffer), 0);
        encoder.set_buffer(1, Some(&mean_buffer), 0);
        encoder.set_buffer(2, Some(&weight_buffer), 0);
        encoder.set_buffer(3, Some(&intercept_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&cols_buffer), 0);
        let threads = context
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(rows as u64);
        encoder.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(threads, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let output =
            unsafe { std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), rows) };
        Ok(output.iter().map(|value| *value as f64).collect())
    })
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
fn metal_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    use metal::{
        CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;

    const DENSE_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void dense_layer_f32(
            const device float* features [[buffer(0)]],
            const device float* weights [[buffer(1)]],
            const device float* biases [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& cols [[buffer(4)]],
            constant uint& out_dim [[buffer(5)]],
            uint2 pos [[thread_position_in_grid]]
        ) {
            uint row = pos.x;
            uint out = pos.y;
            float value = biases[out];
            uint feature_offset = row * cols;
            for (uint col = 0; col < cols; col++) {
                value += features[feature_offset + col] * weights[col * out_dim + out];
            }
            output[row * out_dim + out] = value;
        }
    "#;

    struct MetalDenseContext {
        device: metal::Device,
        command_queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    impl MetalDenseContext {
        fn new() -> Result<Self> {
            let device = Device::system_default().ok_or_else(|| {
                NeuralError::InvalidArgument("no Metal device is available".to_string())
            })?;
            let library = device
                .new_library_with_source(DENSE_SOURCE, &CompileOptions::new())
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to compile Metal dense layer kernel: {err}"
                    ))
                })?;
            let kernel = library
                .get_function("dense_layer_f32", None)
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to load Metal dense layer kernel: {err}"
                    ))
                })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to create Metal dense layer pipeline: {err}"
                    ))
                })?;
            let command_queue = device.new_command_queue();
            Ok(Self {
                device,
                command_queue,
                pipeline,
            })
        }
    }

    thread_local! {
        static METAL_DENSE_CONTEXT: RefCell<Option<MetalDenseContext>> = const { RefCell::new(None) };
    }

    let rows = features.len();
    let cols = features[0].len();
    let out_dim = biases.len();
    let flat_features = features
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let cols_param = cols as u32;
    let out_dim_param = out_dim as u32;
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    METAL_DENSE_CONTEXT.with(|cell| {
        let mut maybe_context = cell.borrow_mut();
        if maybe_context.is_none() {
            *maybe_context = Some(MetalDenseContext::new()?);
        }
        let context = maybe_context
            .as_ref()
            .expect("initialized Metal dense context");
        let feature_buffer = context.device.new_buffer_with_data(
            flat_features.as_ptr().cast(),
            flat_features.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = context.device.new_buffer_with_data(
            weights.as_ptr().cast(),
            weights.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let bias_buffer = context.device.new_buffer_with_data(
            biases.as_ptr().cast(),
            biases.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = context.device.new_buffer(
            (rows * out_dim) as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let cols_buffer = context.device.new_buffer_with_data(
            (&cols_param as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_dim_buffer = context.device.new_buffer_with_data(
            (&out_dim_param as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&context.pipeline);
        encoder.set_buffer(0, Some(&feature_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&bias_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_buffer(4, Some(&cols_buffer), 0);
        encoder.set_buffer(5, Some(&out_dim_buffer), 0);
        let threads_x = context
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(rows as u64);
        encoder.dispatch_threads(
            MTLSize::new(rows as u64, out_dim as u64, 1),
            MTLSize::new(threads_x, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), rows * out_dim)
        };
        Ok(output
            .chunks(out_dim)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>())
    })
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
fn metal_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    use metal::{
        CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;

    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    const PAIR_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void pair_sigmoid_scores_f32(
            const device float* embeddings [[buffer(0)]],
            const device uint2* pairs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& dim [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint source = pairs[id].x;
            uint target = pairs[id].y;
            uint source_offset = source * dim;
            uint target_offset = target * dim;
            float score = 0.0;
            for (uint col = 0; col < dim; col++) {
                score += embeddings[source_offset + col] * embeddings[target_offset + col];
            }
            output[id] = 1.0 / (1.0 + exp(-score));
        }
    "#;

    struct MetalPairContext {
        device: metal::Device,
        command_queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    impl MetalPairContext {
        fn new() -> Result<Self> {
            let device = Device::system_default().ok_or_else(|| {
                NeuralError::InvalidArgument("no Metal device is available".to_string())
            })?;
            let library = device
                .new_library_with_source(PAIR_SOURCE, &CompileOptions::new())
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to compile Metal pair scoring kernel: {err}"
                    ))
                })?;
            let kernel = library
                .get_function("pair_sigmoid_scores_f32", None)
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to load Metal pair scoring kernel: {err}"
                    ))
                })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|err| {
                    NeuralError::InvalidArgument(format!(
                        "failed to create Metal pair scoring pipeline: {err}"
                    ))
                })?;
            let command_queue = device.new_command_queue();
            Ok(Self {
                device,
                command_queue,
                pipeline,
            })
        }
    }

    thread_local! {
        static METAL_PAIR_CONTEXT: RefCell<Option<MetalPairContext>> = const { RefCell::new(None) };
    }

    let dim = embeddings[0].len();
    let flat_embeddings = embeddings
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let pair_indices = pairs
        .iter()
        .flat_map(|&(source, target)| [source as u32, target as u32])
        .collect::<Vec<_>>();
    let dim_param = dim as u32;
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    METAL_PAIR_CONTEXT.with(|cell| {
        let mut maybe_context = cell.borrow_mut();
        if maybe_context.is_none() {
            *maybe_context = Some(MetalPairContext::new()?);
        }
        let context = maybe_context
            .as_ref()
            .expect("initialized Metal pair scoring context");
        let embedding_buffer = context.device.new_buffer_with_data(
            flat_embeddings.as_ptr().cast(),
            flat_embeddings.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let pair_buffer = context.device.new_buffer_with_data(
            pair_indices.as_ptr().cast(),
            pair_indices.len() as u64 * std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = context.device.new_buffer(
            pairs.len() as u64 * f32_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let dim_buffer = context.device.new_buffer_with_data(
            (&dim_param as *const u32).cast(),
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&context.pipeline);
        encoder.set_buffer(0, Some(&embedding_buffer), 0);
        encoder.set_buffer(1, Some(&pair_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&dim_buffer), 0);
        let threads = context
            .pipeline
            .thread_execution_width()
            .max(1)
            .min(pairs.len() as u64);
        encoder.dispatch_threads(
            MTLSize::new(pairs.len() as u64, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), pairs.len())
        };
        Ok(output.iter().map(|value| f64::from(*value)).collect())
    })
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
fn metal_train_tanh_mlp_f32(
    inputs: &[Vec<f32>],
    targets: &[f32],
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    parameters: &mut [f32],
) -> Result<()> {
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void train_tanh_mlp_f32(
            const device float* inputs [[buffer(0)]],
            const device float* targets [[buffer(1)]],
            device float* parameters [[buffer(2)]],
            constant uint& rows [[buffer(3)]],
            constant uint& input_size [[buffer(4)]],
            constant uint& hidden_size [[buffer(5)]],
            constant uint& epochs [[buffer(6)]],
            constant float& learning_rate [[buffer(7)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id != 0) return;
            uint w1_offset = 0;
            uint b1_offset = hidden_size * input_size;
            uint w2_offset = b1_offset + hidden_size;
            uint b2_offset = w2_offset + hidden_size;
            for (uint epoch = 0; epoch < epochs; ++epoch) {
                for (uint row = 0; row < rows; ++row) {
                    float prediction = parameters[b2_offset];
                    for (uint hidden = 0; hidden < hidden_size; ++hidden) {
                        float value = parameters[b1_offset + hidden];
                        for (uint input = 0; input < input_size; ++input) {
                            value += parameters[w1_offset + hidden * input_size + input]
                                * inputs[row * input_size + input];
                        }
                        prediction += tanh(value) * parameters[w2_offset + hidden];
                    }
                    float error_gradient = 2.0f * (prediction - targets[row]);
                    parameters[b2_offset] -= learning_rate * error_gradient;
                    for (uint hidden = 0; hidden < hidden_size; ++hidden) {
                        float value = parameters[b1_offset + hidden];
                        for (uint input = 0; input < input_size; ++input) {
                            value += parameters[w1_offset + hidden * input_size + input]
                                * inputs[row * input_size + input];
                        }
                        float activation = tanh(value);
                        float old_w2 = parameters[w2_offset + hidden];
                        parameters[w2_offset + hidden] -= learning_rate * error_gradient * activation;
                        float gradient = error_gradient * old_w2 * (1.0f - activation * activation);
                        parameters[b1_offset + hidden] -= learning_rate * gradient;
                        for (uint input = 0; input < input_size; ++input) {
                            parameters[w1_offset + hidden * input_size + input] -= learning_rate * gradient
                                * inputs[row * input_size + input];
                        }
                    }
                }
            }
        }
    "#;

    let device = Device::system_default()
        .ok_or_else(|| NeuralError::InvalidArgument("no Metal device is available".to_string()))?;
    let library = device
        .new_library_with_source(SOURCE, &CompileOptions::new())
        .map_err(|err| {
            NeuralError::InvalidArgument(format!(
                "failed to compile Metal tanh-MLP training kernel: {err}"
            ))
        })?;
    let function = library
        .get_function("train_tanh_mlp_f32", None)
        .map_err(|err| {
            NeuralError::InvalidArgument(format!(
                "failed to load Metal tanh-MLP training kernel: {err}"
            ))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|err| {
            NeuralError::InvalidArgument(format!(
                "failed to create Metal tanh-MLP training pipeline: {err}"
            ))
        })?;
    let input_size = inputs[0].len() as u32;
    let rows = inputs.len() as u32;
    let hidden_size = hidden_size as u32;
    let epochs = epochs as u32;
    let flat_inputs = inputs.iter().flatten().copied().collect::<Vec<_>>();
    let queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeShared;
    let inputs_buffer = device.new_buffer_with_data(
        flat_inputs.as_ptr().cast(),
        std::mem::size_of_val(flat_inputs.as_slice()) as u64,
        options,
    );
    let targets_buffer = device.new_buffer_with_data(
        targets.as_ptr().cast(),
        std::mem::size_of_val(targets) as u64,
        options,
    );
    let parameter_buffer = device.new_buffer_with_data(
        parameters.as_ptr().cast(),
        std::mem::size_of_val(parameters) as u64,
        options,
    );
    let rows_buffer = device.new_buffer_with_data((&rows as *const u32).cast(), 4, options);
    let input_buffer = device.new_buffer_with_data((&input_size as *const u32).cast(), 4, options);
    let hidden_buffer =
        device.new_buffer_with_data((&hidden_size as *const u32).cast(), 4, options);
    let epochs_buffer = device.new_buffer_with_data((&epochs as *const u32).cast(), 4, options);
    let learning_buffer = device.new_buffer_with_data(
        (&learning_rate as *const f32).cast(),
        std::mem::size_of::<f32>() as u64,
        options,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&inputs_buffer), 0);
    encoder.set_buffer(1, Some(&targets_buffer), 0);
    encoder.set_buffer(2, Some(&parameter_buffer), 0);
    encoder.set_buffer(3, Some(&rows_buffer), 0);
    encoder.set_buffer(4, Some(&input_buffer), 0);
    encoder.set_buffer(5, Some(&hidden_buffer), 0);
    encoder.set_buffer(6, Some(&epochs_buffer), 0);
    encoder.set_buffer(7, Some(&learning_buffer), 0);
    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let trained = unsafe {
        std::slice::from_raw_parts(parameter_buffer.contents().cast::<f32>(), parameters.len())
    };
    parameters.copy_from_slice(trained);
    Ok(())
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn metal_affine_scores(
    _features: &[Vec<f64>],
    _means: &[f64],
    _weights: &[f64],
    _intercepts: &[f64],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "Metal affine scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn metal_dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "Metal dense layer scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn metal_pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "Metal pair scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
fn metal_train_tanh_mlp_f32(
    _inputs: &[Vec<f32>],
    _targets: &[f32],
    _hidden_size: usize,
    _epochs: usize,
    _learning_rate: f32,
    _parameters: &mut [f32],
) -> Result<()> {
    Err(NeuralError::InvalidArgument(
        "Metal tanh-MLP training is not available in this build".to_string(),
    ))
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

#[cfg(all(feature = "rocm", target_os = "linux"))]
type RocmError = i32;

#[cfg(all(feature = "rocm", target_os = "linux"))]
type HipModule = *mut c_void;

#[cfg(all(feature = "rocm", target_os = "linux"))]
type HipFunction = *mut c_void;

#[cfg(all(feature = "rocm", target_os = "linux"))]
type HiprtcProgram = *mut c_void;

#[cfg(all(feature = "rocm", target_os = "linux"))]
struct RocmRuntime {
    _hip_library: libloading::Library,
    _rtc_library: libloading::Library,
    hip_init: extern "C" fn(u32) -> RocmError,
    hip_get_device_count: extern "C" fn(*mut i32) -> RocmError,
    hip_set_device: extern "C" fn(i32) -> RocmError,
    hip_device_synchronize: extern "C" fn() -> RocmError,
    hip_malloc: extern "C" fn(*mut *mut c_void, usize) -> RocmError,
    hip_free: extern "C" fn(*mut c_void) -> RocmError,
    hip_memcpy_hto_d: extern "C" fn(*mut c_void, *const c_void, usize) -> RocmError,
    hip_memcpy_dto_h: extern "C" fn(*mut c_void, *const c_void, usize) -> RocmError,
    hip_module_load_data: extern "C" fn(*mut HipModule, *const c_void) -> RocmError,
    hip_module_unload: extern "C" fn(HipModule) -> RocmError,
    hip_module_get_function: extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> RocmError,
    hip_module_launch_kernel: extern "C" fn(
        HipFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> RocmError,
    hiprtc_create_program: extern "C" fn(
        *mut HiprtcProgram,
        *const c_char,
        *const c_char,
        i32,
        *const *const c_char,
        *const *const c_char,
    ) -> RocmError,
    hiprtc_compile_program: extern "C" fn(HiprtcProgram, i32, *const *const c_char) -> RocmError,
    hiprtc_get_code_size: extern "C" fn(HiprtcProgram, *mut usize) -> RocmError,
    hiprtc_get_code: extern "C" fn(HiprtcProgram, *mut c_void) -> RocmError,
    hiprtc_destroy_program: extern "C" fn(*mut HiprtcProgram) -> RocmError,
    hiprtc_get_program_log_size: extern "C" fn(HiprtcProgram, *mut usize) -> RocmError,
    hiprtc_get_program_log: extern "C" fn(HiprtcProgram, *mut c_char) -> RocmError,
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl RocmRuntime {
    fn new() -> Result<Self> {
        fn load_library(names: &[&str]) -> Result<libloading::Library> {
            for name in names {
                if let Ok(library) = unsafe { libloading::Library::new(name) } {
                    return Ok(library);
                }
            }
            Err(NeuralError::InvalidArgument(format!(
                "failed to load ROCm libraries from any of: {}",
                names.join(", ")
            )))
        }

        unsafe fn load_symbol<T: Copy>(library: &libloading::Library, symbol: &[u8]) -> Result<T> {
            library.get::<T>(symbol).map(|value| *value).map_err(|err| {
                NeuralError::InvalidArgument(format!(
                    "failed to load ROCm symbol {}: {err}",
                    String::from_utf8_lossy(symbol).trim_end_matches('\0')
                ))
            })
        }

        let hip_library = load_library(&["libamdhip64.so", "libamdhip64.so.6"])?;
        let rtc_library = load_library(&["libhiprtc.so", "libhiprtc.so.6"])?;
        Ok(Self {
            hip_init: unsafe { load_symbol(&hip_library, b"hipInit\0")? },
            hip_get_device_count: unsafe { load_symbol(&hip_library, b"hipGetDeviceCount\0")? },
            hip_set_device: unsafe { load_symbol(&hip_library, b"hipSetDevice\0")? },
            hip_device_synchronize: unsafe {
                load_symbol(&hip_library, b"hipDeviceSynchronize\0")?
            },
            hip_malloc: unsafe { load_symbol(&hip_library, b"hipMalloc\0")? },
            hip_free: unsafe { load_symbol(&hip_library, b"hipFree\0")? },
            hip_memcpy_hto_d: unsafe { load_symbol(&hip_library, b"hipMemcpyHtoD\0")? },
            hip_memcpy_dto_h: unsafe { load_symbol(&hip_library, b"hipMemcpyDtoH\0")? },
            hip_module_load_data: unsafe { load_symbol(&hip_library, b"hipModuleLoadData\0")? },
            hip_module_unload: unsafe { load_symbol(&hip_library, b"hipModuleUnload\0")? },
            hip_module_get_function: unsafe {
                load_symbol(&hip_library, b"hipModuleGetFunction\0")?
            },
            hip_module_launch_kernel: unsafe {
                load_symbol(&hip_library, b"hipModuleLaunchKernel\0")?
            },
            hiprtc_create_program: unsafe { load_symbol(&rtc_library, b"hiprtcCreateProgram\0")? },
            hiprtc_compile_program: unsafe {
                load_symbol(&rtc_library, b"hiprtcCompileProgram\0")?
            },
            hiprtc_get_code_size: unsafe { load_symbol(&rtc_library, b"hiprtcGetCodeSize\0")? },
            hiprtc_get_code: unsafe { load_symbol(&rtc_library, b"hiprtcGetCode\0")? },
            hiprtc_destroy_program: unsafe {
                load_symbol(&rtc_library, b"hiprtcDestroyProgram\0")?
            },
            hiprtc_get_program_log_size: unsafe {
                load_symbol(&rtc_library, b"hiprtcGetProgramLogSize\0")?
            },
            hiprtc_get_program_log: unsafe { load_symbol(&rtc_library, b"hiprtcGetProgramLog\0")? },
            _hip_library: hip_library,
            _rtc_library: rtc_library,
        })
    }

    fn check_hip(&self, code: RocmError, context: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(NeuralError::InvalidArgument(format!(
                "{context} (HIP error code {code})"
            )))
        }
    }

    fn check_rtc(&self, code: RocmError, context: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(NeuralError::InvalidArgument(format!(
                "{context} (HIPRTC error code {code})"
            )))
        }
    }

    fn prepare_device(&self) -> Result<()> {
        self.check_hip((self.hip_init)(0), "failed to initialize the ROCm runtime")?;
        let mut count = 0;
        self.check_hip(
            (self.hip_get_device_count)(&mut count),
            "failed to query ROCm device count",
        )?;
        if count <= 0 {
            return Err(NeuralError::InvalidArgument(
                "no ROCm device is available".to_string(),
            ));
        }
        self.check_hip((self.hip_set_device)(0), "failed to select ROCm device 0")
    }

    fn program_log(&self, program: HiprtcProgram) -> String {
        let mut size = 0usize;
        if (self.hiprtc_get_program_log_size)(program, &mut size) != 0 || size == 0 {
            return String::new();
        }
        let mut buffer = vec![0u8; size];
        if (self.hiprtc_get_program_log)(program, buffer.as_mut_ptr().cast()) != 0 {
            return String::new();
        }
        let len = buffer
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..len]).into_owned()
    }

    fn with_compiled_kernel<T>(
        &self,
        source: &str,
        entry: &str,
        f: impl FnOnce(HipFunction) -> Result<T>,
    ) -> Result<T> {
        let source_c = CString::new(source).map_err(|err| {
            NeuralError::InvalidArgument(format!("ROCm kernel source contains NUL bytes: {err}"))
        })?;
        let name_c = CString::new("kernel.hip").expect("static source name");
        let entry_c = CString::new(entry).map_err(|err| {
            NeuralError::InvalidArgument(format!("ROCm kernel entry contains NUL bytes: {err}"))
        })?;
        let mut program: HiprtcProgram = std::ptr::null_mut();
        self.check_rtc(
            (self.hiprtc_create_program)(
                &mut program,
                source_c.as_ptr(),
                name_c.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ),
            "failed to create a ROCm RTC program",
        )?;

        let compile_result = (self.hiprtc_compile_program)(program, 0, std::ptr::null());
        if compile_result != 0 {
            let log = self.program_log(program);
            let _ = (self.hiprtc_destroy_program)(&mut program);
            return Err(NeuralError::InvalidArgument(if log.is_empty() {
                format!(
                    "failed to compile ROCm kernel {entry:?} (HIPRTC error code {compile_result})"
                )
            } else {
                format!(
                    "failed to compile ROCm kernel {entry:?} (HIPRTC error code {compile_result}): {log}"
                )
            }));
        }

        let mut code_size = 0usize;
        self.check_rtc(
            (self.hiprtc_get_code_size)(program, &mut code_size),
            "failed to query ROCm RTC code size",
        )?;
        let mut code = vec![0u8; code_size];
        self.check_rtc(
            (self.hiprtc_get_code)(program, code.as_mut_ptr().cast()),
            "failed to extract ROCm RTC code",
        )?;
        self.check_rtc(
            (self.hiprtc_destroy_program)(&mut program),
            "failed to destroy the ROCm RTC program",
        )?;

        let mut module: HipModule = std::ptr::null_mut();
        self.check_hip(
            (self.hip_module_load_data)(&mut module, code.as_ptr().cast()),
            "failed to load the ROCm module",
        )?;

        let mut function: HipFunction = std::ptr::null_mut();
        let function_result = self.check_hip(
            (self.hip_module_get_function)(&mut function, module, entry_c.as_ptr()),
            "failed to locate the ROCm kernel entry point",
        );
        if let Err(err) = function_result {
            let _ = (self.hip_module_unload)(module);
            return Err(err);
        }

        let result = f(function);
        let unload_result = self.check_hip(
            (self.hip_module_unload)(module),
            "failed to unload the ROCm module",
        );
        match (result, unload_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(err)) => Err(err),
            (Err(err), _) => Err(err),
        }
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
thread_local! {
    static ROCM_RUNTIME: std::cell::RefCell<Option<RocmRuntime>> = const { std::cell::RefCell::new(None) };
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn with_rocm_runtime<T>(f: impl FnOnce(&RocmRuntime) -> Result<T>) -> Result<T> {
    ROCM_RUNTIME.with(|cell| {
        let mut maybe_runtime = cell.borrow_mut();
        if maybe_runtime.is_none() {
            *maybe_runtime = Some(RocmRuntime::new()?);
        }
        let runtime = maybe_runtime
            .as_ref()
            .expect("initialized ROCm runtime context");
        f(runtime)
    })
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_probe() -> bool {
    RocmRuntime::new()
        .and_then(|runtime| {
            runtime.prepare_device()?;
            Ok(())
        })
        .is_ok()
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
struct RocmDeviceBuffer {
    ptr: *mut c_void,
    free: extern "C" fn(*mut c_void) -> RocmError,
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl RocmDeviceBuffer {
    fn new(runtime: &RocmRuntime, bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        runtime.check_hip(
            (runtime.hip_malloc)(&mut ptr, bytes),
            "failed to allocate ROCm device memory",
        )?;
        Ok(Self {
            ptr,
            free: runtime.hip_free,
        })
    }

    fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl Drop for RocmDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = (self.free)(self.ptr);
        }
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn rocm_launch_kernel(
    runtime: &RocmRuntime,
    function: HipFunction,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    args: &mut [*mut c_void],
) -> Result<()> {
    runtime.check_hip(
        (runtime.hip_module_launch_kernel)(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        ),
        "failed to launch the ROCm kernel",
    )
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_copy_to_device<T: Copy>(
    runtime: &RocmRuntime,
    buffer: &RocmDeviceBuffer,
    values: &[T],
) -> Result<()> {
    runtime.check_hip(
        (runtime.hip_memcpy_hto_d)(
            buffer.as_mut_ptr(),
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
        ),
        "failed to upload data to ROCm device memory",
    )
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_copy_from_device<T: Copy>(
    runtime: &RocmRuntime,
    values: &mut [T],
    buffer: &RocmDeviceBuffer,
) -> Result<()> {
    runtime.check_hip(
        (runtime.hip_memcpy_dto_h)(
            values.as_mut_ptr().cast(),
            buffer.as_mut_ptr(),
            std::mem::size_of_val(values),
        ),
        "failed to read data back from ROCm device memory",
    )
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "ROCm runtime kernels are unsupported; use the pure Rust or CUDA Oxide backend".to_string(),
    ))
}
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_affine_scores(
    _features: &[Vec<f64>],
    _means: &[f64],
    _weights: &[f64],
    _intercepts: &[f64],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "ROCm runtime kernels are unsupported; use the pure Rust or CUDA Oxide backend".to_string(),
    ))
}
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "ROCm runtime kernels are unsupported; use the pure Rust or CUDA Oxide backend".to_string(),
    ))
}
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "ROCm runtime kernels are unsupported; use the pure Rust or CUDA Oxide backend".to_string(),
    ))
}
#[cfg(all(feature = "rocm", target_os = "linux"))]
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
#[cfg(not(all(feature = "rocm", target_os = "linux")))]
fn rocm_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "ROCm dispatch is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
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

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
fn rocm_dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "ROCm dense layer scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
fn rocm_pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "ROCm pair scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
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

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn dummy_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }

    fn wake(_: *const ()) {}

    fn wake_by_ref(_: *const ()) {}

    fn drop(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    let waker = dummy_waker();
    let mut cx = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[cfg(feature = "webgpu")]
fn bytes_of<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

#[cfg(feature = "webgpu")]
async fn webgpu_request_device_async() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|err| {
            NeuralError::InvalidArgument(format!("failed to request a WebGPU adapter: {err}"))
        })?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|err| {
            NeuralError::InvalidArgument(format!("failed to request a WebGPU device: {err}"))
        })
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_request_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    block_on(webgpu_request_device_async())
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output_buffer: &wgpu::Buffer,
    byte_len: u64,
) -> Result<Vec<f32>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let result = loop {
        match rx.try_recv() {
            Ok(result) => break result,
            Err(mpsc::TryRecvError::Empty) => {
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                std::thread::yield_now();
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                staging.unmap();
                return Err(NeuralError::InvalidArgument(
                    "WebGPU readback channel disconnected".to_string(),
                ));
            }
        }
    };
    result.map_err(|err| {
        NeuralError::InvalidArgument(format!("failed to map WebGPU readback buffer: {err}"))
    })?;
    let data = slice.get_mapped_range();
    let values =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), byte_len as usize / 4) }
            .to_vec();
    drop(data);
    staging.unmap();
    Ok(values)
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
async fn webgpu_readback_f32_async(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output_buffer: &wgpu::Buffer,
    byte_len: u64,
) -> Result<Vec<f32>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    rx.await
        .map_err(|_| {
            NeuralError::InvalidArgument("WebGPU readback channel disconnected".to_string())
        })?
        .map_err(|err| {
            NeuralError::InvalidArgument(format!("failed to map WebGPU readback buffer: {err}"))
        })?;
    let data = slice.get_mapped_range();
    let values = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), byte_len as usize / 4).to_vec()
    };
    drop(data);
    staging.unmap();
    Ok(values)
}

/// Runs the WebGPU verification kernel without blocking the browser event loop.
/// This is intentionally an async-only API: browser adapter and buffer mapping
/// callbacks are delivered through JavaScript promises.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub async fn webgpu_dispatch_report_async(len: usize) -> Result<BackendDispatchReport> {
    const SOURCE: &str = r#"
        struct Params { len: u32, };
        @group(0) @binding(0) var<storage, read> left: array<f32>;
        @group(0) @binding(1) var<storage, read> right: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<storage, read> params: Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= params.len) { return; }
            output[id.x] = left[id.x] + right[id.x];
        }
    "#;

    let len = len.max(1);
    let left = (0..len).map(|idx| idx as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|idx| idx as f32 * 1.5).collect::<Vec<_>>();
    let params = [len as u32];
    let expected_checksum = (0..len).map(|idx| idx as f64 * 2.0).sum::<f64>();
    let (device, queue) = webgpu_request_device_async().await?;
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let left_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-left"),
        size: std::mem::size_of_val(left.as_slice()) as u64,
        usage: storage,
        mapped_at_creation: false,
    });
    let right_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-right"),
        size: std::mem::size_of_val(right.as_slice()) as u64,
        usage: storage,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-output"),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-params"),
        size: std::mem::size_of_val(&params) as u64,
        usage: storage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&left_buffer, 0, bytes_of(&left));
    queue.write_buffer(&right_buffer, 0, bytes_of(&right));
    queue.write_buffer(&params_buffer, 0, bytes_of(&params));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("webgpu-vector-add"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu-vector-layout"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, true),
            storage_layout_entry(2, false),
            storage_layout_entry(3, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu-vector-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("webgpu-vector-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu-vector-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: left_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: right_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-vector-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("webgpu-vector-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((len as u32).div_ceil(64), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let output =
        webgpu_readback_f32_async(&device, &queue, &output_buffer, (len * 4) as u64).await?;
    Ok(BackendDispatchReport {
        requested: "webgpu".to_string(),
        selected: "webgpu".to_string(),
        operation: "vector_add".to_string(),
        len,
        checksum: output.into_iter().map(f64::from).sum(),
        expected_checksum,
        // `std::time::Instant` is unavailable on wasm32-unknown-unknown.
        // Browser callers can time the returned Promise with `performance.now()`.
        elapsed_ms: 0.0,
        accelerated: true,
    })
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Executes a dense layer on a browser WebGPU device without blocking the
/// JavaScript event loop.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub async fn webgpu_dense_layer_f32_async(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    validate_dense_layer_inputs(features, weights, biases)?;
    const SOURCE: &str = r#"
        struct Params { rows: u32, cols: u32, out_dim: u32, };
        @group(0) @binding(0) var<storage, read> features: array<f32>;
        @group(0) @binding(1) var<storage, read> weights: array<f32>;
        @group(0) @binding(2) var<storage, read> biases: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
        @group(0) @binding(4) var<storage, read> params: Params;
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= params.rows || id.y >= params.out_dim) { return; }
            var value = biases[id.y];
            var col = 0u;
            loop {
                if (col >= params.cols) { break; }
                value += features[id.x * params.cols + col] * weights[col * params.out_dim + id.y];
                col += 1u;
            }
            output[id.x * params.out_dim + id.y] = value;
        }
    "#;
    let rows = features.len();
    let cols = features[0].len();
    let out_dim = biases.len();
    let flat = features
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let params = [rows as u32, cols as u32, out_dim as u32];
    let (device, queue) = webgpu_request_device_async().await?;
    let read_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let features_buffer = buffer(
        "webgpu-dense-features",
        std::mem::size_of_val(flat.as_slice()) as u64,
        read_storage,
    );
    let weights_buffer = buffer(
        "webgpu-dense-weights",
        std::mem::size_of_val(weights) as u64,
        read_storage,
    );
    let biases_buffer = buffer(
        "webgpu-dense-biases",
        std::mem::size_of_val(biases) as u64,
        read_storage,
    );
    let output_buffer = buffer(
        "webgpu-dense-output",
        (rows * out_dim * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let params_buffer = buffer(
        "webgpu-dense-params",
        std::mem::size_of_val(&params) as u64,
        read_storage,
    );
    queue.write_buffer(&features_buffer, 0, bytes_of(&flat));
    queue.write_buffer(&weights_buffer, 0, bytes_of(weights));
    queue.write_buffer(&biases_buffer, 0, bytes_of(biases));
    queue.write_buffer(&params_buffer, 0, bytes_of(&params));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("webgpu-dense"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu-dense-layout"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, true),
            storage_layout_entry(2, true),
            storage_layout_entry(3, false),
            storage_layout_entry(4, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu-dense-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("webgpu-dense-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu-dense-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: biases_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-dense-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("webgpu-dense-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows.div_ceil(8) as u32, out_dim.div_ceil(8) as u32, 1);
    }
    queue.submit(Some(encoder.finish()));
    let values =
        webgpu_readback_f32_async(&device, &queue, &output_buffer, (rows * out_dim * 4) as u64)
            .await?;
    Ok(values.chunks(out_dim).map(|row| row.to_vec()).collect())
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_vector_add_report(
    selection: BackendSelection,
    len: usize,
) -> Result<BackendDispatchReport> {
    const SOURCE: &str = r#"
        struct Params {
            len: u32,
        };

        @group(0) @binding(0) var<storage, read> left: array<f32>;
        @group(0) @binding(1) var<storage, read> right: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<storage, read> params: Params;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= params.len) {
                return;
            }
            output[idx] = left[idx] + right[idx];
        }
    "#;

    let left = (0..len).map(|idx| idx as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|idx| idx as f32 * 1.5).collect::<Vec<_>>();
    let params = [len as u32];
    let start = Instant::now();
    let (device, queue) = webgpu_request_device()?;
    let left_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-left"),
        size: std::mem::size_of_val(left.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let right_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-right"),
        size: std::mem::size_of_val(right.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-output"),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-vector-params"),
        size: std::mem::size_of_val(&params) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&left_buffer, 0, bytes_of(&left));
    queue.write_buffer(&right_buffer, 0, bytes_of(&right));
    queue.write_buffer(&params_buffer, 0, bytes_of(&params));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("webgpu-vector-add"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu-vector-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu-vector-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("webgpu-vector-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu-vector-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: left_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: right_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-vector-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("webgpu-vector-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(len.div_ceil(64) as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let output = webgpu_readback_f32(&device, &queue, &output_buffer, (len * 4) as u64)?;
    let checksum = output.iter().map(|value| f64::from(*value)).sum::<f64>();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(BackendDispatchReport {
        requested: selection.requested,
        selected: selection.selected,
        operation: "vector_add_f32".to_string(),
        len,
        checksum,
        expected_checksum: expected_vector_add_checksum(len),
        elapsed_ms,
        accelerated: true,
    })
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    const SOURCE: &str = r#"
        struct Params {
            rows: u32,
            cols: u32,
        };

        @group(0) @binding(0) var<storage, read> features: array<f32>;
        @group(0) @binding(1) var<storage, read> means: array<f32>;
        @group(0) @binding(2) var<storage, read> weights: array<f32>;
        @group(0) @binding(3) var<storage, read> intercepts: array<f32>;
        @group(0) @binding(4) var<storage, read_write> output: array<f32>;
        @group(0) @binding(5) var<storage, read> params: Params;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let row = id.x;
            if (row >= params.rows) {
                return;
            }
            var score = intercepts[row];
            let offset = row * params.cols;
            var col = 0u;
            loop {
                if (col >= params.cols) {
                    break;
                }
                score += (features[offset + col] - means[col]) * weights[col];
                col = col + 1u;
            }
            output[row] = score;
        }
    "#;

    let rows = features.len();
    let cols = weights.len();
    let flat_features = features
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as f32))
        .collect::<Vec<_>>();
    let means = means.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let weights = weights
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let intercepts = intercepts
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let params = [rows as u32, cols as u32];
    let (device, queue) = webgpu_request_device()?;
    let feature_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-affine-features"),
        size: std::mem::size_of_val(flat_features.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let means_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-affine-means"),
        size: std::mem::size_of_val(means.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-affine-weights"),
        size: std::mem::size_of_val(weights.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let intercepts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-affine-intercepts"),
        size: std::mem::size_of_val(intercepts.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-affine-output"),
        size: (rows * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-affine-params"),
        size: std::mem::size_of_val(&params) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&feature_buffer, 0, bytes_of(&flat_features));
    queue.write_buffer(&means_buffer, 0, bytes_of(&means));
    queue.write_buffer(&weights_buffer, 0, bytes_of(&weights));
    queue.write_buffer(&intercepts_buffer, 0, bytes_of(&intercepts));
    queue.write_buffer(&params_buffer, 0, bytes_of(&params));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("webgpu-affine"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu-affine-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu-affine-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("webgpu-affine-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu-affine-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: feature_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: means_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: intercepts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-affine-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("webgpu-affine-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let output = webgpu_readback_f32(&device, &queue, &output_buffer, (rows * 4) as u64)?;
    Ok(output.into_iter().map(f64::from).collect())
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    const SOURCE: &str = r#"
        struct Params {
            rows: u32,
            cols: u32,
            out_dim: u32,
        };

        @group(0) @binding(0) var<storage, read> features: array<f32>;
        @group(0) @binding(1) var<storage, read> weights: array<f32>;
        @group(0) @binding(2) var<storage, read> biases: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
        @group(0) @binding(4) var<storage, read> params: Params;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let row = id.x;
            let out = id.y;
            if (row >= params.rows || out >= params.out_dim) {
                return;
            }
            var value = biases[out];
            let feature_offset = row * params.cols;
            var col = 0u;
            loop {
                if (col >= params.cols) {
                    break;
                }
                value += features[feature_offset + col] * weights[col * params.out_dim + out];
                col = col + 1u;
            }
            output[row * params.out_dim + out] = value;
        }
    "#;

    let rows = features.len();
    let cols = features[0].len();
    let out_dim = biases.len();
    let flat_features = features
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let params = [rows as u32, cols as u32, out_dim as u32];
    let (device, queue) = webgpu_request_device()?;
    let feature_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-dense-features"),
        size: std::mem::size_of_val(flat_features.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-dense-weights"),
        size: std::mem::size_of_val(weights) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bias_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-dense-biases"),
        size: std::mem::size_of_val(biases) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-dense-output"),
        size: (rows * out_dim * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-dense-params"),
        size: std::mem::size_of_val(&params) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&feature_buffer, 0, bytes_of(&flat_features));
    queue.write_buffer(&weight_buffer, 0, bytes_of(weights));
    queue.write_buffer(&bias_buffer, 0, bytes_of(biases));
    queue.write_buffer(&params_buffer, 0, bytes_of(&params));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("webgpu-dense"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu-dense-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu-dense-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("webgpu-dense-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu-dense-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: feature_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bias_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-dense-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("webgpu-dense-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows.div_ceil(8) as u32, out_dim.div_ceil(8) as u32, 1);
    }
    queue.submit(Some(encoder.finish()));
    let output = webgpu_readback_f32(&device, &queue, &output_buffer, (rows * out_dim * 4) as u64)?;
    Ok(output.chunks(out_dim).map(|row| row.to_vec()).collect())
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
fn webgpu_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    const SOURCE: &str = r#"
        struct Params {
            pairs: u32,
            dim: u32,
        };

        @group(0) @binding(0) var<storage, read> embeddings: array<f32>;
        @group(0) @binding(1) var<storage, read> pair_indices: array<u32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<storage, read> params: Params;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let pair_id = id.x;
            if (pair_id >= params.pairs) {
                return;
            }
            let source = pair_indices[pair_id * 2u];
            let target_index = pair_indices[pair_id * 2u + 1u];
            let source_offset = source * params.dim;
            let target_offset = target_index * params.dim;
            var score = 0.0f;
            var col = 0u;
            loop {
                if (col >= params.dim) {
                    break;
                }
                score += embeddings[source_offset + col] * embeddings[target_offset + col];
                col = col + 1u;
            }
            output[pair_id] = 1.0 / (1.0 + exp(-score));
        }
    "#;

    let dim = embeddings[0].len();
    let flat_embeddings = embeddings
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let pair_indices = pairs
        .iter()
        .flat_map(|&(source, target)| [source as u32, target as u32])
        .collect::<Vec<_>>();
    let params = [pairs.len() as u32, dim as u32];
    let (device, queue) = webgpu_request_device()?;
    let embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-pair-embeddings"),
        size: std::mem::size_of_val(flat_embeddings.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pair_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-pair-indices"),
        size: (pair_indices.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-pair-output"),
        size: pairs.len().saturating_mul(std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webgpu-pair-params"),
        size: std::mem::size_of_val(&params) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&embedding_buffer, 0, bytes_of(&flat_embeddings));
    queue.write_buffer(&pair_buffer, 0, bytes_of(&pair_indices));
    queue.write_buffer(&params_buffer, 0, bytes_of(&params));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("webgpu-pair"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("webgpu-pair-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("webgpu-pair-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("webgpu-pair-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("webgpu-pair-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: embedding_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pair_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("webgpu-pair-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("webgpu-pair-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(pairs.len().div_ceil(64) as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let output = webgpu_readback_f32(&device, &queue, &output_buffer, (pairs.len() * 4) as u64)?;
    Ok(output.into_iter().map(f64::from).collect())
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
fn webgpu_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "WebGPU dispatch is not available in this build".to_string(),
    ))
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
fn webgpu_affine_scores(
    _features: &[Vec<f64>],
    _means: &[f64],
    _weights: &[f64],
    _intercepts: &[f64],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "WebGPU affine scoring is not available in this build".to_string(),
    ))
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
fn webgpu_dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "WebGPU dense layer scoring is not available in this build".to_string(),
    ))
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
fn webgpu_pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "WebGPU pair scoring is not available in this build".to_string(),
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

    #[cfg(all(feature = "rocm", target_os = "linux"))]
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

    #[cfg(all(feature = "rocm", target_os = "linux"))]
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

    #[cfg(all(feature = "rocm", target_os = "linux"))]
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

    #[cfg(all(feature = "rocm", target_os = "linux"))]
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
}
