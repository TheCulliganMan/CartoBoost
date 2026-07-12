use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};
#[cfg(any(
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "rocm", target_os = "linux")
))]
use std::ffi::{c_char, c_void, CString};
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use std::future::Future;
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use std::sync::mpsc;
#[cfg(any(
    all(feature = "webgpu", not(target_arch = "wasm32")),
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "rocm", target_os = "linux")
))]
use std::sync::OnceLock;
#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Instant;

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

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
static CUDA_AVAILABLE: OnceLock<bool> = OnceLock::new();

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

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_available() -> bool {
    *CUDA_AVAILABLE.get_or_init(cuda_probe)
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

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type CudaError = i32;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type CudaDevice = i32;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type CudaContext = *mut c_void;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type CudaModule = *mut c_void;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type CudaFunction = *mut c_void;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type NvrtcProgram = *mut c_void;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
struct CudaRuntime {
    _driver_library: libloading::Library,
    _rtc_library: libloading::Library,
    cu_init: extern "C" fn(u32) -> CudaError,
    cu_device_get_count: extern "C" fn(*mut i32) -> CudaError,
    cu_device_get: extern "C" fn(*mut CudaDevice, i32) -> CudaError,
    cu_ctx_create_v2: extern "C" fn(*mut CudaContext, u32, CudaDevice) -> CudaError,
    cu_ctx_destroy_v2: extern "C" fn(CudaContext) -> CudaError,
    cu_ctx_synchronize: extern "C" fn() -> CudaError,
    cu_mem_alloc_v2: extern "C" fn(*mut u64, usize) -> CudaError,
    cu_mem_free_v2: extern "C" fn(u64) -> CudaError,
    cu_memcpy_hto_d_v2: extern "C" fn(u64, *const c_void, usize) -> CudaError,
    cu_memcpy_dto_h_v2: extern "C" fn(*mut c_void, u64, usize) -> CudaError,
    cu_module_load_data: extern "C" fn(*mut CudaModule, *const c_void) -> CudaError,
    cu_module_unload: extern "C" fn(CudaModule) -> CudaError,
    cu_module_get_function:
        extern "C" fn(*mut CudaFunction, CudaModule, *const c_char) -> CudaError,
    cu_launch_kernel: extern "C" fn(
        CudaFunction,
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
    ) -> CudaError,
    nvrtc_create_program: extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        i32,
        *const *const c_char,
        *const *const c_char,
    ) -> CudaError,
    nvrtc_compile_program: extern "C" fn(NvrtcProgram, i32, *const *const c_char) -> CudaError,
    nvrtc_get_ptx_size: extern "C" fn(NvrtcProgram, *mut usize) -> CudaError,
    nvrtc_get_ptx: extern "C" fn(NvrtcProgram, *mut c_void) -> CudaError,
    nvrtc_destroy_program: extern "C" fn(*mut NvrtcProgram) -> CudaError,
    nvrtc_get_program_log_size: extern "C" fn(NvrtcProgram, *mut usize) -> CudaError,
    nvrtc_get_program_log: extern "C" fn(NvrtcProgram, *mut c_char) -> CudaError,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaRuntime {
    fn new() -> Result<Self> {
        fn load_library(names: &[&str]) -> Result<libloading::Library> {
            for name in names {
                if let Ok(library) = unsafe { libloading::Library::new(name) } {
                    return Ok(library);
                }
            }
            Err(NeuralError::InvalidArgument(format!(
                "failed to load CUDA libraries from any of: {}",
                names.join(", ")
            )))
        }

        unsafe fn load_symbol<T: Copy>(library: &libloading::Library, symbol: &[u8]) -> Result<T> {
            library.get::<T>(symbol).map(|value| *value).map_err(|err| {
                NeuralError::InvalidArgument(format!(
                    "failed to load CUDA symbol {}: {err}",
                    String::from_utf8_lossy(symbol).trim_end_matches('\0')
                ))
            })
        }

        #[cfg(target_os = "windows")]
        let driver_library = load_library(&["nvcuda.dll"])?;
        #[cfg(target_os = "linux")]
        let driver_library = load_library(&["libcuda.so.1", "libcuda.so"])?;

        #[cfg(target_os = "windows")]
        let rtc_library = load_library(&[
            "nvrtc64_130_0.dll",
            "nvrtc64_129_0.dll",
            "nvrtc64_128_0.dll",
            "nvrtc64_127_0.dll",
            "nvrtc64_126_0.dll",
            "nvrtc64_125_0.dll",
            "nvrtc64_124_0.dll",
            "nvrtc64_123_0.dll",
            "nvrtc64_122_0.dll",
            "nvrtc64_121_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_118_0.dll",
            "nvrtc64_117_0.dll",
        ])?;
        #[cfg(target_os = "linux")]
        let rtc_library = load_library(&["libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11"])?;
        Ok(Self {
            cu_init: unsafe { load_symbol(&driver_library, b"cuInit\0")? },
            cu_device_get_count: unsafe { load_symbol(&driver_library, b"cuDeviceGetCount\0")? },
            cu_device_get: unsafe { load_symbol(&driver_library, b"cuDeviceGet\0")? },
            cu_ctx_create_v2: unsafe { load_symbol(&driver_library, b"cuCtxCreate_v2\0")? },
            cu_ctx_destroy_v2: unsafe { load_symbol(&driver_library, b"cuCtxDestroy_v2\0")? },
            cu_ctx_synchronize: unsafe { load_symbol(&driver_library, b"cuCtxSynchronize\0")? },
            cu_mem_alloc_v2: unsafe { load_symbol(&driver_library, b"cuMemAlloc_v2\0")? },
            cu_mem_free_v2: unsafe { load_symbol(&driver_library, b"cuMemFree_v2\0")? },
            cu_memcpy_hto_d_v2: unsafe { load_symbol(&driver_library, b"cuMemcpyHtoD_v2\0")? },
            cu_memcpy_dto_h_v2: unsafe { load_symbol(&driver_library, b"cuMemcpyDtoH_v2\0")? },
            cu_module_load_data: unsafe { load_symbol(&driver_library, b"cuModuleLoadData\0")? },
            cu_module_unload: unsafe { load_symbol(&driver_library, b"cuModuleUnload\0")? },
            cu_module_get_function: unsafe {
                load_symbol(&driver_library, b"cuModuleGetFunction\0")?
            },
            cu_launch_kernel: unsafe { load_symbol(&driver_library, b"cuLaunchKernel\0")? },
            nvrtc_create_program: unsafe { load_symbol(&rtc_library, b"nvrtcCreateProgram\0")? },
            nvrtc_compile_program: unsafe { load_symbol(&rtc_library, b"nvrtcCompileProgram\0")? },
            nvrtc_get_ptx_size: unsafe { load_symbol(&rtc_library, b"nvrtcGetPTXSize\0")? },
            nvrtc_get_ptx: unsafe { load_symbol(&rtc_library, b"nvrtcGetPTX\0")? },
            nvrtc_destroy_program: unsafe { load_symbol(&rtc_library, b"nvrtcDestroyProgram\0")? },
            nvrtc_get_program_log_size: unsafe {
                load_symbol(&rtc_library, b"nvrtcGetProgramLogSize\0")?
            },
            nvrtc_get_program_log: unsafe { load_symbol(&rtc_library, b"nvrtcGetProgramLog\0")? },
            _driver_library: driver_library,
            _rtc_library: rtc_library,
        })
    }

    fn check_cuda(&self, code: CudaError, context: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(NeuralError::InvalidArgument(format!(
                "{context} (CUDA error code {code})"
            )))
        }
    }

    fn check_nvrtc(&self, code: CudaError, context: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(NeuralError::InvalidArgument(format!(
                "{context} (NVRTC error code {code})"
            )))
        }
    }

    fn prepare_device(&self) -> Result<CudaDevice> {
        self.check_cuda((self.cu_init)(0), "failed to initialize the CUDA driver")?;
        let mut count = 0;
        self.check_cuda(
            (self.cu_device_get_count)(&mut count),
            "failed to query CUDA device count",
        )?;
        if count <= 0 {
            return Err(NeuralError::InvalidArgument(
                "no CUDA device is available".to_string(),
            ));
        }
        let mut device = 0;
        self.check_cuda(
            (self.cu_device_get)(&mut device, 0),
            "failed to select CUDA device 0",
        )?;
        Ok(device)
    }

    fn program_log(&self, program: NvrtcProgram) -> String {
        let mut size = 0usize;
        if (self.nvrtc_get_program_log_size)(program, &mut size) != 0 || size == 0 {
            return String::new();
        }
        let mut buffer = vec![0u8; size];
        if (self.nvrtc_get_program_log)(program, buffer.as_mut_ptr().cast()) != 0 {
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
        f: impl FnOnce(CudaFunction) -> Result<T>,
    ) -> Result<T> {
        let source_c = CString::new(source).map_err(|err| {
            NeuralError::InvalidArgument(format!("CUDA kernel source contains NUL bytes: {err}"))
        })?;
        let name_c = CString::new("kernel.cu").expect("static source name");
        let entry_c = CString::new(entry).map_err(|err| {
            NeuralError::InvalidArgument(format!("CUDA kernel entry contains NUL bytes: {err}"))
        })?;
        let compile_options = [
            CString::new("--std=c++14").expect("static compile option"),
            CString::new("--gpu-architecture=compute_52").expect("static compile option"),
        ];
        let option_ptrs = compile_options
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();

        let mut program: NvrtcProgram = std::ptr::null_mut();
        self.check_nvrtc(
            (self.nvrtc_create_program)(
                &mut program,
                source_c.as_ptr(),
                name_c.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ),
            "failed to create a CUDA NVRTC program",
        )?;

        let compile_result =
            (self.nvrtc_compile_program)(program, option_ptrs.len() as i32, option_ptrs.as_ptr());
        if compile_result != 0 {
            let log = self.program_log(program);
            let _ = (self.nvrtc_destroy_program)(&mut program);
            return Err(NeuralError::InvalidArgument(if log.is_empty() {
                format!(
                    "failed to compile CUDA kernel {entry:?} (NVRTC error code {compile_result})"
                )
            } else {
                format!(
                    "failed to compile CUDA kernel {entry:?} (NVRTC error code {compile_result}): {log}"
                )
            }));
        }

        let mut ptx_size = 0usize;
        self.check_nvrtc(
            (self.nvrtc_get_ptx_size)(program, &mut ptx_size),
            "failed to query CUDA PTX size",
        )?;
        let mut ptx = vec![0u8; ptx_size];
        self.check_nvrtc(
            (self.nvrtc_get_ptx)(program, ptx.as_mut_ptr().cast()),
            "failed to extract CUDA PTX",
        )?;
        self.check_nvrtc(
            (self.nvrtc_destroy_program)(&mut program),
            "failed to destroy the CUDA NVRTC program",
        )?;

        let device = self.prepare_device()?;
        let mut context: CudaContext = std::ptr::null_mut();
        self.check_cuda(
            (self.cu_ctx_create_v2)(&mut context, 0, device),
            "failed to create a CUDA context",
        )?;

        let mut module: CudaModule = std::ptr::null_mut();
        let result = (|| -> Result<T> {
            self.check_cuda(
                (self.cu_module_load_data)(&mut module, ptx.as_ptr().cast()),
                "failed to load the CUDA module",
            )?;
            let mut function: CudaFunction = std::ptr::null_mut();
            self.check_cuda(
                (self.cu_module_get_function)(&mut function, module, entry_c.as_ptr()),
                "failed to locate the CUDA kernel entry point",
            )?;
            f(function)
        })();

        let unload_result = if module.is_null() {
            Ok(())
        } else {
            self.check_cuda(
                (self.cu_module_unload)(module),
                "failed to unload the CUDA module",
            )
        };
        let destroy_result = if context.is_null() {
            Ok(())
        } else {
            self.check_cuda(
                (self.cu_ctx_destroy_v2)(context),
                "failed to destroy the CUDA context",
            )
        };

        match (result, unload_result, destroy_result) {
            (Ok(value), Ok(()), Ok(())) => Ok(value),
            (Ok(_), Err(err), _) => Err(err),
            (Ok(_), Ok(()), Err(err)) => Err(err),
            (Err(err), _, _) => Err(err),
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
thread_local! {
    static CUDA_RUNTIME: std::cell::RefCell<Option<CudaRuntime>> = const { std::cell::RefCell::new(None) };
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn with_cuda_runtime<T>(f: impl FnOnce(&CudaRuntime) -> Result<T>) -> Result<T> {
    CUDA_RUNTIME.with(|cell| {
        let mut maybe_runtime = cell.borrow_mut();
        if maybe_runtime.is_none() {
            *maybe_runtime = Some(CudaRuntime::new()?);
        }
        let runtime = maybe_runtime
            .as_ref()
            .expect("initialized CUDA runtime context");
        f(runtime)
    })
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_probe() -> bool {
    CudaRuntime::new()
        .and_then(|runtime| {
            let device = runtime.prepare_device()?;
            let mut context: CudaContext = std::ptr::null_mut();
            runtime.check_cuda(
                (runtime.cu_ctx_create_v2)(&mut context, 0, device),
                "failed to create a CUDA context during availability probe",
            )?;
            runtime.check_cuda(
                (runtime.cu_ctx_destroy_v2)(context),
                "failed to destroy a CUDA context during availability probe",
            )?;
            Ok(())
        })
        .is_ok()
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
struct CudaDeviceBuffer {
    ptr: u64,
    free: extern "C" fn(u64) -> CudaError,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaDeviceBuffer {
    fn new(runtime: &CudaRuntime, bytes: usize) -> Result<Self> {
        let mut ptr = 0u64;
        runtime.check_cuda(
            (runtime.cu_mem_alloc_v2)(&mut ptr, bytes),
            "failed to allocate CUDA device memory",
        )?;
        Ok(Self {
            ptr,
            free: runtime.cu_mem_free_v2,
        })
    }

    fn as_device_ptr(&self) -> u64 {
        self.ptr
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = (self.free)(self.ptr);
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[allow(clippy::too_many_arguments)]
fn cuda_launch_kernel(
    runtime: &CudaRuntime,
    function: CudaFunction,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    args: &mut [*mut c_void],
) -> Result<()> {
    runtime.check_cuda(
        (runtime.cu_launch_kernel)(
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
        "failed to launch the CUDA kernel",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_copy_to_device<T: Copy>(
    runtime: &CudaRuntime,
    buffer: &CudaDeviceBuffer,
    values: &[T],
) -> Result<()> {
    runtime.check_cuda(
        (runtime.cu_memcpy_hto_d_v2)(
            buffer.as_device_ptr(),
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
        ),
        "failed to upload data to CUDA device memory",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_copy_from_device<T: Copy>(
    runtime: &CudaRuntime,
    values: &mut [T],
    buffer: &CudaDeviceBuffer,
) -> Result<()> {
    runtime.check_cuda(
        (runtime.cu_memcpy_dto_h_v2)(
            values.as_mut_ptr().cast(),
            buffer.as_device_ptr(),
            std::mem::size_of_val(values),
        ),
        "failed to read data back from CUDA device memory",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_vector_add_report(
    selection: BackendSelection,
    len: usize,
) -> Result<BackendDispatchReport> {
    const SOURCE: &str = r#"
        extern "C" __global__ void vector_add_f32(
            const float* left,
            const float* right,
            float* output,
            unsigned int len
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= len) {
                return;
            }
            output[idx] = left[idx] + right[idx];
        }
    "#;

    let left = (0..len).map(|idx| idx as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|idx| idx as f32 * 1.5).collect::<Vec<_>>();
    let start = Instant::now();
    with_cuda_runtime(|runtime| {
        runtime.with_compiled_kernel(SOURCE, "vector_add_f32", |function| {
            let left_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(left.as_slice()))?;
            let right_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(right.as_slice()))?;
            let output_buffer = CudaDeviceBuffer::new(runtime, len * std::mem::size_of::<f32>())?;
            cuda_copy_to_device(runtime, &left_buffer, &left)?;
            cuda_copy_to_device(runtime, &right_buffer, &right)?;
            let mut left_ptr = left_buffer.as_device_ptr();
            let mut right_ptr = right_buffer.as_device_ptr();
            let mut output_ptr = output_buffer.as_device_ptr();
            let mut len_param = len as u32;
            let mut args = [
                (&mut left_ptr as *mut u64).cast::<c_void>(),
                (&mut right_ptr as *mut u64).cast::<c_void>(),
                (&mut output_ptr as *mut u64).cast::<c_void>(),
                (&mut len_param as *mut u32).cast::<c_void>(),
            ];
            cuda_launch_kernel(
                runtime,
                function,
                len.div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_cuda(
                (runtime.cu_ctx_synchronize)(),
                "failed to synchronize CUDA context",
            )?;
            let mut output = vec![0f32; len];
            cuda_copy_from_device(runtime, &mut output, &output_buffer)?;
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
        })
    })
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    const SOURCE: &str = r#"
        extern "C" __global__ void affine_scores_f32(
            const float* features,
            const float* means,
            const float* weights,
            const float* intercepts,
            float* output,
            unsigned int rows,
            unsigned int cols
        ) {
            unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows) {
                return;
            }
            float score = intercepts[row];
            unsigned int offset = row * cols;
            for (unsigned int col = 0; col < cols; ++col) {
                score += (features[offset + col] - means[col]) * weights[col];
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
    with_cuda_runtime(|runtime| {
        runtime.with_compiled_kernel(SOURCE, "affine_scores_f32", |function| {
            let feature_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(flat_features.as_slice()))?;
            let means_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(means.as_slice()))?;
            let weights_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(weights.as_slice()))?;
            let intercepts_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(intercepts.as_slice()))?;
            let output_buffer = CudaDeviceBuffer::new(runtime, rows * std::mem::size_of::<f32>())?;
            cuda_copy_to_device(runtime, &feature_buffer, &flat_features)?;
            cuda_copy_to_device(runtime, &means_buffer, &means)?;
            cuda_copy_to_device(runtime, &weights_buffer, &weights)?;
            cuda_copy_to_device(runtime, &intercepts_buffer, &intercepts)?;
            let mut feature_ptr = feature_buffer.as_device_ptr();
            let mut means_ptr = means_buffer.as_device_ptr();
            let mut weights_ptr = weights_buffer.as_device_ptr();
            let mut intercepts_ptr = intercepts_buffer.as_device_ptr();
            let mut output_ptr = output_buffer.as_device_ptr();
            let mut rows_param = rows as u32;
            let mut cols_param = cols as u32;
            let mut args = [
                (&mut feature_ptr as *mut u64).cast::<c_void>(),
                (&mut means_ptr as *mut u64).cast::<c_void>(),
                (&mut weights_ptr as *mut u64).cast::<c_void>(),
                (&mut intercepts_ptr as *mut u64).cast::<c_void>(),
                (&mut output_ptr as *mut u64).cast::<c_void>(),
                (&mut rows_param as *mut u32).cast::<c_void>(),
                (&mut cols_param as *mut u32).cast::<c_void>(),
            ];
            cuda_launch_kernel(
                runtime,
                function,
                rows.div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_cuda(
                (runtime.cu_ctx_synchronize)(),
                "failed to synchronize CUDA context",
            )?;
            let mut output = vec![0f32; rows];
            cuda_copy_from_device(runtime, &mut output, &output_buffer)?;
            Ok(output.into_iter().map(f64::from).collect::<Vec<_>>())
        })
    })
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    const SOURCE: &str = r#"
        extern "C" __global__ void dense_layer_f32(
            const float* features,
            const float* weights,
            const float* biases,
            float* output,
            unsigned int rows,
            unsigned int cols,
            unsigned int out_dim
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int total = rows * out_dim;
            if (idx >= total) {
                return;
            }
            unsigned int row = idx / out_dim;
            unsigned int out = idx % out_dim;
            float value = biases[out];
            unsigned int feature_offset = row * cols;
            for (unsigned int col = 0; col < cols; ++col) {
                value += features[feature_offset + col] * weights[col * out_dim + out];
            }
            output[idx] = value;
        }
    "#;

    let rows = features.len();
    let cols = features[0].len();
    let out_dim = biases.len();
    let flat_features = features
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    with_cuda_runtime(|runtime| {
        runtime.with_compiled_kernel(SOURCE, "dense_layer_f32", |function| {
            let feature_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(flat_features.as_slice()))?;
            let weight_buffer = CudaDeviceBuffer::new(runtime, std::mem::size_of_val(weights))?;
            let bias_buffer = CudaDeviceBuffer::new(runtime, std::mem::size_of_val(biases))?;
            let output_buffer =
                CudaDeviceBuffer::new(runtime, rows * out_dim * std::mem::size_of::<f32>())?;
            cuda_copy_to_device(runtime, &feature_buffer, &flat_features)?;
            cuda_copy_to_device(runtime, &weight_buffer, weights)?;
            cuda_copy_to_device(runtime, &bias_buffer, biases)?;
            let mut feature_ptr = feature_buffer.as_device_ptr();
            let mut weight_ptr = weight_buffer.as_device_ptr();
            let mut bias_ptr = bias_buffer.as_device_ptr();
            let mut output_ptr = output_buffer.as_device_ptr();
            let mut rows_param = rows as u32;
            let mut cols_param = cols as u32;
            let mut out_dim_param = out_dim as u32;
            let mut args = [
                (&mut feature_ptr as *mut u64).cast::<c_void>(),
                (&mut weight_ptr as *mut u64).cast::<c_void>(),
                (&mut bias_ptr as *mut u64).cast::<c_void>(),
                (&mut output_ptr as *mut u64).cast::<c_void>(),
                (&mut rows_param as *mut u32).cast::<c_void>(),
                (&mut cols_param as *mut u32).cast::<c_void>(),
                (&mut out_dim_param as *mut u32).cast::<c_void>(),
            ];
            cuda_launch_kernel(
                runtime,
                function,
                (rows * out_dim).div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_cuda(
                (runtime.cu_ctx_synchronize)(),
                "failed to synchronize CUDA context",
            )?;
            let mut output = vec![0f32; rows * out_dim];
            cuda_copy_from_device(runtime, &mut output, &output_buffer)?;
            Ok(output
                .chunks(out_dim)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>())
        })
    })
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn cuda_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    const SOURCE: &str = r#"
        extern "C" __global__ void pair_sigmoid_scores_f32(
            const float* embeddings,
            const unsigned int* pairs,
            float* output,
            unsigned int pairs_len,
            unsigned int dim
        ) {
            unsigned int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
            if (pair_id >= pairs_len) {
                return;
            }
            unsigned int source = pairs[pair_id * 2u];
            unsigned int target = pairs[pair_id * 2u + 1u];
            unsigned int source_offset = source * dim;
            unsigned int target_offset = target * dim;
            float score = 0.0f;
            for (unsigned int col = 0; col < dim; ++col) {
                score += embeddings[source_offset + col] * embeddings[target_offset + col];
            }
            output[pair_id] = 1.0f / (1.0f + expf(-score));
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
    with_cuda_runtime(|runtime| {
        runtime.with_compiled_kernel(SOURCE, "pair_sigmoid_scores_f32", |function| {
            let embedding_buffer =
                CudaDeviceBuffer::new(runtime, std::mem::size_of_val(flat_embeddings.as_slice()))?;
            let pair_buffer =
                CudaDeviceBuffer::new(runtime, pair_indices.len() * std::mem::size_of::<u32>())?;
            let output_buffer = CudaDeviceBuffer::new(
                runtime,
                pairs.len().saturating_mul(std::mem::size_of::<f32>()),
            )?;
            cuda_copy_to_device(runtime, &embedding_buffer, &flat_embeddings)?;
            cuda_copy_to_device(runtime, &pair_buffer, &pair_indices)?;
            let mut embedding_ptr = embedding_buffer.as_device_ptr();
            let mut pair_ptr = pair_buffer.as_device_ptr();
            let mut output_ptr = output_buffer.as_device_ptr();
            let mut pairs_param = pairs.len() as u32;
            let mut dim_param = dim as u32;
            let mut args = [
                (&mut embedding_ptr as *mut u64).cast::<c_void>(),
                (&mut pair_ptr as *mut u64).cast::<c_void>(),
                (&mut output_ptr as *mut u64).cast::<c_void>(),
                (&mut pairs_param as *mut u32).cast::<c_void>(),
                (&mut dim_param as *mut u32).cast::<c_void>(),
            ];
            cuda_launch_kernel(
                runtime,
                function,
                pairs.len().div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_cuda(
                (runtime.cu_ctx_synchronize)(),
                "failed to synchronize CUDA context",
            )?;
            let mut output = vec![0f32; pairs.len()];
            cuda_copy_from_device(runtime, &mut output, &output_buffer)?;
            Ok(output.into_iter().map(f64::from).collect::<Vec<_>>())
        })
    })
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn cuda_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "CUDA dispatch is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn cuda_affine_scores(
    _features: &[Vec<f64>],
    _means: &[f64],
    _weights: &[f64],
    _intercepts: &[f64],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "CUDA affine scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn cuda_dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "CUDA dense layer scoring is not available in this build".to_string(),
    ))
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn cuda_pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "CUDA pair scoring is not available in this build".to_string(),
    ))
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
    selection: BackendSelection,
    len: usize,
) -> Result<BackendDispatchReport> {
    const SOURCE: &str = r#"
        extern "C" __global__ void vector_add_f32(
            const float* left,
            const float* right,
            float* output,
            unsigned int len
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= len) {
                return;
            }
            output[idx] = left[idx] + right[idx];
        }
    "#;

    let left = (0..len).map(|idx| idx as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|idx| idx as f32 * 1.5).collect::<Vec<_>>();
    let start = Instant::now();
    with_rocm_runtime(|runtime| {
        runtime.prepare_device()?;
        runtime.with_compiled_kernel(SOURCE, "vector_add_f32", |function| {
            let left_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(left.as_slice()))?;
            let right_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(right.as_slice()))?;
            let output_buffer = RocmDeviceBuffer::new(runtime, len * std::mem::size_of::<f32>())?;
            rocm_copy_to_device(runtime, &left_buffer, &left)?;
            rocm_copy_to_device(runtime, &right_buffer, &right)?;
            let len_param = len as u32;
            let mut left_ptr = left_buffer.as_mut_ptr();
            let mut right_ptr = right_buffer.as_mut_ptr();
            let mut output_ptr = output_buffer.as_mut_ptr();
            let mut len_arg = len_param;
            let mut args = [
                (&mut left_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut right_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut len_arg as *mut u32).cast::<c_void>(),
            ];
            rocm_launch_kernel(
                runtime,
                function,
                len.div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_hip(
                (runtime.hip_device_synchronize)(),
                "failed to synchronize ROCm device",
            )?;
            let mut output = vec![0f32; len];
            rocm_copy_from_device(runtime, &mut output, &output_buffer)?;
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
        })
    })
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    const SOURCE: &str = r#"
        extern "C" __global__ void affine_scores_f32(
            const float* features,
            const float* means,
            const float* weights,
            const float* intercepts,
            float* output,
            unsigned int rows,
            unsigned int cols
        ) {
            unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows) {
                return;
            }
            float score = intercepts[row];
            unsigned int offset = row * cols;
            for (unsigned int col = 0; col < cols; ++col) {
                score += (features[offset + col] - means[col]) * weights[col];
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
    with_rocm_runtime(|runtime| {
        runtime.prepare_device()?;
        runtime.with_compiled_kernel(SOURCE, "affine_scores_f32", |function| {
            let feature_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(flat_features.as_slice()))?;
            let means_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(means.as_slice()))?;
            let weights_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(weights.as_slice()))?;
            let intercepts_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(intercepts.as_slice()))?;
            let output_buffer = RocmDeviceBuffer::new(runtime, rows * std::mem::size_of::<f32>())?;
            rocm_copy_to_device(runtime, &feature_buffer, &flat_features)?;
            rocm_copy_to_device(runtime, &means_buffer, &means)?;
            rocm_copy_to_device(runtime, &weights_buffer, &weights)?;
            rocm_copy_to_device(runtime, &intercepts_buffer, &intercepts)?;
            let rows_arg = rows as u32;
            let cols_arg = cols as u32;
            let mut feature_ptr = feature_buffer.as_mut_ptr();
            let mut means_ptr = means_buffer.as_mut_ptr();
            let mut weights_ptr = weights_buffer.as_mut_ptr();
            let mut intercepts_ptr = intercepts_buffer.as_mut_ptr();
            let mut output_ptr = output_buffer.as_mut_ptr();
            let mut rows_param = rows_arg;
            let mut cols_param = cols_arg;
            let mut args = [
                (&mut feature_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut means_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut weights_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut intercepts_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut rows_param as *mut u32).cast::<c_void>(),
                (&mut cols_param as *mut u32).cast::<c_void>(),
            ];
            rocm_launch_kernel(
                runtime,
                function,
                rows.div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_hip(
                (runtime.hip_device_synchronize)(),
                "failed to synchronize ROCm device",
            )?;
            let mut output = vec![0f32; rows];
            rocm_copy_from_device(runtime, &mut output, &output_buffer)?;
            Ok(output.into_iter().map(f64::from).collect::<Vec<_>>())
        })
    })
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_dense_layer_f32(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    const SOURCE: &str = r#"
        extern "C" __global__ void dense_layer_f32(
            const float* features,
            const float* weights,
            const float* biases,
            float* output,
            unsigned int rows,
            unsigned int cols,
            unsigned int out_dim
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int total = rows * out_dim;
            if (idx >= total) {
                return;
            }
            unsigned int row = idx / out_dim;
            unsigned int out = idx % out_dim;
            float value = biases[out];
            unsigned int feature_offset = row * cols;
            for (unsigned int col = 0; col < cols; ++col) {
                value += features[feature_offset + col] * weights[col * out_dim + out];
            }
            output[idx] = value;
        }
    "#;

    let rows = features.len();
    let cols = features[0].len();
    let out_dim = biases.len();
    let flat_features = features
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    with_rocm_runtime(|runtime| {
        runtime.prepare_device()?;
        runtime.with_compiled_kernel(SOURCE, "dense_layer_f32", |function| {
            let feature_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(flat_features.as_slice()))?;
            let weight_buffer = RocmDeviceBuffer::new(runtime, std::mem::size_of_val(weights))?;
            let bias_buffer = RocmDeviceBuffer::new(runtime, std::mem::size_of_val(biases))?;
            let output_buffer =
                RocmDeviceBuffer::new(runtime, rows * out_dim * std::mem::size_of::<f32>())?;
            rocm_copy_to_device(runtime, &feature_buffer, &flat_features)?;
            rocm_copy_to_device(runtime, &weight_buffer, weights)?;
            rocm_copy_to_device(runtime, &bias_buffer, biases)?;
            let rows_arg = rows as u32;
            let cols_arg = cols as u32;
            let out_dim_arg = out_dim as u32;
            let mut feature_ptr = feature_buffer.as_mut_ptr();
            let mut weight_ptr = weight_buffer.as_mut_ptr();
            let mut bias_ptr = bias_buffer.as_mut_ptr();
            let mut output_ptr = output_buffer.as_mut_ptr();
            let mut rows_param = rows_arg;
            let mut cols_param = cols_arg;
            let mut out_dim_param = out_dim_arg;
            let mut args = [
                (&mut feature_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut weight_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut bias_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut rows_param as *mut u32).cast::<c_void>(),
                (&mut cols_param as *mut u32).cast::<c_void>(),
                (&mut out_dim_param as *mut u32).cast::<c_void>(),
            ];
            rocm_launch_kernel(
                runtime,
                function,
                (rows * out_dim).div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_hip(
                (runtime.hip_device_synchronize)(),
                "failed to synchronize ROCm device",
            )?;
            let mut output = vec![0f32; rows * out_dim];
            rocm_copy_from_device(runtime, &mut output, &output_buffer)?;
            Ok(output
                .chunks(out_dim)
                .map(|row| row.to_vec())
                .collect::<Vec<_>>())
        })
    })
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn rocm_pair_sigmoid_scores_f32(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    const SOURCE: &str = r#"
        extern "C" __global__ void pair_sigmoid_scores_f32(
            const float* embeddings,
            const unsigned int* pairs,
            float* output,
            unsigned int pairs_len,
            unsigned int dim
        ) {
            unsigned int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
            if (pair_id >= pairs_len) {
                return;
            }
            unsigned int source = pairs[pair_id * 2u];
            unsigned int target = pairs[pair_id * 2u + 1u];
            unsigned int source_offset = source * dim;
            unsigned int target_offset = target * dim;
            float score = 0.0f;
            for (unsigned int col = 0; col < dim; ++col) {
                score += embeddings[source_offset + col] * embeddings[target_offset + col];
            }
            output[pair_id] = 1.0f / (1.0f + expf(-score));
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
    with_rocm_runtime(|runtime| {
        runtime.prepare_device()?;
        runtime.with_compiled_kernel(SOURCE, "pair_sigmoid_scores_f32", |function| {
            let embedding_buffer =
                RocmDeviceBuffer::new(runtime, std::mem::size_of_val(flat_embeddings.as_slice()))?;
            let pair_buffer =
                RocmDeviceBuffer::new(runtime, pair_indices.len() * std::mem::size_of::<u32>())?;
            let output_buffer = RocmDeviceBuffer::new(
                runtime,
                pairs.len().saturating_mul(std::mem::size_of::<f32>()),
            )?;
            rocm_copy_to_device(runtime, &embedding_buffer, &flat_embeddings)?;
            rocm_copy_to_device(runtime, &pair_buffer, &pair_indices)?;
            let pairs_arg = pairs.len() as u32;
            let dim_arg = dim as u32;
            let mut embedding_ptr = embedding_buffer.as_mut_ptr();
            let mut pair_ptr = pair_buffer.as_mut_ptr();
            let mut output_ptr = output_buffer.as_mut_ptr();
            let mut pairs_param = pairs_arg;
            let mut dim_param = dim_arg;
            let mut args = [
                (&mut embedding_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut pair_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
                (&mut pairs_param as *mut u32).cast::<c_void>(),
                (&mut dim_param as *mut u32).cast::<c_void>(),
            ];
            rocm_launch_kernel(
                runtime,
                function,
                pairs.len().div_ceil(256) as u32,
                1,
                1,
                256,
                1,
                1,
                &mut args,
            )?;
            runtime.check_hip(
                (runtime.hip_device_synchronize)(),
                "failed to synchronize ROCm device",
            )?;
            let mut output = vec![0f32; pairs.len()];
            rocm_copy_from_device(runtime, &mut output, &output_buffer)?;
            Ok(output.into_iter().map(f64::from).collect::<Vec<_>>())
        })
    })
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
}
