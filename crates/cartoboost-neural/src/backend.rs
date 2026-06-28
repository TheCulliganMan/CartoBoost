use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};
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
        ComputeBackend::Auto => available
            .iter()
            .find(|name| name.as_str() != "cpu")
            .cloned()
            .unwrap_or_else(|| "cpu".to_string()),
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
        "metal" => metal_vector_add_report(selection, len),
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
        "metal" => metal_affine_scores(features, means, weights, intercepts),
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
        "metal" => metal_dense_layer_f32(features, weights, biases),
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
        "metal" => metal_pair_sigmoid_scores_f32(embeddings, pairs),
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

#[cfg(test)]
mod tests {
    use super::*;

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
