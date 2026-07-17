//! WebGPU compute backend implementation.

use crate::backend::{BackendDispatchReport, BackendSelection, CsrDiffusionBackward};
use crate::{NeuralError, Result};
#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
static AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn is_available() -> bool {
    *AVAILABLE.get_or_init(|| request_device().is_ok())
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
async fn request_device_async() -> Result<(wgpu::Device, wgpu::Queue)> {
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
fn request_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    block_on(request_device_async())
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
pub async fn dispatch_report_async(len: usize) -> Result<BackendDispatchReport> {
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
    let (device, queue) = request_device_async().await?;
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
pub async fn dense_layer_f32_async(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    crate::backend::validate_dense_layer_inputs(features, weights, biases)?;
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
    let (device, queue) = request_device_async().await?;
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
pub(crate) fn vector_add_report(
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
    let (device, queue) = request_device()?;
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
        expected_checksum: crate::backend::expected_vector_add_checksum(len),
        elapsed_ms,
        accelerated: true,
    })
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
pub(crate) fn affine_scores(
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
    let (device, queue) = request_device()?;
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
pub(crate) fn dense_layer_f32(
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
    let (device, queue) = request_device()?;
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
pub(crate) fn pair_sigmoid_scores_f32(
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
    let (device, queue) = request_device()?;
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

#[cfg(not(target_arch = "wasm32"))]
fn storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
    writable: bool,
) -> wgpu::Buffer {
    let usage = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | if writable {
            wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::empty()
        };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, bytes);
    }
    buffer
}

#[cfg(not(target_arch = "wasm32"))]
fn dispatch_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    source: &str,
    buffers: &[(&wgpu::Buffer, bool)],
    workgroups: u32,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, (_, read_only))| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: *read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect::<Vec<_>>();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, (buffer, _))| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &entries,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
}

#[cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
pub(crate) fn scalar_graph_f32(
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        struct Params { len: u32, };
        @group(0) @binding(0) var<storage, read_write> values: array<f32>;
        @group(0) @binding(1) var<storage, read> opcodes: array<u32>;
        @group(0) @binding(2) var<storage, read> left: array<u32>;
        @group(0) @binding(3) var<storage, read> right: array<u32>;
        @group(0) @binding(4) var<storage, read> params: Params;
        @compute @workgroup_size(1)
        fn main() {
            var i = 0u;
            loop {
                if (i >= params.len) { break; }
                let op = opcodes[i];
                if (op > 1u) {
                    let a = values[left[i]]; let b = values[right[i]];
                    switch op {
                        case 2u: { values[i] = a + b; }
                        case 3u: { values[i] = a * b; }
                        case 4u: { values[i] = a / max(b, 1.0e-12); }
                        case 5u: { values[i] = tanh(a); }
                        case 6u: { values[i] = exp(a); }
                        case 7u: { values[i] = sqrt(max(a, 1.0e-12)); }
                        case 8u: { values[i] = sin(a); }
                        case 9u: { values[i] = 1.0 / (1.0 + exp(-a)); }
                        case 10u: { values[i] = max(a, b); }
                        case 11u: { values[i] = a; }
                        default: {}
                    }
                }
                i += 1u;
            }
        }
    "#;
    let (device, queue) = request_device()?;
    let opcodes = opcodes
        .iter()
        .map(|&value| u32::from(value))
        .collect::<Vec<_>>();
    let params = [initial_values.len() as u32];
    let values = storage_buffer(
        &device,
        &queue,
        "scalar-values",
        bytes_of(initial_values),
        true,
    );
    let ops = storage_buffer(&device, &queue, "scalar-opcodes", bytes_of(&opcodes), false);
    let lhs = storage_buffer(&device, &queue, "scalar-left", bytes_of(left), false);
    let rhs = storage_buffer(&device, &queue, "scalar-right", bytes_of(right), false);
    let config = storage_buffer(&device, &queue, "scalar-params", bytes_of(&params), false);
    dispatch_shader(
        &device,
        &queue,
        "webgpu-scalar-graph",
        SOURCE,
        &[
            (&values, false),
            (&ops, true),
            (&lhs, true),
            (&rhs, true),
            (&config, true),
        ],
        1,
    );
    webgpu_readback_f32(
        &device,
        &queue,
        &values,
        std::mem::size_of_val(initial_values) as u64,
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn csr_diffusion(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        struct Params { nodes: u32, channels: u32, len: u32, };
        @group(0) @binding(0) var<storage, read> indptr: array<u32>;
        @group(0) @binding(1) var<storage, read> indices: array<u32>;
        @group(0) @binding(2) var<storage, read> weights: array<f32>;
        @group(0) @binding(3) var<storage, read> values: array<f32>;
        @group(0) @binding(4) var<storage, read_write> output: array<f32>;
        @group(0) @binding(5) var<storage, read> params: Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i >= params.len) { return; }
            let node = (i / params.channels) % params.nodes;
            let channel = i % params.channels;
            let batch = i / (params.nodes * params.channels);
            var sum = 0.0;
            for (var edge = indptr[node]; edge < indptr[node + 1u]; edge++) {
                let source = indices[edge];
                sum += weights[edge] * values[(batch * params.nodes + source) * params.channels + channel];
            }
            output[i] = sum;
        }
    "#;
    let (device, queue) = request_device()?;
    let output = vec![0.0_f32; values.len()];
    let params = [
        (indptr.len() - 1) as u32,
        channels as u32,
        values.len() as u32,
    ];
    let b0 = storage_buffer(
        &device,
        &queue,
        "webgpu-csr-indptr",
        bytes_of(indptr),
        false,
    );
    let b1 = storage_buffer(
        &device,
        &queue,
        "webgpu-csr-indices",
        bytes_of(indices),
        false,
    );
    let b2 = storage_buffer(
        &device,
        &queue,
        "webgpu-csr-weights",
        bytes_of(weights),
        false,
    );
    let b3 = storage_buffer(
        &device,
        &queue,
        "webgpu-csr-values",
        bytes_of(values),
        false,
    );
    let b4 = storage_buffer(
        &device,
        &queue,
        "webgpu-csr-output",
        bytes_of(&output),
        true,
    );
    let b5 = storage_buffer(
        &device,
        &queue,
        "webgpu-csr-params",
        bytes_of(&params),
        false,
    );
    dispatch_shader(
        &device,
        &queue,
        "webgpu-csr-diffusion",
        SOURCE,
        &[
            (&b0, true),
            (&b1, true),
            (&b2, true),
            (&b3, true),
            (&b4, false),
            (&b5, true),
        ],
        (values.len() as u32).div_ceil(64),
    );
    webgpu_readback_f32(&device, &queue, &b4, (values.len() * 4) as u64)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn csr_diffusion_backward(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    const SOURCE: &str = r#"
        struct Params { nodes: u32, channels: u32, value_len: u32, edge_len: u32, batches: u32, };
        @group(0) @binding(0) var<storage, read> indptr: array<u32>;
        @group(0) @binding(1) var<storage, read> indices: array<u32>;
        @group(0) @binding(2) var<storage, read> weights: array<f32>;
        @group(0) @binding(3) var<storage, read> values: array<f32>;
        @group(0) @binding(4) var<storage, read> grad: array<f32>;
        @group(0) @binding(5) var<storage, read_write> input_grad: array<f32>;
        @group(0) @binding(6) var<storage, read_write> edge_grad: array<f32>;
        @group(0) @binding(7) var<storage, read> params: Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i < params.value_len) {
                let source = (i / params.channels) % params.nodes;
                let channel = i % params.channels;
                let batch = i / (params.nodes * params.channels);
                var sum = 0.0;
                for (var row = 0u; row < params.nodes; row++) {
                    for (var edge = indptr[row]; edge < indptr[row + 1u]; edge++) {
                        if (indices[edge] == source) {
                            sum += weights[edge] * grad[(batch * params.nodes + row) * params.channels + channel];
                        }
                    }
                }
                input_grad[i] = sum;
            }
            if (i < params.edge_len) {
                var row = 0u;
                while (indptr[row + 1u] <= i) { row++; }
                let source = indices[i];
                var sum = 0.0;
                for (var batch = 0u; batch < params.batches; batch++) {
                    for (var channel = 0u; channel < params.channels; channel++) {
                        sum += grad[(batch * params.nodes + row) * params.channels + channel]
                             * values[(batch * params.nodes + source) * params.channels + channel];
                    }
                }
                edge_grad[i] = sum;
            }
        }
    "#;
    let (device, queue) = request_device()?;
    let nodes = indptr.len() - 1;
    let input = vec![0.0_f32; values.len()];
    let edge = vec![0.0_f32; weights.len()];
    let params = [
        nodes as u32,
        channels as u32,
        values.len() as u32,
        weights.len() as u32,
        (values.len() / (nodes * channels)) as u32,
    ];
    let b0 = storage_buffer(&device, &queue, "csr-bwd-indptr", bytes_of(indptr), false);
    let b1 = storage_buffer(&device, &queue, "csr-bwd-indices", bytes_of(indices), false);
    let b2 = storage_buffer(&device, &queue, "csr-bwd-weights", bytes_of(weights), false);
    let b3 = storage_buffer(&device, &queue, "csr-bwd-values", bytes_of(values), false);
    let b4 = storage_buffer(
        &device,
        &queue,
        "csr-bwd-grad",
        bytes_of(output_grad),
        false,
    );
    let b5 = storage_buffer(&device, &queue, "csr-bwd-input", bytes_of(&input), true);
    let b6 = storage_buffer(&device, &queue, "csr-bwd-edge", bytes_of(&edge), true);
    let b7 = storage_buffer(&device, &queue, "csr-bwd-params", bytes_of(&params), false);
    let count = values.len().max(weights.len()) as u32;
    dispatch_shader(
        &device,
        &queue,
        "webgpu-csr-backward",
        SOURCE,
        &[
            (&b0, true),
            (&b1, true),
            (&b2, true),
            (&b3, true),
            (&b4, true),
            (&b5, false),
            (&b6, false),
            (&b7, true),
        ],
        count.div_ceil(64),
    );
    Ok(CsrDiffusionBackward {
        input_grad: webgpu_readback_f32(&device, &queue, &b5, (values.len() * 4) as u64)?,
        edge_grad: webgpu_readback_f32(&device, &queue, &b6, (weights.len() * 4) as u64)?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn csr_row_softmax(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        struct Params { rows: u32, len: u32, };
        @group(0) @binding(0) var<storage, read> indptr: array<u32>;
        @group(0) @binding(1) var<storage, read> logits: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<storage, read> params: Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let edge = id.x; if (edge >= params.len) { return; }
            var row = 0u; while (indptr[row + 1u] <= edge) { row++; }
            var maximum = -3.402823e38;
            for (var i=indptr[row]; i<indptr[row+1u]; i++) { maximum=max(maximum,logits[i]); }
            var denominator=0.0;
            for (var i=indptr[row]; i<indptr[row+1u]; i++) { denominator+=exp(logits[i]-maximum); }
            output[edge]=exp(logits[edge]-maximum)/denominator;
        }
    "#;
    run_row_kernel(indptr, logits, None, SOURCE)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn csr_row_softmax_backward(
    indptr: &[u32],
    weights: &[f32],
    grad: &[f32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        struct Params { rows: u32, len: u32, };
        @group(0) @binding(0) var<storage, read> indptr: array<u32>;
        @group(0) @binding(1) var<storage, read> weights: array<f32>;
        @group(0) @binding(2) var<storage, read> grad: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
        @group(0) @binding(4) var<storage, read> params: Params;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let edge=id.x; if(edge>=params.len){return;}
            var row=0u; while(indptr[row+1u]<=edge){row++;}
            var dot=0.0; for(var i=indptr[row];i<indptr[row+1u];i++){dot+=weights[i]*grad[i];}
            output[edge]=weights[edge]*(grad[edge]-dot);
        }
    "#;
    run_row_kernel(indptr, weights, Some(grad), SOURCE)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_row_kernel(
    indptr: &[u32],
    values: &[f32],
    extra: Option<&[f32]>,
    source: &str,
) -> Result<Vec<f32>> {
    let (device, queue) = request_device()?;
    let output = vec![0.0_f32; values.len()];
    let params = [(indptr.len() - 1) as u32, values.len() as u32];
    let b0 = storage_buffer(&device, &queue, "row-indptr", bytes_of(indptr), false);
    let b1 = storage_buffer(&device, &queue, "row-values", bytes_of(values), false);
    let b2 = extra.map(|v| storage_buffer(&device, &queue, "row-extra", bytes_of(v), false));
    let out = storage_buffer(&device, &queue, "row-output", bytes_of(&output), true);
    let par = storage_buffer(&device, &queue, "row-params", bytes_of(&params), false);
    if let Some(extra) = b2.as_ref() {
        dispatch_shader(
            &device,
            &queue,
            "webgpu-row-kernel",
            source,
            &[
                (&b0, true),
                (&b1, true),
                (extra, true),
                (&out, false),
                (&par, true),
            ],
            (values.len() as u32).div_ceil(64),
        );
    } else {
        dispatch_shader(
            &device,
            &queue,
            "webgpu-row-kernel",
            source,
            &[(&b0, true), (&b1, true), (&out, false), (&par, true)],
            (values.len() as u32).div_ceil(64),
        );
    }
    webgpu_readback_f32(&device, &queue, &out, (values.len() * 4) as u64)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn adamw(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    grad: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    const SOURCE: &str = r#"
        struct Params { len:u32, step:u32, learning_rate:f32, weight_decay:f32, };
        @group(0) @binding(0) var<storage,read_write> p:array<f32>;
        @group(0) @binding(1) var<storage,read_write> m:array<f32>;
        @group(0) @binding(2) var<storage,read_write> v:array<f32>;
        @group(0) @binding(3) var<storage,read> g:array<f32>;
        @group(0) @binding(4) var<storage,read> params:Params;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
            let i=id.x;if(i>=params.len){return;}
            let gradient=g[i]+params.weight_decay*p[i];
            m[i]=0.9*m[i]+0.1*gradient;v[i]=0.999*v[i]+0.001*gradient*gradient;
            let c1=1.0-pow(0.9,f32(params.step));let c2=1.0-pow(0.999,f32(params.step));
            p[i]-=params.learning_rate*(m[i]/c1)/(sqrt(v[i]/c2)+1e-8);
        }
    "#;
    let (device, queue) = request_device()?;
    let raw = [
        parameters.len() as u32,
        step as u32,
        learning_rate.to_bits(),
        weight_decay.to_bits(),
    ];
    let b0 = storage_buffer(&device, &queue, "adamw-p", bytes_of(parameters), true);
    let b1 = storage_buffer(&device, &queue, "adamw-m", bytes_of(first), true);
    let b2 = storage_buffer(&device, &queue, "adamw-v", bytes_of(second), true);
    let b3 = storage_buffer(&device, &queue, "adamw-g", bytes_of(grad), false);
    let b4 = storage_buffer(&device, &queue, "adamw-params", bytes_of(&raw), false);
    dispatch_shader(
        &device,
        &queue,
        "webgpu-adamw",
        SOURCE,
        &[
            (&b0, false),
            (&b1, false),
            (&b2, false),
            (&b3, true),
            (&b4, true),
        ],
        (parameters.len() as u32).div_ceil(64),
    );
    parameters.copy_from_slice(&webgpu_readback_f32(
        &device,
        &queue,
        &b0,
        (parameters.len() * 4) as u64,
    )?);
    first.copy_from_slice(&webgpu_readback_f32(
        &device,
        &queue,
        &b1,
        (first.len() * 4) as u64,
    )?);
    second.copy_from_slice(&webgpu_readback_f32(
        &device,
        &queue,
        &b2,
        (second.len() * 4) as u64,
    )?);
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn layer_norm(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        struct Params{rows:u32,width:u32,len:u32,};
        @group(0)@binding(0)var<storage,read> x:array<f32>;
        @group(0)@binding(1)var<storage,read> gamma:array<f32>;
        @group(0)@binding(2)var<storage,read> beta:array<f32>;
        @group(0)@binding(3)var<storage,read_write> output:array<f32>;
        @group(0)@binding(4)var<storage,read> params:Params;
        @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){
            let i=id.x;if(i>=params.len){return;}let row=i/params.width;let col=i%params.width;
            var mean=0.0;for(var j=0u;j<params.width;j++){mean+=x[row*params.width+j];}mean/=f32(params.width);
            var variance=0.0;for(var j=0u;j<params.width;j++){let d=x[row*params.width+j]-mean;variance+=d*d;}variance/=f32(params.width);
            output[i]=(x[i]-mean)/sqrt(variance+1e-5)*gamma[col]+beta[col];
        }
    "#;
    let (device, queue) = request_device()?;
    let output = vec![0.0_f32; values.len()];
    let params = [rows as u32, width as u32, values.len() as u32];
    let b0 = storage_buffer(&device, &queue, "ln-x", bytes_of(values), false);
    let b1 = storage_buffer(&device, &queue, "ln-g", bytes_of(gamma), false);
    let b2 = storage_buffer(&device, &queue, "ln-b", bytes_of(beta), false);
    let b3 = storage_buffer(&device, &queue, "ln-out", bytes_of(&output), true);
    let b4 = storage_buffer(&device, &queue, "ln-params", bytes_of(&params), false);
    dispatch_shader(
        &device,
        &queue,
        "webgpu-layer-norm",
        SOURCE,
        &[
            (&b0, true),
            (&b1, true),
            (&b2, true),
            (&b3, false),
            (&b4, true),
        ],
        (values.len() as u32).div_ceil(64),
    );
    webgpu_readback_f32(&device, &queue, &b3, (values.len() * 4) as u64)
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
pub(crate) fn vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "WebGPU dispatch is not available in this build".to_string(),
    ))
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
pub(crate) fn affine_scores(
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
pub(crate) fn dense_layer_f32(
    _features: &[Vec<f32>],
    _weights: &[f32],
    _biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    Err(NeuralError::InvalidArgument(
        "WebGPU dense layer scoring is not available in this build".to_string(),
    ))
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
pub(crate) fn pair_sigmoid_scores_f32(
    _embeddings: &[Vec<f32>],
    _pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    Err(NeuralError::InvalidArgument(
        "WebGPU pair scoring is not available in this build".to_string(),
    ))
}

#[cfg(any(not(feature = "webgpu"), target_arch = "wasm32"))]
pub(crate) fn scalar_graph_f32(
    _initial_values: &[f32],
    _opcodes: &[u8],
    _left: &[u32],
    _right: &[u32],
) -> Result<Vec<f32>> {
    Err(NeuralError::InvalidArgument(
        "native WebGPU scalar-graph inference is not available in this build".to_string(),
    ))
}
