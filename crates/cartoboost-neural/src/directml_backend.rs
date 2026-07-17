//! Windows DirectML tensor backend.
//!
//! All DirectML implementation details live here so the shared dispatcher
//! remains independent from CUDA, ROCm, WebGPU, and CPU implementation work.

use crate::backend::{BackendDispatchReport, BackendSelection, CsrDiffusionBackward};
use crate::{NeuralError, Result};
use std::{ffi::c_void, mem::ManuallyDrop, ptr, sync::OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory4};
use windows::Win32::AI::MachineLearning::DirectML::*;

struct DirectMlRuntime {
    #[allow(dead_code)]
    d3d12: ID3D12Device,
    #[allow(dead_code)]
    directml: IDMLDevice,
}

// ID3D12Device and IDMLDevice are documented as thread-safe COM interfaces.
unsafe impl Send for DirectMlRuntime {}
unsafe impl Sync for DirectMlRuntime {}

// Direct3D 12 and DirectML devices are thread-safe. Keeping one process-wide
// device avoids repeatedly probing adapters or creating duplicate DML devices.
static RUNTIME: OnceLock<Option<DirectMlRuntime>> = OnceLock::new();

fn create_runtime() -> windows::core::Result<DirectMlRuntime> {
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1()?;
        for index in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(index) else {
                break;
            };
            let mut d3d12 = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut d3d12).is_err() {
                continue;
            }
            let d3d12 = d3d12.expect("D3D12CreateDevice succeeded without a device");
            let mut directml = None;
            if DMLCreateDevice(&d3d12, DML_CREATE_DEVICE_FLAG_NONE, &mut directml).is_err() {
                continue;
            }
            return Ok(DirectMlRuntime {
                d3d12,
                directml: directml.expect("DMLCreateDevice succeeded without a device"),
            });
        }
    }
    Err(windows::core::Error::empty())
}

pub(crate) fn is_available() -> bool {
    RUNTIME.get_or_init(|| create_runtime().ok()).is_some()
}

fn as_neural(context: &str, error: windows::core::Error) -> NeuralError {
    NeuralError::InvalidArgument(format!("DirectML {context} failed: {error}"))
}

fn buffer_description(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(4),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

unsafe fn create_buffer(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let description = buffer_description(size, flags);
    let mut output = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            state,
            None,
            &mut output,
        )?;
    }
    Ok(output.expect("CreateCommittedResource succeeded without a resource"))
}

unsafe fn write_upload(buffer: &ID3D12Resource, values: &[f32]) -> windows::core::Result<()> {
    let mut mapped = ptr::null_mut::<c_void>();
    unsafe { buffer.Map(0, None, Some(&mut mapped))? };
    unsafe {
        ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            mapped.cast::<u8>(),
            std::mem::size_of_val(values),
        );
        buffer.Unmap(0, None);
    }
    Ok(())
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

unsafe fn submit_and_wait(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    list: &ID3D12GraphicsCommandList,
) -> windows::core::Result<()> {
    unsafe { list.Close()? };
    let base: ID3D12CommandList = list.cast()?;
    unsafe { queue.ExecuteCommandLists(&[Some(base)]) };
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
    unsafe { queue.Signal(&fence, 1)? };
    while unsafe { fence.GetCompletedValue() } < 1 {
        std::thread::yield_now();
    }
    Ok(())
}

fn buffer_binding(buffer: &ID3D12Resource, size: u64) -> DML_BUFFER_BINDING {
    DML_BUFFER_BINDING {
        Buffer: ManuallyDrop::new(Some(buffer.clone())),
        Offset: 0,
        SizeInBytes: size,
    }
}

fn binding_description(binding: &DML_BUFFER_BINDING) -> DML_BINDING_DESC {
    DML_BINDING_DESC {
        Type: DML_BINDING_TYPE_BUFFER,
        Desc: (binding as *const DML_BUFFER_BINDING).cast(),
    }
}

/// Compiles and executes one DirectML operator. Inputs and output stay in
/// D3D12 default-heap buffers for the dispatch; upload/readback are explicit.
unsafe fn execute_operator(
    operator_description: &DML_OPERATOR_DESC,
    inputs: &[&[f32]],
    output_len: usize,
) -> windows::core::Result<Vec<f32>> {
    let runtime = RUNTIME
        .get_or_init(|| create_runtime().ok())
        .as_ref()
        .ok_or_else(windows::core::Error::empty)?;
    let mut operator: Option<IDMLOperator> = None;
    unsafe {
        runtime
            .directml
            .CreateOperator(operator_description, &mut operator)?
    };
    let operator = operator.expect("CreateOperator succeeded without an operator");
    let mut compiled: Option<IDMLCompiledOperator> = None;
    unsafe {
        runtime
            .directml
            .CompileOperator(&operator, DML_EXECUTION_FLAG_NONE, &mut compiled)?
    };
    let compiled = compiled.expect("CompileOperator succeeded without a compiled operator");
    let initializer: IDMLOperatorInitializer = unsafe {
        runtime
            .directml
            .CreateOperatorInitializer(Some(&[Some(compiled.clone())]))?
    };
    let initialize = unsafe { initializer.GetBindingProperties() };
    let execute = unsafe { compiled.GetBindingProperties() };
    let descriptor_count = initialize
        .RequiredDescriptorCount
        .max(execute.RequiredDescriptorCount)
        .max(1);

    let queue_description = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
    };
    let queue: ID3D12CommandQueue =
        unsafe { runtime.d3d12.CreateCommandQueue(&queue_description)? };
    let allocator: ID3D12CommandAllocator = unsafe {
        runtime
            .d3d12
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?
    };
    let list: ID3D12GraphicsCommandList = unsafe {
        runtime.d3d12.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &allocator,
            None::<&ID3D12PipelineState>,
        )?
    };
    let descriptor_heap: ID3D12DescriptorHeap = unsafe {
        runtime
            .d3d12
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: descriptor_count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
    };
    unsafe { list.SetDescriptorHeaps(&[Some(descriptor_heap.clone())]) };
    let mut table_description = DML_BINDING_TABLE_DESC {
        Dispatchable: ManuallyDrop::new(Some(initializer.cast()?)),
        CPUDescriptorHandle: unsafe { descriptor_heap.GetCPUDescriptorHandleForHeapStart() },
        GPUDescriptorHandle: unsafe { descriptor_heap.GetGPUDescriptorHandleForHeapStart() },
        SizeInDescriptors: descriptor_count,
    };
    let table: IDMLBindingTable = unsafe {
        runtime
            .directml
            .CreateBindingTable(Some(&table_description))?
    };
    let temporary_size = initialize
        .TemporaryResourceSize
        .max(execute.TemporaryResourceSize);
    let temporary = if temporary_size == 0 {
        None
    } else {
        Some(unsafe {
            create_buffer(
                &runtime.d3d12,
                D3D12_HEAP_TYPE_DEFAULT,
                temporary_size,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )?
        })
    };
    let persistent = if execute.PersistentResourceSize == 0 {
        None
    } else {
        Some(unsafe {
            create_buffer(
                &runtime.d3d12,
                D3D12_HEAP_TYPE_DEFAULT,
                execute.PersistentResourceSize,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )?
        })
    };
    if let Some(buffer) = &temporary {
        let binding = buffer_binding(buffer, temporary_size);
        let description = binding_description(&binding);
        unsafe { table.BindTemporaryResource(Some(&description)) };
    }
    if let Some(buffer) = &persistent {
        let binding = buffer_binding(buffer, execute.PersistentResourceSize);
        let description = binding_description(&binding);
        unsafe { table.BindOutputs(Some(&[description])) };
    }
    let recorder: IDMLCommandRecorder = unsafe { runtime.directml.CreateCommandRecorder()? };
    unsafe {
        recorder.RecordDispatch(&list, &initializer, &table);
        submit_and_wait(&runtime.d3d12, &queue, &list)?;
        allocator.Reset()?;
        list.Reset(&allocator, None::<&ID3D12PipelineState>)?;
        list.SetDescriptorHeaps(&[Some(descriptor_heap.clone())]);
    }

    table_description.Dispatchable = ManuallyDrop::new(Some(compiled.cast()?));
    unsafe { table.Reset(Some(&table_description))? };
    if let Some(buffer) = &temporary {
        let binding = buffer_binding(buffer, temporary_size);
        let description = binding_description(&binding);
        unsafe { table.BindTemporaryResource(Some(&description)) };
    }
    if let Some(buffer) = &persistent {
        let binding = buffer_binding(buffer, execute.PersistentResourceSize);
        let description = binding_description(&binding);
        unsafe { table.BindPersistentResource(Some(&description)) };
    }

    let mut device_inputs = Vec::with_capacity(inputs.len());
    for values in inputs {
        let size = std::mem::size_of_val(*values) as u64;
        let upload = unsafe {
            create_buffer(
                &runtime.d3d12,
                D3D12_HEAP_TYPE_UPLOAD,
                size,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
        };
        unsafe { write_upload(&upload, values)? };
        let device = unsafe {
            create_buffer(
                &runtime.d3d12,
                D3D12_HEAP_TYPE_DEFAULT,
                size,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?
        };
        unsafe {
            list.CopyBufferRegion(&device, 0, &upload, 0, size);
            list.ResourceBarrier(&[transition(
                &device,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        // Keep the upload resource alive until the command list containing
        // CopyBufferRegion has completed. Dropping it here leaves DirectML
        // reading zeroed/default-heap inputs on real hardware.
        device_inputs.push((device, size, upload));
    }
    let output_size = (output_len * std::mem::size_of::<f32>()) as u64;
    let output = unsafe {
        create_buffer(
            &runtime.d3d12,
            D3D12_HEAP_TYPE_DEFAULT,
            output_size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?
    };
    let input_bindings = device_inputs
        .iter()
        .map(|(buffer, size, _upload)| buffer_binding(buffer, *size))
        .collect::<Vec<_>>();
    let input_descriptions = input_bindings
        .iter()
        .map(binding_description)
        .collect::<Vec<_>>();
    let output_binding = buffer_binding(&output, output_size);
    let output_description = binding_description(&output_binding);
    unsafe {
        table.BindInputs(Some(&input_descriptions));
        table.BindOutputs(Some(&[output_description]));
        recorder.RecordDispatch(&list, &compiled, &table);
        submit_and_wait(&runtime.d3d12, &queue, &list)?;
        allocator.Reset()?;
        list.Reset(&allocator, None::<&ID3D12PipelineState>)?;
    }
    let readback = unsafe {
        create_buffer(
            &runtime.d3d12,
            D3D12_HEAP_TYPE_READBACK,
            output_size,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?
    };
    unsafe {
        list.ResourceBarrier(&[transition(
            &output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&readback, 0, &output, 0, output_size);
        submit_and_wait(&runtime.d3d12, &queue, &list)?;
    }
    let mut mapped = ptr::null_mut::<c_void>();
    let range = D3D12_RANGE {
        Begin: 0,
        End: output_size as usize,
    };
    unsafe { readback.Map(0, Some(&range), Some(&mut mapped))? };
    let values = unsafe { std::slice::from_raw_parts(mapped.cast::<f32>(), output_len).to_vec() };
    unsafe { readback.Unmap(0, None) };
    Ok(values)
}

fn pending<T>(operation: &str) -> Result<T> {
    Err(NeuralError::InvalidArgument(format!(
        "DirectML {operation} is not initialized"
    )))
}

struct TensorDescription {
    sizes: Vec<u32>,
    strides: Option<Vec<u32>>,
    buffer: Box<DML_BUFFER_TENSOR_DESC>,
    tensor: Box<DML_TENSOR_DESC>,
}

impl TensorDescription {
    fn f32(sizes: &[u32]) -> Self {
        Self::f32_with_strides(sizes, None)
    }

    fn f32_with_strides(sizes: &[u32], strides: Option<&[u32]>) -> Self {
        let sizes = sizes.to_vec();
        let strides = strides.map(<[u32]>::to_vec);
        let elements = if let Some(strides) = &strides {
            sizes
                .iter()
                .zip(strides)
                .fold(1_u64, |total, (size, stride)| {
                    total + u64::from(size.saturating_sub(1)) * u64::from(*stride)
                })
        } else {
            sizes
                .iter()
                .fold(1_u64, |total, value| total * u64::from(*value))
        };
        let buffer = Box::new(DML_BUFFER_TENSOR_DESC {
            DataType: DML_TENSOR_DATA_TYPE_FLOAT32,
            Flags: DML_TENSOR_FLAG_NONE,
            DimensionCount: sizes.len() as u32,
            Sizes: sizes.as_ptr(),
            Strides: strides.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            TotalTensorSizeInBytes: elements * 4,
            GuaranteedBaseOffsetAlignment: 0,
        });
        let tensor = Box::new(DML_TENSOR_DESC {
            Type: DML_TENSOR_TYPE_BUFFER,
            Desc: (&*buffer as *const DML_BUFFER_TENSOR_DESC).cast(),
        });
        Self {
            sizes,
            strides,
            buffer,
            tensor,
        }
    }
}

fn directml_gemm(
    left: &[f32],
    rows: usize,
    cols: usize,
    right: &[f32],
    output_columns: usize,
    bias: &[f32],
) -> Result<Vec<f32>> {
    let left_tensor = TensorDescription::f32(&[1, 1, rows as u32, cols as u32]);
    let right_tensor = TensorDescription::f32(&[1, 1, cols as u32, output_columns as u32]);
    let bias_tensor = TensorDescription::f32_with_strides(
        &[1, 1, rows as u32, output_columns as u32],
        Some(&[0, 0, 0, 1]),
    );
    let output_tensor = TensorDescription::f32(&[1, 1, rows as u32, output_columns as u32]);
    let gemm = DML_GEMM_OPERATOR_DESC {
        ATensor: &*left_tensor.tensor,
        BTensor: &*right_tensor.tensor,
        CTensor: &*bias_tensor.tensor,
        OutputTensor: &*output_tensor.tensor,
        TransA: DML_MATRIX_TRANSFORM_NONE,
        TransB: DML_MATRIX_TRANSFORM_NONE,
        Alpha: 1.0,
        Beta: 1.0,
        FusedActivation: ptr::null(),
    };
    let operator = DML_OPERATOR_DESC {
        Type: DML_OPERATOR_GEMM,
        Desc: (&gemm as *const DML_GEMM_OPERATOR_DESC).cast(),
    };
    unsafe { execute_operator(&operator, &[left, right, bias], rows * output_columns) }
        .map_err(|error| as_neural("GEMM", error))
}

pub(crate) fn vector_add_report(
    selection: BackendSelection,
    len: usize,
    expected_checksum: f64,
) -> Result<BackendDispatchReport> {
    let left = (0..len).map(|index| index as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|index| index as f32 * 1.5).collect::<Vec<_>>();
    let tensor = TensorDescription::f32(&[1, 1, 1, len as u32]);
    let add = DML_ELEMENT_WISE_ADD_OPERATOR_DESC {
        ATensor: &*tensor.tensor,
        BTensor: &*tensor.tensor,
        OutputTensor: &*tensor.tensor,
    };
    let operator = DML_OPERATOR_DESC {
        Type: DML_OPERATOR_ELEMENT_WISE_ADD,
        Desc: (&add as *const DML_ELEMENT_WISE_ADD_OPERATOR_DESC).cast(),
    };
    let start = std::time::Instant::now();
    let output = unsafe { execute_operator(&operator, &[&left, &right], len) }
        .map_err(|error| as_neural("vector add", error))?;
    Ok(BackendDispatchReport {
        requested: selection.requested,
        selected: selection.selected,
        operation: "vector_add_f32".to_string(),
        len,
        checksum: output.iter().map(|value| f64::from(*value)).sum(),
        expected_checksum,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        accelerated: true,
    })
}

pub(crate) fn affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    let rows = features.len();
    let columns = weights.len();
    let centered = features
        .iter()
        .flat_map(|row| {
            row.iter()
                .zip(means)
                .map(|(value, mean)| (value - mean) as f32)
        })
        .collect::<Vec<_>>();
    let weights = weights
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let bias = intercepts
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    directml_gemm(&centered, rows, columns, &weights, 1, &bias)
        .map(|values| values.into_iter().map(f64::from).collect())
}
pub(crate) fn dense_layer(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let rows = features.len();
    let columns = features[0].len();
    let output_columns = biases.len();
    let features = features.iter().flatten().copied().collect::<Vec<_>>();
    let output = directml_gemm(&features, rows, columns, weights, output_columns, biases)?;
    Ok(output.chunks(output_columns).map(<[f32]>::to_vec).collect())
}
pub(crate) fn pair_sigmoid_scores(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    use std::collections::BTreeMap;

    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let dimensions = embeddings[0].len();
    let mut sources = pairs.iter().map(|(source, _)| *source).collect::<Vec<_>>();
    let mut targets = pairs.iter().map(|(_, target)| *target).collect::<Vec<_>>();
    sources.sort_unstable();
    sources.dedup();
    targets.sort_unstable();
    targets.dedup();
    let source_positions = sources
        .iter()
        .enumerate()
        .map(|(position, node)| (*node, position))
        .collect::<BTreeMap<_, _>>();
    let target_positions = targets
        .iter()
        .enumerate()
        .map(|(position, node)| (*node, position))
        .collect::<BTreeMap<_, _>>();
    let left = sources
        .iter()
        .flat_map(|node| embeddings[*node].iter().copied())
        .collect::<Vec<_>>();
    // GEMM expects a [dimensions, targets] right-hand matrix.
    let mut right = vec![0.0_f32; dimensions * targets.len()];
    for (column, node) in targets.iter().enumerate() {
        for (dimension, value) in embeddings[*node].iter().enumerate() {
            right[dimension * targets.len() + column] = *value;
        }
    }
    let bias = vec![0.0_f32; targets.len()];
    let logits = directml_gemm(
        &left,
        sources.len(),
        dimensions,
        &right,
        targets.len(),
        &bias,
    )?;
    Ok(pairs
        .iter()
        .map(|(source, target)| {
            let row = source_positions[source];
            let column = target_positions[target];
            let logit = f64::from(logits[row * targets.len() + column]);
            1.0 / (1.0 + (-logit).exp())
        })
        .collect())
}
pub(crate) fn csr_diffusion(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    let nodes = indptr.len() - 1;
    let batch = values.len() / (nodes * channels);
    let output_columns = batch * channels;
    let mut adjacency = vec![0.0_f32; nodes * nodes];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            adjacency[row * nodes + indices[edge] as usize] += weights[edge];
        }
    }
    // DirectML GEMM consumes [nodes,nodes] @ [nodes,batch*channels].
    // Public tensors are [batch,nodes,channels], so transpose only at the
    // API boundary and keep the matrix multiplication itself on the device.
    let mut node_major = vec![0.0_f32; nodes * output_columns];
    for batch_index in 0..batch {
        for node in 0..nodes {
            for channel in 0..channels {
                node_major[node * output_columns + batch_index * channels + channel] =
                    values[(batch_index * nodes + node) * channels + channel];
            }
        }
    }
    let bias = vec![0.0_f32; output_columns];
    let node_output = directml_gemm(&adjacency, nodes, nodes, &node_major, output_columns, &bias)?;
    let mut output = vec![0.0_f32; values.len()];
    for batch_index in 0..batch {
        for node in 0..nodes {
            for channel in 0..channels {
                output[(batch_index * nodes + node) * channels + channel] =
                    node_output[node * output_columns + batch_index * channels + channel];
            }
        }
    }
    Ok(output)
}
pub(crate) fn csr_diffusion_backward(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    let nodes = indptr.len() - 1;
    let batch = values.len() / (nodes * channels);
    let columns = batch * channels;
    let mut adjacency_transpose = vec![0.0_f32; nodes * nodes];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            adjacency_transpose[indices[edge] as usize * nodes + row] += weights[edge];
        }
    }
    let mut grad_node_major = vec![0.0_f32; nodes * columns];
    for batch_index in 0..batch {
        for node in 0..nodes {
            for channel in 0..channels {
                grad_node_major[node * columns + batch_index * channels + channel] =
                    output_grad[(batch_index * nodes + node) * channels + channel];
            }
        }
    }
    let node_input_grad = directml_gemm(
        &adjacency_transpose,
        nodes,
        nodes,
        &grad_node_major,
        columns,
        &vec![0.0; columns],
    )?;
    let mut input_grad = vec![0.0_f32; values.len()];
    for batch_index in 0..batch {
        for node in 0..nodes {
            for channel in 0..channels {
                input_grad[(batch_index * nodes + node) * channels + channel] =
                    node_input_grad[node * columns + batch_index * channels + channel];
            }
        }
    }
    // Edge gradients are a reduction across batch and channels. DirectML's
    // GEMM computes the dominant input-gradient tensor; this compact gather
    // reduction avoids materializing a dense nodes-by-nodes gradient matrix.
    let mut edge_grad = vec![0.0_f32; weights.len()];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            let source = indices[edge] as usize;
            let mut gradient = 0.0_f32;
            for batch_index in 0..batch {
                for channel in 0..channels {
                    gradient += output_grad[(batch_index * nodes + row) * channels + channel]
                        * values[(batch_index * nodes + source) * channels + channel];
                }
            }
            edge_grad[edge] = gradient;
        }
    }
    Ok(CsrDiffusionBackward {
        input_grad,
        edge_grad,
    })
}
pub(crate) fn csr_row_softmax(_: &[u32], _: &[f32]) -> Result<Vec<f32>> {
    pending("CSR row softmax")
}
pub(crate) fn csr_row_softmax_backward(_: &[u32], _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
    pending("CSR row softmax backward")
}
pub(crate) fn adamw(
    _: &mut [f32],
    _: &mut [f32],
    _: &mut [f32],
    _: &[f32],
    _: u64,
    _: f32,
    _: f32,
) -> Result<()> {
    pending("AdamW")
}
pub(crate) fn layer_norm(_: &[f32], _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
    pending("layer normalization")
}
pub(crate) fn scalar_graph(_: &[f32], _: &[u8], _: &[u32], _: &[u32]) -> Result<Vec<f32>> {
    pending("scalar graph")
}
