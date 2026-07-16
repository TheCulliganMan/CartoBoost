//! Windows DirectML tensor backend.
//!
//! All DirectML implementation details live here so the shared dispatcher
//! remains independent from CUDA, ROCm, WebGPU, and CPU implementation work.

use crate::backend::{BackendDispatchReport, BackendSelection, CsrDiffusionBackward};
use crate::{NeuralError, Result};
use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    ptr,
    sync::{Mutex, OnceLock},
};
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
    execution_lock: Mutex<()>,
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
                execution_lock: Mutex::new(()),
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
    let _execution_guard = runtime
        .execution_lock
        .lock()
        .map_err(|_| windows::core::Error::empty())?;
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
    let mut upload_buffers = Vec::with_capacity(inputs.len());
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
        device_inputs.push((device, size));
        upload_buffers.push(upload);
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
        .map(|(buffer, size)| buffer_binding(buffer, *size))
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
        let barriers = [transition(
            &output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )];
        list.ResourceBarrier(&barriers);
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

struct TensorDescription {
    _sizes: Vec<u32>,
    _strides: Option<Vec<u32>>,
    _buffer: Box<DML_BUFFER_TENSOR_DESC>,
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
            _sizes: sizes,
            _strides: strides,
            _buffer: buffer,
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
    let bias_tensor = TensorDescription::f32(&[1, 1, rows as u32, output_columns as u32]);
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
    let expanded_bias = (0..rows)
        .flat_map(|_| bias.iter().copied())
        .collect::<Vec<_>>();
    unsafe {
        execute_operator(
            &operator,
            &[left, right, &expanded_bias],
            rows * output_columns,
        )
    }
    .map_err(|error| as_neural("GEMM", error))
}

fn directml_batched_gemm(
    left: &[f32],
    batch: usize,
    rows: usize,
    inner: usize,
    right: &[f32],
    output_columns: usize,
    transpose_left: bool,
) -> Result<Vec<f32>> {
    let left_sizes = [batch as u32, 1, rows as u32, inner as u32];
    let output_rows = if transpose_left { inner } else { rows };
    let left_tensor = TensorDescription::f32(&left_sizes);
    let right_tensor =
        TensorDescription::f32(&[batch as u32, 1, inner as u32, output_columns as u32]);
    let output_tensor =
        TensorDescription::f32(&[batch as u32, 1, output_rows as u32, output_columns as u32]);
    let zero_tensor =
        TensorDescription::f32(&[batch as u32, 1, output_rows as u32, output_columns as u32]);
    let gemm = DML_GEMM_OPERATOR_DESC {
        ATensor: &*left_tensor.tensor,
        BTensor: &*right_tensor.tensor,
        CTensor: &*zero_tensor.tensor,
        OutputTensor: &*output_tensor.tensor,
        TransA: if transpose_left {
            DML_MATRIX_TRANSFORM_TRANSPOSE
        } else {
            DML_MATRIX_TRANSFORM_NONE
        },
        TransB: DML_MATRIX_TRANSFORM_NONE,
        Alpha: 1.0,
        Beta: 1.0,
        FusedActivation: ptr::null(),
    };
    let operator = DML_OPERATOR_DESC {
        Type: DML_OPERATOR_GEMM,
        Desc: (&gemm as *const DML_GEMM_OPERATOR_DESC).cast(),
    };
    let zeros = vec![0.0; batch * output_rows * output_columns];
    unsafe {
        execute_operator(
            &operator,
            &[left, right, &zeros],
            batch * output_rows * output_columns,
        )
    }
    .map_err(|error| as_neural("batched GEMM", error))
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Multiply,
    Divide,
    Maximum,
    Subtract,
}

fn elementwise_binary(operation: BinaryOperation, left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    let tensor = TensorDescription::f32(&[1, 1, 1, left.len() as u32]);
    macro_rules! run {
        ($description:ident, $operator_type:ident) => {{
            let description = $description {
                ATensor: &*tensor.tensor,
                BTensor: &*tensor.tensor,
                OutputTensor: &*tensor.tensor,
            };
            let operator = DML_OPERATOR_DESC {
                Type: $operator_type,
                Desc: (&description as *const $description).cast(),
            };
            unsafe { execute_operator(&operator, &[left, right], left.len()) }
        }};
    }
    let result = match operation {
        BinaryOperation::Add => run!(
            DML_ELEMENT_WISE_ADD_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_ADD
        ),
        BinaryOperation::Multiply => run!(
            DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_MULTIPLY
        ),
        BinaryOperation::Divide => run!(
            DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_DIVIDE
        ),
        BinaryOperation::Maximum => run!(
            DML_ELEMENT_WISE_MAX_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_MAX
        ),
        BinaryOperation::Subtract => run!(
            DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_SUBTRACT
        ),
    };
    result.map_err(|error| as_neural("elementwise binary operator", error))
}

#[derive(Clone, Copy)]
enum UnaryOperation {
    Exp,
    Sin,
    Sqrt,
    Tanh,
    Sigmoid,
}

fn elementwise_unary(operation: UnaryOperation, input: &[f32]) -> Result<Vec<f32>> {
    let tensor = TensorDescription::f32(&[1, 1, 1, input.len() as u32]);
    macro_rules! run_scaled {
        ($description:ident, $operator_type:ident) => {{
            let description = $description {
                InputTensor: &*tensor.tensor,
                OutputTensor: &*tensor.tensor,
                ScaleBias: ptr::null(),
            };
            let operator = DML_OPERATOR_DESC {
                Type: $operator_type,
                Desc: (&description as *const $description).cast(),
            };
            unsafe { execute_operator(&operator, &[input], input.len()) }
        }};
    }
    let result = match operation {
        UnaryOperation::Exp => run_scaled!(
            DML_ELEMENT_WISE_EXP_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_EXP
        ),
        UnaryOperation::Sin => run_scaled!(
            DML_ELEMENT_WISE_SIN_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_SIN
        ),
        UnaryOperation::Sqrt => run_scaled!(
            DML_ELEMENT_WISE_SQRT_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_SQRT
        ),
        UnaryOperation::Tanh => {
            let description = DML_ACTIVATION_TANH_OPERATOR_DESC {
                InputTensor: &*tensor.tensor,
                OutputTensor: &*tensor.tensor,
            };
            let operator = DML_OPERATOR_DESC {
                Type: DML_OPERATOR_ACTIVATION_TANH,
                Desc: (&description as *const DML_ACTIVATION_TANH_OPERATOR_DESC).cast(),
            };
            unsafe { execute_operator(&operator, &[input], input.len()) }
        }
        UnaryOperation::Sigmoid => {
            let description = DML_ACTIVATION_SIGMOID_OPERATOR_DESC {
                InputTensor: &*tensor.tensor,
                OutputTensor: &*tensor.tensor,
            };
            let operator = DML_OPERATOR_DESC {
                Type: DML_OPERATOR_ACTIVATION_SIGMOID,
                Desc: (&description as *const DML_ACTIVATION_SIGMOID_OPERATOR_DESC).cast(),
            };
            unsafe { execute_operator(&operator, &[input], input.len()) }
        }
    };
    result.map_err(|error| as_neural("elementwise unary operator", error))
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
    let batch = pairs.len();
    let width = embeddings[0].len();
    let left = pairs
        .iter()
        .flat_map(|(left, _)| embeddings[*left].iter().copied())
        .collect::<Vec<_>>();
    let right = pairs
        .iter()
        .flat_map(|(_, right)| embeddings[*right].iter().copied())
        .collect::<Vec<_>>();
    let left_tensor = TensorDescription::f32(&[batch as u32, 1, 1, width as u32]);
    let right_tensor = TensorDescription::f32(&[batch as u32, 1, width as u32, 1]);
    let output_tensor = TensorDescription::f32(&[batch as u32, 1, 1, 1]);
    let zero_tensor = TensorDescription::f32(&[batch as u32, 1, 1, 1]);
    let sigmoid = DML_ACTIVATION_SIGMOID_OPERATOR_DESC {
        InputTensor: ptr::null(),
        OutputTensor: ptr::null(),
    };
    let fused = DML_OPERATOR_DESC {
        Type: DML_OPERATOR_ACTIVATION_SIGMOID,
        Desc: (&sigmoid as *const DML_ACTIVATION_SIGMOID_OPERATOR_DESC).cast(),
    };
    let gemm = DML_GEMM_OPERATOR_DESC {
        ATensor: &*left_tensor.tensor,
        BTensor: &*right_tensor.tensor,
        CTensor: &*zero_tensor.tensor,
        OutputTensor: &*output_tensor.tensor,
        TransA: DML_MATRIX_TRANSFORM_NONE,
        TransB: DML_MATRIX_TRANSFORM_NONE,
        Alpha: 1.0,
        Beta: 1.0,
        FusedActivation: &fused,
    };
    let operator = DML_OPERATOR_DESC {
        Type: DML_OPERATOR_GEMM,
        Desc: (&gemm as *const DML_GEMM_OPERATOR_DESC).cast(),
    };
    let zeros = vec![0.0; batch];
    unsafe { execute_operator(&operator, &[&left, &right, &zeros], batch) }
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| as_neural("pair sigmoid scoring", error))
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
    let mut matrix = vec![0.0_f32; nodes * nodes];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            matrix[row * nodes + indices[edge] as usize] += weights[edge];
        }
    }
    let matrices = (0..batch)
        .flat_map(|_| matrix.iter().copied())
        .collect::<Vec<_>>();
    directml_batched_gemm(&matrices, batch, nodes, nodes, values, channels, false)
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
    let mut matrix = vec![0.0_f32; nodes * nodes];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            matrix[row * nodes + indices[edge] as usize] += weights[edge];
        }
    }
    let matrices = (0..batch)
        .flat_map(|_| matrix.iter().copied())
        .collect::<Vec<_>>();
    let input_grad =
        directml_batched_gemm(&matrices, batch, nodes, nodes, output_grad, channels, true)?;
    let mut edge_grad = vec![0.0_f32; weights.len()];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            let source = indices[edge] as usize;
            for batch_index in 0..batch {
                for channel in 0..channels {
                    edge_grad[edge] += output_grad
                        [(batch_index * nodes + row) * channels + channel]
                        * values[(batch_index * nodes + source) * channels + channel];
                }
            }
        }
    }
    Ok(CsrDiffusionBackward {
        input_grad,
        edge_grad,
    })
}
pub(crate) fn csr_row_softmax(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    let rows = indptr.len() - 1;
    let mut shifted = logits.to_vec();
    for row in 0..rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        if start == end {
            continue;
        }
        let maximum = logits[start..end]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        for value in &mut shifted[start..end] {
            *value -= maximum;
        }
    }
    let exponentials = elementwise_unary(UnaryOperation::Exp, &shifted)?;
    let mut denominators = vec![1.0_f32; logits.len()];
    for row in 0..rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        if start == end {
            continue;
        }
        denominators[start..end].fill(exponentials[start..end].iter().sum());
    }
    elementwise_binary(BinaryOperation::Divide, &exponentials, &denominators)
}
pub(crate) fn csr_row_softmax_backward(
    indptr: &[u32],
    weights: &[f32],
    output_grad: &[f32],
) -> Result<Vec<f32>> {
    let products = elementwise_binary(BinaryOperation::Multiply, weights, output_grad)?;
    let mut row_dot = vec![0.0_f32; weights.len()];
    for row in 0..indptr.len() - 1 {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let dot = products[start..end].iter().sum::<f32>();
        row_dot[start..end].fill(dot);
    }
    let centered = elementwise_binary(BinaryOperation::Subtract, output_grad, &row_dot)?;
    elementwise_binary(BinaryOperation::Multiply, weights, &centered)
}
pub(crate) fn adamw(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    gradients: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    let len = parameters.len();
    let constant = |value| vec![value; len];
    let first_decay = elementwise_binary(BinaryOperation::Multiply, first, &constant(BETA1))?;
    let first_gradient =
        elementwise_binary(BinaryOperation::Multiply, gradients, &constant(1.0 - BETA1))?;
    let updated_first = elementwise_binary(BinaryOperation::Add, &first_decay, &first_gradient)?;
    let gradient_square = elementwise_binary(BinaryOperation::Multiply, gradients, gradients)?;
    let second_decay = elementwise_binary(BinaryOperation::Multiply, second, &constant(BETA2))?;
    let second_gradient = elementwise_binary(
        BinaryOperation::Multiply,
        &gradient_square,
        &constant(1.0 - BETA2),
    )?;
    let updated_second = elementwise_binary(BinaryOperation::Add, &second_decay, &second_gradient)?;
    let first_hat = elementwise_binary(
        BinaryOperation::Divide,
        &updated_first,
        &constant(1.0 - BETA1.powi(step as i32)),
    )?;
    let second_hat = elementwise_binary(
        BinaryOperation::Divide,
        &updated_second,
        &constant(1.0 - BETA2.powi(step as i32)),
    )?;
    let root = elementwise_unary(UnaryOperation::Sqrt, &second_hat)?;
    let denominator = elementwise_binary(BinaryOperation::Add, &root, &constant(1.0e-8))?;
    let normalized = elementwise_binary(BinaryOperation::Divide, &first_hat, &denominator)?;
    let decay = elementwise_binary(
        BinaryOperation::Multiply,
        parameters,
        &constant(weight_decay),
    )?;
    let update = elementwise_binary(BinaryOperation::Add, &normalized, &decay)?;
    let scaled = elementwise_binary(BinaryOperation::Multiply, &update, &constant(learning_rate))?;
    let updated_parameters = elementwise_binary(BinaryOperation::Subtract, parameters, &scaled)?;
    parameters.copy_from_slice(&updated_parameters);
    first.copy_from_slice(&updated_first);
    second.copy_from_slice(&updated_second);
    Ok(())
}
pub(crate) fn layer_norm(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    let mut means = vec![0.0_f32; values.len()];
    let mut scales = vec![0.0_f32; values.len()];
    let mut expanded_gamma = vec![0.0_f32; values.len()];
    let mut expanded_beta = vec![0.0_f32; values.len()];
    for row in 0..rows {
        let range = row * width..(row + 1) * width;
        let mean = values[range.clone()].iter().sum::<f32>() / width as f32;
        let variance = values[range.clone()]
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / width as f32;
        means[range.clone()].fill(mean);
        scales[range.clone()].fill((variance + 1.0e-5).sqrt());
        expanded_gamma[range.clone()].copy_from_slice(gamma);
        expanded_beta[range].copy_from_slice(beta);
    }
    let centered = elementwise_binary(BinaryOperation::Subtract, values, &means)?;
    let normalized = elementwise_binary(BinaryOperation::Divide, &centered, &scales)?;
    let scaled = elementwise_binary(BinaryOperation::Multiply, &normalized, &expanded_gamma)?;
    elementwise_binary(BinaryOperation::Add, &scaled, &expanded_beta)
}
pub(crate) fn scalar_graph(
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    let mut values = initial_values.to_vec();
    for index in 0..opcodes.len() {
        let opcode = opcodes[index];
        if opcode < 2 {
            continue;
        }
        let a = [values[left[index] as usize]];
        let b = [values[right[index] as usize]];
        let result = match opcode {
            2 => elementwise_binary(BinaryOperation::Add, &a, &b)?,
            3 => elementwise_binary(BinaryOperation::Multiply, &a, &b)?,
            4 => {
                let denominator = elementwise_binary(BinaryOperation::Maximum, &b, &[1.0e-12])?;
                elementwise_binary(BinaryOperation::Divide, &a, &denominator)?
            }
            5 => elementwise_unary(UnaryOperation::Tanh, &a)?,
            6 => elementwise_unary(UnaryOperation::Exp, &a)?,
            7 => {
                let safe = elementwise_binary(BinaryOperation::Maximum, &a, &[1.0e-12])?;
                elementwise_unary(UnaryOperation::Sqrt, &safe)?
            }
            8 => elementwise_unary(UnaryOperation::Sin, &a)?,
            9 => elementwise_unary(UnaryOperation::Sigmoid, &a)?,
            10 => elementwise_binary(BinaryOperation::Maximum, &a, &b)?,
            11 => a.to_vec(),
            _ => vec![0.0],
        };
        values[index] = result[0];
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directml_device_and_vector_dispatch_are_real() {
        assert!(
            is_available(),
            "test host must expose a DirectML-capable D3D12 adapter"
        );
        let report = vector_add_report(
            BackendSelection {
                requested: "directml".to_string(),
                selected: "directml".to_string(),
                available: vec!["cpu".to_string(), "directml".to_string()],
            },
            64,
            4032.0,
        )
        .expect("DirectML vector dispatch should succeed");
        assert!(report.accelerated);
        assert_eq!(report.checksum, report.expected_checksum);
        let selection = crate::backend::select_backend(Some("dml")).unwrap();
        assert_eq!(selection.requested, "directml");
        assert_eq!(selection.selected, "directml");
        let public_report = crate::backend::backend_dispatch_report(Some("directml"), 64).unwrap();
        assert!(public_report.accelerated);
        assert_eq!(public_report.checksum, public_report.expected_checksum);
    }

    #[test]
    fn directml_dense_affine_and_pair_scoring_match_reference_values() {
        let dense = dense_layer(
            &[vec![1.0, 2.0], vec![-1.0, 0.5]],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.5, -0.5],
        )
        .unwrap();
        assert_close(&dense.concat(), &[7.5, 9.5, 1.0, -0.5], 1.0e-5);

        let affine = affine_scores(
            &[vec![2.0, 4.0], vec![0.0, 3.0]],
            &[1.0, 2.0],
            &[0.5, -2.0],
            &[1.0, -1.0],
        )
        .unwrap();
        assert_close(
            &affine.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            &[-2.5, -3.5],
            1.0e-5,
        );

        let pair = pair_sigmoid_scores(
            &[vec![1.0, 2.0], vec![3.0, 4.0], vec![-1.0, 1.0]],
            &[(0, 1), (0, 2)],
        )
        .unwrap();
        let expected = [
            1.0 / (1.0 + (-11.0_f64).exp()),
            1.0 / (1.0 + (-1.0_f64).exp()),
        ];
        for (actual, expected) in pair.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn directml_sparse_normalization_optimizer_and_graph_match_cuda_contract() {
        let indptr = [0, 2, 3];
        let indices = [0, 1, 0];
        let weights = [0.25, 0.75, 2.0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let diffusion = csr_diffusion(&indptr, &indices, &weights, 2, &values).unwrap();
        assert_close(&diffusion, &[2.5, 3.5, 2.0, 4.0], 1.0e-5);
        let backward = csr_diffusion_backward(
            &indptr,
            &indices,
            &weights,
            2,
            &values,
            &[1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        assert_close(&backward.input_grad, &[6.25, 8.5, 0.75, 1.5], 1.0e-5);
        assert_close(&backward.edge_grad, &[5.0, 11.0, 11.0], 1.0e-5);

        let logits = [1.0, 2.0, -1.0];
        let softmax = csr_row_softmax(&indptr, &logits).unwrap();
        assert_close(&softmax, &[0.268_941_43, 0.731_058_6, 1.0], 1.0e-5);
        let softmax_grad = csr_row_softmax_backward(&indptr, &softmax, &[2.0, -1.0, 4.0]).unwrap();
        assert_close(&softmax_grad, &[0.589_835_8, -0.589_835_8, 0.0], 1.0e-5);

        let normalized = layer_norm(&[1.0, 3.0, 2.0, 2.0], 2, 2, &[1.0, 1.0], &[0.0, 0.0]).unwrap();
        assert_close(&normalized, &[-0.999_995, 0.999_995, 0.0, 0.0], 1.0e-4);

        let graph = scalar_graph(
            &[2.0, 3.0, 0.0, 0.0, 0.0],
            &[0, 0, 2, 3, 9],
            &[0, 0, 0, 2, 3],
            &[0, 0, 1, 1, 0],
        )
        .unwrap();
        assert_close(&graph[2..], &[5.0, 15.0, 0.999_999_7], 1.0e-5);

        let mut parameters = [1.0, -2.0];
        let mut first = [0.0, 0.0];
        let mut second = [0.0, 0.0];
        adamw(
            &mut parameters,
            &mut first,
            &mut second,
            &[0.5, -0.25],
            1,
            0.01,
            0.1,
        )
        .unwrap();
        assert_close(&first, &[0.05, -0.025], 1.0e-6);
        assert_close(&second, &[0.00025, 0.0000625], 1.0e-7);
        assert_close(&parameters, &[0.989, -1.988], 1.0e-5);
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }
}
