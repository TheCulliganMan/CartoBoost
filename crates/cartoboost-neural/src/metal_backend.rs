use crate::backend::{BackendDispatchReport, BackendSelection, CsrDiffusionBackward};
use crate::{NeuralError, Result};
#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
use std::time::Instant;

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
fn metal_pipeline(
    source: &str,
    kernel: &str,
) -> Result<(
    metal::Device,
    metal::CommandQueue,
    metal::ComputePipelineState,
)> {
    use metal::{CompileOptions, Device};

    let device = Device::system_default()
        .ok_or_else(|| NeuralError::InvalidArgument("no Metal device is available".to_string()))?;
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            NeuralError::InvalidArgument(format!(
                "failed to compile Metal kernel {kernel}: {error}"
            ))
        })?;
    let function = library.get_function(kernel, None).map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to load Metal kernel {kernel}: {error}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            NeuralError::InvalidArgument(format!(
                "failed to create Metal pipeline {kernel}: {error}"
            ))
        })?;
    let queue = device.new_command_queue();
    Ok((device, queue, pipeline))
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
fn metal_input_buffer<T>(device: &metal::DeviceRef, values: &[T]) -> metal::Buffer {
    use metal::MTLResourceOptions;

    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values).max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    )
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
fn metal_output_buffer<T>(device: &metal::DeviceRef, len: usize) -> metal::Buffer {
    use metal::MTLResourceOptions;

    device.new_buffer(
        (len * std::mem::size_of::<T>()).max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    )
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
fn metal_dispatch(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    buffers: &[&metal::BufferRef],
    threads: usize,
) {
    use metal::MTLSize;

    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (slot, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(slot as u64, Some(buffer), 0);
    }
    let width = pipeline
        .thread_execution_width()
        .max(1)
        .min(threads.max(1) as u64);
    encoder.dispatch_threads(
        MTLSize::new(threads.max(1) as u64, 1, 1),
        MTLSize::new(width, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
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
fn metal_read<T: Copy>(buffer: &metal::BufferRef, len: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), len).to_vec() }
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
pub(crate) fn with_metal_autoreleasepool<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
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
pub(crate) fn with_metal_autoreleasepool<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    operation()
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
pub(crate) fn metal_scalar_graph_f32(
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
pub(crate) fn metal_scalar_graph_train_step_f32(
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
pub(crate) fn metal_scalar_graph_f32(
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
pub(crate) fn metal_scalar_graph_train_step_f32(
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
#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
pub(crate) fn metal_vector_add_report(
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
pub(crate) fn metal_vector_add_report(
    _selection: BackendSelection,
    _len: usize,
) -> Result<BackendDispatchReport> {
    Err(NeuralError::InvalidArgument(
        "Metal dispatch is not available in this build".to_string(),
    ))
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
pub(crate) fn metal_affine_scores(
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
pub(crate) fn metal_csr_diffusion_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void csr_diffusion_f32(
            const device uint* indptr [[buffer(0)]],
            const device uint* indices [[buffer(1)]],
            const device float* weights [[buffer(2)]],
            const device float* values [[buffer(3)]],
            device float* output [[buffer(4)]],
            constant uint& nodes [[buffer(5)]],
            constant uint& channels [[buffer(6)]],
            uint id [[thread_position_in_grid]]) {
            uint channel = id % channels;
            uint row = (id / channels) % nodes;
            uint batch = id / (nodes * channels);
            float sum = 0.0f;
            for (uint edge = indptr[row]; edge < indptr[row + 1]; ++edge) {
                uint source = indices[edge];
                sum += weights[edge] * values[(batch * nodes + source) * channels + channel];
            }
            output[id] = sum;
        }
    "#;

    with_metal_autoreleasepool(|| {
        let nodes = (indptr.len() - 1) as u32;
        let channel_count = channels as u32;
        let (device, queue, pipeline) = metal_pipeline(SOURCE, "csr_diffusion_f32")?;
        let indptr_buffer = metal_input_buffer(&device, indptr);
        let indices_buffer = metal_input_buffer(&device, indices);
        let weights_buffer = metal_input_buffer(&device, weights);
        let values_buffer = metal_input_buffer(&device, values);
        let output_buffer = metal_output_buffer::<f32>(&device, values.len());
        let nodes_buffer = metal_input_buffer(&device, &[nodes]);
        let channels_buffer = metal_input_buffer(&device, &[channel_count]);
        metal_dispatch(
            &queue,
            &pipeline,
            &[
                &indptr_buffer,
                &indices_buffer,
                &weights_buffer,
                &values_buffer,
                &output_buffer,
                &nodes_buffer,
                &channels_buffer,
            ],
            values.len(),
        );
        Ok(metal_read(&output_buffer, values.len()))
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
pub(crate) fn metal_csr_diffusion_backward_f32(
    _indptr: &[u32],
    _indices: &[u32],
    _weights: &[f32],
    _channels: usize,
    _values: &[f32],
    _output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    Err(NeuralError::InvalidArgument(
        "Metal CSR diffusion backward is not available in this build".to_string(),
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
pub(crate) fn metal_csr_row_softmax_f32(_indptr: &[u32], _logits: &[f32]) -> Result<Vec<f32>> {
    Err(NeuralError::InvalidArgument(
        "Metal CSR row softmax is not available in this build".to_string(),
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
pub(crate) fn metal_csr_row_softmax_backward_f32(
    _indptr: &[u32],
    _weights: &[f32],
    _grad: &[f32],
) -> Result<Vec<f32>> {
    Err(NeuralError::InvalidArgument(
        "Metal CSR row softmax backward is not available in this build".to_string(),
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
pub(crate) fn metal_adamw_step_f32(
    _parameters: &mut [f32],
    _first: &mut [f32],
    _second: &mut [f32],
    _gradients: &[f32],
    _step: u64,
    _lr: f32,
    _decay: f32,
) -> Result<()> {
    Err(NeuralError::InvalidArgument(
        "Metal AdamW is not available in this build".to_string(),
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
pub(crate) fn metal_layer_norm_f32(
    _values: &[f32],
    _rows: usize,
    _width: usize,
    _gamma: &[f32],
    _beta: &[f32],
) -> Result<Vec<f32>> {
    Err(NeuralError::InvalidArgument(
        "Metal layer normalization is not available in this build".to_string(),
    ))
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
pub(crate) fn metal_csr_diffusion_backward_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_grad: &[f32],
) -> Result<CsrDiffusionBackward> {
    let nodes = indptr.len() - 1;
    let batches = values.len() / (nodes * channels);
    let mut transpose_rows = vec![Vec::<(u32, f32)>::new(); nodes];
    let mut edge_rows = vec![0_u32; indices.len()];
    for row in 0..nodes {
        for edge in indptr[row] as usize..indptr[row + 1] as usize {
            edge_rows[edge] = row as u32;
            transpose_rows[indices[edge] as usize].push((row as u32, weights[edge]));
        }
    }
    let mut transpose_indptr = Vec::with_capacity(nodes + 1);
    let mut transpose_indices = Vec::with_capacity(indices.len());
    let mut transpose_weights = Vec::with_capacity(weights.len());
    transpose_indptr.push(0);
    for row in transpose_rows {
        for (index, weight) in row {
            transpose_indices.push(index);
            transpose_weights.push(weight);
        }
        transpose_indptr.push(transpose_indices.len() as u32);
    }
    let input_grad = metal_csr_diffusion_f32(
        &transpose_indptr,
        &transpose_indices,
        &transpose_weights,
        channels,
        output_grad,
    )?;

    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void csr_edge_grad_f32(
            const device uint* edge_rows [[buffer(0)]], const device uint* indices [[buffer(1)]],
            const device float* values [[buffer(2)]], const device float* grad [[buffer(3)]],
            device float* edge_grad [[buffer(4)]], constant uint& nodes [[buffer(5)]],
            constant uint& channels [[buffer(6)]], constant uint& batches [[buffer(7)]],
            uint edge [[thread_position_in_grid]]) {
            uint row = edge_rows[edge], source = indices[edge]; float sum = 0.0f;
            for (uint batch = 0; batch < batches; ++batch)
                for (uint channel = 0; channel < channels; ++channel)
                    sum += grad[(batch * nodes + row) * channels + channel] *
                           values[(batch * nodes + source) * channels + channel];
            edge_grad[edge] = sum;
        }
    "#;
    let edge_grad = with_metal_autoreleasepool(|| {
        let (device, queue, pipeline) = metal_pipeline(SOURCE, "csr_edge_grad_f32")?;
        let rows = metal_input_buffer(&device, &edge_rows);
        let idx = metal_input_buffer(&device, indices);
        let vals = metal_input_buffer(&device, values);
        let grad = metal_input_buffer(&device, output_grad);
        let out = metal_output_buffer::<f32>(&device, indices.len());
        let n = metal_input_buffer(&device, &[nodes as u32]);
        let c = metal_input_buffer(&device, &[channels as u32]);
        let b = metal_input_buffer(&device, &[batches as u32]);
        metal_dispatch(
            &queue,
            &pipeline,
            &[&rows, &idx, &vals, &grad, &out, &n, &c, &b],
            indices.len(),
        );
        Ok(metal_read(&out, indices.len()))
    })?;
    Ok(CsrDiffusionBackward {
        input_grad,
        edge_grad,
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
pub(crate) fn metal_csr_row_softmax_f32(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void csr_softmax(const device uint* p [[buffer(0)]], const device float* x [[buffer(1)]],
          device float* y [[buffer(2)]], uint row [[thread_position_in_grid]]) {
          uint begin=p[row], end=p[row+1]; if(begin==end) return; float m=x[begin];
          for(uint e=begin+1;e<end;++e) m=max(m,x[e]); float s=0.0f;
          for(uint e=begin;e<end;++e) s+=exp(x[e]-m); for(uint e=begin;e<end;++e) y[e]=exp(x[e]-m)/s;
        }
    "#;
    with_metal_autoreleasepool(|| {
        let (d, q, k) = metal_pipeline(SOURCE, "csr_softmax")?;
        let p = metal_input_buffer(&d, indptr);
        let x = metal_input_buffer(&d, logits);
        let y = metal_output_buffer::<f32>(&d, logits.len());
        metal_dispatch(&q, &k, &[&p, &x, &y], indptr.len() - 1);
        Ok(metal_read(&y, logits.len()))
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
pub(crate) fn metal_csr_row_softmax_backward_f32(
    indptr: &[u32],
    weights: &[f32],
    grad: &[f32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void csr_softmax_backward(const device uint* p [[buffer(0)]], const device float* w [[buffer(1)]],
          const device float* g [[buffer(2)]], device float* y [[buffer(3)]], uint row [[thread_position_in_grid]]) {
          uint begin=p[row],end=p[row+1]; float dot=0.0f; for(uint e=begin;e<end;++e) dot+=w[e]*g[e];
          for(uint e=begin;e<end;++e) y[e]=w[e]*(g[e]-dot);
        }
    "#;
    with_metal_autoreleasepool(|| {
        let (d, q, k) = metal_pipeline(SOURCE, "csr_softmax_backward")?;
        let p = metal_input_buffer(&d, indptr);
        let w = metal_input_buffer(&d, weights);
        let g = metal_input_buffer(&d, grad);
        let y = metal_output_buffer::<f32>(&d, weights.len());
        metal_dispatch(&q, &k, &[&p, &w, &g, &y], indptr.len() - 1);
        Ok(metal_read(&y, weights.len()))
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
pub(crate) fn metal_adamw_step_f32(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    gradients: &[f32],
    step: u64,
    lr: f32,
    decay: f32,
) -> Result<()> {
    const SOURCE: &str = r#"
      #include <metal_stdlib>
      using namespace metal;
      kernel void adamw(device float* p[[buffer(0)]],device float* m[[buffer(1)]],device float* v[[buffer(2)]],
       const device float* g[[buffer(3)]],constant float& lr[[buffer(4)]],constant float& decay[[buffer(5)]],
       constant float& bc1[[buffer(6)]],constant float& bc2[[buffer(7)]],uint i[[thread_position_in_grid]]){
       float mi=0.9f*m[i]+0.1f*g[i]; float vi=0.999f*v[i]+0.001f*g[i]*g[i]; m[i]=mi; v[i]=vi;
       p[i]=p[i]*(1.0f-lr*decay)-lr*(mi/bc1)/(sqrt(vi/bc2)+1.0e-8f); }
    "#;
    with_metal_autoreleasepool(|| {
        let (d, q, k) = metal_pipeline(SOURCE, "adamw")?;
        let p = metal_input_buffer(&d, parameters);
        let m = metal_input_buffer(&d, first);
        let v = metal_input_buffer(&d, second);
        let g = metal_input_buffer(&d, gradients);
        let l = metal_input_buffer(&d, &[lr]);
        let wd = metal_input_buffer(&d, &[decay]);
        let b1 = metal_input_buffer(&d, &[1.0 - 0.9_f32.powi(step.min(i32::MAX as u64) as i32)]);
        let b2 = metal_input_buffer(
            &d,
            &[1.0 - 0.999_f32.powi(step.min(i32::MAX as u64) as i32)],
        );
        metal_dispatch(
            &q,
            &k,
            &[&p, &m, &v, &g, &l, &wd, &b1, &b2],
            parameters.len(),
        );
        parameters.copy_from_slice(&metal_read(&p, parameters.len()));
        first.copy_from_slice(&metal_read(&m, first.len()));
        second.copy_from_slice(&metal_read(&v, second.len()));
        Ok(())
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
pub(crate) fn metal_layer_norm_f32(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    const SOURCE: &str = r#"
  #include <metal_stdlib>
  using namespace metal;
  kernel void layer_norm(const device float*x[[buffer(0)]],const device float*g[[buffer(1)]],const device float*b[[buffer(2)]],device float*y[[buffer(3)]],constant uint&w[[buffer(4)]],uint row[[thread_position_in_grid]]){
   uint o=row*w; float mean=0.0f; for(uint i=0;i<w;++i) mean+=x[o+i]; mean/=float(w); float var=0.0f; for(uint i=0;i<w;++i){float d=x[o+i]-mean;var+=d*d;} var/=float(w); float inv=rsqrt(var+1.0e-5f); for(uint i=0;i<w;++i)y[o+i]=(x[o+i]-mean)*inv*g[i]+b[i]; }
 "#;
    with_metal_autoreleasepool(|| {
        let (d, q, k) = metal_pipeline(SOURCE, "layer_norm")?;
        let x = metal_input_buffer(&d, values);
        let g = metal_input_buffer(&d, gamma);
        let b = metal_input_buffer(&d, beta);
        let y = metal_output_buffer::<f32>(&d, values.len());
        let w = metal_input_buffer(&d, &[width as u32]);
        metal_dispatch(&q, &k, &[&x, &g, &b, &y, &w], rows);
        Ok(metal_read(&y, values.len()))
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
pub(crate) fn metal_dense_layer_f32(
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

#[cfg(not(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
)))]
pub(crate) fn metal_csr_diffusion_f32(
    _indptr: &[u32],
    _indices: &[u32],
    _weights: &[f32],
    _channels: usize,
    _values: &[f32],
) -> Result<Vec<f32>> {
    Err(NeuralError::InvalidArgument(
        "Metal CSR diffusion is not available in this build".to_string(),
    ))
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
pub(crate) fn metal_pair_sigmoid_scores_f32(
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
pub(crate) fn metal_train_tanh_mlp_f32(
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
pub(crate) fn metal_affine_scores(
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
pub(crate) fn metal_dense_layer_f32(
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
pub(crate) fn metal_pair_sigmoid_scores_f32(
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
pub(crate) fn metal_train_tanh_mlp_f32(
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
