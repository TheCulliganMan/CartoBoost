use crate::{BackendDispatchReport, BackendSelection, NeuralError, Result};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vector_add_f32(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            *output_value = left[raw_index] + right[raw_index];
        }
    }
}

pub(super) fn is_available() -> bool {
    CudaContext::new(0).is_ok()
}

pub(super) fn vector_add_report(
    selection: BackendSelection,
    len: usize,
    expected_checksum: f64,
) -> Result<BackendDispatchReport> {
    let left = (0..len).map(|index| index as f32 * 0.5).collect::<Vec<_>>();
    let right = (0..len).map(|index| index as f32 * 1.5).collect::<Vec<_>>();
    let start = Instant::now();
    let context = CudaContext::new(0).map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to create cuda-oxide context: {error}"))
    })?;
    let stream = context.default_stream();
    let left_device = DeviceBuffer::from_host(&stream, &left).map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to upload cuda-oxide left tensor: {error}"))
    })?;
    let right_device = DeviceBuffer::from_host(&stream, &right).map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to upload cuda-oxide right tensor: {error}"))
    })?;
    let mut output_device = DeviceBuffer::<f32>::zeroed(&stream, len).map_err(|error| {
        NeuralError::InvalidArgument(format!(
            "failed to allocate cuda-oxide output tensor: {error}"
        ))
    })?;
    let module = kernels::load(&context).map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.vector_add_f32(
            &stream,
            LaunchConfig::for_num_elems(len as u32),
            &left_device,
            &right_device,
            &mut output_device,
        )
    }
    .map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to launch cuda-oxide vector add: {error}"))
    })?;
    let output = output_device.to_host_vec(&stream).map_err(|error| {
        NeuralError::InvalidArgument(format!("failed to read cuda-oxide output tensor: {error}"))
    })?;

    Ok(BackendDispatchReport {
        requested: selection.requested,
        selected: selection.selected,
        operation: "vector_add_f32".to_string(),
        len,
        checksum: output.into_iter().map(f64::from).sum(),
        expected_checksum,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        accelerated: true,
    })
}
