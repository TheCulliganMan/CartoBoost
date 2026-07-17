use crate::{AcceleratorError, BackendDispatchReport, BackendSelection, Result};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, device, kernel, thread, DisjointSlice};
use std::sync::Arc;
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    fn device_sqrtf(value: f32) -> f32 {
        value.sqrt()
    }

    #[device]
    fn device_expf(value: f32) -> f32 {
        value.exp()
    }

    #[device]
    fn device_tanhf(value: f32) -> f32 {
        value.tanh()
    }

    #[device]
    fn device_sinf(value: f32) -> f32 {
        value.sin()
    }

    #[device]
    fn device_cosf(value: f32) -> f32 {
        value.cos()
    }

    #[device]
    fn device_powf(value: f32, exponent: f32) -> f32 {
        value.powf(exponent)
    }

    #[device]
    fn device_absf(value: f32) -> f32 {
        if value < 0.0 {
            -value
        } else {
            value
        }
    }

    // Keep existing kernel expressions concise while resolving `libm::...`
    // locally to CUDA-safe device helpers rather than the host libm crate.
    mod libm {
        #[inline(always)]
        pub fn sqrtf(value: f32) -> f32 {
            super::device_sqrtf(value)
        }

        #[inline(always)]
        pub fn expf(value: f32) -> f32 {
            super::device_expf(value)
        }

        #[inline(always)]
        pub fn tanhf(value: f32) -> f32 {
            super::device_tanhf(value)
        }

        #[inline(always)]
        pub fn sinf(value: f32) -> f32 {
            super::device_sinf(value)
        }

        #[inline(always)]
        pub fn cosf(value: f32) -> f32 {
            super::device_cosf(value)
        }

        #[inline(always)]
        pub fn powf(value: f32, exponent: f32) -> f32 {
            super::device_powf(value, exponent)
        }

        #[inline(always)]
        pub fn fabsf(value: f32) -> f32 {
            super::device_absf(value)
        }
    }

    #[kernel]
    pub fn vector_add_f32(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            *output_value = left[raw_index] + right[raw_index];
        }
    }

    #[kernel]
    pub fn affine_scores_f32(
        features: &[f32],
        means: &[f32],
        weights: &[f32],
        intercepts: &[f32],
        mut output: DisjointSlice<f32>,
        cols: u32,
    ) {
        let row = thread::index_1d();
        let raw_row = row.get();
        if let Some(output_value) = output.get_mut(row) {
            let mut score = intercepts[raw_row];
            let mut col = 0;
            while col < cols as usize {
                score += (features[raw_row * cols as usize + col] - means[col]) * weights[col];
                col += 1;
            }
            *output_value = score;
        }
    }

    #[kernel]
    pub fn dense_layer_f32(
        features: &[f32],
        weights: &[f32],
        biases: &[f32],
        mut output: DisjointSlice<f32>,
        cols: u32,
        out_dim: u32,
    ) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let raw_out_dim = out_dim as usize;
            let row = raw_index / raw_out_dim;
            let out = raw_index % raw_out_dim;
            let mut value = biases[out];
            let mut col = 0;
            while col < cols as usize {
                value += features[row * cols as usize + col] * weights[col * raw_out_dim + out];
                col += 1;
            }
            *output_value = value;
        }
    }

    #[kernel]
    pub fn affine_parameter_slice_f32(
        input: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        input_width: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let input_width = input_width as usize;
            let output_width = output_width as usize;
            let row = item / output_width;
            let column = item % output_width;
            let mut value = parameters[bias_offset as usize + column];
            let mut feature = 0;
            while feature < input_width {
                value += input[row * input_width + feature]
                    * parameters[weights_offset as usize + feature * output_width + column];
                feature += 1;
            }
            *output_value = value;
        }
    }

    #[kernel]
    pub fn affine_parameter_slice_input_backward_f32(
        output_gradient: &[f32],
        parameters: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        input_width: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = input_gradient.get_mut(index) {
            let input_width = input_width as usize;
            let output_width = output_width as usize;
            let row = item / input_width;
            let input = item % input_width;
            let mut sum = 0.0;
            let mut output = 0;
            while output < output_width {
                sum += output_gradient[row * output_width + output]
                    * parameters[weights_offset as usize + input * output_width + output];
                output += 1;
            }
            *gradient = sum;
        }
    }

    #[kernel]
    pub fn affine_parameter_slice_weight_backward_f32(
        input: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        rows: u32,
        input_width: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let start = weights_offset as usize;
        let end = start + input_width as usize * output_width as usize;
        if item >= start && item < end {
            if let Some(gradient) = parameter_gradient.get_mut(index) {
                let local = item - start;
                let input_column = local / output_width as usize;
                let output_column = local % output_width as usize;
                let mut sum = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    sum += input[row * input_width as usize + input_column]
                        * output_gradient[row * output_width as usize + output_column];
                    row += 1;
                }
                *gradient += sum;
            }
        }
    }

    #[kernel]
    pub fn affine_parameter_slice_bias_backward_f32(
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        bias_offset: u32,
        rows: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let start = bias_offset as usize;
        let end = start + output_width as usize;
        if item >= start && item < end {
            if let Some(gradient) = parameter_gradient.get_mut(index) {
                let output = item - start;
                let mut sum = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    sum += output_gradient[row * output_width as usize + output];
                    row += 1;
                }
                *gradient += sum;
            }
        }
    }

    #[kernel]
    pub fn affine_input_backward_f32(
        output_gradient: &[f32],
        weights: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        input_width: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            let input_width = input_width as usize;
            let output_width = output_width as usize;
            let row = item / input_width;
            let column = item % input_width;
            let mut sum = 0.0;
            let mut output = 0;
            while output < output_width {
                sum += output_gradient[row * output_width + output]
                    * weights[column * output_width + output];
                output += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn affine_weight_backward_f32(
        input: &[f32],
        output_gradient: &[f32],
        mut weight_gradient: DisjointSlice<f32>,
        rows: u32,
        input_width: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = weight_gradient.get_mut(index) {
            let input_width = input_width as usize;
            let output_width = output_width as usize;
            let column = item / output_width;
            let output = item % output_width;
            let mut sum = 0.0;
            let mut row = 0;
            while row < rows as usize {
                sum += input[row * input_width + column]
                    * output_gradient[row * output_width + output];
                row += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn affine_bias_backward_f32(
        output_gradient: &[f32],
        mut bias_gradient: DisjointSlice<f32>,
        rows: u32,
        output_width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = bias_gradient.get_mut(index) {
            let mut sum = 0.0;
            let mut row = 0;
            while row < rows as usize {
                sum += output_gradient[row * output_width as usize + item];
                row += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn matmul_f32(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
        _rows: u32,
        shared: u32,
        columns: u32,
    ) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let raw_columns = columns as usize;
            let row = raw_index / raw_columns;
            let column = raw_index % raw_columns;
            let raw_shared = shared as usize;
            let mut value = 0.0;
            let mut inner = 0;
            while inner < raw_shared {
                value += left[row * raw_shared + inner] * right[inner * raw_columns + column];
                inner += 1;
            }
            *output_value = value;
        }
    }

    #[kernel]
    pub fn csr_diffusion_f32(
        indptr: &[u32],
        indices: &[u32],
        weights: &[f32],
        values: &[f32],
        mut output: DisjointSlice<f32>,
        nodes: u32,
        channels: u32,
    ) {
        let item = thread::index_1d();
        let raw_item = item.get();
        if let Some(output_value) = output.get_mut(item) {
            let raw_channels = channels as usize;
            let raw_nodes = nodes as usize;
            let channel = raw_item % raw_channels;
            let node_batch = raw_item / raw_channels;
            let row = node_batch % raw_nodes;
            let batch = node_batch / raw_nodes;
            let mut sum = 0.0;
            let mut edge = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            while edge < end {
                let source = indices[edge] as usize;
                sum +=
                    weights[edge] * values[(batch * raw_nodes + source) * raw_channels + channel];
                edge += 1;
            }
            *output_value = sum;
        }
    }

    // Compute each input gradient independently instead of using a floating
    // point atomic accumulation. cuda-oxide's safe slice model keeps the
    // output disjoint, and the sparse graphs used by LSTTN are small enough
    // that scanning CSR rows is preferable to a host-side fallback.
    #[kernel]
    pub fn csr_diffusion_input_backward_f32(
        indptr: &[u32],
        indices: &[u32],
        weights: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        nodes: u32,
        channels: u32,
    ) {
        let item = thread::index_1d();
        let raw_item = item.get();
        if let Some(gradient) = input_gradient.get_mut(item) {
            let raw_nodes = nodes as usize;
            let raw_channels = channels as usize;
            let channel = raw_item % raw_channels;
            let node_batch = raw_item / raw_channels;
            let source = node_batch % raw_nodes;
            let batch = node_batch / raw_nodes;
            let mut value = 0.0;
            let mut row = 0;
            while row < raw_nodes {
                let mut edge = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                while edge < end {
                    if indices[edge] as usize == source {
                        value += weights[edge]
                            * output_gradient[(batch * raw_nodes + row) * raw_channels + channel];
                    }
                    edge += 1;
                }
                row += 1;
            }
            *gradient = value;
        }
    }

    #[kernel]
    pub fn csr_diffusion_edge_backward_f32(
        indptr: &[u32],
        indices: &[u32],
        values: &[f32],
        output_gradient: &[f32],
        mut edge_gradient: DisjointSlice<f32>,
        batches: u32,
        nodes: u32,
        channels: u32,
    ) {
        let edge_index = thread::index_1d();
        let edge = edge_index.get();
        if let Some(gradient) = edge_gradient.get_mut(edge_index) {
            let raw_batches = batches as usize;
            let raw_nodes = nodes as usize;
            let raw_channels = channels as usize;
            let source = indices[edge] as usize;
            let mut row = 0;
            while row + 1 < indptr.len() && indptr[row + 1] as usize <= edge {
                row += 1;
            }
            let mut value = 0.0;
            let mut batch = 0;
            while batch < raw_batches {
                let mut channel = 0;
                while channel < raw_channels {
                    value += output_gradient[(batch * raw_nodes + row) * raw_channels + channel]
                        * values[(batch * raw_nodes + source) * raw_channels + channel];
                    channel += 1;
                }
                batch += 1;
            }
            *gradient = value;
        }
    }

    #[kernel]
    pub fn layer_norm_f32(
        values: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut output: DisjointSlice<f32>,
        width: u32,
    ) {
        let index = thread::index_1d();
        let raw_index = index.get();
        let raw_width = width as usize;
        if let Some(output_value) = output.get_mut(index) {
            let row = raw_index / raw_width;
            let col = raw_index % raw_width;
            let offset = row * raw_width;
            let mut mean = 0.0;
            let mut feature = 0;
            while feature < raw_width {
                mean += values[offset + feature];
                feature += 1;
            }
            mean /= raw_width as f32;
            let mut variance = 0.0;
            feature = 0;
            while feature < raw_width {
                let delta = values[offset + feature] - mean;
                variance += delta * delta;
                feature += 1;
            }
            let inverse_std = 1.0 / libm::sqrtf(variance / raw_width as f32 + 1.0e-5);
            *output_value = (values[offset + col] - mean) * inverse_std * gamma[col] + beta[col];
        }
    }

    #[kernel]
    pub fn layer_norm_input_backward_f32(
        values: &[f32],
        gamma: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = width as usize;
        if let Some(value) = input_gradient.get_mut(index) {
            let row = item / width;
            let column = item % width;
            let offset = row * width;
            let mut mean = 0.0;
            let mut feature = 0;
            while feature < width {
                mean += values[offset + feature];
                feature += 1;
            }
            mean /= width as f32;
            let mut variance = 0.0;
            feature = 0;
            while feature < width {
                let delta = values[offset + feature] - mean;
                variance += delta * delta;
                feature += 1;
            }
            let inverse_std = 1.0 / libm::sqrtf(variance / width as f32 + 1.0e-5);
            let mut sum_gradient = 0.0;
            let mut sum_gradient_xhat = 0.0;
            feature = 0;
            while feature < width {
                let gradient = output_gradient[offset + feature] * gamma[feature];
                let xhat = (values[offset + feature] - mean) * inverse_std;
                sum_gradient += gradient;
                sum_gradient_xhat += gradient * xhat;
                feature += 1;
            }
            let xhat = (values[offset + column] - mean) * inverse_std;
            *value = inverse_std
                * (output_gradient[offset + column] * gamma[column]
                    - sum_gradient / width as f32
                    - xhat * sum_gradient_xhat / width as f32);
        }
    }

    #[kernel]
    pub fn layer_norm_gamma_backward_f32(
        values: &[f32],
        output_gradient: &[f32],
        mut gamma_gradient: DisjointSlice<f32>,
        rows: u32,
        width: u32,
    ) {
        let index = thread::index_1d();
        let column = index.get();
        let width = width as usize;
        if let Some(value) = gamma_gradient.get_mut(index) {
            let mut sum = 0.0;
            let mut row = 0;
            while row < rows as usize {
                let offset = row * width;
                let mut mean = 0.0;
                let mut feature = 0;
                while feature < width {
                    mean += values[offset + feature];
                    feature += 1;
                }
                mean /= width as f32;
                let mut variance = 0.0;
                feature = 0;
                while feature < width {
                    let delta = values[offset + feature] - mean;
                    variance += delta * delta;
                    feature += 1;
                }
                sum += output_gradient[offset + column] * (values[offset + column] - mean)
                    / libm::sqrtf(variance / width as f32 + 1.0e-5);
                row += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn layer_norm_beta_backward_f32(
        output_gradient: &[f32],
        mut beta_gradient: DisjointSlice<f32>,
        rows: u32,
        width: u32,
    ) {
        let index = thread::index_1d();
        let column = index.get();
        if let Some(value) = beta_gradient.get_mut(index) {
            let mut sum = 0.0;
            let mut row = 0;
            while row < rows as usize {
                sum += output_gradient[row * width as usize + column];
                row += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn layer_norm_parameter_slice_f32(
        values: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        gamma_offset: u32,
        beta_offset: u32,
        width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = width as usize;
        if let Some(output_value) = output.get_mut(index) {
            let row = item / width;
            let column = item % width;
            let offset = row * width;
            let mut mean = 0.0;
            let mut feature = 0;
            while feature < width {
                mean += values[offset + feature];
                feature += 1;
            }
            mean /= width as f32;
            let mut variance = 0.0;
            feature = 0;
            while feature < width {
                let delta = values[offset + feature] - mean;
                variance += delta * delta;
                feature += 1;
            }
            let inverse_std = 1.0 / libm::sqrtf(variance / width as f32 + 1.0e-5);
            *output_value = (values[offset + column] - mean)
                * inverse_std
                * parameters[gamma_offset as usize + column]
                + parameters[beta_offset as usize + column];
        }
    }

    #[kernel]
    pub fn layer_norm_parameter_slice_input_backward_f32(
        values: &[f32],
        parameters: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        gamma_offset: u32,
        width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = width as usize;
        if let Some(gradient) = input_gradient.get_mut(index) {
            let row = item / width;
            let column = item % width;
            let offset = row * width;
            let mut mean = 0.0;
            let mut feature = 0;
            while feature < width {
                mean += values[offset + feature];
                feature += 1;
            }
            mean /= width as f32;
            let mut variance = 0.0;
            feature = 0;
            while feature < width {
                let delta = values[offset + feature] - mean;
                variance += delta * delta;
                feature += 1;
            }
            let inverse_std = 1.0 / libm::sqrtf(variance / width as f32 + 1.0e-5);
            let mut sum_gradient = 0.0;
            let mut sum_gradient_xhat = 0.0;
            feature = 0;
            while feature < width {
                let normalized_gradient =
                    output_gradient[offset + feature] * parameters[gamma_offset as usize + feature];
                sum_gradient += normalized_gradient;
                sum_gradient_xhat +=
                    normalized_gradient * (values[offset + feature] - mean) * inverse_std;
                feature += 1;
            }
            let normalized_gradient =
                output_gradient[item] * parameters[gamma_offset as usize + column];
            let xhat = (values[item] - mean) * inverse_std;
            *gradient = inverse_std
                * (normalized_gradient
                    - sum_gradient / width as f32
                    - xhat * sum_gradient_xhat / width as f32);
        }
    }

    #[kernel]
    pub fn layer_norm_parameter_slice_parameter_backward_f32(
        values: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        gamma_offset: u32,
        beta_offset: u32,
        rows: u32,
        width: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = width as usize;
        let gamma_start = gamma_offset as usize;
        let beta_start = beta_offset as usize;
        if item >= gamma_start && item < gamma_start + width {
            if let Some(gradient) = parameter_gradient.get_mut(index) {
                let column = item - gamma_start;
                let mut sum = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    let offset = row * width;
                    let mut mean = 0.0;
                    let mut feature = 0;
                    while feature < width {
                        mean += values[offset + feature];
                        feature += 1;
                    }
                    mean /= width as f32;
                    let mut variance = 0.0;
                    feature = 0;
                    while feature < width {
                        let delta = values[offset + feature] - mean;
                        variance += delta * delta;
                        feature += 1;
                    }
                    let inverse_std = 1.0 / libm::sqrtf(variance / width as f32 + 1.0e-5);
                    sum += output_gradient[offset + column]
                        * (values[offset + column] - mean)
                        * inverse_std;
                    row += 1;
                }
                *gradient += sum;
            }
        } else if item >= beta_start && item < beta_start + width {
            if let Some(gradient) = parameter_gradient.get_mut(index) {
                let column = item - beta_start;
                let mut sum = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    sum += output_gradient[row * width + column];
                    row += 1;
                }
                *gradient += sum;
            }
        }
    }

    #[kernel]
    pub fn batch_norm_channel_stats_f32(
        values: &[f32],
        mut statistics: DisjointSlice<f32>,
        rows: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let channel = index.get();
        if channel < channels as usize {
            if let Some(mean_value) = statistics.get_mut(index) {
                let mut mean = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    mean += values[row * channels as usize + channel];
                    row += 1;
                }
                mean /= rows as f32;
                let mut variance = 0.0;
                row = 0;
                while row < rows as usize {
                    let delta = values[row * channels as usize + channel] - mean;
                    variance += delta * delta;
                    row += 1;
                }
                *mean_value = mean;
                unsafe {
                    *statistics.get_unchecked_mut(channels as usize + channel) =
                        1.0 / libm::sqrtf(variance / rows as f32 + 1.0e-5);
                }
            }
        }
    }

    #[kernel]
    pub fn batch_norm_channel_apply_f32(
        values: &[f32],
        parameters: &[f32],
        statistics: &[f32],
        mut output: DisjointSlice<f32>,
        gamma_offset: u32,
        beta_offset: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let channel = item % channels as usize;
            *value = (values[item] - statistics[channel])
                * statistics[channels as usize + channel]
                * parameters[gamma_offset as usize + channel]
                + parameters[beta_offset as usize + channel];
        }
    }

    #[kernel]
    pub fn batch_norm_channel_input_backward_f32(
        values: &[f32],
        parameters: &[f32],
        statistics: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        gamma_offset: u32,
        rows: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            let channel = item % channels as usize;
            let mean = statistics[channel];
            let inverse = statistics[channels as usize + channel];
            let mut sum_gradient = 0.0;
            let mut sum_gradient_xhat = 0.0;
            let mut row = 0;
            while row < rows as usize {
                let offset = row * channels as usize + channel;
                let normalized =
                    output_gradient[offset] * parameters[gamma_offset as usize + channel];
                sum_gradient += normalized;
                sum_gradient_xhat += normalized * (values[offset] - mean) * inverse;
                row += 1;
            }
            let normalized = output_gradient[item] * parameters[gamma_offset as usize + channel];
            let xhat = (values[item] - mean) * inverse;
            *value = inverse
                * (normalized
                    - sum_gradient / rows as f32
                    - xhat * sum_gradient_xhat / rows as f32);
        }
    }

    #[kernel]
    pub fn batch_norm_channel_parameter_backward_f32(
        values: &[f32],
        statistics: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        gamma_offset: u32,
        beta_offset: u32,
        rows: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let gamma_start = gamma_offset as usize;
        let beta_start = beta_offset as usize;
        if item >= gamma_start && item < gamma_start + channels as usize {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let channel = item - gamma_start;
                let mut sum = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    let offset = row * channels as usize + channel;
                    sum += output_gradient[offset]
                        * (values[offset] - statistics[channel])
                        * statistics[channels as usize + channel];
                    row += 1;
                }
                *value += sum;
            }
        } else if item >= beta_start && item < beta_start + channels as usize {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let channel = item - beta_start;
                let mut sum = 0.0;
                let mut row = 0;
                while row < rows as usize {
                    sum += output_gradient[row * channels as usize + channel];
                    row += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn add_f32(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            *output_value = left[raw_index] + right[raw_index];
        }
    }

    #[kernel]
    pub fn add_in_place_f32(other: &[f32], mut target: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(target_value) = target.get_mut(index) {
            *target_value += other[raw_index];
        }
    }

    #[kernel]
    pub fn double_in_place_f32(mut target: DisjointSlice<f32>) {
        let index = thread::index_1d();
        if let Some(target_value) = target.get_mut(index) {
            *target_value += *target_value;
        }
    }

    #[kernel]
    pub fn scalar_graph_f32(
        initial_values: &[f32],
        opcodes: &[u8],
        left: &[u32],
        right: &[u32],
        mut values: DisjointSlice<f32>,
        len: u32,
    ) {
        let index = thread::index_1d();
        if index.get() != 0 {
            return;
        }
        let values_ptr = values.as_mut_ptr();
        let mut item = 0usize;
        while item < len as usize {
            unsafe { *values_ptr.add(item) = initial_values[item] };
            item += 1;
        }
        item = 0;
        while item < len as usize {
            let opcode = opcodes[item];
            if opcode >= 2 {
                let a = unsafe { *values_ptr.add(left[item] as usize) };
                let b = unsafe { *values_ptr.add(right[item] as usize) };
                let value = match opcode {
                    2 => a + b,
                    3 => a * b,
                    4 => a / if b > 1.0e-12 { b } else { 1.0e-12 },
                    5 => libm::tanhf(a),
                    6 => libm::expf(a),
                    7 => libm::sqrtf(if a > 1.0e-12 { a } else { 1.0e-12 }),
                    8 => libm::sinf(a),
                    9 => 1.0 / (1.0 + libm::expf(-a)),
                    10 => {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    }
                    11 => a,
                    _ => 0.0,
                };
                unsafe { *values_ptr.add(item) = value };
            }
            item += 1;
        }
    }

    #[kernel]
    pub fn scalar_graph_train_f32(
        initial: &[f32],
        opcodes: &[u8],
        left: &[u32],
        right: &[u32],
        parameter_ids: &[u32],
        mut parameters: DisjointSlice<f32>,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
        mut values: DisjointSlice<f32>,
        mut gradients: DisjointSlice<f32>,
        mut parameter_gradients: DisjointSlice<f32>,
        len: u32,
        loss: u32,
        parameter_len: u32,
        step: u32,
        learning_rate: f32,
        weight_decay: f32,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let (vp, gp, pgp, pp, mp, sp) = (
            values.as_mut_ptr(),
            gradients.as_mut_ptr(),
            parameter_gradients.as_mut_ptr(),
            parameters.as_mut_ptr(),
            first.as_mut_ptr(),
            second.as_mut_ptr(),
        );
        let mut i = 0usize;
        while i < len as usize {
            unsafe { *vp.add(i) = initial[i] };
            let op = opcodes[i];
            if op == 1 {
                unsafe { *vp.add(i) = *pp.add(parameter_ids[i] as usize) }
            } else if op > 1 {
                let a = unsafe { *vp.add(left[i] as usize) };
                let b = unsafe { *vp.add(right[i] as usize) };
                let value = match op {
                    2 => a + b,
                    3 => a * b,
                    4 => a / if b > 1e-12 { b } else { 1e-12 },
                    5 => libm::tanhf(a),
                    6 => libm::expf(a),
                    7 => libm::sqrtf(if a > 1e-12 { a } else { 1e-12 }),
                    8 => libm::sinf(a),
                    9 => 1.0 / (1.0 + libm::expf(-a)),
                    10 => {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    }
                    11 => a,
                    _ => 0.0,
                };
                unsafe { *vp.add(i) = value }
            }
            i += 1;
        }
        unsafe { *gp.add(loss as usize) = 1.0 };
        let mut rev = len as usize;
        while rev > 0 {
            rev -= 1;
            let op = opcodes[rev];
            let l = left[rev] as usize;
            let r = right[rev] as usize;
            let g = unsafe { *gp.add(rev) };
            unsafe {
                match op {
                    1 => *pgp.add(parameter_ids[rev] as usize) += g,
                    2 => {
                        *gp.add(l) += g;
                        *gp.add(r) += g
                    }
                    3 => {
                        *gp.add(l) += g * *vp.add(r);
                        *gp.add(r) += g * *vp.add(l)
                    }
                    4 => {
                        let rv = *vp.add(r);
                        let d = if rv > 1e-12 { rv } else { 1e-12 };
                        *gp.add(l) += g / d;
                        *gp.add(r) -= g * *vp.add(l) / (d * d)
                    }
                    5 => *gp.add(l) += g * (1.0 - *vp.add(rev) * *vp.add(rev)),
                    6 => *gp.add(l) += g * *vp.add(rev),
                    7 => {
                        let value = *vp.add(rev);
                        *gp.add(l) += g / (2.0 * if value > 1e-12 { value } else { 1e-12 })
                    }
                    8 => *gp.add(l) += g * libm::cosf(*vp.add(l)),
                    9 => *gp.add(l) += g * *vp.add(rev) * (1.0 - *vp.add(rev)),
                    10 => {
                        if *vp.add(l) >= *vp.add(r) {
                            *gp.add(l) += g
                        } else {
                            *gp.add(r) += g
                        }
                    }
                    11 => *gp.add(l) += g,
                    _ => {}
                }
            }
        }
        let c1 = 1.0 - libm::powf(0.9, step as f32);
        let c2 = 1.0 - libm::powf(0.999, step as f32);
        i = 0;
        while i < parameter_len as usize {
            unsafe {
                let g = *pgp.add(i) + weight_decay * *pp.add(i);
                *mp.add(i) = 0.9 * *mp.add(i) + 0.1 * g;
                *sp.add(i) = 0.999 * *sp.add(i) + 0.001 * g * g;
                *pp.add(i) -=
                    learning_rate * (*mp.add(i) / c1) / (libm::sqrtf(*sp.add(i) / c2) + 1e-8)
            }
            i += 1;
        }
    }

    #[kernel]
    pub fn train_tanh_mlp_f32(
        inputs: &[f32],
        targets: &[f32],
        mut parameters: DisjointSlice<f32>,
        rows: u32,
        input_size: u32,
        hidden_size: u32,
        epochs: u32,
        learning_rate: f32,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let p = parameters.as_mut_ptr();
        let input_size = input_size as usize;
        let hidden_size = hidden_size as usize;
        let b1 = hidden_size * input_size;
        let w2 = b1 + hidden_size;
        let b2 = w2 + hidden_size;
        let mut epoch = 0;
        while epoch < epochs as usize {
            let mut row = 0;
            while row < rows as usize {
                let mut prediction = unsafe { *p.add(b2) };
                let mut hidden = 0;
                while hidden < hidden_size {
                    let mut value = unsafe { *p.add(b1 + hidden) };
                    let mut input = 0;
                    while input < input_size {
                        value += unsafe { *p.add(hidden * input_size + input) }
                            * inputs[row * input_size + input];
                        input += 1;
                    }
                    prediction += libm::tanhf(value) * unsafe { *p.add(w2 + hidden) };
                    hidden += 1;
                }
                let error_gradient = 2.0 * (prediction - targets[row]);
                unsafe {
                    *p.add(b2) -= learning_rate * error_gradient;
                }
                hidden = 0;
                while hidden < hidden_size {
                    let mut value = unsafe { *p.add(b1 + hidden) };
                    let mut input = 0;
                    while input < input_size {
                        value += unsafe { *p.add(hidden * input_size + input) }
                            * inputs[row * input_size + input];
                        input += 1;
                    }
                    let activation = libm::tanhf(value);
                    let old_w2 = unsafe { *p.add(w2 + hidden) };
                    unsafe {
                        *p.add(w2 + hidden) -= learning_rate * error_gradient * activation;
                    }
                    let gradient = error_gradient * old_w2 * (1.0 - activation * activation);
                    unsafe {
                        *p.add(b1 + hidden) -= learning_rate * gradient;
                    }
                    input = 0;
                    while input < input_size {
                        unsafe {
                            *p.add(hidden * input_size + input) -=
                                learning_rate * gradient * inputs[row * input_size + input];
                        }
                        input += 1;
                    }
                    hidden += 1;
                }
                row += 1;
            }
            epoch += 1;
        }
    }

    #[kernel]
    pub fn accumulate_parameter_slice_f32(
        source: &[f32],
        mut destination: DisjointSlice<f32>,
        offset: u32,
        len: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = destination.get_mut(index) {
            if item >= offset as usize && item < offset as usize + len as usize {
                *value += source[item - offset as usize];
            }
        }
    }

    #[kernel]
    pub fn masked_inverse_scale_mae_loss_f32(
        prediction: &[f32],
        target: &[f32],
        mut loss: DisjointSlice<f32>,
        len: u32,
        normalized_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item < 2 {
            let mut total = 0.0;
            let mut count = 0u32;
            let mut item = 0;
            while item < len as usize {
                if libm::fabsf(target[item] - normalized_zero) > 1.0e-12 {
                    total += libm::fabsf((prediction[item] - target[item]) * target_scale);
                    count += 1;
                }
                item += 1;
            }
            if let Some(value) = loss.get_mut(index) {
                *value = if item == 0 {
                    total / if count == 0 { 1.0 } else { count as f32 }
                } else {
                    count as f32
                };
            }
        }
    }

    #[kernel]
    pub fn masked_inverse_scale_mae_gradient_f32(
        prediction: &[f32],
        target: &[f32],
        mut gradient: DisjointSlice<f32>,
        len: u32,
        normalized_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = gradient.get_mut(index) {
            if libm::fabsf(target[item] - normalized_zero) <= 1.0e-12 {
                *value = 0.0;
            } else {
                let residual = prediction[item] - target[item];
                let sign = if residual > 0.0 {
                    1.0
                } else if residual < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let mut count = 0u32;
                let mut target_index = 0;
                while target_index < len as usize {
                    if libm::fabsf(target[target_index] - normalized_zero) > 1.0e-12 {
                        count += 1;
                    }
                    target_index += 1;
                }
                let count = if count == 0 { 1.0 } else { count as f32 };
                *value = sign * target_scale / count;
            }
        }
    }

    #[kernel]
    pub fn weighted_inverse_scale_mae_loss_f32(
        prediction: &[f32],
        target: &[f32],
        weight: &[f32],
        mut loss: DisjointSlice<f32>,
        len: u32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item < 2 {
            let mut total = 0.0;
            let mut weight_total = 0.0;
            let mut position = 0;
            while position < len as usize {
                let value = weight[position];
                if value > 0.0 {
                    total += value
                        * libm::fabsf((prediction[position] - target[position]) * target_scale);
                    weight_total += value;
                }
                position += 1;
            }
            if let Some(value) = loss.get_mut(index) {
                *value = if item == 0 {
                    total
                        / if weight_total == 0.0 {
                            1.0
                        } else {
                            weight_total
                        }
                } else {
                    weight_total
                };
            }
        }
    }

    #[kernel]
    pub fn weighted_inverse_scale_mae_gradient_f32(
        prediction: &[f32],
        target: &[f32],
        weight: &[f32],
        loss: &[f32],
        mut gradient: DisjointSlice<f32>,
        len: u32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item < len as usize {
            if let Some(value) = gradient.get_mut(index) {
                let denominator = loss[1];
                let item_weight = weight[item];
                if item_weight <= 0.0 || denominator <= 0.0 {
                    *value = 0.0;
                } else {
                    let residual = prediction[item] - target[item];
                    let sign = if residual > 0.0 {
                        1.0
                    } else if residual < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    *value = sign * target_scale * item_weight / denominator;
                }
            }
        }
    }

    #[kernel]
    pub fn gradient_l2_norm_f32(gradients: &[f32], mut norm: DisjointSlice<f32>, len: u32) {
        let index = thread::index_1d();
        if index.get() == 0 {
            let mut sum = 0.0;
            let mut item = 0;
            while item < len as usize {
                sum += gradients[item] * gradients[item];
                item += 1;
            }
            if let Some(value) = norm.get_mut(index) {
                *value = libm::sqrtf(sum);
            }
        }
    }

    #[kernel]
    pub fn clip_gradient_l2_f32(norm: &[f32], mut gradients: DisjointSlice<f32>, maximum: f32) {
        let index = thread::index_1d();
        if let Some(value) = gradients.get_mut(index) {
            if norm[0] > maximum {
                *value *= maximum / norm[0];
            }
        }
    }

    #[kernel]
    pub fn scale_f32(values: &[f32], mut output: DisjointSlice<f32>, scale: f32) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            *output_value = values[raw_index] * scale;
        }
    }

    #[kernel]
    pub fn relu_f32(values: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let value = values[raw_index];
            *output_value = if value > 0.0 { value } else { 0.0 };
        }
    }

    #[kernel]
    pub fn relu_in_place_f32(mut values: DisjointSlice<f32>) {
        let index = thread::index_1d();
        if let Some(value) = values.get_mut(index) {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
    }

    #[kernel]
    pub fn gelu_f32(values: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let value = values[raw_index];
            let cubic = value * value * value;
            let inner = 0.797_884_6 * (value + 0.044_715 * cubic);
            *output_value = 0.5 * value * (1.0 + libm::tanhf(inner));
        }
    }

    #[kernel]
    pub fn gated_tanh_sigmoid_f32(filter: &[f32], gate: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + libm::expf(-gate[raw_index]));
            *output_value = libm::tanhf(filter[raw_index]) * sigmoid;
        }
    }

    #[kernel]
    pub fn gated_tanh_sigmoid_filter_backward_f32(
        filter: &[f32],
        gate: &[f32],
        output_gradient: &[f32],
        mut filter_gradient: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = filter_gradient.get_mut(index) {
            let tanh = libm::tanhf(filter[item]);
            let sigmoid = 1.0 / (1.0 + libm::expf(-gate[item]));
            *gradient = output_gradient[item] * (1.0 - tanh * tanh) * sigmoid;
        }
    }

    #[kernel]
    pub fn gated_tanh_sigmoid_gate_backward_f32(
        filter: &[f32],
        gate: &[f32],
        output_gradient: &[f32],
        mut gate_gradient: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = gate_gradient.get_mut(index) {
            let sigmoid = 1.0 / (1.0 + libm::expf(-gate[item]));
            *gradient =
                output_gradient[item] * libm::tanhf(filter[item]) * sigmoid * (1.0 - sigmoid);
        }
    }

    #[kernel]
    pub fn relu_backward_f32(
        activations: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(input_value) = input_gradient.get_mut(index) {
            *input_value = if activations[raw_index] > 0.0 {
                output_gradient[raw_index]
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn causal_conv2_f32(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        mut output: DisjointSlice<f32>,
        output_times: u32,
        nodes: u32,
        input_channels: u32,
        output_channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let out_channels = output_channels as usize;
            let in_channels = input_channels as usize;
            let raw_nodes = nodes as usize;
            let raw_output_times = output_times as usize;
            let out = item % out_channels;
            let q = item / out_channels;
            let node = q % raw_nodes;
            let time = (q / raw_nodes) % raw_output_times;
            let batch = q / (raw_nodes * raw_output_times);
            let mut sum = bias[out];
            let mut tap = 0;
            while tap < 2 {
                let mut channel = 0;
                while channel < in_channels {
                    let source_time = time + tap * dilation as usize;
                    let source = ((batch * (raw_output_times + dilation as usize) + source_time)
                        * raw_nodes
                        + node)
                        * in_channels
                        + channel;
                    sum +=
                        input[source] * weights[(tap * in_channels + channel) * out_channels + out];
                    channel += 1;
                }
                tap += 1;
            }
            *output_value = sum;
        }
    }

    #[kernel]
    pub fn causal_conv2_parameter_slice_f32(
        input: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        output_times: u32,
        nodes: u32,
        input_channels: u32,
        output_channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let output_channels = output_channels as usize;
            let input_channels = input_channels as usize;
            let nodes = nodes as usize;
            let output_times = output_times as usize;
            let output_channel = item % output_channels;
            let q = item / output_channels;
            let node = q % nodes;
            let time = (q / nodes) % output_times;
            let batch = q / (nodes * output_times);
            let mut sum = parameters[bias_offset as usize + output_channel];
            let mut tap = 0;
            while tap < 2 {
                let mut input_channel = 0;
                while input_channel < input_channels {
                    let source_time = time + tap * dilation as usize;
                    let source =
                        ((batch * (output_times + dilation as usize) + source_time) * nodes + node)
                            * input_channels
                            + input_channel;
                    sum += input[source]
                        * parameters[weights_offset as usize
                            + (tap * input_channels + input_channel) * output_channels
                            + output_channel];
                    input_channel += 1;
                }
                tap += 1;
            }
            *output_value = sum;
        }
    }

    #[kernel]
    pub fn causal_conv2_parameter_slice_input_backward_f32(
        parameters: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        times: u32,
        nodes: u32,
        input_channels: u32,
        output_channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            let input_channels = input_channels as usize;
            let output_channels = output_channels as usize;
            let nodes = nodes as usize;
            let times = times as usize;
            let input_channel = item % input_channels;
            let q = item / input_channels;
            let node = q % nodes;
            let time = (q / nodes) % times;
            let batch = q / (nodes * times);
            let output_times = times - dilation as usize;
            let mut sum = 0.0;
            let mut tap = 0;
            while tap < 2 {
                if time >= tap * dilation as usize {
                    let output_time = time - tap * dilation as usize;
                    if output_time < output_times {
                        let mut output_channel = 0;
                        while output_channel < output_channels {
                            sum += output_gradient[((batch * output_times + output_time) * nodes
                                + node)
                                * output_channels
                                + output_channel]
                                * parameters[weights_offset as usize
                                    + (tap * input_channels + input_channel) * output_channels
                                    + output_channel];
                            output_channel += 1;
                        }
                    }
                }
                tap += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn causal_conv2_parameter_slice_parameter_backward_f32(
        input: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        batches: u32,
        output_times: u32,
        nodes: u32,
        input_channels: u32,
        output_channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let input_channels = input_channels as usize;
        let output_channels = output_channels as usize;
        let weight_start = weights_offset as usize;
        let weight_count = 2 * input_channels * output_channels;
        let bias_start = bias_offset as usize;
        if item >= weight_start && item < weight_start + weight_count {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let local = item - weight_start;
                let output_channel = local % output_channels;
                let q = local / output_channels;
                let input_channel = q % input_channels;
                let tap = q / input_channels;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut time = 0;
                    while time < output_times as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            sum += input[((batch * (output_times as usize + dilation as usize)
                                + time
                                + tap * dilation as usize)
                                * nodes as usize
                                + node)
                                * input_channels
                                + input_channel]
                                * output_gradient[((batch * output_times as usize + time)
                                    * nodes as usize
                                    + node)
                                    * output_channels
                                    + output_channel];
                            node += 1;
                        }
                        time += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        } else if item >= bias_start && item < bias_start + output_channels {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let output_channel = item - bias_start;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut time = 0;
                    while time < output_times as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            sum += output_gradient[((batch * output_times as usize + time)
                                * nodes as usize
                                + node)
                                * output_channels
                                + output_channel];
                            node += 1;
                        }
                        time += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn causal_conv2_input_backward_f32(
        weights: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        times: u32,
        nodes: u32,
        input_channels: u32,
        output_channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            let in_width = input_channels as usize;
            let out_width = output_channels as usize;
            let nodes = nodes as usize;
            let times = times as usize;
            let input_channel = item % in_width;
            let q = item / in_width;
            let node = q % nodes;
            let time = (q / nodes) % times;
            let batch = q / (nodes * times);
            let output_times = times - dilation as usize;
            let mut sum = 0.0;
            let mut tap = 0;
            while tap < 2 {
                if time >= tap * dilation as usize {
                    let output_time = time - tap * dilation as usize;
                    if output_time < output_times {
                        let mut out = 0;
                        while out < out_width {
                            sum += output_gradient[((batch * output_times + output_time) * nodes
                                + node)
                                * out_width
                                + out]
                                * weights[(tap * in_width + input_channel) * out_width + out];
                            out += 1;
                        }
                    }
                }
                tap += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn causal_conv2_weight_backward_f32(
        input: &[f32],
        output_gradient: &[f32],
        mut weight_gradient: DisjointSlice<f32>,
        batches: u32,
        output_times: u32,
        nodes: u32,
        input_channels: u32,
        output_channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = weight_gradient.get_mut(index) {
            let in_width = input_channels as usize;
            let out_width = output_channels as usize;
            let out = item % out_width;
            let q = item / out_width;
            let input_channel = q % in_width;
            let tap = q / in_width;
            let mut sum = 0.0;
            let mut batch = 0;
            while batch < batches as usize {
                let mut time = 0;
                while time < output_times as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        sum += input[((batch * (output_times as usize + dilation as usize)
                            + time
                            + tap * dilation as usize)
                            * nodes as usize
                            + node)
                            * in_width
                            + input_channel]
                            * output_gradient[((batch * output_times as usize + time)
                                * nodes as usize
                                + node)
                                * out_width
                                + out];
                        node += 1;
                    }
                    time += 1;
                }
                batch += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn causal_conv2_bias_backward_f32(
        output_gradient: &[f32],
        mut bias_gradient: DisjointSlice<f32>,
        batches: u32,
        output_times: u32,
        nodes: u32,
        output_channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = bias_gradient.get_mut(index) {
            let mut sum = 0.0;
            let mut batch = 0;
            while batch < batches as usize {
                let mut time = 0;
                while time < output_times as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        sum += output_gradient[((batch * output_times as usize + time)
                            * nodes as usize
                            + node)
                            * output_channels as usize
                            + item];
                        node += 1;
                    }
                    time += 1;
                }
                batch += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn lsttn_short_input_projection_parameter_slice_f32(
        input: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        lookback: u32,
        nodes: u32,
        input_channels: u32,
        recent_window: u32,
        hidden: u32,
        left_padding: u32,
        phase_offset: u32,
        periodicity: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let padded_times = (recent_window + left_padding) as usize;
            let channel = item % hidden;
            let q = item / hidden;
            let node = q % nodes;
            let local_time = (q / nodes) % padded_times;
            let batch = q / (nodes * padded_times);
            if local_time < left_padding as usize {
                *value = 0.0;
            } else {
                let source_time =
                    lookback as usize - recent_window as usize + local_time - left_padding as usize;
                let signal = input[((batch * lookback as usize + source_time) * nodes + node)
                    * input_channels as usize];
                let time = ((phase_offset as usize + source_time) % periodicity as usize) as f32
                    / periodicity as f32;
                *value = parameters[bias_offset as usize + channel]
                    + signal * parameters[weights_offset as usize + channel]
                    + time * parameters[weights_offset as usize + hidden + channel];
            }
        }
    }

    #[kernel]
    pub fn transpose_node_time_f32(
        values: &[f32],
        mut output: DisjointSlice<f32>,
        _batches: u32,
        times: u32,
        nodes: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let channels = channels as usize;
            let nodes = nodes as usize;
            let times = times as usize;
            let channel = item % channels;
            let q = item / channels;
            let node = q % nodes;
            let q = q / nodes;
            let time = q % times;
            let batch = q / times;
            let source = ((batch * nodes + node) * times + time) * channels + channel;
            *output_value = values[source];
        }
    }

    #[kernel]
    pub fn patches_to_attention_sequences_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        patches: u32,
        nodes: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let patches = patches as usize;
            let channel = item % hidden;
            let q = item / hidden;
            let node = q % nodes;
            let patch = (q / nodes) % patches;
            let batch = q / (nodes * patches);
            *output_value = input[((batch * patches + patch) * nodes + node) * hidden + channel];
        }
    }

    #[kernel]
    pub fn attention_sequences_to_patches_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        patches: u32,
        nodes: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let patches = patches as usize;
            let channel = item % hidden;
            let q = item / hidden;
            let node = q % nodes;
            let patch = (q / nodes) % patches;
            let batch = q / (nodes * patches);
            *output_value = input[((batch * nodes + node) * patches + patch) * hidden + channel];
        }
    }

    #[kernel]
    pub fn select_node_major_time_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        nodes: u32,
        patches: u32,
        channels: u32,
        patch: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let channels = channels as usize;
            let nodes = nodes as usize;
            let channel = item % channels;
            let q = item / channels;
            let node = q % nodes;
            let batch = q / nodes;
            *output_value = input
                [((batch * nodes + node) * patches as usize + patch as usize) * channels + channel];
        }
    }

    #[kernel]
    pub fn gather_patch_tokens_f32(
        input: &[f32],
        indices: &[u32],
        mut output: DisjointSlice<f32>,
        nodes: u32,
        patches: u32,
        selected: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let selected = selected as usize;
            let nodes = nodes as usize;
            let channel = item % hidden;
            let q = item / hidden;
            let selected_patch = q % selected;
            let node = (q / selected) % nodes;
            let batch = q / (selected * nodes);
            let patch = indices[selected_patch] as usize;
            *output_value = if patch < patches as usize {
                input[((batch * nodes + node) * patches as usize + patch) * hidden + channel]
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn gather_patch_tokens_backward_f32(
        selected_gradient: &[f32],
        indices: &[u32],
        mut full_gradient: DisjointSlice<f32>,
        nodes: u32,
        patches: u32,
        selected: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = full_gradient.get_mut(index) {
            let hidden = hidden as usize;
            let patches = patches as usize;
            let nodes = nodes as usize;
            let channel = item % hidden;
            let q = item / hidden;
            let patch = q % patches;
            let node = (q / patches) % nodes;
            let batch = q / (patches * nodes);
            let mut sum = 0.0;
            let mut selected_patch = 0;
            while selected_patch < selected as usize {
                if indices[selected_patch] as usize == patch {
                    sum += selected_gradient[((batch * nodes + node) * selected as usize
                        + selected_patch)
                        * hidden
                        + channel];
                }
                selected_patch += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn patch_positions_input_backward_f32(
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            *value = output_gradient[item] * scale;
        }
    }

    #[kernel]
    pub fn patch_positions_parameter_backward_f32(
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        positions_offset: u32,
        batches: u32,
        patches: u32,
        nodes: u32,
        hidden: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let start = positions_offset as usize;
        let count = patches as usize * hidden as usize;
        if item >= start && item < start + count {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let local = item - start;
                let patch = local / hidden as usize;
                let channel = local % hidden as usize;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        sum += output_gradient[((batch * patches as usize + patch)
                            * nodes as usize
                            + node)
                            * hidden as usize
                            + channel]
                            * scale;
                        node += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn node_major_horizons_to_output_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        nodes: u32,
        horizons: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let horizons = horizons as usize;
            let nodes = nodes as usize;
            let node = item % nodes;
            let q = item / nodes;
            let horizon = q % horizons;
            let batch = q / horizons;
            *output_value = input[(batch * nodes + node) * horizons + horizon];
        }
    }

    #[kernel]
    pub fn output_to_node_major_horizons_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        nodes: u32,
        horizons: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let horizons = horizons as usize;
            let nodes = nodes as usize;
            let horizon = item % horizons;
            let q = item / horizons;
            let node = q % nodes;
            let batch = q / nodes;
            *output_value = input[(batch * horizons + horizon) * nodes + node];
        }
    }

    #[kernel]
    pub fn concat_channels_f32(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        left_channels: u32,
        right_channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let channels = (left_channels + right_channels) as usize;
            let row = item / channels;
            let channel = item % channels;
            *output_value = if row < rows as usize && channel < left_channels as usize {
                left[row * left_channels as usize + channel]
            } else {
                right[row * right_channels as usize + channel - left_channels as usize]
            };
        }
    }

    #[kernel]
    pub fn add_tail_time_f32(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
        left_times: u32,
        right_times: u32,
        nodes: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let channels = channels as usize;
            let nodes = nodes as usize;
            let right_times = right_times as usize;
            let left_times = left_times as usize;
            let channel = item % channels;
            let q = item / channels;
            let node = q % nodes;
            let time = (q / nodes) % right_times;
            let batch = q / (nodes * right_times);
            let left_time = left_times - right_times + time;
            let left_index = ((batch * left_times + left_time) * nodes + node) * channels + channel;
            *output_value = left[left_index] + right[item];
        }
    }

    #[kernel]
    pub fn add_tail_time_left_backward_f32(
        output_gradient: &[f32],
        mut left_gradient: DisjointSlice<f32>,
        left_times: u32,
        right_times: u32,
        nodes: u32,
        channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = left_gradient.get_mut(index) {
            let channels = channels as usize;
            let nodes = nodes as usize;
            let left_times = left_times as usize;
            let right_times = right_times as usize;
            let channel = item % channels;
            let q = item / channels;
            let node = q % nodes;
            let time = (q / nodes) % left_times;
            let batch = q / (nodes * left_times);
            *gradient = if time < left_times - right_times {
                0.0
            } else {
                let tail_time = time - (left_times - right_times);
                output_gradient
                    [((batch * right_times + tail_time) * nodes + node) * channels + channel]
            };
        }
    }

    #[kernel]
    pub fn deterministic_dropout_f32(
        values: &[f32],
        mut output: DisjointSlice<f32>,
        seed_low: u32,
        seed_high: u32,
        base_low: u32,
        base_high: u32,
        keep_probability: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get() as u64;
        if let Some(output_value) = output.get_mut(index) {
            let seed = ((seed_high as u64) << 32) | seed_low as u64;
            let base = ((base_high as u64) << 32) | base_low as u64;
            let mut state = seed ^ base.wrapping_add(item).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let drop_threshold = ((1.0 - keep_probability) * 10_000.0) as u64;
            *output_value = if state % 10_000 >= drop_threshold {
                values[item as usize] / keep_probability
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn patch_embedding_f32(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        mut output: DisjointSlice<f32>,
        patches: u32,
        nodes: u32,
        channels: u32,
        patch_width: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let patches = patches as usize;
            let out = item % hidden;
            let q = item / hidden;
            let node = q % nodes;
            let patch = (q / nodes) % patches;
            let batch = q / (nodes * patches);
            let mut value = bias[out];
            let mut patch_item = 0;
            let width = patch_width as usize * channels as usize;
            while patch_item < width {
                let time = patch * patch_width as usize + patch_item / channels as usize;
                let channel = patch_item % channels as usize;
                let source = ((batch * (patches * patch_width as usize) + time) * nodes + node)
                    * channels as usize
                    + channel;
                value += input[source] * weights[patch_item * hidden + out];
                patch_item += 1;
            }
            *output_value = value;
        }
    }

    #[kernel]
    pub fn patch_embedding_parameter_slice_f32(
        input: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        patches: u32,
        nodes: u32,
        channels: u32,
        patch_width: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let patches = patches as usize;
            let patch_width = patch_width as usize;
            let channels = channels as usize;
            let channel = item % hidden;
            let q = item / hidden;
            let node = q % nodes;
            let patch = (q / nodes) % patches;
            let batch = q / (nodes * patches);
            let mut value = parameters[bias_offset as usize + channel];
            let mut offset = 0;
            while offset < patch_width {
                let time = patch * patch_width + offset;
                value += input
                    [((batch * (patches * patch_width) + time) * nodes + node) * channels]
                    * parameters[weights_offset as usize + offset * hidden + channel];
                offset += 1;
            }
            *output_value = value;
        }
    }

    #[kernel]
    pub fn patch_embedding_parameter_slice_backward_f32(
        input: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        batches: u32,
        patches: u32,
        nodes: u32,
        channels: u32,
        patch_width: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let hidden = hidden as usize;
        let patch_width = patch_width as usize;
        let weight_start = weights_offset as usize;
        let weight_count = patch_width * hidden;
        let bias_start = bias_offset as usize;
        if item >= weight_start && item < weight_start + weight_count {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let local = item - weight_start;
                let offset = local / hidden;
                let output = local % hidden;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut patch = 0;
                    while patch < patches as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            let source = ((batch * patches as usize * patch_width
                                + patch * patch_width
                                + offset)
                                * nodes as usize
                                + node)
                                * channels as usize;
                            sum += input[source]
                                * output_gradient[((batch * patches as usize + patch)
                                    * nodes as usize
                                    + node)
                                    * hidden
                                    + output];
                            node += 1;
                        }
                        patch += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        } else if item >= bias_start && item < bias_start + hidden {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let output = item - bias_start;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut patch = 0;
                    while patch < patches as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            sum += output_gradient[((batch * patches as usize + patch)
                                * nodes as usize
                                + node)
                                * hidden
                                + output];
                            node += 1;
                        }
                        patch += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn add_patch_positions_f32(
        values: &[f32],
        positions: &[f32],
        mut output: DisjointSlice<f32>,
        patches: u32,
        hidden: u32,
        nodes: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let patch = (item / hidden / nodes) % patches as usize;
            let channel = item % hidden;
            *output_value = values[item] + scale * positions[patch * hidden + channel];
        }
    }

    #[kernel]
    pub fn add_patch_positions_parameter_slice_f32(
        values: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        positions_offset: u32,
        patches: u32,
        hidden: u32,
        nodes: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let nodes = nodes as usize;
            let patch = (item / hidden / nodes) % patches as usize;
            let channel = item % hidden;
            *output_value = (values[item]
                + parameters[positions_offset as usize + patch * hidden + channel])
                * scale;
        }
    }

    #[kernel]
    pub fn add_patch_positions_input_backward_f32_unused(
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            *value = output_gradient[item] * scale;
        }
    }

    #[kernel]
    pub fn add_patch_positions_parameter_backward_f32_unused(
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        positions_offset: u32,
        batches: u32,
        patches: u32,
        nodes: u32,
        hidden: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item >= positions_offset as usize
            && item < positions_offset as usize + patches as usize * hidden as usize
        {
            let Some(value) = parameter_gradient.get_mut(index) else {
                return;
            };
            let hidden = hidden as usize;
            let relative = item - positions_offset as usize;
            let patch = relative / hidden;
            let channel = relative % hidden;
            let mut sum = 0.0;
            let mut batch = 0;
            while batch < batches as usize {
                let mut node = 0;
                while node < nodes as usize {
                    sum += output_gradient[((batch * patches as usize + patch) * nodes as usize
                        + node)
                        * hidden
                        + channel];
                    node += 1;
                }
                batch += 1;
            }
            *value += sum * scale;
        }
    }

    #[kernel]
    pub fn gather_patch_tokens_backward_f32_unused(
        selected_gradient: &[f32],
        patch_indices: &[u32],
        mut full_gradient: DisjointSlice<f32>,
        nodes: u32,
        patches: u32,
        selected: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = full_gradient.get_mut(index) {
            let hidden = hidden as usize;
            let patch = item / hidden % patches as usize;
            let q = item / hidden;
            let node = q / patches as usize % nodes as usize;
            let batch = q / (patches as usize * nodes as usize);
            let mut selected_patch = 0;
            let mut sum = 0.0;
            while selected_patch < selected as usize {
                if patch_indices[selected_patch] as usize == patch {
                    sum += selected_gradient[((batch * nodes as usize + node) * selected as usize
                        + selected_patch)
                        * hidden
                        + item % hidden];
                }
                selected_patch += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn patch_embedding_parameter_slice_backward_f32_unused(
        input: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        batches: u32,
        patches: u32,
        nodes: u32,
        channels: u32,
        patch_width: u32,
        hidden: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item >= weights_offset as usize && item < bias_offset as usize + hidden as usize {
            let Some(value) = parameter_gradient.get_mut(index) else {
                return;
            };
            let hidden = hidden as usize;
            let patch_width = patch_width as usize;
            let _weights = patch_width * hidden;
            if item < bias_offset as usize {
                let relative = item - weights_offset as usize;
                let offset = relative / hidden;
                let out = relative % hidden;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut patch = 0;
                    while patch < patches as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            let source = ((batch * patches as usize * patch_width
                                + patch * patch_width
                                + offset)
                                * nodes as usize
                                + node)
                                * channels as usize;
                            let gradient = ((batch * patches as usize + patch) * nodes as usize
                                + node)
                                * hidden
                                + out;
                            sum += input[source] * output_gradient[gradient];
                            node += 1;
                        }
                        patch += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            } else {
                let out = item - bias_offset as usize;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut patch = 0;
                    while patch < patches as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            sum += output_gradient[((batch * patches as usize + patch)
                                * nodes as usize
                                + node)
                                * hidden
                                + out];
                            node += 1;
                        }
                        patch += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn assemble_masked_decoder_tokens_f32(
        visible_tokens: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        mask_token_offset: u32,
        positions_offset: u32,
        nodes: u32,
        visible_count: u32,
        masked_count: u32,
        hidden: u32,
        position_count: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let hidden = hidden as usize;
            let visible_count = visible_count as usize;
            let masked_count = masked_count as usize;
            let total = visible_count + masked_count;
            let channel = item % hidden;
            let q = item / hidden;
            let token = q % total;
            let node = (q / total) % nodes as usize;
            let batch = q / (total * nodes as usize);
            *value = if token < visible_count {
                visible_tokens
                    [((batch * nodes as usize + node) * visible_count + token) * hidden + channel]
                    * scale
            } else {
                let patch =
                    masked_indices[token - visible_count] as usize % position_count as usize;
                (parameters[mask_token_offset as usize + channel]
                    + parameters[positions_offset as usize + patch * hidden + channel])
                    * scale
            };
        }
    }

    #[kernel]
    pub fn assemble_masked_decoder_tokens_visible_backward_f32(
        output_gradient: &[f32],
        mut visible_gradient: DisjointSlice<f32>,
        nodes: u32,
        visible: u32,
        masked: u32,
        hidden: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = visible_gradient.get_mut(index) {
            let hidden = hidden as usize;
            let visible = visible as usize;
            let node = item / hidden / visible % nodes as usize;
            let batch = item / (hidden * visible * nodes as usize);
            let token = item / hidden % visible;
            *value = output_gradient[((batch * nodes as usize + node)
                * (visible + masked as usize)
                + token)
                * hidden
                + item % hidden]
                * scale;
        }
    }

    #[kernel]
    pub fn assemble_masked_decoder_tokens_parameter_backward_f32(
        output_gradient: &[f32],
        masked_indices: &[u32],
        mut parameter_gradient: DisjointSlice<f32>,
        mask_token_offset: u32,
        positions_offset: u32,
        batches: u32,
        nodes: u32,
        visible: u32,
        masked: u32,
        hidden: u32,
        position_count: u32,
        scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = hidden as usize;
        let mask_start = mask_token_offset as usize;
        let position_start = positions_offset as usize;
        if let Some(value) = parameter_gradient.get_mut(index) {
            if item >= mask_start && item < mask_start + width {
                let channel = item - mask_start;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        let mut token = 0;
                        while token < masked as usize {
                            sum += output_gradient[((batch * nodes as usize + node)
                                * (visible as usize + masked as usize)
                                + visible as usize
                                + token)
                                * width
                                + channel];
                            token += 1;
                        }
                        node += 1;
                    }
                    batch += 1;
                }
                *value += sum * scale;
            } else if item >= position_start
                && item < position_start + position_count as usize * width
            {
                let local = item - position_start;
                let patch = local / width;
                let channel = local % width;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        let mut token = 0;
                        while token < masked as usize {
                            if masked_indices[token] as usize % position_count as usize == patch {
                                sum += output_gradient[((batch * nodes as usize + node)
                                    * (visible as usize + masked as usize)
                                    + visible as usize
                                    + token)
                                    * width
                                    + channel];
                            }
                            token += 1;
                        }
                        node += 1;
                    }
                    batch += 1;
                }
                *value += sum * scale;
            }
        }
    }

    #[kernel]
    pub fn masked_patch_reconstruction_loss_f32(
        decoded: &[f32],
        target: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        mut loss: DisjointSlice<f32>,
        decoder_offset: u32,
        decoder_bias_offset: u32,
        batches: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        visible: u32,
        masked: u32,
        patch_width: u32,
        hidden: u32,
        masked_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item > 1 {
            return;
        }
        let mut count = 0.0;
        let mut total = 0.0;
        let mut batch = 0;
        while batch < batches as usize {
            let mut node = 0;
            while node < nodes as usize {
                let mut mask = 0;
                while mask < masked as usize {
                    let patch = masked_indices[mask] as usize;
                    let base = ((batch * nodes as usize + node)
                        * (visible as usize + masked as usize)
                        + visible as usize
                        + mask)
                        * hidden as usize;
                    let mut offset = 0;
                    while offset < patch_width as usize {
                        let time = patch * patch_width as usize + offset;
                        if time < times as usize {
                            let observed =
                                target[((batch * times as usize + time) * nodes as usize + node)
                                    * channels as usize];
                            if libm::fabsf(observed - masked_zero) > 1.0e-12 {
                                let mut prediction =
                                    parameters[decoder_bias_offset as usize + offset];
                                let mut feature = 0;
                                while feature < hidden as usize {
                                    prediction += decoded[base + feature]
                                        * parameters[decoder_offset as usize
                                            + offset * hidden as usize
                                            + feature];
                                    feature += 1;
                                }
                                let residual = (prediction - observed) * target_scale;
                                total += libm::sqrtf(residual * residual + 1.0e-12);
                                count += 1.0;
                            }
                        }
                        offset += 1;
                    }
                    mask += 1;
                }
                node += 1;
            }
            batch += 1;
        }
        if let Some(value) = loss.get_mut(index) {
            *value = if item == 0 {
                total / if count > 0.0 { count } else { 1.0 }
            } else {
                count
            };
        }
    }

    #[kernel]
    pub fn masked_patch_reconstruction_context_backward_f32(
        decoded: &[f32],
        target: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        mut context_gradient: DisjointSlice<f32>,
        decoder_offset: u32,
        decoder_bias_offset: u32,
        batches: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        visible: u32,
        masked: u32,
        patch_width: u32,
        hidden: u32,
        masked_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = context_gradient.get_mut(index) {
            let width = hidden as usize;
            let total_tokens = visible as usize + masked as usize;
            let channel = item % width;
            let q = item / width;
            let token = q % total_tokens;
            let node = q / total_tokens % nodes as usize;
            let batch = q / (total_tokens * nodes as usize);
            if token < visible as usize {
                *value = 0.0;
                return;
            }
            let mut count = 0.0;
            let mut count_batch = 0;
            while count_batch < batches as usize {
                let mut count_node = 0;
                while count_node < nodes as usize {
                    let mut count_mask = 0;
                    while count_mask < masked as usize {
                        let patch = masked_indices[count_mask] as usize;
                        let mut offset = 0;
                        while offset < patch_width as usize {
                            let time = patch * patch_width as usize + offset;
                            if time < times as usize
                                && libm::fabsf(
                                    target[((count_batch * times as usize + time)
                                        * nodes as usize
                                        + count_node)
                                        * channels as usize]
                                        - masked_zero,
                                ) > 1.0e-12
                            {
                                count += 1.0;
                            }
                            offset += 1;
                        }
                        count_mask += 1;
                    }
                    count_node += 1;
                }
                count_batch += 1;
            }
            let count = if count > 0.0 { count } else { 1.0 };
            let mask = token - visible as usize;
            let patch = masked_indices[mask] as usize;
            let base = ((batch * nodes as usize + node) * total_tokens + token) * width;
            let mut sum = 0.0;
            let mut offset = 0;
            while offset < patch_width as usize {
                let time = patch * patch_width as usize + offset;
                if time < times as usize {
                    let observed = target[((batch * times as usize + time) * nodes as usize
                        + node)
                        * channels as usize];
                    if libm::fabsf(observed - masked_zero) > 1.0e-12 {
                        let mut prediction = parameters[decoder_bias_offset as usize + offset];
                        let mut feature = 0;
                        while feature < width {
                            prediction += decoded[base + feature]
                                * parameters[decoder_offset as usize + offset * width + feature];
                            feature += 1;
                        }
                        let residual = (prediction - observed) * target_scale;
                        sum += residual / libm::sqrtf(residual * residual + 1.0e-12) * target_scale
                            / count
                            * parameters[decoder_offset as usize + offset * width + channel];
                    }
                }
                offset += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn masked_patch_reconstruction_parameter_backward_f32(
        decoded: &[f32],
        target: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        decoder_offset: u32,
        decoder_bias_offset: u32,
        batches: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        visible: u32,
        masked: u32,
        patch_width: u32,
        hidden: u32,
        masked_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = hidden as usize;
        let weight_start = decoder_offset as usize;
        let _weight_count = patch_width as usize * width;
        let bias_start = decoder_bias_offset as usize;
        if item < weight_start || item >= bias_start + patch_width as usize {
            return;
        }
        if let Some(value) = parameter_gradient.get_mut(index) {
            let mut count = 0.0;
            let mut b = 0;
            while b < batches as usize {
                let mut n = 0;
                while n < nodes as usize {
                    let mut m = 0;
                    while m < masked as usize {
                        let patch = masked_indices[m] as usize;
                        let mut offset = 0;
                        while offset < patch_width as usize {
                            let time = patch * patch_width as usize + offset;
                            if time < times as usize
                                && libm::fabsf(
                                    target[((b * times as usize + time) * nodes as usize + n)
                                        * channels as usize]
                                        - masked_zero,
                                ) > 1.0e-12
                            {
                                count += 1.0;
                            }
                            offset += 1;
                        }
                        m += 1;
                    }
                    n += 1;
                }
                b += 1;
            }
            let count = if count > 0.0 { count } else { 1.0 };
            let mut sum = 0.0;
            if item < bias_start {
                let local = item - weight_start;
                let offset = local / width;
                let feature = local % width;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        let mut mask = 0;
                        while mask < masked as usize {
                            let patch = masked_indices[mask] as usize;
                            let time = patch * patch_width as usize + offset;
                            if time < times as usize {
                                let observed = target[((batch * times as usize + time)
                                    * nodes as usize
                                    + node)
                                    * channels as usize];
                                if libm::fabsf(observed - masked_zero) > 1.0e-12 {
                                    let base = ((batch * nodes as usize + node)
                                        * (visible as usize + masked as usize)
                                        + visible as usize
                                        + mask)
                                        * width;
                                    let mut prediction = parameters[bias_start + offset];
                                    let mut f = 0;
                                    while f < width {
                                        prediction += decoded[base + f]
                                            * parameters[weight_start + offset * width + f];
                                        f += 1;
                                    }
                                    let residual = (prediction - observed) * target_scale;
                                    sum += decoded[base + feature] * residual
                                        / libm::sqrtf(residual * residual + 1.0e-12)
                                        * target_scale
                                        / count;
                                }
                            }
                            mask += 1;
                        }
                        node += 1;
                    }
                    batch += 1;
                }
            } else {
                let offset = item - bias_start;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut node = 0;
                    while node < nodes as usize {
                        let mut mask = 0;
                        while mask < masked as usize {
                            let patch = masked_indices[mask] as usize;
                            let time = patch * patch_width as usize + offset;
                            if time < times as usize {
                                let observed = target[((batch * times as usize + time)
                                    * nodes as usize
                                    + node)
                                    * channels as usize];
                                if libm::fabsf(observed - masked_zero) > 1.0e-12 {
                                    let base = ((batch * nodes as usize + node)
                                        * (visible as usize + masked as usize)
                                        + visible as usize
                                        + mask)
                                        * width;
                                    let mut prediction = parameters[bias_start + offset];
                                    let mut f = 0;
                                    while f < width {
                                        prediction += decoded[base + f]
                                            * parameters[weight_start + offset * width + f];
                                        f += 1;
                                    }
                                    let residual = (prediction - observed) * target_scale;
                                    sum += residual / libm::sqrtf(residual * residual + 1.0e-12)
                                        * target_scale
                                        / count;
                                }
                            }
                            mask += 1;
                        }
                        node += 1;
                    }
                    batch += 1;
                }
            }
            *value += sum;
        }
    }

    #[kernel]
    pub fn attention_f32(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: u32,
        heads: u32,
        width: u32,
        causal: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            let width = width as usize;
            let heads = heads as usize;
            let tokens = tokens as usize;
            let dimension = item % width;
            let head = (item / width) % heads;
            let query_token = (item / (width * heads)) % tokens;
            let sequence = item / (width * heads * tokens);
            let base = (sequence * tokens + query_token) * heads * width + head * width;
            let limit = if causal != 0 { query_token + 1 } else { tokens };
            let scale = 1.0 / libm::sqrtf(width as f32);
            let mut maximum = -3.402_823_5e38;
            let mut key_token = 0;
            while key_token < limit {
                let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                let mut score = 0.0;
                let mut feature = 0;
                while feature < width {
                    score += query[base + feature] * key[key_base + feature];
                    feature += 1;
                }
                let scaled = score * scale;
                if scaled > maximum {
                    maximum = scaled;
                }
                key_token += 1;
            }
            let mut normalizer = 0.0;
            key_token = 0;
            while key_token < limit {
                let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                let mut score = 0.0;
                let mut feature = 0;
                while feature < width {
                    score += query[base + feature] * key[key_base + feature];
                    feature += 1;
                }
                normalizer += libm::expf(score * scale - maximum);
                key_token += 1;
            }
            let mut sum = 0.0;
            key_token = 0;
            while key_token < limit {
                let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                let mut score = 0.0;
                let mut feature = 0;
                while feature < width {
                    score += query[base + feature] * key[key_base + feature];
                    feature += 1;
                }
                sum +=
                    libm::expf(score * scale - maximum) / normalizer * value[key_base + dimension];
                key_token += 1;
            }
            *output_value = sum;
        }
    }

    #[kernel]
    pub fn attention_query_backward_f32(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output_gradient: &[f32],
        mut query_gradient: DisjointSlice<f32>,
        tokens: u32,
        heads: u32,
        width: u32,
        causal: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = query_gradient.get_mut(index) {
            let width = width as usize;
            let heads = heads as usize;
            let tokens = tokens as usize;
            let dimension = item % width;
            let head = (item / width) % heads;
            let query_token = (item / (width * heads)) % tokens;
            let sequence = item / (width * heads * tokens);
            let query_base = (sequence * tokens + query_token) * heads * width + head * width;
            let limit = if causal != 0 { query_token + 1 } else { tokens };
            let scale = 1.0 / libm::sqrtf(width as f32);
            let mut maximum = -3.402_823_5e38;
            let mut key_token = 0;
            while key_token < limit {
                let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                let mut score = 0.0;
                let mut d = 0;
                while d < width {
                    score += query[query_base + d] * key[key_base + d];
                    d += 1;
                }
                if score * scale > maximum {
                    maximum = score * scale;
                }
                key_token += 1;
            }
            let mut normalizer = 0.0;
            let mut weighted_dot = 0.0;
            key_token = 0;
            while key_token < limit {
                let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                let mut score = 0.0;
                let mut local = 0.0;
                let mut d = 0;
                while d < width {
                    score += query[query_base + d] * key[key_base + d];
                    local += output_gradient[query_base + d] * value[key_base + d];
                    d += 1;
                }
                let probability = libm::expf(score * scale - maximum);
                normalizer += probability;
                weighted_dot += probability * local;
                key_token += 1;
            }
            weighted_dot /= normalizer;
            let mut sum = 0.0;
            key_token = 0;
            while key_token < limit {
                let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                let mut score = 0.0;
                let mut local = 0.0;
                let mut d = 0;
                while d < width {
                    score += query[query_base + d] * key[key_base + d];
                    local += output_gradient[query_base + d] * value[key_base + d];
                    d += 1;
                }
                let probability = libm::expf(score * scale - maximum) / normalizer;
                sum += probability * (local - weighted_dot) * key[key_base + dimension] * scale;
                key_token += 1;
            }
            *gradient = sum;
        }
    }

    #[kernel]
    pub fn attention_key_backward_f32(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output_gradient: &[f32],
        mut key_gradient: DisjointSlice<f32>,
        tokens: u32,
        heads: u32,
        width: u32,
        causal: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = key_gradient.get_mut(index) {
            let width = width as usize;
            let heads = heads as usize;
            let tokens = tokens as usize;
            let dimension = item % width;
            let head = (item / width) % heads;
            let key_token = (item / (width * heads)) % tokens;
            let sequence = item / (width * heads * tokens);
            let scale = 1.0 / libm::sqrtf(width as f32);
            let mut sum = 0.0;
            let mut query_token = 0;
            while query_token < tokens {
                if causal == 0 || key_token <= query_token {
                    let query_base =
                        (sequence * tokens + query_token) * heads * width + head * width;
                    let limit = if causal != 0 { query_token + 1 } else { tokens };
                    let mut maximum = -3.402_823_5e38;
                    let mut token = 0;
                    while token < limit {
                        let base = (sequence * tokens + token) * heads * width + head * width;
                        let mut score = 0.0;
                        let mut d = 0;
                        while d < width {
                            score += query[query_base + d] * key[base + d];
                            d += 1;
                        }
                        if score * scale > maximum {
                            maximum = score * scale;
                        }
                        token += 1;
                    }
                    let mut normalizer = 0.0;
                    let mut weighted_dot = 0.0;
                    token = 0;
                    while token < limit {
                        let base = (sequence * tokens + token) * heads * width + head * width;
                        let mut score = 0.0;
                        let mut local = 0.0;
                        let mut d = 0;
                        while d < width {
                            score += query[query_base + d] * key[base + d];
                            local += output_gradient[query_base + d] * value[base + d];
                            d += 1;
                        }
                        let probability = libm::expf(score * scale - maximum);
                        normalizer += probability;
                        weighted_dot += probability * local;
                        token += 1;
                    }
                    let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                    let mut score = 0.0;
                    let mut local = 0.0;
                    let mut d = 0;
                    while d < width {
                        score += query[query_base + d] * key[key_base + d];
                        local += output_gradient[query_base + d] * value[key_base + d];
                        d += 1;
                    }
                    sum += libm::expf(score * scale - maximum) / normalizer
                        * (local - weighted_dot / normalizer)
                        * query[query_base + dimension]
                        * scale;
                }
                query_token += 1;
            }
            *gradient = sum;
        }
    }

    #[kernel]
    pub fn attention_value_backward_f32(
        query: &[f32],
        key: &[f32],
        output_gradient: &[f32],
        mut value_gradient: DisjointSlice<f32>,
        tokens: u32,
        heads: u32,
        width: u32,
        causal: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(gradient) = value_gradient.get_mut(index) {
            let width = width as usize;
            let heads = heads as usize;
            let tokens = tokens as usize;
            let dimension = item % width;
            let head = (item / width) % heads;
            let key_token = (item / (width * heads)) % tokens;
            let sequence = item / (width * heads * tokens);
            let scale = 1.0 / libm::sqrtf(width as f32);
            let mut sum = 0.0;
            let mut query_token = 0;
            while query_token < tokens {
                if causal == 0 || key_token <= query_token {
                    let query_base =
                        (sequence * tokens + query_token) * heads * width + head * width;
                    let limit = if causal != 0 { query_token + 1 } else { tokens };
                    let mut maximum = -3.402_823_5e38;
                    let mut token = 0;
                    while token < limit {
                        let base = (sequence * tokens + token) * heads * width + head * width;
                        let mut score = 0.0;
                        let mut d = 0;
                        while d < width {
                            score += query[query_base + d] * key[base + d];
                            d += 1;
                        }
                        if score * scale > maximum {
                            maximum = score * scale;
                        }
                        token += 1;
                    }
                    let mut normalizer = 0.0;
                    token = 0;
                    while token < limit {
                        let base = (sequence * tokens + token) * heads * width + head * width;
                        let mut score = 0.0;
                        let mut d = 0;
                        while d < width {
                            score += query[query_base + d] * key[base + d];
                            d += 1;
                        }
                        normalizer += libm::expf(score * scale - maximum);
                        token += 1;
                    }
                    let key_base = (sequence * tokens + key_token) * heads * width + head * width;
                    let mut score = 0.0;
                    let mut d = 0;
                    while d < width {
                        score += query[query_base + d] * key[key_base + d];
                        d += 1;
                    }
                    sum += libm::expf(score * scale - maximum) / normalizer
                        * output_gradient[query_base + dimension];
                }
                query_token += 1;
            }
            *gradient = sum;
        }
    }

    #[kernel]
    pub fn csr_adaptive_logits_f32(
        indptr: &[u32],
        indices: &[u32],
        parameters: &[f32],
        mut logits: DisjointSlice<f32>,
        source_offset: u32,
        target_offset: u32,
        latent: u32,
    ) {
        let edge_index = thread::index_1d();
        let edge = edge_index.get();
        if let Some(logit) = logits.get_mut(edge_index) {
            let mut row = 0;
            while row + 1 < indptr.len() && indptr[row + 1] as usize <= edge {
                row += 1;
            }
            if row + 1 < indptr.len() {
                let source = indices[edge] as usize;
                let mut score = 0.0;
                let mut feature = 0;
                while feature < latent as usize {
                    score += parameters[source_offset as usize + row * latent as usize + feature]
                        * parameters[target_offset as usize + source * latent as usize + feature];
                    feature += 1;
                }
                *logit = if score > 0.0 { score } else { 0.0 };
            }
        }
    }

    #[kernel]
    pub fn lsttn_long_conv_pool_parameter_slice_f32(
        input: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let channels = channels as usize;
            let nodes = nodes as usize;
            let output_times = (times as usize).div_ceil(2).div_ceil(2);
            let channel = item % channels;
            let q = item / channels;
            let node = q % nodes;
            let pool_time = q / nodes % output_times;
            let batch = q / (nodes * output_times);
            let convolution_times = (times as usize).div_ceil(2);
            let first = pool_time as isize * 2 - 1;
            let mut maximum = -3.402_823_5e38;
            let mut candidate = first;
            while candidate <= first + 2 {
                if candidate >= 0 && candidate < convolution_times as isize {
                    let center = candidate as usize * 2;
                    let mut sum = parameters[bias_offset as usize + channel];
                    let mut tap = 0;
                    while tap < 3 {
                        let source_time = if tap == 0 {
                            center as isize - dilation as isize
                        } else if tap == 1 {
                            center as isize
                        } else {
                            center as isize + dilation as isize
                        };
                        if source_time >= 0 && source_time < times as isize {
                            let mut input_channel = 0;
                            while input_channel < channels {
                                sum += input[((batch * times as usize + source_time as usize)
                                    * nodes
                                    + node)
                                    * channels
                                    + input_channel]
                                    * parameters[weights_offset as usize
                                        + (tap * channels + input_channel) * channels
                                        + channel];
                                input_channel += 1;
                            }
                        }
                        tap += 1;
                    }
                    let gelu = 0.5
                        * sum
                        * (1.0 + libm::tanhf(0.797_884_6 * (sum + 0.044_715 * sum * sum * sum)));
                    if gelu > maximum {
                        maximum = gelu;
                    }
                }
                candidate += 1;
            }
            *value = maximum;
        }
    }

    #[kernel]
    pub fn csr_adaptive_logits_parameter_slice_backward_f32_unused(
        indptr: &[u32],
        indices: &[u32],
        parameters: &[f32],
        logits_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        source_offset: u32,
        target_offset: u32,
        nodes: u32,
        latent: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let latent = latent as usize;
        let source_start = source_offset as usize;
        let target_start = target_offset as usize;
        if item >= source_start && item < source_start + nodes as usize * latent {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let local = item - source_start;
                let row = local / latent;
                let feature = local % latent;
                let mut sum = 0.0;
                let mut edge = indptr[row] as usize;
                while edge < indptr[row + 1] as usize {
                    let source = indices[edge] as usize;
                    let mut score = 0.0;
                    let mut f = 0;
                    while f < latent {
                        score += parameters[source_start + row * latent + f]
                            * parameters[target_start + source * latent + f];
                        f += 1;
                    }
                    if score > 0.0 {
                        sum += logits_gradient[edge]
                            * parameters[target_start + source * latent + feature];
                    }
                    edge += 1;
                }
                *value += sum;
            }
        } else if item >= target_start && item < target_start + nodes as usize * latent {
            if let Some(value) = parameter_gradient.get_mut(index) {
                let local = item - target_start;
                let source = local / latent;
                let feature = local % latent;
                let mut sum = 0.0;
                let mut row = 0;
                while row < nodes as usize {
                    let mut edge = indptr[row] as usize;
                    while edge < indptr[row + 1] as usize {
                        if indices[edge] as usize == source {
                            let mut score = 0.0;
                            let mut f = 0;
                            while f < latent {
                                score += parameters[source_start + row * latent + f]
                                    * parameters[target_start + source * latent + f];
                                f += 1;
                            }
                            if score > 0.0 {
                                sum += logits_gradient[edge]
                                    * parameters[source_start + row * latent + feature];
                            }
                        }
                        edge += 1;
                    }
                    row += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn csr_adaptive_logits_parameter_slice_backward_f32(
        indptr: &[u32],
        indices: &[u32],
        parameters: &[f32],
        logits_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        source_offset: u32,
        target_offset: u32,
        nodes: u32,
        latent: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = latent as usize;
        let count = nodes as usize * width;
        let source_start = source_offset as usize;
        let target_start = target_offset as usize;
        if let Some(value) = parameter_gradient.get_mut(index) {
            if item >= source_start && item < source_start + count {
                let local = item - source_start;
                let row = local / width;
                let feature = local % width;
                let mut sum = 0.0;
                let mut edge = indptr[row] as usize;
                while edge < indptr[row + 1] as usize {
                    let target = indices[edge] as usize;
                    let mut score = 0.0;
                    let mut feature_index = 0;
                    while feature_index < width {
                        score += parameters[source_start + row * width + feature_index]
                            * parameters[target_start + target * width + feature_index];
                        feature_index += 1;
                    }
                    if score > 0.0 {
                        sum += logits_gradient[edge]
                            * parameters[target_start + target * width + feature];
                    }
                    edge += 1;
                }
                *value += sum;
            } else if item >= target_start && item < target_start + count {
                let local = item - target_start;
                let target = local / width;
                let feature = local % width;
                let mut sum = 0.0;
                let mut row = 0;
                while row < nodes as usize {
                    let mut edge = indptr[row] as usize;
                    while edge < indptr[row + 1] as usize {
                        if indices[edge] as usize == target {
                            let mut score = 0.0;
                            let mut feature_index = 0;
                            while feature_index < width {
                                score += parameters[source_start + row * width + feature_index]
                                    * parameters[target_start + target * width + feature_index];
                                feature_index += 1;
                            }
                            if score > 0.0 {
                                sum += logits_gradient[edge]
                                    * parameters[source_start + row * width + feature];
                            }
                        }
                        edge += 1;
                    }
                    row += 1;
                }
                *value += sum;
            }
        }
    }

    #[kernel]
    pub fn lsttn_long_conv_pool_parameter_slice_f32_unused(
        input: &[f32],
        parameters: &[f32],
        mut output: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        times: u32,
        convolution_times: u32,
        output_times: u32,
        nodes: u32,
        channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let width = channels as usize;
            let raw_nodes = nodes as usize;
            let raw_times = times as usize;
            let channel = item % width;
            let q = item / width;
            let node = q % raw_nodes;
            let pool_time = (q / raw_nodes) % output_times as usize;
            let batch = q / (raw_nodes * output_times as usize);
            let mut maximum = -3.402_823_5e38_f32;
            let first = pool_time as isize * 2 - 1;
            let mut candidate = first;
            while candidate <= first + 2 {
                if candidate >= 0 && candidate < convolution_times as isize {
                    let center = candidate as usize * 2;
                    let mut sum = parameters[bias_offset as usize + channel];
                    let mut tap = 0;
                    while tap < 3 {
                        let source_time = center as isize + (tap as isize - 1) * dilation as isize;
                        if source_time >= 0 && source_time < raw_times as isize {
                            let mut input_channel = 0;
                            while input_channel < width {
                                let source =
                                    ((batch * raw_times + source_time as usize) * raw_nodes + node)
                                        * width
                                        + input_channel;
                                sum += input[source]
                                    * parameters[weights_offset as usize
                                        + (tap * width + input_channel) * width
                                        + channel];
                                input_channel += 1;
                            }
                        }
                        tap += 1;
                    }
                    let activated = 0.5
                        * sum
                        * (1.0 + libm::tanhf(0.797_884_6 * (sum + 0.044_715 * sum * sum * sum)));
                    if activated > maximum {
                        maximum = activated;
                    }
                }
                candidate += 1;
            }
            *value = maximum;
        }
    }

    #[kernel]
    pub fn lsttn_long_conv_pool_input_backward_f32(
        input: &[f32],
        parameters: &[f32],
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        times: u32,
        convolution_times: u32,
        output_times: u32,
        nodes: u32,
        channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = input_gradient.get_mut(index) {
            let width = channels as usize;
            let raw_nodes = nodes as usize;
            let raw_times = times as usize;
            let input_channel = item % width;
            let q = item / width;
            let node = q % raw_nodes;
            let source_time = (q / raw_nodes) % raw_times;
            let batch = q / (raw_nodes * raw_times);
            let mut sum = 0.0;
            let mut pool_time = 0;
            while pool_time < output_times as usize {
                let first = pool_time as isize * 2 - 1;
                let mut candidate = first;
                while candidate <= first + 2 {
                    if candidate >= 0 && candidate < convolution_times as isize {
                        let center = candidate as usize * 2;
                        let mut tap = 0;
                        while tap < 3 {
                            let tap_time = center as isize + (tap as isize - 1) * dilation as isize;
                            if tap_time == source_time as isize {
                                let mut out_channel = 0;
                                while out_channel < width {
                                    let mut maximum = -3.402_823_5e38_f32;
                                    let mut winner = -1isize;
                                    let mut choose = first;
                                    while choose <= first + 2 {
                                        if choose >= 0 && choose < convolution_times as isize {
                                            let choose_center = choose as usize * 2;
                                            let mut raw =
                                                parameters[bias_offset as usize + out_channel];
                                            let mut choose_tap = 0;
                                            while choose_tap < 3 {
                                                let choose_time = choose_center as isize
                                                    + (choose_tap as isize - 1) * dilation as isize;
                                                if choose_time >= 0
                                                    && choose_time < raw_times as isize
                                                {
                                                    let mut channel = 0;
                                                    while channel < width {
                                                        raw += input[((batch * raw_times
                                                            + choose_time as usize)
                                                            * raw_nodes
                                                            + node)
                                                            * width
                                                            + channel]
                                                            * parameters[weights_offset as usize
                                                                + (choose_tap * width + channel)
                                                                    * width
                                                                + out_channel];
                                                        channel += 1;
                                                    }
                                                }
                                                choose_tap += 1;
                                            }
                                            let active = 0.5
                                                * raw
                                                * (1.0
                                                    + libm::tanhf(
                                                        0.797_884_6
                                                            * (raw + 0.044_715 * raw * raw * raw),
                                                    ));
                                            if active > maximum {
                                                maximum = active;
                                                winner = choose;
                                            }
                                        }
                                        choose += 1;
                                    }
                                    if winner == candidate {
                                        let mut raw =
                                            parameters[bias_offset as usize + out_channel];
                                        let mut raw_tap = 0;
                                        while raw_tap < 3 {
                                            let raw_time = center as isize
                                                + (raw_tap as isize - 1) * dilation as isize;
                                            if raw_time >= 0 && raw_time < raw_times as isize {
                                                let mut channel = 0;
                                                while channel < width {
                                                    raw += input[((batch * raw_times
                                                        + raw_time as usize)
                                                        * raw_nodes
                                                        + node)
                                                        * width
                                                        + channel]
                                                        * parameters[weights_offset as usize
                                                            + (raw_tap * width + channel) * width
                                                            + out_channel];
                                                    channel += 1;
                                                }
                                            }
                                            raw_tap += 1;
                                        }
                                        let u = 0.797_884_6 * (raw + 0.044_715 * raw * raw * raw);
                                        let t = libm::tanhf(u);
                                        let gelu_gradient = 0.5 * (1.0 + t)
                                            + 0.5
                                                * raw
                                                * (1.0 - t * t)
                                                * 0.797_884_6
                                                * (1.0 + 0.134_145 * raw * raw);
                                        sum += output_gradient[((batch * output_times as usize
                                            + pool_time)
                                            * raw_nodes
                                            + node)
                                            * width
                                            + out_channel]
                                            * gelu_gradient
                                            * parameters[weights_offset as usize
                                                + (tap * width + input_channel) * width
                                                + out_channel];
                                    }
                                    out_channel += 1;
                                }
                            }
                            tap += 1;
                        }
                    }
                    candidate += 1;
                }
                pool_time += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn lsttn_long_conv_pool_parameter_backward_f32(
        input: &[f32],
        parameters: &[f32],
        output_gradient: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        weights_offset: u32,
        bias_offset: u32,
        batches: u32,
        times: u32,
        convolution_times: u32,
        output_times: u32,
        nodes: u32,
        channels: u32,
        dilation: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = channels as usize;
        let weights = 3 * width * width;
        let weight_start = weights_offset as usize;
        let bias_start = bias_offset as usize;
        if let Some(value) = parameter_gradient.get_mut(index) {
            if item >= weight_start && item < weight_start + weights {
                let local = item - weight_start;
                let out_channel = local % width;
                let q = local / width;
                let input_channel = q % width;
                let tap = q / width;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut pool_time = 0;
                    while pool_time < output_times as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            let first = pool_time as isize * 2 - 1;
                            let mut maximum = -3.402_823_5e38_f32;
                            let mut winner = -1isize;
                            let mut candidate = first;
                            while candidate <= first + 2 {
                                if candidate >= 0 && candidate < convolution_times as isize {
                                    let center = candidate as usize * 2;
                                    let mut raw = parameters[bias_start + out_channel];
                                    let mut raw_tap = 0;
                                    while raw_tap < 3 {
                                        let source_time = center as isize
                                            + (raw_tap as isize - 1) * dilation as isize;
                                        if source_time >= 0 && source_time < times as isize {
                                            let mut channel = 0;
                                            while channel < width {
                                                raw += input[((batch * times as usize
                                                    + source_time as usize)
                                                    * nodes as usize
                                                    + node)
                                                    * width
                                                    + channel]
                                                    * parameters[weight_start
                                                        + (raw_tap * width + channel) * width
                                                        + out_channel];
                                                channel += 1;
                                            }
                                        }
                                        raw_tap += 1;
                                    }
                                    let active = 0.5
                                        * raw
                                        * (1.0
                                            + libm::tanhf(
                                                0.797_884_6 * (raw + 0.044_715 * raw * raw * raw),
                                            ));
                                    if active > maximum {
                                        maximum = active;
                                        winner = candidate;
                                    }
                                }
                                candidate += 1;
                            }
                            if winner >= 0 {
                                let center = winner as usize * 2;
                                let source_time =
                                    center as isize + (tap as isize - 1) * dilation as isize;
                                if source_time >= 0 && source_time < times as isize {
                                    let mut raw = parameters[bias_start + out_channel];
                                    let mut raw_tap = 0;
                                    while raw_tap < 3 {
                                        let raw_time = center as isize
                                            + (raw_tap as isize - 1) * dilation as isize;
                                        if raw_time >= 0 && raw_time < times as isize {
                                            let mut channel = 0;
                                            while channel < width {
                                                raw += input[((batch * times as usize
                                                    + raw_time as usize)
                                                    * nodes as usize
                                                    + node)
                                                    * width
                                                    + channel]
                                                    * parameters[weight_start
                                                        + (raw_tap * width + channel) * width
                                                        + out_channel];
                                                channel += 1;
                                            }
                                        }
                                        raw_tap += 1;
                                    }
                                    let u = 0.797_884_6 * (raw + 0.044_715 * raw * raw * raw);
                                    let t = libm::tanhf(u);
                                    let gelu_gradient = 0.5 * (1.0 + t)
                                        + 0.5
                                            * raw
                                            * (1.0 - t * t)
                                            * 0.797_884_6
                                            * (1.0 + 0.134_145 * raw * raw);
                                    sum += input[((batch * times as usize + source_time as usize)
                                        * nodes as usize
                                        + node)
                                        * width
                                        + input_channel]
                                        * output_gradient[((batch * output_times as usize
                                            + pool_time)
                                            * nodes as usize
                                            + node)
                                            * width
                                            + out_channel]
                                        * gelu_gradient;
                                }
                            }
                            node += 1;
                        }
                        pool_time += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            } else if item >= bias_start && item < bias_start + width {
                let out_channel = item - bias_start;
                let mut sum = 0.0;
                let mut batch = 0;
                while batch < batches as usize {
                    let mut pool_time = 0;
                    while pool_time < output_times as usize {
                        let mut node = 0;
                        while node < nodes as usize {
                            let first = pool_time as isize * 2 - 1;
                            let mut maximum = -3.402_823_5e38_f32;
                            let mut winner = -1isize;
                            let mut candidate = first;
                            let mut winner_raw = 0.0;
                            while candidate <= first + 2 {
                                if candidate >= 0 && candidate < convolution_times as isize {
                                    let center = candidate as usize * 2;
                                    let mut raw = parameters[bias_start + out_channel];
                                    let mut tap = 0;
                                    while tap < 3 {
                                        let source_time = center as isize
                                            + (tap as isize - 1) * dilation as isize;
                                        if source_time >= 0 && source_time < times as isize {
                                            let mut channel = 0;
                                            while channel < width {
                                                raw += input[((batch * times as usize
                                                    + source_time as usize)
                                                    * nodes as usize
                                                    + node)
                                                    * width
                                                    + channel]
                                                    * parameters[weight_start
                                                        + (tap * width + channel) * width
                                                        + out_channel];
                                                channel += 1;
                                            }
                                        }
                                        tap += 1;
                                    }
                                    let active = 0.5
                                        * raw
                                        * (1.0
                                            + libm::tanhf(
                                                0.797_884_6 * (raw + 0.044_715 * raw * raw * raw),
                                            ));
                                    if active > maximum {
                                        maximum = active;
                                        winner = candidate;
                                        winner_raw = raw;
                                    }
                                }
                                candidate += 1;
                            }
                            if winner >= 0 {
                                let u = 0.797_884_6
                                    * (winner_raw
                                        + 0.044_715 * winner_raw * winner_raw * winner_raw);
                                let t = libm::tanhf(u);
                                let gelu_gradient = 0.5 * (1.0 + t)
                                    + 0.5
                                        * winner_raw
                                        * (1.0 - t * t)
                                        * 0.797_884_6
                                        * (1.0 + 0.134_145 * winner_raw * winner_raw);
                                sum += output_gradient[((batch * output_times as usize
                                    + pool_time)
                                    * nodes as usize
                                    + node)
                                    * width
                                    + out_channel]
                                    * gelu_gradient;
                            }
                            node += 1;
                        }
                        pool_time += 1;
                    }
                    batch += 1;
                }
                *value += sum;
            }
        }
    }

    #[cfg(any())]
    #[kernel]
    pub fn masked_patch_reconstruction_loss_oxide_temp_f32(
        decoded: &[f32],
        target: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        mut loss: DisjointSlice<f32>,
        decoder_offset: u32,
        decoder_bias_offset: u32,
        batches: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        visible: u32,
        masked: u32,
        patch_width: u32,
        hidden: u32,
        masked_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        if index.get() == 0 {
            let mut count = 0.0;
            let mut total = 0.0;
            let mut batch = 0;
            while batch < batches as usize {
                let mut node = 0;
                while node < nodes as usize {
                    let mut mask = 0;
                    while mask < masked as usize {
                        let patch = masked_indices[mask] as usize;
                        let base = ((batch * nodes as usize + node) * (visible + masked) as usize
                            + visible as usize
                            + mask)
                            * hidden as usize;
                        let mut offset = 0;
                        while offset < patch_width as usize {
                            let time = patch * patch_width as usize + offset;
                            if time < times as usize {
                                let actual = target[((batch * times as usize + time)
                                    * nodes as usize
                                    + node)
                                    * channels as usize];
                                if libm::fabsf(actual - masked_zero) > 1.0e-12 {
                                    let mut prediction =
                                        parameters[decoder_bias_offset as usize + offset];
                                    let mut channel = 0;
                                    while channel < hidden as usize {
                                        prediction += decoded[base + channel]
                                            * parameters[decoder_offset as usize
                                                + offset * hidden as usize
                                                + channel];
                                        channel += 1;
                                    }
                                    let residual = (prediction - actual) * target_scale;
                                    total += libm::sqrtf(residual * residual + 1.0e-12);
                                    count += 1.0;
                                }
                            }
                            offset += 1;
                        }
                        mask += 1;
                    }
                    node += 1;
                }
                batch += 1;
            }
            if let Some(value) = loss.get_mut(index) {
                *value = if item == 0 {
                    total / if count > 1.0 { count } else { 1.0 }
                } else {
                    if count > 1.0 {
                        count
                    } else {
                        1.0
                    }
                };
            }
        }
    }

    #[cfg(any())]
    #[kernel]
    pub fn masked_patch_reconstruction_context_backward_oxide_temp_f32(
        decoded: &[f32],
        target: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        loss: &[f32],
        mut context_gradient: DisjointSlice<f32>,
        decoder_offset: u32,
        decoder_bias_offset: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        visible: u32,
        masked: u32,
        patch_width: u32,
        hidden: u32,
        masked_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = context_gradient.get_mut(index) {
            let width = hidden as usize;
            let tokens = (visible + masked) as usize;
            let channel = item % width;
            let q = item / width;
            let token = q % tokens;
            let node = (q / tokens) % nodes as usize;
            let batch = q / (tokens * nodes as usize);
            if token < visible as usize {
                *value = 0.0;
                return;
            }
            let patch = masked_indices[token - visible as usize] as usize;
            let base = ((batch * nodes as usize + node) * tokens + token) * width;
            let mut sum = 0.0;
            let mut offset = 0;
            while offset < patch_width as usize {
                let time = patch * patch_width as usize + offset;
                if time < times as usize {
                    let actual = target[((batch * times as usize + time) * nodes as usize + node)
                        * channels as usize];
                    if libm::fabsf(actual - masked_zero) > 1.0e-12 {
                        let mut prediction = parameters[decoder_bias_offset as usize + offset];
                        let mut feature = 0;
                        while feature < width {
                            prediction += decoded[base + feature]
                                * parameters[decoder_offset as usize + offset * width + feature];
                            feature += 1;
                        }
                        let residual = (prediction - actual) * target_scale;
                        sum += residual / libm::sqrtf(residual * residual + 1.0e-12) * target_scale
                            / loss[1]
                            * parameters[decoder_offset as usize + offset * width + channel];
                    }
                }
                offset += 1;
            }
            *value = sum;
        }
    }

    #[cfg(any())]
    #[kernel]
    pub fn masked_patch_reconstruction_parameter_backward_oxide_temp_f32(
        decoded: &[f32],
        target: &[f32],
        masked_indices: &[u32],
        parameters: &[f32],
        loss: &[f32],
        mut parameter_gradient: DisjointSlice<f32>,
        decoder_offset: u32,
        decoder_bias_offset: u32,
        batches: u32,
        times: u32,
        nodes: u32,
        channels: u32,
        visible: u32,
        masked: u32,
        patch_width: u32,
        hidden: u32,
        masked_zero: f32,
        target_scale: f32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let width = hidden as usize;
        let weights = patch_width as usize * width;
        if let Some(value) = parameter_gradient.get_mut(index) {
            let is_weight =
                item >= decoder_offset as usize && item < decoder_offset as usize + weights;
            let is_bias = item >= decoder_bias_offset as usize
                && item < decoder_bias_offset as usize + patch_width as usize;
            if !is_weight && !is_bias {
                return;
            }
            let local = if is_weight {
                item - decoder_offset as usize
            } else {
                item - decoder_bias_offset as usize
            };
            let offset = if is_weight { local / width } else { local };
            let channel = if is_weight { local % width } else { 0 };
            let mut sum = 0.0;
            let mut batch = 0;
            while batch < batches as usize {
                let mut node = 0;
                while node < nodes as usize {
                    let mut mask = 0;
                    while mask < masked as usize {
                        let patch = masked_indices[mask] as usize;
                        let time = patch * patch_width as usize + offset;
                        if time < times as usize {
                            let actual = target[((batch * times as usize + time) * nodes as usize
                                + node)
                                * channels as usize];
                            if libm::fabsf(actual - masked_zero) > 1.0e-12 {
                                let base = ((batch * nodes as usize + node)
                                    * (visible + masked) as usize
                                    + visible as usize
                                    + mask)
                                    * width;
                                let mut prediction =
                                    parameters[decoder_bias_offset as usize + offset];
                                let mut feature = 0;
                                while feature < width {
                                    prediction += decoded[base + feature]
                                        * parameters
                                            [decoder_offset as usize + offset * width + feature];
                                    feature += 1;
                                }
                                let residual = (prediction - actual) * target_scale;
                                let gradient = residual
                                    / libm::sqrtf(residual * residual + 1.0e-12)
                                    * target_scale
                                    / loss[1];
                                sum += if is_weight {
                                    decoded[base + channel] * gradient
                                } else {
                                    gradient
                                };
                            }
                        }
                        mask += 1;
                    }
                    node += 1;
                }
                batch += 1;
            }
            *value += sum;
        }
    }

    #[kernel]
    pub fn split_left_channels_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        left_channels: u32,
        right_channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let row = item / left_channels as usize;
            let channel = item % left_channels as usize;
            *value = input[row * (left_channels + right_channels) as usize + channel];
        }
    }

    #[kernel]
    pub fn split_right_channels_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        left_channels: u32,
        right_channels: u32,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = output.get_mut(index) {
            let row = item / right_channels as usize;
            let channel = item % right_channels as usize;
            *value = input[row * (left_channels + right_channels) as usize
                + left_channels as usize
                + channel];
        }
    }

    #[kernel]
    pub fn copy_f32(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output_value) = output.get_mut(index) {
            *output_value = input[item];
        }
    }

    #[kernel]
    pub fn fill_f32(mut output: DisjointSlice<f32>, value: f32) {
        let index = thread::index_1d();
        if let Some(output_value) = output.get_mut(index) {
            *output_value = value;
        }
    }

    #[kernel]
    pub fn adamw_f32(
        mut parameters: DisjointSlice<f32>,
        mut first: DisjointSlice<f32>,
        mut second: DisjointSlice<f32>,
        gradients: &[f32],
        learning_rate: f32,
        weight_decay: f32,
        first_correction: f32,
        second_correction: f32,
    ) {
        let index = thread::index_1d();
        let raw_index = index.get();
        if let (Some(parameter), Some(first_moment), Some(second_moment)) = (
            parameters.get_mut(index),
            first.get_mut(thread::index_1d()),
            second.get_mut(thread::index_1d()),
        ) {
            let gradient = gradients[raw_index] + weight_decay * *parameter;
            *first_moment = 0.9 * *first_moment + 0.1 * gradient;
            *second_moment = 0.999 * *second_moment + 0.001 * gradient * gradient;
            let first_hat = *first_moment / first_correction;
            let second_hat = *second_moment / second_correction;
            *parameter -= learning_rate * first_hat / (libm::sqrtf(second_hat) + 1.0e-8);
        }
    }

    #[kernel]
    pub fn pair_sigmoid_scores_f32(
        embeddings: &[f32],
        pairs: &[u32],
        mut output: DisjointSlice<f32>,
        dim: u32,
    ) {
        let pair = thread::index_1d();
        let raw_pair = pair.get();
        if let Some(output_value) = output.get_mut(pair) {
            let raw_dim = dim as usize;
            let source = pairs[raw_pair * 2] as usize;
            let target = pairs[raw_pair * 2 + 1] as usize;
            let mut score = 0.0;
            let mut col = 0;
            while col < raw_dim {
                score += embeddings[source * raw_dim + col] * embeddings[target * raw_dim + col];
                col += 1;
            }
            *output_value = 1.0 / (1.0 + libm::expf(-score));
        }
    }

    #[kernel]
    pub fn csr_row_softmax_f32(indptr: &[u32], logits: &[f32], mut weights: DisjointSlice<f32>) {
        let edge_index = thread::index_1d();
        let edge = edge_index.get();
        if let Some(weight) = weights.get_mut(edge_index) {
            let mut row = 0;
            while row + 1 < indptr.len() && indptr[row + 1] as usize <= edge {
                row += 1;
            }
            if row + 1 < indptr.len() {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                let mut maximum = logits[start];
                let mut current = start + 1;
                while current < end {
                    if logits[current] > maximum {
                        maximum = logits[current];
                    }
                    current += 1;
                }
                let mut total = 0.0;
                current = start;
                while current < end {
                    total += libm::expf(logits[current] - maximum);
                    current += 1;
                }
                *weight = libm::expf(logits[edge] - maximum) / total;
            }
        }
    }

    #[kernel]
    pub fn csr_row_softmax_backward_f32(
        indptr: &[u32],
        weights: &[f32],
        output_gradient: &[f32],
        mut logits_gradient: DisjointSlice<f32>,
    ) {
        let edge_index = thread::index_1d();
        let edge = edge_index.get();
        if let Some(gradient) = logits_gradient.get_mut(edge_index) {
            let mut row = 0;
            while row + 1 < indptr.len() && indptr[row + 1] as usize <= edge {
                row += 1;
            }
            if row + 1 < indptr.len() {
                let mut dot = 0.0;
                let mut current = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                while current < end {
                    dot += weights[current] * output_gradient[current];
                    current += 1;
                }
                *gradient = weights[edge] * (output_gradient[edge] - dot);
            }
        }
    }
}

pub(super) fn is_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// cuda-oxide-backed reusable sparse graph plan. Graph structure remains on
/// the device while the activation buffer grows only when a larger batch is
/// requested.
pub struct CudaCsrDiffusionWorkspace {
    context: Arc<CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    indptr: DeviceBuffer<u32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    values: Option<DeviceBuffer<f32>>,
    output: Option<DeviceBuffer<f32>>,
    nodes: usize,
    value_capacity: usize,
    allocation_count: usize,
}

/// Persistent cuda-oxide tensor slots. The public arena will switch to this
/// type once every established operation has a Rust kernel equivalent.
pub struct CudaTensorArena {
    context: Arc<CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    f32_slots: Vec<Option<DeviceBuffer<f32>>>,
    f32_capacities: Vec<usize>,
    u32_slots: Vec<Option<DeviceBuffer<u32>>>,
    u32_capacities: Vec<usize>,
    allocation_count: usize,
}

impl CudaTensorArena {
    pub fn new(slots: usize) -> Result<Self> {
        if slots == 0 {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA tensor arena requires at least one slot".to_string(),
            ));
        }
        let context = context()?;
        let stream = context.default_stream();
        Ok(Self {
            context,
            stream,
            f32_slots: (0..slots).map(|_| None).collect(),
            f32_capacities: vec![0; slots],
            u32_slots: (0..slots).map(|_| None).collect(),
            u32_capacities: vec![0; slots],
            allocation_count: 0,
        })
    }

    pub fn affine_f32(
        &mut self,
        input: usize,
        weights: usize,
        bias: usize,
        output: usize,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let input_len = checked_product(&[rows, input_width])?;
        let output_len = checked_product(&[rows, output_width])?;
        if rows == 0
            || input_width == 0
            || output_width == 0
            || input == weights
            || input == bias
            || input == output
            || weights == bias
            || weights == output
            || bias == output
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(weights)? < input_width * output_width
            || self.capacity_f32(bias)? < output_width
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide affine slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, weight_values, bias_values, output_values) =
            get_four_f32_slots(&mut self.f32_slots, input, weights, bias, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.dense_layer_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                weight_values,
                bias_values,
                output_values,
                input_width as u32,
                output_width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide affine: {error}"
            ))
        })
    }

    pub fn affine_backward_f32(
        &mut self,
        input: usize,
        weights: usize,
        output_gradient: usize,
        input_gradient: usize,
        weight_gradient: usize,
        bias_gradient: usize,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let input_len = checked_product(&[rows, input_width])?;
        let output_len = checked_product(&[rows, output_width])?;
        let weight_len = input_width * output_width;
        let slots = [
            input,
            weights,
            output_gradient,
            input_gradient,
            weight_gradient,
            bias_gradient,
        ];
        if rows == 0
            || input_width == 0
            || output_width == 0
            || slots
                .iter()
                .enumerate()
                .any(|(i, slot)| slots[..i].contains(slot))
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(weights)? < weight_len
            || self.capacity_f32(output_gradient)? < output_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide affine backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, input_len)?;
        self.reserve_f32(weight_gradient, weight_len)?;
        self.reserve_f32(bias_gradient, output_width)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (gradient, weight, input_grad) = get_three_f32_slots(
                &mut self.f32_slots,
                output_gradient,
                weights,
                input_gradient,
            )?;
            unsafe {
                module.affine_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(input_len as u32),
                    gradient,
                    weight,
                    input_grad,
                    input_width as u32,
                    output_width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide affine input backward: {error}"
                ))
            })?;
        }
        {
            let (input_values, gradient, weight_grad) =
                get_three_f32_slots(&mut self.f32_slots, input, output_gradient, weight_gradient)?;
            unsafe {
                module.affine_weight_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(weight_len as u32),
                    input_values,
                    gradient,
                    weight_grad,
                    rows as u32,
                    input_width as u32,
                    output_width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide affine weight backward: {error}"
                ))
            })?;
        }
        let (gradient, bias_grad) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, bias_gradient)?;
        unsafe {
            module.affine_bias_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_width as u32),
                gradient,
                bias_grad,
                rows as u32,
                output_width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide affine bias backward: {error}"
            ))
        })
    }

    pub fn layer_norm_f32(
        &mut self,
        values: usize,
        gamma: usize,
        beta: usize,
        output: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let len = checked_product(&[rows, width])?;
        if rows == 0
            || width == 0
            || values == gamma
            || values == beta
            || values == output
            || gamma == beta
            || gamma == output
            || beta == output
            || self.capacity_f32(values)? < len
            || self.capacity_f32(gamma)? < width
            || self.capacity_f32(beta)? < width
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide layer normalization slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (value_values, gamma_values, beta_values, output_values) =
            get_four_f32_slots(&mut self.f32_slots, values, gamma, beta, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.layer_norm_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                value_values,
                gamma_values,
                beta_values,
                output_values,
                width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide layer normalization: {error}"
            ))
        })
    }

    pub fn layer_norm_backward_f32(
        &mut self,
        values: usize,
        gamma: usize,
        output_gradient: usize,
        input_gradient: usize,
        gamma_gradient: usize,
        beta_gradient: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let len = checked_product(&[rows, width])?;
        let slots = [
            values,
            gamma,
            output_gradient,
            input_gradient,
            gamma_gradient,
            beta_gradient,
        ];
        if rows == 0
            || width == 0
            || slots
                .iter()
                .enumerate()
                .any(|(i, slot)| slots[..i].contains(slot))
            || self.capacity_f32(values)? < len
            || self.capacity_f32(gamma)? < width
            || self.capacity_f32(output_gradient)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide layer normalization backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, len)?;
        self.reserve_f32(gamma_gradient, width)?;
        self.reserve_f32(beta_gradient, width)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (value_values, gamma_values, gradient_values, input_gradient_values) =
                get_four_f32_slots(
                    &mut self.f32_slots,
                    values,
                    gamma,
                    output_gradient,
                    input_gradient,
                )?;
            unsafe {
                module.layer_norm_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    value_values,
                    gamma_values,
                    gradient_values,
                    input_gradient_values,
                    width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide layer normalization input backward: {error}"
                ))
            })?;
        }
        {
            let (value_values, gradient_values, gamma_gradient_values) =
                get_three_f32_slots(&mut self.f32_slots, values, output_gradient, gamma_gradient)?;
            unsafe {
                module.layer_norm_gamma_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(width as u32),
                    value_values,
                    gradient_values,
                    gamma_gradient_values,
                    rows as u32,
                    width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide layer normalization gamma backward: {error}"
                ))
            })?;
        }
        let (gradient_values, beta_gradient_values) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, beta_gradient)?;
        unsafe {
            module.layer_norm_beta_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(width as u32),
                gradient_values,
                beta_gradient_values,
                rows as u32,
                width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide layer normalization beta backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv2_backward_f32(
        &mut self,
        input: usize,
        weights: usize,
        output_gradient: usize,
        input_gradient: usize,
        weight_gradient: usize,
        bias_gradient: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        input_channels: usize,
        output_channels: usize,
        dilation: usize,
    ) -> Result<()> {
        if batches == 0
            || times <= dilation
            || nodes == 0
            || input_channels == 0
            || output_channels == 0
            || dilation == 0
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide causal convolution backward dimensions".to_string(),
            ));
        }
        let input_len = checked_product(&[batches, times, nodes, input_channels])?;
        let output_times = times - dilation;
        let output_len = checked_product(&[batches, output_times, nodes, output_channels])?;
        let weight_len = 2 * input_channels * output_channels;
        let slots = [
            input,
            weights,
            output_gradient,
            input_gradient,
            weight_gradient,
            bias_gradient,
        ];
        if slots
            .iter()
            .enumerate()
            .any(|(i, slot)| slots[..i].contains(slot))
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(weights)? < weight_len
            || self.capacity_f32(output_gradient)? < output_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide causal convolution backward slots".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, input_len)?;
        self.reserve_f32(weight_gradient, weight_len)?;
        self.reserve_f32(bias_gradient, output_channels)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (weight_values, gradient_values, input_gradient_values) = get_three_f32_slots(
                &mut self.f32_slots,
                weights,
                output_gradient,
                input_gradient,
            )?;
            unsafe {
                module.causal_conv2_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(input_len as u32),
                    weight_values,
                    gradient_values,
                    input_gradient_values,
                    times as u32,
                    nodes as u32,
                    input_channels as u32,
                    output_channels as u32,
                    dilation as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide causal convolution input backward: {error}"
                ))
            })?;
        }
        {
            let (input_values, gradient_values, weight_gradient_values) =
                get_three_f32_slots(&mut self.f32_slots, input, output_gradient, weight_gradient)?;
            unsafe {
                module.causal_conv2_weight_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(weight_len as u32),
                    input_values,
                    gradient_values,
                    weight_gradient_values,
                    batches as u32,
                    output_times as u32,
                    nodes as u32,
                    input_channels as u32,
                    output_channels as u32,
                    dilation as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide causal convolution weight backward: {error}"
                ))
            })?;
        }
        let (gradient_values, bias_gradient_values) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, bias_gradient)?;
        unsafe {
            module.causal_conv2_bias_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_channels as u32),
                gradient_values,
                bias_gradient_values,
                batches as u32,
                output_times as u32,
                nodes as u32,
                output_channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide causal convolution bias backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv2_f32(
        &mut self,
        input: usize,
        weights: usize,
        bias: usize,
        output: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        input_channels: usize,
        output_channels: usize,
        dilation: usize,
    ) -> Result<()> {
        if batches == 0
            || times <= dilation
            || nodes == 0
            || input_channels == 0
            || output_channels == 0
            || dilation == 0
            || input == weights
            || input == bias
            || input == output
            || weights == bias
            || weights == output
            || bias == output
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide causal convolution slots or dimensions".to_string(),
            ));
        }
        let output_times = times - dilation;
        let input_len = checked_product(&[batches, times, nodes, input_channels])?;
        let output_len = checked_product(&[batches, output_times, nodes, output_channels])?;
        if self.capacity_f32(input)? < input_len
            || self.capacity_f32(weights)? < 2 * input_channels * output_channels
            || self.capacity_f32(bias)? < output_channels
        {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide causal convolution input is smaller than its shape".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, weight_values, bias_values, output_values) =
            get_four_f32_slots(&mut self.f32_slots, input, weights, bias, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.causal_conv2_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                weight_values,
                bias_values,
                output_values,
                output_times as u32,
                nodes as u32,
                input_channels as u32,
                output_channels as u32,
                dilation as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide causal convolution: {error}"
            ))
        })
    }

    #[cfg(any())]
    #[allow(clippy::too_many_arguments)]
    pub fn lsttn_long_conv_pool_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        dilation: usize,
    ) -> Result<()> {
        let convolution_times = times.div_ceil(2);
        let output_times = convolution_times.div_ceil(2);
        let input_len = checked_product(&[batches, times, nodes, channels])?;
        let output_len = checked_product(&[batches, output_times, nodes, channels])?;
        let parameter_len = bias_offset.checked_add(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA long convolution parameter range overflows".to_string(),
            )
        })?;
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || dilation == 0
            || input == parameters
            || input == output
            || parameters == output
            || weights_offset
                .checked_add(3 * channels * channels)
                .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA long convolution slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.lsttn_long_conv_pool_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                parameter_values,
                output_values,
                weights_offset as u32,
                bias_offset as u32,
                times as u32,
                convolution_times as u32,
                output_times as u32,
                nodes as u32,
                channels as u32,
                dilation as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide long convolution: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lsttn_long_conv_pool_parameter_slice_backward_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output_gradient: usize,
        input_gradient: usize,
        parameter_gradient: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        dilation: usize,
    ) -> Result<()> {
        let convolution_times = times.div_ceil(2);
        let output_times = convolution_times.div_ceil(2);
        let input_len = checked_product(&[batches, times, nodes, channels])?;
        let output_len = checked_product(&[batches, output_times, nodes, channels])?;
        let parameter_len = bias_offset.checked_add(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA long convolution gradient range overflows".to_string(),
            )
        })?;
        let slots = [
            input,
            parameters,
            output_gradient,
            input_gradient,
            parameter_gradient,
        ];
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || dilation == 0
            || weights_offset
                .checked_add(3 * channels * channels)
                .is_none_or(|end| end > bias_offset)
            || slots
                .iter()
                .enumerate()
                .any(|(i, slot)| slots[..i].contains(slot))
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(output_gradient)? < output_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA long convolution backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, input_len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (input_values, parameter_values, output_values, input_gradient_values) =
                get_four_f32_slots(
                    &mut self.f32_slots,
                    input,
                    parameters,
                    output_gradient,
                    input_gradient,
                )?;
            unsafe {
                module.lsttn_long_conv_pool_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(input_len as u32),
                    input_values,
                    parameter_values,
                    output_values,
                    input_gradient_values,
                    weights_offset as u32,
                    bias_offset as u32,
                    times as u32,
                    convolution_times as u32,
                    output_times as u32,
                    nodes as u32,
                    channels as u32,
                    dilation as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide long convolution input backward: {error}"
                ))
            })?;
        }
        let (input_values, parameter_values, output_values, parameter_gradient_values) =
            get_four_f32_slots(
                &mut self.f32_slots,
                input,
                parameters,
                output_gradient,
                parameter_gradient,
            )?;
        unsafe {
            module.lsttn_long_conv_pool_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                input_values,
                parameter_values,
                output_values,
                parameter_gradient_values,
                weights_offset as u32,
                bias_offset as u32,
                batches as u32,
                times as u32,
                convolution_times as u32,
                output_times as u32,
                nodes as u32,
                channels as u32,
                dilation as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide long convolution parameter backward: {error}"
            ))
        })
    }

    #[cfg(any())]
    #[allow(clippy::too_many_arguments)]
    pub fn masked_patch_reconstruction_loss_backward_f32_unused(
        &mut self,
        decoded: usize,
        target: usize,
        masked_patch_indices: usize,
        parameters: usize,
        decoder_offset: usize,
        decoder_bias_offset: usize,
        context_gradient: usize,
        parameter_gradient: usize,
        loss: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        visible: usize,
        masked: usize,
        patch_width: usize,
        hidden: usize,
        masked_zero: f32,
        target_scale: f32,
    ) -> Result<()> {
        let token_len = checked_product(&[
            batches,
            nodes,
            visible.checked_add(masked).ok_or_else(|| {
                AcceleratorError::InvalidArgument("CUDA masked token count overflows".to_string())
            })?,
            hidden,
        ])?;
        let target_len = checked_product(&[batches, times, nodes, channels])?;
        let parameter_len = decoder_bias_offset
            .checked_add(patch_width)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked decoder parameter range overflows".to_string(),
                )
            })?;
        let slots = [
            decoded,
            target,
            parameters,
            context_gradient,
            parameter_gradient,
            loss,
        ];
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || visible == 0
            || masked == 0
            || patch_width == 0
            || hidden == 0
            || !times.is_multiple_of(patch_width)
            || decoder_bias_offset < decoder_offset + patch_width * hidden
            || !masked_zero.is_finite()
            || !target_scale.is_finite()
            || slots
                .iter()
                .enumerate()
                .any(|(i, slot)| slots[..i].contains(slot))
            || self.capacity_f32(decoded)? < token_len
            || self.capacity_f32(target)? < target_len
            || self
                .u32_capacities
                .get(masked_patch_indices)
                .copied()
                .unwrap_or(0)
                < masked
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA masked reconstruction slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(context_gradient, token_len)?;
        self.reserve_f32(loss, 2)?;
        let masked_values = self.u32_slots[masked_patch_indices]
            .as_ref()
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked patch indices are not allocated".to_string(),
                )
            })?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (decoded_values, target_values, parameter_values, loss_values) =
                get_four_f32_slots(&mut self.f32_slots, decoded, target, parameters, loss)?;
            unsafe {
                module.masked_patch_reconstruction_loss_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(2),
                    decoded_values,
                    target_values,
                    masked_values,
                    parameter_values,
                    loss_values,
                    decoder_offset as u32,
                    decoder_bias_offset as u32,
                    batches as u32,
                    times as u32,
                    nodes as u32,
                    channels as u32,
                    visible as u32,
                    masked as u32,
                    patch_width as u32,
                    hidden as u32,
                    masked_zero,
                    target_scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide masked reconstruction loss: {error}"
                ))
            })?;
        }
        {
            let (decoded_values, target_values, parameter_values, loss_values, context_values) =
                get_four_refs_one_mut_f32_slots(
                    &mut self.f32_slots,
                    decoded,
                    target,
                    parameters,
                    loss,
                    context_gradient,
                )?;
            unsafe {
                module.masked_patch_reconstruction_context_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(token_len as u32),
                    decoded_values,
                    target_values,
                    masked_values,
                    parameter_values,
                    loss_values,
                    context_values,
                    decoder_offset as u32,
                    decoder_bias_offset as u32,
                    times as u32,
                    nodes as u32,
                    channels as u32,
                    visible as u32,
                    masked as u32,
                    patch_width as u32,
                    hidden as u32,
                    masked_zero,
                    target_scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide masked reconstruction context backward: {error}"
                ))
            })?;
        }
        let (decoded_values, target_values, parameter_values, loss_values, gradient_values) =
            get_four_refs_one_mut_f32_slots(
                &mut self.f32_slots,
                decoded,
                target,
                parameters,
                loss,
                parameter_gradient,
            )?;
        unsafe {
            module.masked_patch_reconstruction_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                decoded_values,
                target_values,
                masked_values,
                parameter_values,
                loss_values,
                gradient_values,
                decoder_offset as u32,
                decoder_bias_offset as u32,
                batches as u32,
                times as u32,
                nodes as u32,
                channels as u32,
                visible as u32,
                masked as u32,
                patch_width as u32,
                hidden as u32,
                masked_zero,
                target_scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide masked reconstruction parameter backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv2_parameter_slice_backward_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output_gradient: usize,
        input_gradient: usize,
        parameter_gradient: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        input_channels: usize,
        output_channels: usize,
        dilation: usize,
    ) -> Result<()> {
        if batches == 0
            || nodes == 0
            || input_channels == 0
            || output_channels == 0
            || dilation == 0
            || times <= dilation
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA causal-convolution backward dimensions".to_string(),
            ));
        }
        let output_times = times - dilation;
        let input_len = checked_product(&[batches, times, nodes, input_channels])?;
        let output_len = checked_product(&[batches, output_times, nodes, output_channels])?;
        let parameter_len = bias_offset.checked_add(output_channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA causal-convolution parameter range overflows".to_string(),
            )
        })?;
        let slots = [
            input,
            parameters,
            output_gradient,
            input_gradient,
            parameter_gradient,
        ];
        if weights_offset
            .checked_add(2 * input_channels * output_channels)
            .is_none_or(|end| end > bias_offset)
            || slots
                .iter()
                .enumerate()
                .any(|(index, slot)| slots[..index].contains(slot))
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(output_gradient)? < output_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide causal-convolution backward slots or parameter slice"
                    .to_string(),
            ));
        }
        self.reserve_f32(input_gradient, input_len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (parameter_values, gradient_values, input_gradient_values) = get_three_f32_slots(
                &mut self.f32_slots,
                parameters,
                output_gradient,
                input_gradient,
            )?;
            unsafe {
                module.causal_conv2_parameter_slice_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(input_len as u32),
                    parameter_values,
                    gradient_values,
                    input_gradient_values,
                    weights_offset as u32,
                    times as u32,
                    nodes as u32,
                    input_channels as u32,
                    output_channels as u32,
                    dilation as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide causal-convolution input backward: {error}"
                ))
            })?;
        }
        let (input_values, gradient_values, parameter_gradient_values) = get_three_f32_slots(
            &mut self.f32_slots,
            input,
            output_gradient,
            parameter_gradient,
        )?;
        unsafe {
            module.causal_conv2_parameter_slice_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                input_values,
                gradient_values,
                parameter_gradient_values,
                weights_offset as u32,
                bias_offset as u32,
                batches as u32,
                output_times as u32,
                nodes as u32,
                input_channels as u32,
                output_channels as u32,
                dilation as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide causal-convolution parameter backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lsttn_short_input_projection_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output: usize,
        batches: usize,
        lookback: usize,
        nodes: usize,
        input_channels: usize,
        recent_window: usize,
        hidden: usize,
        phase_offset: usize,
        periodicity: usize,
    ) -> Result<usize> {
        if batches == 0
            || lookback == 0
            || nodes == 0
            || input_channels == 0
            || recent_window == 0
            || hidden == 0
            || periodicity == 0
            || recent_window > lookback
            || input == parameters
            || input == output
            || parameters == output
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA LSTTN short input projection dimensions".to_string(),
            ));
        }
        let padded_times = (recent_window + 1).max(13);
        let left_padding = padded_times - recent_window;
        let input_len = checked_product(&[batches, lookback, nodes, input_channels])?;
        let output_len = checked_product(&[batches, padded_times, nodes, hidden])?;
        let parameter_len = bias_offset.checked_add(hidden).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA short input projection parameter range overflows".to_string(),
            )
        })?;
        if weights_offset
            .checked_add(2 * hidden)
            .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA short input projection parameter slice".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.lsttn_short_input_projection_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                parameter_values,
                output_values,
                weights_offset as u32,
                bias_offset as u32,
                lookback as u32,
                nodes as u32,
                input_channels as u32,
                recent_window as u32,
                hidden as u32,
                left_padding as u32,
                phase_offset as u32,
                periodicity as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide short input projection: {error}"
            ))
        })?;
        Ok(padded_times)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn patch_embedding_parameter_slice_backward_f32(
        &mut self,
        input: usize,
        output_gradient: usize,
        parameter_gradient: usize,
        weights_offset: usize,
        bias_offset: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        patch_width: usize,
        hidden: usize,
    ) -> Result<()> {
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || patch_width == 0
            || hidden == 0
            || !times.is_multiple_of(patch_width)
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-embedding backward dimensions".to_string(),
            ));
        }
        let patches = times / patch_width;
        let input_len = checked_product(&[batches, times, nodes, channels])?;
        let output_len = checked_product(&[batches, patches, nodes, hidden])?;
        let parameter_len = bias_offset.checked_add(hidden).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA patch-embedding parameter range overflows".to_string(),
            )
        })?;
        if weights_offset
            .checked_add(patch_width * hidden)
            .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(output_gradient)? < output_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
            || input == output_gradient
            || input == parameter_gradient
            || output_gradient == parameter_gradient
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-embedding backward slots or parameter slice".to_string(),
            ));
        }
        let (input_values, gradient_values, parameter_gradient_values) = get_three_f32_slots(
            &mut self.f32_slots,
            input,
            output_gradient,
            parameter_gradient,
        )?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.patch_embedding_parameter_slice_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                input_values,
                gradient_values,
                parameter_gradient_values,
                weights_offset as u32,
                bias_offset as u32,
                batches as u32,
                patches as u32,
                nodes as u32,
                channels as u32,
                patch_width as u32,
                hidden as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch-embedding backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather_patch_tokens_backward_f32(
        &mut self,
        selected_gradient: usize,
        patch_indices: usize,
        full_gradient: usize,
        batches: usize,
        nodes: usize,
        patches: usize,
        selected: usize,
        hidden: usize,
    ) -> Result<()> {
        let selected_len = checked_product(&[batches, nodes, selected, hidden])?;
        let full_len = checked_product(&[batches, nodes, patches, hidden])?;
        if batches == 0
            || nodes == 0
            || patches == 0
            || selected == 0
            || hidden == 0
            || selected_gradient == full_gradient
            || self.capacity_f32(selected_gradient)? < selected_len
            || self.u32_capacities.get(patch_indices).copied().unwrap_or(0) < selected
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-gather backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(full_gradient, full_len)?;
        let indices = self.u32_slots[patch_indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {patch_indices} has not been allocated"
            ))
        })?;
        let (selected_values, full_values) =
            get_two_f32_slots(&mut self.f32_slots, selected_gradient, full_gradient)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.gather_patch_tokens_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(full_len as u32),
                selected_values,
                indices,
                full_values,
                nodes as u32,
                patches as u32,
                selected as u32,
                hidden as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch gather backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_patch_positions_backward_f32(
        &mut self,
        output_gradient: usize,
        input_gradient: usize,
        parameter_gradient: usize,
        positions_offset: usize,
        batches: usize,
        patches: usize,
        nodes: usize,
        hidden: usize,
        scale: f32,
    ) -> Result<()> {
        let len = checked_product(&[batches, patches, nodes, hidden])?;
        let parameter_len = positions_offset
            .checked_add(patches * hidden)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA patch-position parameter range overflows".to_string(),
                )
            })?;
        if batches == 0
            || patches == 0
            || nodes == 0
            || hidden == 0
            || !scale.is_finite()
            || output_gradient == input_gradient
            || output_gradient == parameter_gradient
            || input_gradient == parameter_gradient
            || self.capacity_f32(output_gradient)? < len
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-position backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (output_values, input_values) =
                get_two_f32_slots(&mut self.f32_slots, output_gradient, input_gradient)?;
            unsafe {
                module.patch_positions_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    output_values,
                    input_values,
                    scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide patch-position input backward: {error}"
                ))
            })?;
        }
        let (output_values, parameter_values) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, parameter_gradient)?;
        unsafe {
            module.patch_positions_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                output_values,
                parameter_values,
                positions_offset as u32,
                batches as u32,
                patches as u32,
                nodes as u32,
                hidden as u32,
                scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch-position parameter backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_backward_f32(
        &mut self,
        query: usize,
        key: usize,
        value: usize,
        output_gradient: usize,
        query_gradient: usize,
        key_gradient: usize,
        value_gradient: usize,
        sequences: usize,
        tokens: usize,
        heads: usize,
        head_width: usize,
        causal: bool,
    ) -> Result<()> {
        let len = checked_product(&[sequences, tokens, heads, head_width])?;
        let slots = [
            query,
            key,
            value,
            output_gradient,
            query_gradient,
            key_gradient,
            value_gradient,
        ];
        if sequences == 0
            || tokens == 0
            || heads == 0
            || head_width == 0
            || slots
                .iter()
                .enumerate()
                .any(|(index, slot)| slots[..index].contains(slot))
            || [query, key, value, output_gradient]
                .iter()
                .any(|slot| self.capacity_f32(*slot).unwrap_or(0) < len)
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide attention-backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(query_gradient, len)?;
        self.reserve_f32(key_gradient, len)?;
        self.reserve_f32(value_gradient, len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (query_values, key_values, value_values, gradient_values, query_gradient_values) =
                get_four_refs_one_mut_f32_slots(
                    &mut self.f32_slots,
                    query,
                    key,
                    value,
                    output_gradient,
                    query_gradient,
                )?;
            unsafe {
                module.attention_query_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    query_values,
                    key_values,
                    value_values,
                    gradient_values,
                    query_gradient_values,
                    tokens as u32,
                    heads as u32,
                    head_width as u32,
                    causal as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide attention query backward: {error}"
                ))
            })?;
        }
        {
            let (query_values, key_values, value_values, gradient_values, key_gradient_values) =
                get_four_refs_one_mut_f32_slots(
                    &mut self.f32_slots,
                    query,
                    key,
                    value,
                    output_gradient,
                    key_gradient,
                )?;
            unsafe {
                module.attention_key_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    query_values,
                    key_values,
                    value_values,
                    gradient_values,
                    key_gradient_values,
                    tokens as u32,
                    heads as u32,
                    head_width as u32,
                    causal as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide attention key backward: {error}"
                ))
            })?;
        }
        let (query_values, key_values, gradient_values, value_gradient_values) =
            get_four_f32_slots(
                &mut self.f32_slots,
                query,
                key,
                output_gradient,
                value_gradient,
            )?;
        unsafe {
            module.attention_value_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                query_values,
                key_values,
                gradient_values,
                value_gradient_values,
                tokens as u32,
                heads as u32,
                head_width as u32,
                causal as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide attention value backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_parameter_slice_backward_f32(
        &mut self,
        values: usize,
        parameters: usize,
        gamma_offset: usize,
        beta_offset: usize,
        output_gradient: usize,
        input_gradient: usize,
        parameter_gradient: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let len = checked_product(&[rows, width])?;
        let parameter_len = beta_offset.checked_add(width).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA layer-normalization parameter range overflows".to_string(),
            )
        })?;
        if rows == 0
            || width == 0
            || gamma_offset
                .checked_add(width)
                .is_none_or(|end| end > beta_offset)
            || self.capacity_f32(values)? < len
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(output_gradient)? < len
            || self.capacity_f32(parameter_gradient)? < parameter_len
            || [
                values,
                parameters,
                output_gradient,
                input_gradient,
                parameter_gradient,
            ]
            .iter()
            .enumerate()
            .any(|(index, slot)| {
                [
                    values,
                    parameters,
                    output_gradient,
                    input_gradient,
                    parameter_gradient,
                ][..index]
                    .contains(slot)
            })
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide layer-normalization backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (value_values, parameter_values, output_gradient_values, input_gradient_values) =
                get_four_f32_slots(
                    &mut self.f32_slots,
                    values,
                    parameters,
                    output_gradient,
                    input_gradient,
                )?;
            unsafe {
                module.layer_norm_parameter_slice_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    value_values,
                    parameter_values,
                    output_gradient_values,
                    input_gradient_values,
                    gamma_offset as u32,
                    width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide layer-normalization input backward: {error}"
                ))
            })?;
        }
        let (value_values, output_gradient_values, parameter_gradient_values) =
            get_three_f32_slots(
                &mut self.f32_slots,
                values,
                output_gradient,
                parameter_gradient,
            )?;
        unsafe {
            module.layer_norm_parameter_slice_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                value_values,
                output_gradient_values,
                parameter_gradient_values,
                gamma_offset as u32,
                beta_offset as u32,
                rows as u32,
                width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide layer-normalization parameter backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_channels_parameter_slice_f32(
        &mut self,
        values: usize,
        parameters: usize,
        gamma_offset: usize,
        beta_offset: usize,
        statistics: usize,
        output: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
    ) -> Result<()> {
        let rows = batches
            .checked_mul(times)
            .and_then(|v| v.checked_mul(nodes))
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA batch-normalization dimensions overflow".to_string(),
                )
            })?;
        let len = rows.checked_mul(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA batch-normalization dimensions overflow".to_string(),
            )
        })?;
        let parameter_len = beta_offset.checked_add(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA batch-normalization parameter range overflows".to_string(),
            )
        })?;
        if rows == 0
            || channels == 0
            || gamma_offset
                .checked_add(channels)
                .is_none_or(|end| end > beta_offset)
            || values == parameters
            || values == statistics
            || values == output
            || parameters == statistics
            || parameters == output
            || statistics == output
            || self.capacity_f32(values)? < len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide batch-normalization slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(statistics, 2 * channels)?;
        self.reserve_f32(output, len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (value_values, stats_values) =
                get_two_f32_slots(&mut self.f32_slots, values, statistics)?;
            unsafe {
                module.batch_norm_channel_stats_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(channels as u32),
                    value_values,
                    stats_values,
                    rows as u32,
                    channels as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide batch-normalization statistics: {error}"
                ))
            })?;
        }
        let (value_values, parameter_values, statistics_values, output_values) =
            get_four_f32_slots(&mut self.f32_slots, values, parameters, statistics, output)?;
        unsafe {
            module.batch_norm_channel_apply_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                value_values,
                parameter_values,
                statistics_values,
                output_values,
                gamma_offset as u32,
                beta_offset as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide batch-normalization apply: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_channels_parameter_slice_backward_f32(
        &mut self,
        values: usize,
        parameters: usize,
        gamma_offset: usize,
        beta_offset: usize,
        statistics: usize,
        output_gradient: usize,
        input_gradient: usize,
        parameter_gradient: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
    ) -> Result<()> {
        let rows = batches
            .checked_mul(times)
            .and_then(|v| v.checked_mul(nodes))
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA batch-normalization dimensions overflow".to_string(),
                )
            })?;
        let len = rows.checked_mul(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA batch-normalization dimensions overflow".to_string(),
            )
        })?;
        let parameter_len = beta_offset.checked_add(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA batch-normalization parameter range overflows".to_string(),
            )
        })?;
        let slots = [
            values,
            parameters,
            statistics,
            output_gradient,
            input_gradient,
            parameter_gradient,
        ];
        if rows == 0
            || channels == 0
            || gamma_offset
                .checked_add(channels)
                .is_none_or(|end| end > beta_offset)
            || slots
                .iter()
                .enumerate()
                .any(|(index, slot)| slots[..index].contains(slot))
            || self.capacity_f32(values)? < len
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(statistics)? < 2 * channels
            || self.capacity_f32(output_gradient)? < len
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide batch-normalization backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (
                value_values,
                parameter_values,
                statistics_values,
                gradient_values,
                input_gradient_values,
            ) = get_four_refs_one_mut_f32_slots(
                &mut self.f32_slots,
                values,
                parameters,
                statistics,
                output_gradient,
                input_gradient,
            )?;
            unsafe {
                module.batch_norm_channel_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    value_values,
                    parameter_values,
                    statistics_values,
                    gradient_values,
                    input_gradient_values,
                    gamma_offset as u32,
                    rows as u32,
                    channels as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide batch-normalization input backward: {error}"
                ))
            })?;
        }
        let (value_values, statistics_values, gradient_values, parameter_gradient_values) =
            get_four_f32_slots(
                &mut self.f32_slots,
                values,
                statistics,
                output_gradient,
                parameter_gradient,
            )?;
        unsafe {
            module.batch_norm_channel_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                value_values,
                statistics_values,
                gradient_values,
                parameter_gradient_values,
                gamma_offset as u32,
                beta_offset as u32,
                rows as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide batch-normalization parameter backward: {error}"
            ))
        })
    }

    pub fn clip_gradient_l2_f32(
        &mut self,
        gradients: usize,
        scratch: usize,
        len: usize,
        maximum_norm: f32,
    ) -> Result<()> {
        if len == 0
            || gradients == scratch
            || !maximum_norm.is_finite()
            || maximum_norm <= 0.0
            || self.capacity_f32(gradients)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA gradient-clipping slots or arguments".to_string(),
            ));
        }
        self.reserve_f32(scratch, 1)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (gradient_values, scratch_values) =
                get_two_f32_slots(&mut self.f32_slots, gradients, scratch)?;
            unsafe {
                module.gradient_l2_norm_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(1),
                    gradient_values,
                    scratch_values,
                    len as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide gradient norm: {error}"
                ))
            })?;
        }
        let (scratch_values, gradient_values) =
            get_two_f32_slots(&mut self.f32_slots, scratch, gradients)?;
        unsafe {
            module.clip_gradient_l2_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                scratch_values,
                gradient_values,
                maximum_norm,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide gradient clipping: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assemble_masked_decoder_tokens_f32(
        &mut self,
        visible_tokens: usize,
        masked_patch_indices: usize,
        parameters: usize,
        mask_token_offset: usize,
        positions_offset: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        visible: usize,
        masked: usize,
        hidden: usize,
        position_count: usize,
        scale: f32,
    ) -> Result<()> {
        if batches == 0
            || nodes == 0
            || visible == 0
            || masked == 0
            || hidden == 0
            || position_count == 0
            || !scale.is_finite()
            || visible_tokens == parameters
            || visible_tokens == output
            || parameters == output
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide masked-decoder token slots or dimensions".to_string(),
            ));
        }
        let visible_len = checked_product(&[batches, nodes, visible, hidden])?;
        let output_len = checked_product(&[batches, nodes, visible + masked, hidden])?;
        let parameter_len = positions_offset
            .checked_add(position_count * hidden)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked-decoder parameter range overflows".to_string(),
                )
            })?;
        if mask_token_offset
            .checked_add(hidden)
            .is_none_or(|end| end > positions_offset)
            || self.capacity_f32(visible_tokens)? < visible_len
            || self.capacity_f32(parameters)? < parameter_len
            || self
                .u32_capacities
                .get(masked_patch_indices)
                .copied()
                .unwrap_or(0)
                < masked
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA masked-decoder parameter slice".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let indices = self.u32_slots[masked_patch_indices]
            .as_ref()
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(format!(
                    "CUDA u32 tensor slot {masked_patch_indices} has not been allocated"
                ))
            })?;
        let (visible_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, visible_tokens, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.assemble_masked_decoder_tokens_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                visible_values,
                indices,
                parameter_values,
                output_values,
                mask_token_offset as u32,
                positions_offset as u32,
                nodes as u32,
                visible as u32,
                masked as u32,
                hidden as u32,
                position_count as u32,
                scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide masked-decoder token assembly: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assemble_masked_decoder_tokens_backward_f32(
        &mut self,
        output_gradient: usize,
        masked_patch_indices: usize,
        parameter_gradient: usize,
        mask_token_offset: usize,
        positions_offset: usize,
        visible_gradient: usize,
        batches: usize,
        nodes: usize,
        visible: usize,
        masked: usize,
        hidden: usize,
        position_count: usize,
        scale: f32,
    ) -> Result<()> {
        let output_len = checked_product(&[batches, nodes, visible + masked, hidden])?;
        let visible_len = checked_product(&[batches, nodes, visible, hidden])?;
        let parameter_len = positions_offset
            .checked_add(position_count * hidden)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked-decoder gradient range overflows".to_string(),
                )
            })?;
        if batches == 0
            || nodes == 0
            || visible == 0
            || masked == 0
            || hidden == 0
            || position_count == 0
            || !scale.is_finite()
            || output_gradient == parameter_gradient
            || output_gradient == visible_gradient
            || parameter_gradient == visible_gradient
            || mask_token_offset
                .checked_add(hidden)
                .is_none_or(|end| end > positions_offset)
            || self.capacity_f32(output_gradient)? < output_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
            || self
                .u32_capacities
                .get(masked_patch_indices)
                .copied()
                .unwrap_or(0)
                < masked
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA masked-decoder backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(visible_gradient, visible_len)?;
        let indices = self.u32_slots[masked_patch_indices]
            .as_ref()
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked-decoder indices are not allocated".to_string(),
                )
            })?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (output_values, visible_values) =
                get_two_f32_slots(&mut self.f32_slots, output_gradient, visible_gradient)?;
            unsafe {
                module.assemble_masked_decoder_tokens_visible_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(visible_len as u32),
                    output_values,
                    visible_values,
                    nodes as u32,
                    visible as u32,
                    masked as u32,
                    hidden as u32,
                    scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide masked-decoder visible backward: {error}"
                ))
            })?;
        }
        let (output_values, parameter_values) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, parameter_gradient)?;
        unsafe {
            module.assemble_masked_decoder_tokens_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                output_values,
                indices,
                parameter_values,
                mask_token_offset as u32,
                positions_offset as u32,
                batches as u32,
                nodes as u32,
                visible as u32,
                masked as u32,
                hidden as u32,
                position_count as u32,
                scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide masked-decoder parameter backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn masked_patch_reconstruction_loss_backward_f32(
        &mut self,
        decoded: usize,
        target: usize,
        masked_patch_indices: usize,
        parameters: usize,
        decoder_offset: usize,
        decoder_bias_offset: usize,
        context_gradient: usize,
        parameter_gradient: usize,
        loss: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        visible: usize,
        masked: usize,
        patch_width: usize,
        hidden: usize,
        masked_zero: f32,
        target_scale: f32,
    ) -> Result<()> {
        let decoded_len = checked_product(&[batches, nodes, visible + masked, hidden])?;
        let target_len = checked_product(&[batches, times, nodes, channels])?;
        let parameter_len = decoder_bias_offset
            .checked_add(patch_width)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked-reconstruction parameter range overflows".to_string(),
                )
            })?;
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || visible == 0
            || masked == 0
            || patch_width == 0
            || hidden == 0
            || !masked_zero.is_finite()
            || !target_scale.is_finite()
            || target_scale <= 0.0
            || decoder_offset
                .checked_add(patch_width * hidden)
                .is_none_or(|end| end > decoder_bias_offset)
            || decoded == target
            || decoded == parameters
            || decoded == context_gradient
            || decoded == parameter_gradient
            || target == parameters
            || target == context_gradient
            || target == parameter_gradient
            || parameters == context_gradient
            || parameters == parameter_gradient
            || context_gradient == parameter_gradient
            || self.capacity_f32(decoded)? < decoded_len
            || self.capacity_f32(target)? < target_len
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
            || self
                .u32_capacities
                .get(masked_patch_indices)
                .copied()
                .unwrap_or(0)
                < masked
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA masked-reconstruction slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(context_gradient, decoded_len)?;
        self.reserve_f32(loss, 2)?;
        let indices = self.u32_slots[masked_patch_indices]
            .as_ref()
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA masked-reconstruction indices are not allocated".to_string(),
                )
            })?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (decoded_values, target_values, parameter_values, loss_values) =
                get_four_f32_slots(&mut self.f32_slots, decoded, target, parameters, loss)?;
            unsafe {
                module.masked_patch_reconstruction_loss_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(2),
                    decoded_values,
                    target_values,
                    indices,
                    parameter_values,
                    loss_values,
                    decoder_offset as u32,
                    decoder_bias_offset as u32,
                    batches as u32,
                    times as u32,
                    nodes as u32,
                    channels as u32,
                    visible as u32,
                    masked as u32,
                    patch_width as u32,
                    hidden as u32,
                    masked_zero,
                    target_scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide masked reconstruction loss: {error}"
                ))
            })?;
        }
        {
            let (decoded_values, target_values, parameter_values, context_values) =
                get_four_f32_slots(
                    &mut self.f32_slots,
                    decoded,
                    target,
                    parameters,
                    context_gradient,
                )?;
            unsafe {
                module.masked_patch_reconstruction_context_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(decoded_len as u32),
                    decoded_values,
                    target_values,
                    indices,
                    parameter_values,
                    context_values,
                    decoder_offset as u32,
                    decoder_bias_offset as u32,
                    batches as u32,
                    times as u32,
                    nodes as u32,
                    channels as u32,
                    visible as u32,
                    masked as u32,
                    patch_width as u32,
                    hidden as u32,
                    masked_zero,
                    target_scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide masked reconstruction context backward: {error}"
                ))
            })?;
        }
        let (decoded_values, target_values, parameter_values, gradient_values) =
            get_four_f32_slots(
                &mut self.f32_slots,
                decoded,
                target,
                parameters,
                parameter_gradient,
            )?;
        unsafe {
            module.masked_patch_reconstruction_parameter_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                decoded_values,
                target_values,
                indices,
                parameter_values,
                gradient_values,
                decoder_offset as u32,
                decoder_bias_offset as u32,
                batches as u32,
                times as u32,
                nodes as u32,
                channels as u32,
                visible as u32,
                masked as u32,
                patch_width as u32,
                hidden as u32,
                masked_zero,
                target_scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide masked reconstruction parameter backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn masked_inverse_scale_mae_loss_backward_f32(
        &mut self,
        prediction: usize,
        target: usize,
        prediction_gradient: usize,
        loss: usize,
        len: usize,
        normalized_zero: f32,
        target_scale: f32,
    ) -> Result<()> {
        if len == 0
            || !normalized_zero.is_finite()
            || !target_scale.is_finite()
            || target_scale <= 0.0
            || prediction == target
            || prediction == prediction_gradient
            || target == prediction_gradient
            || prediction == loss
            || target == loss
            || prediction_gradient == loss
            || self.capacity_f32(prediction)? < len
            || self.capacity_f32(target)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA masked inverse-scale MAE slots or arguments".to_string(),
            ));
        }
        self.reserve_f32(prediction_gradient, len)?;
        self.reserve_f32(loss, 2)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (prediction_values, target_values, loss_values) =
                get_three_f32_slots(&mut self.f32_slots, prediction, target, loss)?;
            unsafe {
                module.masked_inverse_scale_mae_loss_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(2),
                    prediction_values,
                    target_values,
                    loss_values,
                    len as u32,
                    normalized_zero,
                    target_scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide masked inverse-scale MAE loss: {error}"
                ))
            })?;
        }
        let (prediction_values, target_values, gradient_values) =
            get_three_f32_slots(&mut self.f32_slots, prediction, target, prediction_gradient)?;
        unsafe {
            module.masked_inverse_scale_mae_gradient_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                prediction_values,
                target_values,
                gradient_values,
                len as u32,
                normalized_zero,
                target_scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide masked inverse-scale MAE gradient: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_inverse_scale_mae_loss_backward_f32(
        &mut self,
        prediction: usize,
        target: usize,
        weight: usize,
        prediction_gradient: usize,
        loss: usize,
        len: usize,
        target_scale: f32,
    ) -> Result<()> {
        let slots = [prediction, target, weight, prediction_gradient, loss];
        if len == 0
            || !target_scale.is_finite()
            || target_scale <= 0.0
            || slots
                .iter()
                .enumerate()
                .any(|(index, slot)| slots[..index].contains(slot))
            || self.capacity_f32(prediction)? < len
            || self.capacity_f32(target)? < len
            || self.capacity_f32(weight)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA weighted MAE slots or arguments".to_string(),
            ));
        }
        self.reserve_f32(prediction_gradient, len)?;
        self.reserve_f32(loss, 2)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (prediction_values, target_values, weight_values, loss_values) =
                get_four_f32_slots(&mut self.f32_slots, prediction, target, weight, loss)?;
            unsafe {
                module.weighted_inverse_scale_mae_loss_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(2),
                    prediction_values,
                    target_values,
                    weight_values,
                    loss_values,
                    len as u32,
                    target_scale,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide weighted inverse-scale MAE loss: {error}"
                ))
            })?;
        }
        let (prediction_values, target_values, weight_values, loss_values, gradient_values) =
            get_five_f32_slots(
                &mut self.f32_slots,
                prediction,
                target,
                weight,
                loss,
                prediction_gradient,
            )?;
        unsafe {
            module.weighted_inverse_scale_mae_gradient_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                prediction_values,
                target_values,
                weight_values,
                loss_values,
                gradient_values,
                len as u32,
                target_scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide weighted inverse-scale MAE gradient: {error}"
            ))
        })
    }

    pub fn gated_tanh_sigmoid_backward_f32(
        &mut self,
        filter: usize,
        gate: usize,
        output_gradient: usize,
        filter_gradient: usize,
        gate_gradient: usize,
        len: usize,
    ) -> Result<()> {
        let slots = [
            filter,
            gate,
            output_gradient,
            filter_gradient,
            gate_gradient,
        ];
        if len == 0
            || slots
                .iter()
                .enumerate()
                .any(|(index, slot)| slots[..index].contains(slot))
            || self.capacity_f32(filter)? < len
            || self.capacity_f32(gate)? < len
            || self.capacity_f32(output_gradient)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA gated-activation-backward slots or length".to_string(),
            ));
        }
        self.reserve_f32(filter_gradient, len)?;
        self.reserve_f32(gate_gradient, len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (filter_values, gate_values, output_gradient_values, filter_gradient_values) =
                get_four_f32_slots(
                    &mut self.f32_slots,
                    filter,
                    gate,
                    output_gradient,
                    filter_gradient,
                )?;
            unsafe {
                module.gated_tanh_sigmoid_filter_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    filter_values,
                    gate_values,
                    output_gradient_values,
                    filter_gradient_values,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide gated filter backward: {error}"
                ))
            })?;
        }
        let (filter_values, gate_values, output_gradient_values, gate_gradient_values) =
            get_four_f32_slots(
                &mut self.f32_slots,
                filter,
                gate,
                output_gradient,
                gate_gradient,
            )?;
        unsafe {
            module.gated_tanh_sigmoid_gate_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                filter_values,
                gate_values,
                output_gradient_values,
                gate_gradient_values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide gated gate backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn csr_adaptive_logits_parameter_slice_f32(
        &mut self,
        indptr: usize,
        indices: usize,
        parameters: usize,
        source_offset: usize,
        target_offset: usize,
        logits: usize,
        nodes: usize,
        edges: usize,
        latent: usize,
    ) -> Result<()> {
        let parameter_len = target_offset.checked_add(nodes * latent).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA adaptive-logit parameter range overflows".to_string(),
            )
        })?;
        if nodes == 0
            || edges == 0
            || latent == 0
            || parameters == logits
            || source_offset
                .checked_add(nodes * latent)
                .is_none_or(|end| end > target_offset)
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.u32_capacities.get(indices).copied().unwrap_or(0) < edges
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide adaptive CSR-logit slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(logits, edges)?;
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indptr} has not been allocated"
            ))
        })?;
        let indices_values = self.u32_slots[indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indices} has not been allocated"
            ))
        })?;
        let (parameter_values, logits_values) =
            get_two_f32_slots(&mut self.f32_slots, parameters, logits)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_adaptive_logits_f32(
                &self.stream,
                LaunchConfig::for_num_elems(edges as u32),
                indptr_values,
                indices_values,
                parameter_values,
                logits_values,
                source_offset as u32,
                target_offset as u32,
                latent as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide adaptive CSR logits: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn csr_adaptive_logits_parameter_slice_backward_f32_unused(
        &mut self,
        indptr: usize,
        indices: usize,
        parameters: usize,
        source_offset: usize,
        target_offset: usize,
        logits_gradient: usize,
        parameter_gradient: usize,
        nodes: usize,
        edges: usize,
        latent: usize,
    ) -> Result<()> {
        let parameter_len = target_offset.checked_add(nodes * latent).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA adaptive-logit parameter range overflows".to_string(),
            )
        })?;
        if nodes == 0
            || edges == 0
            || latent == 0
            || source_offset
                .checked_add(nodes * latent)
                .is_none_or(|end| end > target_offset)
            || parameters == logits_gradient
            || parameters == parameter_gradient
            || logits_gradient == parameter_gradient
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.u32_capacities.get(indices).copied().unwrap_or(0) < edges
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(logits_gradient)? < edges
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide adaptive CSR-logit backward slots or dimensions".to_string(),
            ));
        }
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indptr} has not been allocated"
            ))
        })?;
        let indices_values = self.u32_slots[indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indices} has not been allocated"
            ))
        })?;
        let (parameter_values, logits_gradient_values, parameter_gradient_values) =
            get_three_f32_slots(
                &mut self.f32_slots,
                parameters,
                logits_gradient,
                parameter_gradient,
            )?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_adaptive_logits_parameter_slice_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                indptr_values,
                indices_values,
                parameter_values,
                logits_gradient_values,
                parameter_gradient_values,
                source_offset as u32,
                target_offset as u32,
                nodes as u32,
                latent as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide adaptive CSR-logit backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn csr_adaptive_logits_parameter_slice_backward_f32(
        &mut self,
        indptr: usize,
        indices: usize,
        parameters: usize,
        source_offset: usize,
        target_offset: usize,
        logits_gradient: usize,
        parameter_gradient: usize,
        nodes: usize,
        edges: usize,
        latent: usize,
    ) -> Result<()> {
        let parameter_len = target_offset.checked_add(nodes * latent).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA adaptive-logit gradient range overflows".to_string(),
            )
        })?;
        if nodes == 0
            || edges == 0
            || latent == 0
            || parameters == parameter_gradient
            || parameters == logits_gradient
            || logits_gradient == parameter_gradient
            || source_offset
                .checked_add(nodes * latent)
                .is_none_or(|end| end > target_offset)
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.u32_capacities.get(indices).copied().unwrap_or(0) < edges
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(logits_gradient)? < edges
            || self.capacity_f32(parameter_gradient)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA adaptive-logit backward slots or dimensions".to_string(),
            ));
        }
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA adaptive-logit indptr is not allocated".to_string(),
            )
        })?;
        let index_values = self.u32_slots[indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA adaptive-logit indices are not allocated".to_string(),
            )
        })?;
        let (parameter_values, logits_values, gradient_values) = get_three_f32_slots(
            &mut self.f32_slots,
            parameters,
            logits_gradient,
            parameter_gradient,
        )?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_adaptive_logits_parameter_slice_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                indptr_values,
                index_values,
                parameter_values,
                logits_values,
                gradient_values,
                source_offset as u32,
                target_offset as u32,
                nodes as u32,
                latent as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide adaptive-logit backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lsttn_long_conv_pool_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        dilation: usize,
    ) -> Result<()> {
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || dilation == 0
            || weights_offset
                .checked_add(3 * channels * channels)
                .is_none_or(|end| end > bias_offset)
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA LSTTN long-convolution dimensions".to_string(),
            ));
        }
        let input_len = checked_product(&[batches, times, nodes, channels])?;
        let output_times = times.div_ceil(2).div_ceil(2);
        let output_len = checked_product(&[batches, output_times, nodes, channels])?;
        let parameter_len = bias_offset.checked_add(channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA LSTTN long-convolution parameter range overflows".to_string(),
            )
        })?;
        if input == parameters
            || input == output
            || parameters == output
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA LSTTN long-convolution slots".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.lsttn_long_conv_pool_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                parameter_values,
                output_values,
                weights_offset as u32,
                bias_offset as u32,
                times as u32,
                nodes as u32,
                channels as u32,
                dilation as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide LSTTN long convolution: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv2_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        input_channels: usize,
        output_channels: usize,
        dilation: usize,
    ) -> Result<()> {
        if batches == 0
            || nodes == 0
            || input_channels == 0
            || output_channels == 0
            || dilation == 0
            || times <= dilation
            || input == parameters
            || input == output
            || parameters == output
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide causal-convolution slots or dimensions".to_string(),
            ));
        }
        let output_times = times - dilation;
        let input_len = checked_product(&[batches, times, nodes, input_channels])?;
        let output_len = checked_product(&[batches, output_times, nodes, output_channels])?;
        let weights_len = 2usize
            .checked_mul(input_channels)
            .and_then(|value| value.checked_mul(output_channels))
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA causal-convolution dimensions overflow".to_string(),
                )
            })?;
        let parameter_len = bias_offset.checked_add(output_channels).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA causal-convolution parameter range overflows".to_string(),
            )
        })?;
        if weights_offset
            .checked_add(weights_len)
            .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA causal-convolution parameter slice".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.causal_conv2_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                parameter_values,
                output_values,
                weights_offset as u32,
                bias_offset as u32,
                output_times as u32,
                nodes as u32,
                input_channels as u32,
                output_channels as u32,
                dilation as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide causal convolution: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn affine_backward_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weight_offset: usize,
        bias_offset: usize,
        output_gradient: usize,
        input_gradient: usize,
        parameter_gradient: usize,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let input_len = checked_product(&[rows, input_width])?;
        let output_len = checked_product(&[rows, output_width])?;
        let parameter_len = bias_offset.checked_add(output_width).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA affine-backward parameter range overflows".to_string(),
            )
        })?;
        if rows == 0
            || input_width == 0
            || output_width == 0
            || weight_offset
                .checked_add(input_width * output_width)
                .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
            || self.capacity_f32(output_gradient)? < output_len
            || self.capacity_f32(parameter_gradient)? < parameter_len
            || [
                input,
                parameters,
                output_gradient,
                input_gradient,
                parameter_gradient,
            ]
            .iter()
            .enumerate()
            .any(|(index, slot)| {
                [
                    input,
                    parameters,
                    output_gradient,
                    input_gradient,
                    parameter_gradient,
                ][..index]
                    .contains(slot)
            })
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide affine-backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, input_len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (output_gradient_values, parameter_values, input_gradient_values) =
                get_three_f32_slots(
                    &mut self.f32_slots,
                    output_gradient,
                    parameters,
                    input_gradient,
                )?;
            unsafe {
                module.affine_parameter_slice_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(input_len as u32),
                    output_gradient_values,
                    parameter_values,
                    input_gradient_values,
                    weight_offset as u32,
                    input_width as u32,
                    output_width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide affine input backward: {error}"
                ))
            })?;
        }
        {
            let (input_values, output_gradient_values, parameter_gradient_values) =
                get_three_f32_slots(
                    &mut self.f32_slots,
                    input,
                    output_gradient,
                    parameter_gradient,
                )?;
            unsafe {
                module.affine_parameter_slice_weight_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(parameter_len as u32),
                    input_values,
                    output_gradient_values,
                    parameter_gradient_values,
                    weight_offset as u32,
                    rows as u32,
                    input_width as u32,
                    output_width as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide affine weight backward: {error}"
                ))
            })?;
        }
        let (output_gradient_values, parameter_gradient_values) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, parameter_gradient)?;
        unsafe {
            module.affine_parameter_slice_bias_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_len as u32),
                output_gradient_values,
                parameter_gradient_values,
                bias_offset as u32,
                rows as u32,
                output_width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide affine bias backward: {error}"
            ))
        })
    }

    pub fn node_major_horizons_to_output_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        horizons: usize,
    ) -> Result<()> {
        self.reorder_horizons_f32(input, output, batches, nodes, horizons, true)
    }

    pub fn output_to_node_major_horizons_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        horizons: usize,
    ) -> Result<()> {
        self.reorder_horizons_f32(input, output, batches, nodes, horizons, false)
    }

    fn reorder_horizons_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        horizons: usize,
        to_output: bool,
    ) -> Result<()> {
        let len = checked_product(&[batches, nodes, horizons])?;
        if batches == 0
            || nodes == 0
            || horizons == 0
            || input == output
            || self.capacity_f32(input)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA horizon-layout slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        if to_output {
            unsafe {
                module.node_major_horizons_to_output_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    input_values,
                    output_values,
                    nodes as u32,
                    horizons as u32,
                )
            }
        } else {
            unsafe {
                module.output_to_node_major_horizons_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    input_values,
                    output_values,
                    nodes as u32,
                    horizons as u32,
                )
            }
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide horizon-layout transform: {error}"
            ))
        })
    }

    pub fn accumulate_parameter_slice_f32(
        &mut self,
        source: usize,
        parameter_gradient: usize,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        let parameter_capacity = self.capacity_f32(parameter_gradient)?;
        if len == 0
            || source == parameter_gradient
            || self.capacity_f32(source)? < len
            || offset
                .checked_add(len)
                .is_none_or(|end| end > parameter_capacity)
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA parameter-gradient slice".to_string(),
            ));
        }
        let (source_values, gradient_values) =
            get_two_f32_slots(&mut self.f32_slots, source, parameter_gradient)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.accumulate_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(parameter_capacity as u32),
                source_values,
                gradient_values,
                offset as u32,
                len as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide parameter-gradient accumulation: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step_f32(
        &mut self,
        parameters: usize,
        first: usize,
        second: usize,
        gradients: usize,
        len: usize,
        step: u64,
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<()> {
        let slots = [parameters, first, second, gradients];
        if len == 0
            || step == 0
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !weight_decay.is_finite()
            || slots
                .iter()
                .enumerate()
                .any(|(index, slot)| slots[..index].contains(slot))
            || slots
                .iter()
                .any(|slot| self.capacity_f32(*slot).unwrap_or(0) < len)
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide AdamW slots or state".to_string(),
            ));
        }
        let (parameter_values, first_values, second_values, gradient_values) =
            get_three_mut_one_ref_f32_slots(
                &mut self.f32_slots,
                parameters,
                first,
                second,
                gradients,
            )?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.adamw_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                parameter_values,
                first_values,
                second_values,
                gradient_values,
                learning_rate,
                weight_decay,
                1.0 - 0.9_f32.powi(step as i32),
                1.0 - 0.999_f32.powi(step as i32),
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to launch cuda-oxide AdamW: {error}"))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_patch_positions_f32(
        &mut self,
        input: usize,
        parameters: usize,
        positions_offset: usize,
        output: usize,
        batches: usize,
        patches: usize,
        nodes: usize,
        hidden: usize,
        scale: f32,
    ) -> Result<()> {
        let len = checked_product(&[batches, patches, nodes, hidden])?;
        let parameter_len = positions_offset
            .checked_add(patches * hidden)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(
                    "CUDA patch-position parameter range overflows".to_string(),
                )
            })?;
        if batches == 0
            || patches == 0
            || nodes == 0
            || hidden == 0
            || !scale.is_finite()
            || input == parameters
            || input == output
            || parameters == output
            || self.capacity_f32(input)? < len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide patch-position slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.add_patch_positions_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                input_values,
                parameter_values,
                output_values,
                positions_offset as u32,
                patches as u32,
                hidden as u32,
                nodes as u32,
                scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch-position addition: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn patch_embedding_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output: usize,
        batches: usize,
        times: usize,
        nodes: usize,
        channels: usize,
        patch_width: usize,
        hidden: usize,
    ) -> Result<()> {
        if batches == 0
            || times == 0
            || nodes == 0
            || channels == 0
            || patch_width == 0
            || hidden == 0
            || !times.is_multiple_of(patch_width)
            || input == parameters
            || input == output
            || parameters == output
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide patch-embedding slots or dimensions".to_string(),
            ));
        }
        let patches = times / patch_width;
        let input_len = checked_product(&[batches, times, nodes, channels])?;
        let output_len = checked_product(&[batches, patches, nodes, hidden])?;
        let parameter_len = bias_offset.checked_add(hidden).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA patch-embedding parameter range overflows".to_string(),
            )
        })?;
        if weights_offset
            .checked_add(patch_width * hidden)
            .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-embedding parameter slice".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.patch_embedding_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                parameter_values,
                output_values,
                weights_offset as u32,
                bias_offset as u32,
                patches as u32,
                nodes as u32,
                channels as u32,
                patch_width as u32,
                hidden as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch embedding: {error}"
            ))
        })
    }

    pub fn patches_to_attention_sequences_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        patches: usize,
        nodes: usize,
        hidden: usize,
    ) -> Result<()> {
        self.reorder_patch_attention_f32(input, output, batches, patches, nodes, hidden, true)
    }

    pub fn attention_sequences_to_patches_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        patches: usize,
        nodes: usize,
        hidden: usize,
    ) -> Result<()> {
        self.reorder_patch_attention_f32(input, output, batches, patches, nodes, hidden, false)
    }

    fn reorder_patch_attention_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        patches: usize,
        nodes: usize,
        hidden: usize,
        to_attention: bool,
    ) -> Result<()> {
        let len = checked_product(&[batches, patches, nodes, hidden])?;
        if batches == 0
            || patches == 0
            || nodes == 0
            || hidden == 0
            || input == output
            || self.capacity_f32(input)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-attention layout slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        if to_attention {
            unsafe {
                module.patches_to_attention_sequences_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    input_values,
                    output_values,
                    patches as u32,
                    nodes as u32,
                    hidden as u32,
                )
            }
        } else {
            unsafe {
                module.attention_sequences_to_patches_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    input_values,
                    output_values,
                    patches as u32,
                    nodes as u32,
                    hidden as u32,
                )
            }
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch-attention layout transform: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn select_node_major_time_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        patches: usize,
        channels: usize,
        patch: usize,
    ) -> Result<()> {
        let input_len = checked_product(&[batches, nodes, patches, channels])?;
        let output_len = checked_product(&[batches, nodes, channels])?;
        if batches == 0
            || nodes == 0
            || patches == 0
            || channels == 0
            || patch >= patches
            || input == output
            || self.capacity_f32(input)? < input_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA node-major temporal selection arguments".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.select_node_major_time_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                output_values,
                nodes as u32,
                patches as u32,
                channels as u32,
                patch as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide node-major temporal selection: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather_patch_tokens_f32(
        &mut self,
        input: usize,
        patch_indices: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        patches: usize,
        selected: usize,
        hidden: usize,
    ) -> Result<()> {
        let input_len = checked_product(&[batches, nodes, patches, hidden])?;
        let output_len = checked_product(&[batches, nodes, selected, hidden])?;
        if batches == 0
            || nodes == 0
            || patches == 0
            || selected == 0
            || hidden == 0
            || input == output
            || self.capacity_f32(input)? < input_len
            || self.u32_capacities.get(patch_indices).copied().unwrap_or(0) < selected
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA patch-gather slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let indices = self.u32_slots[patch_indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {patch_indices} has not been allocated"
            ))
        })?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.gather_patch_tokens_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                indices,
                output_values,
                nodes as u32,
                patches as u32,
                selected as u32,
                hidden as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide patch gather: {error}"
            ))
        })
    }

    pub fn transpose_node_time_f32(
        &mut self,
        input: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        times: usize,
        channels: usize,
    ) -> Result<()> {
        let len = checked_product(&[batches, nodes, times, channels])?;
        if input == output || self.capacity_f32(input)? < len {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA node/time transpose slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.transpose_node_time_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                input_values,
                output_values,
                batches as u32,
                times as u32,
                nodes as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide node/time transpose: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_f32(
        &mut self,
        query: usize,
        key: usize,
        value: usize,
        output: usize,
        sequences: usize,
        tokens: usize,
        heads: usize,
        head_width: usize,
        causal: bool,
    ) -> Result<()> {
        let len = checked_product(&[sequences, tokens, heads, head_width])?;
        if sequences == 0
            || tokens == 0
            || heads == 0
            || head_width == 0
            || [query, key, value, output]
                .iter()
                .enumerate()
                .any(|(index, slot)| [query, key, value, output][..index].contains(slot))
            || self.capacity_f32(query)? < len
            || self.capacity_f32(key)? < len
            || self.capacity_f32(value)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide attention slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (query_values, key_values, value_values, output_values) =
            get_four_f32_slots(&mut self.f32_slots, query, key, value, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.attention_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                query_values,
                key_values,
                value_values,
                output_values,
                tokens as u32,
                heads as u32,
                head_width as u32,
                causal as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide attention: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn csr_diffuse_f32(
        &mut self,
        indptr: usize,
        indices: usize,
        weights: usize,
        values: usize,
        output: usize,
        batches: usize,
        nodes: usize,
        channels: usize,
        edges: usize,
    ) -> Result<()> {
        let len = checked_product(&[batches, nodes, channels])?;
        if batches == 0
            || nodes == 0
            || channels == 0
            || output == weights
            || output == values
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.u32_capacities.get(indices).copied().unwrap_or(0) < edges
            || self.capacity_f32(weights)? < edges
            || self.capacity_f32(values)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide CSR diffusion slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indptr} has not been allocated"
            ))
        })?;
        let indices_values = self.u32_slots[indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indices} has not been allocated"
            ))
        })?;
        let (weight_values, value_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, weights, values, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_diffusion_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                indptr_values,
                indices_values,
                weight_values,
                value_values,
                output_values,
                nodes as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide CSR diffusion: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn csr_diffuse_backward_f32(
        &mut self,
        indptr: usize,
        indices: usize,
        weights: usize,
        values: usize,
        output_gradient: usize,
        input_gradient: usize,
        edge_gradient: usize,
        batches: usize,
        nodes: usize,
        channels: usize,
        edges: usize,
    ) -> Result<()> {
        let value_len = checked_product(&[batches, nodes, channels])?;
        if batches == 0
            || nodes == 0
            || channels == 0
            || [
                weights,
                values,
                output_gradient,
                input_gradient,
                edge_gradient,
            ]
            .iter()
            .enumerate()
            .any(|(index, slot)| {
                [
                    weights,
                    values,
                    output_gradient,
                    input_gradient,
                    edge_gradient,
                ][..index]
                    .contains(slot)
            })
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.u32_capacities.get(indices).copied().unwrap_or(0) < edges
            || self.capacity_f32(weights)? < edges
            || self.capacity_f32(values)? < value_len
            || self.capacity_f32(output_gradient)? < value_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide CSR diffusion-backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, value_len)?;
        self.reserve_f32(edge_gradient, edges)?;
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indptr} has not been allocated"
            ))
        })?;
        let indices_values = self.u32_slots[indices].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indices} has not been allocated"
            ))
        })?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (weight_values, output_gradient_values, input_gradient_values) =
                get_three_f32_slots(
                    &mut self.f32_slots,
                    weights,
                    output_gradient,
                    input_gradient,
                )?;
            unsafe {
                module.csr_diffusion_input_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(value_len as u32),
                    indptr_values,
                    indices_values,
                    weight_values,
                    output_gradient_values,
                    input_gradient_values,
                    nodes as u32,
                    channels as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide CSR input backward: {error}"
                ))
            })?;
        }
        let (value_values, output_gradient_values, edge_gradient_values) =
            get_three_f32_slots(&mut self.f32_slots, values, output_gradient, edge_gradient)?;
        unsafe {
            module.csr_diffusion_edge_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(edges as u32),
                indptr_values,
                indices_values,
                value_values,
                output_gradient_values,
                edge_gradient_values,
                batches as u32,
                nodes as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide CSR edge backward: {error}"
            ))
        })
    }

    pub fn csr_row_softmax_f32(
        &mut self,
        indptr: usize,
        logits: usize,
        weights: usize,
        nodes: usize,
        edges: usize,
    ) -> Result<()> {
        if nodes == 0
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.capacity_f32(logits)? < edges
            || logits == weights
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA CSR row-softmax slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(weights, edges)?;
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indptr} has not been allocated"
            ))
        })?;
        let (logit_values, weight_values) =
            get_two_f32_slots(&mut self.f32_slots, logits, weights)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_row_softmax_f32(
                &self.stream,
                LaunchConfig::for_num_elems(edges as u32),
                indptr_values,
                logit_values,
                weight_values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide CSR row-softmax: {error}"
            ))
        })
    }

    pub fn csr_row_softmax_backward_f32(
        &mut self,
        indptr: usize,
        weights: usize,
        output_gradient: usize,
        logits_gradient: usize,
        nodes: usize,
        edges: usize,
    ) -> Result<()> {
        if nodes == 0
            || self.u32_capacities.get(indptr).copied().unwrap_or(0) < nodes + 1
            || self.capacity_f32(weights)? < edges
            || self.capacity_f32(output_gradient)? < edges
            || weights == output_gradient
            || weights == logits_gradient
            || output_gradient == logits_gradient
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA CSR row-softmax-backward slots or dimensions".to_string(),
            ));
        }
        self.reserve_f32(logits_gradient, edges)?;
        let indptr_values = self.u32_slots[indptr].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {indptr} has not been allocated"
            ))
        })?;
        let (weight_values, gradient_values, logits_gradient_values) = get_three_f32_slots(
            &mut self.f32_slots,
            weights,
            output_gradient,
            logits_gradient,
        )?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_row_softmax_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(edges as u32),
                indptr_values,
                weight_values,
                gradient_values,
                logits_gradient_values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide CSR row-softmax backward: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn affine_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        weights_offset: usize,
        bias_offset: usize,
        output: usize,
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        if rows == 0 || input_width == 0 || output_width == 0 {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA affine dimensions must be non-zero".to_string(),
            ));
        }
        if input == parameters || input == output || parameters == output {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide affine requires distinct input, parameter, and output slots"
                    .to_string(),
            ));
        }
        let input_len = checked_product(&[rows, input_width])?;
        let output_len = checked_product(&[rows, output_width])?;
        let weights_len = checked_product(&[input_width, output_width])?;
        let parameter_len = bias_offset.checked_add(output_width).ok_or_else(|| {
            AcceleratorError::InvalidArgument("CUDA affine parameter range overflows".to_string())
        })?;
        if weights_offset
            .checked_add(weights_len)
            .is_none_or(|end| end > bias_offset)
            || self.capacity_f32(input)? < input_len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA affine tensor shape or parameter slice".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.affine_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                input_values,
                parameter_values,
                output_values,
                weights_offset as u32,
                bias_offset as u32,
                input_width as u32,
                output_width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide affine: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_parameter_slice_f32(
        &mut self,
        input: usize,
        parameters: usize,
        gamma_offset: usize,
        beta_offset: usize,
        output: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        if rows == 0 || width == 0 || input == parameters || input == output || parameters == output
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid cuda-oxide layer-normalization slots or dimensions".to_string(),
            ));
        }
        let len = checked_product(&[rows, width])?;
        let parameter_len = beta_offset.checked_add(width).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA layer-normalization parameter range overflows".to_string(),
            )
        })?;
        if gamma_offset
            .checked_add(width)
            .is_none_or(|end| end > beta_offset)
            || self.capacity_f32(input)? < len
            || self.capacity_f32(parameters)? < parameter_len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA layer-normalization parameter slice".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (input_values, parameter_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, input, parameters, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.layer_norm_parameter_slice_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                input_values,
                parameter_values,
                output_values,
                gamma_offset as u32,
                beta_offset as u32,
                width as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide layer normalization: {error}"
            ))
        })
    }

    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to synchronize cuda-oxide stream: {error}"
            ))
        })
    }

    pub fn reserve_f32(&mut self, slot: usize, len: usize) -> Result<()> {
        let capacity = self.f32_capacities.get_mut(slot).ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!("CUDA tensor slot {slot} is out of range"))
        })?;
        if len <= *capacity {
            return Ok(());
        }
        self.f32_slots[slot] = Some(zeroed(&self.stream, len, "arena slot")?);
        *capacity = len;
        self.allocation_count += 1;
        Ok(())
    }

    pub fn reserve_u32(&mut self, slot: usize, len: usize) -> Result<()> {
        let capacity = self.u32_capacities.get_mut(slot).ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA u32 tensor slot {slot} is out of range"
            ))
        })?;
        if len <= *capacity {
            return Ok(());
        }
        self.u32_slots[slot] = Some(DeviceBuffer::zeroed(&self.stream, len).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to allocate cuda-oxide u32 arena slot: {error}"
            ))
        })?);
        *capacity = len;
        self.allocation_count += 1;
        Ok(())
    }

    pub fn capacity_f32(&self, slot: usize) -> Result<usize> {
        self.f32_capacities.get(slot).copied().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!("CUDA tensor slot {slot} is out of range"))
        })
    }

    pub fn upload_f32(&mut self, slot: usize, values: &[f32]) -> Result<()> {
        self.reserve_f32(slot, values.len())?;
        let capacity = self.f32_capacities[slot];
        let mut padded = vec![0.0; capacity];
        padded[..values.len()].copy_from_slice(values);
        self.f32_slots[slot]
            .as_mut()
            .expect("allocated f32 slot")
            .copy_from_host(&self.stream, &padded)
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to upload cuda-oxide arena slot: {error}"
                ))
            })
    }

    pub fn upload_u32(&mut self, slot: usize, values: &[u32]) -> Result<()> {
        self.reserve_u32(slot, values.len())?;
        let capacity = self.u32_capacities[slot];
        let mut padded = vec![0; capacity];
        padded[..values.len()].copy_from_slice(values);
        self.u32_slots[slot]
            .as_mut()
            .expect("allocated u32 slot")
            .copy_from_host(&self.stream, &padded)
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to upload cuda-oxide u32 arena slot: {error}"
                ))
            })
    }

    pub fn download_f32(&self, slot: usize, values: &mut [f32]) -> Result<()> {
        let capacity = self.capacity_f32(slot)?;
        if values.len() > capacity {
            return Err(AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {slot} has capacity {capacity}, cannot download {} values",
                values.len()
            )));
        }
        let host_values = self
            .f32_slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(format!(
                    "CUDA tensor slot {slot} has not been allocated"
                ))
            })?
            .to_host_vec(&self.stream)
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to download cuda-oxide arena slot: {error}"
                ))
            })?;
        values.copy_from_slice(&host_values[..values.len()]);
        Ok(())
    }

    pub fn add_f32(&mut self, left: usize, right: usize, output: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA add length must be non-zero".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let left_capacity = self.capacity_f32(left)?;
        let right_capacity = self.capacity_f32(right)?;
        if left_capacity < len || right_capacity < len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA add input slot is smaller than the requested length".to_string(),
            ));
        }
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        let launch = LaunchConfig::for_num_elems(len as u32);
        if output == left && output == right {
            unsafe {
                module.double_in_place_f32(
                    &self.stream,
                    launch,
                    get_mut_f32_slot(&mut self.f32_slots, output)?,
                )
            }
        } else if output == left {
            let (other, target) = get_two_f32_slots(&mut self.f32_slots, right, output)?;
            unsafe { module.add_in_place_f32(&self.stream, launch, other, target) }
        } else if output == right {
            let (other, target) = get_two_f32_slots(&mut self.f32_slots, left, output)?;
            unsafe { module.add_in_place_f32(&self.stream, launch, other, target) }
        } else {
            let (left_values, right_values, output_values) =
                get_three_f32_slots(&mut self.f32_slots, left, right, output)?;
            unsafe {
                module.add_f32(
                    &self.stream,
                    launch,
                    left_values,
                    right_values,
                    output_values,
                )
            }
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide arena add: {error}"
            ))
        })
    }

    /// Computes row-major `output = left × right` without materializing an
    /// intermediate tensor on the host. `left` is `[rows, shared]`, `right`
    /// is `[shared, columns]`, and `output` is `[rows, columns]`.
    fn matmul_f32(
        &mut self,
        left: usize,
        right: usize,
        output: usize,
        rows: usize,
        shared: usize,
        columns: usize,
    ) -> Result<()> {
        if rows == 0 || shared == 0 || columns == 0 {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA matrix multiplication dimensions must be non-zero".to_string(),
            ));
        }
        if output == left || output == right {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide matmul requires an output slot distinct from both inputs".to_string(),
            ));
        }
        let left_len = rows.checked_mul(shared).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA matrix multiplication dimensions overflow".to_string(),
            )
        })?;
        let right_len = shared.checked_mul(columns).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA matrix multiplication dimensions overflow".to_string(),
            )
        })?;
        let output_len = rows.checked_mul(columns).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA matrix multiplication dimensions overflow".to_string(),
            )
        })?;
        if self.capacity_f32(left)? < left_len || self.capacity_f32(right)? < right_len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA matrix multiplication input slot is smaller than its matrix shape"
                    .to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (left_values, right_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, left, right, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.matmul_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                left_values,
                right_values,
                output_values,
                rows as u32,
                shared as u32,
                columns as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide arena matrix multiplication: {error}"
            ))
        })
    }

    pub fn fill_f32(&mut self, slot: usize, len: usize, value: f32) -> Result<()> {
        self.reserve_f32(slot, len)?;
        let output = self.f32_slots[slot].as_mut().expect("allocated f32 slot");
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.fill_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                output,
                value,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide arena fill: {error}"
            ))
        })
    }

    pub fn scale_f32(&mut self, input: usize, output: usize, len: usize, scale: f32) -> Result<()> {
        if input == output {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide scale requires distinct input and output slots".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (source, target) = if input < output {
            let (before_output, from_output) = self.f32_slots.split_at_mut(output);
            (before_output[input].as_ref(), from_output[0].as_mut())
        } else {
            let (before_input, from_input) = self.f32_slots.split_at_mut(input);
            (from_input[0].as_ref(), before_input[output].as_mut())
        };
        let source = source.ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {input} has not been allocated"
            ))
        })?;
        let target = target.ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {output} has not been allocated"
            ))
        })?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.scale_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                source,
                target,
                scale,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide arena scale: {error}"
            ))
        })
    }

    pub fn relu_in_place_f32(&mut self, slot: usize, len: usize) -> Result<()> {
        let values = self
            .f32_slots
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                AcceleratorError::InvalidArgument(format!(
                    "CUDA tensor slot {slot} has not been allocated"
                ))
            })?;
        if len > values.len() {
            return Err(AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {slot} has capacity {}, cannot apply ReLU to {len} values",
                values.len()
            )));
        }
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.relu_in_place_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide in-place ReLU: {error}"
            ))
        })
    }

    pub fn relu_f32(&mut self, input: usize, output: usize, len: usize) -> Result<()> {
        if input == output {
            return self.relu_in_place_f32(input, len);
        }
        self.launch_relu_like_f32(input, output, len, true)
    }

    pub fn gelu_f32(&mut self, input: usize, output: usize, len: usize) -> Result<()> {
        self.launch_relu_like_f32(input, output, len, false)
    }

    fn launch_relu_like_f32(
        &mut self,
        input: usize,
        output: usize,
        len: usize,
        relu: bool,
    ) -> Result<()> {
        let name = if relu { "ReLU" } else { "GELU" };
        if len == 0 || input == output || self.capacity_f32(input)? < len {
            return Err(AcceleratorError::InvalidArgument(format!(
                "invalid CUDA {name} tensor slots or length"
            )));
        }
        self.reserve_f32(output, len)?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        if relu {
            unsafe {
                module.relu_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    input_values,
                    output_values,
                )
            }
        } else {
            unsafe {
                module.gelu_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(len as u32),
                    input_values,
                    output_values,
                )
            }
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide {name}: {error}"
            ))
        })
    }

    pub fn gated_tanh_sigmoid_f32(
        &mut self,
        filter: usize,
        gate: usize,
        output: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 || filter == gate || filter == output || gate == output {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide gated activation requires three distinct slots".to_string(),
            ));
        }
        if self.capacity_f32(filter)? < len || self.capacity_f32(gate)? < len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA gated activation input slot is smaller than the requested length".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (filter_values, gate_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, filter, gate, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.gated_tanh_sigmoid_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                filter_values,
                gate_values,
                output_values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide gated activation: {error}"
            ))
        })
    }

    pub fn relu_backward_f32(
        &mut self,
        activations: usize,
        output_gradient: usize,
        input_gradient: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0
            || activations == output_gradient
            || activations == input_gradient
            || output_gradient == input_gradient
            || self.capacity_f32(activations)? < len
            || self.capacity_f32(output_gradient)? < len
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA ReLU-backward slots or length".to_string(),
            ));
        }
        self.reserve_f32(input_gradient, len)?;
        let (activation_values, output_gradient_values, input_gradient_values) =
            get_three_f32_slots(
                &mut self.f32_slots,
                activations,
                output_gradient,
                input_gradient,
            )?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.relu_backward_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                activation_values,
                output_gradient_values,
                input_gradient_values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide ReLU backward: {error}"
            ))
        })
    }

    pub fn deterministic_dropout_f32(
        &mut self,
        input: usize,
        output: usize,
        len: usize,
        seed: u64,
        base: usize,
        training: bool,
        keep_probability: f32,
    ) -> Result<()> {
        if len == 0
            || input == output
            || self.capacity_f32(input)? < len
            || !keep_probability.is_finite()
            || !(0.0..=1.0).contains(&keep_probability)
        {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA deterministic-dropout arguments".to_string(),
            ));
        }
        self.reserve_f32(output, len)?;
        let (input_values, output_values) = get_two_f32_slots(&mut self.f32_slots, input, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.deterministic_dropout_f32(
                &self.stream,
                LaunchConfig::for_num_elems(len as u32),
                input_values,
                output_values,
                seed as u32,
                (seed >> 32) as u32,
                base as u32,
                (base >> 32) as u32,
                if training {
                    1.0 - keep_probability
                } else {
                    1.0
                },
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide deterministic dropout: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tail_time_f32(
        &mut self,
        left: usize,
        right: usize,
        output: usize,
        batches: usize,
        left_times: usize,
        right_times: usize,
        nodes: usize,
        channels: usize,
    ) -> Result<()> {
        validate_tail_dimensions(batches, left_times, right_times, nodes, channels)?;
        if left == right || left == output || right == output {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide tail add requires three distinct slots".to_string(),
            ));
        }
        let left_len = checked_product(&[batches, left_times, nodes, channels])?;
        let right_len = checked_product(&[batches, right_times, nodes, channels])?;
        if self.capacity_f32(left)? < left_len || self.capacity_f32(right)? < right_len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA tail-add input slot is smaller than its tensor shape".to_string(),
            ));
        }
        self.reserve_f32(output, right_len)?;
        let (left_values, right_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, left, right, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.add_tail_time_f32(
                &self.stream,
                LaunchConfig::for_num_elems(right_len as u32),
                left_values,
                right_values,
                output_values,
                left_times as u32,
                right_times as u32,
                nodes as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide tail add: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tail_time_backward_f32(
        &mut self,
        output_gradient: usize,
        left_gradient: usize,
        right_gradient: usize,
        batches: usize,
        left_times: usize,
        right_times: usize,
        nodes: usize,
        channels: usize,
    ) -> Result<()> {
        validate_tail_dimensions(batches, left_times, right_times, nodes, channels)?;
        if left_gradient == right_gradient {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide tail-add backward requires distinct branch-gradient slots".to_string(),
            ));
        }
        let left_len = checked_product(&[batches, left_times, nodes, channels])?;
        let right_len = checked_product(&[batches, right_times, nodes, channels])?;
        if self.capacity_f32(output_gradient)? < right_len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA tail-add output-gradient slot is smaller than its tensor shape".to_string(),
            ));
        }
        self.reserve_f32(left_gradient, left_len)?;
        if right_gradient != output_gradient {
            self.reserve_f32(right_gradient, right_len)?;
        }
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        if output_gradient == left_gradient {
            if right_gradient == output_gradient {
                return Err(AcceleratorError::InvalidArgument(
                    "cuda-oxide tail-add backward cannot alias all gradient slots".to_string(),
                ));
            }
            {
                let (output_gradient_values, right_gradient_values) =
                    get_two_f32_slots(&mut self.f32_slots, output_gradient, right_gradient)?;
                unsafe {
                    module.copy_f32(
                        &self.stream,
                        LaunchConfig::for_num_elems(right_len as u32),
                        output_gradient_values,
                        right_gradient_values,
                    )
                }
                .map_err(|error| {
                    AcceleratorError::InvalidArgument(format!(
                        "failed to preserve cuda-oxide tail-add output gradient: {error}"
                    ))
                })?;
            }
            let (right_gradient_values, left_gradient_values) =
                get_two_f32_slots(&mut self.f32_slots, right_gradient, left_gradient)?;
            unsafe {
                module.add_tail_time_left_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(left_len as u32),
                    right_gradient_values,
                    left_gradient_values,
                    left_times as u32,
                    right_times as u32,
                    nodes as u32,
                    channels as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide aliased tail-add left gradient: {error}"
                ))
            })?;
            return Ok(());
        }
        {
            let (output_gradient_values, left_gradient_values) =
                get_two_f32_slots(&mut self.f32_slots, output_gradient, left_gradient)?;
            unsafe {
                module.add_tail_time_left_backward_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(left_len as u32),
                    output_gradient_values,
                    left_gradient_values,
                    left_times as u32,
                    right_times as u32,
                    nodes as u32,
                    channels as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide tail-add left gradient: {error}"
                ))
            })?;
        }
        if right_gradient == output_gradient {
            return Ok(());
        }
        let (output_gradient_values, right_gradient_values) =
            get_two_f32_slots(&mut self.f32_slots, output_gradient, right_gradient)?;
        unsafe {
            module.copy_f32(
                &self.stream,
                LaunchConfig::for_num_elems(right_len as u32),
                output_gradient_values,
                right_gradient_values,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide tail-add right gradient: {error}"
            ))
        })
    }

    pub fn concat_channels_f32(
        &mut self,
        left: usize,
        right: usize,
        output: usize,
        rows: usize,
        left_channels: usize,
        right_channels: usize,
    ) -> Result<()> {
        if rows == 0 || left_channels == 0 || right_channels == 0 {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA channel concatenation dimensions".to_string(),
            ));
        }
        if left == right || left == output || right == output {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide channel concatenation requires distinct slots".to_string(),
            ));
        }
        let left_len = checked_product(&[rows, left_channels])?;
        let right_len = checked_product(&[rows, right_channels])?;
        let output_len = left_len.checked_add(right_len).ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA channel concatenation dimensions overflow".to_string(),
            )
        })?;
        if self.capacity_f32(left)? < left_len || self.capacity_f32(right)? < right_len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA channel concatenation input slot is smaller than its shape".to_string(),
            ));
        }
        self.reserve_f32(output, output_len)?;
        let (left_values, right_values, output_values) =
            get_three_f32_slots(&mut self.f32_slots, left, right, output)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.concat_channels_f32(
                &self.stream,
                LaunchConfig::for_num_elems(output_len as u32),
                left_values,
                right_values,
                output_values,
                rows as u32,
                left_channels as u32,
                right_channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide channel concatenation: {error}"
            ))
        })
    }

    pub fn split_channels_f32(
        &mut self,
        input: usize,
        left: usize,
        right: usize,
        rows: usize,
        left_channels: usize,
        right_channels: usize,
    ) -> Result<()> {
        if rows == 0 || left_channels == 0 || right_channels == 0 {
            return Err(AcceleratorError::InvalidArgument(
                "invalid CUDA channel split dimensions".to_string(),
            ));
        }
        if input == left || input == right || left == right {
            return Err(AcceleratorError::InvalidArgument(
                "cuda-oxide channel split requires distinct slots".to_string(),
            ));
        }
        let left_len = checked_product(&[rows, left_channels])?;
        let right_len = checked_product(&[rows, right_channels])?;
        let input_len = left_len.checked_add(right_len).ok_or_else(|| {
            AcceleratorError::InvalidArgument("CUDA channel split dimensions overflow".to_string())
        })?;
        if self.capacity_f32(input)? < input_len {
            return Err(AcceleratorError::InvalidArgument(
                "CUDA channel split input slot is smaller than its shape".to_string(),
            ));
        }
        self.reserve_f32(left, left_len)?;
        self.reserve_f32(right, right_len)?;
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        {
            let (input_values, left_values) = get_two_f32_slots(&mut self.f32_slots, input, left)?;
            unsafe {
                module.split_left_channels_f32(
                    &self.stream,
                    LaunchConfig::for_num_elems(left_len as u32),
                    input_values,
                    left_values,
                    left_channels as u32,
                    right_channels as u32,
                )
            }
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to launch cuda-oxide left channel split: {error}"
                ))
            })?;
        }
        let (input_values, right_values) = get_two_f32_slots(&mut self.f32_slots, input, right)?;
        unsafe {
            module.split_right_channels_f32(
                &self.stream,
                LaunchConfig::for_num_elems(right_len as u32),
                input_values,
                right_values,
                left_channels as u32,
                right_channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide right channel split: {error}"
            ))
        })
    }
}

impl CudaCsrDiffusionWorkspace {
    pub fn new(indptr: &[u32], indices: &[u32], weights: &[f32]) -> Result<Self> {
        if indptr.len() < 2 || indices.len() != weights.len() {
            return Err(AcceleratorError::InvalidArgument(
                "CSR workspace graph buffers are invalid".to_string(),
            ));
        }
        let context = context()?;
        let stream = context.default_stream();
        let indptr_device = DeviceBuffer::from_host(&stream, indptr).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to upload cuda-oxide CSR indptr: {error}"
            ))
        })?;
        let indices_device = DeviceBuffer::from_host(&stream, indices).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to upload cuda-oxide CSR indices: {error}"
            ))
        })?;
        let weights_device = DeviceBuffer::from_host(&stream, weights).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to upload cuda-oxide CSR weights: {error}"
            ))
        })?;
        Ok(Self {
            context,
            stream,
            indptr: indptr_device,
            indices: indices_device,
            weights: weights_device,
            values: None,
            output: None,
            nodes: indptr.len() - 1,
            value_capacity: 0,
            allocation_count: 3,
        })
    }

    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }
    pub fn value_capacity(&self) -> usize {
        self.value_capacity
    }

    pub fn diffuse(&mut self, channels: usize, values: &[f32]) -> Result<Vec<f32>> {
        if channels == 0 || values.len() % (self.nodes * channels) != 0 {
            return Err(AcceleratorError::InvalidArgument(
                "CSR workspace values must be [batch, nodes, channels] data".to_string(),
            ));
        }
        if values.len() > self.value_capacity {
            self.values = Some(zeroed(&self.stream, values.len(), "CSR workspace values")?);
            self.output = Some(zeroed(&self.stream, values.len(), "CSR workspace output")?);
            self.value_capacity = values.len();
            self.allocation_count += 2;
        }
        let values_device = self.values.as_mut().expect("allocated workspace values");
        values_device
            .copy_from_host(&self.stream, values)
            .map_err(|error| {
                AcceleratorError::InvalidArgument(format!(
                    "failed to upload cuda-oxide CSR workspace values: {error}"
                ))
            })?;
        let output_device = self.output.as_mut().expect("allocated workspace output");
        let module = kernels::load(&self.context).map_err(|error| {
            AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
        })?;
        unsafe {
            module.csr_diffusion_f32(
                &self.stream,
                LaunchConfig::for_num_elems(values.len() as u32),
                &self.indptr,
                &self.indices,
                &self.weights,
                values_device,
                output_device,
                self.nodes as u32,
                channels as u32,
            )
        }
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to launch cuda-oxide CSR workspace: {error}"
            ))
        })?;
        output_device.to_host_vec(&self.stream).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to read cuda-oxide CSR workspace output: {error}"
            ))
        })
    }
}

pub(super) fn scalar_graph(
    initial_values: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
) -> Result<Vec<f32>> {
    let context = context()?;
    let stream = context.default_stream();
    let initial_values_device =
        DeviceBuffer::from_host(&stream, initial_values).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to upload CUDA scalar-graph values: {error}"
            ))
        })?;
    let opcodes_device = DeviceBuffer::from_host(&stream, opcodes).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload CUDA scalar-graph opcodes: {error}"
        ))
    })?;
    let left_device = DeviceBuffer::from_host(&stream, left).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload CUDA scalar-graph left indices: {error}"
        ))
    })?;
    let right_device = DeviceBuffer::from_host(&stream, right).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload CUDA scalar-graph right indices: {error}"
        ))
    })?;
    let mut values_device = zeroed(&stream, initial_values.len(), "scalar-graph values")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to load CUDA scalar-graph module: {error}"
        ))
    })?;
    unsafe {
        module.scalar_graph_f32(
            &stream,
            LaunchConfig::for_num_elems(1),
            &initial_values_device,
            &opcodes_device,
            &left_device,
            &right_device,
            &mut values_device,
            initial_values.len() as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch CUDA scalar-graph kernel: {error}"
        ))
    })?;
    let values = values_device.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to download CUDA scalar-graph values: {error}"
        ))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(AcceleratorError::InvalidArgument(
            "CUDA scalar-graph inference produced a non-finite value".to_string(),
        ));
    }
    Ok(values)
}

pub(super) fn train_tanh_mlp(
    inputs: &[Vec<f32>],
    targets: &[f32],
    hidden_size: usize,
    epochs: usize,
    learning_rate: f32,
    parameters: &mut [f32],
) -> Result<()> {
    let context = context()?;
    let stream = context.default_stream();
    let flat = inputs.iter().flatten().copied().collect::<Vec<_>>();
    let inputs_device = DeviceBuffer::from_host(&stream, &flat).map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to upload CUDA tanh-MLP inputs: {e}"))
    })?;
    let targets_device = DeviceBuffer::from_host(&stream, targets).map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to upload CUDA tanh-MLP targets: {e}"))
    })?;
    let mut parameters_device = DeviceBuffer::from_host(&stream, parameters).map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to upload CUDA tanh-MLP parameters: {e}"))
    })?;
    let module = kernels::load(&context).map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to load CUDA tanh-MLP module: {e}"))
    })?;
    unsafe {
        module.train_tanh_mlp_f32(
            &stream,
            LaunchConfig::for_num_elems(1),
            &inputs_device,
            &targets_device,
            &mut parameters_device,
            inputs.len() as u32,
            inputs[0].len() as u32,
            hidden_size as u32,
            epochs as u32,
            learning_rate,
        )
    }
    .map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to launch CUDA tanh-MLP kernel: {e}"))
    })?;
    let trained = parameters_device.to_host_vec(&stream).map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to read CUDA tanh-MLP parameters: {e}"))
    })?;
    parameters.copy_from_slice(&trained);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scalar_graph_train_step(
    initial: &[f32],
    opcodes: &[u8],
    left: &[u32],
    right: &[u32],
    parameter_ids: &[u32],
    loss: usize,
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<f32> {
    let context = context()?;
    let stream = context.default_stream();
    let initial_d = DeviceBuffer::from_host(&stream, initial)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let ops = DeviceBuffer::from_host(&stream, opcodes)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let l = DeviceBuffer::from_host(&stream, left)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let r = DeviceBuffer::from_host(&stream, right)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let ids = DeviceBuffer::from_host(&stream, parameter_ids)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let mut p = DeviceBuffer::from_host(&stream, parameters)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let mut m = DeviceBuffer::from_host(&stream, first)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let mut v = DeviceBuffer::from_host(&stream, second)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    let mut values = zeroed(&stream, initial.len(), "scalar values")?;
    let mut gradients = zeroed(&stream, initial.len(), "scalar gradients")?;
    let mut pgrad = zeroed(&stream, parameters.len(), "parameter gradients")?;
    let module =
        kernels::load(&context).map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    unsafe {
        module.scalar_graph_train_f32(
            &stream,
            LaunchConfig::for_num_elems(1),
            &initial_d,
            &ops,
            &l,
            &r,
            &ids,
            &mut p,
            &mut m,
            &mut v,
            &mut values,
            &mut gradients,
            &mut pgrad,
            initial.len() as u32,
            loss as u32,
            parameters.len() as u32,
            step as u32,
            learning_rate,
            weight_decay,
        )
    }
    .map_err(|e| {
        AcceleratorError::InvalidArgument(format!("failed to launch CUDA scalar training: {e}"))
    })?;
    let computed = values
        .to_host_vec(&stream)
        .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?;
    parameters.copy_from_slice(
        &p.to_host_vec(&stream)
            .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?,
    );
    first.copy_from_slice(
        &m.to_host_vec(&stream)
            .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?,
    );
    second.copy_from_slice(
        &v.to_host_vec(&stream)
            .map_err(|e| AcceleratorError::InvalidArgument(e.to_string()))?,
    );
    Ok(computed[loss])
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
        AcceleratorError::InvalidArgument(format!("failed to create cuda-oxide context: {error}"))
    })?;
    let stream = context.default_stream();
    let left_device = DeviceBuffer::from_host(&stream, &left).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide left tensor: {error}"
        ))
    })?;
    let right_device = DeviceBuffer::from_host(&stream, &right).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide right tensor: {error}"
        ))
    })?;
    let mut output_device = DeviceBuffer::<f32>::zeroed(&stream, len).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to allocate cuda-oxide output tensor: {error}"
        ))
    })?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
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
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide vector add: {error}"
        ))
    })?;
    let output = output_device.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to read cuda-oxide output tensor: {error}"
        ))
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

pub(super) fn affine_scores(
    features: &[Vec<f64>],
    means: &[f64],
    weights: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>> {
    let rows = features.len();
    let cols = weights.len();
    let features = features
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
    let context = context()?;
    let stream = context.default_stream();
    let features = buffer(&stream, &features, "features")?;
    let means = buffer(&stream, &means, "means")?;
    let weights = buffer(&stream, &weights, "weights")?;
    let intercepts = buffer(&stream, &intercepts, "intercepts")?;
    let mut output = zeroed(&stream, rows, "affine output")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.affine_scores_f32(
            &stream,
            LaunchConfig::for_num_elems(rows as u32),
            &features,
            &means,
            &weights,
            &intercepts,
            &mut output,
            cols as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide affine scores: {error}"
        ))
    })?;
    output
        .to_host_vec(&stream)
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to read cuda-oxide affine output: {error}"
            ))
        })
}

pub(super) fn dense_layer(
    features: &[Vec<f32>],
    weights: &[f32],
    biases: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let rows = features.len();
    let cols = features[0].len();
    let out_dim = biases.len();
    let features = features
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let context = context()?;
    let stream = context.default_stream();
    let features = buffer(&stream, &features, "features")?;
    let weights = buffer(&stream, weights, "weights")?;
    let biases = buffer(&stream, biases, "biases")?;
    let mut output = zeroed(&stream, rows * out_dim, "dense output")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.dense_layer_f32(
            &stream,
            LaunchConfig::for_num_elems((rows * out_dim) as u32),
            &features,
            &weights,
            &biases,
            &mut output,
            cols as u32,
            out_dim as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide dense layer: {error}"
        ))
    })?;
    output
        .to_host_vec(&stream)
        .map(|values| values.chunks(out_dim).map(ToOwned::to_owned).collect())
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to read cuda-oxide dense output: {error}"
            ))
        })
}

pub(super) fn csr_diffusion(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    let nodes = indptr.len() - 1;
    let len = values.len();
    let context = context()?;
    let stream = context.default_stream();
    let indptr = DeviceBuffer::from_host(&stream, indptr).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide CSR indptr: {error}"
        ))
    })?;
    let indices = DeviceBuffer::from_host(&stream, indices).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide CSR indices: {error}"
        ))
    })?;
    let weights = buffer(&stream, weights, "CSR weights")?;
    let values = buffer(&stream, values, "CSR values")?;
    let mut output = zeroed(&stream, len, "CSR output")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.csr_diffusion_f32(
            &stream,
            LaunchConfig::for_num_elems(len as u32),
            &indptr,
            &indices,
            &weights,
            &values,
            &mut output,
            nodes as u32,
            channels as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide CSR diffusion: {error}"
        ))
    })?;
    output.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to read cuda-oxide CSR output: {error}"))
    })
}

pub(super) fn csr_diffusion_backward(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    channels: usize,
    values: &[f32],
    output_gradient: &[f32],
) -> Result<crate::CsrDiffusionBackward> {
    let nodes = indptr.len() - 1;
    let batches = values.len() / (nodes * channels);
    let context = context()?;
    let stream = context.default_stream();
    let indptr = DeviceBuffer::from_host(&stream, indptr).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide CSR indptr: {error}"
        ))
    })?;
    let indices = DeviceBuffer::from_host(&stream, indices).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide CSR indices: {error}"
        ))
    })?;
    let weights = buffer(&stream, weights, "CSR weights")?;
    let values = buffer(&stream, values, "CSR values")?;
    let output_gradient = buffer(&stream, output_gradient, "CSR output gradient")?;
    let mut input_gradient = zeroed(&stream, values.len(), "CSR input gradient")?;
    let mut edge_gradient = zeroed(&stream, indices.len(), "CSR edge gradient")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.csr_diffusion_input_backward_f32(
            &stream,
            LaunchConfig::for_num_elems(values.len() as u32),
            &indptr,
            &indices,
            &weights,
            &output_gradient,
            &mut input_gradient,
            nodes as u32,
            channels as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide CSR input-gradient kernel: {error}"
        ))
    })?;
    unsafe {
        module.csr_diffusion_edge_backward_f32(
            &stream,
            LaunchConfig::for_num_elems(indices.len() as u32),
            &indptr,
            &indices,
            &values,
            &output_gradient,
            &mut edge_gradient,
            batches as u32,
            nodes as u32,
            channels as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide CSR edge-gradient kernel: {error}"
        ))
    })?;
    Ok(crate::CsrDiffusionBackward {
        input_grad: input_gradient.to_host_vec(&stream).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to read cuda-oxide CSR input gradient: {error}"
            ))
        })?,
        edge_grad: edge_gradient.to_host_vec(&stream).map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to read cuda-oxide CSR edge gradient: {error}"
            ))
        })?,
    })
}

pub(super) fn layer_norm(
    values: &[f32],
    rows: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    let context = context()?;
    let stream = context.default_stream();
    let values = buffer(&stream, values, "layer norm values")?;
    let gamma = buffer(&stream, gamma, "layer norm gamma")?;
    let beta = buffer(&stream, beta, "layer norm beta")?;
    let mut output = zeroed(&stream, rows * width, "layer norm output")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.layer_norm_f32(
            &stream,
            LaunchConfig::for_num_elems((rows * width) as u32),
            &values,
            &gamma,
            &beta,
            &mut output,
            width as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide layer norm: {error}"
        ))
    })?;
    output.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to read cuda-oxide layer norm: {error}"))
    })
}

pub(super) fn adamw(
    parameters: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    gradients: &[f32],
    step: u64,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<()> {
    let context = context()?;
    let stream = context.default_stream();
    let mut parameters_device = buffer(&stream, parameters, "AdamW parameters")?;
    let mut first_device = buffer(&stream, first, "AdamW first moment")?;
    let mut second_device = buffer(&stream, second, "AdamW second moment")?;
    let gradients = buffer(&stream, gradients, "AdamW gradients")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.adamw_f32(
            &stream,
            LaunchConfig::for_num_elems(parameters.len() as u32),
            &mut parameters_device,
            &mut first_device,
            &mut second_device,
            &gradients,
            learning_rate,
            weight_decay,
            1.0 - 0.9_f32.powi(step as i32),
            1.0 - 0.999_f32.powi(step as i32),
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to launch cuda-oxide AdamW: {error}"))
    })?;
    let updated_parameters = parameters_device.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to read cuda-oxide AdamW parameters: {error}"
        ))
    })?;
    let updated_first = first_device.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to read cuda-oxide AdamW first moment: {error}"
        ))
    })?;
    let updated_second = second_device.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to read cuda-oxide AdamW second moment: {error}"
        ))
    })?;
    parameters.copy_from_slice(&updated_parameters);
    first.copy_from_slice(&updated_first);
    second.copy_from_slice(&updated_second);
    Ok(())
}

pub(super) fn pair_sigmoid_scores(
    embeddings: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> Result<Vec<f64>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let dim = embeddings[0].len();
    let embeddings = embeddings
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let pairs = pairs
        .iter()
        .flat_map(|&(source, target)| [source as u32, target as u32])
        .collect::<Vec<_>>();
    let pair_len = pairs.len() / 2;
    let context = context()?;
    let stream = context.default_stream();
    let embeddings = buffer(&stream, &embeddings, "pair embeddings")?;
    let pairs = DeviceBuffer::from_host(&stream, &pairs).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to upload cuda-oxide pairs: {error}"))
    })?;
    let mut output = zeroed(&stream, pair_len, "pair scores")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.pair_sigmoid_scores_f32(
            &stream,
            LaunchConfig::for_num_elems(pair_len as u32),
            &embeddings,
            &pairs,
            &mut output,
            dim as u32,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide pair scores: {error}"
        ))
    })?;
    output
        .to_host_vec(&stream)
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| {
            AcceleratorError::InvalidArgument(format!(
                "failed to read cuda-oxide pair scores: {error}"
            ))
        })
}

pub(super) fn csr_row_softmax(indptr: &[u32], logits: &[f32]) -> Result<Vec<f32>> {
    let len = logits.len();
    let context = context()?;
    let stream = context.default_stream();
    let indptr = DeviceBuffer::from_host(&stream, indptr).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide CSR indptr: {error}"
        ))
    })?;
    let logits = buffer(&stream, logits, "CSR logits")?;
    let mut weights = zeroed(&stream, len, "CSR softmax weights")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.csr_row_softmax_f32(
            &stream,
            LaunchConfig::for_num_elems(len as u32),
            &indptr,
            &logits,
            &mut weights,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide CSR softmax: {error}"
        ))
    })?;
    weights.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to read cuda-oxide CSR softmax: {error}"))
    })
}

pub(super) fn csr_row_softmax_backward(
    indptr: &[u32],
    weights: &[f32],
    output_gradient: &[f32],
) -> Result<Vec<f32>> {
    let len = weights.len();
    let context = context()?;
    let stream = context.default_stream();
    let indptr = DeviceBuffer::from_host(&stream, indptr).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide CSR indptr: {error}"
        ))
    })?;
    let weights = buffer(&stream, weights, "CSR weights")?;
    let output_gradient = buffer(&stream, output_gradient, "CSR output gradient")?;
    let mut logits_gradient = zeroed(&stream, len, "CSR logits gradient")?;
    let module = kernels::load(&context).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to load cuda-oxide module: {error}"))
    })?;
    unsafe {
        module.csr_row_softmax_backward_f32(
            &stream,
            LaunchConfig::for_num_elems(len as u32),
            &indptr,
            &weights,
            &output_gradient,
            &mut logits_gradient,
        )
    }
    .map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to launch cuda-oxide CSR softmax backward: {error}"
        ))
    })?;
    logits_gradient.to_host_vec(&stream).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to read cuda-oxide CSR logits gradient: {error}"
        ))
    })
}

fn context() -> Result<Arc<CudaContext>> {
    CudaContext::new(0).map_err(|error| {
        AcceleratorError::InvalidArgument(format!("failed to create cuda-oxide context: {error}"))
    })
}

fn buffer(stream: &cuda_core::CudaStream, values: &[f32], name: &str) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::from_host(stream, values).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to upload cuda-oxide {name} tensor: {error}"
        ))
    })
}

fn zeroed(stream: &cuda_core::CudaStream, len: usize, name: &str) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::zeroed(stream, len).map_err(|error| {
        AcceleratorError::InvalidArgument(format!(
            "failed to allocate cuda-oxide {name} tensor: {error}"
        ))
    })
}

fn validate_tail_dimensions(
    batches: usize,
    left_times: usize,
    right_times: usize,
    nodes: usize,
    channels: usize,
) -> Result<()> {
    if batches == 0 || right_times == 0 || left_times < right_times || nodes == 0 || channels == 0 {
        return Err(AcceleratorError::InvalidArgument(
            "invalid CUDA causal tail-add dimensions".to_string(),
        ));
    }
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            AcceleratorError::InvalidArgument("CUDA tensor dimensions overflow".to_string())
        })
    })
}

fn get_two_f32_slots(
    slots: &mut [Option<DeviceBuffer<f32>>],
    input: usize,
    output: usize,
) -> Result<(&DeviceBuffer<f32>, &mut DeviceBuffer<f32>)> {
    if input == output || input >= slots.len() || output >= slots.len() {
        return Err(AcceleratorError::InvalidArgument(
            "cuda-oxide arena requires distinct in-range tensor slots".to_string(),
        ));
    }
    if input < output {
        let (before_output, from_output) = slots.split_at_mut(output);
        let input = before_output[input].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {input} has not been allocated"
            ))
        })?;
        let output = from_output[0].as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {output} has not been allocated"
            ))
        })?;
        Ok((input, output))
    } else {
        let (before_input, from_input) = slots.split_at_mut(input);
        let output = before_input[output].as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {output} has not been allocated"
            ))
        })?;
        let input = from_input[0].as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {input} has not been allocated"
            ))
        })?;
        Ok((input, output))
    }
}

fn get_mut_f32_slot(
    slots: &mut [Option<DeviceBuffer<f32>>],
    slot: usize,
) -> Result<&mut DeviceBuffer<f32>> {
    slots
        .get_mut(slot)
        .ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!("CUDA tensor slot {slot} is out of range"))
        })?
        .as_mut()
        .ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {slot} has not been allocated"
            ))
        })
}

fn get_three_f32_slots(
    slots: &mut [Option<DeviceBuffer<f32>>],
    left: usize,
    right: usize,
    output: usize,
) -> Result<(
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
)> {
    if left == right || left == output || right == output {
        return Err(AcceleratorError::InvalidArgument(
            "cuda-oxide arena add requires three distinct slots".to_string(),
        ));
    }
    if left >= slots.len() || right >= slots.len() || output >= slots.len() {
        return Err(AcceleratorError::InvalidArgument(
            "CUDA tensor slot is out of range".to_string(),
        ));
    }
    // Distinct slot indices make these simultaneous references non-overlapping.
    let ptr = slots.as_mut_ptr();
    unsafe {
        let left = (*ptr.add(left)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA left tensor slot has not been allocated".to_string(),
            )
        })?;
        let right = (*ptr.add(right)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA right tensor slot has not been allocated".to_string(),
            )
        })?;
        let output = (*ptr.add(output)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA output tensor slot has not been allocated".to_string(),
            )
        })?;
        Ok((left, right, output))
    }
}

fn get_four_f32_slots(
    slots: &mut [Option<DeviceBuffer<f32>>],
    first: usize,
    second: usize,
    third: usize,
    output: usize,
) -> Result<(
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
)> {
    let indices = [first, second, third, output];
    if indices.iter().any(|index| *index >= slots.len())
        || indices
            .iter()
            .enumerate()
            .any(|(index, value)| indices[..index].contains(value))
    {
        return Err(AcceleratorError::InvalidArgument(
            "cuda-oxide arena requires four distinct in-range tensor slots".to_string(),
        ));
    }
    // The distinct-index check makes these independently borrowed slots disjoint.
    let ptr = slots.as_mut_ptr();
    unsafe {
        let first = (*ptr.add(first)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {first} has not been allocated"
            ))
        })?;
        let second = (*ptr.add(second)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {second} has not been allocated"
            ))
        })?;
        let third = (*ptr.add(third)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {third} has not been allocated"
            ))
        })?;
        let output = (*ptr.add(output)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {output} has not been allocated"
            ))
        })?;
        Ok((first, second, third, output))
    }
}

fn get_five_f32_slots(
    slots: &mut [Option<DeviceBuffer<f32>>],
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
    output: usize,
) -> Result<(
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
)> {
    let indices = [first, second, third, fourth, output];
    if indices.iter().any(|index| *index >= slots.len())
        || indices
            .iter()
            .enumerate()
            .any(|(index, value)| indices[..index].contains(value))
    {
        return Err(AcceleratorError::InvalidArgument(
            "cuda-oxide arena requires five distinct in-range tensor slots".to_string(),
        ));
    }
    // The distinct-index check makes these independently borrowed slots disjoint.
    let ptr = slots.as_mut_ptr();
    unsafe {
        let first = (*ptr.add(first)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA first tensor slot has not been allocated".to_string(),
            )
        })?;
        let second = (*ptr.add(second)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA second tensor slot has not been allocated".to_string(),
            )
        })?;
        let third = (*ptr.add(third)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA third tensor slot has not been allocated".to_string(),
            )
        })?;
        let fourth = (*ptr.add(fourth)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA fourth tensor slot has not been allocated".to_string(),
            )
        })?;
        let output = (*ptr.add(output)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "CUDA output tensor slot has not been allocated".to_string(),
            )
        })?;
        Ok((first, second, third, fourth, output))
    }
}

fn get_three_mut_one_ref_f32_slots(
    slots: &mut [Option<DeviceBuffer<f32>>],
    first: usize,
    second: usize,
    third: usize,
    input: usize,
) -> Result<(
    &mut DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
)> {
    let indices = [first, second, third, input];
    if indices.iter().any(|index| *index >= slots.len())
        || indices
            .iter()
            .enumerate()
            .any(|(index, value)| indices[..index].contains(value))
    {
        return Err(AcceleratorError::InvalidArgument(
            "cuda-oxide arena requires four distinct in-range tensor slots".to_string(),
        ));
    }
    // Each index was proven distinct before forming these simultaneous references.
    let ptr = slots.as_mut_ptr();
    unsafe {
        let first = (*ptr.add(first)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {first} has not been allocated"
            ))
        })?;
        let second = (*ptr.add(second)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {second} has not been allocated"
            ))
        })?;
        let third = (*ptr.add(third)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {third} has not been allocated"
            ))
        })?;
        let input = (*ptr.add(input)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {input} has not been allocated"
            ))
        })?;
        Ok((first, second, third, input))
    }
}

fn get_four_refs_one_mut_f32_slots(
    slots: &mut [Option<DeviceBuffer<f32>>],
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
    output: usize,
) -> Result<(
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
)> {
    let indices = [first, second, third, fourth, output];
    if indices.iter().any(|index| *index >= slots.len())
        || indices
            .iter()
            .enumerate()
            .any(|(index, value)| indices[..index].contains(value))
    {
        return Err(AcceleratorError::InvalidArgument(
            "cuda-oxide arena requires five distinct in-range tensor slots".to_string(),
        ));
    }
    // The index validation establishes all references below are non-overlapping.
    let ptr = slots.as_mut_ptr();
    unsafe {
        let first = (*ptr.add(first)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {first} has not been allocated"
            ))
        })?;
        let second = (*ptr.add(second)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {second} has not been allocated"
            ))
        })?;
        let third = (*ptr.add(third)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {third} has not been allocated"
            ))
        })?;
        let fourth = (*ptr.add(fourth)).as_ref().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {fourth} has not been allocated"
            ))
        })?;
        let output = (*ptr.add(output)).as_mut().ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "CUDA tensor slot {output} has not been allocated"
            ))
        })?;
        Ok((first, second, third, fourth, output))
    }
}
