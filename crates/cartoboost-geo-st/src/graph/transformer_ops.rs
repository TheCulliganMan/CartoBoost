fn tape_linear(
    tape: &AutodiffTape,
    parameter_nodes: &[usize],
    offset: usize,
    input: &[usize],
    input_width: usize,
    output_width: usize,
) -> Vec<usize> {
    (0..output_width)
        .map(|output| {
            let mut value = parameter_nodes[offset + input_width * output_width + output];
            for (index, input_value) in input.iter().enumerate().take(input_width) {
                value = tape.add(
                    value,
                    tape.mul(
                        parameter_nodes[offset + index * output_width + output],
                        *input_value,
                    ),
                );
            }
            value
        })
        .collect()
}

fn numeric_linear(
    parameters: &[f64],
    offset: usize,
    input: &[f64],
    input_width: usize,
    output_width: usize,
) -> Vec<f64> {
    (0..output_width)
        .map(|output| {
            input.iter().enumerate().take(input_width).fold(
                parameters[offset + input_width * output_width + output],
                |sum, (index, value)| {
                    sum + parameters[offset + index * output_width + output] * value
                },
            )
        })
        .collect()
}

fn numeric_layer_norm(parameters: &[f64], offset: usize, values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let denominator = (variance + 1e-5).sqrt();
    values
        .iter()
        .enumerate()
        .map(|(channel, value)| {
            (value - mean) / denominator * parameters[offset + channel]
                + parameters[offset + values.len() + channel]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn numeric_transformer_encoder_layer(
    parameters: &[f64],
    sequence: &[Vec<f64>],
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    out_offset: usize,
    ffn_offset: usize,
    norm_offset: usize,
    hidden: usize,
    heads: usize,
) -> Vec<Vec<f64>> {
    let queries = sequence
        .iter()
        .map(|token| numeric_linear(parameters, q_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let keys = sequence
        .iter()
        .map(|token| numeric_linear(parameters, k_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let values = sequence
        .iter()
        .map(|token| numeric_linear(parameters, v_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    sequence
        .iter()
        .enumerate()
        .map(|(token_index, residual)| {
            let mut attended = vec![0.0; hidden];
            for head in 0..heads {
                let start = head * hidden / heads;
                let end = (head + 1) * hidden / heads;
                let scale = 1.0 / ((end - start) as f64).sqrt();
                let logits = keys
                    .iter()
                    .map(|key| {
                        queries[token_index][start..end]
                            .iter()
                            .zip(&key[start..end])
                            .map(|(left, right)| left * right)
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut weights = logits
                    .iter()
                    .map(|value| (value - max).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f64>().max(1e-12);
                for weight in &mut weights {
                    *weight /= denominator;
                }
                for channel in start..end {
                    attended[channel] = weights
                        .iter()
                        .zip(&values)
                        .map(|(weight, value)| weight * value[channel])
                        .sum();
                }
            }
            let projected = numeric_linear(parameters, out_offset, &attended, hidden, hidden);
            let first = residual
                .iter()
                .zip(projected)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>();
            let normalized = numeric_layer_norm(parameters, norm_offset, &first);
            let expanded = numeric_linear(parameters, ffn_offset, &normalized, hidden, 4 * hidden)
                .into_iter()
                .map(|value| value.max(0.0))
                .collect::<Vec<_>>();
            let contracted = numeric_linear(
                parameters,
                ffn_offset + (hidden + 1) * 4 * hidden,
                &expanded,
                4 * hidden,
                hidden,
            );
            numeric_layer_norm(
                parameters,
                norm_offset + 2 * hidden,
                &normalized
                    .iter()
                    .zip(contracted)
                    .map(|(left, right)| left + right)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn tape_layer_norm(
    tape: &AutodiffTape,
    parameters: &[usize],
    offset: usize,
    values: &[usize],
) -> Vec<usize> {
    let width = values.len();
    let inverse = tape.constant(1.0 / width as f64);
    let mean = values.iter().fold(tape.constant(0.0), |sum, value| {
        tape.add(sum, tape.mul(*value, inverse))
    });
    let variance = values.iter().fold(tape.constant(0.0), |sum, value| {
        let centered = tape.add(*value, tape.mul(mean, tape.constant(-1.0)));
        tape.add(sum, tape.mul(tape.mul(centered, centered), inverse))
    });
    let denominator = tape.sqrt(tape.add(variance, tape.constant(1e-5)));
    values
        .iter()
        .enumerate()
        .map(|(channel, value)| {
            let centered = tape.add(*value, tape.mul(mean, tape.constant(-1.0)));
            tape.add(
                tape.mul(
                    tape.div(centered, denominator),
                    parameters[offset + channel],
                ),
                parameters[offset + width + channel],
            )
        })
        .collect()
}

fn tape_deterministic_dropout(
    tape: &AutodiffTape,
    value: usize,
    seed: u64,
    index: usize,
    enabled: bool,
) -> usize {
    tape_deterministic_dropout_rate(tape, value, seed, index, enabled, 0.1)
}

fn tape_deterministic_dropout_rate(
    tape: &AutodiffTape,
    value: usize,
    seed: u64,
    index: usize,
    enabled: bool,
    probability: f64,
) -> usize {
    if !enabled {
        return value;
    }
    let mut state = seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let threshold = (probability * 10_000.0).round() as u64;
    if state % 10_000 < threshold {
        tape.constant(0.0)
    } else {
        tape.mul(value, tape.constant(1.0 / (1.0 - probability)))
    }
}

fn tape_gelu(tape: &AutodiffTape, value: usize) -> usize {
    let cube = tape.mul(tape.mul(value, value), value);
    let inner = tape.mul(
        tape.constant((2.0 / std::f64::consts::PI).sqrt()),
        tape.add(value, tape.mul(tape.constant(0.044715), cube)),
    );
    tape.mul(
        tape.constant(0.5),
        tape.mul(value, tape.add(tape.constant(1.0), tape.tanh(inner))),
    )
}

#[allow(clippy::too_many_arguments)]
fn tape_transformer_encoder_layer(
    tape: &AutodiffTape,
    parameters: &[usize],
    sequence: &[Vec<usize>],
    q_offset: usize,
    k_offset: usize,
    v_offset: usize,
    out_offset: usize,
    ffn_offset: usize,
    norm_offset: usize,
    hidden: usize,
    heads: usize,
    dropout_seed: u64,
    dropout: bool,
) -> Vec<Vec<usize>> {
    let queries = sequence
        .iter()
        .map(|token| tape_linear(tape, parameters, q_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let keys = sequence
        .iter()
        .map(|token| tape_linear(tape, parameters, k_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    let values = sequence
        .iter()
        .map(|token| tape_linear(tape, parameters, v_offset, token, hidden, hidden))
        .collect::<Vec<_>>();
    sequence
        .iter()
        .enumerate()
        .map(|(token_index, residual)| {
            let mut attended = vec![tape.constant(0.0); hidden];
            for head in 0..heads {
                let start = head * hidden / heads;
                let end = (head + 1) * hidden / heads;
                let scale = tape.constant(1.0 / ((end - start) as f64).sqrt());
                let logits = keys
                    .iter()
                    .map(|key| {
                        tape.mul(
                            scale,
                            tape_dot(tape, &queries[token_index][start..end], &key[start..end]),
                        )
                    })
                    .collect::<Vec<_>>();
                let weights = tape_softmax(tape, &logits)
                    .into_iter()
                    .enumerate()
                    .map(|(key_index, weight)| {
                        tape_deterministic_dropout(
                            tape,
                            weight,
                            dropout_seed ^ 0x6a09_e667_f3bc_c909,
                            (token_index * sequence.len() + key_index) * heads + head,
                            dropout,
                        )
                    })
                    .collect::<Vec<_>>();
                let head_values = values
                    .iter()
                    .map(|value| value[start..end].to_vec())
                    .collect::<Vec<_>>();
                attended[start..end].copy_from_slice(&tape_weighted_sum(
                    tape,
                    &weights,
                    &head_values,
                    end - start,
                ));
            }
            let projected = tape_linear(tape, parameters, out_offset, &attended, hidden, hidden);
            let first_residual = residual
                .iter()
                .zip(projected)
                .enumerate()
                .map(|(channel, (skip, value))| {
                    tape.add(
                        *skip,
                        tape_deterministic_dropout(
                            tape,
                            value,
                            dropout_seed,
                            token_index * hidden + channel,
                            dropout,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let normalized = tape_layer_norm(tape, parameters, norm_offset, &first_residual);
            let expanded = tape_linear(
                tape,
                parameters,
                ffn_offset,
                &normalized,
                hidden,
                4 * hidden,
            )
            .into_iter()
            .map(|value| tape.max(value, tape.constant(0.0)))
            .collect::<Vec<_>>();
            let contracted = tape_linear(
                tape,
                parameters,
                ffn_offset + (hidden + 1) * 4 * hidden,
                &expanded,
                4 * hidden,
                hidden,
            );
            let second_residual = normalized
                .iter()
                .zip(contracted)
                .enumerate()
                .map(|(channel, (skip, value))| {
                    tape.add(
                        *skip,
                        tape_deterministic_dropout(
                            tape,
                            value,
                            dropout_seed ^ 0xa5a5_a5a5_a5a5_a5a5,
                            token_index * hidden + channel,
                            dropout,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            tape_layer_norm(tape, parameters, norm_offset + 2 * hidden, &second_residual)
        })
        .collect()
}

fn periodic_phase(absolute_step: usize, period: usize) -> f64 {
    (absolute_step as f64 * std::f64::consts::TAU / period.max(1) as f64).sin()
}

fn tape_dot(tape: &AutodiffTape, left: &[usize], right: &[usize]) -> usize {
    left.iter()
        .zip(right)
        .fold(tape.constant(0.0), |sum, (a, b)| {
            tape.add(sum, tape.mul(*a, *b))
        })
}

fn clip_gradient_norm(gradients: &mut [f64], maximum_norm: f64) {
    let norm = gradients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > maximum_norm {
        let scale = maximum_norm / norm;
        for gradient in gradients {
            *gradient *= scale;
        }
    }
}

/// Logistic noise obtained from the difference of two Gumbel variates.  With
/// the sigmoid relaxation in the graphon branch, this is the binary form of
/// Gumbel-Softmax.  It is deterministic for a serialized optimizer step so a
/// saved model cannot change predictions merely by being reloaded.
fn graphon_gumbel_logistic_noise(
    step: u64,
    expert: usize,
    target: usize,
    source: usize,
    channel: usize,
) -> f64 {
    let mut value = step
        ^ (expert as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (target as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (source as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ (channel as u64).wrapping_mul(0xD6E8_FD50_6A6A_5A93);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let uniform = ((value >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64);
    (uniform / (1.0 - uniform)).ln()
}

/// Summary shared by every query in one STGformer attention head.  Construct
/// this once rather than materializing an N-by-N attention matrix.
fn tape_stgformer_attention_summary(
    tape: &AutodiffTape,
    keys: &[Vec<usize>],
    values: &[Vec<usize>],
) -> (Vec<usize>, Vec<Vec<usize>>, usize) {
    let keys = keys
        .iter()
        .map(|key| tape_l2_normalize(tape, key))
        .collect::<Vec<_>>();
    let width = keys.first().map_or(0, Vec::len);
    let key_sum = (0..width)
        .map(|channel| {
            keys.iter()
                .fold(tape.constant(0.0), |sum, key| tape.add(sum, key[channel]))
        })
        .collect::<Vec<_>>();
    let key_value = (0..key_sum.len())
        .map(|key_channel| {
            (0..key_sum.len())
                .map(|value_channel| {
                    keys.iter()
                        .zip(values)
                        .fold(tape.constant(0.0), |sum, (key, value)| {
                            tape.add(sum, tape.mul(key[key_channel], value[value_channel]))
                        })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (key_sum, key_value, keys.len())
}

/// STGformer's scaling-normalized efficient attention.  The official layer
/// computes `Q(K^T V) + N V` over `Q(sum K) + N`; the residual V belongs to
/// the current query position, not an arbitrary key/value position.
fn tape_stgformer_fast_attention(
    tape: &AutodiffTape,
    query: &[usize],
    summary: &(Vec<usize>, Vec<Vec<usize>>, usize),
    residual_value: &[usize],
) -> Vec<usize> {
    let query = tape_l2_normalize(tape, query);
    let (key_sum, key_value, count) = summary;
    let positions = tape.constant(*count as f64);
    let denominator = tape.add(tape_dot(tape, &query, key_sum), positions);
    (0..query.len())
        .map(|output| {
            let mut numerator = tape.mul(positions, residual_value[output]);
            for channel in 0..query.len() {
                numerator = tape.add(
                    numerator,
                    tape.mul(query[channel], key_value[channel][output]),
                );
            }
            tape.div(numerator, denominator)
        })
        .collect()
}

fn tape_l2_normalize(tape: &AutodiffTape, values: &[usize]) -> Vec<usize> {
    let squared_norm = values.iter().fold(tape.constant(1e-12), |sum, value| {
        tape.add(sum, tape.mul(*value, *value))
    });
    let norm = tape.sqrt(squared_norm);
    values.iter().map(|value| tape.div(*value, norm)).collect()
}

#[cfg(test)]
fn adaptive_neighbor_indices<'a>(
    adjacency: &'a CsrAdjacency,
    node: usize,
    fallback: &'a [usize; 1],
) -> &'a [usize] {
    let neighbors = &adjacency.indices[adjacency.indptr[node]..adjacency.indptr[node + 1]];
    if neighbors.is_empty() {
        fallback
    } else {
        neighbors
    }
}

fn tape_softmax(tape: &AutodiffTape, logits: &[usize]) -> Vec<usize> {
    let max_logit = logits
        .iter()
        .copied()
        .reduce(|left, right| tape.max(left, right))
        .expect("softmax requires at least one logit");
    let shift = tape.mul(tape.constant(-1.0), tape.stop_gradient(max_logit));
    let exponentials = logits
        .iter()
        // Attention and MoE routing use the actual Transformer softmax.  The
        // detached max shift is algebraically invariant and avoids overflow.
        .map(|value| tape.exp(tape.add(*value, shift)))
        .collect::<Vec<_>>();
    let denominator = exponentials
        .iter()
        .fold(tape.constant(0.0), |sum, value| tape.add(sum, *value));
    exponentials
        .into_iter()
        .map(|value| tape.div(value, denominator))
        .collect()
}
fn tape_weighted_sum(
    tape: &AutodiffTape,
    weights: &[usize],
    values: &[Vec<usize>],
    width: usize,
) -> Vec<usize> {
    (0..width)
        .map(|channel| {
            weights
                .iter()
                .zip(values)
                .fold(tape.constant(0.0), |sum, (weight, value)| {
                    tape.add(sum, tape.mul(*weight, value[channel]))
                })
        })
        .collect()
}

fn tape_csr_diffuse(
    tape: &AutodiffTape,
    adjacency: &CsrAdjacency,
    weights: &[usize],
    values: &[Vec<usize>],
    hidden: usize,
) -> Vec<Vec<usize>> {
    (0..values.len())
        .map(|target| {
            (0..hidden)
                .map(|channel| {
                    (adjacency.indptr[target]..adjacency.indptr[target + 1]).fold(
                        tape.constant(0.0),
                        |sum, edge| {
                            tape.add(
                                sum,
                                tape.mul(weights[edge], values[adjacency.indices[edge]][channel]),
                            )
                        },
                    )
                })
                .collect()
        })
        .collect()
}

fn tape_add_vectors(tape: &AutodiffTape, left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .zip(right)
        .map(|(a, b)| tape.add(*a, *b))
        .collect()
}

/// Preserve a graphon value for episodic mixup while intentionally stopping
/// its gradient.  The episode then learns only how to combine independently
/// trained expert graphons for the held-out environment.
fn tape_detach_vectors(tape: &AutodiffTape, values: &[usize]) -> Vec<usize> {
    values
        .iter()
        .map(|value| tape.stop_gradient(*value))
        .collect()
}

