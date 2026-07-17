#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_compute_backend_selects_cpu() {
        let selection = select_compute_backend(Some("auto")).unwrap();
        assert_eq!(selection.requested, "auto");
        assert_eq!(selection.selected, "cpu");

        let default_selection = select_compute_backend(None).unwrap();
        assert_eq!(default_selection.requested, "auto");
        assert_eq!(default_selection.selected, "cpu");
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_lsttn_fit_uses_tensor_executor_without_scalar_fallback() {
        let backend = match select_compute_backend(Some("cuda")) {
            Ok(backend) => backend,
            Err(_) => return,
        };
        let mut model = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend,
        })
        .unwrap();
        model.fit(&traffic_style_fixture_frame()).unwrap();
        assert!(model
            .trainable_state
            .as_ref()
            .is_some_and(|state| state.steps > 0));
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_lsttn_pretraining_updates_mst_parameters_on_device() {
        if select_compute_backend(Some("cuda")).is_err() {
            return;
        }
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 17);
        state.target_scale = 1.0;
        state.normalized_zero = 0.0;
        let before = state.parameters.clone();
        let mut executor = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        let loss = executor
            .cuda_train_masked_subseries_reconstruction(&mut state, &frame.target[..8], 0.001, 0.0)
            .unwrap();
        assert!(loss.is_finite(), "{loss}");
        assert_eq!(state.steps, 1);
        let layout = state.layout();
        let changed_pretrain = (layout.pretrain_mask_token..layout.total)
            .any(|idx| (state.parameters[idx] - before[idx]).abs() > 1.0e-9);
        assert!(
            changed_pretrain,
            "CUDA pretraining must update MST/pretraining parameters"
        );
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn cuda_lsttn_executor_keeps_model_and_all_three_graphs_resident() {
        if select_compute_backend(Some("cuda")).is_err() {
            return;
        }
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 1, 4, 8, 4, 2, 2, 17);
        let layout = state.layout();
        for offset in [
            layout.lsttn_adaptive_source,
            layout.lsttn_adaptive_target,
            layout.lsttn_weekly_adaptive_source,
            layout.lsttn_weekly_adaptive_target,
            layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_target,
        ] {
            for (idx, value) in state.parameters[offset..offset + state.nodes * state.hidden]
                .iter_mut()
                .enumerate()
            {
                *value = 0.05 + (idx % state.hidden) as f64 * 0.01;
            }
        }
        let mut executor = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        assert_eq!(executor.forward_edges, adjacency.indices.len());
        assert_eq!(executor.reverse_edges, adjacency.indices.len());
        assert!(executor.adaptive_edges >= adjacency.indices.len());
        assert_eq!(executor.node_tiles().collect::<Vec<_>>(), vec![0..3]);
        // Parameters, moments, and six CSR buffers were allocated during the
        // one-time construction; no batch activation has been allocated yet.
        assert!(executor.allocation_count() >= 9);
        executor
            .upload_supervised_batch(&[&frame.target[..8]], &[&frame.target[8..12]], 1)
            .unwrap();
        assert_eq!(
            executor
                .short_input_projection(state.layout(), 1, 8, 3, 1, 4, 4, 0, 1)
                .unwrap(),
            13
        );
        let short = executor
            .short_branch(state.layout(), 1, 8, 1, 4, 4, 0, 1, true, state.steps)
            .unwrap();
        assert!(
            executor.arena.capacity_f32(short).unwrap() >= 12,
            "eight dilated short layers must retain [batch, one step, nodes, hidden]"
        );
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::SUPERVISED_INPUT)
                .unwrap(),
            24
        );
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::SUPERVISED_TARGET)
                .unwrap(),
            12
        );
        executor
            .patch_embedding(state.layout(), 1, 8, 3, 1, 1, 4)
            .unwrap();
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::PATCH_EMBEDDING)
                .unwrap(),
            96
        );
        executor
            .add_patch_positions(state.layout(), 1, 8, 3, 4)
            .unwrap();
        executor.patch_attention_layout(1, 8, 3, 4).unwrap();
        executor
            .frozen_encoder(state.layout(), 1, 8, 3, 4, 2)
            .unwrap();
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::ATTENTION_SEQUENCE)
                .unwrap(),
            96
        );
        let mut cuda_encoder = vec![0.0_f32; 96];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::ATTENTION_SEQUENCE,
                &mut cuda_encoder,
            )
            .unwrap();
        let cpu_encoder =
            state.frozen_lsttn_patch_representations(&frame.target[..8], &adjacency, 0);
        for node in 0..3 {
            for patch in 0..8 {
                for channel in 0..4 {
                    let device = cuda_encoder[(node * 8 + patch) * 4 + channel];
                    let host = cpu_encoder[patch][node][channel];
                    assert!(
                        (device - host).abs() < 2e-4,
                        "encoder mismatch node={node} patch={patch} channel={channel}: {device} vs {host}"
                    );
                }
            }
        }
        executor
            .adaptive_weights(
                layout.lsttn_short_adaptive_source,
                layout.lsttn_short_adaptive_target,
            )
            .unwrap();
        let mut adaptive_weights = vec![0.0_f32; executor.adaptive_edges];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::ADAPTIVE_WEIGHTS,
                &mut adaptive_weights,
            )
            .unwrap();
        let adaptive = adjacency.with_adaptive_self_candidates(3);
        for row in 0..3 {
            let sum = adaptive_weights[adaptive.indptr[row]..adaptive.indptr[row + 1]]
                .iter()
                .sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-5);
        }
        let long = executor.long_branch(layout, 1, 8, 3, 4).unwrap();
        assert!(
            executor.arena.capacity_f32(long).unwrap() >= 12,
            "long branch must retain [batch, final_time, nodes, hidden]"
        );
        executor
            .periodic_feature(
                layout,
                CudaLsttnTensorExecutor::PERIODIC_SHORT,
                1,
                8,
                4,
                1,
                false,
            )
            .unwrap();
        executor
            .periodic_feature(
                layout,
                CudaLsttnTensorExecutor::PERIODIC_SEASONAL,
                1,
                8,
                4,
                2,
                true,
            )
            .unwrap();
        assert_eq!(
            executor
                .arena
                .capacity_f32(CudaLsttnTensorExecutor::PERIODIC_SHORT)
                .unwrap(),
            12
        );
        let direct = executor
            .fuse_and_direct_output(layout, long, short, 1, 4, 4)
            .unwrap();
        assert_eq!(executor.arena.capacity_f32(direct).unwrap(), 12);
        let direct = executor.supervised_forward(&state, 1, 1, 0, true).unwrap();
        assert_eq!(executor.arena.capacity_f32(direct).unwrap(), 12);
        executor.direct_head_loss_and_backward(&state, 1).unwrap();
        executor.fusion_backward(&state, 1).unwrap();
        executor
            .short_branch_backward(&state, 1, 1, 0, true)
            .unwrap();
        let mut gradient_after_short = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_after_short,
            )
            .unwrap();
        let short_width =
            3 * state.hidden + 8 * (12 * state.hidden * state.hidden + 6 * state.hidden);
        assert!(
            gradient_after_short[layout.lsttn_short_wave..layout.lsttn_short_wave + short_width]
                .iter()
                .any(|value| value.abs() > 1.0e-8),
            "Graph WaveNet short-branch reverse must accumulate short-wave gradients"
        );
        executor.long_branch_backward(&state, 1).unwrap();
        let mut gradient_after_long = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_after_long,
            )
            .unwrap();
        assert!(
            gradient_after_long[layout.lsttn_dilated_convolution
                ..layout.lsttn_dilated_convolution
                    + 4 * (3 * state.hidden * state.hidden + state.hidden)]
                .iter()
                .any(|value| value.abs() > 1.0e-8),
            "long branch reverse must accumulate dilated-convolution gradients"
        );
        executor
            .periodic_projection_backward(&state, 1, false)
            .unwrap();
        executor
            .periodic_projection_backward(&state, 1, true)
            .unwrap();
        let mut gradient_before_periodic_graph = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_before_periodic_graph,
            )
            .unwrap();
        executor
            .arena
            .deterministic_dropout_f32(
                CudaLsttnTensorExecutor::PERIODIC_SHORT_INPUT,
                CudaLsttnTensorExecutor::PERIODIC_SHORT_INPUT_GRADIENT,
                state.nodes * 7 * state.hidden,
                0,
                0,
                false,
                0.0,
            )
            .unwrap();
        executor
            .arena
            .deterministic_dropout_f32(
                CudaLsttnTensorExecutor::PERIODIC_SEASONAL_INPUT,
                CudaLsttnTensorExecutor::PERIODIC_SEASONAL_INPUT_GRADIENT,
                state.nodes * 7 * state.hidden,
                0,
                0,
                false,
                0.0,
            )
            .unwrap();
        executor
            .periodic_graph_backward(&state, 1, 1, false)
            .unwrap();
        executor
            .periodic_graph_backward(&state, 1, 1, true)
            .unwrap();
        let mut gradient_after_periodic_graph = vec![0.0_f32; state.parameters.len()];
        executor
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut gradient_after_periodic_graph,
            )
            .unwrap();
        let adaptive_gradient_changed = [
            layout.lsttn_adaptive_source,
            layout.lsttn_adaptive_target,
            layout.lsttn_weekly_adaptive_source,
            layout.lsttn_weekly_adaptive_target,
        ]
        .into_iter()
        .any(|offset| {
            gradient_after_periodic_graph[offset..offset + state.nodes * state.hidden]
                .iter()
                .zip(&gradient_before_periodic_graph[offset..offset + state.nodes * state.hidden])
                .any(|(after, before)| (after - before).abs() > 1.0e-8)
        });
        assert!(
            adaptive_gradient_changed,
            "periodic graph reverse must backpropagate through adaptive CSR logits"
        );
        let mut loss = [0.0_f32; 2];
        executor
            .arena
            .download_f32(CudaLsttnTensorExecutor::SUPERVISED_LOSS, &mut loss)
            .unwrap();
        assert!(loss[0].is_finite());
        let expected = state
            .parameters
            .iter()
            .map(|value| *value as f32 as f64)
            .collect::<Vec<_>>();
        let mut checkpoint_state = state.clone();
        checkpoint_state.parameters.fill(0.0);
        checkpoint_state.first_moment.fill(0.0);
        checkpoint_state.second_moment.fill(0.0);
        executor
            .synchronize_portable_state(&mut checkpoint_state)
            .unwrap();
        assert_eq!(checkpoint_state.parameters, expected);

        let mut supervised = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        supervised
            .upload_supervised_batch(&[&frame.target[..8]], &[&frame.target[8..12]], 1)
            .unwrap();
        supervised
            .supervised_forward(&state, 1, 1, 0, true)
            .unwrap();
        supervised
            .supervised_backward(&state, 1, 1, 0, true)
            .unwrap();
        let mut supervised_gradient = vec![0.0_f32; state.parameters.len()];
        supervised
            .arena
            .download_f32(
                CudaLsttnTensorExecutor::PARAMETER_GRADIENT,
                &mut supervised_gradient,
            )
            .unwrap();
        assert!(
            supervised_gradient.iter().any(|value| value.abs() > 1.0e-8),
            "consolidated CUDA supervised backward must accumulate model gradients"
        );

        let mut tiled = CudaLsttnTensorExecutor::new(&state, &adjacency).unwrap();
        tiled
            .upload_supervised_node_tile(&[&frame.target[..8]], &[&frame.target[8..12]], 1, 1..3)
            .unwrap();
        let mut input_tile = vec![0.0_f32; 16];
        let mut target_tile = vec![0.0_f32; 8];
        tiled
            .arena
            .download_f32(CudaLsttnTensorExecutor::SUPERVISED_INPUT, &mut input_tile)
            .unwrap();
        tiled
            .arena
            .download_f32(CudaLsttnTensorExecutor::SUPERVISED_TARGET, &mut target_tile)
            .unwrap();
        let expected_input = frame.target[..8]
            .iter()
            .flat_map(|row| row[1..3].iter().map(|value| *value as f32))
            .collect::<Vec<_>>();
        let expected_target = frame.target[8..12]
            .iter()
            .flat_map(|row| row[1..3].iter().map(|value| *value as f32))
            .collect::<Vec<_>>();
        assert_eq!(input_tile, expected_input);
        assert_eq!(target_tile, expected_target);
    }

    #[test]
    fn webgpu_compute_backend_matches_runtime_availability() {
        let available = available_compute_backends()
            .iter()
            .any(|backend| backend == "webgpu");
        match select_compute_backend(Some("webgpu")) {
            Ok(selection) => {
                assert!(available);
                assert_eq!(selection.selected, "webgpu");
            }
            Err(error) => {
                assert!(!available);
                assert!(error.to_string().contains("not available in this build"));
            }
        }
    }

    #[test]
    fn paper_graph_transformer_profiles_fit_predict_and_round_trip() {
        let frame = traffic_style_fixture_frame();
        for profile in [
            GraphTransformerProfile::HeterogeneousMoE,
            GraphTransformerProfile::EfficientHighOrder,
            GraphTransformerProfile::LongShortFusion,
            GraphTransformerProfile::GatedGraphTemporal,
            GraphTransformerProfile::SpatialShiftGraphonMoE,
        ] {
            let mut model = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
                profile: profile.clone(),
                lookback: 8,
                hidden_size: 8,
                attention_heads: 2,
                graph_order: 2,
                experts: 3,
                periodicity: if profile == GraphTransformerProfile::LongShortFusion {
                    1
                } else {
                    6
                },
                recent_window: 4,
                epochs: 8,
                learning_rate: 0.01,
                weight_decay: 0.0,
                backend: select_compute_backend(Some("cpu")).unwrap(),
            })
            .unwrap();
            model.fit(&frame).unwrap();
            let prediction = model.predict(4).unwrap();
            assert_eq!(prediction.len(), 4);
            assert!(prediction.iter().flatten().all(|value| value.is_finite()));
            let restored =
                PaperGraphTransformerForecaster::from_json_string(&model.to_json_string().unwrap())
                    .unwrap();
            for (actual, expected) in restored.predict(4).unwrap().iter().zip(&prediction) {
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            if profile == GraphTransformerProfile::SpatialShiftGraphonMoE {
                assert_eq!(model.architecture_report().graphon_expert_count, 3);
            }
            if profile == GraphTransformerProfile::LongShortFusion {
                let supervised_examples = 48usize - 8 - 4 + 1;
                let supervised_steps = supervised_examples.div_ceil(32) * 8;
                let pretraining_windows = (48usize - 8).div_ceil(8) + 1;
                let pretraining_steps = pretraining_windows * (8 / 4);
                assert_eq!(
                    model.trainable_state.as_ref().unwrap().steps as usize,
                    supervised_steps + pretraining_steps
                );
            }
            let report = model.architecture_report();
            let required_component = match profile {
                GraphTransformerProfile::HeterogeneousMoE => "moe_load_balancing_loss",
                GraphTransformerProfile::EfficientHighOrder => {
                    "recursive_pointwise_high_order_interaction"
                }
                GraphTransformerProfile::LongShortFusion => {
                    "seventy_five_percent_masked_patch_pretraining"
                }
                GraphTransformerProfile::GatedGraphTemporal => "normalized_graph_convolution",
                GraphTransformerProfile::SpatialShiftGraphonMoE => {
                    "maximum_spatiotemporal_graph_division"
                }
            };
            assert!(report
                .components
                .iter()
                .any(|component| component == required_component));
        }
    }

    #[test]
    fn lsttn_checkpoint_is_resumable_and_fingerprint_guarded() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 2,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("lsttn-checkpoint.json");
        let mut first = PaperGraphTransformerForecaster::new(config.clone()).unwrap();
        first.fit_checkpointed(&frame, &checkpoint).unwrap();
        let first_prediction = first.predict(4).unwrap();
        let first_steps = first.trainable_state.as_ref().unwrap().steps;

        let mut resumed = PaperGraphTransformerForecaster::new(config).unwrap();
        resumed.fit_checkpointed(&frame, &checkpoint).unwrap();
        assert_eq!(resumed.trainable_state.as_ref().unwrap().steps, first_steps);
        assert_eq!(resumed.predict(4).unwrap(), first_prediction);

        let mut changed = frame.clone();
        changed.target[0][0] += 1.0;
        assert!(resumed.fit_checkpointed(&changed, checkpoint).is_err());
    }

    #[test]
    fn frozen_lsttn_patch_cache_preserves_trainable_gradients() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 8, 4, 2, 2, 41);
        let cache = state.frozen_lsttn_patch_representations(&frame.target[..8], &adjacency, 0);
        let (uncached_loss, uncached) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            None,
        );
        let (cached_loss, cached) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            Some(&cache),
            None,
        );
        assert!((cached_loss - uncached_loss).abs() < 1e-5);
        let layout = state.layout();
        for (actual, expected) in cached[layout.spatial_q..layout.pretrain_position]
            .iter()
            .zip(&uncached[layout.spatial_q..layout.pretrain_position])
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn graph_transformer_tape_backpropagates_through_attention_parameters() {
        let tape = AutodiffTape::new();
        let parameter = tape.parameter(0, 3.0);
        let squared = tape.mul(parameter, parameter);
        let gradients = tape.backward(squared, 1);
        assert!((gradients[0] - 6.0).abs() < 1e-12);

        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
            ..PaperGraphTransformerConfig::default()
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for range in [
            layout.temporal_q..layout.temporal_k,
            layout.spatial_q..layout.spatial_k,
            layout.router..layout.expert_heads,
        ] {
            assert!(trained.parameters[range.clone()]
                .iter()
                .zip(&initial.parameters[range])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn attention_projection_ranges_include_independent_biases() {
        let layout = GraphParameterLayout::new(3, 4, 2, 2, 1, 6, 8);
        let width = 4 * (4 + 1);
        let block_width = 4 * width;
        assert_eq!(layout.temporal_k, layout.temporal_q + block_width);
        assert_eq!(layout.temporal_v, layout.temporal_k + block_width);
        assert_eq!(layout.spatial_q, layout.temporal_v + block_width);
        assert_eq!(layout.spatial_k, layout.spatial_q + block_width);
        assert_eq!(layout.spatial_v, layout.spatial_k + block_width);
        assert_eq!(layout.shortest_path_bias, layout.spatial_v + block_width);
        assert_eq!((layout.pretrain_decoder - layout.pretrain_position) / 4, 8);
    }

    #[test]
    fn periodic_phase_is_stable_across_sliding_window_origins() {
        let absolute_step = 37;
        let first_window = periodic_phase(12 + (absolute_step - 12), 24);
        let second_window = periodic_phase(29 + (absolute_step - 29), 24);

        assert!((first_window - second_window).abs() < 1e-12);
    }

    #[test]
    fn efficient_high_order_profile_uses_stgformer_scaling_normalized_attention() {
        let frame = traffic_style_fixture_frame();
        let state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 8, 2, 2, 2, 19);
        let adjacency = frame.adjacency.row_normalized();
        let window = &frame.target[..8];
        let linear = state.predict_window(
            &GraphTransformerProfile::EfficientHighOrder,
            window,
            &adjacency,
        );
        let softmax = state.predict_window(
            &GraphTransformerProfile::HeterogeneousMoE,
            window,
            &adjacency,
        );
        assert!(linear.iter().flatten().all(|value| value.is_finite()));
        assert!(softmax.iter().flatten().all(|value| value.is_finite()));
        assert!(linear
            .iter()
            .flatten()
            .zip(softmax.iter().flatten())
            .any(|(left, right)| (left - right).abs() > 1e-12));
    }

    #[test]
    fn stgformer_trains_each_order_pointwise_interaction() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::EfficientHighOrder,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for order in 0..2 {
            let start = layout.stgformer_pointwise + order * 4 * 5;
            assert!(trained.parameters[start..start + 20]
                .iter()
                .zip(&initial.parameters[start..start + 20])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn stgformer_fast_attention_reuses_summary_and_keeps_query_value_residual() {
        let tape = AutodiffTape::new();
        let keys = vec![
            vec![tape.constant(1.0), tape.constant(0.0)],
            vec![tape.constant(0.0), tape.constant(1.0)],
        ];
        let values = vec![
            vec![tape.constant(2.0), tape.constant(3.0)],
            vec![tape.constant(5.0), tape.constant(7.0)],
        ];
        let summary = tape_stgformer_attention_summary(&tape, &keys, &values);
        let output = tape_stgformer_fast_attention(
            &tape,
            &[tape.constant(1.0), tape.constant(0.0)],
            &summary,
            &values[1],
        );
        // Q(K^T V) + N * V_query over Q(sum K) + N, for N = 2.
        assert!((tape.value(output[0]) - 4.0).abs() < 1e-12);
        assert!((tape.value(output[1]) - 17.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn attention_and_router_softmax_preserve_true_logit_ratios() {
        let tape = AutodiffTape::new();
        let weights = tape_softmax(&tape, &[tape.constant(0.0), tape.constant(2.0)]);
        let ratio = tape.value(weights[1]) / tape.value(weights[0]);
        assert!((ratio - 2.0_f64.exp()).abs() < 1e-12);
    }

    #[test]
    fn training_tape_interns_each_model_parameter_once() {
        let tape = AutodiffTape::new();
        let parameters = [0.25, -0.5];
        let parameter_nodes = parameters
            .iter()
            .enumerate()
            .map(|(index, value)| tape.parameter(index, *value))
            .collect::<Vec<_>>();

        let first_use = tape.mul(parameter_nodes[0], tape.constant(2.0));
        let second_use = tape.add(parameter_nodes[0], parameter_nodes[1]);
        let loss = tape.add(first_use, second_use);
        let gradients = tape.backward(loss, parameters.len());

        assert_eq!(parameter_nodes.len(), parameters.len());
        assert_eq!(tape.nodes.borrow().len(), 6);
        assert_eq!(gradients, vec![3.0, 1.0]);
    }

    #[test]
    fn adaptive_diffusion_uses_csr_neighbors_with_isolated_node_fallback() {
        let adjacency =
            CsrAdjacency::new(vec![0, 2, 3, 3], vec![1, 2, 0], vec![1.0; 3], 3).unwrap();
        let first_fallback = [0];
        let isolated_fallback = [2];

        assert_eq!(
            adaptive_neighbor_indices(&adjacency, 0, &first_fallback),
            &[1, 2]
        );
        assert_eq!(
            adaptive_neighbor_indices(&adjacency, 2, &isolated_fallback),
            &[2]
        );

        let adaptive = adjacency.with_adaptive_self_candidates(3);
        assert_eq!(adaptive.indptr, vec![0, 3, 5, 6]);
        assert_eq!(adaptive.indices, vec![1, 2, 0, 0, 1, 2]);
        // The learned adaptive support remains O(E + N), not N².
        assert_eq!(adaptive.indices.len(), adjacency.indices.len() + 3);
    }

    #[test]
    fn maximum_graph_division_groups_contiguous_rank_stable_environments() {
        let values = [
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 2.0, 1.0],
            vec![6.0, 4.0, 2.0],
        ]
        .into_iter()
        .cycle()
        .take(16)
        .collect::<Vec<_>>();
        assert_eq!(
            maximum_spatiotemporal_graph_division(&values, 4, 2).unwrap(),
            vec![0, 0, 1, 1]
        );
    }

    #[test]
    fn maximum_graph_division_rejects_an_incomplete_cycle() {
        let error = maximum_spatiotemporal_graph_division(&[vec![1.0, 2.0]], 2, 2)
            .expect_err("one observation cannot define a two-step graph cycle");
        assert!(error.to_string().contains("complete period"));
    }

    #[test]
    fn episodic_graphon_detach_stops_expert_gradient() {
        let tape = AutodiffTape::new();
        let expert = tape.parameter(0, 3.0);
        let detached = tape_detach_vectors(&tape, &[expert]);
        let loss = tape.mul(detached[0], detached[0]);
        assert_eq!(tape.backward(loss, 1), vec![0.0]);
    }

    #[test]
    fn stgormer_trains_degree_and_shortest_path_embedding_tables() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::HeterogeneousMoE,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for range in [
            layout.in_degree_embedding..layout.out_degree_embedding,
            layout.temporal_q..layout.temporal_k,
            layout.temporal_k..layout.temporal_v,
            layout.temporal_v..layout.spatial_q,
            layout.spatial_q..layout.spatial_k,
            layout.spatial_k..layout.spatial_v,
            layout.spatial_v..layout.shortest_path_bias,
            layout.shortest_path_bias..layout.router,
            layout.router..layout.spatial_router,
            layout.spatial_router..layout.spatial_expert_heads,
            layout.spatial_expert_heads..layout.expert_heads,
        ] {
            assert!(trained.parameters[range.clone()]
                .iter()
                .zip(&initial.parameters[range])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn long_short_profile_trains_dynamic_periodic_graph_parameters() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for range in [
            layout.lsttn_adaptive_source..layout.lsttn_adaptive_target,
            layout.lsttn_adaptive_target..layout.lsttn_weekly_adaptive_source,
            layout.lsttn_weekly_adaptive_source..layout.lsttn_weekly_adaptive_target,
            layout.lsttn_weekly_adaptive_target..layout.lsttn_short_adaptive_source,
            layout.lsttn_short_adaptive_source..layout.lsttn_short_adaptive_target,
            layout.lsttn_short_adaptive_target..layout.lsttn_periodic_projection,
        ] {
            assert!(trained.parameters[range.clone()]
                .iter()
                .zip(&initial.parameters[range])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
        assert!(trained.parameters[layout.pretrain_mask_token..layout.total]
            .iter()
            .zip(&initial.parameters[layout.pretrain_mask_token..layout.total])
            .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        assert!(
            trained.parameters[layout.lsttn_fusion..layout.graphon_nodes]
                .iter()
                .zip(&initial.parameters[layout.lsttn_fusion..layout.graphon_nodes])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
        assert!(
            trained.parameters[layout.lsttn_dilated_convolution..layout.stgformer_pointwise]
                .iter()
                .zip(
                    &initial.parameters
                        [layout.lsttn_dilated_convolution..layout.stgformer_pointwise],
                )
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
        assert!(
            trained.parameters[layout.lsttn_short_wave..layout.stgformer_pointwise]
                .iter()
                .zip(&initial.parameters[layout.lsttn_short_wave..layout.stgformer_pointwise],)
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12)
        );
    }

    #[test]
    fn long_short_profile_rejects_recent_context_larger_than_history() {
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 24,
            recent_window: 25,
            ..PaperGraphTransformerConfig::default()
        };

        assert!(PaperGraphTransformerForecaster::new(config).is_err());
    }

    #[test]
    fn long_short_fit_requires_a_real_weekly_transformer_state() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            periodicity: 2,
            recent_window: 4,
            epochs: 1,
            ..PaperGraphTransformerConfig::default()
        };
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        let error = model.fit(&frame).unwrap_err();
        assert!(error.to_string().contains("exceed its seasonal lag"));
    }

    #[test]
    fn long_short_report_exposes_every_paper_spatial_temporal_branch() {
        let model = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 24 * 14,
            periodicity: 24,
            recent_window: 12,
            ..PaperGraphTransformerConfig::default()
        })
        .unwrap();
        let report = model.architecture_report();
        for component in [
            "seventy_five_percent_masked_patch_pretraining",
            "four_layer_multihead_transformer_encoder",
            "four_stage_dilated_long_trend_convolution",
            "previous_day_and_week_transformer_states",
            "independent_forward_backward_adaptive_periodic_graph_convolutions",
            "eight_layer_causal_graph_wavenet_short_branch",
            "signal_and_time_of_day_short_term_channels",
            "long_periodic_short_feature_fusion",
            "all_origin_thirty_two_window_supervision",
        ] {
            assert!(report.components.iter().any(|actual| actual == component));
        }
    }

    #[test]
    fn long_short_graph_wavenet_consumes_supplied_time_of_day_channel() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let state = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            1,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let first = vec![vec![0.0; 3]; 8];
        let second = (0..8)
            .map(|time| vec![time as f64 / 8.0; 3])
            .collect::<Vec<_>>();
        let (first_loss, _) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            Some(&first),
        );
        let (second_loss, second_gradients) = state.lsttn_example_loss_and_gradients(
            &frame.target[..8],
            &adjacency,
            &frame.target[8..12],
            0,
            None,
            Some(&second),
        );
        let layout = state.layout();
        assert!(second_gradients
            [layout.lsttn_short_wave + state.hidden..layout.lsttn_short_wave + 2 * state.hidden]
            .iter()
            .any(|gradient| gradient.abs() > 0.0));
        assert!(first_loss.is_finite() && second_loss.is_finite());
    }

    #[test]
    fn long_short_supervised_path_freezes_pretrained_context_encoder() {
        let frame = traffic_style_fixture_frame();
        let adjacency = frame.adjacency.row_normalized();
        let mut state = TrainableGraphTransformerState::initialized(3, 4, 2, 6, 4, 8, 4, 2, 2, 29);
        let initial = state.parameters.clone();
        let layout = state.layout();

        state
            .train_example_with_context(
                &GraphTransformerProfile::LongShortFusion,
                &frame.target[..8],
                &adjacency,
                &frame.target[8..12],
                None,
                0.01,
                0.0,
                0,
                false,
                None,
            )
            .unwrap();

        for range in [
            layout.temporal_q..layout.temporal_k,
            layout.temporal_k..layout.temporal_v,
            layout.temporal_v..layout.spatial_q,
            layout.pretrain_position..layout.pretrain_decoder,
        ] {
            assert!(state.parameters[range.clone()]
                .iter()
                .zip(&initial[range])
                .all(|(trained, initial)| (trained - initial).abs() < 1e-12));
        }
    }

    #[test]
    fn long_short_direct_horizon_decoder_does_not_rebase_predictions() {
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 2,
            hidden_size: 1,
            attention_heads: 1,
            graph_order: 1,
            experts: 1,
            periodicity: 1,
            recent_window: 2,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let mut state = TrainableGraphTransformerState::initialized(1, 1, 1, 1, 2, 2, 3, 1, 1, 31);
        state.parameters.fill(0.0);
        let layout = state.layout();
        for (horizon, delta) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            state.parameters[layout.output + horizon * 2 + 1] = delta;
        }
        let model = PaperGraphTransformerForecaster {
            config,
            node_ids: vec!["lane".into()],
            frequency: "hourly".into(),
            horizon: 3,
            adjacency: Some(CsrAdjacency::new(vec![0, 0], vec![], vec![], 1).unwrap()),
            trainable_state: Some(state),
            history: vec![vec![9.0], vec![10.0]],
            history_time_features: None,
            target_mean: 0.0,
            target_scale: 1.0,
        };

        assert_eq!(
            model.predict(3).unwrap(),
            vec![vec![1.0], vec![2.0], vec![3.0]]
        );
        let mut artifact: serde_json::Value =
            serde_json::from_str(&model.to_json_string().unwrap()).unwrap();
        artifact["trainable_state"]["parameters"]
            .as_array_mut()
            .unwrap()
            .pop();
        let error = PaperGraphTransformerForecaster::from_json_string(
            &serde_json::to_string(&artifact).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("incompatible"));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn long_short_full_metal_inference_matches_cpu_predictions() {
        let frame = traffic_style_fixture_frame();
        let mut cpu = PaperGraphTransformerForecaster::new(PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        })
        .unwrap();
        cpu.fit(&frame).unwrap();
        let expected = cpu.predict(4).unwrap();

        let mut metal = cpu.clone();
        metal.config.backend = select_compute_backend(Some("metal")).unwrap();
        let actual = metal.predict(4).unwrap();

        for (actual_row, expected_row) in actual.iter().zip(expected) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-4);
            }
        }
        assert!(metal
            .architecture_report()
            .components
            .iter()
            .any(|component| component == "metal_full_graph_training_and_inference"));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn long_short_metal_trains_and_predicts_complete_graph() {
        let mut frame = traffic_style_fixture_frame();
        frame.timestamps.truncate(13);
        frame.target.truncate(13);
        let config = |backend| PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::LongShortFusion,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 1,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some(backend)).unwrap(),
        };
        let mut cpu = PaperGraphTransformerForecaster::new(config("cpu")).unwrap();
        let mut metal = PaperGraphTransformerForecaster::new(config("metal")).unwrap();
        cpu.fit(&frame).unwrap();
        metal.fit(&frame).unwrap();
        let expected = cpu.predict(4).unwrap();
        let actual = metal.predict(4).unwrap();

        for (actual_row, expected_row) in actual.iter().zip(expected) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-1, "{actual} != {expected}");
            }
        }
        let state = metal.trainable_state.as_ref().unwrap();
        assert!(state.steps > 0);
        assert!(state.first_moment.iter().any(|value| value.abs() > 0.0));
        assert!(metal
            .architecture_report()
            .components
            .iter()
            .any(|component| component == "metal_full_graph_training_and_inference"));
    }

    #[test]
    fn spatial_shift_profile_trains_each_graphon_expert_before_router_mixup() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::SpatialShiftGraphonMoE,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 2,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        for expert in 0..2 {
            let start = layout.graphon_nodes + expert * 3;
            assert!(trained.parameters[start..start + 3]
                .iter()
                .zip(&initial.parameters[start..start + 3])
                .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
        }
    }

    #[test]
    fn spatial_shift_prediction_reuses_fitted_experts_without_test_time_updates() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::SpatialShiftGraphonMoE,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let before = model.to_json_string().unwrap();
        let prediction = model.predict(3).unwrap();
        assert!(prediction.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(model.to_json_string().unwrap(), before);
    }

    #[test]
    fn graphon_gumbel_sampling_is_step_seeded_and_finite() {
        let first = graphon_gumbel_logistic_noise(7, 1, 2, 3, 4);
        let repeated = graphon_gumbel_logistic_noise(7, 1, 2, 3, 4);
        let next_step = graphon_gumbel_logistic_noise(8, 1, 2, 3, 4);
        assert!(first.is_finite());
        assert_eq!(first, repeated);
        assert_ne!(first, next_step);
    }

    #[test]
    fn gated_profile_trains_the_graph_convolution_projection() {
        let frame = traffic_style_fixture_frame();
        let config = PaperGraphTransformerConfig {
            profile: GraphTransformerProfile::GatedGraphTemporal,
            lookback: 8,
            hidden_size: 4,
            attention_heads: 2,
            graph_order: 2,
            experts: 2,
            periodicity: 6,
            recent_window: 4,
            epochs: 1,
            learning_rate: 0.01,
            weight_decay: 0.0,
            backend: select_compute_backend(Some("cpu")).unwrap(),
        };
        let initial = TrainableGraphTransformerState::initialized(
            3,
            4,
            2,
            6,
            4,
            8,
            4,
            2,
            2,
            0x5354_474d_4f45,
        );
        let mut model = PaperGraphTransformerForecaster::new(config).unwrap();
        model.fit(&frame).unwrap();
        let trained = model.trainable_state.as_ref().unwrap();
        let layout = trained.layout();
        assert!(trained.parameters[layout.spatial_v..layout.router]
            .iter()
            .zip(&initial.parameters[layout.spatial_v..layout.router])
            .any(|(actual, initial)| (actual - initial).abs() > 1e-12));
    }

    #[test]
    fn reconstruction_masks_are_patchwise_reproducible_and_seventy_five_percent() {
        let first = masked_patch_indices(8, 41);
        assert_eq!(first.len(), 6);
        assert_eq!(first, masked_patch_indices(8, 41));
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(first.iter().all(|index| *index < 8));
        assert_eq!(masked_patch_indices(2, 41).len(), 1);
        assert_eq!(masked_patch_indices(3, 41).len(), 2);
    }
}
