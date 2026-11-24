# Deep Models

`cartoboost.deep` covers
ordered pickup/dropoff pairs, competing candidate choices, calibrated events,
residual corrections, graph sequences, and spatial scenarios.

Choose a guide by its required inputs and output below. Each guide includes a
Python example, a browser example, validation advice, and limitations. Evidence
varies by model: distinguish an API demonstration, a synthetic mechanism check,
and measured performance on real taxi data.

## Maturity And Evidence

| Model surface | Architecture | Evidence label |
| --- | --- | --- |
| `DirectionalPairForecaster(architecture="pair_embedding_mlp")` | `pair_embedding_mlp` | Synthetic mechanism checks |
| `InvertedTemporalTransformer` | `inverted_transformer` | Synthetic mechanism checks |
| `PropagationDelayGraphForecaster` | `delay_aware_graph_transformer` | Synthetic mechanism checks |
| `ConditionalFlowDistributionHead` | `conditional_residual_sampler` | Synthetic mechanism checks |
| `ChoiceSetTransformer` | `choice_set_utility_softmax` | Synthetic mechanism checks |
| `ResponseCurveModel`, `EventOutcomeModel`, `ServiceTimeResidualModel`, `ConstrainedDecisionOptimizer` | Utility and residual heads | API behavior only |
| `GeoTemporalDiffusionScenarioModel` | `conditional_residual_diffusion` | Experimental |
| `GraphNeuralOperator` | `graph_neural_operator` | Experimental |

See the [Model Capabilities](../reference/capability-matrix.md) table for
backend, parameter, evidence, and maturity details.

## Choose A Guide

| Need | Guide |
| --- | --- |
| Mixed geo-temporal regimes with named experts | [CartoBoost RegimeMoEForecaster](deep-models/cartoboost-regime-moe-forecaster.md) |
| Wide synchronized panels with entity-token attention | [CartoBoost InvertedTemporalTransformer](deep-models/cartoboost-inverted-temporal-transformer.md) |
| Directed graph propagation with known lag priors | [CartoBoost PropagationDelayGraphForecaster](deep-models/cartoboost-propagation-delay-graph-forecaster.md) |
| Joint multi-horizon uncertainty from hidden-state context | [CartoBoost ConditionalFlowDistributionHead](deep-models/cartoboost-conditional-flow-distribution-head.md) |
| Experimental graph-wide residual scenario generation | [CartoBoost GeoTemporalDiffusionScenarioModel](deep-models/cartoboost-geotemporal-diffusion-scenario-model.md) |
| Advanced experimental spatial field-to-field mapping | [CartoBoost GraphNeuralOperator](deep-models/cartoboost-graph-neural-operator.md) |
| Candidate competition and counterfactual best selection | [CartoBoost ChoiceSetTransformer](deep-models/cartoboost-choice-set-transformer.md) |
| Repeated ordered source-target rows | [CartoBoost DirectionalPairForecaster](deep-models/cartoboost-directional-pair-forecaster.md) |
| Candidate values with monotone response | [CartoBoost ResponseCurveModel](deep-models/cartoboost-response-curve-model.md) |
| Calibrated binary event probability | [CartoBoost EventOutcomeModel](deep-models/cartoboost-event-outcome-model.md) |
| Correct a known baseline numeric estimate | [CartoBoost ServiceTimeResidualModel](deep-models/cartoboost-service-time-residual-model.md) |
| Node-time forecasting on directed weighted edges | [CartoBoost SpatioTemporalGraphForecaster](deep-models/cartoboost-spatiotemporal-graph-forecaster.md) |
| Select one candidate per decision group | [CartoBoost ConstrainedDecisionOptimizer](deep-models/cartoboost-constrained-decision-optimizer.md) |

## Backend Choice

CPU is the default. Check the installed build before requesting an accelerator:

```python
from cartoboost.deep import available_deep_backends

print(available_deep_backends())
```

A backend request needs both a compatible build and a usable device. An explicit
request fails when either is unavailable. Read the fitted metadata to see which
backend and operations ran; accelerating one stage does not mean the entire
pipeline ran on the GPU.

| Backend | Environment |
| --- | --- |
| `cpu` | Default execution |
| `metal` | Apple-platform build with Metal support |
| `cuda` | Compatible Windows or Linux build, CUDA driver, and NVRTC |
| `rocm` | Linux or WSL build with ROCm support and a usable HIP device |
| `directml` | Windows build with DirectML support and a DirectX 12 adapter |
| `webgpu` | WebGPU-enabled build with a compatible adapter |

Supported operations depend on the model. Some neural forecasting paths
accelerate dense prediction while retaining training and feature construction
on the CPU. The [capability matrix](../reference/capability-matrix.md) records
backend and model details.

Browser examples run through the Rust/WASM exports. Where a WebGPU path is
available, capability checks perform a real compute pass and readback before
reporting availability. The example's reported backend describes that run.

## Read Model Names Alongside Their Contracts

Several specialized names need a precise interpretation:

| Model | Current behavior |
| --- | --- |
| `ConditionalFlowDistributionHead` | A conditional location/scale residual sampler, not an invertible normalizing flow |
| `ChoiceSetTransformer` | Utility scoring and softmax within a choice set, not candidate-to-candidate attention |
| `GeoTemporalDiffusionScenarioModel` | Experimental residual scenarios around an existing point forecast |
| `GraphNeuralOperator` | Experimental field mapping through graph smoothing, coordinate signals, and temporal changes |

The dedicated guides describe aliases, inputs, outputs, and limitations. Use
the architecture reported by the model when interpreting a result.

## Input Validation

Targets, features, covariates, baselines, candidate values, coordinates, and
edge weights must be finite. Invalid numeric inputs fail at frame construction
or fitting. Handle meaningful missingness explicitly before training, and keep
unknown identifiers distinct when the model supports an unknown-ID policy.

## Validation Defaults

| Model family | Useful comparison |
| --- | --- |
| Ordered pairs | Pair baseline or row-level CartoBoost on the same covariates |
| Response curves | Simple candidate rules, with whole decisions held out |
| Event outcomes | Dummy probabilities, calibration, and threshold-free classification metrics |
| Residual correction | The required baseline alone versus the corrected prediction |
| Graph sequences | Seasonal naive, CartoBoost lag, and a graph-free panel model |
| Decision optimization | The baseline rule, selected utility, and constraint violations |

Use the [deep-model benchmark report](../benchmarks/deep-claims.md) for the
recorded experiments and their limits. A successful browser run establishes
that the example executes; it does not establish performance on your data.
