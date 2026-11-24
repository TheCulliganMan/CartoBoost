/* tslint:disable */
/* eslint-disable */

export function availableDeepBackends(): any;

export function availableForecastModels(): any;

export function deepChoiceSetTransformerReport(candidates: any, temperature: number, monotone_candidate_value?: string | null, backend?: string | null): any;

export function deepConditionalFlowFit(hidden: any, residuals: Float64Array, quantiles: Float64Array, sample_count: number, backend?: string | null): any;

export function deepConditionalFlowPredict(artifact: any, hidden: any, actual?: Float64Array | null): any;

/**
 * Conditional-flow inference with both learned affine projections executed
 * by browser WebGPU. Sampling and metrics reuse the native probability
 * contract so this export remains numerically aligned with native backends.
 */
export function deepConditionalFlowPredictWebgpu(artifact: any, hidden: any, actual?: Float64Array | null): Promise<any>;

export function deepConstrainedDecisionSelect(candidates: any, objective: string, constraints: any, fallback: string): any;

export function deepDiffusionScenarioGenerate(point_forecast: any, edges: any, scenario_count: number, diffusion_steps: number, shock_scale: number, backend?: string | null): any;

export function deepDiffusionScenarioGenerateWebgpu(point_forecast: any, edges: any, scenario_count: number, diffusion_steps: number, shock_scale: number): Promise<any>;

export function deepDirectionalPairPredict(rows: any, backend?: string | null): any;

export function deepEventOutcomeFit(features: any, labels: Float64Array, backend?: string | null): any;

export function deepEventOutcomePredict(artifact: any, features: any): any;

/**
 * Predicts event probabilities with the artifact's hidden layer dispatched on
 * WebGPU. The export is asynchronous because browser GPU work is asynchronous.
 */
export function deepEventOutcomePredictWebgpu(artifact: any, features: any): Promise<any>;

export function deepGraphNeuralOperatorPredict(field_values: any, coordinates: any, edges: any, exogenous_fields: any, smoothing: number, coordinate_scale: number, backend?: string | null): any;

/**
 * Graph-neural-operator inference with graph aggregation executed by browser
 * WebGPU. This is async because browsers do not expose synchronous adapter
 * discovery or command completion.
 */
export function deepGraphNeuralOperatorPredictWebgpu(field_values: any, coordinates: any, edges: any, exogenous_fields: any, smoothing: number, coordinate_scale: number): Promise<any>;

export function deepInvertedTemporalFit(y: any, lookback: number, backend?: string | null): any;

export function deepInvertedTemporalPredict(artifact: any, horizon: number, cross_entity: boolean): any;

export function deepNeuralOperatorSyntheticBenchmark(): any;

export function deepRegimeMoeReport(features: any, target: Float64Array): any;

export function deepResponseCurveFit(rows: any, response_type: string, monotone?: string | null, backend?: string | null): any;

export function deepResponseCurvePredict(artifact: any, rows: any): any;

export function deepResponseCurvePredictWebgpu(artifact: any, rows: any): Promise<any>;

export function deepServiceResidualFit(rows: any, backend?: string | null): any;

export function deepServiceResidualPredict(artifact: any, rows: any): any;

export function deepServiceResidualPredictWebgpu(artifact: any, rows: any): Promise<any>;

export function deepTemporalEntityFit(y: any, lookback: number, horizon: number, backend?: string | null): any;

export function deepTemporalEntityPredict(artifact: any, horizon: number): any;

export function empiricalSemivariogramWebgpu(coords: any, values: Float32Array, bin_count: number, max_distance: number | null | undefined, anisotropy_angle_degrees: number, anisotropy_scaling: number): Promise<any>;

export function fitPiecewiseLinearSeasonalArtifact(request: any): any;

export function geoSplitManifestHash(manifest_json: string): string;

export function predictPiecewiseLinearSeasonalArtifact(artifact: string, horizon: number): any;

export function predictPiecewiseLinearSeasonalArtifactWithOptions(artifact: string, horizon: number, options: any): any;

export function runForecast(request: any): any;

export function runGeoCausalExperiment(request: any): any;

export function runGeoFeatureExamples(request: any): any;

export function runGeoFeatureExamplesWebgpu(request: any): Promise<any>;

export function runGeostatisticsModel(request: any): any;

export function runGeostatisticsWebgpu(request: any): Promise<any>;

export function runGeotemporalDiagnostics(request: any): any;

/**
 * Runs a graph-diffusion forecaster whose sparse propagation remains on
 * browser WebGPU for every diffusion and horizon step.
 */
export function runGraphDiffusionWebgpu(request: any): Promise<any>;

export function runGraphForecast(request: any): any;

/**
 * Fits the generic market structure model in-browser and returns the full
 * analyst-facing payload: directional forecasts, explanations, and kernels.
 */
export function runMarketStructureExplorer(request: any): any;

/**
 * Fits and recursively predicts N-BEATS/N-HiTS with browser WebGPU training
 * and dense hidden-layer inference. Kept separate from the synchronous
 * dispatcher because browser GPU promises cannot be synchronously blocked.
 */
export function runNeuralForecastWebgpu(request: any): Promise<any>;

export function runNeuralModel(request: any): any;

export function runNode2VecModelWebgpu(request: any): Promise<any>;

export function runRegressionModel(request: any): any;

export function runSequence(request: any): any;

export function webgpuAdamwStep(parameters: Float32Array, first_moment: Float32Array, second_moment: Float32Array, gradient: Float32Array, step: bigint, learning_rate: number, weight_decay: number): Promise<any>;

export function webgpuAffineScores(features: any, means: Float64Array, weights: Float64Array, intercepts: Float64Array): Promise<any>;

/**
 * Probes the browser adapter and reports the same complete operation contract
 * used by native backends. This is async because browser adapter discovery is
 * promise-driven and must never be guessed from compile-time features alone.
 */
export function webgpuCapabilities(): Promise<any>;

export function webgpuCsrDiffusion(indptr: Uint32Array, indices: Uint32Array, weights: Float32Array, channels: number, values: Float32Array): Promise<Float32Array>;

export function webgpuCsrDiffusionBackward(indptr: Uint32Array, indices: Uint32Array, weights: Float32Array, channels: number, values: Float32Array, output_grad: Float32Array): Promise<any>;

export function webgpuCsrRowSoftmax(indptr: Uint32Array, logits: Float32Array): Promise<Float32Array>;

export function webgpuCsrRowSoftmaxBackward(indptr: Uint32Array, weights: Float32Array, output_grad: Float32Array): Promise<Float32Array>;

export function webgpuDenseLayer(features: any, weights: Float32Array, biases: Float32Array): Promise<any>;

/**
 * Executes a browser WebGPU compute pass and resolves once the mapped output
 * has been verified. Unlike the synchronous modeling exports, this function
 * can await browser adapter and buffer-map promises safely.
 */
export function webgpuDispatchReport(len: number): Promise<any>;

export function webgpuGraphSmooth(indptr: Uint32Array, indices: Uint32Array, weights: Float32Array, values: Float64Array, smoothing: number, iterations: number): Promise<any>;

export function webgpuLayerNorm(values: Float32Array, rows: number, width: number, gamma: Float32Array, beta: Float32Array): Promise<Float32Array>;

export function webgpuPairSigmoidScores(embeddings: any, pairs: any): Promise<any>;

export function webgpuPairwiseSquaredDistances(left: any, right: any): Promise<any>;

export function webgpuScalarGraph(initial_values: Float32Array, opcodes: Uint8Array, left: Uint32Array, right: Uint32Array): Promise<Float32Array>;

export function webgpuScalarGraphTrainStep(initial_values: Float32Array, opcodes: Uint8Array, left: Uint32Array, right: Uint32Array, parameter_ids: Uint32Array, loss: number, parameters: Float32Array, first_moment: Float32Array, second_moment: Float32Array, step: bigint, learning_rate: number, weight_decay: number): Promise<any>;

export function webgpuTrainTanhMlp(inputs: any, targets: Float32Array, hidden_size: number, epochs: number, learning_rate: number, parameters: Float32Array): Promise<Float32Array>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly availableDeepBackends: () => [number, number, number];
    readonly availableForecastModels: () => [number, number, number];
    readonly deepChoiceSetTransformerReport: (a: any, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly deepConditionalFlowFit: (a: any, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly deepConditionalFlowPredict: (a: any, b: any, c: number, d: number) => [number, number, number];
    readonly deepConditionalFlowPredictWebgpu: (a: any, b: any, c: number, d: number) => any;
    readonly deepConstrainedDecisionSelect: (a: any, b: number, c: number, d: any, e: number, f: number) => [number, number, number];
    readonly deepDiffusionScenarioGenerate: (a: any, b: any, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly deepDiffusionScenarioGenerateWebgpu: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly deepDirectionalPairPredict: (a: any, b: number, c: number) => [number, number, number];
    readonly deepEventOutcomeFit: (a: any, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly deepEventOutcomePredict: (a: any, b: any) => [number, number, number];
    readonly deepEventOutcomePredictWebgpu: (a: any, b: any) => any;
    readonly deepGraphNeuralOperatorPredict: (a: any, b: any, c: any, d: any, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly deepGraphNeuralOperatorPredictWebgpu: (a: any, b: any, c: any, d: any, e: number, f: number) => any;
    readonly deepNeuralOperatorSyntheticBenchmark: () => [number, number, number];
    readonly deepRegimeMoeReport: (a: any, b: number, c: number) => [number, number, number];
    readonly deepResponseCurveFit: (a: any, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly deepResponseCurvePredict: (a: any, b: any) => [number, number, number];
    readonly deepResponseCurvePredictWebgpu: (a: any, b: any) => any;
    readonly deepServiceResidualFit: (a: any, b: number, c: number) => [number, number, number];
    readonly deepServiceResidualPredict: (a: any, b: any) => [number, number, number];
    readonly deepServiceResidualPredictWebgpu: (a: any, b: any) => any;
    readonly deepTemporalEntityFit: (a: any, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly deepTemporalEntityPredict: (a: any, b: number) => [number, number, number];
    readonly empiricalSemivariogramWebgpu: (a: any, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
    readonly fitPiecewiseLinearSeasonalArtifact: (a: any) => [number, number, number];
    readonly geoSplitManifestHash: (a: number, b: number) => [number, number, number, number];
    readonly predictPiecewiseLinearSeasonalArtifact: (a: number, b: number, c: number) => [number, number, number];
    readonly predictPiecewiseLinearSeasonalArtifactWithOptions: (a: number, b: number, c: number, d: any) => [number, number, number];
    readonly runForecast: (a: any) => [number, number, number];
    readonly runGeoCausalExperiment: (a: any) => [number, number, number];
    readonly runGeoFeatureExamples: (a: any) => [number, number, number];
    readonly runGeoFeatureExamplesWebgpu: (a: any) => any;
    readonly runGeostatisticsModel: (a: any) => [number, number, number];
    readonly runGeostatisticsWebgpu: (a: any) => any;
    readonly runGeotemporalDiagnostics: (a: any) => [number, number, number];
    readonly runGraphDiffusionWebgpu: (a: any) => any;
    readonly runGraphForecast: (a: any) => [number, number, number];
    readonly runMarketStructureExplorer: (a: any) => [number, number, number];
    readonly runNeuralForecastWebgpu: (a: any) => any;
    readonly runNeuralModel: (a: any) => [number, number, number];
    readonly runNode2VecModelWebgpu: (a: any) => any;
    readonly runRegressionModel: (a: any) => [number, number, number];
    readonly runSequence: (a: any) => [number, number, number];
    readonly webgpuAdamwStep: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: bigint, j: number, k: number) => any;
    readonly webgpuAffineScores: (a: any, b: number, c: number, d: number, e: number, f: number, g: number) => any;
    readonly webgpuCapabilities: () => any;
    readonly webgpuCsrDiffusion: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => any;
    readonly webgpuCsrDiffusionBackward: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => any;
    readonly webgpuCsrRowSoftmax: (a: number, b: number, c: number, d: number) => any;
    readonly webgpuCsrRowSoftmaxBackward: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly webgpuDenseLayer: (a: any, b: number, c: number, d: number, e: number) => any;
    readonly webgpuDispatchReport: (a: number) => any;
    readonly webgpuGraphSmooth: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => any;
    readonly webgpuLayerNorm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
    readonly webgpuPairSigmoidScores: (a: any, b: any) => any;
    readonly webgpuPairwiseSquaredDistances: (a: any, b: any) => any;
    readonly webgpuScalarGraph: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
    readonly webgpuScalarGraphTrainStep: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: bigint, s: number, t: number) => any;
    readonly webgpuTrainTanhMlp: (a: any, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
    readonly deepInvertedTemporalFit: (a: any, b: number, c: number, d: number) => [number, number, number];
    readonly deepInvertedTemporalPredict: (a: any, b: number, c: number) => [number, number, number];
    readonly wasm_bindgen_e13304a9bf4bdd25___convert__closures_____invoke___wasm_bindgen_e13304a9bf4bdd25___JsValue__core_608f92abc48d28da___result__Result_____wasm_bindgen_e13304a9bf4bdd25___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_e13304a9bf4bdd25___convert__closures_____invoke___js_sys_9a5df1e41d83eac9___Function_fn_wasm_bindgen_e13304a9bf4bdd25___JsValue_____wasm_bindgen_e13304a9bf4bdd25___sys__Undefined___js_sys_9a5df1e41d83eac9___Function_fn_wasm_bindgen_e13304a9bf4bdd25___JsValue_____wasm_bindgen_e13304a9bf4bdd25___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_e13304a9bf4bdd25___convert__closures_____invoke___wasm_bindgen_e13304a9bf4bdd25___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
