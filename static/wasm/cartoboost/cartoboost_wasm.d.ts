/* tslint:disable */
/* eslint-disable */

export function availableDeepBackends(): any;

export function availableForecastModels(): any;

export function deepChoiceSetTransformerReport(candidates: any, temperature: number, monotone_candidate_value?: string | null): any;

export function deepConditionalFlowFit(hidden: any, residuals: Float64Array, quantiles: Float64Array, sample_count: number): any;

export function deepConditionalFlowPredict(artifact: any, hidden: any, actual?: Float64Array | null): any;

export function deepConstrainedDecisionSelect(candidates: any, objective: string, constraints: any, fallback: string): any;

export function deepDiffusionScenarioGenerate(point_forecast: any, edges: any, scenario_count: number, diffusion_steps: number, shock_scale: number): any;

export function deepDirectionalPairPredict(rows: any): any;

export function deepEventOutcomeFit(features: any, labels: Float64Array, backend?: string | null): any;

export function deepEventOutcomePredict(artifact: any, features: any): any;

export function deepGraphNeuralOperatorPredict(field_values: any, coordinates: any, edges: any, exogenous_fields: any, smoothing: number, coordinate_scale: number): any;

export function deepNeuralOperatorSyntheticBenchmark(): any;

export function deepRegimeMoeReport(features: any, target: Float64Array): any;

export function deepResponseCurveFit(rows: any, response_type: string, monotone?: string | null, backend?: string | null): any;

export function deepResponseCurvePredict(artifact: any, rows: any): any;

export function deepServiceResidualFit(rows: any, backend?: string | null): any;

export function deepServiceResidualPredict(artifact: any, rows: any): any;

export function deepTemporalEntityFit(y: any, lookback: number, horizon: number): any;

export function deepTemporalEntityPredict(artifact: any, horizon: number): any;

export function fitPiecewiseLinearSeasonalArtifact(request: any): any;

export function geoSplitManifestHash(manifest_json: string): string;

export function predictPiecewiseLinearSeasonalArtifact(artifact: string, horizon: number): any;

export function predictPiecewiseLinearSeasonalArtifactWithOptions(artifact: string, horizon: number, options: any): any;

export function runForecast(request: any): any;

export function runGeoCausalExperiment(request: any): any;

export function runGeoFeatureExamples(request: any): any;

export function runGeostatisticsModel(request: any): any;

export function runGeotemporalDiagnostics(request: any): any;

export function runGraphForecast(request: any): any;

export function runNeuralModel(request: any): any;

export function runRegressionModel(request: any): any;

export function runSequence(request: any): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly runForecast: (a: any) => [number, number, number];
    readonly runGraphForecast: (a: any) => [number, number, number];
    readonly deepResponseCurveFit: (a: any, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly deepResponseCurvePredict: (a: any, b: any) => [number, number, number];
    readonly deepEventOutcomeFit: (a: any, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly deepEventOutcomePredict: (a: any, b: any) => [number, number, number];
    readonly deepDirectionalPairPredict: (a: any) => [number, number, number];
    readonly deepServiceResidualFit: (a: any, b: number, c: number) => [number, number, number];
    readonly availableDeepBackends: () => [number, number, number];
    readonly deepServiceResidualPredict: (a: any, b: any) => [number, number, number];
    readonly deepTemporalEntityFit: (a: any, b: number, c: number) => [number, number, number];
    readonly deepTemporalEntityPredict: (a: any, b: number) => [number, number, number];
    readonly deepConditionalFlowFit: (a: any, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly deepConditionalFlowPredict: (a: any, b: any, c: number, d: number) => [number, number, number];
    readonly deepDiffusionScenarioGenerate: (a: any, b: any, c: number, d: number, e: number) => [number, number, number];
    readonly deepGraphNeuralOperatorPredict: (a: any, b: any, c: any, d: any, e: number, f: number) => [number, number, number];
    readonly deepNeuralOperatorSyntheticBenchmark: () => [number, number, number];
    readonly deepChoiceSetTransformerReport: (a: any, b: number, c: number, d: number) => [number, number, number];
    readonly deepRegimeMoeReport: (a: any, b: number, c: number) => [number, number, number];
    readonly deepConstrainedDecisionSelect: (a: any, b: number, c: number, d: any, e: number, f: number) => [number, number, number];
    readonly fitPiecewiseLinearSeasonalArtifact: (a: any) => [number, number, number];
    readonly predictPiecewiseLinearSeasonalArtifact: (a: number, b: number, c: number) => [number, number, number];
    readonly predictPiecewiseLinearSeasonalArtifactWithOptions: (a: number, b: number, c: number, d: any) => [number, number, number];
    readonly runRegressionModel: (a: any) => [number, number, number];
    readonly runNeuralModel: (a: any) => [number, number, number];
    readonly runSequence: (a: any) => [number, number, number];
    readonly runGeotemporalDiagnostics: (a: any) => [number, number, number];
    readonly runGeoCausalExperiment: (a: any) => [number, number, number];
    readonly runGeostatisticsModel: (a: any) => [number, number, number];
    readonly runGeoFeatureExamples: (a: any) => [number, number, number];
    readonly availableForecastModels: () => [number, number, number];
    readonly geoSplitManifestHash: (a: number, b: number) => [number, number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
