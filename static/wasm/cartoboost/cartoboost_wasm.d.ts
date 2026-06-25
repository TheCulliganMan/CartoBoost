/* tslint:disable */
/* eslint-disable */

export function availableForecastModels(): any;

export function fitPiecewiseLinearSeasonalArtifact(request: any): any;

export function predictPiecewiseLinearSeasonalArtifact(artifact: string, horizon: number): any;

export function predictPiecewiseLinearSeasonalArtifactWithOptions(artifact: string, horizon: number, options: any): any;

export function runForecast(request: any): any;

export function runGeotemporalDiagnostics(request: any): any;

export function runNeuralModel(request: any): any;

export function runRegressionModel(request: any): any;

export function runSequence(request: any): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly availableForecastModels: () => [number, number, number];
    readonly fitPiecewiseLinearSeasonalArtifact: (a: any) => [number, number, number];
    readonly predictPiecewiseLinearSeasonalArtifact: (a: number, b: number, c: number) => [number, number, number];
    readonly predictPiecewiseLinearSeasonalArtifactWithOptions: (a: number, b: number, c: number, d: any) => [number, number, number];
    readonly runForecast: (a: any) => [number, number, number];
    readonly runGeotemporalDiagnostics: (a: any) => [number, number, number];
    readonly runNeuralModel: (a: any) => [number, number, number];
    readonly runRegressionModel: (a: any) => [number, number, number];
    readonly runSequence: (a: any) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
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
