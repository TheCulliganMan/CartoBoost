/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const availableForecastModels: () => [number, number, number];
export const fitPiecewiseLinearSeasonalArtifact: (a: any) => [number, number, number];
export const predictPiecewiseLinearSeasonalArtifact: (a: number, b: number, c: number) => [number, number, number];
export const predictPiecewiseLinearSeasonalArtifactWithOptions: (a: number, b: number, c: number, d: any) => [number, number, number];
export const runForecast: (a: any) => [number, number, number];
export const runGeotemporalDiagnostics: (a: any) => [number, number, number];
export const runNeuralModel: (a: any) => [number, number, number];
export const runRegressionModel: (a: any) => [number, number, number];
export const runSequence: (a: any) => [number, number, number];
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
export const __wbindgen_exn_store: (a: number) => void;
export const __externref_table_alloc: () => number;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_start: () => void;
