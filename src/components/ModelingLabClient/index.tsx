import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import type {PointerEvent as ReactPointerEvent} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {contourDensity, geoGraticule, geoMercator, geoPath, interpolateTurbo, scaleSequential} from 'd3';
import 'maplibre-gl/dist/maplibre-gl.css';

import {assertForecastResponseRecords, coerceFiniteNumber, formatFixed, formatPercent} from './numberFormat';
import styles from './styles.module.css';

type ParsedTable = {
  columns: string[];
  rows: Record<string, string>[];
  fileName: string;
};

type ForecastRecord = {
  series_id: string;
  timestamp: string;
  horizon: number;
  model: string;
  prediction: number;
  base_mean?: number;
  spatial_correction?: number;
  kriging_variance?: number;
  selected_neighbors?: string[];
};

type ForecastChartPoint = {
  kind: 'actual' | 'forecast';
  label: string;
  timestamp: string;
  value: number;
  index: number;
  horizon?: number;
};

type ForecastComponentRecord = {
  series_id: string;
  timestamp: string;
  horizon: number;
  prediction: number;
  trend: number;
  adjusted_trend?: number;
  trend_adjustment?: number;
  trend_adjustment_multiplier?: number;
  residual_shock?: number;
  linear_predictor: number;
  component_scale?: string;
  components: Record<string, number | Record<string, number>>;
};

type ForecastHistoryComponentRecord = {
  series_id: string;
  timestamp: string;
  index: number;
  actual: number;
  fitted: number;
  residual: number;
  trend: number;
  adjusted_trend?: number;
  trend_adjustment?: number;
  trend_adjustment_multiplier?: number;
  trend_movement?: number | null;
  fitted_movement?: number | null;
  components: Record<string, number | Record<string, number>>;
};

type ForecastQuantileRecord = {
  series_id: string;
  timestamp: string;
  horizon: number;
  quantile: number;
  prediction: number;
  mean: number;
};

type WasmModule = {
  default: (
    input?: string | URL | Request | Response | BufferSource | WebAssembly.Module | {module_or_path: string},
  ) => Promise<unknown>;
  runForecast: (request: unknown) => ForecastResponse;
  runRegressionModel: (request: unknown) => RegressionResponse;
  runNeuralModel: (request: unknown) => RegressionResponse;
  runGeostatisticsModel?: (request: unknown) => GeostatisticsResponse;
  runGeoFeatureExamples?: (request: unknown) => GeoFeatureResponse;
  runGraphForecast?: (request: unknown) => unknown;
  deepResponseCurveFit?: (rows: unknown, responseType: string, monotone?: string, backend?: string) => unknown;
  deepResponseCurvePredict?: (artifact: unknown, rows: unknown) => unknown;
  deepEventOutcomeFit?: (features: unknown, labels: Float64Array, backend?: string) => unknown;
  deepEventOutcomePredict?: (artifact: unknown, features: unknown) => unknown;
  deepDirectionalPairPredict?: (rows: unknown) => unknown;
  deepServiceResidualFit?: (rows: unknown, backend?: string) => unknown;
  deepServiceResidualPredict?: (artifact: unknown, rows: unknown) => unknown;
  deepTemporalEntityFit?: (y: unknown, lookback: number, horizon: number) => unknown;
  deepTemporalEntityPredict?: (artifact: unknown, horizon: number) => unknown;
  deepConditionalFlowFit?: (
    hidden: unknown,
    residuals: Float64Array,
    quantiles: Float64Array,
    sampleCount: number,
  ) => unknown;
  deepConditionalFlowPredict?: (artifact: unknown, hidden: unknown, actual?: Float64Array) => unknown;
  deepDiffusionScenarioGenerate?: (
    pointForecast: unknown,
    edges: unknown,
    scenarioCount: number,
    diffusionSteps: number,
    shockScale: number,
  ) => unknown;
  deepGraphNeuralOperatorPredict?: (
    fieldValues: unknown,
    coordinates: unknown,
    edges: unknown,
    exogenousFields: unknown,
    smoothing: number,
    coordinateScale: number,
  ) => unknown;
  deepNeuralOperatorSyntheticBenchmark?: () => unknown;
  deepChoiceSetTransformerReport?: (
    candidates: unknown,
    temperature: number,
    monotoneCandidateValue?: string,
  ) => unknown;
  deepRegimeMoeReport?: (features: unknown, target: Float64Array) => unknown;
  deepConstrainedDecisionSelect?: (
    candidates: unknown,
    objective: string,
    constraints: unknown,
    fallback: string,
  ) => unknown;
  runSequence?: (request: unknown) => unknown;
  runGeotemporalDiagnostics?: (request: unknown) => unknown;
  availableForecastModels?: () => WasmModelMetadata[];
};

type ParquetWasmModule = {
  default: () => Promise<unknown>;
  readParquet: (data: Uint8Array) => {intoIPCStream: () => Uint8Array; free?: () => void};
};

type ForecastResponse = {
  metadata: {
    model: string;
    warning?: {
      requestedModel?: string;
      fallbackModel?: string;
      reason?: string;
      policy?: string;
    };
    input: {
      n_rows: number;
      is_panel: boolean;
      series_ids: string[];
      frequency: string;
    };
  };
  forecast: {
    records: ForecastRecord[];
  };
  components?: {
    records: ForecastComponentRecord[];
  };
  historyComponents?: {
    records: ForecastHistoryComponentRecord[];
  };
  samples?: {
    records: unknown[];
  };
  quantiles?: {
    records: ForecastQuantileRecord[];
  };
};

type GeostatisticsResponse = {
  predictions: {
    x: number;
    y: number;
    mean: number;
    variance: number;
    std: number;
    neighborIndices: number[];
  }[];
  metadata: Record<string, unknown>;
};

type GeoFeatureResponse = {
  planar: BearingFeatureRow[];
  latlng: BearingFeatureRow[];
  routes: RouteFeatureRow[];
  radial: AnchorFeatureRow[];
  rbf: AnchorFeatureRow[];
  localFrame: LocalFrameFeatureRow[];
  metadata: Record<string, unknown>;
};

type BearingFeatureRow = {
  label: string;
  east?: number | null;
  north?: number | null;
  zeroDistance: boolean;
};

type RouteFeatureRow = {
  label: string;
  midX?: number | null;
  midY?: number | null;
  distance?: number | null;
  bearingEast?: number | null;
  bearingNorth?: number | null;
  zeroDistance: boolean;
};

type AnchorFeatureRow = {
  label: string;
  values: number[];
};

type LocalFrameFeatureRow = {
  label: string;
  alongAxis?: number | null;
  crossAxis?: number | null;
  invalidAxis: boolean;
};

type ComparisonResult = {
  runId?: string;
  runLabel?: string;
  requestedModel: string;
  label: string;
  pipeline: string;
  response?: ForecastResponse;
  error?: string;
};

type BacktestResult = {
  runId?: string;
  runLabel?: string;
  requestedModel: string;
  label: string;
  pipeline: string;
  rmse?: number;
  mae?: number;
  wape?: number;
  comparedRows?: number;
  response?: ForecastResponse;
  error?: string;
};

type BenchmarkResult = ComparisonResult | BacktestResult;

type RunProgress = {
  label: string;
  current?: number;
  total?: number;
};

type RunLogEntry = {
  id: number;
  message: string;
};

const BACKTEST_FORECAST_TIMEOUT_MS = 120_000;
const MAX_PARALLEL_BACKTEST_WORKERS = 3;

type RegressionResponse = {
  metadata: {
    model: string;
    featureNames?: string[];
    sparseFeatureNames?: string[];
    denseFeatureNames?: string[];
    pipeline?: string;
    splitterMode?: string;
    loss?: string;
    monotonicConstraints?: number[];
    treeCount: number;
  };
  metrics: {
    rmse: number;
    mae: number;
    r2: number;
    trainRows: number;
    holdoutRows: number;
  };
  predictions: {
    rowIndex: number;
    actual: number;
    prediction: number;
    lowerPrediction?: number;
    upperPrediction?: number;
    residual: number;
  }[];
  featureImportance: {
    feature: string;
    splitCount: number;
  }[];
  modelVisualization?: ModelVisualization;
};

type ModelVisualization = {
  summary: {
    treeCount: number;
    nodeCount: number;
    branchCount: number;
    leafCount: number;
    maxDepth: number;
    meanLeafValue: number;
    meanGain: number;
  };
  splitKinds: {
    kind: string;
    count: number;
  }[];
  splitterRules: {
    kind: string;
    label: string;
    count: number;
    totalGain: number;
    meanGain: number;
  }[];
  featureSplitCounts: {
    feature: string;
    kind: string;
    count: number;
    totalGain: number;
  }[];
  depthHistogram: {
    depth: number;
    count: number;
  }[];
  treeBlueprints: TreeBlueprint[];
};

type TreeBlueprint = {
  treeIndex: number;
  nodeCount: number;
  branchCount: number;
  leafCount: number;
  maxDepth: number;
  totalGain: number;
  root: TreeNodeBlueprint;
};

type TreeNodeBlueprint = {
  id: number;
  depth: number;
  kind: string;
  label: string;
  value?: number;
  gain?: number;
  sampleWeightSum?: number;
  left?: TreeNodeBlueprint;
  right?: TreeNodeBlueprint;
};

type ModelOption = {
  value: string;
  label: string;
  group: string;
};

type GeoDemandPoint = {
  lon: number;
  lat: number;
  target: number;
  dropLon: number;
  dropLat: number;
  routeKey?: string;
};

type WasmModelMetadata = {
  name: string;
  label: string;
  pipeline: string;
};

const fallbackModelOptions: ModelOption[] = [
  {value: 'auto_forecast', label: 'CartoBoost AutoForecast', group: 'global'},
  {value: 'cartoboost_lag', label: 'CartoBoost Lag', group: 'global'},
  {value: 'cartoboost_direct', label: 'CartoBoost Direct', group: 'global'},
  {value: 'rectified_recursive', label: 'Rectified Recursive', group: 'global'},
  {value: 'lag_plus', label: 'Lag Plus', group: 'global'},
  {value: 'scaled_cartoboost_lag', label: 'Scaled CartoBoost Lag', group: 'transform'},
  {value: 'log1p_cartoboost_lag', label: 'Log1p CartoBoost Lag', group: 'transform'},
  {value: 'neural_panel', label: 'Neural Panel', group: 'neural'},
  {value: 'classical_expert_bank', label: 'Classical Expert Bank', group: 'selection'},
  {value: 'autostats_bank', label: 'AutoStats Bank', group: 'selection'},
  {value: 'intermittent_demand', label: 'Intermittent Demand', group: 'demand'},
  {value: 'croston', label: 'Croston', group: 'demand'},
  {value: 'sba', label: 'SBA', group: 'demand'},
  {value: 'tsb', label: 'TSB', group: 'demand'},
  {value: 'stl_cartoboost', label: 'STL + ARIMA', group: 'decomposition'},
  {value: 'mstl_cartoboost', label: 'MSTL + ARIMA', group: 'decomposition'},
  {value: 'seasonal_naive', label: 'Seasonal Naive', group: 'local'},
  {value: 'window_average', label: 'Window Average', group: 'local'},
  {value: 'seasonal_window_average', label: 'Seasonal Window Average', group: 'local'},
  {value: 'theta', label: 'Theta', group: 'local'},
  {value: 'auto_ets', label: 'Auto ETS', group: 'local'},
  {value: 'ets', label: 'ETS', group: 'local'},
  {value: 'seasonal_ets', label: 'Seasonal ETS', group: 'local'},
  {value: 'auto_arima', label: 'Auto ARIMA', group: 'local'},
  {value: 'arima', label: 'ARIMA', group: 'local'},
  {value: 'kalman', label: 'Kalman', group: 'local'},
  {value: 'local_level_kalman', label: 'Local Level Kalman', group: 'local'},
  {value: 'auto_kalman', label: 'Auto Kalman', group: 'local'},
  {value: 'auto_local_level_kalman', label: 'Auto Local Level Kalman', group: 'local'},
  {value: 'piecewise_linear_seasonal', label: 'Piecewise Linear Seasonal', group: 'local'},
  {value: 'kriging', label: 'Kriging', group: 'spatial'},
  {value: 'spatial_piecewise_kriging', label: 'Spatial Piecewise Kriging', group: 'spatial'},
  {value: 'optimized_theta', label: 'Optimized Theta', group: 'local'},
  {value: 'naive', label: 'Naive', group: 'local'},
];

export function RegressionModelExample({
  title,
  mode = 'axis',
  loss = 'l2',
}: {
  title: string;
  mode?: string;
  loss?: string;
}): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const [status, setStatus] = useState('Ready to run in this page.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<RegressionResponse | null>(null);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Running boosted tree model in this page.');
    try {
      const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
      const response = wasmModule.runRegressionModel(regressionExampleRequest(mode, loss));
      setResult(response);
      setStatus(`Model complete with ${response.metrics.holdoutRows.toLocaleString()} holdout rows.`);
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [loss, mode, wasmBinaryUrl, wasmJsUrl]);

  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap'}}>
        <div>
          <strong>{title}</strong>
          <p style={{margin: '0.25rem 0 0'}}>
            Runs <code>runRegressionModel</code> in Wasm with <code>{mode}</code> splitters and <code>{loss}</code> loss.
          </p>
        </div>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run model'}
        </button>
      </div>
      <p style={{margin: '0.75rem 0'}}>{status}</p>
      {result && (
        <>
          <RegressionMetricSummary result={result} />
          <CartoBoostModelVisualizer result={result} />
          <RegressionPredictionChart result={result} />
          <FeatureImportanceChart result={result} />
          <RegressionPredictionTable result={result} />
        </>
      )}
    </section>
  );
}

export function DeepModelWasmExample({model}: {model: string}): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const [status, setStatus] = useState('Ready to run in this page.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<unknown>(null);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Loading CartoBoost Wasm.');
    try {
      const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
      const nextResult = runDeepModelWasmExample(wasmModule, model);
      setResult(nextResult);
      setStatus('Wasm example complete.');
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [model, wasmBinaryUrl, wasmJsUrl]);

  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap'}}>
        <div>
          <strong>{model} Wasm example</strong>
          <p style={{margin: '0.25rem 0 0'}}>
            Runs the Rust-backed browser export with a small taxi-shaped sample.
          </p>
        </div>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run example'}
        </button>
      </div>
      <p style={{margin: '0.5rem 0 0'}}>
        {status}
      </p>
      {result && <DeepModelWasmResult result={result} />}
    </section>
  );
}

function DeepModelWasmResult({result}: {result: unknown}) {
  const rows = deepExampleResultRows(result);
  return (
    <div style={{overflowX: 'auto', marginTop: '0.75rem'}}>
      <table>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th>{label}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function deepExampleResultRows(result: unknown): [string, string][] {
  if (typeof result !== 'object' || result === null) {
    return [['Result', String(result)]];
  }
  const value = result as Record<string, unknown>;
  if (Array.isArray(value.predictions)) {
    const firstPrediction = value.predictions[0];
    if (Array.isArray(firstPrediction)) {
      return [
        ['Horizons', String(value.horizon ?? value.predictions.length)],
        ['Nodes', Array.isArray(value.nodeIds) ? value.nodeIds.join(', ') : '-'],
        ['First horizon', firstPrediction.map(formatMetric).join(', ')],
      ];
    }
    if (typeof firstPrediction === 'object' && firstPrediction !== null) {
      const row = firstPrediction as Record<string, unknown>;
      return Object.entries(row).slice(0, 5).map(([key, rowValue]) => [key, formatMetric(rowValue)]);
    }
  }
  if (Array.isArray(value.choices)) {
    const firstChoice = value.choices[0] as Record<string, unknown> | undefined;
    return [
      ['Selected candidates', String(value.choices.length)],
      ['First group', String(firstChoice?.group_id ?? firstChoice?.groupId ?? '-')],
      ['First candidate', String(firstChoice?.candidate_id ?? firstChoice?.candidateId ?? '-')],
    ];
  }
  return Object.entries(value).slice(0, 6).map(([key, rowValue]) => [key, formatMetric(rowValue)]);
}

function runDeepModelWasmExample(wasmModule: WasmModule, model: string): unknown {
  switch (model) {
    case 'DirectionalPairForecaster': {
      if (!wasmModule.deepDirectionalPairPredict) {
        throw new Error('CartoBoost directional pair Wasm export is not available from this bundle.');
      }
      return {predictions: wasmModule.deepDirectionalPairPredict(deepDirectionalPairRows())};
    }
    case 'ResponseCurveModel': {
      if (!wasmModule.deepResponseCurveFit || !wasmModule.deepResponseCurvePredict) {
        throw new Error('CartoBoost response curve Wasm exports are not available from this bundle.');
      }
      const rows = deepResponseCurveRows();
      const artifact = wasmModule.deepResponseCurveFit(rows.slice(0, 6), 'binary', 'decreasing', 'cpu');
      return {predictions: wasmModule.deepResponseCurvePredict(artifact, rows.slice(6))};
    }
    case 'EventOutcomeModel': {
      if (!wasmModule.deepEventOutcomeFit || !wasmModule.deepEventOutcomePredict) {
        throw new Error('CartoBoost event outcome Wasm exports are not available from this bundle.');
      }
      const features = deepEventFeatures();
      const artifact = wasmModule.deepEventOutcomeFit(features.slice(0, 6), new Float64Array([0, 0, 1, 1, 0, 1]), 'cpu');
      return {predictions: wasmModule.deepEventOutcomePredict(artifact, features.slice(6))};
    }
    case 'ServiceTimeResidualModel': {
      if (!wasmModule.deepServiceResidualFit || !wasmModule.deepServiceResidualPredict) {
        throw new Error('CartoBoost service residual Wasm exports are not available from this bundle.');
      }
      const rows = deepServiceResidualRows();
      const artifact = wasmModule.deepServiceResidualFit(rows.slice(0, 6), 'cpu');
      return {predictions: wasmModule.deepServiceResidualPredict(artifact, rows.slice(6))};
    }
    case 'SpatioTemporalGraphForecaster': {
      if (!wasmModule.runGraphForecast) {
        throw new Error('CartoBoost graph forecast Wasm export is not available from this bundle.');
      }
      return wasmModule.runGraphForecast(deepGraphForecastRequest());
    }
    case 'DelayAwareGraphTransformer':
    case 'PropagationDelayGraphForecaster':
    case 'DynamicAdjacencyTransformer': {
      if (!wasmModule.runGraphForecast) {
        throw new Error('CartoBoost graph forecast Wasm export is not available from this bundle.');
      }
      return wasmModule.runGraphForecast(deepGraphForecastRequest());
    }
    case 'TemporalEntityTransformer':
    case 'InvertedTemporalTransformer':
    case 'InvertedEntityTransformer': {
      if (!wasmModule.deepTemporalEntityFit || !wasmModule.deepTemporalEntityPredict) {
        throw new Error('CartoBoost temporal entity Wasm exports are not available from this bundle.');
      }
      const panel = deepTemporalPanel();
      const artifact = wasmModule.deepTemporalEntityFit(panel, 4, 2);
      return wasmModule.deepTemporalEntityPredict(artifact, 2);
    }
    case 'ConditionalFlowDistributionHead':
    case 'JointHorizonFlowHead':
    case 'ResidualFlowCalibrator': {
      if (!wasmModule.deepConditionalFlowFit || !wasmModule.deepConditionalFlowPredict) {
        throw new Error('CartoBoost conditional flow Wasm exports are not available from this bundle.');
      }
      const hidden = deepFlowHidden();
      const artifact = wasmModule.deepConditionalFlowFit(
        hidden.slice(0, 6),
        new Float64Array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2]),
        new Float64Array([0.05, 0.5, 0.95]),
        12,
      );
      return wasmModule.deepConditionalFlowPredict(artifact, hidden.slice(6), new Float64Array([0.15, -0.05]));
    }
    case 'GeoTemporalDiffusionScenarioModel':
    case 'FlowScenarioGenerator':
    case 'ConditionalResidualDiffusion': {
      if (!wasmModule.deepDiffusionScenarioGenerate) {
        throw new Error('CartoBoost diffusion scenario Wasm export is not available from this bundle.');
      }
      return wasmModule.deepDiffusionScenarioGenerate(
        [
          [42, 35, 18],
          [44, 36, 19],
          [51, 40, 24],
        ],
        deepSpatialEdges(),
        8,
        2,
        0.6,
      );
    }
    case 'GraphNeuralOperator':
    case 'FourierGeoOperator':
    case 'SpatioTemporalOperator': {
      if (!wasmModule.deepGraphNeuralOperatorPredict) {
        throw new Error('CartoBoost graph neural operator Wasm export is not available from this bundle.');
      }
      return wasmModule.deepGraphNeuralOperatorPredict(
        [
          [42, 35, 18],
          [44, 36, 19],
          [51, 40, 24],
        ],
        [
          [0, 0],
          [0.5, 0.4],
          [1, 0.1],
        ],
        deepSpatialEdges(),
        [
          [0.1, 0.2, 0.0],
          [0.1, 0.3, 0.1],
          [0.2, 0.2, 0.1],
        ],
        0.25,
        0.1,
      );
    }
    case 'ChoiceSetTransformer':
    case 'UtilityNet':
    case 'NestedChoiceHead':
    case 'CounterfactualCandidateScorer': {
      if (!wasmModule.deepChoiceSetTransformerReport) {
        throw new Error('CartoBoost choice-set Wasm export is not available from this bundle.');
      }
      return wasmModule.deepChoiceSetTransformerReport(deepDecisionCandidates(), 0.85, 'decreasing');
    }
    case 'RegimeMoEForecaster':
    case 'GeoTemporalMixtureOfExperts':
    case 'PairRegimeRouter':
    case 'EntityRegimeRouter': {
      if (!wasmModule.deepRegimeMoeReport) {
        throw new Error('CartoBoost regime MoE Wasm export is not available from this bundle.');
      }
      return wasmModule.deepRegimeMoeReport(deepRegimeFeatures(), new Float64Array([18, 21, 48, 51, 24, 55]));
    }
    case 'ConstrainedDecisionOptimizer': {
      if (!wasmModule.deepConstrainedDecisionSelect) {
        throw new Error('CartoBoost constrained decision Wasm export is not available from this bundle.');
      }
      const choices = wasmModule.deepConstrainedDecisionSelect(
        deepDecisionCandidates(),
        'utility',
        {fare: 35, eta_minutes: 28},
        'raise',
      );
      return {choices};
    }
    default:
      throw new Error(`Unknown deep model example: ${model}`);
  }
}

function deepTemporalPanel() {
  return [
    [42, 35, 18],
    [44, 36, 19],
    [51, 40, 24],
    [58, 46, 31],
    [55, 45, 34],
    [49, 43, 30],
  ];
}

function deepFlowHidden() {
  return [
    [1.2, 8, 0.35],
    [2.4, 9, 0.44],
    [4.8, 17, 0.78],
    [5.2, 18, 0.82],
    [1.6, 11, 0.38],
    [6.1, 19, 0.88],
    [3.2, 16, 0.65],
    [5.8, 20, 0.9],
  ];
}

function deepSpatialEdges() {
  return [
    {source: 0, target: 1, weight: 0.7},
    {source: 0, target: 2, weight: 0.3},
    {source: 1, target: 2, weight: 1.0},
  ];
}

function deepRegimeFeatures() {
  return [
    [1.8, 8, 0.35],
    [2.1, 9, 0.42],
    [12.4, 17, 0.83],
    [13.1, 18, 0.88],
    [3.2, 11, 0.51],
    [14.8, 19, 0.91],
  ];
}

function deepDirectionalPairRows() {
  return [
    {source_id: 'PULocationID:161', target_id: 'DOLocationID:236', timestamp: 0, features: [1.8, 8, 0.42], target: 18.4},
    {source_id: 'PULocationID:161', target_id: 'DOLocationID:237', timestamp: 1, features: [2.1, 9, 0.48], target: 20.2},
    {source_id: 'PULocationID:132', target_id: 'DOLocationID:161', timestamp: 2, features: [17.4, 17, 0.83], target: 58.7},
  ];
}

function deepResponseCurveRows() {
  return [18, 22, 26, 20, 24, 28, 21, 25].map((fare, index) => ({
    features: [index % 2, 8 + (index % 4), 1.6 + index * 0.2],
    candidate_value: fare,
    response: index < 6 ? Number(fare <= 24) : undefined,
    group_id: `route_offer_${Math.floor(index / 2)}`,
    candidate_id: `fare_${fare}`,
  }));
}

function deepEventFeatures() {
  return [
    [1.2, 8, 0.35],
    [2.4, 9, 0.44],
    [4.8, 17, 0.78],
    [5.2, 18, 0.82],
    [1.6, 11, 0.38],
    [6.1, 19, 0.88],
    [3.2, 16, 0.65],
    [5.8, 20, 0.9],
  ];
}

function deepServiceResidualRows() {
  return [
    {baseline_value: 16.5, actual_value: 17.4, features: [1.8, 8, 0.35]},
    {baseline_value: 18.2, actual_value: 19.1, features: [2.1, 9, 0.42]},
    {baseline_value: 46.0, actual_value: 49.8, features: [12.4, 17, 0.83]},
    {baseline_value: 51.0, actual_value: 53.2, features: [13.1, 18, 0.88]},
    {baseline_value: 22.4, actual_value: 23.0, features: [3.2, 11, 0.51]},
    {baseline_value: 55.5, actual_value: 59.4, features: [14.8, 19, 0.91]},
    {baseline_value: 24.1, features: [3.6, 16, 0.62]},
    {baseline_value: 58.0, features: [15.2, 20, 0.94]},
  ];
}

function deepGraphForecastRequest() {
  return {
    frame: {
      nodeIds: ['PULocationID:161', 'PULocationID:236', 'PULocationID:132'],
      timestamps: [0, 1, 2, 3, 4, 5],
      target: [
        [42, 35, 18],
        [44, 36, 19],
        [51, 40, 24],
        [58, 46, 31],
        [55, 45, 34],
        [49, 43, 30],
      ],
      adjacency: {indptr: [0, 2, 3, 3], indices: [1, 2, 2], data: [0.7, 0.3, 1.0]},
      horizon: 2,
      frequency: 'hourly',
    },
    options: {diffusionSteps: 1, hiddenSize: 4, epochs: 5, backend: 'cpu'},
  };
}

function deepDecisionCandidates() {
  return [
    {decision_id: 'ride_request_1', candidate_id: 'standard_route', candidate_value: 26, utility: 0.74, fare: 26, eta_minutes: 24},
    {decision_id: 'ride_request_1', candidate_id: 'express_route', candidate_value: 34, utility: 0.82, fare: 34, eta_minutes: 18},
    {decision_id: 'ride_request_2', candidate_id: 'local_pickup', candidate_value: 22, utility: 0.68, fare: 22, eta_minutes: 26},
    {decision_id: 'ride_request_2', candidate_id: 'airport_queue', candidate_value: 42, utility: 0.91, fare: 42, eta_minutes: 32},
  ];
}

export function ForecastModelExample({
  model,
  title,
  sample = 'lane',
  horizon = 8,
}: ForecastModelExampleProps): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const taxiLaneSampleUrl = useBaseUrl('/samples/yellow_taxi_2024-01-single-lane-5000.parquet');
  const taxiVariedRouteSampleUrl = useBaseUrl('/samples/yellow_taxi_2024-01-varied-routes-2500.parquet');
  const [status, setStatus] = useState('Ready to run in this page.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [table, setTable] = useState<ParsedTable>(() => embeddedForecastExampleTable(sample, 'hourly'));
  const frequency = 'hourly';
  const seasonLength = 24;
  const selectedModel = normalizedForecastModel(model);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Loading real taxi forecast sample.');
    try {
      const nextTable = await loadDocsForecastExampleTable(sample, taxiLaneSampleUrl, taxiVariedRouteSampleUrl);
      const trainingTable = forecastExampleTrainingTable(nextTable, horizon, 'series_id');
      setTable(nextTable);
      setStatus('Running forecast in this page.');
      const response = await runBrowserForecast({
        wasmJsUrl,
        wasmBinaryUrl,
        table: trainingTable,
        timestampCol: 'timestamp',
        targetCol: 'target',
        seriesCol: 'series_id',
        frequency,
        horizon,
        model: selectedModel,
        seasonLength,
      });
      setResult(response);
      setStatus(`Forecast complete with ${response.forecast.records.length.toLocaleString()} holdout rows.`);
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [frequency, horizon, sample, seasonLength, selectedModel, taxiLaneSampleUrl, taxiVariedRouteSampleUrl, wasmBinaryUrl, wasmJsUrl]);

  const records = result?.forecast.records.slice(0, 6) ?? [];
  const hasSpatialCorrection = records.some((record) => typeof record.spatial_correction === 'number');
  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap'}}>
        <div>
          <strong>{title}</strong>
          <p style={{margin: '0.25rem 0 0'}}>
            Runs <code>{selectedModel}</code> against a bundled {sample === 'spatial' ? 'multi-location demand panel' : 'route-demand'} sample.
          </p>
        </div>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run forecast'}
        </button>
      </div>
      <p style={{margin: '0.75rem 0'}}>{status}</p>
      {result &&
        (isSpatialPiecewiseKrigingModel(selectedModel) ? (
          <SpatialForecastExampleDiagnostics records={result.forecast.records} />
        ) : (
          <ForecastExampleChart records={result.forecast.records} table={table} />
        ))}
      {records.length > 0 && (
        <div style={{overflowX: 'auto'}}>
          <table>
            <thead>
              <tr>
                <th>series_id</th>
                <th>timestamp</th>
                <th>horizon</th>
                <th>prediction</th>
                {hasSpatialCorrection && <th>spatial_correction</th>}
              </tr>
            </thead>
            <tbody>
              {records.map((record) => (
                <tr key={`${record.series_id}-${record.timestamp}-${record.horizon}`}>
                  <td>{record.series_id}</td>
                  <td>{record.timestamp}</td>
                  <td>{record.horizon}</td>
                  <td>{formatMetric(record.prediction)}</td>
                  {hasSpatialCorrection && <td>{typeof record.spatial_correction === 'number' ? formatMetric(record.spatial_correction) : '-'}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function NeuralModelExample({pipeline, title}: NeuralModelExampleProps): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const [status, setStatus] = useState('Ready to run in this page.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<RegressionResponse | null>(null);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Running neural model in this page.');
    try {
      const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
      const response = wasmModule.runNeuralModel(neuralExampleRequest(pipeline));
      setResult(response);
      setStatus(`Model complete with ${response.metrics.holdoutRows.toLocaleString()} holdout rows.`);
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [pipeline, wasmBinaryUrl, wasmJsUrl]);

  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap'}}>
        <div>
          <strong>{title}</strong>
          <p style={{margin: '0.25rem 0 0'}}>
            Runs <code>{pipeline}</code> in the browser with the bundled CartoBoost Wasm model.
          </p>
        </div>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run model'}
        </button>
      </div>
      <p style={{margin: '0.75rem 0'}}>{status}</p>
      {result && (
        <>
          <RegressionMetricSummary result={result} />
          <RegressionPredictionChart result={result} />
          <FeatureImportanceChart result={result} />
          <RegressionPredictionTable result={result} />
        </>
      )}
    </section>
  );
}

export function GeostatisticsModelExample({title}: {title: string}): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const [status, setStatus] = useState('Ready to run in this page.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<GeostatisticsResponse | null>(null);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Fitting nearest-neighbor GP interpolation in this page.');
    try {
      const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
      if (!wasmModule.runGeostatisticsModel) {
        throw new Error('CartoBoost geostatistics Wasm export is not available from this bundle.');
      }
      const response = wasmModule.runGeostatisticsModel(geostatisticsExampleRequest());
      setResult(response);
      setStatus(`NNGP complete with ${response.predictions.length.toLocaleString()} uncertainty predictions.`);
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [wasmBinaryUrl, wasmJsUrl]);

  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap'}}>
        <div>
          <strong>{title}</strong>
          <p style={{margin: '0.25rem 0 0'}}>
            Runs nearest-neighbor GP interpolation with mean and standard deviation in the browser.
          </p>
        </div>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run NNGP'}
        </button>
      </div>
      <p style={{margin: '0.75rem 0'}}>{status}</p>
      {result && (
        <div style={{overflowX: 'auto'}}>
          <table>
            <thead>
              <tr>
                <th>pickup lon</th>
                <th>pickup lat</th>
                <th>mean</th>
                <th>std</th>
                <th>neighbors</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.slice(0, 6).map((row, index) => (
                <tr key={`${row.x}-${row.y}-${index}`}>
                  <td>{formatFixed(row.x, 4)}</td>
                  <td>{formatFixed(row.y, 4)}</td>
                  <td>{formatMetric(row.mean)}</td>
                  <td>{formatMetric(row.std)}</td>
                  <td>{row.neighborIndices.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function GeoFeatureExample({title}: {title: string}): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const [status, setStatus] = useState('Ready to run in this page.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<GeoFeatureResponse | null>(null);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Computing bearing unit-vector features in this page.');
    try {
      const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
      if (!wasmModule.runGeoFeatureExamples) {
        throw new Error('CartoBoost geo feature Wasm export is not available from this bundle.');
      }
      const response = wasmModule.runGeoFeatureExamples(geoFeatureExampleRequest());
      setResult(response);
      setStatus(`Computed ${response.planar.length + response.latlng.length} bearing feature rows.`);
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [wasmBinaryUrl, wasmJsUrl]);

  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap'}}>
        <div>
          <strong>{title}</strong>
          <p style={{margin: '0.25rem 0 0'}}>
            Runs Rust geo feature helpers in Wasm and returns continuous <code>east</code>/<code>north</code> bearing columns.
          </p>
        </div>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run geo features'}
        </button>
      </div>
      <p style={{margin: '0.75rem 0'}}>{status}</p>
      {result && (
        <>
          <BearingFeatureTable title="Projected coordinate routes" rows={result.planar} />
          <BearingFeatureTable title="Latitude/longitude taxi routes" rows={result.latlng} />
          <RouteFeatureTable rows={result.routes} />
          <AnchorFeatureTable title="Radial distance features" rows={result.radial} />
          <AnchorFeatureTable title="RBF anchor features" rows={result.rbf} />
          <LocalFrameFeatureTable rows={result.localFrame} />
        </>
      )}
    </section>
  );
}

function BearingFeatureTable({title, rows}: {title: string; rows: BearingFeatureRow[]}): React.ReactElement {
  return (
    <div style={{overflowX: 'auto', marginTop: '0.75rem'}}>
      <strong>{title}</strong>
      <table>
        <thead>
          <tr>
            <th>route</th>
            <th>east</th>
            <th>north</th>
            <th>zero distance</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{typeof row.east === 'number' ? formatFixed(row.east, 4) : '-'}</td>
              <td>{typeof row.north === 'number' ? formatFixed(row.north, 4) : '-'}</td>
              <td>{row.zeroDistance ? 'true' : 'false'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RouteFeatureTable({rows}: {rows: RouteFeatureRow[]}): React.ReactElement {
  return (
    <div style={{overflowX: 'auto', marginTop: '0.75rem'}}>
      <strong>Route midpoint, distance, and bearing</strong>
      <table>
        <thead>
          <tr>
            <th>route</th>
            <th>mid_x</th>
            <th>mid_y</th>
            <th>distance</th>
            <th>bearing_east</th>
            <th>bearing_north</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{typeof row.midX === 'number' ? formatFixed(row.midX, 4) : '-'}</td>
              <td>{typeof row.midY === 'number' ? formatFixed(row.midY, 4) : '-'}</td>
              <td>{typeof row.distance === 'number' ? formatFixed(row.distance, 4) : '-'}</td>
              <td>{typeof row.bearingEast === 'number' ? formatFixed(row.bearingEast, 4) : '-'}</td>
              <td>{typeof row.bearingNorth === 'number' ? formatFixed(row.bearingNorth, 4) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AnchorFeatureTable({title, rows}: {title: string; rows: AnchorFeatureRow[]}): React.ReactElement {
  const maxWidth = Math.max(0, ...rows.map((row) => row.values.length));
  return (
    <div style={{overflowX: 'auto', marginTop: '0.75rem'}}>
      <strong>{title}</strong>
      <table>
        <thead>
          <tr>
            <th>point</th>
            {Array.from({length: maxWidth}, (_, index) => (
              <th key={index}>anchor_{index}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              {Array.from({length: maxWidth}, (_, index) => (
                <td key={index}>{typeof row.values[index] === 'number' ? formatFixed(row.values[index], 4) : '-'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LocalFrameFeatureTable({rows}: {rows: LocalFrameFeatureRow[]}): React.ReactElement {
  return (
    <div style={{overflowX: 'auto', marginTop: '0.75rem'}}>
      <strong>Local frame features</strong>
      <table>
        <thead>
          <tr>
            <th>point</th>
            <th>along_axis</th>
            <th>cross_axis</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{typeof row.alongAxis === 'number' ? formatFixed(row.alongAxis, 4) : '-'}</td>
              <td>{typeof row.crossAxis === 'number' ? formatFixed(row.crossAxis, 4) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ForecastModelRosterExample(): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const taxiLaneSampleUrl = useBaseUrl('/samples/yellow_taxi_2024-01-single-lane-5000.parquet');
  const taxiVariedRouteSampleUrl = useBaseUrl('/samples/yellow_taxi_2024-01-varied-routes-2500.parquet');
  const [models, setModels] = useState<ModelOption[]>(fallbackModelOptions);
  const [model, setModel] = useState('auto_forecast');
  const [status, setStatus] = useState('Choose any forecast model and run the same route-demand example.');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [table, setTable] = useState<ParsedTable>(() => embeddedForecastExampleTable('lane', 'hourly'));
  const selectedModel = normalizedForecastModel(model);
  const sample = docsExampleSample(selectedModel);
  const frequency = 'hourly';
  const seasonLength = 24;

  useEffect(() => {
    let cancelled = false;
    getForecastModelOptions(wasmJsUrl, wasmBinaryUrl)
      .then((options) => {
        if (!cancelled && options.length > 0) {
          setModels(options);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : String(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [wasmBinaryUrl, wasmJsUrl]);

  const runExample = useCallback(async () => {
    setIsRunning(true);
    setStatus('Loading real taxi forecast sample.');
    try {
      const nextTable = await loadDocsForecastExampleTable(sample, taxiLaneSampleUrl, taxiVariedRouteSampleUrl);
      setTable(nextTable);
      setStatus(`Running ${selectedModel} in this page.`);
      const response = await runBrowserForecast({
        wasmJsUrl,
        wasmBinaryUrl,
        table: nextTable,
        timestampCol: 'timestamp',
        targetCol: 'target',
        seriesCol: 'series_id',
        frequency,
        horizon: 6,
        model: selectedModel,
        seasonLength,
      });
      setResult(response);
      setStatus(`${modelLabel(models, selectedModel)} complete with ${response.forecast.records.length.toLocaleString()} rows.`);
    } catch (error) {
      setResult(null);
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [frequency, models, sample, seasonLength, selectedModel, taxiLaneSampleUrl, taxiVariedRouteSampleUrl, wasmBinaryUrl, wasmJsUrl]);

  const records = result?.forecast.records.slice(0, 5) ?? [];
  return (
    <section
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 8,
        padding: '1rem',
        margin: '1rem 0',
      }}
    >
      <div style={{display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'end'}}>
        <label style={{display: 'grid', gap: '0.25rem', minWidth: 260}}>
          <span>Forecast model</span>
          <select value={model} onChange={(event) => setModel(event.target.value)} disabled={isRunning}>
            {models.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <button className="button button--primary" type="button" disabled={isRunning} onClick={() => void runExample()}>
          {isRunning ? 'Running' : 'Run selected model'}
        </button>
      </div>
      <p style={{margin: '0.75rem 0'}}>{status}</p>
      <p style={{margin: '0 0 0.75rem'}}>
        This control reads the available model list and runs the selected model against a bundled taxi-style {sample === 'spatial' ? 'coordinate panel' : 'lane'} sample.
      </p>
      {result && <ForecastExampleChart records={result.forecast.records} table={table} />}
      {records.length > 0 && (
        <div style={{overflowX: 'auto'}}>
          <table>
            <thead>
              <tr>
                <th>series_id</th>
                <th>timestamp</th>
                <th>horizon</th>
                <th>prediction</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record) => (
                <tr key={`${record.series_id}-${record.timestamp}-${record.horizon}`}>
                  <td>{record.series_id}</td>
                  <td>{record.timestamp}</td>
                  <td>{record.horizon}</td>
                  <td>{formatMetric(record.prediction)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function neuralExampleRequest(pipeline: NeuralModelExampleProps['pipeline']) {
  const nodeTypes = [0, 1, 1, 0, 0, 1, 1, 0];
  const rows = Array.from({length: 64}, (_, index) => {
    const source = index % nodeTypes.length;
    const sourceType = nodeTypes[source];
    const compatibleTargets = nodeTypes
      .map((nodeType, node) => ({nodeType, node}))
      .filter((candidate) => candidate.nodeType !== sourceType);
    const targetNode = compatibleTargets[(Math.floor(index / nodeTypes.length) + source) % compatibleTargets.length].node;
    const edgeType = sourceType === 0 ? 0 : 1;
    const id = 100 + (source * 3 + targetNode) % 10;
    const distance = 1.2 + ((source + targetNode) % 6) * 0.45 + (index % 4) * 0.08;
    const hour = 6 + ((index * 3 + source) % 18);
    const routeEffect = sourceType === 0 ? 0.35 : -0.15;
    const target = 1.8 + distance * 0.42 + hour * 0.055 + routeEffect + (targetNode % 3) * 0.18 + (id % 5) * 0.07;
    return {
      id,
      source,
      targetNode,
      edgeType,
      dense: [Number(distance.toFixed(3)), hour],
      target: Number(target.toFixed(3)),
    };
  });

  return {
    pipeline,
    denseFeatureNames: ['distance', 'hour'],
    nodeFeatures: [
      [1.0, 0.0],
      [0.0, 1.0],
      [0.6, 0.3],
      [0.2, 0.7],
      [0.9, 0.2],
      [0.1, 0.8],
      [0.5, 0.6],
      [0.4, 0.1],
    ],
    nodeTypes,
    edgeTypeTriples: [
      [0, 0, 1],
      [1, 1, 0],
    ],
    rows,
    options: {
      holdoutFraction: 0.25,
      embeddingDim: 6,
      randomState: 42,
      nEstimators: 48,
      learningRate: 0.07,
      maxDepth: 3,
      minSamplesLeaf: 3,
      node2vecWalkLength: 10,
      node2vecWalksPerNode: 6,
      node2vecWindowSize: 3,
      node2vecEpochs: 4,
      node2vecSeed: 42,
      graphSageEpochs: 6,
      graphSageNegativeSamples: 3,
      graphSageSeed: 42,
      includeModelVisualization: true,
    },
  };
}

function regressionExampleRequest(mode: string, loss: string) {
  const rows = Array.from({length: 72}, (_, index) => {
    const pickupLon = -74.02 + (index % 12) * 0.012;
    const pickupLat = 40.68 + (Math.floor(index / 12) % 6) * 0.018;
    const tripDistance = 0.9 + (index % 8) * 0.42;
    const hour = (index * 3) % 24;
    const airportZone = index % 9 === 0 ? 1 : 0;
    const target =
      4.5 +
      tripDistance * 2.8 +
      Math.sin((hour / 24) * Math.PI * 2) * 1.4 +
      airportZone * 7.5 +
      (pickupLon + 74.02) * 18 +
      (pickupLat - 40.68) * 9;
    return {
      features: [
        Number(pickupLon.toFixed(5)),
        Number(pickupLat.toFixed(5)),
        Number(tripDistance.toFixed(3)),
        hour,
        airportZone,
      ],
      sparseSets: [[100 + (index % 6), 200 + (index % 4)]],
      target: Number(target.toFixed(3)),
    };
  });

  return {
    rows,
    featureNames: ['pickup_lon', 'pickup_lat', 'trip_distance', 'pickup_hour', 'airport_zone'],
    sparseFeatureNames: ['taxi_zone_memberships'],
    options: {
      holdoutFraction: 0.25,
      splitterMode: mode,
      loss,
      quantileAlpha: 0.5,
      huberDelta: 4,
      logOffset: 1,
      intervalLowerAlpha: 0.1,
      intervalUpperAlpha: 0.9,
      featureKinds: {
        pickup_lon: 'spatial',
        pickup_lat: 'spatial',
        trip_distance: 'numeric',
        pickup_hour: 'periodic',
        airport_zone: 'numeric',
      },
      periodicPeriods: {
        pickup_hour: 24,
      },
      nEstimators: 36,
      learningRate: 0.07,
      maxDepth: 3,
      minSamplesLeaf: 3,
      includeModelVisualization: true,
    },
  };
}

function geostatisticsExampleRequest() {
  const observations = [
    {x: -73.9851, y: 40.7589, value: 12.4},
    {x: -73.9772, y: 40.7527, value: 10.8},
    {x: -73.968, y: 40.759, value: 9.7},
    {x: -73.9969, y: 40.742, value: 13.6},
    {x: -74.006, y: 40.7128, value: 16.2},
    {x: -73.9897, y: 40.7336, value: 14.1},
    {x: -73.958, y: 40.8006, value: 8.9},
    {x: -73.9496, y: 40.7831, value: 9.4},
    {x: -73.9213, y: 40.7433, value: 11.7},
    {x: -73.873, y: 40.7769, value: 21.5},
    {x: -73.7903, y: 40.6437, value: 31.2},
    {x: -73.8628, y: 40.7681, value: 23.0},
  ];
  return {
    observations,
    targets: [
      {x: -73.981, y: 40.756},
      {x: -73.99, y: 40.725},
      {x: -73.94, y: 40.776},
      {x: -73.88, y: 40.77},
      {x: -73.81, y: 40.65},
      {x: -74.02, y: 40.7},
    ],
    options: {
      kernel: 'matern_3_2',
      range: 0.045,
      sill: 36.0,
      nugget: 0.15,
      nNeighbors: 6,
    },
  };
}

function geoFeatureExampleRequest() {
  return {
    planarRoutes: [
      {label: 'north', origin: [0.0, 0.0], destination: [0.0, 4.0]},
      {label: 'east', origin: [0.0, 0.0], destination: [4.0, 0.0]},
      {label: 'northwest', origin: [0.0, 0.0], destination: [-3.0, 3.0]},
      {label: 'same point', origin: [2.0, 2.0], destination: [2.0, 2.0]},
    ],
    latlngRoutes: [
      {
        label: 'Times Square to Upper East Side',
        origin: [40.758, -73.9855],
        destination: [40.7804, -73.957],
      },
      {
        label: 'JFK to Midtown',
        origin: [40.647, -73.7865],
        destination: [40.7535, -73.9888],
      },
      {
        label: 'LaGuardia to Upper West Side',
        origin: [40.7769, -73.873],
        destination: [40.7917, -73.973],
      },
    ],
    radialPoints: [
      {label: 'pickup midtown', point: [0.5, 2.0]},
      {label: 'pickup west', point: [-2.0, 1.5]},
      {label: 'pickup south', point: [0.0, -1.0]},
    ],
    anchors: [
      {label: 'hub', point: [0.0, 0.0]},
      {label: 'airport', point: [4.0, -2.0]},
      {label: 'stadium', point: [-3.0, 3.0]},
    ],
    lengthScale: 3.0,
    localFrame: {
      origin: [0.0, 0.0],
      axis: [1.0, 1.0],
      points: [
        {label: 'on corridor', point: [2.0, 2.0]},
        {label: 'left of corridor', point: [0.0, 2.0]},
        {label: 'right of corridor', point: [2.0, 0.0]},
      ],
    },
  };
}

function ForecastExampleChart({records, table}: {records: ForecastRecord[]; table: ParsedTable}): React.ReactElement | null {
  const [hovered, setHovered] = useState<ForecastChartPoint | null>(null);
  const points = useMemo(() => forecastChartPoints(records, table), [records, table]);
  if (points.length < 2) {
    return null;
  }
  const width = 720;
  const height = 300;
  const padding = {top: 24, right: 18, bottom: 58, left: 58};
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const values = points.map((point) => point.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valuePadding = Math.max((maxValue - minValue) * 0.12, 1);
  const yMin = minValue - valuePadding;
  const yMax = maxValue + valuePadding;
  const xFor = (index: number) => padding.left + (points.length === 1 ? innerWidth / 2 : (index / (points.length - 1)) * innerWidth);
  const yFor = (value: number) => padding.top + (1 - (value - yMin) / (yMax - yMin)) * innerHeight;
  const actualPoints = points.filter((point) => point.kind === 'actual');
  const forecastPoints = points.filter((point) => point.kind === 'forecast');
  const pointIndex = new Map(points.map((point, index) => [point, index]));
  const actualPath = chartPath(actualPoints, pointIndex, xFor, yFor);
  const forecastPath = chartPath(forecastPoints, pointIndex, xFor, yFor);
  const boundaryIndex = actualPoints.length - 0.5;
  const boundaryX = actualPoints.length > 0 && forecastPoints.length > 0 ? xFor(boundaryIndex) : null;
  const joinPath =
    actualPoints.length > 0 && forecastPoints.length > 0
      ? `M ${xFor(pointIndex.get(actualPoints[actualPoints.length - 1]) ?? 0)} ${yFor(actualPoints[actualPoints.length - 1].value)} L ${xFor(pointIndex.get(forecastPoints[0]) ?? 0)} ${yFor(forecastPoints[0].value)}`
      : '';
  const tooltip = hovered ?? points[points.length - 1];

  return (
    <figure style={{margin: '1rem 0'}}>
      <figcaption style={{fontWeight: 700, marginBottom: '0.35rem'}}>Recent actuals and browser forecast</figcaption>
      <div style={{overflowX: 'auto', border: '1px solid var(--ifm-color-emphasis-200)', borderRadius: 8, padding: '0.5rem'}}>
        <svg
          role="img"
          aria-label="Recent actual values and forecast path"
          viewBox={`0 0 ${width} ${height}`}
          style={{width: '100%', minWidth: 520, maxWidth: width, display: 'block'}}
        >
          <rect x={padding.left} y={padding.top} width={innerWidth} height={innerHeight} fill="var(--ifm-color-emphasis-100)" opacity="0.3" />
          {boundaryX !== null && (
            <>
              <rect x={boundaryX} y={padding.top} width={width - padding.right - boundaryX} height={innerHeight} fill="#fed7aa" opacity="0.22" />
              <line x1={boundaryX} x2={boundaryX} y1={padding.top} y2={height - padding.bottom} stroke="#c2410c" strokeDasharray="5 5" />
            </>
          )}
          <line x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} stroke="var(--ifm-color-emphasis-400)" />
          <line x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} stroke="var(--ifm-color-emphasis-400)" />
          {[0, 0.5, 1].map((tick) => {
            const value = yMin + (yMax - yMin) * tick;
            const y = yFor(value);
            return (
              <g key={tick}>
                <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} stroke="var(--ifm-color-emphasis-200)" />
                <text x={padding.left - 8} y={y + 4} textAnchor="end" fontSize="12" fill="var(--ifm-font-color-base)">
                  {formatMetric(value)}
                </text>
              </g>
            );
          })}
          {actualPath && <path d={actualPath} fill="none" stroke="var(--ifm-color-primary)" strokeWidth="3" />}
          {joinPath && <path d={joinPath} fill="none" stroke="var(--ifm-color-primary)" strokeWidth="2" strokeDasharray="4 4" />}
          {forecastPath && <path d={forecastPath} fill="none" stroke="#c2410c" strokeWidth="3" />}
          {points.map((point, index) => {
            const isEndpoint = index === 0 || index === points.length - 1 || point.kind === 'forecast';
            return (
              <g key={`${point.kind}-${point.timestamp}-${index}`}>
                <circle
                  cx={xFor(index)}
                  cy={yFor(point.value)}
                  r={hovered === point ? 6 : isEndpoint ? 4 : 2.5}
                  fill={point.kind === 'actual' ? 'var(--ifm-color-primary)' : '#c2410c'}
                  stroke="white"
                  strokeWidth="1.5"
                />
                <circle
                  cx={xFor(index)}
                  cy={yFor(point.value)}
                  r="10"
                  fill="transparent"
                  onPointerEnter={() => setHovered(point)}
                  onPointerLeave={() => setHovered(null)}
                />
              </g>
            );
          })}
          {boundaryX !== null && (
            <text x={boundaryX + 6} y={padding.top + 14} fontSize="12" fill="#9a3412">
              forecast starts
            </text>
          )}
          <text x={padding.left} y={height - 20} fontSize="12" fill="var(--ifm-font-color-base)">
            Actual
          </text>
          <circle cx={padding.left + 42} cy={height - 24} r="4" fill="var(--ifm-color-primary)" />
          <text x={padding.left + 64} y={height - 20} fontSize="12" fill="var(--ifm-font-color-base)">
            Forecast
          </text>
          <circle cx={padding.left + 120} cy={height - 24} r="4" fill="#c2410c" />
        </svg>
      </div>
      <p style={{margin: '0.35rem 0 0'}}>
        {tooltip.kind === 'forecast' ? `Horizon ${tooltip.horizon}: ` : 'Actual: '}
        <strong>{formatMetric(tooltip.value)}</strong> at <code>{tooltip.timestamp}</code>
      </p>
    </figure>
  );
}

function SpatialForecastExampleDiagnostics({records}: {records: ForecastRecord[]}): React.ReactElement | null {
  if (records.length === 0) {
    return null;
  }
  const predictions = records.map((record) => record.prediction).filter(Number.isFinite);
  const corrections = records
    .map((record) => record.spatial_correction)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const variances = records
    .map((record) => record.kriging_variance)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const neighborCounts = records
    .map((record) => record.selected_neighbors?.length)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const seriesCount = new Set(records.map((record) => record.series_id)).size;
  const rows = [
    ['Series', seriesCount.toLocaleString()],
    ['Prediction range', metricRange(predictions)],
    ['Mean spatial correction', corrections.length > 0 ? formatMetric(mean(corrections)) : '-'],
    ['Mean kriging variance', variances.length > 0 ? formatMetric(mean(variances)) : '-'],
    ['Mean neighbors', neighborCounts.length > 0 ? formatMetric(mean(neighborCounts)) : '-'],
  ];
  return (
    <section style={{margin: '1rem 0'}}>
      <h3 style={{fontSize: '1rem', marginBottom: '0.5rem'}}>Spatial forecast diagnostics</h3>
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem'}}>
        {rows.map(([label, value]) => (
          <div key={label} style={{border: '1px solid var(--ifm-color-emphasis-200)', borderRadius: 8, padding: '0.75rem'}}>
            <strong style={{display: 'block', fontSize: '0.85rem'}}>{label}</strong>
            <span>{value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function metricRange(values: number[]) {
  if (values.length === 0) {
    return '-';
  }
  return `${formatMetric(Math.min(...values))} to ${formatMetric(Math.max(...values))}`;
}

function mean(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function forecastChartPoints(records: ForecastRecord[], table: ParsedTable): ForecastChartPoint[] {
  const selectedSeries = selectForecastChartSeriesId(records, table);
  if (!selectedSeries) {
    return [];
  }
  const forecasts = records
    .filter((record) => record.series_id === selectedSeries)
    .sort((left, right) => left.horizon - right.horizon || left.timestamp.localeCompare(right.timestamp))
    .map((record, index) => ({
      kind: 'forecast' as const,
      label: 'Forecast',
      timestamp: record.timestamp,
      value: record.prediction,
      index,
      horizon: record.horizon,
    }))
    .filter((point) => Number.isFinite(point.value));
  const firstForecastTimestamp = forecasts[0]?.timestamp;
  const actuals = table.rows
    .filter((row) => row.series_id === selectedSeries)
    .filter((row) => !firstForecastTimestamp || row.timestamp < firstForecastTimestamp)
    .map((row, index) => ({
      kind: 'actual' as const,
      label: 'Actual',
      timestamp: row.timestamp,
      value: Number(row.target),
      index,
    }))
    .filter((point) => Number.isFinite(point.value))
    .slice(-36);
  return [...actuals, ...forecasts];
}

function forecastExampleTrainingTable(table: ParsedTable, horizon: number, seriesCol: string): ParsedTable {
  if (!seriesCol || !table.columns.includes(seriesCol) || horizon <= 0) {
    return table;
  }
  const remainingBySeries = new Map<string, number>();
  for (const row of table.rows) {
    const seriesId = row[seriesCol] ?? '';
    remainingBySeries.set(seriesId, (remainingBySeries.get(seriesId) ?? 0) + 1);
  }
  const rows = table.rows.filter((row) => {
    const seriesId = row[seriesCol] ?? '';
    const remaining = remainingBySeries.get(seriesId) ?? 0;
    remainingBySeries.set(seriesId, remaining - 1);
    return remaining > horizon;
  });
  return rows.length > 0 ? {...table, rows} : table;
}

function selectForecastChartSeriesId(records: ForecastRecord[], table: ParsedTable): string | null {
  const seriesIds = Array.from(new Set(records.map((record) => record.series_id).filter(Boolean)));
  let best: {seriesId: string; score: number} | null = null;
  for (const seriesId of seriesIds) {
    const actuals = table.rows
      .filter((row) => row.series_id === seriesId)
      .map((row) => Number(row.target))
      .filter((value) => Number.isFinite(value));
    const forecasts = records
      .filter((record) => record.series_id === seriesId)
      .sort((left, right) => left.horizon - right.horizon || left.timestamp.localeCompare(right.timestamp))
      .map((record) => record.prediction)
      .filter((value) => Number.isFinite(value));
    if (actuals.length === 0 || forecasts.length === 0) {
      continue;
    }
    const recentActuals = actuals.slice(-36);
    const range = Math.max(...recentActuals) - Math.min(...recentActuals);
    const boundaryGap = Math.abs(actuals[actuals.length - 1] - forecasts[0]);
    const score = boundaryGap / Math.max(range, 1);
    if (!best || score < best.score) {
      best = {seriesId, score};
    }
  }
  return best?.seriesId ?? seriesIds[0] ?? null;
}

function chartPath(
  points: ForecastChartPoint[],
  pointIndex: Map<ForecastChartPoint, number>,
  xFor: (index: number) => number,
  yFor: (value: number) => number,
) {
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(pointIndex.get(point) ?? 0)} ${yFor(point.value)}`)
    .join(' ');
}

function docsExampleSample(model: string): 'lane' | 'spatial' {
  return ['auto_forecast', 'cartoboost_lag', 'kriging', 'spatial_piecewise_kriging'].includes(model) ? 'spatial' : 'lane';
}

function modelLabel(models: ModelOption[], value: string) {
  return models.find((option) => option.value === value)?.label ?? value;
}

const forecastPipelineLabels: Record<string, string> = {
  global: 'CartoBoost global',
  transform: 'CartoBoost transformed',
  selection: 'Selector banks',
  demand: 'Sparse demand',
  decomposition: 'Decomposition',
  spatial: 'Geographic',
  local: 'Local statistical',
};

const neuralPipelineLabels: Record<string, string> = {
  embedding: 'Embedding regressor',
  node2vec: 'Node2Vec graph',
  graphsage: 'GraphSAGE graph',
  hetero_graphsage: 'HeteroGraphSAGE typed graph',
  hinsage: 'HinSAGE source-target graph',
};

const graphNeuralPipelines = new Set(['node2vec', 'graphsage', 'hetero_graphsage', 'hinsage']);
type ActiveModelingSurface = 'forecast' | 'model' | 'neural' | 'deep';

type PendingLabRun = {
  action: 'forecast' | 'compare' | 'backtest' | 'benchmark';
  model: string;
};

type ForecastModelExampleProps = {
  model: string;
  title: string;
  sample?: 'lane' | 'spatial';
  horizon?: number;
};

type NeuralModelExampleProps = {
  pipeline: 'embedding' | 'node2vec' | 'graphsage' | 'hetero_graphsage' | 'hinsage';
  title: string;
};
const TAXI_LANE_SAMPLE_ROWS = 5000;
const TAXI_VARIED_ROUTE_SAMPLE_ROWS = 2500;
const VISUALIZED_MODEL_MAX_ROWS = 800;
const VISUALIZED_NEURAL_MAX_ROWS = 240;
const TAXI_ZONE_CENTROIDS: Record<string, {lat: number; lon: number}> = {
  4: {lat: 40.723752, lon: -73.976968},
  7: {lat: 40.761493, lon: -73.919694},
  13: {lat: 40.709139, lon: -74.016103},
  24: {lat: 40.798641, lon: -73.967191},
  33: {lat: 40.696011, lon: -73.995475},
  41: {lat: 40.804333, lon: -73.951292},
  43: {lat: 40.782478, lon: -73.965553},
  48: {lat: 40.762252, lon: -73.990376},
  50: {lat: 40.766948, lon: -73.995548},
  68: {lat: 40.748428, lon: -73.999917},
  74: {lat: 40.801169, lon: -73.937345},
  75: {lat: 40.790011, lon: -73.94575},
  79: {lat: 40.727705, lon: -73.986388},
  87: {lat: 40.706808, lon: -74.007495},
  88: {lat: 40.703357, lon: -74.011515},
  90: {lat: 40.742279, lon: -73.996971},
  100: {lat: 40.753513, lon: -73.988787},
  107: {lat: 40.736824, lon: -73.984053},
  113: {lat: 40.733217, lon: -73.994583},
  114: {lat: 40.729277, lon: -74.002972},
  125: {lat: 40.724159, lon: -74.004778},
  132: {lat: 40.646985, lon: -73.786533},
  137: {lat: 40.740439, lon: -73.976494},
  138: {lat: 40.77375, lon: -73.872923},
  140: {lat: 40.766558, lon: -73.95353},
  141: {lat: 40.766948, lon: -73.959635},
  142: {lat: 40.775932, lon: -73.982196},
  143: {lat: 40.775965, lon: -73.987646},
  144: {lat: 40.720889, lon: -73.996919},
  148: {lat: 40.718939, lon: -73.990896},
  151: {lat: 40.799717, lon: -73.970552},
  158: {lat: 40.735035, lon: -74.008984},
  161: {lat: 40.758028, lon: -73.977698},
  162: {lat: 40.756687, lon: -73.972356},
  163: {lat: 40.764421, lon: -73.977569},
  164: {lat: 40.748574, lon: -73.985156},
  166: {lat: 40.812887, lon: -73.962663},
  170: {lat: 40.747745, lon: -73.978492},
  186: {lat: 40.748497, lon: -73.992437},
  209: {lat: 40.707062, lon: -74.003757},
  211: {lat: 40.722327, lon: -74.001905},
  224: {lat: 40.73182, lon: -73.976848},
  229: {lat: 40.756728, lon: -73.965146},
  230: {lat: 40.759818, lon: -73.984196},
  231: {lat: 40.717773, lon: -74.008584},
  232: {lat: 40.715761, lon: -73.986782},
  233: {lat: 40.749948, lon: -73.970771},
  234: {lat: 40.740337, lon: -73.990457},
  236: {lat: 40.780436, lon: -73.957012},
  237: {lat: 40.768615, lon: -73.965635},
  238: {lat: 40.791705, lon: -73.973049},
  239: {lat: 40.783962, lon: -73.978632},
  246: {lat: 40.753309, lon: -74.004015},
  249: {lat: 40.734576, lon: -74.005281},
  261: {lat: 40.709729, lon: -74.013379},
  262: {lat: 40.77699, lon: -73.94615},
  263: {lat: 40.778766, lon: -73.95101},
  264: {lat: 40.758896, lon: -73.98513},
  265: {lat: 40.710051, lon: -73.865438},
};

export default function ModelingLabClient(): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const taxiLaneSampleUrl = useBaseUrl('/samples/yellow_taxi_2024-01-single-lane-5000.parquet');
  const taxiVariedRouteSampleUrl = useBaseUrl('/samples/yellow_taxi_2024-01-varied-routes-2500.parquet');
  const [table, setTable] = useState<ParsedTable | null>(null);
  const [timestampCol, setTimestampCol] = useState('timestamp');
  const [targetCol, setTargetCol] = useState('target');
  const [seriesCol, setSeriesCol] = useState('series_id');
  const [frequency, setFrequency] = useState('daily');
  const [model, setModel] = useState('auto_forecast');
  const [horizon, setHorizon] = useState(14);
  const [seasonLength, setSeasonLength] = useState(7);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([]);
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [hiddenBenchmarkSeries, setHiddenBenchmarkSeries] = useState<string[]>([]);
  const [regressionResult, setRegressionResult] = useState<RegressionResponse | null>(null);
  const [neuralResult, setNeuralResult] = useState<RegressionResponse | null>(null);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [sparseFeatureCols, setSparseFeatureCols] = useState<string[]>([]);
  const [modelingMode, setModelingMode] = useState('full');
  const [modelingLoss, setModelingLoss] = useState('l2');
  const [activeModelingSurface, setActiveModelingSurface] = useState<ActiveModelingSurface>('forecast');
  const [neuralPipeline, setNeuralPipeline] = useState('embedding');
  const [neuralIdCol, setNeuralIdCol] = useState('');
  const [graphSourceCol, setGraphSourceCol] = useState('');
  const [graphTargetCol, setGraphTargetCol] = useState('');
  const [graphWeightCol, setGraphWeightCol] = useState('');
  const [status, setStatus] = useState('Drop a CSV, TSV, or Parquet file to start.');
  const [isRunning, setIsRunning] = useState(false);
  const [isLoadingTaxiLane, setIsLoadingTaxiLane] = useState(false);
  const [isLoadingTaxiVariedRoutes, setIsLoadingTaxiVariedRoutes] = useState(false);
  const [runProgress, setRunProgress] = useState<RunProgress | null>(null);
  const [runLog, setRunLog] = useState<RunLogEntry[]>([]);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(fallbackModelOptions);
  const presetAppliedRef = useRef(false);
  const runLogIdRef = useRef(0);
  const [pendingLabRun, setPendingLabRun] = useState<PendingLabRun | null>(null);

  const previewRows = table?.rows.slice(0, 6) ?? [];
  const selectedForecastModel = modelOptions.find((option) => option.value === model) ?? modelOptions[0];
  const isLoadingTaxiSample = isLoadingTaxiLane || isLoadingTaxiVariedRoutes;
  const selectedColumnsReady =
    table !== null && timestampCol !== '' && targetCol !== '' && table.columns.includes(timestampCol) && table.columns.includes(targetCol);

  const appendRunLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
    runLogIdRef.current += 1;
    setRunLog((current) => [{id: runLogIdRef.current, message: `${timestamp} ${message}`}, ...current].slice(0, 8));
  }, []);

  const scheduleRun = useCallback((runner: () => Promise<void>) => {
    window.setTimeout(() => {
      void runner();
    }, 30);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void getForecastModelOptions(wasmJsUrl, wasmBinaryUrl)
      .then((options) => {
        if (cancelled) {
          return;
        }
        setModelOptions(options);
        setModel((current) => (options.some((option) => option.value === current) ? current : options[0]?.value ?? 'auto_forecast'));
      })
      .catch(() => {
        if (!cancelled) {
          setModelOptions(fallbackModelOptions);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [wasmBinaryUrl, wasmJsUrl]);

  const loadText = useCallback((text: string, fileName: string) => {
    const parsed = parseDelimited(text, fileName);
    const nextTimestampCol = guessColumn(parsed.columns, ['timestamp', 'tpep_pickup_datetime', 'pickup_datetime', 'date', 'ds', 'time']) ?? parsed.columns[0] ?? '';
    const nextTargetCol = guessColumn(parsed.columns, ['target', 'total_amount', 'fare_amount', 'trip_distance', 'y', 'demand', 'trips', 'count', 'fare', 'duration']) ?? parsed.columns[1] ?? '';
    const nextSeriesCol = guessColumn(parsed.columns, ['series_id', 'unique_id', 'PULocationID', 'DOLocationID', 'zone', 'route']) ?? '';
    setTable(parsed);
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setHiddenBenchmarkSeries([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setTimestampCol(nextTimestampCol);
    setTargetCol(nextTargetCol);
    setSeriesCol(nextSeriesCol);
    setFeatureCols(defaultFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
    setSparseFeatureCols(defaultSparseFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
    setNeuralIdCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'zone_id', 'id']) ?? '');
    setGraphSourceCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) ?? '');
    setGraphTargetCol(guessColumn(parsed.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']) ?? '');
    setGraphWeightCol(guessColumn(parsed.columns, ['edge_weight', 'weight', 'trip_count']) ?? '');
    setStatus(`Loaded ${parsed.rows.length.toLocaleString()} rows from ${fileName}.`);
  }, []);

  const loadFile = useCallback(
    async (file: File) => {
      if (file.name.toLowerCase().endsWith('.parquet')) {
        const parsed = await parseParquet(file);
        const nextTimestampCol = guessColumn(parsed.columns, ['timestamp', 'tpep_pickup_datetime', 'pickup_datetime', 'date', 'ds', 'time']) ?? parsed.columns[0] ?? '';
        const nextTargetCol = guessColumn(parsed.columns, ['target', 'total_amount', 'fare_amount', 'trip_distance', 'y', 'demand', 'trips', 'count', 'fare', 'duration']) ?? parsed.columns[1] ?? '';
        const nextSeriesCol = guessColumn(parsed.columns, ['series_id', 'unique_id', 'PULocationID', 'DOLocationID', 'zone', 'route']) ?? '';
        setTable(parsed);
        setResult(null);
        setComparisonResults([]);
        setBacktestResults([]);
        setHiddenBenchmarkSeries([]);
        setRegressionResult(null);
        setNeuralResult(null);
        setTimestampCol(nextTimestampCol);
        setTargetCol(nextTargetCol);
        setSeriesCol(nextSeriesCol);
        setFeatureCols(defaultFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
        setSparseFeatureCols(defaultSparseFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
        setNeuralIdCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'zone_id', 'id']) ?? '');
        setGraphSourceCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) ?? '');
        setGraphTargetCol(guessColumn(parsed.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']) ?? '');
        setGraphWeightCol(guessColumn(parsed.columns, ['edge_weight', 'weight', 'trip_count']) ?? '');
        setStatus(`Loaded ${parsed.rows.length.toLocaleString()} rows from ${file.name}.`);
        return;
      }
      const text = await file.text();
      loadText(text, file.name);
    },
    [loadText],
  );

  const applyTaxiTable = useCallback((parsed: ParsedTable, message: string) => {
    const nextTimestampCol = guessColumn(parsed.columns, ['timestamp', 'tpep_pickup_datetime', 'pickup_datetime', 'date', 'ds', 'time']) ?? parsed.columns[0] ?? '';
    const nextTargetCol = guessColumn(parsed.columns, ['target', 'total_amount', 'fare_amount', 'trip_distance', 'y', 'demand', 'trips', 'count', 'fare', 'duration']) ?? parsed.columns[1] ?? '';
    const nextSeriesCol = guessColumn(parsed.columns, ['series_id', 'unique_id', 'PULocationID', 'DOLocationID', 'zone', 'route']) ?? '';
    const nextFeatureCols = defaultFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol);
    const nextSparseFeatureCols = defaultSparseFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol);
    const profile = buildTargetingProfile(
      parsed,
      nextTargetCol,
      nextTimestampCol,
      nextSeriesCol,
      nextFeatureCols,
      nextSparseFeatureCols,
      modelOptions,
    );
    setTable(parsed);
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setHiddenBenchmarkSeries([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setTimestampCol(nextTimestampCol);
    setTargetCol(nextTargetCol);
    setSeriesCol(nextSeriesCol);
    setFrequency('hourly');
    setSeasonLength(24);
    setModel(profile.forecastModelValue);
    setModelingMode(profile.splitterModeValue);
    setModelingLoss(profile.lossValue);
    setFeatureCols(profile.featureCols);
    setSparseFeatureCols(profile.sparseFeatureCols);
    setNeuralPipeline(profile.neuralPipeline);
    setNeuralIdCol(profile.neuralIdCol || (guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'zone_id', 'id']) ?? ''));
    setGraphSourceCol(profile.graphSourceCol || (guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) ?? ''));
    setGraphTargetCol(profile.graphTargetCol || (guessColumn(parsed.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']) ?? ''));
    setGraphWeightCol(profile.graphWeightCol || (guessColumn(parsed.columns, ['edge_weight', 'weight', 'trip_count']) ?? ''));
    setStatus(`${message} Recommended ${profile.forecastModel} is selected.`);
  }, [modelOptions]);

  const loadTaxiLaneSample = useCallback(async () => {
    setIsLoadingTaxiLane(true);
    setStatus('Loading the 5,000-row single-lane taxi demand sample.');
    try {
      await waitForBrowserPaint();
      const response = await fetch(taxiLaneSampleUrl);
      if (!response.ok) {
        throw new Error(`Unable to load taxi lane sample (${response.status}).`);
      }
      setStatus('Parsing single-lane taxi demand sample.');
      await waitForBrowserPaint();
      const parsed = buildTaxiRouteHourSampleTable(
        await parseParquetBuffer(
          await response.arrayBuffer(),
          'yellow_taxi_2024-01-single-lane-5000.parquet',
        ),
        TAXI_LANE_SAMPLE_ROWS,
        {fileName: 'yellow_taxi_2024-01-single-lane-5000.parquet'},
      );
      applyTaxiTable(parsed, `Loaded ${parsed.rows.length.toLocaleString()} hourly rows for taxi lane PU132-DO236.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingTaxiLane(false);
    }
  }, [applyTaxiTable, taxiLaneSampleUrl]);

  const loadTaxiVariedRouteSample = useCallback(async () => {
    setIsLoadingTaxiVariedRoutes(true);
    setStatus('Loading the varied-route taxi geography sample.');
    try {
      await waitForBrowserPaint();
      const response = await fetch(taxiVariedRouteSampleUrl);
      if (!response.ok) {
        throw new Error(`Unable to load varied-route taxi sample (${response.status}).`);
      }
      setStatus('Parsing varied-route taxi geography sample.');
      await waitForBrowserPaint();
      const parsed = buildTaxiRouteHourSampleTable(
        await parseParquetBuffer(
          await response.arrayBuffer(),
          'yellow_taxi_2024-01-varied-routes-2500.parquet',
        ),
        TAXI_VARIED_ROUTE_SAMPLE_ROWS,
        {
          balancedManhattanRoutes: true,
          fileName: 'yellow_taxi_2024-01-varied-routes-2500.parquet',
        },
      );
      const laneCount = new Set(parsed.rows.map((row) => row.series_id).filter(Boolean)).size;
      applyTaxiTable(parsed, `Loaded ${parsed.rows.length.toLocaleString()} route-hour rows across ${laneCount.toLocaleString()} varied Manhattan pickup/dropoff lanes.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingTaxiVariedRoutes(false);
    }
  }, [applyTaxiTable, taxiVariedRouteSampleUrl]);

  useEffect(() => {
    if (presetAppliedRef.current) {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const requestedModel = normalizedForecastModel(params.get('model') ?? '');
    const requestedAction = params.get('run')?.trim().toLowerCase() ?? '';
    const requestedSample = params.get('sample')?.trim().toLowerCase() ?? '';
    if (!requestedModel && !requestedAction && !requestedSample) {
      return;
    }

    presetAppliedRef.current = true;
    const modelValue = modelOptions.some((option) => option.value === requestedModel)
      ? requestedModel
      : modelOptions[0]?.value ?? 'auto_forecast';
    const sample =
      requestedSample ||
      (modelValue === 'kriging' || modelValue === 'spatial_piecewise_kriging' ? 'varied' : 'lane');
    const action =
      requestedAction === 'benchmark' || requestedAction === 'compare' || requestedAction === 'backtest' || requestedAction === 'forecast'
        ? requestedAction
        : null;

    void (async () => {
      if (sample === 'varied' || sample === 'routes' || sample === 'spatial') {
        await loadTaxiVariedRouteSample();
      } else if (sample === 'lane' || sample === 'single' || sample === 'taxi') {
        await loadTaxiLaneSample();
      }
      setModel(modelValue);
      setActiveModelingSurface('forecast');
      if (action) {
        setPendingLabRun({action, model: modelValue});
      }
    })();
  }, [loadTaxiLaneSample, loadTaxiVariedRouteSample, modelOptions]);

  const onDrop = useCallback(
    async (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      const file = event.dataTransfer.files.item(0);
      if (file) {
        await loadFile(file);
      }
    },
    [loadFile],
  );

  const runForecast = useCallback(async () => {
    if (!table || !selectedColumnsReady) {
      setStatus('Choose timestamp and target columns before running a forecast.');
      return;
    }
    setIsRunning(true);
    setRunProgress({label: 'Fitting forecast'});
    setStatus('Fitting the forecast in this page.');
    try {
      const response = await runBrowserForecast({
        wasmJsUrl,
        wasmBinaryUrl,
        table,
        timestampCol,
        targetCol,
        seriesCol,
        frequency,
        horizon,
        model,
        seasonLength,
      });
      setResult(response);
      setComparisonResults([]);
      setBacktestResults([]);
      setHiddenBenchmarkSeries([]);
      setRegressionResult(null);
      setNeuralResult(null);
      setStatus(`Forecast complete with ${response.forecast.records.length.toLocaleString()} prediction rows.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setResult(null);
    } finally {
      setIsRunning(false);
      setRunProgress(null);
    }
  }, [
    frequency,
    horizon,
    model,
    seasonLength,
    selectedColumnsReady,
    seriesCol,
    table,
    targetCol,
    timestampCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const runComparison = useCallback(async () => {
    if (!table || !selectedColumnsReady) {
      setStatus('Choose timestamp and target columns before running a benchmark.');
      return;
    }
    setIsRunning(true);
    setRunProgress({label: 'Preparing model roster', current: 0, total: modelOptions.length});
    setResult(null);
    setRegressionResult(null);
    setNeuralResult(null);
    setRunLog([]);
    setStatus('Running model roster benchmark.');
    const roster = modelOptions.filter((option) => forecastModelCanRun(option, table.columns));
    setRunProgress({label: 'Running model roster', current: 0, total: roster.length});
    const runId = benchmarkRunId();
    const runLabel = benchmarkRunLabel([...comparisonResults, ...backtestResults]);
    appendRunLog(`Started ${runLabel} with ${roster.length.toLocaleString()} models.`);
    await waitForBrowserPaint();
    const nextResults: ComparisonResult[] = [...comparisonResults];
    for (const [index, option] of roster.entries()) {
      setRunProgress({label: `Running ${option.label}`, current: index, total: roster.length});
      setStatus(`Running ${option.label}.`);
      appendRunLog(`Running ${option.label}.`);
      await waitForBrowserPaint();
      const started = performance.now();
      try {
        const response = await runBrowserForecast({
          wasmJsUrl,
          wasmBinaryUrl,
          table,
          timestampCol,
          targetCol,
          seriesCol,
          frequency,
          horizon,
          model: option.value,
          seasonLength,
        });
        nextResults.push({
          runId,
          runLabel,
          requestedModel: option.value,
          label: option.label,
          pipeline: option.group,
          response,
        });
        appendRunLog(`Finished ${option.label} in ${formatElapsedMs(performance.now() - started)}.`);
      } catch (error) {
        appendRunLog(`${option.label} reported a constraint after ${formatElapsedMs(performance.now() - started)}.`);
        nextResults.push({
          runId,
          runLabel,
          requestedModel: option.value,
          label: option.label,
          pipeline: option.group,
          error: error instanceof Error ? error.message : String(error),
        });
      }
      setComparisonResults([...nextResults]);
      setRunProgress({label: `Checked ${option.label}`, current: index + 1, total: roster.length});
      await waitForBrowserPaint();
    }
    const successes = nextResults.filter((item) => item.response).length;
    const runRows = nextResults.filter((item) => item.runId === runId);
    const runSuccesses = runRows.filter((item) => item.response).length;
    setStatus(`${runLabel} complete: ${runSuccesses.toLocaleString()} succeeded, ${(runRows.length - runSuccesses).toLocaleString()} reported constraints. Previous plotted runs are still available.`);
    appendRunLog(`${runLabel} complete: ${runSuccesses.toLocaleString()} succeeded.`);
    setIsRunning(false);
    setRunProgress(null);
  }, [
    appendRunLog,
    backtestResults,
    comparisonResults,
    frequency,
    horizon,
    modelOptions,
    seasonLength,
    selectedColumnsReady,
    seriesCol,
    table,
    targetCol,
    timestampCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const runBacktest = useCallback(async () => {
    if (!table || !selectedColumnsReady) {
      setStatus('Choose timestamp and target columns before running a holdout backtest.');
      return;
    }
    setIsRunning(true);
    setRunProgress({label: 'Preparing holdout backtest'});
    setResult(null);
    setRegressionResult(null);
    setNeuralResult(null);
    setRunLog([]);
    setStatus('Running holdout benchmark across the model roster.');
    try {
      const split = holdoutSplit(table, timestampCol, targetCol, seriesCol, horizon);
      const roster = modelOptions.filter((option) => forecastModelCanRun(option, table.columns));
      setRunProgress({label: 'Running holdout backtest', current: 0, total: roster.length});
      const runId = benchmarkRunId();
      const runLabel = benchmarkRunLabel([...backtestResults, ...comparisonResults]);
      appendRunLog(`Started ${runLabel} with ${roster.length.toLocaleString()} models.`);
      await waitForBrowserPaint();
      const nextResults: BacktestResult[] = [...backtestResults];
      let completed = 0;
      await runWithConcurrency(roster, backtestConcurrency(), async (option, index) => {
        setRunProgress({label: `Backtesting ${option.label}`, current: completed, total: roster.length});
        setStatus(`Backtesting ${option.label}.`);
        appendRunLog(`Backtesting ${option.label}.`);
        const started = performance.now();
        let result: BacktestResult;
        try {
          const response = await runBrowserForecast({
            wasmJsUrl,
            wasmBinaryUrl,
            table: split.train,
            timestampCol,
            targetCol,
            seriesCol,
            frequency,
            horizon,
            model: option.value,
            seasonLength,
            isolatedWorker: true,
            timeoutMs: BACKTEST_FORECAST_TIMEOUT_MS,
          });
          const metrics = evaluateHoldout(response.forecast.records, split.actuals);
          result = {
            runId,
            runLabel,
            requestedModel: option.value,
            label: option.label,
            pipeline: option.group,
            response,
            ...metrics,
          };
          appendRunLog(`Scored ${option.label} in ${formatElapsedMs(performance.now() - started)}.`);
        } catch (error) {
          appendRunLog(`${option.label} reported a constraint after ${formatElapsedMs(performance.now() - started)}.`);
          result = {
            runId,
            runLabel,
            requestedModel: option.value,
            label: option.label,
            pipeline: option.group,
            error: error instanceof Error ? error.message : String(error),
          };
        }
        nextResults.push(result);
        completed += 1;
        setBacktestResults([...nextResults]);
        setRunProgress({label: `Checked ${option.label}`, current: completed, total: roster.length});
        await waitForBrowserPaint();
      });
      const runRows = nextResults.filter((item) => item.runId === runId);
      const successes = runRows.filter((item) => item.rmse !== undefined).length;
      setStatus(`${runLabel} complete: ${successes.toLocaleString()} scored, ${(runRows.length - successes).toLocaleString()} reported constraints. Earlier plotted models were kept.`);
      appendRunLog(`${runLabel} complete: ${successes.toLocaleString()} scored.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      appendRunLog(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
      setRunProgress(null);
    }
  }, [
    appendRunLog,
    backtestResults,
    comparisonResults.length,
    frequency,
    horizon,
    modelOptions,
    seasonLength,
    selectedColumnsReady,
    seriesCol,
    table,
    targetCol,
    timestampCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  useEffect(() => {
    if (!pendingLabRun || !table || !selectedColumnsReady || isLoadingTaxiSample || isRunning) {
      return;
    }
    if (model !== pendingLabRun.model) {
      return;
    }
    setPendingLabRun(null);
    if (pendingLabRun.action === 'compare' || pendingLabRun.action === 'backtest' || pendingLabRun.action === 'benchmark') {
      scheduleRun(runBacktest);
    } else {
      scheduleRun(runForecast);
    }
  }, [
    isLoadingTaxiSample,
    isRunning,
    model,
    pendingLabRun,
    runBacktest,
    runForecast,
    scheduleRun,
    selectedColumnsReady,
    table,
  ]);

  const runRegression = useCallback(async () => {
    if (!table || !selectedColumnsReady || featureCols.length === 0) {
      setStatus('Choose a target and at least one numeric feature before running modeling.');
      return;
    }
    setIsRunning(true);
    setRunProgress({label: 'Fitting regression model'});
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setHiddenBenchmarkSeries([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setStatus('Fitting CartoBoost regression in this page.');
    try {
      await waitForBrowserPaint();
      const response = await runBrowserRegression({
        wasmJsUrl,
        wasmBinaryUrl,
        table,
        targetCol,
        featureCols,
        sparseFeatureCols,
        modelingMode,
        modelingLoss,
      });
      setRegressionResult(response);
      setStatus(modelVisualizerStatus('Modeling complete', response, table.rows.length));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
      setRunProgress(null);
    }
  }, [featureCols, modelingLoss, modelingMode, selectedColumnsReady, sparseFeatureCols, table, targetCol, wasmBinaryUrl, wasmJsUrl]);

  const runNeural = useCallback(async () => {
    if (!table || !selectedColumnsReady || featureCols.length === 0) {
      setStatus('Choose a target and at least one dense feature before running neural modeling.');
      return;
    }
    if (neuralPipeline === 'embedding' && !neuralIdCol) {
      setStatus('Choose an ID column before running embedding modeling.');
      return;
    }
    if (graphNeuralPipelines.has(neuralPipeline) && (!graphSourceCol || !graphTargetCol)) {
      setStatus(`Choose source and target node columns before running ${neuralPipelineLabels[neuralPipeline] ?? neuralPipeline}.`);
      return;
    }
    setIsRunning(true);
    setRunProgress({label: `Fitting ${neuralPipelineLabels[neuralPipeline] ?? neuralPipeline}`});
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setHiddenBenchmarkSeries([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setStatus(`Fitting ${neuralPipelineLabels[neuralPipeline] ?? neuralPipeline} in this page.`);
    try {
      await waitForBrowserPaint();
      const response = await runBrowserNeural({
        wasmJsUrl,
        wasmBinaryUrl,
        table,
        targetCol,
        featureCols,
        neuralPipeline,
        neuralIdCol,
        graphSourceCol,
        graphTargetCol,
        graphWeightCol,
      });
      setNeuralResult(response);
      setStatus(modelVisualizerStatus('Neural modeling complete', response, table.rows.length));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
      setRunProgress(null);
    }
  }, [
    featureCols,
    graphSourceCol,
    graphTargetCol,
    graphWeightCol,
    neuralIdCol,
    neuralPipeline,
    selectedColumnsReady,
    table,
    targetCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const exportSuggestedConfig = useCallback(() => {
    if (!table) {
      setStatus('Load a dataset before exporting a suggested config.');
      return;
    }
    const config = buildSuggestedConfig({
      table,
      timestampCol,
      targetCol,
      seriesCol,
      frequency,
      horizon,
      seasonLength,
      model,
      modelOptions,
      featureCols,
      sparseFeatureCols,
      modelingMode,
      modelingLoss,
      neuralPipeline,
      neuralIdCol,
      graphSourceCol,
      graphTargetCol,
      graphWeightCol,
    });
    downloadJson(`${stripExtension(table.fileName)}-cartoboost-config.json`, config);
    setStatus(`Exported suggested config for ${table.fileName}.`);
  }, [featureCols, frequency, graphSourceCol, graphTargetCol, graphWeightCol, horizon, model, modelOptions, modelingLoss, modelingMode, neuralIdCol, neuralPipeline, seasonLength, seriesCol, sparseFeatureCols, table, targetCol, timestampCol]);

  const applyRecommendedSettings = useCallback(() => {
    if (!table) {
      setStatus('Load a dataset before applying recommended settings.');
      return;
    }
    const profile = buildTargetingProfile(table, targetCol, timestampCol, seriesCol, featureCols, sparseFeatureCols, modelOptions);
    setModel(profile.forecastModelValue);
    setModelingMode(profile.splitterModeValue);
    setModelingLoss(profile.lossValue);
    setFeatureCols(profile.featureCols);
    setSparseFeatureCols(profile.sparseFeatureCols);
    if (profile.graphSourceCol) {
      setGraphSourceCol(profile.graphSourceCol);
    }
    if (profile.graphTargetCol) {
      setGraphTargetCol(profile.graphTargetCol);
    }
    if (profile.graphWeightCol) {
      setGraphWeightCol(profile.graphWeightCol);
    }
    if (profile.neuralIdCol) {
      setNeuralIdCol(profile.neuralIdCol);
    }
    setNeuralPipeline(profile.neuralPipeline);
    setStatus(`Applied recommended ${profile.forecastModel} forecast and ${profile.splitterMode} modeling settings.`);
  }, [featureCols, modelOptions, seriesCol, sparseFeatureCols, table, targetCol, timestampCol]);

  const chartRows = useMemo(() => {
    if (!result) {
      return [];
    }
    const firstSeries = result.forecast.records[0]?.series_id;
    return result.forecast.records.filter((row) => row.series_id === firstSeries);
  }, [result]);
  const actualRows = useMemo(
    () => actualRowsForFirstSeries(table, timestampCol, targetCol, seriesCol),
    [seriesCol, table, targetCol, timestampCol],
  );
  const benchmarkRows = backtestResults.length > 0 ? backtestResults : comparisonResults;
  const visibleBenchmarkRows = benchmarkRows.filter((item) => !hiddenBenchmarkSeries.includes(benchmarkSeriesKey(item)));
  const toggleBenchmarkSeries = useCallback((key: string) => {
    setHiddenBenchmarkSeries((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key],
    );
  }, []);

  return (
    <main className={styles.shell}>
      <section className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Browser-local modeling lab</span>
          <h1>Model taxi demand, routes, and neural signals interactively</h1>
          <p>
            Run CartoBoost forecasts, regression, graph models, and neural signals on taxi data.
            Load a bundled sample or upload your own file for an interactive check.
          </p>
        </div>
        <div className={styles.headerActions}>
          <button className={styles.secondaryButton} type="button" disabled={isLoadingTaxiSample || isRunning} onClick={() => void loadTaxiLaneSample()}>
            {isLoadingTaxiLane ? 'Loading taxi lane' : 'Load taxi lane'}
          </button>
          <button className={styles.secondaryButton} type="button" disabled={isLoadingTaxiSample || isRunning} onClick={() => void loadTaxiVariedRouteSample()}>
            {isLoadingTaxiVariedRoutes ? 'Loading varied routes' : 'Load varied routes'}
          </button>
          {isLoadingTaxiSample && (
            <div className={styles.loadingBar} role="progressbar" aria-label={isLoadingTaxiVariedRoutes ? 'Loading varied routes' : 'Loading taxi lane'}>
              <span />
            </div>
          )}
        </div>
      </section>

      <section className={styles.workspace}>
        <div className={styles.controlPanel}>
          <label
            className={styles.dropZone}
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
            <input
              accept=".csv,.tsv,.parquet,text/csv,text/tab-separated-values,application/vnd.apache.parquet"
              type="file"
              onChange={(event) => {
                const file = event.target.files?.item(0);
                if (file) {
                  void loadFile(file);
                }
              }}
            />
            <strong>{table ? table.fileName : 'Drop CSV, TSV, or Parquet'}</strong>
            <span>{table ? `${table.rows.length.toLocaleString()} rows ready` : 'Choose CSV, TSV, or Parquet, then map timestamp, target, and series columns.'}</span>
          </label>

          <div className={styles.controlStack}>
            <ControlSection title="Dataset mapping" step="1">
              <div className={`${styles.controlsGrid} ${styles.mappingGrid}`}>
                <Select label="Timestamp" value={timestampCol} onChange={setTimestampCol} options={table?.columns ?? []} />
                <Select label="Target" value={targetCol} onChange={setTargetCol} options={table?.columns ?? []} />
                <Select label="Series" value={seriesCol} onChange={setSeriesCol} options={table?.columns ?? []} allowBlank blankLabel="Single series" />
              </div>
            </ControlSection>

            <div className={styles.surfaceBlock}>
              <div className={styles.stepHeader}>
                <span>2</span>
                <strong>Modeling surface</strong>
              </div>
              <div className={styles.surfaceTabs} aria-label="Modeling surface">
                <button
                  className={activeModelingSurface === 'forecast' ? styles.surfaceTabActive : undefined}
                  type="button"
                  onClick={() => setActiveModelingSurface('forecast')}
                >
                  Forecast
                </button>
                <button
                  className={activeModelingSurface === 'model' ? styles.surfaceTabActive : undefined}
                  type="button"
                  onClick={() => setActiveModelingSurface('model')}
                >
                  Model
                </button>
                <button
                  className={activeModelingSurface === 'neural' ? styles.surfaceTabActive : undefined}
                  type="button"
                  onClick={() => setActiveModelingSurface('neural')}
                >
                  Neural
                </button>
                <button
                  className={activeModelingSurface === 'deep' ? styles.surfaceTabActive : undefined}
                  type="button"
                  onClick={() => setActiveModelingSurface('deep')}
                >
                  Deep
                </button>
              </div>
            </div>

            {activeModelingSurface === 'forecast' && (
              <>
                <ControlSection title="Forecast model" step="3">
                  <ModelPicker
                    modelOptions={modelOptions}
                    value={model}
                    onChange={setModel}
                  />
                </ControlSection>
                <ControlSection title="Forecast settings" step="4">
                  <ForecastModelSettings
                    selectedModel={selectedForecastModel}
                    columns={table?.columns ?? []}
                    frequency={frequency}
                    horizon={horizon}
                    seasonLength={seasonLength}
                    onFrequencyChange={setFrequency}
                    onHorizonChange={setHorizon}
                    onSeasonLengthChange={setSeasonLength}
                  />
                </ControlSection>
              </>
            )}

            {activeModelingSurface === 'model' && (
              <ControlSection title="Regression modeling" step="3">
                <div className={styles.controlsGrid}>
                  <Select
                    label="Splitter menu"
                    value={modelingMode}
                    onChange={setModelingMode}
                    options={['full', 'auto', 'axis', 'spatial', 'periodic']}
                    labels={{
                      full: 'Spatial + periodic toolkit',
                      auto: 'Auto dense',
                      axis: 'Axis only',
                      spatial: 'Spatial splitters',
                      periodic: 'Periodic splitters',
                    }}
                  />
                  <Select
                    label="Loss"
                    value={modelingLoss}
                    onChange={setModelingLoss}
                    options={['l2', 'l1', 'huber', 'log_l2', 'quantile']}
                    labels={{
                      l2: 'L2 mean',
                      l1: 'L1 median',
                      huber: 'Huber robust',
                      log_l2: 'Log-L2 positive',
                      quantile: 'Quantile median',
                    }}
                  />
                </div>
                {table && (
                  <FeatureSelector
                    columns={numericFeatureColumns(table, targetCol, timestampCol, seriesCol)}
                    selected={featureCols}
                    onChange={setFeatureCols}
                  />
                )}
                {table && shouldShowSparseFeatures(modelingMode) && (
                  <FeatureSelector
                    title="Sparse set features"
                    columns={sparseFeatureColumns(table, targetCol, timestampCol, seriesCol)}
                    selected={sparseFeatureCols}
                    onChange={setSparseFeatureCols}
                  />
                )}
              </ControlSection>
            )}

            {activeModelingSurface === 'neural' && (
              <ControlSection title="Neural and graph settings" step="3">
                <div className={styles.neuralSummary}>
                  <strong>{neuralPipelineLabels[neuralPipeline]}</strong>
                  <span>{graphNeuralPipelines.has(neuralPipeline) ? 'Source and target node columns are required.' : 'Choose an ID column for embedding features.'}</span>
                </div>
                <div className={styles.controlsGrid}>
                  <GroupedSelect
                    label="Neural pipeline"
                    value={neuralPipeline}
                    onChange={setNeuralPipeline}
                    groups={[
                      {label: 'Embeddings', options: [{value: 'embedding', label: neuralPipelineLabels.embedding}]},
                      {
                        label: 'Graph pipelines',
                        options: [
                          {value: 'node2vec', label: neuralPipelineLabels.node2vec},
                          {value: 'graphsage', label: neuralPipelineLabels.graphsage},
                          {value: 'hetero_graphsage', label: neuralPipelineLabels.hetero_graphsage},
                          {value: 'hinsage', label: neuralPipelineLabels.hinsage},
                        ],
                      },
                    ]}
                  />
                  {neuralPipeline === 'embedding' && (
                    <Select label="ID" value={neuralIdCol} onChange={setNeuralIdCol} options={table?.columns ?? []} allowBlank blankLabel="No ID column" />
                  )}
                  {graphNeuralPipelines.has(neuralPipeline) && (
                    <>
                      <Select label="Graph Source" value={graphSourceCol} onChange={setGraphSourceCol} options={table?.columns ?? []} allowBlank blankLabel="No source column" />
                      <Select label="Graph Target" value={graphTargetCol} onChange={setGraphTargetCol} options={table?.columns ?? []} allowBlank blankLabel="No target column" />
                      <Select label="Graph Weight" value={graphWeightCol} onChange={setGraphWeightCol} options={table?.columns ?? []} allowBlank blankLabel="Unweighted graph" />
                    </>
                  )}
                </div>
                {table && (
                  <FeatureSelector
                    columns={numericFeatureColumns(table, targetCol, timestampCol, seriesCol)}
                    selected={featureCols}
                    onChange={setFeatureCols}
                  />
                )}
                {table && graphNeuralPipelines.has(neuralPipeline) && (
                  <FeatureSelector
                    title="Sparse set features"
                    columns={sparseFeatureColumns(table, targetCol, timestampCol, seriesCol)}
                    selected={sparseFeatureCols}
                    onChange={setSparseFeatureCols}
                  />
                )}
              </ControlSection>
            )}

            {activeModelingSurface === 'deep' && (
              <ControlSection title="Deep model Wasm examples" step="3">
                <div className={styles.neuralSummary}>
                  <strong>Browser-native deep models</strong>
                  <span>Run the Rust-backed deep-model examples directly in this page with tiny synthetic taxi-shaped samples.</span>
                </div>
                <div className={styles.controlsGrid}>
                  <DeepModelWasmExample model="DirectionalPairForecaster" />
                  <DeepModelWasmExample model="ResponseCurveModel" />
                  <DeepModelWasmExample model="EventOutcomeModel" />
                  <DeepModelWasmExample model="ServiceTimeResidualModel" />
                  <DeepModelWasmExample model="SpatioTemporalGraphForecaster" />
                  <DeepModelWasmExample model="ConditionalFlowDistributionHead" />
                  <DeepModelWasmExample model="GeoTemporalDiffusionScenarioModel" />
                  <DeepModelWasmExample model="GraphNeuralOperator" />
                  <DeepModelWasmExample model="ChoiceSetTransformer" />
                  <DeepModelWasmExample model="RegimeMoEForecaster" />
                  <DeepModelWasmExample model="ConstrainedDecisionOptimizer" />
                </div>
              </ControlSection>
            )}

          </div>

          <div className={styles.actionBlock}>
            <div className={styles.actionHeader}>
              <span>5</span>
              <strong>Run and export</strong>
            </div>
            <div className={styles.actionGrid}>
            {activeModelingSurface === 'forecast' && (
              <>
                <button className={styles.primaryButton} type="button" disabled={!selectedColumnsReady || isRunning || isLoadingTaxiSample} onClick={() => scheduleRun(runForecast)}>
                  {isRunning ? 'Running forecast' : 'Run forecast'}
                </button>
                <button className={styles.secondaryActionButton} type="button" disabled={!selectedColumnsReady || isRunning || isLoadingTaxiSample} onClick={() => scheduleRun(runBacktest)}>
                  Benchmark roster
                </button>
              </>
            )}
            {activeModelingSurface === 'model' && (
              <button className={styles.primaryButton} type="button" disabled={!selectedColumnsReady || featureCols.length === 0 || isRunning || isLoadingTaxiSample} onClick={() => scheduleRun(runRegression)}>
                {isRunning ? 'Running model' : 'Run model'}
              </button>
            )}
            {activeModelingSurface === 'neural' && (
              <button className={styles.primaryButton} type="button" disabled={!selectedColumnsReady || featureCols.length === 0 || isRunning || isLoadingTaxiSample} onClick={() => scheduleRun(runNeural)}>
                {isRunning ? 'Running neural' : 'Run neural'}
              </button>
            )}
            {activeModelingSurface === 'deep' && (
              <div className={styles.settingHint}>These examples run directly in the browser and do not require uploaded data.</div>
            )}
            <button className={styles.secondaryActionButton} type="button" disabled={!table || isRunning || isLoadingTaxiSample} onClick={exportSuggestedConfig}>
              Export config
            </button>
            </div>
          </div>
          {(runProgress || isLoadingTaxiSample) && (
            <ProgressBar progress={runProgress} label={isLoadingTaxiVariedRoutes ? 'Loading varied routes' : isLoadingTaxiLane ? 'Loading taxi lane' : undefined} />
          )}
          {runLog.length > 0 && (
            <div className={styles.runLog} aria-live="polite" aria-label="Run activity">
              <strong>Run activity</strong>
              <ol>
                {runLog.map((entry) => (
                  <li key={entry.id}>{entry.message}</li>
                ))}
              </ol>
            </div>
          )}
          <p className={styles.status}>{status}</p>
        </div>

        <div className={styles.outputPanel}>
          {neuralResult ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Neural modeling</span>
                  <h2>{neuralResult.metadata.model}</h2>
                </div>
                <dl>
                  <div>
                    <dt>Train</dt>
                    <dd>{neuralResult.metrics.trainRows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Holdout</dt>
                    <dd>{neuralResult.metrics.holdoutRows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Visualized</dt>
                    <dd>{(neuralResult.metrics.trainRows + neuralResult.metrics.holdoutRows).toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <RegressionMetricSummary result={neuralResult} />
              <CartoBoostModelVisualizer result={neuralResult} />
              <RegressionPredictionChart result={neuralResult} />
              <FeatureImportanceChart result={neuralResult} />
              <RegressionPredictionTable result={neuralResult} />
            </>
          ) : regressionResult ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Modeling</span>
                  <h2>CartoBoost regression</h2>
                </div>
                <dl>
                  <div>
                    <dt>Train</dt>
                    <dd>{regressionResult.metrics.trainRows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Holdout</dt>
                    <dd>{regressionResult.metrics.holdoutRows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Visualized</dt>
                    <dd>{(regressionResult.metrics.trainRows + regressionResult.metrics.holdoutRows).toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <RegressionMetricSummary result={regressionResult} />
              <CartoBoostModelVisualizer result={regressionResult} />
              <RegressionPredictionChart result={regressionResult} />
              <FeatureImportanceChart result={regressionResult} />
              <RegressionPredictionTable result={regressionResult} />
            </>
          ) : benchmarkRows.length > 0 ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Benchmark</span>
                  <h2>Roster results</h2>
                </div>
                <dl>
                  <div>
                    <dt>Visible</dt>
                    <dd>{visibleBenchmarkRows.length}</dd>
                  </div>
                  <div>
                    <dt>Checked</dt>
                    <dd>{benchmarkRows.length}</dd>
                  </div>
                </dl>
              </div>
              <BenchmarkModelToggles results={benchmarkRows} hidden={hiddenBenchmarkSeries} onToggle={toggleBenchmarkSeries} />
              <ComparisonChart actualRows={actualRows} results={visibleBenchmarkRows} />
              {backtestResults.length > 0 && <BacktestMetricChart results={visibleBenchmarkRows as BacktestResult[]} />}
              {backtestResults.length > 0 ? <BacktestTable results={visibleBenchmarkRows as BacktestResult[]} /> : <ComparisonTable results={visibleBenchmarkRows as ComparisonResult[]} />}
            </>
          ) : result ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Result</span>
                  <h2>{result.metadata.model}</h2>
                </div>
                <dl>
                  <div>
                    <dt>Rows</dt>
                    <dd>{result.metadata.input.n_rows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Series</dt>
                    <dd>{result.metadata.input.series_ids.length.toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <ForecastSignalSummary actualRows={actualRows} forecastRows={chartRows} frequency={frequency} seasonLength={seasonLength} />
              <ForecastChart actualRows={actualRows} rows={chartRows} quantiles={firstSeriesQuantileRows(result)} />
              {table && <GeoDatasetVisualization table={table} targetCol={targetCol} seriesCol={seriesCol} />}
              <ForecastComponentSummary components={result.components} />
              <ProphetDebuggerPanel response={result} />
              <ForecastTable records={result.forecast.records.slice(0, 12)} />
            </>
          ) : (
            <>
              <div className={styles.emptyState}>
                <div>
                  <span className={styles.eyebrow}>Preview</span>
                  <h2>{table ? 'Dataset loaded' : 'Waiting for data'}</h2>
                </div>
              </div>
              {table && (
                <>
                  <div className={styles.previewGrid}>
                    <DatasetProfile table={table} timestampCol={timestampCol} targetCol={targetCol} seriesCol={seriesCol} />
                    <TargetRunPlan
                      table={table}
                      targetCol={targetCol}
                      timestampCol={timestampCol}
                      seriesCol={seriesCol}
                      featureCols={featureCols}
                      sparseFeatureCols={sparseFeatureCols}
                      modelOptions={modelOptions}
                      onApply={applyRecommendedSettings}
                    />
                    <TargetDiagnostics table={table} targetCol={targetCol} timestampCol={timestampCol} seriesCol={seriesCol} />
                    <TargetOpportunityPanel table={table} targetCol={targetCol} timestampCol={timestampCol} seriesCol={seriesCol} />
                    <FeatureRoleMatrix
                      table={table}
                      targetCol={targetCol}
                      timestampCol={timestampCol}
                      seriesCol={seriesCol}
                      featureCols={featureCols}
                      sparseFeatureCols={sparseFeatureCols}
                    />
                    <GeoDatasetVisualization table={table} targetCol={targetCol} seriesCol={seriesCol} />
                  </div>
                  <PreviewTable columns={table.columns} rows={previewRows} />
                </>
              )}
            </>
          )}
        </div>
      </section>
    </main>
  );
}

type SelectGroup = {
  label: string;
  options: {value: string; label: string}[];
};

function ControlSection({title, step, children}: {title: string; step?: string; children: React.ReactNode}) {
  return (
    <section className={styles.controlSection}>
      <div className={styles.stepHeader}>
        {step && <span>{step}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Select({
  label,
  value,
  options,
  labels = {},
  allowBlank = false,
  blankLabel = 'None',
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  labels?: Record<string, string>;
  allowBlank?: boolean;
  blankLabel?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {allowBlank && <option value="">{blankLabel}</option>}
        {options.map((option) => (
          <option value={option} key={option}>
            {labels[option] ?? option}
          </option>
        ))}
      </select>
    </label>
  );
}

function GroupedSelect({
  label,
  value,
  groups,
  onChange,
}: {
  label: string;
  value: string;
  groups: SelectGroup[];
  onChange: (value: string) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {groups.map((group) => (
          <optgroup label={group.label} key={group.label}>
            {group.options.map((option) => (
              <option value={option.value} key={option.value}>
                {option.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

function NumberInput({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SeasonalityControl({
  frequency,
  value,
  onChange,
}: {
  frequency: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const presets = seasonalityPresets(frequency);
  const matched = presets.find((preset) => preset.length === value);
  return (
    <div className={styles.seasonalityControl}>
      <div className={styles.seasonalityHeader}>
        <span>Seasonality</span>
        <strong>{matched?.label ?? `${value.toLocaleString()} steps`}</strong>
      </div>
      <div className={styles.seasonalityButtons}>
        {presets.map((preset) => (
          <button
            className={preset.length === value ? styles.seasonalityButtonActive : undefined}
            type="button"
            onClick={() => onChange(preset.length)}
            key={preset.key}
            title={`${preset.description}: ${preset.length.toLocaleString()} steps`}
          >
            <strong>{preset.label}</strong>
            <span>{preset.length.toLocaleString()}</span>
          </button>
        ))}
      </div>
      <NumberInput label="Custom Period" value={value} min={1} max={8784} onChange={onChange} />
    </div>
  );
}

function ForecastModelSettings({
  selectedModel,
  columns,
  frequency,
  horizon,
  seasonLength,
  onFrequencyChange,
  onHorizonChange,
  onSeasonLengthChange,
}: {
  selectedModel?: ModelOption;
  columns: string[];
  frequency: string;
  horizon: number;
  seasonLength: number;
  onFrequencyChange: (value: string) => void;
  onHorizonChange: (value: number) => void;
  onSeasonLengthChange: (value: number) => void;
}) {
  const profile = forecastSettingsProfile(selectedModel);
  return (
    <>
      <div className={styles.controlsGrid}>
        <Select label="Frequency" value={frequency} onChange={onFrequencyChange} options={['hourly', 'daily', 'weekly']} />
        <NumberInput label="Horizon" value={horizon} min={1} max={365} onChange={onHorizonChange} />
        {profile.showSeasonality && (
          <SeasonalityControl frequency={frequency} value={seasonLength} onChange={onSeasonLengthChange} />
        )}
      </div>
      {profile.notice && <p className={styles.settingHint}>{profile.notice}</p>}
      {profile.needsCoordinates && !hasCoordinateColumns(columns) && (
        <p className={styles.settingHint}>Kriging requires pickup/dropoff or longitude/latitude columns before it can run.</p>
      )}
    </>
  );
}

function forecastSettingsProfile(selectedModel?: ModelOption) {
  const value = selectedModel?.value ?? '';
  const levelOnlyModels = new Set([
    'intermittent_demand',
    'croston',
    'sba',
    'tsb',
  ]);
  const seasonalModels = new Set([
    'auto_forecast',
    'cartoboost_lag',
    'cartoboost_direct',
    'rectified_recursive',
    'lag_plus',
    'scaled_cartoboost_lag',
    'log1p_cartoboost_lag',
    'classical_expert_bank',
    'autostats_bank',
    'stl_cartoboost',
    'mstl_cartoboost',
    'seasonal_naive',
    'seasonal_window_average',
    'theta',
    'optimized_theta',
    'auto_ets',
    'seasonal_ets',
    'piecewise_linear_seasonal',
  ]);
  const needsCoordinates = forecastModelNeedsCoordinates(selectedModel);
  if (needsCoordinates) {
    if (isSpatialPiecewiseKrigingModel(value)) {
      return {
        needsCoordinates,
        showSeasonality: true,
        notice: 'Spatial piecewise kriging uses stable coordinates, the piecewise seasonal base, and optional numeric spatial regressors.',
      };
    }
    return {
      needsCoordinates,
      showSeasonality: false,
      notice: 'Spatial kriging uses coordinates and horizon; seasonality is not part of this model setup.',
    };
  }
  if (levelOnlyModels.has(value)) {
    return {
      needsCoordinates,
      showSeasonality: false,
      notice: 'Intermittent-demand methods estimate a constant expected demand level. They do not model seasonal cycles.',
    };
  }
  if (value === 'ets') {
    return {
      needsCoordinates,
      showSeasonality: false,
      notice: 'ETS models level and trend only. Select Seasonal ETS or Auto ETS when a seasonal cycle is part of the signal.',
    };
  }
  if (!seasonalModels.has(value)) {
    return {
      needsCoordinates,
      showSeasonality: false,
      notice: 'This local baseline only needs frequency and horizon.',
    };
  }
  return {
    needsCoordinates,
    showSeasonality: true,
    notice: '',
  };
}

function forecastModelNeedsCoordinates(option?: ModelOption) {
  return option?.value === 'kriging' || option?.value === 'spatial_piecewise_kriging' || option?.group === 'spatial';
}

function forecastModelCanRun(option: ModelOption, columns: string[]) {
  return !forecastModelNeedsCoordinates(option) || hasCoordinateColumns(columns);
}

function shouldShowSparseFeatures(modelingMode: string) {
  return modelingMode === 'full' || modelingMode === 'auto' || modelingMode === 'spatial';
}

function forecastModelGroups(modelOptions: ModelOption[]): SelectGroup[] {
  const grouped = new Map<string, {value: string; label: string}[]>();
  for (const option of modelOptions) {
    const groupLabel = forecastPipelineLabels[option.group] ?? option.group;
    const group = grouped.get(groupLabel) ?? [];
    group.push({value: option.value, label: option.label});
    grouped.set(groupLabel, group);
  }
  return Array.from(grouped, ([label, options]) => ({label, options}));
}

function ModelPicker({
  modelOptions,
  value,
  onChange,
}: {
  modelOptions: ModelOption[];
  value: string;
  onChange: (value: string) => void;
}) {
  const groups = useMemo(() => forecastModelGroups(modelOptions), [modelOptions]);
  return (
    <div className={styles.modelPicker}>
      <GroupedSelect label="Forecast model" value={value} onChange={onChange} groups={groups} />
    </div>
  );
}

function ProgressBar({progress, label}: {progress: RunProgress | null; label?: string}) {
  const total = progress?.total ?? 0;
  const current = progress?.current ?? 0;
  const hasValue = total > 0;
  const progressLabel = label ?? progress?.label ?? 'Working';
  const clampedCurrent = Math.min(current, total);
  const percentage = hasValue ? Math.min(100, Math.max(0, (clampedCurrent / total) * 100)) : undefined;
  return (
    <div className={styles.progressWrap}>
      <div className={styles.progressMeta}>
        <span>{progressLabel}</span>
        {hasValue && <strong>{`${clampedCurrent.toLocaleString()} / ${total.toLocaleString()}`}</strong>}
      </div>
      <div
        className={hasValue ? styles.progressBar : `${styles.progressBar} ${styles.progressBarIndeterminate}`}
        role="progressbar"
        aria-label={progressLabel}
        aria-valuemin={hasValue ? 0 : undefined}
        aria-valuemax={hasValue ? total : undefined}
        aria-valuenow={hasValue ? clampedCurrent : undefined}
      >
        <span style={hasValue ? {width: `${percentage}%`} : undefined} />
      </div>
    </div>
  );
}

function FeatureSelector({
  title = 'Model features',
  columns,
  selected,
  onChange,
}: {
  title?: string;
  columns: string[];
  selected: string[];
  onChange: (value: string[]) => void;
}) {
  if (columns.length === 0) {
    return null;
  }
  return (
    <div className={styles.featureSelector}>
      <span>{title}</span>
      <div>
        {columns.map((column) => (
          <label key={column}>
            <input
              type="checkbox"
              checked={selected.includes(column)}
              onChange={(event) => {
                if (event.target.checked) {
                  onChange([...selected, column]);
                } else {
                  onChange(selected.filter((item) => item !== column));
                }
              }}
            />
            {column}
          </label>
        ))}
      </div>
    </div>
  );
}

function ForecastChart({
  actualRows,
  rows,
  quantiles = [],
}: {
  actualRows: ActualRecord[];
  rows: ForecastRecord[];
  quantiles?: ForecastQuantileRecord[];
}) {
  if (rows.length === 0 && actualRows.length === 0) {
    return null;
  }
  const visibleActualRows = recentActualRowsForForecastChart(actualRows, rows.length);
  const quantileSeries = quantileForecastSeries(quantiles);
  const series = buildLineSeries(visibleActualRows, [
    {label: rows[0]?.model ?? 'forecast', records: rows},
    ...quantileSeries,
  ]);
  return <LineChart caption={rows[0]?.series_id ? `First series: ${rows[0].series_id}` : 'Forecast'} series={series} />;
}

function ForecastComponentSummary({components}: {components?: ForecastResponse['components']}) {
  const first = components?.records[0];
  if (!first) {
    return null;
  }
  const rows = summarizeComponentRecord(first);
  return (
    <section className={styles.componentPanel}>
      <div className={styles.componentHeader}>
        <div>
          <span className={styles.eyebrow}>Components</span>
          <h3>{first.series_id}</h3>
        </div>
        <p>{first.timestamp}</p>
      </div>
      <div className={styles.componentGrid}>
        <p>
          <span>Prediction</span>
          {formatCompact(first.prediction)}
        </p>
        <p>
          <span>Trend</span>
          {formatCompact(first.trend)}
        </p>
        <p>
          <span>Non-trend</span>
          {formatCompact(numberComponent(first.components.non_trend_total))}
        </p>
      </div>
      <div className={styles.tableScroller}>
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Contribution</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{formatCompact(row.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ProphetDebuggerPanel({response}: {response: ForecastResponse}) {
  const componentRows = firstSeriesComponentRows(response);
  const historyRows = firstSeriesHistoryComponentRows(response);
  if (componentRows.length === 0 && historyRows.length === 0) {
    return null;
  }
  const lastHistory = historyRows.at(-1);
  const trendMovement = lastHistory?.trend_movement ?? null;
  const seasonalTotal = lastHistory ? numberComponent(lastHistory.components.seasonal_total) : null;
  const residual = lastHistory?.residual ?? null;
  const forecastComponentSeries = forecastComponentChartSeries(componentRows);
  const historyTrendSeries = historyTrendChartSeries(historyRows);
  const historySeasonalitySeries = historySeasonalityChartSeries(historyRows);
  return (
    <section className={styles.debuggerPanel}>
      <div className={styles.componentHeader}>
        <div>
          <span className={styles.eyebrow}>Prophet-style debugger</span>
          <h3>{componentRows[0]?.series_id ?? historyRows[0]?.series_id ?? 'Forecast components'}</h3>
        </div>
        <p>{componentRows.length > 0 ? `${componentRows.length.toLocaleString()} forecast steps` : `${historyRows.length.toLocaleString()} history rows`}</p>
      </div>
      <div className={styles.debuggerMetrics}>
        <p>
          <span>Last trend move</span>
          {formatCompact(trendMovement)}
        </p>
        <p>
          <span>Last seasonality</span>
          {formatCompact(seasonalTotal)}
        </p>
        <p>
          <span>Last residual</span>
          {formatCompact(residual)}
        </p>
      </div>
      {forecastComponentSeries.length > 0 && (
        <LineChart caption="Forecast trend and seasonal components" series={forecastComponentSeries} showForecastBoundary={false} />
      )}
      {historyTrendSeries.length > 0 && (
        <LineChart caption="Historical actual, fitted, and trend" series={historyTrendSeries} showForecastBoundary={false} />
      )}
      {historySeasonalitySeries.length > 0 && (
        <LineChart caption="Historical seasonality and residual diagnostics" series={historySeasonalitySeries} showForecastBoundary={false} />
      )}
      {historyRows.length > 0 && (
        <div className={styles.tableScroller}>
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Actual</th>
                <th>Fitted</th>
                <th>Trend</th>
                <th>Trend move</th>
                <th>Seasonality</th>
                <th>Residual</th>
              </tr>
            </thead>
            <tbody>
              {historyRows.slice(-8).map((row) => (
                <tr key={`${row.series_id}-${row.timestamp}-${row.index}`}>
                  <td>{row.timestamp}</td>
                  <td>{formatCompact(row.actual)}</td>
                  <td>{formatCompact(row.fitted)}</td>
                  <td>{formatCompact(row.trend)}</td>
                  <td>{formatCompact(row.trend_movement)}</td>
                  <td>{formatCompact(numberComponent(row.components.seasonal_total))}</td>
                  <td>{formatCompact(row.residual)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function ForecastSignalSummary({
  actualRows,
  forecastRows,
  frequency,
  seasonLength,
}: {
  actualRows: ActualRecord[];
  forecastRows: ForecastRecord[];
  frequency: string;
  seasonLength: number;
}) {
  const lastActual = actualRows.at(-1)?.value;
  const firstForecast = forecastRows[0]?.prediction;
  const lastForecast = forecastRows.at(-1)?.prediction;
  const lift = lastActual === undefined || firstForecast === undefined || lastActual === 0 ? null : (firstForecast - lastActual) / Math.abs(lastActual);
  const drift = firstForecast === undefined || lastForecast === undefined || firstForecast === 0 ? null : (lastForecast - firstForecast) / Math.abs(firstForecast);
  return (
    <div className={styles.signalCards}>
      <p>
        <span>Seasonality</span>
        {seasonalityLabel(frequency, seasonLength)}
        <em>{seasonLength.toLocaleString()} steps</em>
      </p>
      <p>
        <span>First forecast</span>
        {firstForecast === undefined ? '-' : formatCompact(firstForecast)}
        <em>{lift === null ? 'no lift' : `${formatPercent(lift)} vs last actual`}</em>
      </p>
      <p>
        <span>Horizon drift</span>
        {formatPercent(drift)}
        <em>{forecastRows.length.toLocaleString()} forecast rows</em>
      </p>
    </div>
  );
}

function RegressionMetricSummary({result}: {result: RegressionResponse}) {
  return (
    <div className={styles.metricCards}>
      <p>
        <span>RMSE</span>
        {formatMetric(result.metrics.rmse)}
      </p>
      <p>
        <span>MAE</span>
        {formatMetric(result.metrics.mae)}
      </p>
      <p>
        <span>R2</span>
        {formatMetric(result.metrics.r2)}
      </p>
      <p>
        <span>Trees</span>
        {result.metadata.treeCount.toLocaleString()}
      </p>
      <p>
        <span>Mode</span>
        {result.metadata.splitterMode ?? 'auto'}
      </p>
      <p>
        <span>Loss</span>
        {result.metadata.loss ?? 'l2'}
      </p>
    </div>
  );
}

function modelVisualizerStatus(prefix: string, response: RegressionResponse, loadedRows: number) {
  const fittedRows = response.metrics.trainRows + response.metrics.holdoutRows;
  const sampleNote = loadedRows > fittedRows
    ? ` Visualizer fit used ${fittedRows.toLocaleString()} sampled rows from ${loadedRows.toLocaleString()} loaded rows.`
    : '';
  return `${prefix}: ${response.metrics.trainRows.toLocaleString()} train rows, ${response.metrics.holdoutRows.toLocaleString()} holdout rows.${sampleNote}`;
}

function CartoBoostModelVisualizer({result}: {result: RegressionResponse}) {
  const visualization = result.modelVisualization;
  if (!visualization) {
    return null;
  }
  const maxSplitCount = Math.max(...visualization.splitKinds.map((row) => row.count), 1);
  const maxRuleGain = Math.max(...visualization.splitterRules.map((row) => row.totalGain), 1);
  const maxDepthCount = Math.max(...visualization.depthHistogram.map((row) => row.count), 1);
  const residualRows = [...result.predictions]
    .sort((left, right) => Math.abs(right.residual) - Math.abs(left.residual))
    .slice(0, 10);
  const residualMax = Math.max(...residualRows.map((row) => Math.abs(row.residual)), 1);
  return (
    <section className={styles.modelVisualizer}>
      <div className={styles.modelVisualizerHeader}>
        <div>
          <span className={styles.eyebrow}>CartoBoost structure</span>
          <h3>Boosted tree visualizer</h3>
        </div>
        <dl>
          <div>
            <dt>Nodes</dt>
            <dd>{visualization.summary.nodeCount.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Leaves</dt>
            <dd>{visualization.summary.leafCount.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Depth</dt>
            <dd>{visualization.summary.maxDepth.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Gain</dt>
            <dd>{formatCompact(visualization.summary.meanGain)}</dd>
          </div>
        </dl>
      </div>
      <SplitterAtlas visualization={visualization} />
      <div className={styles.modelVisualizerGrid}>
        <TreeForestSvg trees={visualization.treeBlueprints} />
        <div className={styles.modelVisualizerSide}>
          <TreeRoster trees={visualization.treeBlueprints} />
          <div className={styles.structurePanel}>
            <strong>Split mix</strong>
            {visualization.splitKinds.map((row) => (
              <div className={styles.structureRow} key={row.kind}>
                <span>{splitKindLabel(row.kind)}</span>
                <i>
                  <em style={{width: `${Math.max((row.count / maxSplitCount) * 100, 3)}%`}} />
                </i>
                <b>{row.count.toLocaleString()}</b>
              </div>
            ))}
          </div>
          <FeatureSplitterMatrix visualization={visualization} />
          <div className={styles.structurePanel}>
            <strong>Top splitter rules</strong>
            {visualization.splitterRules.slice(0, 7).map((row) => (
              <div className={styles.splitterRuleRow} key={`${row.kind}-${row.label}`}>
                <span>
                  <b>{splitKindLabel(row.kind)}</b>
                  {row.label}
                </span>
                <i>
                  <em style={{width: `${Math.max((row.totalGain / maxRuleGain) * 100, 3)}%`}} />
                </i>
                <strong>{`${row.count.toLocaleString()} / ${formatCompact(row.meanGain)}`}</strong>
              </div>
            ))}
          </div>
          <div className={styles.structurePanel}>
            <strong>Depth profile</strong>
            <div className={styles.depthProfile}>
              {visualization.depthHistogram.map((row) => (
                <span title={`Depth ${row.depth}: ${row.count.toLocaleString()} nodes`} key={row.depth}>
                  <i style={{height: `${Math.max((row.count / maxDepthCount) * 100, 5)}%`}} />
                  <em>{row.depth}</em>
                </span>
              ))}
            </div>
          </div>
          <div className={styles.structurePanel}>
            <strong>Largest holdout residuals</strong>
            {residualRows.map((row) => (
              <div className={styles.residualRow} key={row.rowIndex}>
                <span>Row {row.rowIndex.toLocaleString()}</span>
                <i>
                  <em
                    className={row.residual >= 0 ? styles.residualPositive : styles.residualNegative}
                    style={{width: `${Math.max((Math.abs(row.residual) / residualMax) * 100, 3)}%`}}
                  />
                </i>
                <b>{formatCompact(row.residual)}</b>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function FeatureSplitterMatrix({visualization}: {visualization: ModelVisualization}) {
  if (visualization.featureSplitCounts.length === 0) {
    return null;
  }
  const topFeatures = Array.from(new Set(visualization.featureSplitCounts.map((row) => row.feature))).slice(0, 8);
  const kinds = visualization.splitKinds.map((row) => row.kind);
  const maxCount = Math.max(...visualization.featureSplitCounts.map((row) => row.count), 1);
  const countByFeatureKind = new Map(visualization.featureSplitCounts.map((row) => [`${row.feature}\u0000${row.kind}`, row]));
  const gridTemplateColumns = `minmax(7rem, 1fr) repeat(${Math.max(kinds.length, 1)}, minmax(1.65rem, 0.28fr))`;
  return (
    <div className={styles.featureSplitterMatrix}>
      <strong>Feature splitter matrix</strong>
      <div className={styles.featureSplitterHeader} style={{gridTemplateColumns}}>
        <span />
        {kinds.map((kind) => (
          <b title={splitKindLabel(kind)} key={kind}>{splitKindLabel(kind).replace(' threshold', '').replace(' spatial', '')}</b>
        ))}
      </div>
      {topFeatures.map((feature) => (
        <div className={styles.featureSplitterRow} style={{gridTemplateColumns}} key={feature}>
          <span title={feature}>{feature}</span>
          {kinds.map((kind) => {
            const row = countByFeatureKind.get(`${feature}\u0000${kind}`);
            return (
              <i title={`${feature} / ${splitKindLabel(kind)}: ${row?.count ?? 0} splits`} key={kind}>
                {row && <em style={{opacity: Math.max(row.count / maxCount, 0.18)}} />}
              </i>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function TreeRoster({trees}: {trees: TreeBlueprint[]}) {
  if (trees.length === 0) {
    return null;
  }
  const maxGain = Math.max(...trees.map((tree) => tree.totalGain), 1);
  return (
    <div className={styles.treeRoster}>
      <strong>Tree roster</strong>
      {trees.map((tree) => (
        <div className={styles.treeRosterRow} key={tree.treeIndex}>
          <span>{`Tree ${tree.treeIndex + 1}`}</span>
          <i>
            <em style={{width: `${Math.max((tree.totalGain / maxGain) * 100, 3)}%`}} />
          </i>
          <b>{formatCompact(tree.totalGain)}</b>
          <small>{`${tree.nodeCount.toLocaleString()} nodes / ${tree.leafCount.toLocaleString()} leaves / d${tree.maxDepth}`}</small>
        </div>
      ))}
    </div>
  );
}

function SplitterAtlas({visualization}: {visualization: ModelVisualization}) {
  if (visualization.splitKinds.length === 0) {
    return null;
  }
  const totalSplits = visualization.splitKinds.reduce((sum, row) => sum + row.count, 0);
  return (
    <div className={styles.splitterAtlas}>
      {visualization.splitKinds.map((row) => {
        const sampleRule = visualization.splitterRules.find((rule) => rule.kind === row.kind);
        return (
          <article className={styles.splitterCard} key={row.kind}>
            <SplitterGlyph kind={row.kind} />
            <div>
              <strong>{splitKindLabel(row.kind)}</strong>
              <span>{formatPercent(totalSplits === 0 ? 0 : row.count / totalSplits).replace(/^\+/, '')}</span>
            </div>
            <p>{sampleRule?.label ?? splitterKindHint(row.kind)}</p>
          </article>
        );
      })}
    </div>
  );
}

function SplitterGlyph({kind}: {kind: string}) {
  const color = splitKindColor(kind);
  if (kind === 'diagonal_2d') {
    return (
      <svg className={styles.splitterGlyph} viewBox="0 0 64 46" aria-hidden="true">
        <rect x="4" y="4" width="56" height="38" rx="6" />
        <line x1="10" y1="38" x2="54" y2="8" style={{stroke: color}} />
        <circle cx="20" cy="14" r="3" />
        <circle cx="44" cy="32" r="3" />
      </svg>
    );
  }
  if (kind === 'gaussian_2d') {
    return (
      <svg className={styles.splitterGlyph} viewBox="0 0 64 46" aria-hidden="true">
        <rect x="4" y="4" width="56" height="38" rx="6" />
        <circle className={styles.splitterRing} cx="32" cy="23" r="14" style={{stroke: color}} />
        <circle cx="32" cy="23" r="3" />
      </svg>
    );
  }
  if (kind === 'periodic') {
    return (
      <svg className={styles.splitterGlyph} viewBox="0 0 64 46" aria-hidden="true">
        <rect x="4" y="4" width="56" height="38" rx="6" />
        <path d="M8 26 C18 8, 28 8, 38 26 S54 44, 60 24" />
        <rect className={styles.splitterBand} x="25" y="5" width="16" height="36" style={{fill: color}} />
      </svg>
    );
  }
  if (kind === 'sparse_set' || kind === 'sparse_list') {
    return (
      <svg className={styles.splitterGlyph} viewBox="0 0 64 46" aria-hidden="true">
        <rect x="4" y="4" width="56" height="38" rx="6" />
        {[14, 25, 36, 47].map((x, index) => (
          <circle cx={x} cy={index % 2 === 0 ? 17 : 29} r="5" style={{fill: index < 2 ? color : undefined}} key={x} />
        ))}
      </svg>
    );
  }
  if (kind === 'fuzzy') {
    return (
      <svg className={styles.splitterGlyph} viewBox="0 0 64 46" aria-hidden="true">
        <rect x="4" y="4" width="56" height="38" rx="6" />
        <rect className={styles.splitterBand} x="26" y="7" width="12" height="32" style={{fill: color}} />
        <line x1="32" y1="8" x2="32" y2="38" style={{stroke: color}} />
      </svg>
    );
  }
  return (
    <svg className={styles.splitterGlyph} viewBox="0 0 64 46" aria-hidden="true">
      <rect x="4" y="4" width="56" height="38" rx="6" />
      <line x1="32" y1="8" x2="32" y2="38" style={{stroke: color}} />
      <circle cx="20" cy="18" r="3" />
      <circle cx="45" cy="29" r="3" />
    </svg>
  );
}

function TreeForestSvg({trees}: {trees: TreeBlueprint[]}) {
  const plottedTrees = trees.slice(0, 4);
  if (plottedTrees.length === 0) {
    return null;
  }
  const width = 940;
  const height = 410;
  const gutter = 24;
  const treeWidth = (width - gutter * (plottedTrees.length + 1)) / plottedTrees.length;
  const nodes = plottedTrees.flatMap((tree, treeOffset) =>
    layoutTreeNodes(tree.root, gutter + treeOffset * (treeWidth + gutter), treeWidth, tree.treeIndex),
  );
  const edges = nodes.filter((node) => node.parentKey);
  return (
    <figure className={styles.treeForest}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Boosted tree structure">
        <defs>
          <pattern id="tree-grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <path className={styles.treeGridPath} d="M 28 0 L 0 0 0 28" />
          </pattern>
        </defs>
        <rect className={styles.treeForestBackdrop} x="0" y="0" width={width} height={height} rx="10" />
        <rect className={styles.treeForestGrid} x="0" y="0" width={width} height={height} rx="10" />
        {plottedTrees.map((tree, index) => (
          <g className={styles.treeMiniSummary} transform={`translate(${gutter + index * (treeWidth + gutter)}, 16)`} key={tree.treeIndex}>
            <text x="0" y="0">{`Tree ${tree.treeIndex + 1}`}</text>
            <text x="0" y="18">{`${tree.nodeCount} nodes, gain ${formatCompact(tree.totalGain)}`}</text>
          </g>
        ))}
        {edges.map((node) => {
          const parent = nodes.find((candidate) => candidate.key === node.parentKey);
          if (!parent) {
            return null;
          }
          return (
            <line
              className={styles.treeEdge}
              x1={parent.x}
              x2={node.x}
              y1={parent.y + 14}
              y2={node.y - 16}
              key={`${parent.key}-${node.key}`}
            />
          );
        })}
        {nodes.map((node) => (
          <g className={`${styles.treeNode} ${node.kind.includes('leaf') ? styles.treeLeaf : styles.treeBranch}`} transform={`translate(${node.x}, ${node.y})`} key={node.key}>
            <title>{node.tooltip}</title>
            <rect
              x="-52"
              y="-18"
              width="104"
              height="36"
              rx="8"
              style={node.kind.includes('leaf') ? undefined : {fill: splitKindFill(node.kind), stroke: splitKindColor(node.kind)}}
            />
            <text x="0" y="-2">{truncateLabel(node.label, 20)}</text>
            <text x="0" y="12">{node.detail}</text>
          </g>
        ))}
      </svg>
      <figcaption>First {plottedTrees.length.toLocaleString()} trees, expanded from model metadata</figcaption>
    </figure>
  );
}

type PositionedTreeNode = {
  key: string;
  parentKey?: string;
  x: number;
  y: number;
  kind: string;
  label: string;
  detail: string;
  tooltip: string;
};

function layoutTreeNodes(
  root: TreeNodeBlueprint,
  xOffset: number,
  treeWidth: number,
  treeIndex: number,
): PositionedTreeNode[] {
  const positioned: PositionedTreeNode[] = [];
  const visit = (node: TreeNodeBlueprint, depth: number, slot: number, slots: number, parentKey?: string) => {
    const key = `${treeIndex}-${node.id}`;
    const x = xOffset + ((slot + 0.5) / slots) * treeWidth;
    const y = 78 + depth * 78;
    positioned.push({
      key,
      parentKey,
      x,
      y,
      kind: node.kind,
      label: node.label,
      detail: node.gain === undefined ? formatCompact(node.value) : `gain ${formatCompact(node.gain)}`,
      tooltip: treeNodeTooltip(node),
    });
    if (node.left) {
      visit(node.left, depth + 1, slot * 2, slots * 2, key);
    }
    if (node.right) {
      visit(node.right, depth + 1, slot * 2 + 1, slots * 2, key);
    }
  };
  visit(root, 0, 0, 1);
  return positioned;
}

function treeNodeTooltip(node: TreeNodeBlueprint) {
  const detail = node.gain === undefined
    ? `leaf value ${formatMetric(node.value)}`
    : `gain ${formatMetric(node.gain)}`;
  const weight = node.sampleWeightSum === undefined ? '' : `, weight ${formatCompact(node.sampleWeightSum)}`;
  return `${splitKindLabel(node.kind)}: ${node.label} (${detail}${weight})`;
}

function RegressionPredictionChart({result}: {result: RegressionResponse}) {
  const values = result.predictions.flatMap((row) =>
    [row.actual, row.prediction, row.lowerPrediction, row.upperPrediction].filter((value): value is number => typeof value === 'number'),
  );
  if (values.length === 0) {
    return null;
  }
  const width = 820;
  const height = 300;
  const padding = 38;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const xFor = (value: number) => padding + ((value - min) / span) * (width - padding * 2);
  const yFor = (value: number) => height - padding - ((value - min) / span) * (height - padding * 2);
  return (
    <figure className={styles.chart}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Actual versus predicted holdout values">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <line className={styles.referenceLine} x1={padding} y1={height - padding} x2={width - padding} y2={padding} />
        {result.predictions
          .filter((row) => typeof row.lowerPrediction === 'number' && typeof row.upperPrediction === 'number')
          .map((row) => (
            <line
              className={styles.intervalLine}
              x1={xFor(row.actual)}
              x2={xFor(row.actual)}
              y1={yFor(row.lowerPrediction as number)}
              y2={yFor(row.upperPrediction as number)}
              key={`interval-${row.rowIndex}`}
            />
          ))}
        {result.predictions.map((row) => (
          <circle cx={xFor(row.actual)} cy={yFor(row.prediction)} r="4" key={row.rowIndex} />
        ))}
      </svg>
      <figcaption>Actual versus predicted holdout values</figcaption>
    </figure>
  );
}

function FeatureImportanceChart({result}: {result: RegressionResponse}) {
  const rows = result.featureImportance.filter((item) => item.splitCount > 0).slice(0, 12);
  if (rows.length === 0) {
    return null;
  }
  const maxSplits = Math.max(...rows.map((row) => row.splitCount), 1);
  return (
    <figure className={styles.metricChart}>
      <figcaption>Split-count feature importance</figcaption>
      {rows.map((row, index) => (
        <div className={styles.metricRow} key={row.feature}>
          <span>{index + 1}. {row.feature}</span>
          <div>
            <i style={{width: `${Math.max((row.splitCount / maxSplits) * 100, 2)}%`}} />
          </div>
          <strong>{row.splitCount.toLocaleString()}</strong>
        </div>
      ))}
    </figure>
  );
}

function RegressionPredictionTable({result}: {result: RegressionResponse}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>Row</th>
            <th>Actual</th>
            <th>Prediction</th>
            <th>Lower</th>
            <th>Upper</th>
            <th>Residual</th>
          </tr>
        </thead>
        <tbody>
          {result.predictions.slice(0, 20).map((row) => (
            <tr key={row.rowIndex}>
              <td>{row.rowIndex.toLocaleString()}</td>
              <td>{formatMetric(row.actual)}</td>
              <td>{formatMetric(row.prediction)}</td>
              <td>{formatMetric(row.lowerPrediction)}</td>
              <td>{formatMetric(row.upperPrediction)}</td>
              <td>{formatMetric(row.residual)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonChart({actualRows, results}: {actualRows: ActualRecord[]; results: BenchmarkResult[]}) {
  const successfulResults = results.filter((result) => result.response);
  // Backtest responses are fitted on a truncated training frame.  Select the
  // same series that supplies the actual trace instead of relying on whatever
  // series happens to be first in each Rust response.
  const selectedSeriesId = actualRows[0]?.series_id;
  const forecastSeries = successfulResults.map((result) => ({
      label: benchmarkSeriesLabel(result),
      records: firstSeriesForecastRows(result.response as ForecastResponse, selectedSeriesId),
    }));
  const maxForecastLength = Math.max(0, ...forecastSeries.map((series) => series.records.length));
  const series = buildLineSeries(recentActualRowsForForecastChart(actualRows, maxForecastLength), forecastSeries);
  return <LineChart caption="First series comparison" series={series} />;
}

type ActiveChartPoint = {
  series: string;
  index: number;
  value: number;
  x: number;
  y: number;
};

function LineChart({caption, series, showForecastBoundary = true}: {caption: string; series: ChartSeries[]; showForecastBoundary?: boolean}) {
  const [hoverPoint, setHoverPoint] = useState<ActiveChartPoint | null>(null);
  const [pinnedPoint, setPinnedPoint] = useState<ActiveChartPoint | null>(null);
  const [zoom, setZoom] = useState(1);
  const [selectedRange, setSelectedRange] = useState<{min: number; max: number} | null>(null);
  const [brush, setBrush] = useState<{startX: number; currentX: number; pointerId: number} | null>(null);
  const drawable = series.filter((item) => item.points.length > 0);
  if (drawable.length === 0) {
    return null;
  }
  const width = 900;
  const height = 340;
  const leftPadding = 58;
  const rightPadding = 22;
  const topPadding = 24;
  const bottomPadding = 46;
  const plotWidth = width - leftPadding - rightPadding;
  const plotHeight = height - topPadding - bottomPadding;
  const indexes = drawable.flatMap((item) => item.points.map((point) => point.index));
  const minIndex = Math.min(...indexes, 0);
  const maxIndex = Math.max(...indexes, minIndex + 1);
  const indexSpan = maxIndex - minIndex || 1;
  const visibleSpan = indexSpan / zoom;
  const rangeMinIndex = selectedRange === null ? null : Math.max(minIndex, Math.min(selectedRange.min, maxIndex));
  const rangeMaxIndex = selectedRange === null ? null : Math.max(minIndex, Math.min(selectedRange.max, maxIndex));
  const hasSelectedRange = rangeMinIndex !== null && rangeMaxIndex !== null && rangeMaxIndex - rangeMinIndex > 1;
  const visibleMinIndex = hasSelectedRange ? rangeMinIndex : zoom === 1 ? minIndex : Math.max(minIndex, maxIndex - visibleSpan);
  const visibleMaxIndex = hasSelectedRange ? rangeMaxIndex : maxIndex;
  const visibleIndexSpan = visibleMaxIndex - visibleMinIndex || 1;
  const values = drawable
    .flatMap((item) =>
      item.points
        .filter((point) => point.index >= visibleMinIndex && point.index <= visibleMaxIndex)
        .map((point) => coerceFiniteNumber(point.value)),
    )
    .filter((value): value is number => value !== null);
  if (values.length === 0) {
    return null;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const actualSeries = drawable[0]?.points ?? [];
  const forecastStarts = drawable
    .slice(1)
    .map((item) => item.forecastStartIndex)
    .filter((index): index is number => index !== undefined && Number.isFinite(index));
  const actualMaxIndex = !showForecastBoundary || actualSeries.length === 0
    ? null
    : forecastStarts.length > 0
      ? Math.min(...forecastStarts) - 1
      : Math.max(...actualSeries.map((point) => point.index));
  const plotSeries = drawable.map((item, seriesIndex) => ({
    ...item,
    color: chartColor(seriesIndex),
    points: item.points
      .map((point) => {
        const value = coerceFiniteNumber(point.value);
        if (value === null) {
          return null;
        }
        return {
          ...point,
          value,
          x: leftPadding + ((point.index - visibleMinIndex) / visibleIndexSpan) * plotWidth,
          y: topPadding + plotHeight - ((value - min) / span) * plotHeight,
        };
      })
      .filter((point): point is {index: number; value: number; x: number; y: number} =>
        point !== null && point.index >= visibleMinIndex && point.index <= visibleMaxIndex,
      ),
  }));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => ({
    value: min + span * ratio,
    y: topPadding + plotHeight - ratio * plotHeight,
  }));
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const index = visibleMinIndex + visibleIndexSpan * ratio;
    return {
      index,
      x: leftPadding + ratio * plotWidth,
    };
  });
  const forecastBoundaryX = actualMaxIndex === null
    ? null
    : leftPadding + ((actualMaxIndex - visibleMinIndex) / visibleIndexSpan) * plotWidth;
  const markerLimit = zoom >= 8 ? 42 : zoom >= 4 ? 28 : zoom >= 2 ? 16 : 0;
  const pointStrideFor = (pointCount: number, limit: number) =>
    Math.max(1, Math.ceil(pointCount / limit));
  const shouldSamplePoint = (index: number, pointCount: number, limit: number) =>
    index === 0 || index === pointCount - 1 || index % pointStrideFor(pointCount, limit) === 0;
  const shouldShowMarker = (index: number, pointCount: number) =>
    pointCount <= 80 || (markerLimit > 0 && shouldSamplePoint(index, pointCount, markerLimit));
  const pointerLocation = (event: ReactPointerEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) {
      return null;
    }
    const pointerX = leftPadding + ((event.clientX - bounds.left) / bounds.width) * plotWidth;
    const pointerY = topPadding + ((event.clientY - bounds.top) / bounds.height) * plotHeight;
    const x = Math.max(leftPadding, Math.min(leftPadding + plotWidth, pointerX));
    const y = Math.max(topPadding, Math.min(topPadding + plotHeight, pointerY));
    const index = visibleMinIndex + ((x - leftPadding) / plotWidth) * visibleIndexSpan;
    return {
      x,
      y,
      index: Math.max(minIndex, Math.min(maxIndex, index)),
    };
  };
  const nearestPointFromPointer = (event: ReactPointerEvent<SVGRectElement>): ActiveChartPoint | null => {
    const pointer = pointerLocation(event);
    if (pointer === null) {
      return null;
    }
    let nearest: ActiveChartPoint | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    plotSeries.forEach((item) => {
      item.points.forEach((point) => {
        const distance = Math.abs(point.x - pointer.x) * 3 + Math.abs(point.y - pointer.y);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = {series: item.label, index: point.index, value: point.value, x: point.x, y: point.y};
        }
      });
    });
    return nearest;
  };
  const outcomeRows = plotSeries
    .filter((item) => item.points.length > 0)
    .map((item) => {
      const first = item.points[0];
      const last = item.points[item.points.length - 1];
      const change = first.value === 0 ? null : (last.value - first.value) / Math.abs(first.value);
      return {label: item.label, color: item.color, first: first.value, last: last.value, change};
    });
  const zoomOptions = [1, 2, 4, 8];
  const activePoint = pinnedPoint ?? hoverPoint;
  const brushMinX = brush === null ? null : Math.min(brush.startX, brush.currentX);
  const brushWidth = brush === null ? null : Math.abs(brush.currentX - brush.startX);
  const rangeLabel = hasSelectedRange
    ? `${Math.round(visibleMinIndex).toLocaleString()}-${Math.round(visibleMaxIndex).toLocaleString()}`
    : zoom === 1 ? 'Full range' : `Last ${formatFixed(100 / zoom, 0)}% of steps`;

  return (
    <figure className={styles.chart}>
      <div className={styles.chartToolbar}>
        <div>
          <strong>{caption}</strong>
          <span>{rangeLabel}</span>
        </div>
        <div className={styles.chartZoomControls} aria-label="Chart zoom">
          {zoomOptions.map((option) => (
            <button
              className={!hasSelectedRange && zoom === option ? styles.chartZoomActive : undefined}
              type="button"
              onClick={() => {
                setHoverPoint(null);
                setPinnedPoint(null);
                setSelectedRange(null);
                setBrush(null);
                setZoom(option);
              }}
              key={option}
            >
              {option === 1 ? 'Fit' : `${option}x`}
            </button>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={caption}>
        <defs>
          <linearGradient id="chart-surface" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="var(--lab-panel-soft)" />
            <stop offset="100%" stopColor="var(--lab-panel)" />
          </linearGradient>
        </defs>
        <rect className={styles.chartPlotArea} x={leftPadding} y={topPadding} width={plotWidth} height={plotHeight} rx="8" />
        {ticks.map((tick) => (
          <g className={styles.chartGridline} key={tick.y}>
            <line x1={leftPadding} y1={tick.y} x2={width - rightPadding} y2={tick.y} />
            <text x={leftPadding - 10} y={tick.y + 4}>
              {formatCompact(tick.value)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g className={styles.chartXTick} key={tick.x}>
            <line x1={tick.x} y1={topPadding + plotHeight} x2={tick.x} y2={topPadding + plotHeight + 5} />
            <text x={tick.x} y={height - 18}>{Math.round(tick.index).toLocaleString()}</text>
          </g>
        ))}
        <line className={styles.chartAxis} x1={leftPadding} y1={topPadding + plotHeight} x2={width - rightPadding} y2={topPadding + plotHeight} />
        <line className={styles.chartAxis} x1={leftPadding} y1={topPadding} x2={leftPadding} y2={topPadding + plotHeight} />
        {forecastBoundaryX !== null && forecastBoundaryX > leftPadding && forecastBoundaryX < width - rightPadding && (
          <g className={styles.forecastBoundary}>
            <line x1={forecastBoundaryX} y1={topPadding + 8} x2={forecastBoundaryX} y2={topPadding + plotHeight} />
            <text x={Math.min(forecastBoundaryX + 8, width - 112)} y={topPadding + 18}>forecast</text>
          </g>
        )}
        {plotSeries.map((item, seriesIndex) => {
          const points = item.points
            .map((point) => `${point.x},${point.y}`)
            .join(' ');
          const normalizedLabel = item.label.toLowerCase();
          const lineClassName = seriesIndex === 0
            ? styles.actualLine
            : normalizedLabel.includes('trend')
              ? styles.trendLine
              : styles.fittedLine;
          return (
            <polyline
              className={lineClassName}
              points={points}
              style={{stroke: item.color}}
              key={item.label}
            />
          );
        })}
        {plotSeries.flatMap((item, seriesIndex) =>
          item.points.filter((point, index, points) => shouldShowMarker(index, points.length)).map((point, index) => (
            <circle
              className={seriesIndex === 0 ? styles.actualPointMarker : styles.forecastPointMarker}
              cx={point.x}
              cy={point.y}
              r={seriesIndex === 0 ? 1.7 : 2.2}
              style={{stroke: item.color}}
              key={`marker-${item.label}-${point.index}-${index}`}
            />
          )),
        )}
        {brushMinX !== null && brushWidth !== null && brushWidth > 0 && (
          <rect
            className={styles.chartBrushSelection}
            x={brushMinX}
            y={topPadding}
            width={brushWidth}
            height={plotHeight}
          />
        )}
        <rect
          className={styles.chartHitLayer}
          x={leftPadding}
          y={topPadding}
          width={plotWidth}
          height={plotHeight}
          tabIndex={0}
          role="button"
          aria-label={`Inspect ${caption}`}
          onBlur={() => setHoverPoint(null)}
          onDoubleClick={() => {
            setHoverPoint(null);
            setPinnedPoint(null);
            setSelectedRange(null);
            setBrush(null);
            setZoom(1);
          }}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              setHoverPoint(null);
              setPinnedPoint(null);
              setSelectedRange(null);
              setBrush(null);
              setZoom(1);
            }
          }}
          onPointerDown={(event) => {
            if (event.button !== 0) {
              return;
            }
            const pointer = pointerLocation(event);
            if (pointer === null) {
              return;
            }
            event.currentTarget.setPointerCapture(event.pointerId);
            setPinnedPoint(null);
            setBrush({startX: pointer.x, currentX: pointer.x, pointerId: event.pointerId});
            setHoverPoint(nearestPointFromPointer(event));
          }}
          onPointerLeave={() => setHoverPoint(null)}
          onPointerMove={(event) => {
            if (brush !== null && brush.pointerId === event.pointerId) {
              const pointer = pointerLocation(event);
              if (pointer !== null) {
                setBrush({...brush, currentX: pointer.x});
                setHoverPoint(nearestPointFromPointer(event));
              }
              return;
            }
            if (pinnedPoint === null) {
              setHoverPoint(nearestPointFromPointer(event));
            }
          }}
          onPointerUp={(event) => {
            if (brush === null || brush.pointerId !== event.pointerId) {
              return;
            }
            const pointer = pointerLocation(event);
            event.currentTarget.releasePointerCapture(event.pointerId);
            setBrush(null);
            if (pointer === null) {
              return;
            }
            const dragWidth = Math.abs(pointer.x - brush.startX);
            if (dragWidth >= 10) {
              const startIndex = visibleMinIndex + ((brush.startX - leftPadding) / plotWidth) * visibleIndexSpan;
              const minSelected = Math.max(minIndex, Math.min(startIndex, pointer.index));
              const maxSelected = Math.min(maxIndex, Math.max(startIndex, pointer.index));
              if (maxSelected - minSelected > 1) {
                setSelectedRange({min: minSelected, max: maxSelected});
                setZoom(1);
                setHoverPoint(null);
                setPinnedPoint(null);
              }
              return;
            }
            setPinnedPoint(nearestPointFromPointer(event));
          }}
        />
        {activePoint && (
          <g className={styles.chartActivePoint}>
            <line x1={activePoint.x} y1={topPadding} x2={activePoint.x} y2={topPadding + plotHeight} />
            <circle cx={activePoint.x} cy={activePoint.y} r="4.5" />
          </g>
        )}
        {activePoint && (
          <g className={styles.chartTooltip} transform={`translate(${Math.min(activePoint.x + 12, width - 214)}, ${Math.max(activePoint.y - 50, 14)})`}>
            <rect width="196" height="54" rx="6" />
            <text x="10" y="20">{activePoint.series}</text>
            <text x="10" y="40">{`step ${activePoint.index}: ${formatMetric(activePoint.value)}`}</text>
          </g>
        )}
        <text className={styles.chartAxisLabel} x={leftPadding + plotWidth / 2} y={height - 4}>Step</text>
      </svg>
      <figcaption>{caption}</figcaption>
      <div className={styles.legend}>
        {plotSeries.map((item) => (
          <span key={item.label}>
            <i style={{background: item.color}} />
            {item.label}
          </span>
        ))}
      </div>
      <div className={styles.outcomeStrip}>
        {outcomeRows.map((row) => (
          <p key={row.label}>
            <i style={{background: row.color}} />
            <span>{row.label}</span>
            <strong>{formatMetric(row.last)}</strong>
            <em>{row.change === null ? 'flat' : `${formatPercent(row.change)} visible change`}</em>
          </p>
        ))}
      </div>
    </figure>
  );
}

function PreviewTable({columns, rows}: {columns: string[]; rows: Record<string, string>[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>{columns.slice(0, 8).map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.slice(0, 8).map((column) => <td key={column}>{row[column]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DatasetProfile({
  table,
  timestampCol,
  targetCol,
  seriesCol,
}: {
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
}) {
  const numericColumns = table.columns.filter((column) =>
    table.rows.some((row) => {
      const value = row[column]?.trim();
      return Boolean(value) && Number.isFinite(Number(value));
    }),
  );
  const seriesCount = seriesCol
    ? new Set(table.rows.map((row) => row[seriesCol]).filter(Boolean)).size
    : 1;
  const targetValues = table.rows
    .map((row) => Number(row[targetCol]))
    .filter((value) => Number.isFinite(value));
  const timestamps = table.rows
    .map((row) => row[timestampCol])
    .filter(Boolean)
    .sort((a, b) => a.localeCompare(b));
  const targetExtent = numericExtent(targetValues);
  const minTarget = targetExtent ? targetExtent.min : null;
  const maxTarget = targetExtent ? targetExtent.max : null;
  return (
    <div className={styles.datasetProfile}>
      <dl>
        <div>
          <dt>Rows</dt>
          <dd>{table.rows.length.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Columns</dt>
          <dd>{table.columns.length.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Series</dt>
          <dd>{seriesCount.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Numeric</dt>
          <dd>{numericColumns.length.toLocaleString()}</dd>
        </div>
      </dl>
      <div className={styles.profileGrid}>
        <p>
          <span>Time range</span>
          {timestamps[0] ?? '-'} to {timestamps[timestamps.length - 1] ?? '-'}
        </p>
        <p>
          <span>Target range</span>
          {minTarget === null || maxTarget === null ? '-' : `${formatMetric(minTarget)} to ${formatMetric(maxTarget)}`}
        </p>
      </div>
      <div className={styles.columnChips}>
        {table.columns.map((column) => (
          <span
            className={
              column === timestampCol || column === targetCol || column === seriesCol
                ? styles.columnChipSelected
                : undefined
            }
            key={column}
          >
            {column}
          </span>
        ))}
      </div>
    </div>
  );
}

function TargetRunPlan({
  table,
  targetCol,
  timestampCol,
  seriesCol,
  featureCols,
  sparseFeatureCols,
  modelOptions,
  onApply,
}: {
  table: ParsedTable;
  targetCol: string;
  timestampCol: string;
  seriesCol: string;
  featureCols: string[];
  sparseFeatureCols: string[];
  modelOptions: ModelOption[];
  onApply: () => void;
}) {
  const profile = buildTargetingProfile(table, targetCol, timestampCol, seriesCol, featureCols, sparseFeatureCols, modelOptions);
  return (
    <section className={styles.runPlanPanel}>
      <div className={styles.diagnosticsHeader}>
        <div>
          <span className={styles.eyebrow}>Targeting</span>
          <h3>Recommended run plan</h3>
        </div>
        <span>{profile.score >= 80 ? 'Ready' : 'Review'}</span>
      </div>
      <div className={styles.planGrid}>
        <p>
          <span>Forecast model</span>
          {profile.forecastModel}
        </p>
        <p>
          <span>Splitter</span>
          {profile.splitterMode}
        </p>
        <p>
          <span>Loss</span>
          {profile.loss}
        </p>
        <p>
          <span>Relationships</span>
          {profile.graphPath}
        </p>
      </div>
      <div className={styles.planReasons}>
        {profile.reasons.map((reason) => (
          <span key={reason}>{reason}</span>
        ))}
      </div>
      <div className={styles.readinessList}>
        {profile.readinessChecks.map((check) => (
          <span className={check.status === 'ready' ? styles.readinessReady : styles.readinessReview} key={check.label}>
            <strong>{check.label}</strong>
            {check.detail}
          </span>
        ))}
      </div>
      <button className={styles.inlineActionButton} type="button" onClick={onApply}>
        Apply run plan
      </button>
    </section>
  );
}

function FeatureRoleMatrix({
  table,
  targetCol,
  timestampCol,
  seriesCol,
  featureCols,
  sparseFeatureCols,
}: {
  table: ParsedTable;
  targetCol: string;
  timestampCol: string;
  seriesCol: string;
  featureCols: string[];
  sparseFeatureCols: string[];
}) {
  const rows = rankFeatureRoles(table, targetCol, timestampCol, seriesCol, featureCols, sparseFeatureCols).slice(0, 14);
  if (rows.length === 0) {
    return null;
  }
  const maxScore = Math.max(...rows.map((row) => row.score), 1);
  return (
    <section className={styles.featureRolePanel}>
      <div className={styles.diagnosticsHeader}>
        <div>
          <span className={styles.eyebrow}>Features</span>
          <h3>Target feature map</h3>
        </div>
        <span>{rows.length.toLocaleString()} ranked columns</span>
      </div>
      {rows.map((row) => (
        <div className={styles.featureRoleRow} key={row.column}>
          <span>{row.column}</span>
          <strong>{row.role}</strong>
          <div>
            <i style={{width: `${Math.max((row.score / maxScore) * 100, 3)}%`}} />
          </div>
          <em>{row.status}</em>
        </div>
      ))}
    </section>
  );
}

function TargetDiagnostics({
  table,
  targetCol,
  timestampCol,
  seriesCol,
}: {
  table: ParsedTable;
  targetCol: string;
  timestampCol: string;
  seriesCol: string;
}) {
  if (!table.columns.includes(targetCol)) {
    return null;
  }
  const targetValues = table.rows
    .map((row) => Number(row[targetCol]))
    .filter((value) => Number.isFinite(value));
  if (targetValues.length === 0) {
    return null;
  }
  const summary = summarizeValues(targetValues);
  const topSeries = seriesCol ? topTargetGroups(table, seriesCol, targetCol, 5) : [];
  const timeBuckets = summarizeTimeBuckets(table, timestampCol, targetCol);
  return (
    <section className={styles.diagnosticsPanel}>
      <div className={styles.diagnosticsHeader}>
        <div>
          <span className={styles.eyebrow}>Targeting</span>
          <h3>Target signal profile</h3>
        </div>
        <span>{targetValues.length.toLocaleString()} finite values</span>
      </div>
      <div className={styles.diagnosticGrid}>
        <p>
          <span>Mean</span>
          {formatCompact(summary.mean)}
        </p>
        <p>
          <span>P90 / P50</span>
          {summary.median === 0 ? '-' : `${formatFixed(summary.p90 / summary.median, 2)}x`}
        </p>
        <p>
          <span>Zero share</span>
          {formatPercent(summary.zeroShare).replace(/^\+/, '')}
        </p>
        <p>
          <span>Outlier band</span>
          {`${formatCompact(summary.p10)} to ${formatCompact(summary.p90)}`}
        </p>
      </div>
      <Histogram values={targetValues} />
      {timeBuckets.length > 0 && <TimeBucketStrip buckets={timeBuckets} />}
      {topSeries.length > 0 && (
        <div className={styles.targetList}>
          {topSeries.map((row, index) => (
            <span key={row.key}>
              <strong>{index + 1}. {row.key}</strong>
              <i>{formatCompact(row.mean)}</i>
              <em>{row.count.toLocaleString()} rows</em>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

function TargetOpportunityPanel({
  table,
  targetCol,
  timestampCol,
  seriesCol,
}: {
  table: ParsedTable;
  targetCol: string;
  timestampCol: string;
  seriesCol: string;
}) {
  const opportunities = rankTargetOpportunities(table, targetCol, timestampCol, seriesCol, 8);
  if (opportunities.length === 0) {
    return null;
  }
  const maxScore = Math.max(...opportunities.map((row) => row.score), 1);
  return (
    <section className={styles.opportunityPanel}>
      <div className={styles.diagnosticsHeader}>
        <div>
          <span className={styles.eyebrow}>Targeting</span>
          <h3>Opportunity targets</h3>
        </div>
        <span>{opportunities.length.toLocaleString()} ranked segments</span>
      </div>
      <div className={styles.opportunityTable}>
        <div className={styles.opportunityHeader}>
          <span>Segment</span>
          <span>Lift</span>
          <span>Trend</span>
          <span>Rows</span>
          <span>Action</span>
        </div>
        {opportunities.map((row) => (
          <div className={styles.opportunityRow} key={`${row.segmentType}-${row.key}`}>
            <strong>
              <span>{row.segmentType}</span>
              {row.key}
            </strong>
            <em>{formatPercent(row.lift)}</em>
            <em>{formatPercent(row.trend)}</em>
            <i>{row.count.toLocaleString()}</i>
            <p>
              {row.action}
              <span className={styles.opportunityScoreBar}>
                <span style={{width: `${Math.max((row.score / maxScore) * 100, 4)}%`}} />
              </span>
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Histogram({values}: {values: number[]}) {
  const bins = histogramBins(values, 12);
  const maxCount = Math.max(numericMax(bins.map((bin) => bin.count)) ?? 0, 1);
  return (
    <div className={styles.histogram} aria-label="Target histogram">
      {bins.map((bin) => (
        <span title={`${formatCompact(bin.start)} to ${formatCompact(bin.end)}: ${bin.count.toLocaleString()}`} key={`${bin.start}-${bin.end}`}>
          <i style={{height: `${Math.max((bin.count / maxCount) * 100, 4)}%`}} />
        </span>
      ))}
    </div>
  );
}

function TimeBucketStrip({buckets}: {buckets: {label: string; mean: number; count: number}[]}) {
  const means = buckets.map((bucket) => bucket.mean);
  const extent = numericExtent(means);
  const min = extent?.min ?? 0;
  const max = extent?.max ?? 1;
  return (
    <div className={styles.timeBuckets}>
      {buckets.map((bucket) => (
        <span
          title={`${bucket.label}: ${formatCompact(bucket.mean)} mean across ${bucket.count.toLocaleString()} rows`}
          style={{background: targetColor(bucket.mean, min, max)}}
          key={bucket.label}
        >
          {bucket.label}
        </span>
      ))}
    </div>
  );
}

function WebGlGeoMap({
  points,
  minTarget,
  maxTarget,
  hasDropoff,
  selectedRouteKey,
  onSelectRoute,
}: {
  points: GeoDemandPoint[];
  minTarget: number;
  maxTarget: number;
  hasDropoff: boolean;
  selectedRouteKey: string | null;
  onSelectRoute: (routeKey: string | null) => void;
}) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState('Loading route map');

  useEffect(() => {
    if (!mapContainerRef.current || points.length === 0) {
      return undefined;
    }
    let cleanup: (() => void) | undefined;
    let cancelled = false;
    void (async () => {
      try {
        const [{default: maplibregl}, {MapboxOverlay}, {ScatterplotLayer, ArcLayer}, {HeatmapLayer}] = await Promise.all([
          import('maplibre-gl'),
          import('@deck.gl/mapbox'),
          import('@deck.gl/layers'),
          import('@deck.gl/aggregation-layers'),
        ]);
        if (cancelled || !mapContainerRef.current) {
          return;
        }
        const routePoints = points.filter((point) => Number.isFinite(point.dropLon) && Number.isFinite(point.dropLat)).slice(0, 180);
        const allCoordinates = points.flatMap((point) =>
          Number.isFinite(point.dropLon) && Number.isFinite(point.dropLat)
            ? [
                [point.lon, point.lat],
                [point.dropLon, point.dropLat],
              ]
            : [[point.lon, point.lat]],
        ) as [number, number][];
        const center = allCoordinates.reduce(
          (sum, coordinate) => [sum[0] + coordinate[0] / allCoordinates.length, sum[1] + coordinate[1] / allCoordinates.length] as [number, number],
          [0, 0] as [number, number],
        );
        const map = new maplibregl.Map({
          attributionControl: false,
          center,
          container: mapContainerRef.current,
          cooperativeGestures: true,
          style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
          zoom: 10,
        });
        map.addControl(new maplibregl.NavigationControl({showCompass: false}), 'top-right');
        map.addControl(new maplibregl.ScaleControl({unit: 'imperial'}), 'bottom-left');
        const overlay = new MapboxOverlay({
          interleaved: false,
          layers: [
            new HeatmapLayer<GeoDemandPoint>({
              id: 'target-heatmap',
              data: points,
              getPosition: (point) => [point.lon, point.lat],
              getWeight: (point) => point.target,
              intensity: 1.1,
              radiusPixels: 42,
              threshold: 0.025,
            }),
            new ArcLayer<GeoDemandPoint>({
              id: 'route-arcs',
              data: routePoints,
              getHeight: 0.18,
              getSourceColor: (point) =>
                selectedRouteKey && point.routeKey !== selectedRouteKey
                  ? targetRgb(point.target, minTarget, maxTarget, 48)
                  : targetRgb(point.target, minTarget, maxTarget, 210),
              getSourcePosition: (point) => [point.lon, point.lat],
              getTargetColor: (point) =>
                selectedRouteKey && point.routeKey !== selectedRouteKey
                  ? targetRgb(point.target, minTarget, maxTarget, 38)
                  : targetRgb(point.target, minTarget, maxTarget, 150),
              getTargetPosition: (point) => [point.dropLon, point.dropLat],
              getWidth: (point) =>
                (point.routeKey === selectedRouteKey ? 5 : 1) +
                Math.max(0, (point.target - minTarget) / (maxTarget - minTarget || 1)) * 4,
              onClick: ({object}) => onSelectRoute(object?.routeKey ?? null),
              pickable: true,
            }),
            new ScatterplotLayer<GeoDemandPoint>({
              id: 'pickup-points',
              data: points,
              getFillColor: (point) =>
                selectedRouteKey && point.routeKey !== selectedRouteKey
                  ? targetRgb(point.target, minTarget, maxTarget, 70)
                  : targetRgb(point.target, minTarget, maxTarget, 230),
              getLineColor: [255, 255, 255, 220],
              getLineWidth: (point) => (point.routeKey === selectedRouteKey ? 3 : 1),
              getPosition: (point) => [point.lon, point.lat],
              getRadius: (point) =>
                (point.routeKey === selectedRouteKey ? 8 : 5) +
                Math.max(0, (point.target - minTarget) / (maxTarget - minTarget || 1)) * 8,
              lineWidthUnits: 'pixels',
              onClick: ({object}) => onSelectRoute(object?.routeKey ?? null),
              pickable: true,
              radiusUnits: 'pixels',
              stroked: true,
            }),
          ],
        });
        map.addControl(overlay);
        map.once('load', () => {
          if (allCoordinates.length > 1) {
            const bounds = allCoordinates.reduce((nextBounds, coordinate) => nextBounds.extend(coordinate), new maplibregl.LngLatBounds(allCoordinates[0], allCoordinates[0]));
            map.fitBounds(bounds, {duration: 0, padding: 42});
          }
          setStatus('Pickup demand and route paths');
        });
        cleanup = () => {
          overlay.finalize();
          map.remove();
        };
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : 'WebGL map unavailable');
        }
      }
    })();
    return () => {
      cancelled = true;
      cleanup?.();
    };
  }, [hasDropoff, maxTarget, minTarget, onSelectRoute, points, selectedRouteKey]);

  return (
    <div className={styles.webglMapShell}>
      <div className={styles.webglMap} ref={mapContainerRef} />
      <div className={styles.webglMapOverlay}>
        <span>{status}</span>
        <strong>{selectedRouteKey ?? (hasDropoff ? 'Pickup demand, hotspots, and dropoff paths' : 'Pickup demand and hotspots')}</strong>
      </div>
    </div>
  );
}

function GeoDatasetVisualization({table, targetCol, seriesCol}: {table: ParsedTable; targetCol: string; seriesCol: string}) {
  const [selectedRouteKey, setSelectedRouteKey] = useState<string | null>(null);
  const pickup = coordinatePair(table.columns, ['pickup', 'pu', 'origin', '']);
  const dropoff = coordinatePair(table.columns, ['dropoff', 'do', 'destination']);
  const h3Columns = table.columns.filter((column) => {
    const normalized = column.toLowerCase();
    return normalized.includes('h3') || table.rows.some((row) => isH3Like(row[column] ?? ''));
  });
  if (!pickup && h3Columns.length === 0) {
    return null;
  }
  const points: GeoDemandPoint[] = pickup
    ? table.rows
        .map((row) => ({
          lon: Number(row[pickup.lon]),
          lat: Number(row[pickup.lat]),
          target: Number(row[targetCol]),
          dropLon: dropoff ? Number(row[dropoff.lon]) : Number.NaN,
          dropLat: dropoff ? Number(row[dropoff.lat]) : Number.NaN,
          routeKey: dropoff
            ? routeKeyFromCoordinates(Number(row[pickup.lat]), Number(row[pickup.lon]), Number(row[dropoff.lat]), Number(row[dropoff.lon]))
            : undefined,
        }))
        .filter((point) => Number.isFinite(point.lon) && Number.isFinite(point.lat) && Number.isFinite(point.target))
        .slice(0, 600)
    : [];
  const routePoints = points.filter((point) => Number.isFinite(point.dropLon) && Number.isFinite(point.dropLat)).slice(0, 140);
  const hotCells = spatialBins(points, 5);
  const topRoutes = dropoff ? topRoutePairs(table, pickup, dropoff, targetCol, 6) : [];
  const selectedRoute = selectedRouteKey ? topRoutes.find((row) => row.key === selectedRouteKey) ?? null : null;
  const zoneLeaders = seriesCol ? topTargetGroups(table, seriesCol, targetCol, 6) : [];
  const allLon = points.flatMap((point) => (Number.isFinite(point.dropLon) ? [point.lon, point.dropLon] : [point.lon]));
  const allLat = points.flatMap((point) => (Number.isFinite(point.dropLat) ? [point.lat, point.dropLat] : [point.lat]));
  const minLon = allLon.length ? Math.min(...allLon) : 0;
  const maxLon = allLon.length ? Math.max(...allLon) : 1;
  const minLat = allLat.length ? Math.min(...allLat) : 0;
  const maxLat = allLat.length ? Math.max(...allLat) : 1;
  const targetValues = points.map((point) => point.target);
  const minTarget = targetValues.length ? Math.min(...targetValues) : 0;
  const maxTarget = targetValues.length ? Math.max(...targetValues) : 1;
  const width = 820;
  const height = 330;
  const padding = 28;
  const projection = geoMercator().fitExtent(
    [
      [padding, padding],
      [width - padding, height - padding],
    ],
    {
      type: 'MultiPoint',
      coordinates: points.flatMap((point) =>
        Number.isFinite(point.dropLon) && Number.isFinite(point.dropLat)
          ? [
              [point.lon, point.lat],
              [point.dropLon, point.dropLat],
            ]
          : [[point.lon, point.lat]],
      ),
    },
  );
  const geoPathForProjection = geoPath(projection);
  const graticulePath = geoPathForProjection(
    geoGraticule()
      .extent([
        [minLon, minLat],
        [maxLon, maxLat],
      ])
      .step([0.02, 0.02])(),
  );
  const projectedPoints = points
    .map((point): ProjectedGeoPoint | null => {
      const projectedPickup = projection([point.lon, point.lat]);
      const projectedDropoff =
        Number.isFinite(point.dropLon) && Number.isFinite(point.dropLat)
          ? projection([point.dropLon, point.dropLat])
          : null;
      if (!projectedPickup) {
        return null;
      }
      return {
        ...point,
        x: projectedPickup[0],
        y: projectedPickup[1],
        dropX: projectedDropoff?.[0],
        dropY: projectedDropoff?.[1],
      };
    })
    .filter((point): point is ProjectedGeoPoint => point !== null);
  const projectedRoutes = projectedPoints.filter(
    (point) => Number.isFinite(point.dropX) && Number.isFinite(point.dropY),
  ).slice(0, 140);
  const densityColor = scaleSequential(interpolateTurbo).domain([minTarget, maxTarget]);
  const contourRows =
    projectedPoints.length > 4
      ? contourDensity<ProjectedGeoPoint>()
          .x((point) => point.x)
          .y((point) => point.y)
          .weight((point) => 0.25 + Math.max(0, (point.target - minTarget) / (maxTarget - minTarget || 1)))
          .size([width, height])
          .bandwidth(24)
          .thresholds(7)(projectedPoints)
      : [];
  const h3Rows = h3Columns
    .flatMap((column) => h3Summary(table, column).map((row) => ({...row, column})))
    .filter((row) => h3CellLabel(row) !== '' && h3CellCount(row) > 0)
    .slice(0, 10);
  const h3DisplayRows =
    h3Rows.length > 0
      ? h3Rows
      : h3Columns
          .map((column) => ({
            column,
            cell: 'detected column',
            count: table.rows.filter((row) => (row[column]?.trim() ?? '') !== '').length || table.rows.length,
          }))
          .slice(0, 4);
  return (
    <section className={styles.geoPanel}>
      <div className={styles.geoHeader}>
        <div>
          <span className={styles.eyebrow}>Geography</span>
          <h3>{pickup ? 'Pickup and dropoff geography' : 'Spatial cell coverage'}</h3>
        </div>
        <span>
          {points.length > 0
            ? `${points.length.toLocaleString()} rows, ${routePoints.length.toLocaleString()} route links`
            : `${h3DisplayRows.length.toLocaleString()} H3 cells`}
        </span>
      </div>
      {points.length > 0 && (
        <div className={styles.geoStats}>
          <p>
            <span>Bounds</span>
            {`${formatMetric(minLat)}, ${formatMetric(minLon)} to ${formatMetric(maxLat)}, ${formatMetric(maxLon)}`}
          </p>
          <p>
            <span>Demand range</span>
            {`${formatCompact(minTarget)} to ${formatCompact(maxTarget)}`}
          </p>
          <p>
            <span>Map shows</span>
            {dropoff ? 'Pickup volume, high-demand areas, and pickup-to-dropoff paths' : 'Pickup volume and high-demand areas'}
          </p>
        </div>
      )}
      {points.length > 0 && (
        <WebGlGeoMap
          points={points}
          minTarget={minTarget}
          maxTarget={maxTarget}
          hasDropoff={Boolean(dropoff)}
          selectedRouteKey={selectedRouteKey}
          onSelectRoute={setSelectedRouteKey}
        />
      )}
      {points.length > 0 && (
        <svg className={styles.geoSvg} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Geographic dataset preview">
          <rect x="0" y="0" width={width} height={height} />
          <text className={styles.geoAxisTitle} x={width / 2} y={height - 9}>
            Longitude
          </text>
          <text className={styles.geoAxisTitle} x={15} y={height / 2} transform={`rotate(-90 15 ${height / 2})`}>
            Latitude
          </text>
          <text className={styles.geoAxisTick} x={padding} y={height - 30}>
            {formatMetric(minLon)}
          </text>
          <text className={styles.geoAxisTick} x={width - padding} y={height - 30} textAnchor="end">
            {formatMetric(maxLon)}
          </text>
          <text className={styles.geoAxisTick} x={34} y={height - padding}>
            {formatMetric(minLat)}
          </text>
          <text className={styles.geoAxisTick} x={34} y={padding + 5}>
            {formatMetric(maxLat)}
          </text>
          {graticulePath && <path className={styles.graticuleLine} d={graticulePath} />}
          {contourRows.map((contour, index) => (
            <path
              className={styles.densityContour}
              d={geoPath()(contour) ?? undefined}
              style={{fill: densityColor(minTarget + ((maxTarget - minTarget || 1) * (index + 1)) / contourRows.length)}}
              key={`contour-${index}`}
            />
          ))}
          {hotCells.map((cell) => (
            <path
              className={styles.hotCell}
              d={projectedCellPath(cell, projection) ?? undefined}
              style={{fill: targetColor(cell.mean, minTarget, maxTarget)}}
              key={`${cell.x}-${cell.y}`}
            />
          ))}
          {projectedRoutes.map((point, index) => (
            <line
              className={point.routeKey === selectedRouteKey ? styles.selectedGeoRoute : undefined}
              x1={point.x}
              y1={point.y}
              x2={point.dropX}
              y2={point.dropY}
              onClick={() => setSelectedRouteKey(point.routeKey ?? null)}
              key={`route-${index}`}
            />
          ))}
          {projectedPoints.map((point, index) => (
            <circle
              className={point.routeKey === selectedRouteKey ? styles.selectedGeoPoint : undefined}
              cx={point.x}
              cy={point.y}
              r="4"
              style={{fill: targetColor(point.target, minTarget, maxTarget)}}
              onClick={() => setSelectedRouteKey(point.routeKey ?? null)}
              key={`point-${index}`}
            />
          ))}
          <g className={styles.geoLegend} transform={`translate(${width - 196} 18)`}>
            <rect x="0" y="0" width="170" height="46" rx="6" />
            <text x="10" y="16">Demand</text>
            {[0, 1, 2, 3, 4].map((step) => (
              <path
                d={`M${10 + step * 28} 25H${38 + step * 28}V35H${10 + step * 28}Z`}
                fill={targetColor(minTarget + ((maxTarget - minTarget || 1) * step) / 4, minTarget, maxTarget)}
                key={step}
              />
            ))}
            <text x="10" y="43">{formatCompact(minTarget)}</text>
            <text x="150" y="43" textAnchor="end">{formatCompact(maxTarget)}</text>
          </g>
        </svg>
      )}
      {topRoutes.length > 0 && (
        <section className={styles.geoListSection} aria-label="Top pickup to dropoff routes">
          <div className={styles.geoListHeader}>
            <strong>Top pickup to dropoff routes</strong>
            {selectedRoute && (
              <span>
                Selected mean {formatCompact(selectedRoute.mean)} from {selectedRoute.count.toLocaleString()} rows
              </span>
            )}
          </div>
          <div className={styles.routeList}>
            {topRoutes.map((row, index) => (
              <button
                className={row.key === selectedRouteKey ? styles.routeButtonActive : undefined}
                type="button"
                onClick={() => setSelectedRouteKey((current) => (current === row.key ? null : row.key))}
                key={row.key}
              >
                <strong>{index + 1}. {row.key}</strong>
                <i>{formatCompact(row.mean)}</i>
                <em>{row.count.toLocaleString()} rows</em>
              </button>
            ))}
          </div>
        </section>
      )}
      {zoneLeaders.length > 0 && (
        <section className={styles.geoListSection} aria-label="Highest demand zones or series">
          <div className={styles.geoListHeader}>
            <strong>Highest demand zones or series</strong>
            <span>Ranked by mean {targetCol}</span>
          </div>
          <div className={styles.routeList}>
            {zoneLeaders.map((row, index) => (
              <span key={row.key}>
                <strong>{index + 1}. {row.key}</strong>
                <i>{formatCompact(row.mean)}</i>
                <em>{row.count.toLocaleString()} rows</em>
              </span>
            ))}
          </div>
        </section>
      )}
      {h3DisplayRows.length > 0 && (
        <section className={styles.geoListSection} aria-label="Detected H3 cells">
          <div className={styles.geoListHeader}>
            <strong>Detected H3 cells</strong>
            <span>Column and row frequency</span>
          </div>
          <div className={styles.h3List}>
            {h3DisplayRows.map((row) => (
              <span key={`${row.column}-${h3CellLabel(row)}`}>
                <strong>{row.column}</strong>
                {h3CellLabel(row)}
                <i>{h3CellCount(row).toLocaleString()}</i>
              </span>
            ))}
          </div>
        </section>
      )}
    </section>
  );
}

function ForecastTable({records}: {records: ForecastRecord[]}) {
  const hasSpatialDetails = records.some(
    (record) =>
      typeof record.base_mean === 'number' ||
      typeof record.spatial_correction === 'number' ||
      typeof record.kriging_variance === 'number' ||
      Array.isArray(record.selected_neighbors),
  );
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>series_id</th>
            <th>timestamp</th>
            <th>horizon</th>
            <th>prediction</th>
            {hasSpatialDetails && <th>base_mean</th>}
            {hasSpatialDetails && <th>spatial_correction</th>}
            {hasSpatialDetails && <th>kriging_variance</th>}
            {hasSpatialDetails && <th>neighbors</th>}
          </tr>
        </thead>
        <tbody>
          {records.map((record) => (
            <tr key={`${record.series_id}-${record.timestamp}-${record.horizon}`}>
              <td>{record.series_id}</td>
              <td>{record.timestamp}</td>
              <td>{record.horizon}</td>
              <td>{formatMetric(record.prediction)}</td>
              {hasSpatialDetails && <td>{typeof record.base_mean === 'number' ? formatMetric(record.base_mean) : '-'}</td>}
              {hasSpatialDetails && <td>{typeof record.spatial_correction === 'number' ? formatMetric(record.spatial_correction) : '-'}</td>}
              {hasSpatialDetails && <td>{typeof record.kriging_variance === 'number' ? formatMetric(record.kriging_variance) : '-'}</td>}
              {hasSpatialDetails && <td>{Array.isArray(record.selected_neighbors) ? record.selected_neighbors.length : '-'}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function forecastWarningLabel(response?: ForecastResponse) {
  const warning = response?.metadata.warning;
  if (!warning) {
    return null;
  }
  const fallback = warning.fallbackModel ?? response?.metadata.model ?? 'fallback';
  const reason = warning.reason ? `: ${warning.reason}` : '';
  return `fallback to ${fallback}${reason}`;
}

function benchmarkRunId() {
  return `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function benchmarkRunLabel(existingResults: Pick<ComparisonResult, 'runId'>[]) {
  const existingRunIds = new Set(existingResults.map((result) => result.runId).filter(Boolean));
  return `Benchmark ${existingRunIds.size + 1}`;
}

function benchmarkSeriesKey(result: Pick<ComparisonResult, 'runId' | 'requestedModel'>) {
  return `${result.runId ?? 'run'}::${result.requestedModel}`;
}

function benchmarkSeriesLabel(result: Pick<ComparisonResult, 'runLabel' | 'label'>) {
  return result.runLabel ? `${result.runLabel} / ${result.label}` : result.label;
}

function BenchmarkModelToggles({
  results,
  hidden,
  onToggle,
}: {
  results: BenchmarkResult[];
  hidden: string[];
  onToggle: (key: string) => void;
}) {
  const successful = results.filter((result) => result.response || (result as BacktestResult).rmse !== undefined);
  if (successful.length === 0) {
    return null;
  }
  return (
    <div className={styles.benchmarkToggles} aria-label="Benchmark model visibility">
      {successful.map((result) => {
        const key = benchmarkSeriesKey(result);
        return (
          <label key={key}>
            <input
              type="checkbox"
              checked={!hidden.includes(key)}
              onChange={() => onToggle(key)}
            />
            <span>{benchmarkSeriesLabel(result)}</span>
          </label>
        );
      })}
    </div>
  );
}

function ComparisonTable({results}: {results: ComparisonResult[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Pipeline</th>
            <th>Rows</th>
            <th>First prediction</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result) => {
            const first = result.response?.forecast.records[0];
            return (
              <tr key={benchmarkSeriesKey(result)}>
                <td>{benchmarkSeriesLabel(result)}</td>
                <td>{result.pipeline}</td>
                <td>{result.response ? result.response.forecast.records.length.toLocaleString() : '-'}</td>
                <td>{first ? formatMetric(first.prediction) : '-'}</td>
                <td>{result.error ?? forecastWarningLabel(result.response) ?? 'ok'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BacktestMetricChart({results}: {results: BacktestResult[]}) {
  const scored = [...results]
    .filter((result) => result.rmse !== undefined)
    .sort((a, b) => (a.rmse as number) - (b.rmse as number))
    .slice(0, 12);
  if (scored.length === 0) {
    return null;
  }
  const maxRmse = Math.max(...scored.map((result) => result.rmse as number), 1);
  return (
    <figure className={styles.metricChart}>
      <figcaption>RMSE by model, lower is better</figcaption>
      {scored.map((result, index) => (
        <div className={styles.metricRow} key={benchmarkSeriesKey(result)}>
          <span>{index + 1}. {benchmarkSeriesLabel(result)}</span>
          <div>
            <i style={{width: `${Math.max(((result.rmse as number) / maxRmse) * 100, 2)}%`}} />
          </div>
          <strong>{formatMetric(result.rmse)}</strong>
        </div>
      ))}
    </figure>
  );
}

function BacktestTable({results}: {results: BacktestResult[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Pipeline</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>WAPE</th>
            <th>Rows</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {[...results].sort(sortBacktestRows).map((result) => (
            <tr key={benchmarkSeriesKey(result)}>
              <td>{benchmarkSeriesLabel(result)}</td>
              <td>{result.pipeline}</td>
              <td>{formatMetric(result.rmse)}</td>
              <td>{formatMetric(result.mae)}</td>
              <td>{result.wape === undefined ? '-' : formatPercent(result.wape, 2).replace(/^\+/, '')}</td>
              <td>{result.comparedRows?.toLocaleString() ?? '-'}</td>
                <td>{result.error ?? forecastWarningLabel(result.response) ?? 'ok'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function sortBacktestRows(a: BacktestResult, b: BacktestResult) {
  if (a.rmse === undefined && b.rmse === undefined) {
    return a.label.localeCompare(b.label);
  }
  if (a.rmse === undefined) {
    return 1;
  }
  if (b.rmse === undefined) {
    return -1;
  }
  return a.rmse - b.rmse;
}

async function loadDocsForecastExampleTable(sample: 'lane' | 'spatial', laneUrl: string, variedRouteUrl: string): Promise<ParsedTable> {
  const isSpatial = sample === 'spatial';
  const response = await fetch(isSpatial ? variedRouteUrl : laneUrl);
  if (!response.ok) {
    throw new Error(`Unable to load bundled taxi forecast sample (${response.status}).`);
  }
  return buildTaxiRouteHourSampleTable(
    await parseParquetBuffer(
      await response.arrayBuffer(),
      isSpatial ? 'yellow_taxi_2024-01-varied-routes-2500.parquet' : 'yellow_taxi_2024-01-single-lane-5000.parquet',
    ),
    isSpatial ? TAXI_VARIED_ROUTE_SAMPLE_ROWS : TAXI_LANE_SAMPLE_ROWS,
    {
      balancedManhattanRoutes: isSpatial,
      fileName: isSpatial ? 'yellow_taxi_2024-01-varied-routes-2500.parquet' : 'yellow_taxi_2024-01-single-lane-5000.parquet',
    },
  );
}

function embeddedForecastExampleTable(sample: 'lane' | 'spatial', frequency: 'daily' | 'hourly'): ParsedTable {
  const spatialSeries =
    sample === 'spatial'
      ? [
          {series_id: 'north-core', base: 42, longitude: -73.991, latitude: 40.748, queue: 1.0, slope: 0.035},
          {series_id: 'waterfront', base: 34, longitude: -73.966, latitude: 40.769, queue: 0.6, slope: 0.018},
          {series_id: 'airport-loop', base: 29, longitude: -73.986, latitude: 40.728, queue: 0.2, slope: 0.028},
        ]
      : [{series_id: 'primary-route', base: 38, longitude: -73.787, latitude: 40.647, queue: 1.2, slope: 0.018}];
  const rows: ParsedTable['rows'] = [];
  const start = Date.UTC(2024, 0, 1, 0, 0, 0);
  const rowCount = frequency === 'daily' ? 140 : 336;
  const stepMs = frequency === 'daily' ? 24 * 60 * 60 * 1000 : 60 * 60 * 1000;
  for (let index = 0; index < rowCount; index += 1) {
    const timestamp = new Date(start + index * stepMs).toISOString().slice(0, 19);
    const hour = frequency === 'daily' ? 12 : index % 24;
    const dayOfWeek = frequency === 'daily' ? index % 7 : Math.floor(index / 24) % 7;
    const weekIndex = frequency === 'daily' ? Math.floor(index / 7) : Math.floor(index / (24 * 7));
    const commute = frequency === 'hourly' && hour >= 7 && hour <= 9 ? 6.5 : frequency === 'hourly' && hour >= 16 && hour <= 18 ? 5.2 : 0;
    const midday = frequency === 'hourly' && hour >= 11 && hour <= 14 ? 2.4 : 0;
    const overnight = frequency === 'hourly' && hour <= 5 ? -7.0 : 0;
    const weekend = dayOfWeek === 5 || dayOfWeek === 6 ? (frequency === 'daily' ? -4.8 : -3.2) : 0;
    const dailyWave = frequency === 'hourly' ? Math.sin(((hour - 6) / 24) * Math.PI * 2) * 3.2 : 0;
    const weeklyWave = Math.sin(((dayOfWeek + 1) / 7) * Math.PI * 2) * (frequency === 'daily' ? 5.5 : 2.2);
    const eventLift = weekIndex === 5 || weekIndex === 12 ? 7.5 : weekIndex === 9 ? -4.0 : 0;
    for (const series of spatialSeries) {
      const trend = index * series.slope;
      const routeCycle = series.queue * Math.cos(index / (frequency === 'daily' ? 5 : 18)) * 2.4;
      const target =
        series.base +
        commute +
        midday +
        overnight +
        weekend +
        dailyWave +
        weeklyWave +
        eventLift * series.queue +
        trend +
        routeCycle;
      rows.push({
        timestamp,
        series_id: series.series_id,
        target: target.toFixed(3),
        hour: String(hour),
        day_of_week: String(dayOfWeek),
        week_index: String(weekIndex),
        is_weekend: weekend === 0 ? '0' : '1',
        event_pressure: (eventLift * series.queue).toFixed(3),
        longitude: String(series.longitude),
        latitude: String(series.latitude),
        airport_queue_pressure: series.queue.toFixed(2),
      });
    }
  }
  return {
    fileName: sample === 'spatial' ? 'embedded-spatial-demand-panel.csv' : 'embedded-route-demand.csv',
    columns: [
      'timestamp',
      'series_id',
      'target',
      'hour',
      'day_of_week',
      'week_index',
      'is_weekend',
      'event_pressure',
      'longitude',
      'latitude',
      'airport_queue_pressure',
    ],
    rows,
  };
}

function formatMetric(value: unknown) {
  return formatFixed(value, 3);
}

function buildTargetingProfile(
  table: ParsedTable,
  targetCol: string,
  timestampCol: string,
  seriesCol: string,
  featureCols: string[],
  sparseFeatureCols: string[],
  modelOptions: ModelOption[],
) {
  const targetValues = table.rows
    .map((row) => Number(row[targetCol]))
    .filter((value) => Number.isFinite(value));
  const summary = targetValues.length > 0 ? summarizeValues(targetValues) : null;
  const roles = rankFeatureRoles(table, targetCol, timestampCol, seriesCol, featureCols, sparseFeatureCols);
  const hasSpatial = roles.some((role) => role.role === 'spatial' && role.status !== 'excluded');
  const hasPeriodic = roles.some((role) => role.role === 'periodic' && role.status !== 'excluded');
  const hasSparse = sparseFeatureCols.length > 0;
  const hasGraph = graphColumnReadiness(table.columns);
  const intermittent = summary ? summary.zeroShare > 0.18 : false;
  const skewed = summary ? summary.p90 > summary.median * 1.65 : false;
  const forecastModelValue =
    intermittent && modelOptions.some((option) => option.value === 'intermittent_demand')
      ? 'intermittent_demand'
      : modelOptions.some((option) => option.value === 'auto_forecast')
        ? 'auto_forecast'
      : modelOptions.some((option) => option.value === 'cartoboost_lag')
        ? 'cartoboost_lag'
        : modelOptions[0]?.value ?? 'auto_forecast';
  const forecastModel = modelOptions.find((option) => option.value === forecastModelValue)?.label ?? forecastModelValue;
  const splitterModeValue = hasSpatial && hasPeriodic ? 'full' : hasSpatial ? 'spatial' : hasPeriodic ? 'periodic' : 'auto';
  const splitterMode = {
    full: 'Spatial + periodic toolkit',
    spatial: 'Spatial splitters',
    periodic: 'Periodic splitters',
    auto: 'Auto dense',
  }[splitterModeValue];
  const lossValue = skewed ? 'huber' : summary && summary.zeroShare > 0.05 ? 'l1' : 'l2';
  const loss = {
    huber: 'Huber robust',
    l1: 'L1 median',
    l2: 'L2 mean',
  }[lossValue];
  const graphPath = hasGraph ? 'Pickup/dropoff columns detected' : hasSparse ? 'Sparse features selected' : seriesCol ? 'Series ID selected' : 'Dense features only';
  const recommendedFeatureCols = roles
    .filter((role) => role.role !== 'sparse set' && role.status !== 'review')
    .slice(0, 12)
    .map((role) => role.column);
  const recommendedSparseFeatureCols = roles
    .filter((role) => role.role === 'sparse set')
    .slice(0, 4)
    .map((role) => role.column);
  const graphSourceCol = guessColumn(table.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) ?? '';
  const graphTargetCol = guessColumn(table.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']) ?? '';
  const graphWeightCol = guessColumn(table.columns, ['edge_weight', 'weight', 'trip_count']) ?? '';
  const neuralIdCol = guessColumn(table.columns, ['pickup_zone_id', 'PULocationID', 'zone_id', 'id']) ?? '';
  const neuralPipeline = 'embedding';
  const reasons = [
    `${featureCols.length.toLocaleString()} dense features selected`,
    `${sparseFeatureCols.length.toLocaleString()} sparse sets selected`,
    hasSpatial ? 'spatial coordinates detected' : 'no coordinate pair selected',
    hasPeriodic ? 'calendar phase detected' : 'no periodic feature selected',
    hasSpatial ? 'spatial targeting applied to features and splitters' : 'dense forecast path preferred',
    skewed ? 'skew-resistant loss preferred' : 'mean target loss is suitable',
  ];
  const readinessChecks = [
    {
      label: 'Target',
      detail: summary ? `${targetValues.length.toLocaleString()} finite rows, ${formatPercent(summary.zeroShare).replace(/^\+/, '')} zero share` : 'no numeric target values',
      status: summary ? 'ready' : 'review',
    },
    {
      label: 'Time',
      detail: timestampCol && table.columns.includes(timestampCol) ? `${timestampCol} selected` : 'choose a timestamp column',
      status: timestampCol && table.columns.includes(timestampCol) ? 'ready' : 'review',
    },
    {
      label: 'Features',
      detail: `${recommendedFeatureCols.length.toLocaleString()} dense, ${recommendedSparseFeatureCols.length.toLocaleString()} sparse recommended`,
      status: recommendedFeatureCols.length > 0 || recommendedSparseFeatureCols.length > 0 ? 'ready' : 'review',
    },
    {
      label: 'Spatial',
      detail: hasSpatial ? 'coordinate features available for maps and spatial splitters' : 'no coordinate pair found in selected features',
      status: hasSpatial ? 'ready' : 'review',
    },
    {
      label: 'Graph',
      detail: hasGraph ? `${graphSourceCol} to ${graphTargetCol}` : 'source and target node columns not both detected',
      status: hasGraph ? 'ready' : 'review',
    },
  ];
  const score = Math.min(100, 35 + featureCols.length * 4 + sparseFeatureCols.length * 8 + (hasSpatial ? 18 : 0) + (hasPeriodic ? 10 : 0) + (hasGraph ? 12 : 0));
  return {
    forecastModel,
    forecastModelValue,
    splitterMode,
    splitterModeValue,
    loss,
    lossValue,
    graphPath,
    reasons,
    readinessChecks,
    score,
    featureCols: recommendedFeatureCols,
    sparseFeatureCols: recommendedSparseFeatureCols,
    graphSourceCol,
    graphTargetCol,
    graphWeightCol,
    neuralIdCol,
    neuralPipeline,
  };
}

function rankFeatureRoles(
  table: ParsedTable,
  targetCol: string,
  timestampCol: string,
  seriesCol: string,
  featureCols: string[],
  sparseFeatureCols: string[],
) {
  const targetValues = table.rows.map((row) => Number(row[targetCol]));
  return table.columns
    .filter((column) => column !== targetCol && column !== timestampCol && column !== seriesCol)
    .map((column) => {
      const role = sparseFeatureCols.includes(column) ? 'sparse set' : inferFeatureKind(column);
      const selected = featureCols.includes(column) || sparseFeatureCols.includes(column);
      const numericValues = table.rows.map((row) => Number(row[column]));
      const finitePairs = numericValues
        .map((value, index) => ({value, target: targetValues[index]}))
        .filter((row) => Number.isFinite(row.value) && Number.isFinite(row.target));
      const signal = finitePairs.length >= 3 ? Math.abs(pearsonCorrelation(finitePairs.map((row) => row.value), finitePairs.map((row) => row.target))) : 0;
      const coverage = table.rows.length === 0 ? 0 : table.rows.filter((row) => (row[column]?.trim() ?? '') !== '').length / table.rows.length;
      const roleBoost = role === 'spatial' ? 0.24 : role === 'periodic' ? 0.18 : role === 'sparse set' ? 0.2 : 0.08;
      const leakagePenalty = looksLikeLeakageColumn(column, targetCol) ? 0.35 : 0;
      const score = Math.max(0, signal * 0.55 + coverage * 0.25 + roleBoost + (selected ? 0.18 : 0) - leakagePenalty);
      return {
        column,
        role,
        score,
        status: looksLikeLeakageColumn(column, targetCol) ? 'review' : selected ? 'selected' : coverage > 0.8 ? 'available' : 'sparse',
      };
    })
    .sort((left, right) => right.score - left.score || left.column.localeCompare(right.column));
}

function pearsonCorrelation(left: number[], right: number[]) {
  const count = Math.min(left.length, right.length);
  if (count === 0) {
    return 0;
  }
  const leftMean = left.slice(0, count).reduce((sum, value) => sum + value, 0) / count;
  const rightMean = right.slice(0, count).reduce((sum, value) => sum + value, 0) / count;
  let numerator = 0;
  let leftDenominator = 0;
  let rightDenominator = 0;
  for (let index = 0; index < count; index += 1) {
    const leftDelta = left[index] - leftMean;
    const rightDelta = right[index] - rightMean;
    numerator += leftDelta * rightDelta;
    leftDenominator += leftDelta ** 2;
    rightDenominator += rightDelta ** 2;
  }
  const denominator = Math.sqrt(leftDenominator * rightDenominator);
  return denominator === 0 ? 0 : numerator / denominator;
}

function graphColumnReadiness(columns: string[]) {
  return Boolean(
    guessColumn(columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) &&
      guessColumn(columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']),
  );
}

function looksLikeLeakageColumn(column: string, targetCol: string) {
  const normalized = column.toLowerCase();
  const target = targetCol.toLowerCase();
  return normalized !== target && (normalized.includes(`${target}_future`) || normalized.includes('future_target') || normalized.includes('label'));
}

function summarizeValues(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((total, value) => total + value, 0);
  return {
    mean: sum / sorted.length,
    median: percentile(sorted, 0.5),
    p10: percentile(sorted, 0.1),
    p90: percentile(sorted, 0.9),
    zeroShare: sorted.filter((value) => value === 0).length / sorted.length,
  };
}

function percentile(sortedValues: number[], percentileRank: number) {
  if (sortedValues.length === 0) {
    return 0;
  }
  const index = Math.max(0, Math.min(sortedValues.length - 1, Math.round((sortedValues.length - 1) * percentileRank)));
  return sortedValues[index];
}

function numericExtent(values: number[]) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return min === Infinity ? null : {min, max};
}

function numericMax(values: number[]) {
  let max = -Infinity;
  for (const value of values) {
    if (Number.isFinite(value)) {
      max = Math.max(max, value);
    }
  }
  return max === -Infinity ? null : max;
}

function histogramBins(values: number[], count: number) {
  const extent = numericExtent(values);
  const min = extent?.min ?? 0;
  const max = extent?.max ?? 1;
  const span = max - min || 1;
  const bins = Array.from({length: count}, (_, index) => ({
    start: min + (span * index) / count,
    end: min + (span * (index + 1)) / count,
    count: 0,
  }));
  for (const value of values) {
    const index = Math.min(count - 1, Math.max(0, Math.floor(((value - min) / span) * count)));
    bins[index].count += 1;
  }
  return bins;
}

function summarizeTimeBuckets(table: ParsedTable, timestampCol: string, targetCol: string) {
  if (!table.columns.includes(timestampCol)) {
    return [];
  }
  const buckets = new Map<string, {sum: number; count: number}>();
  for (const row of table.rows) {
    const timestamp = row[timestampCol] ?? '';
    const target = Number(row[targetCol]);
    if (!Number.isFinite(target)) {
      continue;
    }
    const parsed = new Date(timestamp);
    if (Number.isNaN(parsed.getTime())) {
      continue;
    }
    const label = timestamp.includes('T') || timestamp.includes(':') ? `${parsed.getUTCHours().toString().padStart(2, '0')}:00` : parsed.toUTCString().slice(0, 3);
    const bucket = buckets.get(label) ?? {sum: 0, count: 0};
    bucket.sum += target;
    bucket.count += 1;
    buckets.set(label, bucket);
  }
  return Array.from(buckets.entries())
    .map((entry) => {
      const label = entry[0];
      const bucket = entry[1];
      return {label, mean: bucket.sum / bucket.count, count: bucket.count};
    })
    .sort((a, b) => a.label.localeCompare(b.label))
    .slice(0, 24);
}

function topTargetGroups(table: ParsedTable, groupCol: string, targetCol: string, limit: number) {
  const groups = new Map<string, {sum: number; count: number}>();
  for (const row of table.rows) {
    const key = row[groupCol]?.trim();
    const target = Number(row[targetCol]);
    if (!key || !Number.isFinite(target)) {
      continue;
    }
    const group = groups.get(key) ?? {sum: 0, count: 0};
    group.sum += target;
    group.count += 1;
    groups.set(key, group);
  }
  return Array.from(groups.entries())
    .map((entry) => {
      const key = entry[0];
      const group = entry[1];
      return {key, mean: group.sum / group.count, count: group.count};
    })
    .sort((left, right) => right.mean - left.mean || right.count - left.count || left.key.localeCompare(right.key))
    .slice(0, limit);
}

function topRoutePairs(
  table: ParsedTable,
  pickup: {lat: string; lon: string},
  dropoff: {lat: string; lon: string},
  targetCol: string,
  limit: number,
) {
  const groups = new Map<string, {sum: number; count: number}>();
  for (const row of table.rows) {
    const target = Number(row[targetCol]);
    const key = routeKeyFromCoordinates(
      Number(row[pickup.lat]),
      Number(row[pickup.lon]),
      Number(row[dropoff.lat]),
      Number(row[dropoff.lon]),
    );
    if (!Number.isFinite(target) || !key) {
      continue;
    }
    const group = groups.get(key) ?? {sum: 0, count: 0};
    group.sum += target;
    group.count += 1;
    groups.set(key, group);
  }
  return Array.from(groups.entries())
    .map((entry) => {
      const key = entry[0];
      const group = entry[1];
      return {key, mean: group.sum / group.count, count: group.count};
    })
    .sort((left, right) => right.mean - left.mean || right.count - left.count || left.key.localeCompare(right.key))
    .slice(0, limit);
}

function routeKeyFromCoordinates(pickupLat: number, pickupLon: number, dropoffLat: number, dropoffLon: number) {
  if (![pickupLat, pickupLon, dropoffLat, dropoffLon].every(Number.isFinite)) {
    return undefined;
  }
  return `${pickupLat.toFixed(3)},${pickupLon.toFixed(3)} -> ${dropoffLat.toFixed(3)},${dropoffLon.toFixed(3)}`;
}

function rankTargetOpportunities(
  table: ParsedTable,
  targetCol: string,
  timestampCol: string,
  seriesCol: string,
  limit: number,
) {
  const globalTargets = table.rows.map((row) => Number(row[targetCol])).filter((value) => Number.isFinite(value));
  if (globalTargets.length === 0) {
    return [];
  }
  const globalMean = globalTargets.reduce((sum, value) => sum + value, 0) / globalTargets.length;
  const segmentSpecs = opportunitySegmentSpecs(table, seriesCol);
  const seen = new Set<string>();
  return segmentSpecs
    .flatMap((spec) => scoreOpportunitySegments(table, targetCol, timestampCol, spec, globalMean, globalTargets.length))
    .filter((row) => {
      const key = `${row.segmentType}:${row.key}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    })
    .sort((left, right) => right.score - left.score || right.mean - left.mean || left.key.localeCompare(right.key))
    .slice(0, limit);
}

function opportunitySegmentSpecs(table: ParsedTable, seriesCol: string) {
  const sourceCol = guessColumn(table.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']);
  const targetCol = guessColumn(table.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']);
  const h3Col = table.columns.find((column) => {
    const normalized = column.toLowerCase();
    return normalized.includes('h3') || table.rows.some((row) => isH3Like(row[column] ?? ''));
  });
  const specs: {segmentType: string; columns: string[]}[] = [];
  if (sourceCol && targetCol) {
    specs.push({segmentType: 'Route', columns: [sourceCol, targetCol]});
  }
  if (h3Col) {
    specs.push({segmentType: 'H3 cell', columns: [h3Col]});
  }
  if (seriesCol && table.columns.includes(seriesCol)) {
    specs.push({segmentType: 'Series', columns: [seriesCol]});
  }
  return specs;
}

function scoreOpportunitySegments(
  table: ParsedTable,
  targetCol: string,
  timestampCol: string,
  spec: {segmentType: string; columns: string[]},
  globalMean: number,
  totalRows: number,
) {
  const groups = new Map<string, {values: {target: number; timestamp: string; rowIndex: number}[]; label: string}>();
  table.rows.forEach((row, rowIndex) => {
    const target = Number(row[targetCol]);
    if (!Number.isFinite(target)) {
      return;
    }
    const keys = segmentKeysForRow(row, spec.columns);
    for (const key of keys) {
      const group = groups.get(key) ?? {values: [], label: key};
      group.values.push({target, timestamp: row[timestampCol] ?? '', rowIndex});
      groups.set(key, group);
    }
  });
  return Array.from(groups.entries())
    .map(([key, group]) => {
      const targets = group.values.map((value) => value.target);
      const mean = targets.reduce((sum, value) => sum + value, 0) / targets.length;
      const lift = globalMean === 0 ? 0 : (mean - globalMean) / Math.abs(globalMean);
      const share = targets.length / Math.max(totalRows, 1);
      const trend = segmentTrend(group.values);
      const volatility = coefficientOfVariation(targets);
      const score = Math.max(0, lift * 0.52 + share * 0.24 + Math.max(trend, 0) * 0.18 + Math.min(volatility, 1.5) * 0.06);
      return {
        key,
        segmentType: spec.segmentType,
        count: targets.length,
        mean,
        lift,
        share,
        trend,
        volatility,
        score,
        action: opportunityAction(lift, trend, volatility, targets.length),
      };
    })
    .filter((row) => row.count >= Math.max(2, Math.floor(totalRows * 0.025)) && row.lift > -0.05)
    .sort((left, right) => right.score - left.score || right.mean - left.mean)
    .slice(0, 8);
}

function segmentKeysForRow(row: Record<string, string>, columns: string[]) {
  if (columns.length === 1) {
    const value = textValue(row[columns[0]]).trim();
    if (!value) {
      return [];
    }
    if (looksDelimitedSegment(value)) {
      return value.split(/[|;,\s]+/).map((item) => item.trim()).filter(Boolean).slice(0, 8);
    }
    return [value];
  }
  const values = columns.map((column) => textValue(row[column]).trim());
  return values.every(Boolean) ? [values.join(' -> ')] : [];
}

function looksDelimitedSegment(value: string) {
  return /[|;,\s]/.test(value) && value.split(/[|;,\s]+/).filter(Boolean).length > 1;
}

function segmentTrend(values: {target: number; timestamp: string; rowIndex: number}[]) {
  if (values.length < 4) {
    return 0;
  }
  const ordered = [...values].sort((left, right) => {
    const leftTime = Date.parse(left.timestamp);
    const rightTime = Date.parse(right.timestamp);
    if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
      return leftTime - rightTime;
    }
    return left.rowIndex - right.rowIndex;
  });
  const midpoint = Math.floor(ordered.length / 2);
  const earlier = ordered.slice(0, midpoint).map((row) => row.target);
  const later = ordered.slice(midpoint).map((row) => row.target);
  const earlierMean = earlier.reduce((sum, value) => sum + value, 0) / Math.max(earlier.length, 1);
  const laterMean = later.reduce((sum, value) => sum + value, 0) / Math.max(later.length, 1);
  return earlierMean === 0 ? 0 : (laterMean - earlierMean) / Math.abs(earlierMean);
}

function coefficientOfVariation(values: number[]) {
  if (values.length < 2) {
    return 0;
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return mean === 0 ? 0 : Math.sqrt(variance) / Math.abs(mean);
}

function opportunityAction(lift: number, trend: number, volatility: number, count: number) {
  if (lift > 0.2 && trend > 0.08) {
    return 'Prioritize capacity';
  }
  if (lift > 0.12 && volatility > 0.22) {
    return 'Stabilize service';
  }
  if (lift > 0.12) {
    return 'Target offer';
  }
  if (trend > 0.12 && count >= 4) {
    return 'Watch rising demand';
  }
  return 'Monitor segment';
}

type ProjectedGeoPoint = {
  lon: number;
  lat: number;
  target: number;
  dropLon: number;
  dropLat: number;
  routeKey?: string;
  x: number;
  y: number;
  dropX?: number;
  dropY?: number;
};

function spatialBins(points: {lon: number; lat: number; target: number}[], limit: number) {
  if (points.length === 0) {
    return [];
  }
  const minLon = Math.min(...points.map((point) => point.lon));
  const maxLon = Math.max(...points.map((point) => point.lon));
  const minLat = Math.min(...points.map((point) => point.lat));
  const maxLat = Math.max(...points.map((point) => point.lat));
  const gridSize = 6;
  const cells = new Map<string, {x: number; y: number; sum: number; count: number}>();
  for (const point of points) {
    const x = Math.min(gridSize - 1, Math.max(0, Math.floor(((point.lon - minLon) / (maxLon - minLon || 1)) * gridSize)));
    const y = Math.min(gridSize - 1, Math.max(0, Math.floor(((point.lat - minLat) / (maxLat - minLat || 1)) * gridSize)));
    const key = `${x}:${y}`;
    const cell = cells.get(key) ?? {x, y, sum: 0, count: 0};
    cell.sum += point.target;
    cell.count += 1;
    cells.set(key, cell);
  }
  return Array.from(cells.values())
    .map((cell) => {
      const lonSpan = (maxLon - minLon || 1) / gridSize;
      const latSpan = (maxLat - minLat || 1) / gridSize;
      return {
        x: cell.x,
        y: cell.y,
        count: cell.count,
        mean: cell.sum / cell.count,
        minLon: minLon + lonSpan * cell.x,
        maxLon: minLon + lonSpan * (cell.x + 1),
        minLat: minLat + latSpan * cell.y,
        maxLat: minLat + latSpan * (cell.y + 1),
      };
    })
    .sort((left, right) => right.mean - left.mean || right.count - left.count)
    .slice(0, limit);
}

function projectedCellPath(
  cell: {minLon: number; maxLon: number; minLat: number; maxLat: number},
  projection: ReturnType<typeof geoMercator>,
) {
  const corners = [
    projection([cell.minLon, cell.minLat]),
    projection([cell.maxLon, cell.minLat]),
    projection([cell.maxLon, cell.maxLat]),
    projection([cell.minLon, cell.maxLat]),
  ];
  if (corners.some((corner) => corner === null)) {
    return null;
  }
  const projected = corners as [number, number][];
  return `M${projected.map((corner) => `${corner[0]},${corner[1]}`).join('L')}Z`;
}

function formatCompact(value: unknown) {
  const numeric = coerceFiniteNumber(value);
  if (numeric === null) {
    return '-';
  }
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: Math.abs(numeric) >= 100 ? 0 : 2,
    notation: Math.abs(numeric) >= 10000 ? 'compact' : 'standard',
  }).format(numeric);
}

function splitKindLabel(kind: string) {
  const labels: Record<string, string> = {
    axis: 'Axis threshold',
    diagonal_2d: 'Diagonal spatial',
    gaussian_2d: 'Gaussian spatial',
    periodic: 'Periodic interval',
    sparse_set: 'Sparse set',
    sparse_list: 'Sparse list',
    fuzzy: 'Fuzzy boundary',
  };
  return labels[kind] ?? kind.replaceAll('_', ' ');
}

function splitterKindHint(kind: string) {
  const hints: Record<string, string> = {
    axis: 'One feature threshold sends rows left or right.',
    diagonal_2d: 'A learned line separates two feature dimensions.',
    gaussian_2d: 'Rows inside a learned radius follow one branch.',
    periodic: 'A wrapped interval captures clock or calendar phases.',
    sparse_set: 'Membership in a set-valued feature controls routing.',
    sparse_list: 'Any matching id in a sparse list controls routing.',
    fuzzy: 'A soft boundary blends routing around a base split.',
  };
  return hints[kind] ?? 'Native CartoBoost splitter rule.';
}

function splitKindColor(kind: string) {
  const colors: Record<string, string> = {
    axis: '#168f86',
    diagonal_2d: '#4f8cff',
    gaussian_2d: '#d05ca6',
    periodic: '#b7791f',
    sparse_set: '#6c5ce7',
    sparse_list: '#6c5ce7',
    fuzzy: '#d14b57',
  };
  return colors[kind] ?? '#168f86';
}

function splitKindFill(kind: string) {
  const fills: Record<string, string> = {
    axis: '#e0f5f3',
    diagonal_2d: '#e8f0ff',
    gaussian_2d: '#fae7f3',
    periodic: '#fff7df',
    sparse_set: '#eeeafb',
    sparse_list: '#eeeafb',
    fuzzy: '#ffe8eb',
  };
  return fills[kind] ?? '#e0f5f3';
}

function truncateLabel(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, Math.max(1, maxLength - 1))}...`;
}

function summarizeComponentRecord(record: ForecastComponentRecord) {
  const ignored = new Set(['trend_linear', 'changepoint_delta', 'seasonal_total', 'event_total', 'regressor_total', 'non_trend_total']);
  const rows: {label: string; value: number}[] = [];
  Object.entries(record.components).forEach(([name, value]) => {
    if (ignored.has(name)) {
      return;
    }
    if (typeof value === 'number') {
      rows.push({label: componentLabel(name), value});
      return;
    }
    Object.entries(value).forEach(([childName, childValue]) => {
      if (Number.isFinite(childValue)) {
        rows.push({label: `${componentLabel(name)}: ${childName}`, value: childValue});
      }
    });
  });
  return rows
    .filter((row) => Math.abs(row.value) > 1.0e-9)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8);
}

function numberComponent(value: number | Record<string, number> | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function componentPointValue(record: ForecastComponentRecord | ForecastHistoryComponentRecord, name: string) {
  return numberComponent(flattenNumericComponents(record.components).get(name));
}

function forecastComponentChartSeries(records: ForecastComponentRecord[]): ChartSeries[] {
  if (records.length === 0) {
    return [];
  }
  const baseSeries: ChartSeries[] = [
    {
      label: 'Prediction',
      points: records.map((record) => ({index: record.horizon, value: record.prediction})),
    },
    {
      label: 'Trend',
      points: records.map((record) => ({index: record.horizon, value: record.trend})),
    },
    optionalRecordSeries('Adjusted trend', records, (record) => record.adjusted_trend, (record) => record.horizon),
    optionalRecordSeries('Trend adjustment', records, (record) => record.trend_adjustment, (record) => record.horizon),
    optionalRecordSeries('Residual shock', records, (record) => record.residual_shock, (record) => record.horizon),
  ].filter((series): series is ChartSeries => series !== null);
  return [
    ...baseSeries,
    ...componentChartSeries(records, (record) => record.horizon),
  ].filter((series) => hasVisibleSeriesSignal(series) || series.label === 'Prediction' || series.label === 'Trend');
}

function historyTrendChartSeries(records: ForecastHistoryComponentRecord[]): ChartSeries[] {
  if (records.length === 0) {
    return [];
  }
  return [
    {
      label: 'Actual',
      points: records.map((record) => ({index: record.index, value: record.actual})),
    },
    {
      label: 'Fitted',
      points: records.map((record) => ({index: record.index, value: record.fitted})),
    },
    {
      label: 'Trend',
      points: records.map((record) => ({index: record.index, value: record.trend})),
    },
  ];
}

function historySeasonalityChartSeries(records: ForecastHistoryComponentRecord[]): ChartSeries[] {
  if (records.length === 0) {
    return [];
  }
  return [
    ...componentChartSeries(records, (record) => record.index),
    {
      label: 'Residual',
      points: records.map((record) => ({index: record.index, value: record.residual})),
    },
    {
      label: 'Trend movement',
      points: records.map((record) => ({index: record.index, value: record.trend_movement})),
    },
    {
      label: 'Fitted movement',
      points: records.map((record) => ({index: record.index, value: record.fitted_movement})),
    },
  ].filter(hasVisibleSeriesSignal);
}

function optionalRecordSeries<T>(
  label: string,
  records: T[],
  valueFor: (record: T) => number | null | undefined,
  indexFor: (record: T) => number,
): ChartSeries | null {
  const points = records.map((record) => ({index: indexFor(record), value: valueFor(record)}));
  return points.some((point) => point.value !== undefined && point.value !== null) ? {label, points} : null;
}

function componentChartSeries<T extends {components: Record<string, number | Record<string, number>>}>(
  records: T[],
  indexFor: (record: T) => number,
): ChartSeries[] {
  const keys = componentKeys(records);
  return keys.map((key) => ({
    label: componentSeriesLabel(key),
    points: records.map((record) => ({
      index: indexFor(record),
      value: flattenNumericComponents(record.components).get(key),
    })),
  }));
}

function componentKeys<T extends {components: Record<string, number | Record<string, number>>}>(
  records: T[],
) {
  const ignored = new Set(['trend_linear', 'changepoint_delta']);
  const keys = new Set<string>();
  for (const record of records) {
    for (const key of flattenNumericComponents(record.components).keys()) {
      if (!ignored.has(key)) {
        keys.add(key);
      }
    }
  }
  return [...keys].sort(componentKeySort);
}

function flattenNumericComponents(
  components: Record<string, number | Record<string, number>>,
  prefix = '',
): Map<string, number> {
  const values = new Map<string, number>();
  Object.entries(components).forEach(([key, value]) => {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'number' && Number.isFinite(value)) {
      values.set(fullKey, value);
      return;
    }
    if (value && typeof value === 'object') {
      flattenNumericComponents(value, fullKey).forEach((childValue, childKey) => {
        values.set(childKey, childValue);
      });
    }
  });
  return values;
}

function componentKeySort(left: string, right: string) {
  const rank = (key: string) => {
    const order = [
      'seasonal_total',
      'weekly',
      'yearly',
      'daily',
      'event_total',
      'regressor_total',
      'non_trend_total',
    ];
    const index = order.indexOf(key);
    return index === -1 ? order.length : index;
  };
  return rank(left) - rank(right) || left.localeCompare(right);
}

function componentSeriesLabel(key: unknown) {
  return textValue(key).split('.').map(componentLabel).join(': ');
}

function hasVisibleSeriesSignal(series: ChartSeries) {
  return series.points.some((point) => Math.abs(numberComponent(point.value as number | null | undefined)) > 1.0e-9);
}

function componentLabel(name: unknown) {
  return textValue(name)
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function guessColumn(columns: string[], candidates: string[]) {
  const normalized = new Map(columns.map((column) => [column.toLowerCase(), column]));
  for (const candidate of candidates) {
    const match = normalized.get(candidate.toLowerCase());
    if (match) {
      return match;
    }
  }
  return null;
}

function coordinatePair(columns: string[], prefixes: string[]) {
  for (const prefix of prefixes) {
    const prefixPart = prefix ? `${prefix}_` : '';
    const lat = guessColumn(columns, [
      `${prefixPart}latitude`,
      `${prefixPart}lat`,
      `${prefixPart}y`,
      `${prefixPart}centroid_lat`,
      `${prefixPart}centroid_y`,
      prefix ? `${prefix}Latitude` : 'latitude',
      prefix ? `${prefix}Lat` : 'lat',
      prefix ? `${prefix}CentroidLat` : 'centroid_lat',
      prefix ? `${prefix}CentroidY` : 'centroid_y',
    ]);
    const lon = guessColumn(columns, [
      `${prefixPart}longitude`,
      `${prefixPart}lon`,
      `${prefixPart}lng`,
      `${prefixPart}x`,
      `${prefixPart}centroid_lon`,
      `${prefixPart}centroid_lng`,
      `${prefixPart}centroid_x`,
      prefix ? `${prefix}Longitude` : 'longitude',
      prefix ? `${prefix}Lon` : 'lon',
      prefix ? `${prefix}Lng` : 'lng',
      prefix ? `${prefix}CentroidLon` : 'centroid_lon',
      prefix ? `${prefix}CentroidLng` : 'centroid_lng',
      prefix ? `${prefix}CentroidX` : 'centroid_x',
    ]);
    if (lat && lon) {
      return {lat, lon};
    }
  }
  return null;
}

function h3Summary(table: ParsedTable, column: string) {
  const counts = new Map<string, number>();
  const acceptNamedH3Values = column.toLowerCase().includes('h3');
  for (const row of table.rows) {
    const cells = textValue(row[column])
      .split(/[|;,\s]+/)
      .map((cell) => cell.trim())
      .filter((cell) => cell !== '' && (acceptNamedH3Values || isH3Like(cell)));
    for (const cell of cells) {
      counts.set(cell, (counts.get(cell) ?? 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 8)
    .map(([cell, count]) => ({cell, count}));
}

function h3CellLabel(row: {cell?: string; count?: number; 0?: string; 1?: number}) {
  return row.cell ?? row[0] ?? '';
}

function h3CellCount(row: {cell?: string; count?: number; 0?: string; 1?: number}) {
  return Number(row.count ?? row[1] ?? 0);
}

function targetColor(value: number, min: number, max: number) {
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const hue = 190 - ratio * 150;
  return `hsl(${hue.toFixed(1)} 78% 48%)`;
}

function targetRgb(value: number, min: number, max: number, alpha: number): [number, number, number, number] {
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const red = Math.round(18 + ratio * 218);
  const green = Math.round(143 - ratio * 70);
  const blue = Math.round(134 - ratio * 82);
  return [red, green, blue, alpha];
}

function defaultFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  return numericFeatureColumns(table, targetCol, timestampCol, seriesCol).slice(0, 12);
}

function defaultSparseFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  return sparseFeatureColumns(table, targetCol, timestampCol, seriesCol).slice(0, 4);
}

function numericFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  const excluded = new Set([targetCol, timestampCol, seriesCol].filter(Boolean));
  return table.columns.filter((column) => {
    if (excluded.has(column)) {
      return false;
    }
    return table.rows.some((row) => {
      const value = row[column]?.trim();
      return Boolean(value) && Number.isFinite(Number(value));
    });
  });
}

function sparseFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  const excluded = new Set([targetCol, timestampCol, seriesCol].filter(Boolean));
  return table.columns.filter((column) => {
    if (excluded.has(column) || numericFeatureColumns(table, targetCol, timestampCol, seriesCol).includes(column)) {
      return false;
    }
    return looksSparseSetColumn(table, column);
  });
}

function looksSparseSetColumn(table: ParsedTable, column: string) {
  const normalized = column.toLowerCase();
  const namedLikeSparse =
    normalized.includes('h3') ||
    normalized.includes('s2') ||
    normalized.includes('membership') ||
    normalized.includes('zone_ids') ||
    normalized.includes('cell') ||
    normalized.includes('cell_ids') ||
    normalized.includes('sparse') ||
    normalized.endsWith('_set') ||
    normalized.endsWith('_ids');
  return table.rows.some((row) => {
    const value = textValue(row[column]).trim();
    return value !== '' && (namedLikeSparse || /[|;,\s]/.test(value) || isH3Like(value)) && parseSparseSet(value).length > 0;
  });
}

function parseSparseSet(value: unknown) {
  return Array.from(
    new Set(
      textValue(value)
        .split(/[|;,\s]+/)
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => {
          const numeric = Number(item);
          if (Number.isInteger(numeric) && numeric >= 0) {
            return numeric;
          }
          return isH3Like(item) ? stableStringId(item) : Number.NaN;
        })
        .filter((item) => Number.isInteger(item) && item >= 0),
    ),
  );
}

function isH3Like(value: unknown) {
  const normalized = textValue(value).trim().toLowerCase().replace(/^0x/, '');
  return /^[0-9a-f]{15,16}$/.test(normalized);
}

function textValue(value: unknown) {
  return typeof value === 'string' ? value : value == null ? '' : String(value);
}

function inferFeatureKind(column: string) {
  const normalized = column.toLowerCase();
  if (
    normalized.includes('hour') ||
    normalized === 'dow' ||
    normalized.includes('day_of_week') ||
    normalized.includes('weekday')
  ) {
    return 'periodic';
  }
  if (
    normalized.includes('lat') ||
    normalized.includes('lon') ||
    normalized.includes('longitude') ||
    normalized.includes('latitude') ||
    normalized.endsWith('_x') ||
    normalized.endsWith('_y')
  ) {
    return 'spatial';
  }
  return 'numeric';
}

function inferMonotonicConstraint(column: string) {
  const normalized = column.toLowerCase();
  if (
    normalized.includes('trip_distance') ||
    normalized === 'distance' ||
    normalized.endsWith('_distance') ||
    normalized.includes('duration') ||
    normalized.includes('elapsed') ||
    normalized.includes('fare') ||
    normalized.includes('tip') ||
    normalized.includes('toll') ||
    normalized.includes('surcharge') ||
    normalized.includes('passenger_count')
  ) {
    return 1;
  }
  if (normalized.includes('discount') || normalized.includes('promo')) {
    return -1;
  }
  return 0;
}

function inferPeriodicPeriod(column: string) {
  return column.toLowerCase().includes('hour') ? 24 : 7;
}

function seasonalityPresets(frequency: string) {
  if (frequency === 'hourly') {
    return [
      {key: 'day', label: 'Day', description: 'Daily hourly cycle', length: 24},
      {key: 'week', label: 'Week', description: 'Weekly hourly cycle', length: 168},
      {key: 'quarter', label: 'Quarter', description: 'Quarter-year hourly cycle', length: 2184},
      {key: 'year', label: 'Year', description: 'Yearly hourly cycle', length: 8760},
    ];
  }
  if (frequency === 'weekly') {
    return [
      {key: 'quarter', label: 'Quarter', description: 'Quarter-year weekly cycle', length: 13},
      {key: 'year', label: 'Year', description: 'Yearly weekly cycle', length: 52},
      {key: 'two_year', label: '2 years', description: 'Two-year weekly cycle', length: 104},
    ];
  }
  return [
    {key: 'week', label: 'Week', description: 'Weekly daily cycle', length: 7},
    {key: 'quarter', label: 'Quarter', description: 'Quarter-year daily cycle', length: 91},
    {key: 'year', label: 'Year', description: 'Yearly daily cycle', length: 365},
  ];
}

function seasonalityLabel(frequency: string, seasonLength: number) {
  return seasonalityPresets(frequency).find((preset) => preset.length === seasonLength)?.label ?? 'Custom';
}

function buildSuggestedConfig({
  table,
  timestampCol,
  targetCol,
  seriesCol,
  frequency,
  horizon,
  seasonLength,
  model,
  modelOptions,
  featureCols,
  sparseFeatureCols,
  modelingMode,
  modelingLoss,
  neuralPipeline,
  neuralIdCol,
  graphSourceCol,
  graphTargetCol,
  graphWeightCol,
}: {
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
  frequency: string;
  horizon: number;
  seasonLength: number;
  model: string;
  modelOptions: ModelOption[];
  featureCols: string[];
  sparseFeatureCols: string[];
  modelingMode: string;
  modelingLoss: string;
  neuralPipeline: string;
  neuralIdCol: string;
  graphSourceCol: string;
  graphTargetCol: string;
  graphWeightCol: string;
}) {
  const featureKinds = Object.fromEntries(featureCols.map((column) => [column, inferFeatureKind(column)]));
  const sparseFeatureKinds = Object.fromEntries(sparseFeatureCols.map((column) => [column, 'sparse_set']));
  const periodicPeriods = Object.fromEntries(
    featureCols
      .filter((column) => inferFeatureKind(column) === 'periodic')
      .map((column) => [column, inferPeriodicPeriod(column)]),
  );
  const selectedModel = modelOptions.find((option) => option.value === model);
  const targetingProfile = buildTargetingProfile(table, targetCol, timestampCol, seriesCol, featureCols, sparseFeatureCols, modelOptions);
  const rankedFeatures = rankFeatureRoles(table, targetCol, timestampCol, seriesCol, featureCols, sparseFeatureCols).slice(0, 20);
  const opportunityTargets = rankTargetOpportunities(table, targetCol, timestampCol, seriesCol, 20);
  const modelSpecificOptions = browserForecastModelOptions(model, table, timestampCol, targetCol, seriesCol, horizon, seasonLength);
  return {
    cartoboostConfigVersion: 1,
    source: {
      fileName: table.fileName,
      rowCount: table.rows.length,
      columns: table.columns,
      acceptedFormats: ['csv', 'tsv', 'parquet'],
    },
    browserWasm: {
      page: '/modeling-lab',
      crate: 'cartoboost-wasm',
      entrypoints: [
        'runForecast',
        'runRegressionModel',
        'runNeuralModel',
        'runSequence',
        'runGeotemporalDiagnostics',
        'availableForecastModels',
      ],
    },
    geography: {
      enabled: true,
      h3FeatureCols: table.columns.filter((column) => table.rows.some((row) => isH3Like(row[column] ?? ''))),
      coordinatePairs: {
        pickup: coordinatePair(table.columns, ['pickup', 'pu', 'origin', '']),
        dropoff: coordinatePair(table.columns, ['dropoff', 'do', 'destination']),
      },
      visualizations: ['spatial_target_scatter', 'route_endpoint_lines', 'spatial_hot_cells', 'h3_cell_frequency', 'target_ranked_routes'],
    },
    targeting: {
      enabled: true,
      diagnostics: ['target_histogram', 'zero_share', 'p10_p90_band', 'time_bucket_mean', 'series_mean_rank'],
      targetCol,
      seriesIdCol: seriesCol || null,
      recommendedPlan: {
        forecastModel: targetingProfile.forecastModelValue,
        forecastModelLabel: targetingProfile.forecastModel,
        splitterMode: targetingProfile.splitterModeValue,
        splitterLabel: targetingProfile.splitterMode,
        loss: targetingProfile.lossValue,
        lossLabel: targetingProfile.loss,
        graphPath: targetingProfile.graphPath,
        readinessScore: targetingProfile.score,
        readinessChecks: targetingProfile.readinessChecks,
        reasons: targetingProfile.reasons,
        denseFeatureCols: targetingProfile.featureCols,
        sparseFeatureCols: targetingProfile.sparseFeatureCols,
        neuralPipeline: targetingProfile.neuralPipeline,
        neuralIdCol: targetingProfile.neuralIdCol || null,
        graphSourceCol: targetingProfile.graphSourceCol || null,
        graphTargetCol: targetingProfile.graphTargetCol || null,
        graphWeightCol: targetingProfile.graphWeightCol || null,
      },
      rankedFeatures,
      opportunityTargets,
    },
    forecast: {
      enabled: true,
      timestampCol,
      targetCol,
      seriesIdCol: seriesCol || null,
      frequency,
      horizon,
      model,
      modelLabel: selectedModel?.label ?? model,
      pipeline: selectedModel?.group ?? 'custom',
      options: {
        seasonLength,
        seasonality: seasonalityLabel(frequency, seasonLength),
        ...modelSpecificOptions,
      },
      roster: modelOptions.map((option) => ({
        model: option.value,
        label: option.label,
        pipeline: option.group,
      })),
    },
    modeling: {
      enabled: featureCols.length > 0,
      targetCol,
      featureCols,
      sparseFeatureCols,
      splitterMode: modelingMode,
      loss: modelingLoss,
      featureKinds,
      sparseFeatureKinds,
      sparseSetFormat: {
        delimiter: 'one or more of | ; , or whitespace',
        valueType: 'non-negative integer ids',
      },
      periodicPeriods,
      holdoutFraction: 0.2,
      booster: {
        nEstimators: 120,
        learningRate: 0.06,
        maxDepth: 3,
        minSamplesLeafRule: 'max(2, min(20, floor(finite_rows / 20)))',
        intervalLowerAlpha: 0.1,
        intervalUpperAlpha: 0.9,
      },
      metrics: ['rmse', 'mae', 'r2'],
      visualizations: ['actual_vs_predicted', 'split_count_feature_importance', 'prediction_table'],
    },
    neuralModeling: {
      enabled: featureCols.length > 0,
      pipeline: neuralPipeline,
      targetCol,
      denseFeatureCols: featureCols,
      idCol: neuralIdCol || null,
      graphSourceCol: graphSourceCol || null,
      graphTargetCol: graphTargetCol || null,
      graphWeightCol: graphWeightCol || null,
      holdoutFraction: 0.2,
      embeddingDim: 8,
      supportedPipelines: Object.keys(neuralPipelineLabels),
      node2vec: {
        walkLength: 8,
        walksPerNode: 4,
        windowSize: 3,
        epochs: 2,
      },
      graphSage: {
        epochs: 3,
        negativeSamples: 2,
        nodeFeatures: 'mean of selected denseFeatureCols by source/target node',
        nodeTypes: 'inferred from graph source/target column roles',
        edgeTypes: 'inferred from source node type and target node type',
      },
      visualizations: ['actual_vs_predicted', 'embedding_or_graph_split_importance', 'prediction_table'],
    },
  };
}

function downloadJson(fileName: string, payload: unknown) {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function stripExtension(fileName: string) {
  return fileName.replace(/\.[^.]+$/, '') || 'dataset';
}

function numericCovariates(row: Record<string, string>, excludedColumns: string[]) {
  const excluded = new Set(excludedColumns.filter(Boolean));
  return Object.fromEntries(
    Object.entries(row)
      .filter(([column]) => !excluded.has(column))
      .map(([column, value]) => [column, Number(value)])
      .filter(([, value]) => Number.isFinite(value)),
  );
}

function normalizedForecastModel(model: string) {
  return model.trim().toLowerCase().replace(/-/g, '_');
}

function isPiecewiseForecastModel(model: string) {
  return normalizedForecastModel(model) === 'piecewise_linear_seasonal';
}

function isSpatialPiecewiseKrigingModel(model: string) {
  return normalizedForecastModel(model) === 'spatial_piecewise_kriging';
}

function isThetaForecastModel(model: string) {
  return ['theta', 'optimized_theta'].includes(normalizedForecastModel(model));
}

function browserForecastModelOptions(
  model: string,
  table: ParsedTable,
  timestampCol: string,
  targetCol: string,
  seriesCol: string,
  horizon: number,
  seasonLength: number,
) {
  if (isSpatialPiecewiseKrigingModel(model)) {
    return spatialPiecewiseKrigingForecastOptions(table, timestampCol, targetCol, seriesCol, horizon, seasonLength);
  }
  if (normalizedForecastModel(model) === 'kriging') {
    return coordinateForecastOptions(table);
  }
  if (isPiecewiseForecastModel(model)) {
    return piecewiseForecastOptions(table, targetCol, seriesCol, [], horizon, seasonLength);
  }
  if (isThetaForecastModel(model)) {
    return {thetaSeasonality: 'additive'};
  }
  return {};
}

function piecewiseForecastOptions(
  table: ParsedTable,
  targetCol: string,
  seriesCol: string,
  covariateColumns: string[],
  horizon: number,
  seasonLength: number,
) {
  return {
    nChangepoints: 25,
    changepointRange: 1.0,
    ...(seasonLength === 24 ? {dailyFourierOrder: 4} : {}),
    seasonalityMode: 'additive',
    holidaysMode: 'additive',
    extraRegressors: covariateColumns,
    extraRegressorMonotonicConstraints: Object.fromEntries(
      covariateColumns
        .map((column) => [column, inferMonotonicConstraint(column)] as const)
        .filter(([, direction]) => direction !== 0),
    ),
    futureRegressors: carriedFutureRegressors(table, covariateColumns, horizon),
    futureRegressorsBySeries: seasonalFutureRegressorsBySeries(
      table,
      covariateColumns,
      horizon,
      seriesCol,
      seasonLength,
    ),
    intervalWidth: 0.8,
    quantileLevels: [0.1, 0.5, 0.9],
    uncertaintySamples: 128,
    includeSamples: false,
    includeQuantiles: true,
    ...inferLogisticBoundRegressors(table, targetCol, covariateColumns),
  };
}

function spatialPiecewiseKrigingForecastOptions(
  table: ParsedTable,
  timestampCol: string,
  targetCol: string,
  seriesCol: string,
  horizon: number,
  seasonLength: number,
) {
  const candidateRegressors = spatialRegressorColumns(table, [timestampCol, targetCol, seriesCol]);
  const knownFutureRegressors = knownFutureRegressorColumns(candidateRegressors);
  const knownFutureSet = new Set(knownFutureRegressors);
  const spatialRegressors = candidateRegressors.filter((column) => !knownFutureSet.has(column));
  return {
    ...piecewiseForecastOptions(table, targetCol, seriesCol, knownFutureRegressors, horizon, seasonLength),
    ...coordinateForecastOptions(table),
    includeHistoryComponents: false,
    includeQuantiles: false,
    uncertaintySamples: 0,
    spatialKrigingMode: spatialRegressors.length > 0 ? 'hybrid' : 'residual_kriging',
    spatialRegressors,
    residualShrinkage: 1.0,
    allowNeighborFallback: true,
  };
}

function coordinateForecastOptions(table: ParsedTable) {
  const coordinateX = inferCoordinateColumn(table.columns, ['longitude', 'lon', 'lng', 'x']);
  const coordinateY = inferCoordinateColumn(table.columns, ['latitude', 'lat', 'y']);
  return {
    ...(coordinateX ? {coordinateX} : {}),
    ...(coordinateY ? {coordinateY} : {}),
  };
}

function inferCoordinateColumn(columns: string[], candidates: string[]) {
  const normalizedCandidates = new Set(candidates.map((candidate) => candidate.toLowerCase()));
  return (
    columns.find((column) => normalizedCandidates.has(column.toLowerCase())) ??
    columns.find((column) => candidates.some((candidate) => column.toLowerCase().includes(candidate)))
  );
}

function spatialRegressorColumns(table: ParsedTable, excludedColumns: string[]) {
  return numericCovariateColumns(table, excludedColumns)
    .filter((column) => !isSpatialCoordinateColumn(column) && !isIdentifierColumn(column))
    .slice(0, 8);
}

function knownFutureRegressorColumns(columns: string[]) {
  return columns.filter((column) => {
    const normalized = column.toLowerCase();
    return (
      normalized === 'hour' ||
      normalized === 'day_of_week' ||
      normalized === 'dayofweek' ||
      normalized === 'is_weekend' ||
      normalized === 'is_commute_peak' ||
      normalized === 'is_airport_lane' ||
      normalized.endsWith('_hour') ||
      normalized.endsWith('_dow')
    );
  });
}

function isSpatialCoordinateColumn(column: string) {
  const normalized = column.toLowerCase();
  return (
    normalized.includes('latitude') ||
    normalized.includes('longitude') ||
    normalized === 'lat' ||
    normalized === 'lon' ||
    normalized === 'lng' ||
    normalized.endsWith('_lat') ||
    normalized.endsWith('_lon') ||
    normalized.endsWith('_lng') ||
    normalized.endsWith('_x') ||
    normalized.endsWith('_y')
  );
}

function isIdentifierColumn(column: string) {
  const normalized = column.toLowerCase();
  return normalized === 'id' || normalized.endsWith('_id') || normalized.endsWith('id');
}

function numericCovariateColumns(table: ParsedTable, excludedColumns: string[]) {
  const excluded = new Set(excludedColumns.filter(Boolean));
  return table.columns.filter((column) => {
    if (excluded.has(column)) {
      return false;
    }
    return table.rows.some((row) => Number.isFinite(Number(row[column])));
  });
}

function carriedFutureRegressors(table: ParsedTable, columns: string[], horizon: number) {
  const lastFinite: Record<string, number> = {};
  for (const column of columns) {
    for (let index = table.rows.length - 1; index >= 0; index -= 1) {
      const value = Number(table.rows[index][column]);
      if (Number.isFinite(value)) {
        lastFinite[column] = value;
        break;
      }
    }
  }
  return Object.fromEntries(
    columns
      .filter((column) => Number.isFinite(lastFinite[column]))
      .map((column) => [column, Array.from({length: horizon}, () => lastFinite[column])]),
  );
}

function carriedFutureRegressorsBySeries(
  table: ParsedTable,
  columns: string[],
  horizon: number,
  seriesCol: string,
) {
  if (!seriesCol) {
    return {};
  }
  const lastFiniteBySeries: Record<string, Record<string, number>> = {};
  for (const row of table.rows) {
    const seriesId = String(row[seriesCol] ?? '');
    if (!seriesId) {
      continue;
    }
    const values = (lastFiniteBySeries[seriesId] ??= {});
    for (const column of columns) {
      const value = Number(row[column]);
      if (Number.isFinite(value)) {
        values[column] = value;
      }
    }
  }
  return Object.fromEntries(
    Object.entries(lastFiniteBySeries).map(([seriesId, values]) => [
      seriesId,
      Object.fromEntries(
        columns
          .filter((column) => Number.isFinite(values[column]))
          .map((column) => [column, Array.from({length: horizon}, () => values[column])]),
      ),
    ]),
  );
}

function seasonalFutureRegressorsBySeries(
  table: ParsedTable,
  columns: string[],
  horizon: number,
  seriesCol: string,
  seasonLength: number,
) {
  if (!seriesCol || columns.length === 0) {
    return {};
  }
  const rowsBySeries = new Map<string, Record<string, string>[]>();
  for (const row of table.rows) {
    const seriesId = String(row[seriesCol] ?? '');
    if (!seriesId) {
      continue;
    }
    const rows = rowsBySeries.get(seriesId) ?? [];
    rows.push(row);
    rowsBySeries.set(seriesId, rows);
  }
  const period = Math.max(1, Math.floor(seasonLength));
  return Object.fromEntries(
    Array.from(rowsBySeries.entries()).map(([seriesId, rows]) => [
      seriesId,
      Object.fromEntries(
        columns
          .map((column) => {
            const fallback = lastFiniteColumnValue(rows, column);
            const values = Array.from({length: horizon}, (_, offset) => {
              const seasonalIndex = rows.length - period + (offset % period);
              const seasonalRow = seasonalIndex >= 0 ? rows[seasonalIndex] : undefined;
              const seasonalValue = seasonalRow ? Number(seasonalRow[column]) : NaN;
              return Number.isFinite(seasonalValue) ? seasonalValue : fallback;
            }).filter((value) => Number.isFinite(value));
            return values.length === horizon ? ([column, values] as const) : null;
          })
          .filter((entry): entry is readonly [string, number[]] => entry !== null),
      ),
    ]),
  );
}

function lastFiniteColumnValue(rows: Record<string, string>[], column: string) {
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const value = Number(rows[index][column]);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return NaN;
}

function inferLogisticBoundRegressors(table: ParsedTable, targetCol: string, columns: string[]) {
  const capRegressor = columns.find((column) => {
    const normalized = column.toLowerCase();
    return (
      (normalized.includes('capacity') ||
        normalized.includes('cap') ||
        normalized.includes('ceiling') ||
        normalized.includes('upper') ||
        normalized === 'max' ||
        normalized.endsWith('_max')) &&
      observedBoundsTarget(table, targetCol, column, 'upper')
    );
  });
  const floorRegressor = columns.find((column) => {
    const normalized = column.toLowerCase();
    return (
      (normalized.includes('floor') ||
        normalized.includes('minimum') ||
        normalized.includes('lower') ||
        normalized === 'min' ||
        normalized.endsWith('_min')) &&
      observedBoundsTarget(table, targetCol, column, 'lower')
    );
  });
  if (!capRegressor) {
    return {};
  }
  return {
    growth: 'logistic',
    capRegressor,
    ...(floorRegressor ? {floorRegressor} : {}),
  };
}

function observedBoundsTarget(
  table: ParsedTable,
  targetCol: string,
  column: string,
  direction: 'upper' | 'lower',
) {
  let compared = 0;
  for (const row of table.rows) {
    const target = Number(row[targetCol]);
    const bound = Number(row[column]);
    if (!Number.isFinite(target) || !Number.isFinite(bound)) {
      continue;
    }
    compared += 1;
    if (direction === 'upper' && bound <= target) {
      return false;
    }
    if (direction === 'lower' && bound >= target) {
      return false;
    }
  }
  return compared >= Math.max(3, Math.floor(table.rows.length * 0.5));
}

async function importExternalModule(url: string) {
  const dynamicImport = new Function('url', 'return import(url);') as (url: string) => Promise<unknown>;
  const response = await fetch(url, {cache: 'no-store'});
  const contentType = response.headers.get('content-type') ?? '';
  if (!response.ok || contentType.includes('text/html')) {
    throw new Error('CartoBoost interactive model bundle is not available from this dev server.');
  }
  const blobUrl = URL.createObjectURL(
    new Blob([await response.text()], {type: contentType.includes('javascript') ? contentType : 'text/javascript'}),
  );
  try {
    return await dynamicImport(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

async function ensureJavaScriptModule(url: string) {
  const response = await fetch(url, {method: 'HEAD', cache: 'no-store'});
  const contentType = response.headers.get('content-type') ?? '';
  if (!response.ok || contentType.includes('text/html')) {
    throw new Error('CartoBoost interactive model bundle is not available from this dev server.');
  }
}

let initializedWasmModule:
  | {
      key: string;
      promise: Promise<WasmModule>;
    }
  | null = null;
let forecastWorkerClient: ForecastWorkerClient | null = null;

type ForecastWorkerClient = {
  key: string;
  worker: Worker;
  nextId: number;
  pending: Map<number, {resolve: (value: ForecastResponse) => void; reject: (error: Error) => void}>;
};

async function getInitializedWasmModule(wasmJsUrl: string, wasmBinaryUrl: string) {
  const key = `${wasmJsUrl}\u0000${wasmBinaryUrl}`;
  if (!initializedWasmModule || initializedWasmModule.key !== key) {
    initializedWasmModule = {
      key,
      promise: (async () => {
        await ensureJavaScriptModule(wasmJsUrl);
        const wasmModule = (await importExternalModule(wasmJsUrl)) as WasmModule;
        await wasmModule.default({module_or_path: wasmBinaryUrl});
        return wasmModule;
      })(),
    };
  }
  try {
    return await initializedWasmModule.promise;
  } catch (error) {
    initializedWasmModule = null;
    throw error;
  }
}

async function getForecastModelOptions(wasmJsUrl: string, wasmBinaryUrl: string): Promise<ModelOption[]> {
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  const registry = wasmModule.availableForecastModels?.() ?? [];
  const options = registry
    .filter((model) => model.name && model.label && model.pipeline)
    .map((model) => ({
      value: model.name,
      label: model.label,
      group: model.pipeline,
    }));
  return options.length > 0 ? options : fallbackModelOptions;
}

async function runBrowserForecast({
  wasmJsUrl,
  wasmBinaryUrl,
  table,
  timestampCol,
  targetCol,
  seriesCol,
  frequency,
  horizon,
  model,
  seasonLength,
  isolatedWorker = false,
  timeoutMs,
}: {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
  frequency: string;
  horizon: number;
  model: string;
  seasonLength: number;
  isolatedWorker?: boolean;
  timeoutMs?: number;
}) {
  const request = buildBrowserForecastRequest({
    table,
    timestampCol,
    targetCol,
    seriesCol,
    frequency,
    horizon,
    model,
    seasonLength,
  });
  const response =
    typeof Worker === 'undefined'
      ? (await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl)).runForecast(request)
      : isolatedWorker
        ? await runForecastRequestInIsolatedWorker(wasmJsUrl, wasmBinaryUrl, request, timeoutMs)
        : await runForecastRequestInWorker(wasmJsUrl, wasmBinaryUrl, request);
  assertForecastResponseRecords(response, model);
  return response;
}

function buildBrowserForecastRequest({
  table,
  timestampCol,
  targetCol,
  seriesCol,
  frequency,
  horizon,
  model,
  seasonLength,
}: {
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
  frequency: string;
  horizon: number;
  model: string;
  seasonLength: number;
}) {
  const modelSpecificOptions = browserForecastModelOptions(model, table, timestampCol, targetCol, seriesCol, horizon, seasonLength);
  const browserAutoOptions = model === 'auto_forecast' ? {maxAutoCandidateCount: 4} : {};
  return {
    rows: table.rows.map((row) => ({
      timestamp: row[timestampCol],
      target: Number(row[targetCol]),
      seriesId: seriesCol ? row[seriesCol] : undefined,
      covariates: numericCovariates(row, [timestampCol, targetCol, seriesCol]),
    })),
    frequency,
    horizon,
    model,
    options: {
      seasonLength,
      includeComponents: true,
      includeHistoryComponents: true,
      ...browserAutoOptions,
      ...modelSpecificOptions,
    },
    metadata: {
      timestampCol,
      targetCol,
      seriesIdCol: seriesCol || undefined,
    },
  };
}

function runForecastRequestInWorker(wasmJsUrl: string, wasmBinaryUrl: string, request: unknown): Promise<ForecastResponse> {
  const absoluteWasmJsUrl = absoluteBrowserUrl(wasmJsUrl);
  const absoluteWasmBinaryUrl = absoluteBrowserUrl(wasmBinaryUrl);
  const client = getForecastWorkerClient(absoluteWasmJsUrl, absoluteWasmBinaryUrl);
  const id = client.nextId;
  client.nextId += 1;
  return new Promise((resolve, reject) => {
    client.pending.set(id, {resolve, reject});
    client.worker.postMessage({id, wasmJsUrl: absoluteWasmJsUrl, wasmBinaryUrl: absoluteWasmBinaryUrl, request});
  });
}

function runForecastRequestInIsolatedWorker(
  wasmJsUrl: string,
  wasmBinaryUrl: string,
  request: unknown,
  timeoutMs = BACKTEST_FORECAST_TIMEOUT_MS,
): Promise<ForecastResponse> {
  const absoluteWasmJsUrl = absoluteBrowserUrl(wasmJsUrl);
  const absoluteWasmBinaryUrl = absoluteBrowserUrl(wasmBinaryUrl);
  const worker = new Worker(URL.createObjectURL(new Blob([forecastWorkerSource], {type: 'text/javascript'})), {type: 'module'});
  let settled = false;
  return new Promise((resolve, reject) => {
    const finish = (callback: () => void) => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeout);
      worker.terminate();
      callback();
    };
    const timeout = window.setTimeout(() => {
      finish(() => reject(new Error(`forecast worker timed out after ${formatElapsedMs(timeoutMs)}`)));
    }, timeoutMs);
    worker.onmessage = (event: MessageEvent<{id: number; response?: ForecastResponse; error?: string}>) => {
      const {response, error} = event.data;
      if (error) {
        finish(() => reject(new Error(error)));
        return;
      }
      if (!response) {
        finish(() => reject(new Error('forecast worker returned an empty response')));
        return;
      }
      finish(() => resolve(response));
    };
    worker.onerror = (event) => {
      finish(() => reject(new Error(event.message || 'forecast worker failed')));
    };
    worker.postMessage({id: 1, wasmJsUrl: absoluteWasmJsUrl, wasmBinaryUrl: absoluteWasmBinaryUrl, request});
  });
}

function absoluteBrowserUrl(url: string) {
  return new URL(url, window.location.href).toString();
}

function getForecastWorkerClient(wasmJsUrl: string, wasmBinaryUrl: string): ForecastWorkerClient {
  const key = `${wasmJsUrl}\u0000${wasmBinaryUrl}`;
  if (forecastWorkerClient?.key === key) {
    return forecastWorkerClient;
  }
  forecastWorkerClient?.worker.terminate();
  const worker = new Worker(URL.createObjectURL(new Blob([forecastWorkerSource], {type: 'text/javascript'})), {type: 'module'});
  const pending: ForecastWorkerClient['pending'] = new Map();
  worker.onmessage = (event: MessageEvent<{id: number; response?: ForecastResponse; error?: string}>) => {
    const {id, response, error} = event.data;
    const callbacks = pending.get(id);
    if (!callbacks) {
      return;
    }
    pending.delete(id);
    if (error) {
      callbacks.reject(new Error(error));
      return;
    }
    if (!response) {
      callbacks.reject(new Error('forecast worker returned an empty response'));
      return;
    }
    callbacks.resolve(response);
  };
  worker.onerror = (event) => {
    const message = event.message || 'forecast worker failed';
    for (const callbacks of pending.values()) {
      callbacks.reject(new Error(message));
    }
    pending.clear();
  };
  forecastWorkerClient = {key, worker, nextId: 1, pending};
  return forecastWorkerClient;
}

const forecastWorkerSource = `
let wasmKey = '';
let wasmPromise = null;

async function getWasmModule(wasmJsUrl, wasmBinaryUrl) {
  const key = wasmJsUrl + '\\0' + wasmBinaryUrl;
  if (wasmPromise && wasmKey === key) {
    return wasmPromise;
  }
  wasmKey = key;
  wasmPromise = (async () => {
    const response = await fetch(wasmJsUrl, {cache: 'no-store'});
    const contentType = response.headers.get('content-type') || '';
    if (!response.ok || contentType.includes('text/html')) {
      throw new Error('CartoBoost interactive model bundle is not available from this dev server.');
    }
    const blobUrl = URL.createObjectURL(
      new Blob([await response.text()], {type: contentType.includes('javascript') ? contentType : 'text/javascript'}),
    );
    const wasmModule = await import(blobUrl);
    URL.revokeObjectURL(blobUrl);
    await wasmModule.default({module_or_path: wasmBinaryUrl});
    return wasmModule;
  })();
  return wasmPromise;
}

self.onmessage = async (event) => {
  const {id, wasmJsUrl, wasmBinaryUrl, request} = event.data;
  try {
    const wasmModule = await getWasmModule(wasmJsUrl, wasmBinaryUrl);
    self.postMessage({id, response: wasmModule.runForecast(request)});
  } catch (error) {
    self.postMessage({id, error: error instanceof Error ? error.message : String(error)});
  }
};
`;

async function runBrowserRegression({
  wasmJsUrl,
  wasmBinaryUrl,
  table,
  targetCol,
  featureCols,
  sparseFeatureCols,
  modelingMode,
  modelingLoss,
}: {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  table: ParsedTable;
  targetCol: string;
  featureCols: string[];
  sparseFeatureCols: string[];
  modelingMode: string;
  modelingLoss: string;
}) {
  const rows = sampleModelRows(
    table.rows
      .map((row) => ({
        features: featureCols.map((column) => Number(row[column])),
        sparseSets: sparseFeatureCols.map((column) => parseSparseSet(row[column] ?? '')),
        target: Number(row[targetCol]),
      }))
      .filter((row) => Number.isFinite(row.target) && row.features.every(Number.isFinite)),
    VISUALIZED_MODEL_MAX_ROWS,
  );
  if (rows.length < 4) {
    throw new Error('CartoBoost modeling needs at least four rows with finite target and feature values.');
  }
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  return wasmModule.runRegressionModel({
    rows,
    featureNames: featureCols,
    sparseFeatureNames: sparseFeatureCols,
    options: {
      holdoutFraction: 0.2,
      splitterMode: modelingMode,
      loss: modelingLoss,
      quantileAlpha: 0.5,
      huberDelta: 8,
      logOffset: 1,
      intervalLowerAlpha: 0.1,
      intervalUpperAlpha: 0.9,
      featureKinds: Object.fromEntries(featureCols.map((column) => [column, inferFeatureKind(column)])),
      periodicPeriods: Object.fromEntries(
        featureCols
          .filter((column) => inferFeatureKind(column) === 'periodic')
          .map((column) => [column, inferPeriodicPeriod(column)]),
      ),
      nEstimators: 24,
      learningRate: 0.06,
      maxDepth: 3,
      minSamplesLeaf: Math.max(2, Math.min(20, Math.floor(rows.length / 20))),
      monotonicConstraints: modelingMode === 'axis' ? featureCols.map(inferMonotonicConstraint) : undefined,
      includeModelVisualization: true,
    },
  });
}

async function runBrowserNeural({
  wasmJsUrl,
  wasmBinaryUrl,
  table,
  targetCol,
  featureCols,
  neuralPipeline,
  neuralIdCol,
  graphSourceCol,
  graphTargetCol,
  graphWeightCol,
}: {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  table: ParsedTable;
  targetCol: string;
  featureCols: string[];
  neuralPipeline: string;
  neuralIdCol: string;
  graphSourceCol: string;
  graphTargetCol: string;
  graphWeightCol: string;
}) {
  const needsGraphTopology = graphNeuralPipelines.has(neuralPipeline);
  const graphTopology = needsGraphTopology
    ? buildGraphTopology(table, graphSourceCol, graphTargetCol, featureCols)
    : null;
  const rows = sampleModelRows(
    table.rows
      .map((row, rowIndex) => ({
        id: neuralIdCol ? parseBrowserId(row[neuralIdCol]) : undefined,
        source: graphSourceCol && graphTopology ? graphTopology.nodeIndex.get(row[graphSourceCol]) : undefined,
        targetNode: graphTargetCol && graphTopology ? graphTopology.nodeIndex.get(row[graphTargetCol]) : undefined,
        edgeWeight: graphWeightCol ? Number(row[graphWeightCol]) : undefined,
        edgeType: graphTopology?.edgeTypesByRow[rowIndex] ?? 0,
        dense: featureCols.map((column) => Number(row[column])),
        target: Number(row[targetCol]),
      }))
      .filter((row) => {
        const validCommon = Number.isFinite(row.target) && row.dense.every(Number.isFinite);
        if (!validCommon) {
          return false;
        }
        if (graphNeuralPipelines.has(neuralPipeline)) {
          return row.source !== undefined && row.targetNode !== undefined && (row.edgeWeight === undefined || Number.isFinite(row.edgeWeight));
        }
        return row.id !== undefined;
      }),
    VISUALIZED_NEURAL_MAX_ROWS,
  );
  if (rows.length < 4) {
    throw new Error('CartoBoost neural modeling needs at least four usable rows.');
  }
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  return wasmModule.runNeuralModel({
    rows,
    denseFeatureNames: featureCols,
    nodeFeatures: graphTopology?.nodeFeatures ?? [],
    nodeTypes: graphTopology?.nodeTypes ?? [],
    edgeTypeTriples: graphTopology?.edgeTypeTriples ?? [],
    pipeline: neuralPipeline,
    options: {
      holdoutFraction: 0.2,
      embeddingDim: 8,
      randomState: 42,
      nEstimators: 20,
      learningRate: 0.07,
      maxDepth: 4,
      minSamplesLeaf: Math.max(2, Math.min(20, Math.floor(rows.length / 20))),
      node2vecWalkLength: 8,
      node2vecWalksPerNode: 4,
      node2vecWindowSize: 3,
      node2vecEpochs: 2,
      node2vecSeed: 42,
      graphSageEpochs: 3,
      graphSageNegativeSamples: 2,
      graphSageSeed: 42,
      includeModelVisualization: true,
    },
  });
}

function parseBrowserId(value: string | undefined) {
  if (!value) {
    return undefined;
  }
  const numeric = Number(value);
  if (Number.isInteger(numeric) && numeric >= 0) {
    return numeric;
  }
  return stableStringId(value);
}

function buildNodeIndex(table: ParsedTable, columns: string[]) {
  const index = new Map<string, number>();
  for (const column of columns.filter(Boolean)) {
    for (const row of table.rows) {
      const value = row[column];
      if (value && !index.has(value)) {
        index.set(value, index.size);
      }
    }
  }
  return index;
}

function buildGraphTopology(table: ParsedTable, sourceCol: string, targetCol: string, featureCols: string[]) {
  const nodeIndex = buildNodeIndex(table, [sourceCol, targetCol]);
  const width = Math.max(1, featureCols.length);
  const sums = Array.from({length: nodeIndex.size}, () => Array.from({length: width}, () => 0));
  const counts = Array.from({length: nodeIndex.size}, () => 0);
  const sourceNodes = new Set<number>();
  const targetNodes = new Set<number>();
  for (const row of table.rows) {
    const dense = featureCols.map((column) => Number(row[column]));
    const values = dense.length > 0 ? dense : [1];
    if (!values.every(Number.isFinite)) {
      continue;
    }
    const source = sourceCol ? nodeIndex.get(row[sourceCol]) : undefined;
    const target = targetCol ? nodeIndex.get(row[targetCol]) : undefined;
    for (const node of [source, target]) {
      if (node === undefined) {
        continue;
      }
      counts[node] += 1;
      for (const [index, value] of values.entries()) {
        sums[node][index] += value;
      }
    }
    if (source !== undefined) {
      sourceNodes.add(source);
    }
    if (target !== undefined) {
      targetNodes.add(target);
    }
  }
  const nodeFeatures = sums.map((row, index) => {
    const count = counts[index] || 1;
    return row.map((value) => value / count);
  });
  const nodeTypes = Array.from({length: nodeIndex.size}, (_, index) => (targetNodes.has(index) && !sourceNodes.has(index) ? 1 : 0));
  const relationIndex = new Map<string, number>();
  const edgeTypeTriples: number[][] = [];
  const edgeTypesByRow = table.rows.map((row) => {
    const source = sourceCol ? nodeIndex.get(row[sourceCol]) : undefined;
    const target = targetCol ? nodeIndex.get(row[targetCol]) : undefined;
    const sourceType = source === undefined ? 0 : nodeTypes[source] ?? 0;
    const targetType = target === undefined ? 0 : nodeTypes[target] ?? 0;
    const key = `${sourceType}->${targetType}`;
    const existing = relationIndex.get(key);
    if (existing !== undefined) {
      return existing;
    }
    const next = relationIndex.size;
    relationIndex.set(key, next);
    edgeTypeTriples.push([sourceType, next, targetType]);
    return next;
  });
  return {
    nodeIndex,
    nodeFeatures,
    nodeTypes,
    edgeTypesByRow,
    edgeTypeTriples: edgeTypeTriples.length > 0 ? edgeTypeTriples : [[0, 0, 0]],
  };
}

function stableStringId(value: string) {
  let hash = 2166136261;
  for (const char of value.trim()) {
    hash ^= char.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash;
}

type ActualRecord = {
  series_id: string;
  timestamp: string;
  index: number;
  value: number;
};

type ChartSeries = {
  label: string;
  points: {index: number; value: unknown}[];
  /** Index at which forecast values begin, before the optional anchor point. */
  forecastStartIndex?: number;
};

function actualRowsForFirstSeries(
  table: ParsedTable | null,
  timestampCol: string,
  targetCol: string,
  seriesCol: string,
): ActualRecord[] {
  if (!table || !table.columns.includes(timestampCol) || !table.columns.includes(targetCol)) {
    return [];
  }
  const firstSeries = seriesCol ? table.rows.find((row) => row[seriesCol] !== undefined)?.[seriesCol] : undefined;
  const rows = table.rows
    .filter((row) => !seriesCol || row[seriesCol] === firstSeries)
    .map((row, index) => ({
      // The WASM frame uses an empty id when no series column was supplied.
      // Keep the browser-side id canonical so comparison traces can align.
      series_id: seriesCol ? row[seriesCol] ?? '' : '',
      timestamp: row[timestampCol],
      index,
      value: Number(row[targetCol]),
    }))
    .filter((row) => Number.isFinite(row.value));
  return rows
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp))
    .map((row, index) => ({...row, index}));
}

function firstSeriesForecastRows(response: ForecastResponse, seriesId?: string) {
  const firstSeries = seriesId ?? response.forecast.records[0]?.series_id;
  return response.forecast.records
    .filter((row) => firstSeries !== undefined && row.series_id === firstSeries && Number.isFinite(row.prediction))
    .sort((left, right) => left.horizon - right.horizon || left.timestamp.localeCompare(right.timestamp));
}

function firstSeriesComponentRows(response: ForecastResponse) {
  const firstSeries = response.forecast.records[0]?.series_id ?? response.components?.records[0]?.series_id;
  return (response.components?.records ?? []).filter((row) => row.series_id === firstSeries);
}

function firstSeriesHistoryComponentRows(response: ForecastResponse) {
  const firstSeries = response.forecast.records[0]?.series_id ?? response.historyComponents?.records[0]?.series_id;
  return (response.historyComponents?.records ?? []).filter((row) => row.series_id === firstSeries);
}

function firstSeriesQuantileRows(response: ForecastResponse) {
  const firstSeries = response.forecast.records[0]?.series_id;
  return (response.quantiles?.records ?? []).filter((row) => row.series_id === firstSeries);
}

function quantileForecastSeries(rows: ForecastQuantileRecord[]) {
  const levels = [...new Set(rows.map((row) => coerceFiniteNumber(row.quantile)).filter((level): level is number => level !== null))].sort((a, b) => a - b);
  return levels.map((level) => ({
    label: `q${formatFixed(level, 2)}`,
    records: rows
      .filter((row) => coerceFiniteNumber(row.quantile) === level)
      .map((row) => ({
        series_id: row.series_id,
        timestamp: row.timestamp,
        horizon: row.horizon,
        model: `q${formatFixed(level, 2)}`,
        prediction: row.prediction,
      })),
  }));
}

function buildLineSeries(
  actualRows: ActualRecord[],
  forecastSeries: {label: string; records: ForecastRecord[]}[],
): ChartSeries[] {
  const actualLength = actualRows.length;
  const lastActual = actualRows[actualRows.length - 1];
  const actualIndexByTimestamp = new Map(
    actualRows.map((row) => [normalizeTimestamp(row.timestamp), row.index]),
  );
  return [
    {
      label: 'Actual',
      points: actualRows.map((row) => ({index: row.index, value: row.value})),
    },
    ...forecastSeries.map((series) => {
      const records = [...series.records]
        .filter((row) => Number.isFinite(row.prediction))
        .sort((left, right) => left.horizon - right.horizon || left.timestamp.localeCompare(right.timestamp));
      const firstMatchedIndex = records.length > 0
        ? actualIndexByTimestamp.get(normalizeTimestamp(records[0].timestamp))
        : undefined;
      const forecastStartIndex = firstMatchedIndex ?? actualLength;
      const anchor = records.length > 0
        ? actualRows.find((row) => row.index === forecastStartIndex - 1) ?? (
            firstMatchedIndex === undefined ? lastActual : undefined
          )
        : undefined;
      return {
        label: series.label,
        forecastStartIndex,
        points: [
          ...(anchor ? [{index: anchor.index, value: anchor.value}] : []),
          ...records.map((row, index) => ({
            // Holdout forecasts share timestamps with the actual trace; a
            // regular future forecast starts immediately after its last point.
            index: actualIndexByTimestamp.get(normalizeTimestamp(row.timestamp))
              ?? (firstMatchedIndex === undefined ? actualLength + index : forecastStartIndex + index),
            value: row.prediction,
          })),
        ],
      };
    }),
  ];
}

function recentActualRowsForForecastChart(actualRows: ActualRecord[], forecastLength: number): ActualRecord[] {
  const windowSize = Math.min(
    actualRows.length,
    Math.max(72, forecastLength * 6),
  );
  const offset = Math.max(0, actualRows.length - windowSize);
  return actualRows.slice(offset).map((row, index) => ({...row, index}));
}

function chartColor(index: number) {
  return ['#526274', '#168f86', '#ffb454', '#4f8cff', '#d84d7f', '#8b6cff', '#2f9d62', '#a15c38', '#65c9ff'][index % 9];
}

function hasCoordinateColumns(columns: string[]) {
  return coordinatePair(columns, ['pickup', 'pu', 'origin', '']) !== null;
}

type HoldoutActual = {
  series_id: string;
  timestamp: string;
  value: number;
};

function holdoutSplit(
  table: ParsedTable,
  timestampCol: string,
  targetCol: string,
  seriesCol: string,
  horizon: number,
) {
  const groups = new Map<string, Record<string, string>[]>();
  for (const row of table.rows) {
    const seriesId = seriesCol ? row[seriesCol] : '__single__';
    const rows = groups.get(seriesId) ?? [];
    rows.push(row);
    groups.set(seriesId, rows);
  }
  const trainRows: Record<string, string>[] = [];
  const actuals: HoldoutActual[] = [];
  for (const [seriesId, rows] of groups) {
    if (rows.length <= horizon + 2) {
      throw new Error(`Series ${seriesId} needs more than ${horizon + 2} rows for a ${horizon}-step holdout backtest.`);
    }
    const sortedRows = [...rows].sort((a, b) => a[timestampCol].localeCompare(b[timestampCol]));
    const cutoff = sortedRows.length - horizon;
    trainRows.push(...sortedRows.slice(0, cutoff));
    actuals.push(
      ...sortedRows.slice(cutoff).map((row) => ({
        series_id: seriesId,
        timestamp: normalizeTimestamp(row[timestampCol]),
        value: Number(row[targetCol]),
      })),
    );
  }
  return {
    train: {
      ...table,
      fileName: `${table.fileName} holdout train`,
      rows: trainRows,
    },
    actuals: actuals.filter((actual) => Number.isFinite(actual.value)),
  };
}

function evaluateHoldout(predictions: ForecastRecord[], actuals: HoldoutActual[]) {
  const actualByKey = new Map(actuals.map((actual) => [`${actual.series_id}\u0000${actual.timestamp}`, actual]));
  const pairs = predictions
    .map((prediction) => ({
      prediction,
      actual: actualByKey.get(`${prediction.series_id}\u0000${normalizeTimestamp(prediction.timestamp)}`),
    }))
    .filter((pair): pair is {prediction: ForecastRecord; actual: HoldoutActual} => pair.actual !== undefined);
  if (pairs.length === 0) {
    throw new Error('No forecast rows aligned with the holdout timestamps.');
  }
  const absErrors = pairs.map((pair) => Math.abs(pair.prediction.prediction - pair.actual.value));
  const squaredErrors = pairs.map((pair) => (pair.prediction.prediction - pair.actual.value) ** 2);
  const actualMagnitude = pairs.reduce((sum, pair) => sum + Math.abs(pair.actual.value), 0);
  return {
    rmse: Math.sqrt(squaredErrors.reduce((sum, value) => sum + value, 0) / pairs.length),
    mae: absErrors.reduce((sum, value) => sum + value, 0) / pairs.length,
    wape: actualMagnitude === 0 ? 0 : absErrors.reduce((sum, value) => sum + value, 0) / actualMagnitude,
    comparedRows: pairs.length,
  };
}

function normalizeTimestamp(value: string) {
  return value.length === 10 ? `${value}T00:00:00` : value.replace(' ', 'T').replace(/\.\d{3}Z$/, '');
}

function backtestConcurrency() {
  const availableWorkers = typeof navigator === 'undefined' ? MAX_PARALLEL_BACKTEST_WORKERS : Math.max(1, (navigator.hardwareConcurrency ?? 2) - 1);
  return Math.max(1, Math.min(MAX_PARALLEL_BACKTEST_WORKERS, availableWorkers));
}

async function runWithConcurrency<T>(
  items: T[],
  concurrency: number,
  runner: (item: T, index: number) => Promise<void>,
) {
  let nextIndex = 0;
  const workerCount = Math.min(Math.max(1, concurrency), items.length);
  await Promise.all(
    Array.from({length: workerCount}, async () => {
      while (nextIndex < items.length) {
        const index = nextIndex;
        nextIndex += 1;
        await runner(items[index], index);
      }
    }),
  );
}

async function waitForBrowserPaint() {
  await new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => window.setTimeout(resolve, 0));
  });
}

function formatElapsedMs(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}s`;
  }
  return `${Math.max(1, Math.round(value)).toLocaleString()}ms`;
}

function parseDelimited(text: string, fileName: string): ParsedTable {
  const delimiter = text.includes('\t') ? '\t' : ',';
  const rows = parseRows(text.trim(), delimiter);
  if (rows.length < 2) {
    throw new Error('The uploaded file must include a header row and at least one data row.');
  }
  const columns = rows[0].map((column) => column.trim());
  return {
    columns,
    rows: rows.slice(1).filter((row) => row.some(Boolean)).map((row) => Object.fromEntries(columns.map((column, index) => [column, row[index] ?? '']))),
    fileName,
  };
}

async function parseParquet(file: File): Promise<ParsedTable> {
  return parseParquetBuffer(await file.arrayBuffer(), file.name);
}

async function parseParquetBuffer(buffer: ArrayBuffer, fileName: string): Promise<ParsedTable> {
  const [{tableFromIPC}, parquet] = await Promise.all([
    import('apache-arrow'),
    import('parquet-wasm') as Promise<ParquetWasmModule>,
  ]);
  await parquet.default();
  const wasmTable = parquet.readParquet(new Uint8Array(buffer));
  try {
    const arrowTable = tableFromIPC(wasmTable.intoIPCStream()) as unknown as {
      numRows: number;
      schema: {fields: {name: string}[]};
      getChild: (name: string) => {get: (index: number) => unknown} | null;
      getChildAt: (index: number) => {get: (rowIndex: number) => unknown} | null;
    };
    const columns = arrowTable.schema.fields.map((field) => field.name);
    const vectors = columns.map((column, index) => arrowTable.getChild(column) ?? arrowTable.getChildAt(index));
    const rows = Array.from({length: arrowTable.numRows}, (_, rowIndex) =>
      Object.fromEntries(
        columns.map((column, columnIndex) => [
          column,
          formatCellValue(vectors[columnIndex]?.get(rowIndex), column),
        ]),
      ),
    );
    if (columns.length === 0 || rows.length === 0) {
      throw new Error('The uploaded Parquet file must include at least one column and one row.');
    }
    return {columns, rows, fileName};
  } finally {
    try {
      wasmTable.free?.();
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes('null pointer')) {
        throw error;
      }
    }
  }
}

function downsampleTable(table: ParsedTable, maxRows: number): ParsedTable {
  if (table.rows.length <= maxRows) {
    return table;
  }
  const step = table.rows.length / maxRows;
  const rows = Array.from({length: maxRows}, (_, index) => table.rows[Math.floor(index * step)]).filter(
    (row): row is Record<string, string> => row !== undefined,
  );
  return {
    ...table,
    rows,
    fileName: `${stripExtension(table.fileName)}-sample-${rows.length.toLocaleString()}.parquet`,
  };
}

function sampleModelRows<T>(rows: T[], maxRows: number): T[] {
  if (rows.length <= maxRows) {
    return rows;
  }
  const step = rows.length / maxRows;
  return Array.from({length: maxRows}, (_, index) => rows[Math.floor(index * step)]).filter((row): row is T => row !== undefined);
}

function buildTaxiRouteHourSampleTable(
  table: ParsedTable,
  maxRows: number,
  options: {balancedManhattanRoutes?: boolean; fileName?: string} = {},
): ParsedTable {
  const requiredColumns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID'];
  if (!requiredColumns.every((column) => table.columns.includes(column))) {
    const enriched = enrichTaxiRouteGeoColumns(table);
    return downsampleTable(
      {
        ...enriched,
        rows: options.balancedManhattanRoutes ? balancedManhattanRouteRows(enriched.rows, maxRows) : enriched.rows,
        fileName: options.fileName ?? enriched.fileName,
      },
      maxRows,
    );
  }
  const groups = new Map<
    string,
    {
      timestamp: string;
      pickup: string;
      dropoff: string;
      count: number;
      totalAmount: number;
      fareAmount: number;
      tripDistance: number;
      durationMinutes: number;
    }
  >();
  for (const row of table.rows) {
    const pickupTime = new Date(row.tpep_pickup_datetime);
    const dropoffTime = new Date(row.tpep_dropoff_datetime);
    const pickup = row.PULocationID;
    const dropoff = row.DOLocationID;
    if (Number.isNaN(pickupTime.getTime()) || Number.isNaN(dropoffTime.getTime()) || pickup === '' || dropoff === '') {
      continue;
    }
    pickupTime.setMinutes(0, 0, 0);
    const timestamp = pickupTime.toISOString().slice(0, 19);
    const key = `${timestamp}|${pickup}|${dropoff}`;
    const group =
      groups.get(key) ??
      {
        timestamp,
        pickup,
        dropoff,
        count: 0,
        totalAmount: 0,
        fareAmount: 0,
        tripDistance: 0,
        durationMinutes: 0,
      };
    group.count += 1;
    group.totalAmount += coerceFiniteNumber(row.total_amount) ?? 0;
    group.fareAmount += coerceFiniteNumber(row.fare_amount) ?? 0;
    group.tripDistance += coerceFiniteNumber(row.trip_distance) ?? 0;
    group.durationMinutes += Math.max(0, (dropoffTime.getTime() - new Date(row.tpep_pickup_datetime).getTime()) / 60000);
    groups.set(key, group);
  }
  const rows = Array.from(groups.values())
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp) || left.pickup.localeCompare(right.pickup) || left.dropoff.localeCompare(right.dropoff))
    .map((group) => {
      const pickupCentroid = TAXI_ZONE_CENTROIDS[group.pickup];
      const dropoffCentroid = TAXI_ZONE_CENTROIDS[group.dropoff];
      return {
        timestamp: group.timestamp,
        series_id: `${group.pickup}-${group.dropoff}`,
        PULocationID: group.pickup,
        DOLocationID: group.dropoff,
        pickup_latitude: formatCoordinate(pickupCentroid?.lat),
        pickup_longitude: formatCoordinate(pickupCentroid?.lon),
        dropoff_latitude: formatCoordinate(dropoffCentroid?.lat),
        dropoff_longitude: formatCoordinate(dropoffCentroid?.lon),
        target: String(group.count),
        trip_count: String(group.count),
        avg_total_amount: formatFixed(group.totalAmount / group.count, 2),
        avg_fare_amount: formatFixed(group.fareAmount / group.count, 2),
        avg_trip_distance: formatFixed(group.tripDistance / group.count, 2),
        avg_duration_minutes: formatFixed(group.durationMinutes / group.count, 2),
      };
    });
  const selectedRows = options.balancedManhattanRoutes ? balancedManhattanRouteRows(rows, maxRows) : rows;
  return downsampleTable(enrichTaxiRouteGeoColumns({
    columns: [
      'timestamp',
      'series_id',
      'PULocationID',
      'DOLocationID',
      'target',
      'trip_count',
      'avg_total_amount',
      'avg_fare_amount',
      'avg_trip_distance',
      'avg_duration_minutes',
    ],
    rows: selectedRows,
    fileName: options.fileName ?? table.fileName,
  }), maxRows);
}

function balancedManhattanRouteRows(rows: Record<string, string>[], maxRows: number) {
  const routeGroups = new Map<string, Record<string, string>[]>();
  for (const row of rows) {
    if (!isManhattanRoute(row)) {
      continue;
    }
    const route = row.series_id;
    const group = routeGroups.get(route) ?? [];
    group.push(row);
    routeGroups.set(route, group);
  }
  const groups = Array.from(routeGroups.entries())
    .filter(([, group]) => group.length > 0)
    .sort((left, right) => {
      const leftScore = routeCoverageScore(left[1]);
      const rightScore = routeCoverageScore(right[1]);
      return rightScore - leftScore || left[0].localeCompare(right[0]);
    });
  if (groups.length < 2) {
    return rows;
  }
  const targetPerRoute = Math.max(1, Math.floor(maxRows / groups.length));
  const selected = groups.flatMap(([, group]) => evenlySampleRows(group, targetPerRoute));
  let cursor = 0;
  while (selected.length < maxRows && groups.some(([, group]) => group.length > targetPerRoute)) {
    const group = groups[cursor % groups.length][1];
    const row = group[targetPerRoute + Math.floor(cursor / groups.length)];
    if (row) {
      selected.push(row);
    }
    cursor += 1;
    if (cursor > rows.length + groups.length) {
      break;
    }
  }
  return selected
    .slice(0, maxRows)
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp) || left.series_id.localeCompare(right.series_id));
}

function isManhattanRoute(row: Record<string, string>) {
  const pickup = TAXI_ZONE_CENTROIDS[row.PULocationID ?? ''];
  const dropoff = TAXI_ZONE_CENTROIDS[row.DOLocationID ?? ''];
  return Boolean(pickup && dropoff && isManhattanCentroid(pickup) && isManhattanCentroid(dropoff) && row.PULocationID !== row.DOLocationID);
}

function isManhattanCentroid(point: {lat: number; lon: number}) {
  return point.lat >= 40.70 && point.lat <= 40.83 && point.lon >= -74.02 && point.lon <= -73.93;
}

function routeCoverageScore(rows: Record<string, string>[]) {
  if (rows.length === 0) {
    return 0;
  }
  const pickup = TAXI_ZONE_CENTROIDS[rows[0].PULocationID ?? ''];
  const dropoff = TAXI_ZONE_CENTROIDS[rows[0].DOLocationID ?? ''];
  const distance = pickup && dropoff ? Math.hypot(pickup.lat - dropoff.lat, pickup.lon - dropoff.lon) : 0;
  return rows.length + distance * 1000;
}

function evenlySampleRows<T>(rows: T[], maxRows: number): T[] {
  if (rows.length <= maxRows) {
    return rows;
  }
  const step = rows.length / maxRows;
  return Array.from({length: maxRows}, (_, index) => rows[Math.floor(index * step)]).filter((row): row is T => row !== undefined);
}

function enrichTaxiRouteGeoColumns(table: ParsedTable): ParsedTable {
  if (!table.columns.includes('PULocationID') || !table.columns.includes('DOLocationID')) {
    return table;
  }
  const geoColumns = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'];
  const columns = [
    ...table.columns.filter((column) => !geoColumns.includes(column)),
    ...geoColumns,
  ];
  const rows = table.rows.map((row) => {
    const pickupCentroid = TAXI_ZONE_CENTROIDS[row.PULocationID ?? ''];
    const dropoffCentroid = TAXI_ZONE_CENTROIDS[row.DOLocationID ?? ''];
    return {
      ...row,
      pickup_latitude: formatCoordinate(pickupCentroid?.lat) || row.pickup_latitude || '',
      pickup_longitude: formatCoordinate(pickupCentroid?.lon) || row.pickup_longitude || '',
      dropoff_latitude: formatCoordinate(dropoffCentroid?.lat) || row.dropoff_latitude || '',
      dropoff_longitude: formatCoordinate(dropoffCentroid?.lon) || row.dropoff_longitude || '',
    };
  });
  return {
    ...table,
    columns,
    rows,
  };
}

function formatCoordinate(value: number | undefined) {
  return Number.isFinite(value) ? formatFixed(value, 6) : '';
}

function formatCellValue(value: unknown, columnName = '') {
  if (value == null) {
    return '';
  }
  if (value instanceof Date) {
    return value.toISOString().replace(/\.\d{3}Z$/, '');
  }
  if (typeof value === 'number' && Number.isFinite(value) && isTimestampColumn(columnName) && value > 1000000000000) {
    return new Date(value).toISOString().replace(/\.\d{3}Z$/, '');
  }
  if (typeof value === 'bigint') {
    return value.toString();
  }
  if (ArrayBuffer.isView(value)) {
    return Array.from(value as Uint8Array).join(',');
  }
  return String(value);
}

function isTimestampColumn(columnName: string) {
  const normalized = columnName.toLowerCase();
  return normalized.includes('datetime') || normalized.includes('timestamp') || normalized === 'date' || normalized === 'ds';
}

function parseRows(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === delimiter && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && next === '\n') {
        index += 1;
      }
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += char;
    }
  }
  row.push(cell);
  rows.push(row);
  return rows;
}
