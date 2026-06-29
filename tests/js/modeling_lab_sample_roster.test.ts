import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import test from 'node:test';

import {tableFromIPC} from 'apache-arrow';
import initParquet, {readParquet} from 'parquet-wasm/esm';

import initCartoboost, {
  availableForecastModels,
  runForecast,
} from '../../static/wasm/cartoboost/cartoboost_wasm.js';

type ArrowRow = Record<string, unknown>;

type ForecastRecord = {
  series_id: string;
  timestamp: string;
  horizon: number;
  prediction: number;
};

function scalar(value: unknown): unknown {
  return typeof value === 'bigint' ? Number(value) : value;
}

function numeric(value: unknown): number | null {
  const parsed = Number(scalar(value));
  return Number.isFinite(parsed) ? parsed : null;
}

const shippedSamples = [
  'static/samples/yellow_taxi_2024-01-single-lane-5000.parquet',
  'static/samples/yellow_taxi_2024-01-varied-routes-2500.parquet',
] as const;

const manhattanBounds = {
  minLat: 40.70,
  maxLat: 40.83,
  minLon: -74.02,
  maxLon: -73.93,
};

test(
  'shipped taxi samples return finite records for every registered forecast model without fallback',
  {timeout: 360_000},
  async () => {
    await initParquet({
      module_or_path: await readFile('node_modules/parquet-wasm/esm/parquet_wasm_bg.wasm'),
    });
    await initCartoboost({
      module_or_path: await readFile('static/wasm/cartoboost/cartoboost_wasm_bg.wasm'),
    });

    const options = {
      seasonLength: 24,
      nEstimators: 4,
      maxDepth: 2,
      minSamplesLeaf: 16,
      lags: [1, 24],
      rollingMeanWindows: [24],
      calendarFeatures: ['day_of_week'],
      maxDirectHorizon: 14,
      coordinateX: 'pickup_longitude',
      coordinateY: 'pickup_latitude',
      windowSize: 24,
      windowCount: 3,
      validationWindow: 24,
      maxP: 1,
      maxD: 1,
      maxQ: 1,
      mstlSeasonLengths: [24],
    };

    for (const samplePath of shippedSamples) {
      const parquet = await readFile(samplePath);
      const arrowTable = tableFromIPC(readParquet(new Uint8Array(parquet)).intoIPCStream());
      const rows = Array.from({length: arrowTable.numRows}, (_, rowIndex) => {
        const source = arrowTable.get(rowIndex)?.toJSON() as ArrowRow | undefined;
        assert.ok(source, `${samplePath} row ${rowIndex} should exist`);
        const covariates: Record<string, number> = {};
        for (const [key, value] of Object.entries(source)) {
          if (['timestamp', 'target', 'series_id'].includes(key)) {
            continue;
          }
          const parsed = numeric(value);
          if (parsed !== null) {
            covariates[key] = parsed;
          }
        }
        return {
          timestamp: String(source.timestamp),
          target: Number(scalar(source.target)),
          seriesId: String(source.series_id),
          covariates,
        };
      });

      const expectedRecords = 14 * new Set(rows.map((row) => row.seriesId)).size;
      for (const model of availableForecastModels() as Array<{name: string}>) {
        const started = performance.now();
        const autoOptions = model.name === 'auto_forecast' ? {maxAutoCandidateCount: 4} : {};
        const response = runWithTimeout(
          () =>
            runForecast({
              rows,
              frequency: 'hourly',
              horizon: 14,
              model: model.name,
              options: {...options, ...autoOptions},
              metadata: {
                timestampCol: 'timestamp',
                targetCol: 'target',
                seriesIdCol: 'series_id',
              },
            }) as {metadata: {warning?: unknown}; forecast: {records: ForecastRecord[]}},
          30_000,
          `${samplePath} ${model.name}`,
        );
        const elapsedMs = Math.round(performance.now() - started);
        console.log(`${samplePath} ${model.name} ${elapsedMs}ms ${response.forecast.records.length} records`);

        assert.equal(
          response.metadata.warning,
          undefined,
          `${model.name} should fit ${samplePath} directly without browser fallback`,
        );
        assert.equal(response.forecast.records.length, expectedRecords, `${model.name} ${samplePath} record count`);
        for (const [index, record] of response.forecast.records.entries()) {
          assert.ok(record.series_id, `${model.name} ${samplePath} record ${index} series_id`);
          assert.ok(record.timestamp, `${model.name} ${samplePath} record ${index} timestamp`);
          assert.ok(record.horizon > 0, `${model.name} ${samplePath} record ${index} horizon`);
          assert.ok(
            Number.isFinite(record.prediction),
            `${model.name} ${samplePath} record ${index} finite prediction`,
          );
        }
      }
    }
  },
);

test('varied-route taxi sample includes multiple Manhattan pickup/dropoff lanes', async () => {
  await initParquet({
    module_or_path: await readFile('node_modules/parquet-wasm/esm/parquet_wasm_bg.wasm'),
  });
  const parquet = await readFile('static/samples/yellow_taxi_2024-01-varied-routes-2500.parquet');
  const arrowTable = tableFromIPC(readParquet(new Uint8Array(parquet)).intoIPCStream());
  const laneIds = new Set<string>();
  const pickupIds = new Set<string>();
  const dropoffIds = new Set<string>();
  const manhattanLaneIds = new Set<string>();
  const manhattanPickupIds = new Set<string>();
  const manhattanDropoffIds = new Set<string>();

  for (let rowIndex = 0; rowIndex < arrowTable.numRows; rowIndex += 1) {
    const row = arrowTable.get(rowIndex)?.toJSON() as ArrowRow | undefined;
    assert.ok(row, `varied-route row ${rowIndex} should exist`);
    const pickup = String(scalar(row.PULocationID));
    const dropoff = String(scalar(row.DOLocationID));
    const lane = `${pickup}-${dropoff}`;
    laneIds.add(lane);
    pickupIds.add(pickup);
    dropoffIds.add(dropoff);

    const pickupLat = numeric(row.pickup_latitude);
    const pickupLon = numeric(row.pickup_longitude);
    const dropoffLat = numeric(row.dropoff_latitude);
    const dropoffLon = numeric(row.dropoff_longitude);
    if (
      pickupLat !== null &&
      pickupLon !== null &&
      dropoffLat !== null &&
      dropoffLon !== null &&
      isManhattanCoordinate(pickupLat, pickupLon) &&
      isManhattanCoordinate(dropoffLat, dropoffLon)
    ) {
      manhattanLaneIds.add(lane);
      manhattanPickupIds.add(pickup);
      manhattanDropoffIds.add(dropoff);
    }
  }

  assert.ok(laneIds.size >= 5, `expected at least 5 OD lanes, found ${laneIds.size}`);
  assert.ok(pickupIds.size >= 5, `expected at least 5 pickup zones, found ${pickupIds.size}`);
  assert.ok(dropoffIds.size >= 5, `expected at least 5 dropoff zones, found ${dropoffIds.size}`);
  assert.ok(manhattanLaneIds.size >= 3, `expected at least 3 Manhattan OD lanes, found ${manhattanLaneIds.size}`);
  assert.ok(manhattanPickupIds.size >= 3, `expected at least 3 Manhattan pickup zones, found ${manhattanPickupIds.size}`);
  assert.ok(manhattanDropoffIds.size >= 3, `expected at least 3 Manhattan dropoff zones, found ${manhattanDropoffIds.size}`);
});

test('wasm bundle can initialize when served below the Docusaurus base path', {timeout: 30_000}, async () => {
  const server = createServer(async (request, response) => {
    const url = request.url ?? '';
    if (url === '/CartoBoost/wasm/cartoboost/cartoboost_wasm.js') {
      response.setHeader('content-type', 'text/javascript');
      response.end(await readFile('static/wasm/cartoboost/cartoboost_wasm.js', 'utf8'));
      return;
    }
    if (url === '/CartoBoost/wasm/cartoboost/cartoboost_wasm_bg.wasm') {
      response.setHeader('content-type', 'application/wasm');
      response.end(await readFile('static/wasm/cartoboost/cartoboost_wasm_bg.wasm'));
      return;
    }
    response.statusCode = 404;
    response.end('not found');
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    const address = server.address();
    assert.ok(address && typeof address === 'object');
    const baseUrl = `http://127.0.0.1:${address.port}/CartoBoost/wasm/cartoboost`;
    const moduleResponse = await fetch(`${baseUrl}/cartoboost_wasm.js`);
    assert.equal(moduleResponse.ok, true);
    const moduleText = await moduleResponse.text();
    const moduleUrl = `data:text/javascript;base64,${Buffer.from(moduleText).toString('base64')}`;
    const wasmModule = (await import(moduleUrl)) as {
      default: (input: {module_or_path: string}) => Promise<unknown>;
      availableForecastModels: () => Array<{name: string}>;
    };
    await wasmModule.default({module_or_path: `${baseUrl}/cartoboost_wasm_bg.wasm`});
    assert.ok(wasmModule.availableForecastModels().some((model) => model.name === 'auto_forecast'));
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
});

function isManhattanCoordinate(lat: number, lon: number): boolean {
  return (
    lat >= manhattanBounds.minLat &&
    lat <= manhattanBounds.maxLat &&
    lon >= manhattanBounds.minLon &&
    lon <= manhattanBounds.maxLon
  );
}

function runWithTimeout<T>(fn: () => T, timeoutMs: number, label: string): T {
  const started = performance.now();
  const result = fn();
  const elapsedMs = performance.now() - started;
  assert.ok(
    elapsedMs <= timeoutMs,
    `${label} exceeded ${timeoutMs}ms browser roster freeze guard (${Math.round(elapsedMs)}ms)`,
  );
  return result;
}
