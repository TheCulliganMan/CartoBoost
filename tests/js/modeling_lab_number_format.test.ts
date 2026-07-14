import assert from 'node:assert/strict';
import test from 'node:test';

import {
  assertForecastResponseRecords,
  coerceFiniteNumber,
  forecastCellDiagnostics,
  formatFixed,
  formatPercent,
} from '../../src/components/ModelingLabClient/numberFormat';

test('coerceFiniteNumber accepts finite numbers and numeric strings', () => {
  assert.equal(coerceFiniteNumber(12.5), 12.5);
  assert.equal(coerceFiniteNumber('12.5'), 12.5);
  assert.equal(coerceFiniteNumber(' 0 '), 0);
});

test('coerceFiniteNumber rejects non-finite and non-numeric values', () => {
  assert.equal(coerceFiniteNumber(Number.NaN), null);
  assert.equal(coerceFiniteNumber(Number.POSITIVE_INFINITY), null);
  assert.equal(coerceFiniteNumber(''), null);
  assert.equal(coerceFiniteNumber('not a number'), null);
  assert.equal(coerceFiniteNumber(null), null);
});

test('formatters do not call toFixed on non-number payloads', () => {
  assert.equal(formatFixed('424.1114', 3), '424.111');
  assert.equal(formatFixed(undefined, 3), '-');
  assert.equal(formatFixed('bad-payload', 3), '-');
  assert.equal(formatPercent('0.1874', 1), '+18.7%');
  assert.equal(formatPercent('bad-payload', 1), '-');
});

test('LSTTN H3 diagnostics preserve signed native counterfactual effects', () => {
  assert.deepEqual(
    forecastCellDiagnostics({
      forecast: 120,
      actual: 114,
      latest: 100,
      isolated: 108,
      recentNeutral: 115,
      rhythmNeutral: 124,
    }),
    {volume: 120, rate: 20, error: 6, network: 12, recent: 5, rhythm: -4},
  );
  assert.equal(
    forecastCellDiagnostics({
      forecast: 8,
      actual: 7,
      latest: 0,
      isolated: 8,
      recentNeutral: 8,
      rhythmNeutral: 8,
    }).rate,
    0,
  );
  assert.throws(
    () => forecastCellDiagnostics({
      forecast: Number.NaN,
      actual: 7,
      latest: 6,
      isolated: 5,
      recentNeutral: 4,
      rhythmNeutral: 3,
    }),
    /finite native forecast values/,
  );
});

test('forecast responses must contain finite non-empty records', () => {
  assertForecastResponseRecords(
    {
      metadata: {
        model: 'naive',
        input: {
          n_rows: 10,
          is_panel: false,
          series_ids: ['__single__'],
          frequency: 'hourly',
        },
      },
      forecast: {
        records: [
          {
            series_id: '__single__',
            timestamp: '2024-01-01T01:00:00',
            horizon: 1,
            model: 'naive',
            prediction: 8.492,
          },
        ],
      },
    },
    'naive',
  );

  assert.throws(
    () =>
      assertForecastResponseRecords(
        {
          metadata: {
            model: 'ets',
            input: {
              n_rows: 12,
              is_panel: false,
              series_ids: ['__single__'],
              frequency: 'hourly',
            },
          },
          forecast: {
            records: [
              {
                series_id: '__single__',
                timestamp: '2024-01-01T01:00:00',
                horizon: 1,
                model: 'ets',
                prediction: 8,
              },
              {
                series_id: '__single__',
                timestamp: '2024-01-01T03:00:00',
                horizon: 3,
                model: 'ets',
                prediction: 9,
              },
            ],
          },
        },
        'ets',
      ),
    /non-contiguous horizons/,
  );

  assert.throws(
    () =>
      assertForecastResponseRecords(
        {
          metadata: {
            model: 'cartoboost_lag',
            input: {
              n_rows: 1750,
              is_panel: true,
              series_ids: ['PU132-DO236'],
              frequency: 'hourly',
            },
          },
          forecast: {
            records: [],
          },
        },
        'cartoboost_lag',
      ),
    /returned no forecast records/,
  );
});
