import assert from 'node:assert/strict';
import test from 'node:test';

import {assertForecastResponseRecords, coerceFiniteNumber, formatFixed, formatPercent} from '../../src/components/ModelingLabClient/numberFormat';

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
