export function coerceFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed === '') {
      return null;
    }
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function formatFixed(value: unknown, digits = 3) {
  const numeric = coerceFiniteNumber(value);
  return numeric === null ? '-' : numeric.toFixed(digits);
}

export function formatPercent(value: unknown, digits = 1) {
  const numeric = coerceFiniteNumber(value);
  return numeric === null ? '-' : `${numeric >= 0 ? '+' : ''}${(numeric * 100).toFixed(digits)}%`;
}

export function assertForecastResponseRecords(response: unknown, requestedModel: string): void {
  const typedResponse = response as {
    metadata?: {input?: {series_ids?: unknown}};
    forecast?: {records?: unknown[]};
  } | null;
  const records = typedResponse?.forecast?.records;
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error(`${requestedModel} returned no forecast records.`);
  }
  const horizonsBySeries = new Map<string, number[]>();
  for (const [index, record] of records.entries()) {
    const row = record as {
      series_id?: unknown;
      timestamp?: unknown;
      horizon?: unknown;
      prediction?: unknown;
    };
    if (
      typeof row.series_id !== 'string' ||
      row.series_id.length === 0 ||
      typeof row.timestamp !== 'string' ||
      row.timestamp.length === 0 ||
      !Number.isInteger(row.horizon) ||
      Number(row.horizon) <= 0 ||
      coerceFiniteNumber(row.prediction) === null
    ) {
      throw new Error(`${requestedModel} returned an invalid forecast record at index ${index}.`);
    }
    const horizons = horizonsBySeries.get(row.series_id) ?? [];
    horizons.push(Number(row.horizon));
    horizonsBySeries.set(row.series_id, horizons);
  }
  const expectedSeries = Array.isArray(typedResponse?.metadata?.input?.series_ids)
    ? typedResponse.metadata.input.series_ids.filter((value): value is string => typeof value === 'string' && value.length > 0)
    : [];
  for (const seriesId of expectedSeries) {
    if (!horizonsBySeries.has(seriesId)) {
      throw new Error(`${requestedModel} returned no forecast records for series ${seriesId}.`);
    }
  }
  for (const [seriesId, horizons] of horizonsBySeries) {
    const sorted = [...horizons].sort((left, right) => left - right);
    for (let index = 0; index < sorted.length; index += 1) {
      if (sorted[index] !== index + 1) {
        throw new Error(`${requestedModel} returned non-contiguous horizons for series ${seriesId}.`);
      }
    }
  }
}
