/**
 * G1: resolveStatsUpdate's error-envelope guard. Live, GET
 * {API_PREFIX}/stats/dataset (and the SSE `snapshot`/`stats` frames that
 * share its shape) can return `{error: "..."}` instead of a real
 * DatasetStats body — trusting that blindly crashed the dashboard with
 * `Cannot read properties of undefined (reading
 * 'region_drain_total_unfinished')`.
 */
import { describe, expect, it } from 'vitest';
import {
  appendDrainSample,
  drainRateOf,
  etaSecondsOf,
  formatRate,
  fmtRelative,
  labeledRowsOf,
  resolveStatsUpdate,
  summarizeStatsError,
} from './datasetStats';
import type { DatasetStats } from '$lib/api_stats';
import { datasetStatsFixture } from './test/fixtures/datasetStats';

const GOOD: DatasetStats = datasetStatsFixture({
  total_crops: 422,
  validated: 297,
  test_holdout: 12,
  by_source: [{ key: 'tag_holdout_sample', doc_count: 300 }],
  labeled: { by_human: 100, by_vlm: 50, by_classifier: 10, other: 0 },
  in_progress: { region_drain_total_unfinished: 3, region_stall_reason: null },
});

describe('resolveStatsUpdate', () => {
  it('accepts a real DatasetStats payload and clears any prior error', () => {
    const result = resolveStatsUpdate(GOOD as unknown as Record<string, unknown>, null);
    expect(result.stats).toEqual(GOOD);
    expect(result.error).toBeNull();
  });

  it('keeps the last good stats and surfaces the message on an {error} payload', () => {
    const payload = {
      error:
        'HTTPException: 503: Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [region_status]',
    };
    const result = resolveStatsUpdate(payload, GOOD);
    expect(result.stats).toBe(GOOD);
    expect(result.error).toContain('fielddata=true on [region_status]');
  });

  it('keeps the last good stats when the payload is missing total_crops', () => {
    const result = resolveStatsUpdate({ as_of: '2026-09-24T00:00:00Z' }, GOOD);
    expect(result.stats).toBe(GOOD);
    expect(result.error).not.toBeNull();
  });

  it('reports the error with null stats when there is no prior good payload', () => {
    const result = resolveStatsUpdate({ error: 'boom' }, null);
    expect(result.stats).toBeNull();
    expect(result.error).toBe('boom');
  });
});

describe('summarizeStatsError (m29, 2026-09-24 interactive pass)', () => {
  it('passes a short message through unchanged', () => {
    expect(summarizeStatsError('opensearch unavailable')).toBe('opensearch unavailable');
  });

  it('truncates a long raw exception to a headline, appending an ellipsis', () => {
    const raw =
      "HTTPException: 503: opensearch error: RequestError(400, 'search_phase_execution_exception', 'Text fields are not optimised for operations that require per-document field data')";
    const result = summarizeStatsError(raw);
    expect(result.length).toBeLessThan(raw.length);
    expect(result.endsWith('…')).toBe(true);
    expect(raw.startsWith(result.slice(0, -1))).toBe(true);
  });

  it('respects a custom max length', () => {
    expect(summarizeStatsError('abcdefghij', 5)).toBe('abcde…');
  });
});

describe('drain rate and breakdown helpers', () => {
  it('needs two samples spanning 10s before reporting a rate', () => {
    expect(drainRateOf([{ unfinished: 100, t: 0 }])).toBeNull();
    expect(
      drainRateOf([
        { unfinished: 100, t: 0 },
        { unfinished: 90, t: 5000 },
      ]),
    ).toBeNull();
  });

  it('reports a positive rate while draining, negative while growing', () => {
    const draining = [
      { unfinished: 100, t: 0 },
      { unfinished: 80, t: 20_000 },
    ];
    expect(drainRateOf(draining)).toBe(1);
    expect(etaSecondsOf(1, 80)).toBe(80);
    expect(
      drainRateOf([
        { unfinished: 10, t: 0 },
        { unfinished: 30, t: 20_000 },
      ]),
    ).toBe(-1);
    expect(etaSecondsOf(-1, 30)).toBeNull();
    expect(etaSecondsOf(null, 30)).toBeNull();
  });

  it('drops samples older than twice the window', () => {
    const out = appendDrainSample([{ unfinished: 1, t: 0 }], 2, 200_000);
    expect(out).toEqual([{ unfinished: 2, t: 200_000 }]);
  });

  it('formats rates and relative times', () => {
    expect(formatRate(12)).toBe('12 crops/s');
    expect(formatRate(2)).toBe('2.0 crops/s');
    expect(formatRate(0.5)).toBe('30.0 crops/min');
    expect(fmtRelative(null)).toBe('—');
    expect(fmtRelative(0, 90_000)).toBe('1m ago');
  });

  it('builds labeled rows with one-decimal percentages and none when empty', () => {
    const rows = labeledRowsOf(
      datasetStatsFixture({
        labeled: { by_human: 1, by_vlm: 3, by_classifier: 0, other: 0 },
      } as Partial<DatasetStats>),
    );
    expect(rows.map((r) => [r.key, r.pct])).toEqual([
      ['by_human', 25],
      ['by_vlm', 75],
      ['by_classifier', 0],
      ['other', 0],
    ]);
    expect(labeledRowsOf(null)).toEqual([]);
  });
});
