/**
 * G1: resolveStatsUpdate's error-envelope guard. Live, GET
 * {API_PREFIX}/stats/dataset (and the SSE `snapshot`/`stats` frames that
 * share its shape) can return `{error: "..."}` instead of a real
 * DatasetStats body — trusting that blindly crashed the dashboard with
 * `Cannot read properties of undefined (reading
 * 'region_drain_total_unfinished')`.
 */
import { describe, expect, it } from 'vitest';
import { resolveStatsUpdate, summarizeStatsError } from './datasetStats';
import type { DatasetStats } from './api';

const GOOD: DatasetStats = {
  as_of: '2026-09-24T00:00:00Z',
  total_crops: 422,
  validated: 297,
  test_holdout: 12,
  by_source: [{ key: 'tag_holdout_sample', doc_count: 300 }],
  labeled: { by_human: 100, by_vlm: 50, by_classifier: 10, by_proposal: 5, other: 0 },
  regions: { total_detected: 0, by_detector: 0, by_segmenter: 0, by_human: 0 },
  unlabeled: { pending_detection: 0, pending_verification: 0, no_label_source: 0 },
  in_progress: { region_drain_total_unfinished: 3 },
  clusters: {
    last_run_at: null,
    cluster_count: 0,
    residual_count: 0,
    noise_count: 0,
    method: null,
  },
};

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
