/*
 * Pure guard for DatasetStats.svelte's applyStats(). Factored out so the
 * error-envelope handling (G1) is unit-testable without mounting the
 * component or standing up the SSE transport.
 */
import { type DatasetStats } from '$lib/api';

export interface StatsUpdateResult {
  stats: DatasetStats | null;
  error: string | null;
}

/**
 * The SSE `snapshot`/`stats` frames and the REST `{API_PREFIX}/stats/dataset`
 * body share this shape. Live, a backend outage sends `{error: "..."}` (the
 * op_items region_status mapping isn't aggregatable — see G1) instead of a
 * real DatasetStats payload. Trusting that blindly crashes the dashboard
 * (`Cannot read properties of undefined (reading
 * 'sam_drain_total_unfinished')`) once a caller reads into it. Guard here:
 * an error envelope, or any payload missing `total_crops` (present on
 * every real response), keeps the last-known-good `stats` and reports the
 * error instead.
 */
export function resolveStatsUpdate(
  payload: Record<string, unknown>,
  previous: DatasetStats | null,
): StatsUpdateResult {
  const err = payload?.error;
  if (typeof err === 'string' && err.length > 0) {
    return { stats: previous, error: err };
  }
  if (typeof payload?.total_crops !== 'number') {
    return { stats: previous, error: 'malformed stats payload (missing total_crops)' };
  }
  return { stats: payload as unknown as DatasetStats, error: null };
}
