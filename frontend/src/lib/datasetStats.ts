/*
 * Pure guard for DatasetStats.svelte's applyStats(). Factored out so the
 * error-envelope handling (G1) is unit-testable without mounting the
 * component or standing up the SSE transport.
 */
import type { DatasetStats } from '$lib/api_stats';

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
 * 'region_drain_total_unfinished')`) once a caller reads into it. Guard here:
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

/**
 * m29 (2026-09-24 interactive pass): the stats-unavailable banner dumped
 * the raw served exception verbatim ("HTTPException: 503: opensearch
 * error: RequestError(400, 'search_phase_execution_exception', 'Text
 * fields are not optimised…"), which is backend internals, not an
 * operator-facing message. This shortens it to a headline — still the
 * server's own words, just capped — the full string stays available via
 * a `title` attribute in the component, never dropped.
 */
export function summarizeStatsError(raw: string, maxLen = 90): string {
  const trimmed = raw.trim();
  if (trimmed.length <= maxLen) return trimmed;
  return `${trimmed.slice(0, maxLen).trimEnd()}…`;
}

export interface BreakdownRow {
  key: string;
  label: string;
  count: number;
  tone: string;
  pct: number;
}

type RawRow = Omit<BreakdownRow, 'pct'>;

function withPct(rows: RawRow[], denominator: number): BreakdownRow[] {
  return rows.map((r) => ({
    ...r,
    pct: Math.round((r.count / denominator) * 1000) / 10,
  }));
}

const LABELED_TONES: Record<string, string> = {
  by_human: 'bg-green-500',
  by_import: 'bg-amber-500',
  by_vlm: 'bg-blue-500',
  by_classifier: 'bg-purple-500',
  other: 'bg-zinc-500',
};

/** The served `labeled.rows` with a bar colour per provenance; counts, labels and shares are the server's. */
export function labeledRowsOf(stats: DatasetStats | null): BreakdownRow[] {
  const l = stats?.labeled;
  if (!l || l.total === 0) return [];
  return l.rows.map((r) => ({
    key: r.key,
    label: r.label,
    count: r.count,
    tone: LABELED_TONES[r.key] ?? 'bg-zinc-500',
    pct: r.share * 100,
  }));
}

/**
 * Region-detection breakdown, separate from class labeling. The backend's
 * `regions` block counts by role in the active detection profile, so these
 * rows are generic. Denominator is total_crops (fraction of crops where a
 * region was detected), not the labeled total.
 */
export function slotRowsOf(
  stats: DatasetStats | null,
  statsKey: string | undefined,
): BreakdownRow[] {
  const p = statsKey
    ? (stats as unknown as Record<string, DatasetStats['regions'] | undefined> | null)?.[
        statsKey
      ]
    : undefined;
  const total = stats?.total_crops ?? 0;
  if (!p || total === 0) return [];
  return withPct(
    [
      {
        key: 'by_detector',
        label: 'Detector',
        count: p.by_detector,
        tone: 'bg-cyan-500',
      },
      {
        key: 'by_segmenter',
        label: 'Segmenter fallback',
        count: p.by_segmenter,
        tone: 'bg-teal-500',
      },
      {
        key: 'by_human_drew',
        label: 'Human-drew bbox',
        count: p.by_human_drew,
        tone: 'bg-green-500',
      },
      // The operator confirmed an AI-proposed bbox; by_human_drew is only a
      // from-scratch bbox, validated_by_human is the union of both.
      {
        key: 'validated_by_human',
        label: 'Confirmed by human',
        count: p.validated_by_human,
        tone: 'bg-emerald-500',
      },
    ],
    total,
  );
}

export interface DrainSample {
  unfinished: number;
  t: number;
}

/** Ring-buffer window for the drain-rate average (about a minute). */
export const SAMPLE_WINDOW_MS = 60_000;

export function appendDrainSample(
  samples: DrainSample[],
  unfinished: number,
  now: number,
): DrainSample[] {
  return [...samples, { unfinished, t: now }].filter(
    (s) => now - s.t <= SAMPLE_WINDOW_MS * 2,
  );
}

/**
 * Crops/sec over the sample window: null until two samples span at least
 * 10s, positive while the queue drains, negative while it grows.
 */
export function drainRateOf(samples: DrainSample[]): number | null {
  if (samples.length < 2) return null;
  const oldest = samples[0];
  const newest = samples[samples.length - 1];
  const dt = (newest.t - oldest.t) / 1000;
  if (dt < 10) return null;
  return (oldest.unfinished - newest.unfinished) / dt;
}

/** Remaining / rate; null when the rate is unknown or not draining. */
export function etaSecondsOf(drainRate: number | null, remaining: number): number | null {
  if (drainRate == null || drainRate <= 0) return null;
  return remaining / drainRate;
}

export function formatRate(rate: number): string {
  if (rate >= 10) return `${rate.toFixed(0)} crops/s`;
  if (rate >= 1) return `${rate.toFixed(1)} crops/s`;
  return `${(rate * 60).toFixed(1)} crops/min`;
}

export function fmt(n: number | undefined | null): string {
  return (n ?? 0).toLocaleString();
}

export function fmtRelative(ms: number | null, now: number = Date.now()): string {
  if (ms == null) return '—';
  const secs = Math.max(0, Math.round((now - ms) / 1000));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}
