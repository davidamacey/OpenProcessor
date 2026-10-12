/** Display formatting for a served bulk-ingest run. No decisions: every number
 *  shown is the one the backend served. */
import type { BulkIngestProgress, BulkIngestRun } from '$lib/types_bulkIngest';

/** `1h05m`, `4m09s`, `12s`; `--` when the backend has no estimate yet. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '--';
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}

/** The rate to headline: while the run is live the last 30 s once it exists, else the
 *  run average (the recent window is empty once the run has stopped). */
export function headlineRate(p: BulkIngestProgress, live: boolean): number | null {
  return live
    ? (p.images_per_s_recent ?? p.images_per_s ?? null)
    : (p.images_per_s ?? null);
}

/** 0-100 for the bar; the denominator is still growing while a folder is walked. */
export function progressPercent(p: BulkIngestProgress): number {
  if (p.discovered <= 0) return 0;
  return Math.min(100, Math.round((p.processed / p.discovered) * 100));
}

/** Stage seconds, largest first. */
export function stageRows(run: BulkIngestRun): [string, number][] {
  return Object.entries(run.stage_s ?? {}).sort((a, b) => b[1] - a[1]);
}
