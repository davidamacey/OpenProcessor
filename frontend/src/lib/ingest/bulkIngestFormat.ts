/** Display formatting for a served bulk-ingest run. No decisions: every number
 *  shown is the one the backend served. */
import type { BulkIngestProgress, BulkIngestRun } from '$lib/types_bulkIngest';

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
