/**
 * Pure label/formatting helpers for the `/train` finished-run results
 * view (`RunResults.svelte`). No metric math lives here — every number
 * rendered is exactly what the backend served; this module only decides
 * *what to call it* and how to spell a missing value.
 */
import type { TrainEpochMetric, TrainEval, TrainState } from './types_train';

/** States a run no longer progresses past — the point at which a
 *  Results section has something to show (metrics for `finished`, the
 *  served error for `failed`, etc.). */
export const TERMINAL_TRAIN_STATES: ReadonlySet<TrainState> = new Set([
  'finished',
  'failed',
  'cancelled',
  'skipped',
  'lost',
]);

export function isTerminalTrainState(state: TrainState | string): boolean {
  return TERMINAL_TRAIN_STATES.has(state as TrainState);
}

/**
 * Label for `TrainEval`'s figures (`map50`/`map50_95`/`precision`/
 * `recall` and `per_class`) — the run's headline number, labelled
 * by the backend's own `eval.split`.
 */
export function evalSplitLabel(ev: TrainEval | null | undefined): string {
  if (!ev) return '';
  return ev.split === 'test' ? 'test split (frozen holdout)' : 'validation';
}

/** A metric value (map/precision/recall/f1/ap50 — a 0-1 score, not a
 *  count) — `null`/`undefined` render "—", never a false 0. */
export function formatMetric(v: number | null | undefined, digits = 3): string {
  return v == null ? '—' : v.toFixed(digits);
}

/** Any other scalar (sha, path, id, seed…) — `null`/`undefined`/empty
 *  string render "—". */
export function formatScalar(v: string | number | null | undefined): string {
  return v == null || v === '' ? '—' : String(v);
}

/** `"epoch N"` for a `TrainEpochMetric` that carries one, else `null` —
 *  used to caption `last_epoch_metric`/`best_checkpoint_metric` (a run
 *  whose status.json predates these fields carries neither the metric
 *  nor an epoch number, and formatMetric already renders "—" for that
 *  case; this only decides whether to show the epoch caption at all). */
export function metricEpochLabel(m: TrainEpochMetric | null | undefined): string | null {
  return m?.epoch != null ? `epoch ${m.epoch}` : null;
}
