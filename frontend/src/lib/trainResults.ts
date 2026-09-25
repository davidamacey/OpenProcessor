/**
 * Pure label/formatting helpers for the `/train` finished-run results
 * view (`RunResults.svelte`). No metric math lives here — every number
 * rendered is exactly what the backend served; this module only decides
 * *what to call it* and how to spell a missing value.
 */
import type { TrainEval, TrainState } from './types_train';

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
 * Label for `TrainEval`'s overall figures (`map50`/`map50_95`/
 * `precision`/`recall`).
 *
 * Today's backend doesn't serve `eval.split` — on that shape the
 * overall figures are actually the last VAL epoch's numbers (confirmed
 * against the live trainer), never the test split. Once the backend
 * ships `split`, label by it directly.
 */
export function evalOverallLabel(ev: TrainEval | null | undefined): string {
  if (!ev) return '';
  if (ev.split === 'test') return 'test split (frozen holdout)';
  if (ev.split === 'val') return 'validation';
  return 'validation (last epoch)';
}

/**
 * Label for `TrainEval.per_class` — on today's split-less shape this
 * really is computed over the frozen test holdout (unlike the overall
 * figures above), so it gets a different label from the same object.
 */
export function evalPerClassLabel(ev: TrainEval | null | undefined): string {
  if (!ev) return '';
  if (ev.split === 'test') return 'test split (frozen holdout)';
  if (ev.split === 'val') return 'validation';
  return 'test split (frozen holdout)';
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
