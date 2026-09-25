/**
 * Pure helper for `/train`'s past-runs table single mAP50 column
 * (coordinator finding 6, 2026-09-24 visual review): the column used to
 * always show `best_metric.map50` — a per-key VAL max across epochs —
 * even for a run whose own `eval` already reports a real test-split
 * figure (`eval.split === 'test'`), so a served test mAP50 of 0.917 sat
 * unused next to a val 0.995 shown as if it were the run's headline
 * number. This picks which already-served field to show; it never
 * blends or derives a new number.
 */
import type { TrainEval } from './types_train';

export interface BestMapSource {
  eval?: TrainEval | null;
  best_metric?: { map50?: number } | null;
}

export interface BestMapDisplay {
  value: number | null;
  source: 'test' | 'val';
}

export function bestMapDisplay(r: BestMapSource): BestMapDisplay {
  if (r.eval?.split === 'test' && r.eval.map50 != null) {
    return { value: r.eval.map50, source: 'test' };
  }
  return { value: r.best_metric?.map50 ?? null, source: 'val' };
}
