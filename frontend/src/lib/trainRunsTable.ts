/**
 * Pure helper for `/train`'s past-runs table single mAP50 column
 * (coordinator finding 6, 2026-09-24 visual review, updated for
 * OpenProcessor #34 W1's `best_metric`/`last_metric` removal): the
 * column shows the run's own headline number, `eval.map50`, labelled by
 * the backend's own `eval.split` ('test' when the frozen-holdout pass
 * ran, 'val' when it fell back). `best_checkpoint_metric` (a per-epoch
 * training-time figure) is never shown here — see `TrainEval`'s doc
 * comment in `types_train.ts` for why. This picks which already-served
 * field to show; it never blends or derives a new number.
 */
import type { TrainEval } from './types_train';

export interface BestMapSource {
  eval?: TrainEval | null;
}

export interface BestMapDisplay {
  value: number | null;
  source: 'test' | 'val' | null;
}

export function bestMapDisplay(r: BestMapSource): BestMapDisplay {
  return { value: r.eval?.map50 ?? null, source: r.eval?.split ?? null };
}
