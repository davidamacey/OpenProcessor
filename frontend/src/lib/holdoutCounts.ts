/**
 * Per-class test-holdout counts (visual audit 2026-09-24, D2/E1).
 *
 * The served per-class `validated_count` (`GET /stats/classes`) counts the
 * frozen test crops too, so the dashboard balance chart and `/export`'s
 * table read "35 validated" for a class whose 5 test crops can never be
 * trained on. Both surfaces show the served holdout count
 * (`GET /test_holdout/stats` `by_class`) next to it; the trainable count
 * itself is the served `trainable` on `/stats/classes`.
 */
import type { TestHoldoutStats } from '$lib/types';

/** class_id -> served test-holdout count; empty when holdout stats are absent. */
export function holdoutByClass(
  holdout: TestHoldoutStats | null | undefined,
): Map<number, number> {
  const m = new Map<number, number>();
  for (const b of holdout?.by_class ?? []) m.set(b.key, b.doc_count);
  return m;
}
