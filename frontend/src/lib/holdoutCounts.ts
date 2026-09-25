/**
 * Per-class test-holdout counts (visual audit 2026-09-24, D2/E1).
 *
 * The served per-class `validated_count` (`GET /stats/classes`) counts the
 * frozen test crops too, so the dashboard balance chart and `/export`'s
 * table read "35 validated" for a class whose 5 test crops can never be
 * trained on. Both surfaces now show the served holdout count
 * (`GET /test_holdout/stats` `by_class`) next to it and the difference as
 * "trainable".
 *
 * TODO(backend): serve a trainable (holdout-excluded) per-class count and
 * gap on `/stats/classes` so this subtraction moves server-side.
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

/** Validated crops that are not frozen test crops (never below 0). */
export function trainableCount(validated: number, test: number): number {
  return Math.max(0, validated - test);
}
