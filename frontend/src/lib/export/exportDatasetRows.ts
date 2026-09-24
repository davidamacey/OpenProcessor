/**
 * Pure row-building for the `/export` dataset table.
 *
 * `aug_target`/`aug_gap` are server-computed
 * (`GET {API_PREFIX}/stats/classes`, `docs/design/
 * logic-moves-adoption-plan-2026-09-24.md` §1.7) — this module never
 * clamps or derives them from `validated_count`. Likewise `testDeficient`
 * prefers the server's per-bucket `deficient` flag
 * (`GET {API_PREFIX}/test_holdout/stats`) and only falls back to comparing
 * against the served `min_test_per_class` when a bucket omits the flag —
 * never a hardcoded "< 5".
 */
import type { StatsSummary, TestHoldoutStats } from '$lib/types';

export interface ExportRow {
  class_id: number;
  class_name: string;
  total: number;
  validated: number;
  aug_target: number;
  gap: number;
  test_count: number;
  testDeficient: boolean;
}

const DEFAULT_MIN_TEST_PER_CLASS = 5;

export function buildExportRows(
  perClass: StatsSummary['per_class'] | undefined,
  holdout: TestHoldoutStats | null,
): ExportRow[] {
  if (!perClass) return [];
  const minTest = holdout?.min_test_per_class ?? DEFAULT_MIN_TEST_PER_CLASS;
  const testMap = new Map<number, { count: number; deficient?: boolean }>();
  for (const b of holdout?.by_class ?? []) {
    testMap.set(b.key, { count: b.doc_count, deficient: b.deficient });
  }
  return perClass.map((c) => {
    const validated = c.validated_count ?? 0;
    const target = c.aug_target ?? 0;
    const test = testMap.get(c.class_id);
    return {
      class_id: c.class_id,
      class_name: c.class_name,
      total: c.count ?? 0,
      validated,
      aug_target: target,
      gap: c.aug_gap ?? target - validated,
      test_count: test?.count ?? 0,
      testDeficient: test?.deficient ?? (test?.count ?? 0) < minTest,
    };
  });
}
