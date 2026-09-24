/**
 * Pure row-building for the `/export` dataset table.
 *
 * `aug_target`/`aug_gap` are server-computed
 * (`GET {API_PREFIX}/stats/classes`, `docs/design/
 * logic-moves-adoption-plan-2026-09-24.md` §1.7) — this module never
 * clamps or derives them from `validated_count`. Likewise `testDeficient`
 * prefers the server's per-bucket `deficient` flag
 * (`GET {API_PREFIX}/test_holdout/stats`). A class with no holdout bucket
 * has zero test crops, which is compared against the served
 * `min_test_per_class`. With no holdout stats at all nothing is flagged.
 */
import type { ExportDataset, StatsSummary, TestHoldoutStats } from '$lib/types';

export interface ExportRow {
  class_id: number;
  class_name: string;
  total: number;
  validated: number;
  aug_target: number;
  /** Served `aug_gap`; null when the server didn't send one. */
  gap: number | null;
  test_count: number;
  testDeficient: boolean;
  /** Served adequacy tier (`block`/`warn`/`ok`); null when unserved (m16). */
  adequacy: string | null;
}

export function buildExportRows(
  perClass: StatsSummary['per_class'] | undefined,
  holdout: TestHoldoutStats | null,
): ExportRow[] {
  if (!perClass) return [];
  const minTest = holdout?.min_test_per_class ?? null;
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
      gap: c.aug_gap ?? null,
      test_count: test?.count ?? 0,
      testDeficient: test?.deficient ?? (minTest != null && (test?.count ?? 0) < minTest),
      adequacy: c.adequacy ?? null,
    };
  });
}

/**
 * m15 (2026-09-24 interactive pass): the registry download buttons
 * (class_registry.json/data.yaml/manifest.json) only make sense for a
 * frozen multi-class (`yolo`) export — they used to gate on the shared
 * export-job-status slot instead, which stayed "success" with no current
 * `yolo` dataset on disk and 404ed. Whether a served `yolo` dataset is
 * current is exactly what `GET {API_PREFIX}/export/datasets` reports —
 * no client-side inference.
 */
export function hasCurrentMulticlassExport(datasets: ExportDataset[]): boolean {
  return datasets.some((d) => d.kind === 'yolo' && d.is_current);
}
