/**
 * Pure row-building for the `/export` dataset table.
 *
 * `aug_target`/`aug_gap` are server-computed
 * (`GET {API_PREFIX}/stats/classes`, `docs/design/
 * logic-moves-adoption-plan-2026-09-24.md` §1.7) — this module never
 * clamps or derives them from `validated_count`. Likewise `testDeficient`
 * is the server's per-bucket `deficient` flag
 * (`GET {API_PREFIX}/test_holdout/stats`). A class with NO holdout bucket at
 * all (e.g. 0 validated crops — never sampled into the holdout in the
 * first place) is never flagged deficient by this module: the server's
 * `by_class` list only ever covers classes it actually considered, and
 * live evidence (`test_holdout/stats` on a 84-class deployment with only
 * 5 classes validated) shows the other 79 simply absent, not flagged —
 * treating "absent" as "deficient" was the bug (every un-validated class
 * showed a red "below N test crops" badge on `/export`).
 */
import type { ExportDataset, StatsSummary, TestHoldoutStats } from '$lib/types';

export interface ExportRow {
  class_id: number;
  class_name: string;
  total: number;
  validated: number;
  aug_target: number;
  /** Served `aug_gap`. */
  gap: number;
  test_count: number;
  /** Served `trainable`. */
  trainable: number;
  /** Served `trainable_gap` (shortfall against the per-class hard minimum). */
  trainableGap: number;
  testDeficient: boolean;
  /** Served adequacy tier (`block`/`warn`/`ok`). */
  adequacy: string;
}

export function buildExportRows(
  perClass: StatsSummary['per_class'] | undefined,
  holdout: TestHoldoutStats | null,
): ExportRow[] {
  if (!perClass) return [];
  const testMap = new Map<number, { count: number; deficient: boolean }>();
  for (const b of holdout?.by_class ?? []) {
    testMap.set(b.key, { count: b.doc_count, deficient: b.deficient });
  }
  return perClass.map((c) => {
    const test = testMap.get(c.class_id);
    return {
      class_id: c.class_id,
      class_name: c.class_name,
      total: c.count,
      validated: c.validated_count,
      aug_target: c.aug_target,
      gap: c.aug_gap,
      test_count: test?.count ?? 0,
      trainable: c.trainable,
      trainableGap: c.trainable_gap,
      testDeficient: test?.deficient ?? false,
      adequacy: c.adequacy,
    };
  });
}

/**
 * DQ-M9 frontend half (docs/design/data-quality-pass-2026-09-24.md):
 * whether the served per-class rows add up to "nothing to export" — 0
 * `class_validated` items total, or every class at the served `block`
 * adequacy tier. The audit found `POST /export/yolo` has no readiness
 * gate of its own and the Export button was enabled over a table that
 * was all red "block" (every class at 0 validated) — this is the pure
 * rule the button's `disabled` reads, built entirely from server-served
 * `validated`/`adequacy`, never a hardcoded count/threshold (those —
 * `block_below`/`warn_below`/`min_test` — already come from the server
 * and are what produced `adequacy` in the first place; this function
 * doesn't re-derive or duplicate them).
 */
export function isNothingExportable(rows: ExportRow[]): boolean {
  if (rows.length === 0) return true;
  const totalValidated = rows.reduce((sum, r) => sum + r.validated, 0);
  if (totalValidated === 0) return true;
  if (rows.every((r) => r.adequacy === 'block')) return true;
  return false;
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

/**
 * F-61: the registry/data.yaml/manifest downloads serve the current
 * multi-class export. The served datasets list flags it `is_current`;
 * `GET /export/status` reporting a finished (`success`) export with an
 * `export_dir` is the same fact from the other endpoint, so either one
 * makes the downloads available (after a fresh UI export the page said
 * "No frozen multi-class export yet" beside downloads that worked).
 */
export function registryArtifactsAvailable(
  datasets: ExportDataset[] | null,
  status: { status?: string | null; export_dir?: string | null } | null,
): boolean {
  if (datasets && hasCurrentMulticlassExport(datasets)) return true;
  return status?.status === 'success' && !!status.export_dir;
}

/**
 * E2 (visual audit 2026-09-24): the export summary said "classes 84" when
 * only 5 classes had any exported object. Splits the served
 * `class_split_counts` into classes with at least one object and the rest.
 */
export function splitExportClasses<
  T extends { train: number; val: number; test: number },
>(counts: readonly T[]): { withObjects: T[]; empty: T[] } {
  const withObjects: T[] = [];
  const empty: T[] = [];
  for (const c of counts) (c.train + c.val + c.test > 0 ? withObjects : empty).push(c);
  return { withObjects, empty };
}

/** Header tooltip for the Gap column: the served `trainable_gap`. */
export const GAP_COLUMN_TITLE =
  'Trainable crops still needed to reach the per-class minimum (served trainable_gap)';

/** Header tooltip for the Trainable column: the served `trainable`. */
export const TRAINABLE_COLUMN_TITLE =
  'Validated crops training can use: the frozen test holdout and excluded crops are left out (served trainable)';

/**
 * Per-cell tooltip for the served `trainable_gap`: the shortfall of
 * `trainable` against the served per-class hard minimum
 * (`thresholds.block_below`, passed in when `/stats/classes` served it).
 */
export function gapCellTitle(gap: number, minimum: number | null): string {
  const floor = minimum != null ? ` of ${minimum}` : '';
  return gap <= 0
    ? `meets the per-class minimum${floor}`
    : `${gap} more trainable crops needed to reach the per-class minimum${floor}`;
}
