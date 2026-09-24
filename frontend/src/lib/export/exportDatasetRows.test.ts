import { describe, expect, it } from 'vitest';
import { buildExportRows } from './exportDatasetRows';
import type { StatsSummary, TestHoldoutStats } from '$lib/types';

function perClass(
  over: Partial<StatsSummary['per_class'][number]>,
): StatsSummary['per_class'] {
  return [
    {
      class_id: 8,
      class_name: 'bmw',
      count: 8,
      validated_count: 8,
      ...over,
    },
  ];
}

describe('buildExportRows — W4: server-served aug_target/aug_gap/deficient, no client rule', () => {
  it('returns [] when per_class is absent', () => {
    expect(buildExportRows(undefined, null)).toEqual([]);
  });

  it('uses the server aug_target/aug_gap verbatim — never clamps validated to [500,3000] client-side', () => {
    // A validated_count of 8 would clamp to 500 under the old client rule
    // (Math.min(3000, Math.max(500, 8))); the server here instead reports
    // an aug_target of 50 for some other reason (e.g. a smaller deployment
    // threshold) — the row must reflect exactly that, not the old clamp.
    const rows = buildExportRows(perClass({ aug_target: 50, aug_gap: 42 }), null);
    expect(rows[0]).toMatchObject({ aug_target: 50, gap: 42 });
  });

  it('derives gap from aug_target - validated only when the server omits aug_gap', () => {
    const rows = buildExportRows(perClass({ aug_target: 500, aug_gap: undefined }), null);
    expect(rows[0]?.gap).toBe(500 - 8);
  });

  it('prefers the server per-bucket "deficient" flag over any local count comparison', () => {
    const holdout: TestHoldoutStats = {
      total: 1,
      min_test_per_class: 5,
      by_class: [{ key: 8, doc_count: 999, deficient: true }],
    };
    // doc_count (999) is far above any threshold, but the server explicitly
    // flagged this class deficient (e.g. a per-source stratification gap) —
    // the row must trust that over recomputing from the count.
    const rows = buildExportRows(perClass({}), holdout);
    expect(rows[0]?.testDeficient).toBe(true);
    expect(rows[0]?.test_count).toBe(999);
  });

  it('falls back to comparing against the served min_test_per_class when a bucket omits "deficient"', () => {
    const holdout: TestHoldoutStats = {
      total: 1,
      min_test_per_class: 10,
      by_class: [{ key: 8, doc_count: 7 }],
    };
    const rows = buildExportRows(perClass({}), holdout);
    expect(rows[0]?.testDeficient).toBe(true); // 7 < 10

    const holdout2: TestHoldoutStats = {
      total: 1,
      min_test_per_class: 5,
      by_class: [{ key: 8, doc_count: 7 }],
    };
    const rows2 = buildExportRows(perClass({}), holdout2);
    expect(rows2[0]?.testDeficient).toBe(false); // 7 >= 5
  });

  it('a class with no test-holdout row at all is deficient against the served minimum, not a hardcoded 5', () => {
    const holdout: TestHoldoutStats = { total: 0, min_test_per_class: 0, by_class: [] };
    const rows = buildExportRows(perClass({}), holdout);
    // min_test_per_class: 0 means "no minimum enforced" — 0 test crops is
    // NOT deficient. A hardcoded "< 5" would wrongly flag this row.
    expect(rows[0]?.testDeficient).toBe(false);
  });
});
