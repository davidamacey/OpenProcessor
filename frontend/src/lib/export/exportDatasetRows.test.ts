import { describe, expect, it } from 'vitest';
import {
  buildExportRows,
  gapCellTitle,
  hasCurrentMulticlassExport,
  registryArtifactsAvailable,
  isNothingExportable,
  splitExportClasses,
  type ExportRow,
} from './exportDatasetRows';
import type { ExportDataset, StatsSummary, TestHoldoutStats } from '$lib/types';

function dataset(over: Partial<ExportDataset>): ExportDataset {
  return {
    kind: 'yolo',
    export_dir: '/exports/x',
    version_tag: 'v1',
    is_current: false,
    ...over,
  };
}

function perClass(
  over: Partial<StatsSummary['per_class'][number]>,
): StatsSummary['per_class'] {
  return [
    {
      class_id: 8,
      class_name: 'widget_a',
      count: 8,
      validated_count: 8,
      adequacy: 'block',
      aug_target: 500,
      aug_gap: 492,
      trainable: 8,
      trainable_gap: 22,
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

  it('flags nothing when there are no holdout stats at all', () => {
    const rows = buildExportRows(perClass({}), null);
    expect(rows[0]?.testDeficient).toBe(false);
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

  it('a class with no test-holdout row at all is deficient against the served minimum, not a hardcoded 5', () => {
    const holdout: TestHoldoutStats = { total: 0, min_test_per_class: 0, by_class: [] };
    const rows = buildExportRows(perClass({}), holdout);
    // min_test_per_class: 0 means "no minimum enforced" — 0 test crops is
    // NOT deficient. A hardcoded "< 5" would wrongly flag this row.
    expect(rows[0]?.testDeficient).toBe(false);
  });

  it('a class absent from by_class entirely (0 validated crops, never sampled) is NOT deficient, even with a real min_test_per_class — the live /export bug (79/84 classes wrongly red-badged)', () => {
    const holdout: TestHoldoutStats = {
      total: 25,
      min_test_per_class: 5,
      by_class: [{ key: 44, doc_count: 5, deficient: false }],
    };
    // class_id 8 (widget_a) has never been validated/frozen, so it has no
    // bucket in `by_class` at all — it must not inherit class 44's
    // min_test_per_class comparison.
    const rows = buildExportRows(perClass({ class_id: 8 }), holdout);
    expect(rows[0]?.testDeficient).toBe(false);
    expect(rows[0]?.test_count).toBe(0);
  });

  it('m16: surfaces the served adequacy tier verbatim', () => {
    const rows = buildExportRows(perClass({ adequacy: 'warn' }), null);
    expect(rows[0]?.adequacy).toBe('warn');
  });
});

describe('hasCurrentMulticlassExport (m15)', () => {
  it('false when the list is empty', () => {
    expect(hasCurrentMulticlassExport([])).toBe(false);
  });

  it('false when only a single_class dataset (e.g. widget_tag) is current — the actual bug case', () => {
    expect(
      hasCurrentMulticlassExport([
        dataset({
          kind: 'single_class',
          profile_name: 'widget_tag',
          is_current: true,
        }),
      ]),
    ).toBe(false);
  });

  it('false when a yolo dataset exists but is not the current one', () => {
    expect(
      hasCurrentMulticlassExport([dataset({ kind: 'yolo', is_current: false })]),
    ).toBe(false);
  });

  it('true only for a current yolo (multi-class) dataset', () => {
    expect(
      hasCurrentMulticlassExport([dataset({ kind: 'yolo', is_current: true })]),
    ).toBe(true);
  });
});

describe('registryArtifactsAvailable (F-61)', () => {
  it('a finished export reported by /export/status makes the downloads available', () => {
    expect(
      registryArtifactsAvailable([dataset({ kind: 'yolo', is_current: false })], {
        status: 'success',
        export_dir: '/exports/20260925T185030Z',
      }),
    ).toBe(true);
    expect(
      registryArtifactsAvailable(null, { status: 'success', export_dir: '/x' }),
    ).toBe(true);
  });

  it('not available with neither a current dataset nor a finished export', () => {
    expect(
      registryArtifactsAvailable([dataset({ kind: 'yolo', is_current: false })], {
        status: 'unknown',
        export_dir: null,
      }),
    ).toBe(false);
    expect(
      registryArtifactsAvailable(null, { status: 'success', export_dir: null }),
    ).toBe(false);
    expect(registryArtifactsAvailable([], null)).toBe(false);
  });
});

function row(over: Partial<ExportRow>): ExportRow {
  return {
    class_id: 1,
    class_name: 'widget_a',
    total: 10,
    validated: 5,
    aug_target: 0,
    gap: 0,
    test_count: 0,
    trainable: 5,
    trainableGap: 0,
    testDeficient: false,
    adequacy: 'ok',
    ...over,
  };
}

describe('isNothingExportable (DQ-M9 frontend half)', () => {
  it('true when there are no rows at all', () => {
    expect(isNothingExportable([])).toBe(true);
  });

  it('true when every row has 0 validated crops (the live repro: 0 class_validated dataset-wide)', () => {
    expect(
      isNothingExportable([
        row({ validated: 0 }),
        row({ validated: 0, adequacy: 'block' }),
      ]),
    ).toBe(true);
  });

  it('true when every class is at the served block adequacy tier, even with some validated crops', () => {
    expect(
      isNothingExportable([
        row({ validated: 3, adequacy: 'block' }),
        row({ validated: 2, adequacy: 'block' }),
      ]),
    ).toBe(true);
  });

  it('false when at least one class has validated crops and is not blocked', () => {
    expect(
      isNothingExportable([
        row({ validated: 0, adequacy: 'block' }),
        row({ validated: 20, adequacy: 'ok' }),
      ]),
    ).toBe(false);
  });
});

describe('E1/E2 (visual audit 2026-09-24): served trainable, classes with objects', () => {
  it('uses the served trainable/trainable_gap, never validated-minus-holdout math', () => {
    // Deliberately served numbers that DISAGREE with what a client-side
    // validated-minus-holdout formula would produce (30/470) — proves the
    // served fields drive the row, not a recomputation.
    const rows = buildExportRows(
      perClass({
        class_id: 52,
        validated_count: 35,
        aug_target: 500,
        aug_gap: 465,
        trainable: 174,
        trainable_gap: 326,
      }),
      {
        total: 5,
        min_test_per_class: 5,
        by_class: [{ key: 52, doc_count: 5, deficient: false }],
      },
    );
    expect(rows[0]).toMatchObject({ trainable: 174, trainableGap: 326 });
  });

  it('splitExportClasses separates classes with any exported object from empty ones', () => {
    const { withObjects, empty } = splitExportClasses([
      { class_id: 1, train: 0, val: 0, test: 0 },
      { class_id: 2, train: 0, val: 0, test: 5 },
      { class_id: 3, train: 24, val: 1, test: 5 },
    ]);
    expect(withObjects.map((c) => c.class_id)).toEqual([2, 3]);
    expect(empty.map((c) => c.class_id)).toEqual([1]);
  });
});

describe('gapCellTitle', () => {
  it('names the served per-class minimum when known', () => {
    expect(gapCellTitle(8, 20)).toBe(
      '8 more trainable crops needed to reach the per-class minimum of 20',
    );
    expect(gapCellTitle(0, 20)).toBe('meets the per-class minimum of 20');
  });

  it('omits the number when the minimum was not served', () => {
    expect(gapCellTitle(3, null)).toBe(
      '3 more trainable crops needed to reach the per-class minimum',
    );
  });
});
