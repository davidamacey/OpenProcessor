import { describe, expect, it } from 'vitest';
import {
  EMPTY_METHODS,
  exportDedupDefault,
  normalizeMethodStatus,
  parseMethodsResponse,
} from '$lib/strategies';
import type { DatasetExportInfo } from '$lib/strategies';

describe('normalizeMethodStatus', () => {
  it('passes through every known status value', () => {
    expect(normalizeMethodStatus('stable')).toBe('stable');
    expect(normalizeMethodStatus('experimental')).toBe('experimental');
    expect(normalizeMethodStatus('shadow')).toBe('shadow');
    expect(normalizeMethodStatus('disabled')).toBe('disabled');
  });

  it("normalizes an unrecognized status string to 'disabled' instead of throwing", () => {
    expect(normalizeMethodStatus('deprecated')).toBe('disabled');
    expect(normalizeMethodStatus('beta')).toBe('disabled');
    expect(normalizeMethodStatus('')).toBe('disabled');
  });

  it('normalizes non-string status values (null/number/object) to disabled', () => {
    expect(normalizeMethodStatus(null)).toBe('disabled');
    expect(normalizeMethodStatus(undefined)).toBe('disabled');
    expect(normalizeMethodStatus(42)).toBe('disabled');
    expect(normalizeMethodStatus({ status: 'stable' })).toBe('disabled');
  });
});

// The real {API_PREFIX}/methods wire shape (confirmed 2026-09-10 against the live
// backend, strategy_registry.py's get_registry()): one flat `strategies`
// array, each entry carrying an `axis` field (`'cluster' | 'sort' |
// 'score' | 'overlay'`), plus a top-level `flags` object this file
// doesn't consume. parseMethodsResponse groups by `axis` into the four
// buckets every downstream consumer (isDiverseOverlayAvailable, etc.)
// already expects — these tests exercise that grouping directly rather
// than the four-separate-top-level-arrays shape an earlier version of
// this file assumed before the real contract was confirmed.
describe('parseMethodsResponse', () => {
  it('parses a well-formed full payload into typed lists, grouped by axis', () => {
    const raw = {
      strategies: [
        {
          id: 'ivf',
          axis: 'cluster',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
        {
          id: 'hdbscan',
          axis: 'cluster',
          label: 'HDBSCAN (dormant)',
          status: 'disabled',
        },
        {
          id: 'default',
          axis: 'sort',
          label: 'Recent first',
          status: 'stable',
          default: true,
        },
        {
          id: 'uncertainty_entropy',
          axis: 'sort',
          label: 'Uncertainty margin',
          status: 'experimental',
          requires_field: 'probe_pred_margin',
          field_coverage: 0.42,
        },
        {
          id: 'viz_projection',
          axis: 'overlay',
          label: 'UMAP scatter',
          status: 'shadow',
          requires_banner: true,
          purity: 0.472,
        },
        {
          id: 'uniqueness',
          axis: 'score',
          label: 'Uniqueness (kNN)',
          status: 'experimental',
          version: '1',
          field_coverage: 0.9,
        },
      ],
      flags: { scores_enabled: false },
    };
    const parsed = parseMethodsResponse(raw);
    expect(parsed.cluster_methods).toEqual([
      { id: 'ivf', label: 'FAISS IVF-512 (production)', status: 'stable', default: true },
      {
        id: 'hdbscan',
        label: 'HDBSCAN (dormant)',
        status: 'disabled',
        default: undefined,
      },
    ]);
    expect(parsed.review_sorts[1]).toMatchObject({
      id: 'uncertainty_entropy',
      requires_field: 'probe_pred_margin',
      field_coverage: 0.42,
    });
    expect(parsed.overlays).toHaveLength(1);
    expect(parsed.overlays[0]?.status).toBe('shadow');
    expect(parsed.overlays[0]?.requires_banner).toBe(true);
    expect(parsed.overlays[0]?.purity).toBe(0.472);
    expect(parsed.scores[0]).toMatchObject({ id: 'uniqueness', version: '1' });
  });

  it('defaults requires_banner to undefined when the server omits it', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        {
          id: 'viz_projection',
          axis: 'overlay',
          label: 'UMAP scatter',
          status: 'experimental',
        },
      ],
    });
    expect(parsed.overlays[0]?.requires_banner).toBeUndefined();
  });

  it('ignores a non-boolean requires_banner rather than throwing', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        {
          id: 'viz_projection',
          axis: 'overlay',
          label: 'UMAP scatter',
          status: 'stable',
          requires_banner: 'yes',
        },
      ],
    });
    expect(parsed.overlays[0]?.requires_banner).toBeUndefined();
  });

  it('never throws on a completely unusable payload (null / string / number / array)', () => {
    for (const bad of [null, undefined, 'nope', 42, [], true]) {
      expect(() => parseMethodsResponse(bad)).not.toThrow();
      const parsed = parseMethodsResponse(bad);
      expect(parsed).toEqual({
        cluster_methods: [],
        review_sorts: [],
        overlays: [],
        scores: [],
        dataset_exports: [],
        detection_profiles: [],
        prompt_packs: [],
        vlm: [],
        axes: [],
      });
    }
  });

  it('defaults a missing/non-array/malformed strategies key to empty lists without throwing', () => {
    for (const bad of ['not-an-array', null, undefined, 42]) {
      const parsed = parseMethodsResponse({ strategies: bad });
      expect(parsed).toEqual({
        cluster_methods: [],
        review_sorts: [],
        overlays: [],
        scores: [],
        dataset_exports: [],
        detection_profiles: [],
        prompt_packs: [],
        vlm: [],
        axes: [],
      });
    }
  });

  it("routes axis:'export' entries into dataset_exports (real T-C2 wire shape)", () => {
    // Verbatim from strategy_registry.py's _export_strategies() @ 4a30e36.
    const parsed = parseMethodsResponse({
      strategies: [
        {
          id: 'yolo',
          axis: 'export',
          label: 'YOLO detection dataset export',
          status: 'stable',
          default: true,
        },
      ],
      flags: {},
    });
    expect(parsed.dataset_exports).toEqual([
      {
        id: 'yolo',
        label: 'YOLO detection dataset export',
        status: 'stable',
        default: true,
      },
    ]);
    // An export entry must not leak into any other bucket.
    expect(parsed.cluster_methods).toEqual([]);
    expect(parsed.review_sorts).toEqual([]);
    expect(parsed.overlays).toEqual([]);
    expect(parsed.scores).toEqual([]);
  });

  it("routes axis:'detection_profile' and axis:'prompt_pack' entries into their own buckets", () => {
    // The agreed-but-not-yet-live wire shape (this plan §1.3): one new
    // `axis` value per entry in the same flat `strategies` array, no new
    // response envelope.
    const parsed = parseMethodsResponse({
      strategies: [
        {
          id: 'grounding_v2',
          axis: 'detection_profile',
          label: 'Grounding detector v2',
          status: 'stable',
          default: true,
        },
        {
          id: 'sam3_dense',
          axis: 'detection_profile',
          label: 'SAM3 dense proposals',
          status: 'experimental',
        },
        {
          id: 'warehouse_v1',
          axis: 'prompt_pack',
          label: 'Warehouse vocabulary',
          status: 'stable',
          default: true,
        },
      ],
      flags: {},
    });
    expect(parsed.detection_profiles).toEqual([
      {
        id: 'grounding_v2',
        label: 'Grounding detector v2',
        status: 'stable',
        default: true,
      },
      {
        id: 'sam3_dense',
        label: 'SAM3 dense proposals',
        status: 'experimental',
        default: undefined,
      },
    ]);
    expect(parsed.prompt_packs).toEqual([
      {
        id: 'warehouse_v1',
        label: 'Warehouse vocabulary',
        status: 'stable',
        default: true,
      },
    ]);
    // Neither axis leaks into any pre-existing bucket.
    expect(parsed.cluster_methods).toEqual([]);
    expect(parsed.review_sorts).toEqual([]);
    expect(parsed.overlays).toEqual([]);
    expect(parsed.scores).toEqual([]);
    expect(parsed.dataset_exports).toEqual([]);
  });

  it('drops entries missing a usable id or label, or an unrecognized axis, instead of crashing the whole parse', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        { id: 'ivf', axis: 'cluster', label: 'FAISS IVF-512', status: 'stable' },
        { axis: 'cluster', label: 'no id' },
        { id: 'no-label', axis: 'cluster' },
        { id: '', axis: 'cluster', label: 'empty id' },
        { id: 123, axis: 'cluster', label: 'non-string id' },
        { id: 'no-axis', label: 'missing axis entirely' },
        { id: 'future-axis', axis: 'quantum', label: "an axis this build doesn't route" },
        null,
        'garbage',
        42,
      ],
    });
    expect(parsed.cluster_methods).toEqual([
      { id: 'ivf', label: 'FAISS IVF-512', status: 'stable', default: undefined },
    ]);
    expect(parsed.review_sorts).toEqual([]);
    expect(parsed.overlays).toEqual([]);
    expect(parsed.scores).toEqual([]);
  });

  it('carries an unrecognized-but-well-formed id through untouched (forward-tolerant)', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        {
          id: 'some_future_method_v9',
          axis: 'cluster',
          label: 'Future Method',
          status: 'experimental',
        },
      ],
    });
    expect(parsed.cluster_methods[0]?.id).toBe('some_future_method_v9');
    expect(parsed.cluster_methods[0]?.status).toBe('experimental');
  });

  it('normalizes an unrecognized status on a real entry to disabled rather than throwing', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        { id: 'mistakenness', axis: 'sort', label: 'Mistakenness', status: 'beta_v2' },
      ],
    });
    expect(parsed.review_sorts).toHaveLength(1);
    expect(parsed.review_sorts[0]?.status).toBe('disabled');
  });

  it('the same id may legitimately appear in more than one axis bucket (score vs sort)', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        {
          id: 'mistakenness',
          axis: 'score',
          label: 'Mistakenness',
          status: 'experimental',
        },
        {
          id: 'mistakenness',
          axis: 'sort',
          label: 'Mistakenness',
          status: 'experimental',
        },
      ],
    });
    expect(parsed.scores).toHaveLength(1);
    expect(parsed.review_sorts).toHaveLength(1);
  });
});

describe('served bounds (v0.4.0 facts)', () => {
  it('parses the diverse overlay caps and the export dedup threshold with bounds', () => {
    const m = parseMethodsResponse({
      strategies: [
        {
          id: 'diverse',
          axis: 'overlay',
          label: 'Diverse',
          status: 'stable',
          max_k: 10000,
          select_max_k: 50000,
        },
        {
          id: 'single_class',
          axis: 'export',
          label: 'Single class',
          status: 'stable',
          dedup_threshold_default: 0.98,
          dedup_threshold_min: 0,
          dedup_threshold_max: 1,
        },
        { id: 'yolo', axis: 'export', label: 'YOLO', status: 'stable' },
      ],
    });
    expect(m.overlays[0]).toMatchObject({ max_k: 10000, select_max_k: 50000 });
    expect(m.dataset_exports[0]).toMatchObject({
      dedup_threshold_default: 0.98,
      dedup_threshold_min: 0,
      dedup_threshold_max: 1,
    });
    // Absent stays absent: nothing is filled in client-side.
    expect(m.dataset_exports[1]!.dedup_threshold_default).toBeUndefined();
  });
});

describe('exportDedupDefault', () => {
  const exports = [
    { id: 'single_class', label: 'S', status: 'stable', dedup_threshold_default: 0.9 },
    { id: 'yolo', label: 'Y', status: 'stable' },
  ] as DatasetExportInfo[];
  it('is the served default of the kind', () => {
    expect(exportDedupDefault(exports, 'single_class')).toBe(0.9);
  });
  it('is null for a kind that serves none, an unknown kind and no kind', () => {
    expect(exportDedupDefault(exports, 'yolo')).toBeNull();
    expect(exportDedupDefault(exports, 'nope')).toBeNull();
    expect(exportDedupDefault(exports, undefined)).toBeNull();
  });
});

describe('EMPTY_METHODS', () => {
  it('advertises nothing on any axis', () => {
    for (const list of Object.values(EMPTY_METHODS)) expect(list).toEqual([]);
  });
});

describe('settable flag on /methods entries', () => {
  it('keeps a boolean settable and drops anything else', () => {
    const parsed = parseMethodsResponse({
      strategies: [
        { id: 'ivf', axis: 'cluster', label: 'IVF', status: 'stable', settable: true },
        {
          id: 'lp',
          axis: 'detection_profile',
          label: 'LP',
          status: 'stable',
          settable: false,
        },
        { id: 'p', axis: 'prompt_pack', label: 'P', status: 'stable', settable: 'yes' },
      ],
    });
    expect(parsed.cluster_methods[0]!.settable).toBe(true);
    expect(parsed.detection_profiles[0]!.settable).toBe(false);
    expect(parsed.prompt_packs[0]!.settable).toBeUndefined();
  });
});
