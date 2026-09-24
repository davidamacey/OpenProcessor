import { describe, expect, it } from 'vitest';
import {
  FALLBACK_METHODS,
  hasFieldCoverage,
  isDatasetExportAvailable,
  isDiverseOverlayAvailable,
  isEmbeddingVizAvailable,
  isEmbeddingVizBannerRequired,
  isPromptPackAvailable,
  isScopedAssistAvailable,
  isSemanticSearchAvailable,
  normalizeMethodStatus,
  parseKbMethodsResponse,
  selectableAxisEntries,
} from './strategies';
import type {
  DatasetExportInfo,
  DetectionProfileInfo,
  OverlayInfo,
  PromptPackInfo,
  ReviewSortInfo,
} from './strategies';

// Two fixtures backing the mock-backed verification for the whole scoped-
// assist feature (docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md
// §6). Kept in sync by eye with the JSON stubbed in
// scripts/playwright_assist_scope.py.

/** Today's real `/methods` shape — no assist axes at all. */
export const METHODS_TODAY = {
  strategies: [
    {
      id: 'ivf',
      axis: 'cluster',
      label: 'FAISS IVF-512 (production)',
      status: 'stable',
      default: true,
    },
    {
      id: 'default',
      axis: 'sort',
      label: 'Recent first',
      status: 'stable',
      default: true,
    },
    {
      id: 'representativeness',
      axis: 'sort',
      label: 'Representativeness',
      status: 'stable',
    },
    {
      id: 'yolo',
      axis: 'export',
      label: 'YOLO detection dataset export',
      status: 'stable',
      default: true,
    },
  ],
  flags: {},
};

/** Same payload as `METHODS_TODAY` plus the two assist axes — one usable
 *  entry each, plus one deliberately-shadow profile and one
 *  deliberately-disabled pack to exercise the status filter. */
export const METHODS_WITH_ASSIST_AXES = {
  strategies: [
    ...METHODS_TODAY.strategies,
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
      id: 'legacy_profile',
      axis: 'detection_profile',
      label: 'Legacy profile (mid-validation)',
      status: 'shadow',
    },
    {
      id: 'warehouse_v1',
      axis: 'prompt_pack',
      label: 'Warehouse vocabulary',
      status: 'stable',
      default: true,
    },
    {
      id: 'retired_pack',
      axis: 'prompt_pack',
      label: 'Retired prompt pack',
      status: 'disabled',
    },
  ],
  flags: {},
};

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

// The real /curation/methods wire shape (confirmed 2026-09-10 against the live
// backend, strategy_registry.py's get_registry()): one flat `strategies`
// array, each entry carrying an `axis` field (`'cluster' | 'sort' |
// 'score' | 'overlay'`), plus a top-level `flags` object this file
// doesn't consume. parseKbMethodsResponse groups by `axis` into the four
// buckets every downstream consumer (isDiverseOverlayAvailable, etc.)
// already expects — these tests exercise that grouping directly rather
// than the four-separate-top-level-arrays shape an earlier version of
// this file assumed before the real contract was confirmed.
describe('parseKbMethodsResponse', () => {
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
      flags: { op_scores_enabled: false },
    };
    const parsed = parseKbMethodsResponse(raw);
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
    const parsed = parseKbMethodsResponse({
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
    const parsed = parseKbMethodsResponse({
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
      expect(() => parseKbMethodsResponse(bad)).not.toThrow();
      const parsed = parseKbMethodsResponse(bad);
      expect(parsed).toEqual({
        cluster_methods: [],
        review_sorts: [],
        overlays: [],
        scores: [],
        dataset_exports: [],
        detection_profiles: [],
        prompt_packs: [],
      });
    }
  });

  it('defaults a missing/non-array/malformed strategies key to empty lists without throwing', () => {
    for (const bad of ['not-an-array', null, undefined, 42]) {
      const parsed = parseKbMethodsResponse({ strategies: bad });
      expect(parsed).toEqual({
        cluster_methods: [],
        review_sorts: [],
        overlays: [],
        scores: [],
        dataset_exports: [],
        detection_profiles: [],
        prompt_packs: [],
      });
    }
  });

  it("routes axis:'export' entries into dataset_exports (real T-C2 wire shape)", () => {
    // Verbatim from strategy_registry.py's _export_strategies() @ d8cb9dc.
    const parsed = parseKbMethodsResponse({
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
    const parsed = parseKbMethodsResponse({
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
    const parsed = parseKbMethodsResponse({
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
    const parsed = parseKbMethodsResponse({
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
    const parsed = parseKbMethodsResponse({
      strategies: [
        { id: 'mistakenness', axis: 'sort', label: 'Mistakenness', status: 'beta_v2' },
      ],
    });
    expect(parsed.review_sorts).toHaveLength(1);
    expect(parsed.review_sorts[0]?.status).toBe('disabled');
  });

  it('the same id may legitimately appear in more than one axis bucket (score vs sort)', () => {
    const parsed = parseKbMethodsResponse({
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

/**
 * isDiverseOverlayAvailable is the single gate both `/clusters/[id]`
 * (widening allowedIds) and `StrategyBar.svelte` (rendering the k
 * stepper) call — this is the real regression guard for "the diverse UI
 * must not render against a backend that hasn't shipped it," since this
 * repo has no component-mount test harness to assert absence in the DOM
 * directly (see StrategyBar.test.ts's header comment).
 */
describe('isDiverseOverlayAvailable', () => {
  it('is false when overlays is empty (pre-Phase-4 / Phase-0/3-only backend)', () => {
    expect(isDiverseOverlayAvailable([])).toBe(false);
  });

  it('is false when /curation/methods does not report a diverse entry at all', () => {
    const overlays: OverlayInfo[] = [
      { id: 'near_dup', label: 'Near-duplicates', status: 'stable' },
      { id: 'viz_projection', label: 'UMAP scatter', status: 'experimental' },
    ];
    expect(isDiverseOverlayAvailable(overlays)).toBe(false);
  });

  it('is false when diverse is reported but shadow (mid-validation, never selectable)', () => {
    expect(
      isDiverseOverlayAvailable([
        { id: 'diverse', label: 'Diversity', status: 'shadow' },
      ]),
    ).toBe(false);
  });

  it('is false when diverse is reported but disabled (OP_SELECT_DIVERSE_ENABLED off)', () => {
    expect(
      isDiverseOverlayAvailable([
        { id: 'diverse', label: 'Diversity', status: 'disabled' },
      ]),
    ).toBe(false);
  });

  it('is true when diverse is reported experimental', () => {
    expect(
      isDiverseOverlayAvailable([
        { id: 'diverse', label: 'Diversity (core-set)', status: 'experimental' },
      ]),
    ).toBe(true);
  });

  it('is true when diverse is reported stable', () => {
    expect(
      isDiverseOverlayAvailable([
        { id: 'diverse', label: 'Diversity', status: 'stable' },
      ]),
    ).toBe(true);
  });

  it('never throws on FALLBACK_METHODS.overlays (empty today)', () => {
    expect(isDiverseOverlayAvailable(FALLBACK_METHODS.overlays)).toBe(false);
  });
});

/**
 * isEmbeddingVizAvailable is the Phase 5 analog of isDiverseOverlayAvailable
 * — the single gate `/clusters` uses to decide whether the "Embedding
 * plot" toggle exists at all. Same case coverage, same reasoning: this
 * repo has no component-mount test harness, so this predicate (not a DOM
 * assertion) is the real regression guard.
 *
 * `'viz_projection'` is the real id (confirmed against
 * `strategy_registry.py`'s `_viz_projection_strategy`) — an earlier
 * placeholder id, `'umap_viz'`, has been corrected throughout this file.
 */
describe('isEmbeddingVizAvailable', () => {
  it('is false when overlays is empty (pre-Phase-5 backend, or OP_VIZ_PROJECTION_ENABLED off)', () => {
    expect(isEmbeddingVizAvailable([])).toBe(false);
  });

  it('is false when /curation/methods does not report a viz_projection entry at all', () => {
    const overlays: OverlayInfo[] = [
      { id: 'diverse', label: 'Diversity', status: 'stable' },
      { id: 'near_dup', label: 'Near-duplicates', status: 'experimental' },
    ];
    expect(isEmbeddingVizAvailable(overlays)).toBe(false);
  });

  it('is false when viz_projection is reported but shadow (mid-validation — the UMAP purity gate has not passed yet)', () => {
    expect(
      isEmbeddingVizAvailable([
        { id: 'viz_projection', label: 'UMAP scatter', status: 'shadow' },
      ]),
    ).toBe(false);
  });

  it('is false when viz_projection is reported but disabled (OP_VIZ_PROJECTION_ENABLED off, or the purity gate failed outright)', () => {
    expect(
      isEmbeddingVizAvailable([
        { id: 'viz_projection', label: 'UMAP scatter', status: 'disabled' },
      ]),
    ).toBe(false);
  });

  it('is true when viz_projection is reported experimental', () => {
    expect(
      isEmbeddingVizAvailable([
        { id: 'viz_projection', label: 'UMAP scatter', status: 'experimental' },
      ]),
    ).toBe(true);
  });

  it('is true when viz_projection is reported stable', () => {
    expect(
      isEmbeddingVizAvailable([
        { id: 'viz_projection', label: 'UMAP scatter', status: 'stable' },
      ]),
    ).toBe(true);
  });

  it('never throws on FALLBACK_METHODS.overlays (empty today)', () => {
    expect(isEmbeddingVizAvailable(FALLBACK_METHODS.overlays)).toBe(false);
  });
});

/**
 * isSemanticSearchAvailable (P2-14) is the gate for whether
 * SemanticSearchBox renders at all on /review and /clusters. Same
 * case coverage/reasoning as isDiverseOverlayAvailable/
 * isEmbeddingVizAvailable above.
 */
describe('isSemanticSearchAvailable', () => {
  it('is false when overlays is empty (pre-P2-14 backend, or KB flag off)', () => {
    expect(isSemanticSearchAvailable([])).toBe(false);
  });

  it('is false when /curation/methods does not report a semantic_search entry at all', () => {
    const overlays: OverlayInfo[] = [
      { id: 'diverse', label: 'Diversity', status: 'stable' },
      { id: 'viz_projection', label: 'UMAP scatter', status: 'experimental' },
    ];
    expect(isSemanticSearchAvailable(overlays)).toBe(false);
  });

  it('is false when semantic_search is reported but shadow', () => {
    expect(
      isSemanticSearchAvailable([
        { id: 'semantic_search', label: 'Text search', status: 'shadow' },
      ]),
    ).toBe(false);
  });

  it('is false when semantic_search is reported but disabled', () => {
    expect(
      isSemanticSearchAvailable([
        { id: 'semantic_search', label: 'Text search', status: 'disabled' },
      ]),
    ).toBe(false);
  });

  it('is true when semantic_search is reported experimental', () => {
    expect(
      isSemanticSearchAvailable([
        { id: 'semantic_search', label: 'Text search', status: 'experimental' },
      ]),
    ).toBe(true);
  });

  it('is true when semantic_search is reported stable', () => {
    expect(
      isSemanticSearchAvailable([
        { id: 'semantic_search', label: 'Text search', status: 'stable' },
      ]),
    ).toBe(true);
  });

  it('never throws on FALLBACK_METHODS.overlays (empty today — no semantic_search entry)', () => {
    expect(isSemanticSearchAvailable(FALLBACK_METHODS.overlays)).toBe(false);
  });
});

/**
 * isDatasetExportAvailable is the T-C3 analog of isDiverseOverlayAvailable/
 * isSemanticSearchAvailable — the single gate `/train` uses to decide
 * whether the optional single-class export panel exists at all.
 */
describe('isDatasetExportAvailable', () => {
  it('is false when dataset_exports is empty', () => {
    expect(isDatasetExportAvailable([], 'lpr')).toBe(false);
  });

  it('is true when the kind is reported stable', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'yolo', label: 'YOLO detection dataset export', status: 'stable' },
    ];
    expect(isDatasetExportAvailable(exports, 'yolo')).toBe(true);
  });

  it('is true when the kind is reported experimental', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'lpr', label: 'LPR plate dataset', status: 'experimental' },
    ];
    expect(isDatasetExportAvailable(exports, 'lpr')).toBe(true);
  });

  it('is false when the kind is reported but shadow (mid-validation, never selectable)', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'lpr', label: 'LPR plate dataset', status: 'shadow' },
    ];
    expect(isDatasetExportAvailable(exports, 'lpr')).toBe(false);
  });

  it('is false when the kind is reported but disabled', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'lpr', label: 'LPR plate dataset', status: 'disabled' },
    ];
    expect(isDatasetExportAvailable(exports, 'lpr')).toBe(false);
  });

  // Absence, not a status. OpenProcessor omits `lpr` entirely from the
  // export axis rather than advertising it disabled, because a proprietary
  // overlay the repo doesn't contain isn't "not yet, but could be later"
  // (curation_api_contract.md's `export` axis section). A consumer must
  // treat "no entry" identically to "entry at shadow/disabled".
  it('is false when the kind is absent entirely (the real lpr-on-OpenProcessor case)', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'yolo', label: 'YOLO detection dataset export', status: 'stable' },
    ];
    expect(isDatasetExportAvailable(exports, 'lpr')).toBe(false);
  });

  it('is false when a different kind is present', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'yolo', label: 'YOLO detection dataset export', status: 'stable' },
    ];
    expect(isDatasetExportAvailable(exports, 'coco')).toBe(false);
  });

  it('never throws on FALLBACK_METHODS.dataset_exports (empty today)', () => {
    expect(isDatasetExportAvailable(FALLBACK_METHODS.dataset_exports, 'lpr')).toBe(false);
  });
});

describe('isEmbeddingVizBannerRequired', () => {
  it('is false when the overlay is not available at all (empty/absent/shadow/disabled)', () => {
    expect(isEmbeddingVizBannerRequired([])).toBe(false);
    expect(
      isEmbeddingVizBannerRequired([
        {
          id: 'viz_projection',
          label: 'UMAP scatter',
          status: 'shadow',
          requires_banner: true,
        },
      ]),
    ).toBe(false);
    expect(
      isEmbeddingVizBannerRequired([
        {
          id: 'viz_projection',
          label: 'UMAP scatter',
          status: 'disabled',
          requires_banner: true,
        },
      ]),
    ).toBe(false);
  });

  it('is false when the overlay is available but does not carry requires_banner', () => {
    expect(
      isEmbeddingVizBannerRequired([
        { id: 'viz_projection', label: 'UMAP scatter', status: 'stable' },
      ]),
    ).toBe(false);
  });

  it('is false when requires_banner is explicitly false', () => {
    expect(
      isEmbeddingVizBannerRequired([
        {
          id: 'viz_projection',
          label: 'UMAP scatter',
          status: 'experimental',
          requires_banner: false,
        },
      ]),
    ).toBe(false);
  });

  it('is true when the overlay is available (stable/experimental) and requires_banner is true', () => {
    expect(
      isEmbeddingVizBannerRequired([
        {
          id: 'viz_projection',
          label: 'UMAP scatter (approximate)',
          status: 'experimental',
          requires_banner: true,
        },
      ]),
    ).toBe(true);
  });

  it('never throws on FALLBACK_METHODS.overlays (empty today)', () => {
    expect(isEmbeddingVizBannerRequired(FALLBACK_METHODS.overlays)).toBe(false);
  });
});

/**
 * hasFieldCoverage is the audit-remediation plan Phase 6 fix (P1-2/P1-3):
 * `StrategyBar.svelte`'s old local `hasCoverage()` used
 * `(entry.field_coverage ?? 0) > 0`, which treats `null`/`undefined`
 * ("coverage unknown") identically to `0` ("coverage confirmed empty").
 * Before Phase 6, no real backend ever sent `field_coverage` at all, so
 * every entry hit the `?? 0` branch and every chip/gated-sort was hidden
 * regardless of real data. These are the tests that would have caught
 * that: `field_coverage: undefined`/`null` must render, only a real `0`
 * must hide.
 */
describe('hasFieldCoverage', () => {
  it('is true when coverage is a positive count', () => {
    expect(hasFieldCoverage({ field_coverage: 124_921 })).toBe(true);
  });

  it('is false when coverage is a confirmed zero (the one real hide case)', () => {
    expect(hasFieldCoverage({ field_coverage: 0 })).toBe(false);
  });

  it('is true when coverage is null (unknown -- e.g. requires_field is null, or a transient backend failure)', () => {
    expect(hasFieldCoverage({ field_coverage: null })).toBe(true);
  });

  it('is true when coverage is undefined/absent (pre-Phase-6 backend, or the FALLBACK_METHODS/synthetic sentinel path)', () => {
    expect(hasFieldCoverage({})).toBe(true);
    expect(hasFieldCoverage({ field_coverage: undefined })).toBe(true);
  });
});

/**
 * The exact filter StrategyBar.svelte's sortOptions applies: stable/
 * experimental status AND hasFieldCoverage. Exercised here against ids
 * and coverage values lifted straight from the plan's live-verification
 * snippet (audit-remediation-plan-2026-09.md Phase 6) so this test would
 * have caught the bug against the real reported numbers, not a synthetic
 * stand-in.
 */
describe('sort dropdown filtering (mirrors StrategyBar.svelte sortOptions)', () => {
  function selectable(sorts: ReviewSortInfo[]): string[] {
    return sorts
      .filter(
        (s) =>
          (s.status === 'stable' || s.status === 'experimental') && hasFieldCoverage(s),
      )
      .map((s) => s.id);
  }

  it('omits sorts with real, confirmed-zero coverage; keeps ones with real positive coverage', () => {
    const sorts: ReviewSortInfo[] = [
      { id: 'recent', label: 'Recently updated', status: 'stable', field_coverage: null },
      {
        id: 'representativeness',
        label: 'Representativeness',
        status: 'stable',
        requires_field: 'cluster_distance',
        field_coverage: 124_921,
      },
      {
        id: 'atypicality',
        label: 'Atypicality',
        status: 'stable',
        requires_field: 'cluster_distance',
        field_coverage: 124_921,
      },
      {
        id: 'uncertainty_entropy',
        label: 'Uncertainty (probe entropy)',
        status: 'stable',
        requires_field: 'probe_pred_entropy',
        field_coverage: 0,
      },
      {
        id: 'disagreement_entropy_asc',
        label: 'Model disagreement',
        status: 'stable',
        requires_field: 'probe_pred_entropy',
        field_coverage: 0,
      },
      {
        id: 'mistakenness',
        label: 'Mistakenness · beta',
        status: 'experimental',
        requires_field: 'mistakenness_score',
        field_coverage: 0,
      },
    ];

    expect(selectable(sorts)).toEqual(['recent', 'representativeness', 'atypicality']);
  });

  it('keeps a sort whose coverage is unknown (null) rather than hiding it like a confirmed zero', () => {
    const sorts: ReviewSortInfo[] = [
      {
        id: 'plate_score',
        label: 'Plate detection score',
        status: 'stable',
        requires_field: 'plate_score',
        field_coverage: null,
      },
    ];
    expect(selectable(sorts)).toEqual(['plate_score']);
  });
});

describe('FALLBACK_METHODS', () => {
  it('is a stable-only list matching what is actually implemented today', () => {
    expect(FALLBACK_METHODS.cluster_methods).toEqual([
      {
        id: 'ivf',
        label: 'FAISS IVF-512 (production)',
        status: 'stable',
        default: true,
      },
    ]);
    expect(FALLBACK_METHODS.review_sorts).toEqual([
      {
        id: 'default',
        label: 'Recent first',
        status: 'stable',
        default: true,
        field_coverage: null,
      },
    ]);
    expect(FALLBACK_METHODS.overlays).toEqual([]);
    expect(FALLBACK_METHODS.scores).toEqual([]);
    expect(FALLBACK_METHODS.dataset_exports).toEqual([]);
    expect(FALLBACK_METHODS.detection_profiles).toEqual([]);
    expect(FALLBACK_METHODS.prompt_packs).toEqual([]);
  });

  it('never contains an experimental/shadow/disabled entry', () => {
    const all = [
      ...FALLBACK_METHODS.cluster_methods,
      ...FALLBACK_METHODS.review_sorts,
      ...FALLBACK_METHODS.overlays,
      ...FALLBACK_METHODS.scores,
      ...FALLBACK_METHODS.dataset_exports,
      ...FALLBACK_METHODS.detection_profiles,
      ...FALLBACK_METHODS.prompt_packs,
    ];
    expect(all.every((m) => m.status === 'stable')).toBe(true);
  });

  // FALLBACK_METHODS is the already-parsed *output* shape (four buckets),
  // not a valid raw /curation/methods *input* (the real wire format is a flat
  // `strategies` array with an `axis` field per entry — see the header
  // comment on parseKbMethodsResponse's describe block above). It is
  // never fed back through the parser in real usage (api.ts's getMethods
  // returns it directly on a fetch failure), so there is no round-trip
  // invariant to assert here anymore.
});

describe('selectableAxisEntries', () => {
  it('keeps a stable entry', () => {
    const entries: DetectionProfileInfo[] = [{ id: 'a', label: 'A', status: 'stable' }];
    expect(selectableAxisEntries(entries)).toEqual(entries);
  });

  it('keeps an experimental entry', () => {
    const entries: DetectionProfileInfo[] = [
      { id: 'a', label: 'A', status: 'experimental' },
    ];
    expect(selectableAxisEntries(entries)).toEqual(entries);
  });

  it('drops a shadow entry', () => {
    expect(
      selectableAxisEntries([
        { id: 'a', label: 'A', status: 'shadow' },
      ] as DetectionProfileInfo[]),
    ).toEqual([]);
  });

  it('drops a disabled entry', () => {
    expect(
      selectableAxisEntries([
        { id: 'a', label: 'A', status: 'disabled' },
      ] as DetectionProfileInfo[]),
    ).toEqual([]);
  });

  it('is empty in, empty out', () => {
    expect(selectableAxisEntries([])).toEqual([]);
  });

  it('preserves input order', () => {
    const entries: DetectionProfileInfo[] = [
      { id: 'b', label: 'B', status: 'experimental' },
      { id: 'a', label: 'A', status: 'stable' },
    ];
    expect(selectableAxisEntries(entries).map((e) => e.id)).toEqual(['b', 'a']);
  });
});

describe('isPromptPackAvailable', () => {
  it('is false for an empty list', () => {
    expect(isPromptPackAvailable([])).toBe(false);
  });

  it('is true with one stable entry', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'stable' }])).toBe(true);
  });

  it('is true with one experimental entry', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'experimental' }])).toBe(
      true,
    );
  });

  it('is false with only shadow entries', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'shadow' }])).toBe(
      false,
    );
  });

  it('is false with only disabled entries', () => {
    expect(isPromptPackAvailable([{ id: 'a', label: 'A', status: 'disabled' }])).toBe(
      false,
    );
  });

  it('is true for a mixed list with at least one usable entry', () => {
    expect(
      isPromptPackAvailable([
        { id: 'a', label: 'A', status: 'disabled' },
        { id: 'b', label: 'B', status: 'experimental' },
      ]),
    ).toBe(true);
  });

  it('is false for FALLBACK_METHODS.prompt_packs', () => {
    expect(isPromptPackAvailable(FALLBACK_METHODS.prompt_packs)).toBe(false);
  });
});

/**
 * The gate that matters most: whether `AutoLabelPanel` renders
 * `<AssistScopeBar>` at all. Written against the two real fixtures
 * (§6 of the plan) rather than synthetic entries, because the
 * "must degrade to fully invisible against today's real backend" case
 * is the regression guard for the whole feature.
 */
describe('isScopedAssistAvailable', () => {
  it('is false for FALLBACK_METHODS (the /methods-404 path)', () => {
    expect(isScopedAssistAvailable(FALLBACK_METHODS)).toBe(false);
  });

  it("is false for today's real backend shape (METHODS_TODAY) — must degrade to fully invisible", () => {
    const parsed = parseKbMethodsResponse(METHODS_TODAY);
    expect(isScopedAssistAvailable(parsed)).toBe(false);
  });

  it('is true once the backend advertises the assist axes (METHODS_WITH_ASSIST_AXES)', () => {
    const parsed = parseKbMethodsResponse(METHODS_WITH_ASSIST_AXES);
    expect(isScopedAssistAvailable(parsed)).toBe(true);
  });

  // Detection profiles never gate the bar: a deployment advertising
  // profiles but no usable prompt pack gets no scope bar.
  it('is false when no prompt pack is usable', () => {
    const packs: PromptPackInfo[] = [{ id: 'legacy', label: 'Legacy', status: 'shadow' }];
    expect(isScopedAssistAvailable({ prompt_packs: packs })).toBe(false);
    expect(isScopedAssistAvailable({ prompt_packs: [] })).toBe(false);
  });

  it('is true when a prompt pack is usable', () => {
    const packs: PromptPackInfo[] = [
      { id: 'warehouse_v1', label: 'Warehouse', status: 'stable' },
    ];
    expect(isScopedAssistAvailable({ prompt_packs: packs })).toBe(true);
  });
});

describe('settable flag on /methods entries', () => {
  it('keeps a boolean settable and drops anything else', () => {
    const parsed = parseKbMethodsResponse({
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
