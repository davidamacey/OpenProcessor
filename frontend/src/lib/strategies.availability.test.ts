import { describe, expect, it } from 'vitest';
import {
  EMPTY_METHODS,
  hasFieldCoverage,
  isDatasetExportAvailable,
  isDiverseOverlayAvailable,
  isEmbeddingVizAvailable,
  isEmbeddingVizBannerRequired,
  isSemanticSearchAvailable,
} from '$lib/strategies';
import type { DatasetExportInfo, OverlayInfo, ReviewSortInfo } from '$lib/strategies';

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

  it('is false when {API_PREFIX}/methods does not report a diverse entry at all', () => {
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

  it('never throws on EMPTY_METHODS.overlays (empty today)', () => {
    expect(isDiverseOverlayAvailable(EMPTY_METHODS.overlays)).toBe(false);
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

  it('is false when {API_PREFIX}/methods does not report a viz_projection entry at all', () => {
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

  it('never throws on EMPTY_METHODS.overlays (empty today)', () => {
    expect(isEmbeddingVizAvailable(EMPTY_METHODS.overlays)).toBe(false);
  });
});

/**
 * isSemanticSearchAvailable (P2-14) is the gate for whether
 * SemanticSearchBox renders at all on /review and /clusters. Same
 * case coverage/reasoning as isDiverseOverlayAvailable/
 * isEmbeddingVizAvailable above.
 */
describe('isSemanticSearchAvailable', () => {
  it('is false when overlays is empty (pre-P2-14 backend, or its OP_ flag off)', () => {
    expect(isSemanticSearchAvailable([])).toBe(false);
  });

  it('is false when {API_PREFIX}/methods does not report a semantic_search entry at all', () => {
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

  it('never throws on EMPTY_METHODS.overlays (empty today — no semantic_search entry)', () => {
    expect(isSemanticSearchAvailable(EMPTY_METHODS.overlays)).toBe(false);
  });
});

/**
 * isDatasetExportAvailable is the T-C3 analog of isDiverseOverlayAvailable/
 * isSemanticSearchAvailable — the single gate `/train` uses to decide
 * whether the optional single-class export panel exists at all.
 */
describe('isDatasetExportAvailable', () => {
  it('is false when dataset_exports is empty', () => {
    expect(isDatasetExportAvailable([], 'single_class')).toBe(false);
  });

  it('is true when the kind is reported stable', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'yolo', label: 'YOLO detection dataset export', status: 'stable' },
    ];
    expect(isDatasetExportAvailable(exports, 'yolo')).toBe(true);
  });

  it('is true when the kind is reported experimental', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'single_class', label: 'Single-class dataset', status: 'experimental' },
    ];
    expect(isDatasetExportAvailable(exports, 'single_class')).toBe(true);
  });

  it('is false when the kind is reported but shadow (mid-validation, never selectable)', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'single_class', label: 'Single-class dataset', status: 'shadow' },
    ];
    expect(isDatasetExportAvailable(exports, 'single_class')).toBe(false);
  });

  it('is false when the kind is reported but disabled', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'single_class', label: 'Single-class dataset', status: 'disabled' },
    ];
    expect(isDatasetExportAvailable(exports, 'single_class')).toBe(false);
  });

  // Absence, not a status. OpenProcessor omits an unservable kind entirely from the
  // export axis rather than advertising it disabled, because a proprietary
  // overlay the repo doesn't contain isn't "not yet, but could be later"
  // (curation_api_contract.md's `export` axis section). A consumer must
  // treat "no entry" identically to "entry at shadow/disabled".
  it('is false when the kind is absent entirely (an export kind this backend does not serve)', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'yolo', label: 'YOLO detection dataset export', status: 'stable' },
    ];
    expect(isDatasetExportAvailable(exports, 'single_class')).toBe(false);
  });

  it('is false when a different kind is present', () => {
    const exports: DatasetExportInfo[] = [
      { id: 'yolo', label: 'YOLO detection dataset export', status: 'stable' },
    ];
    expect(isDatasetExportAvailable(exports, 'coco')).toBe(false);
  });

  it('never throws on EMPTY_METHODS.dataset_exports (empty today)', () => {
    expect(isDatasetExportAvailable(EMPTY_METHODS.dataset_exports, 'single_class')).toBe(
      false,
    );
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

  it('never throws on EMPTY_METHODS.overlays (empty today)', () => {
    expect(isEmbeddingVizBannerRequired(EMPTY_METHODS.overlays)).toBe(false);
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

  it('is true when coverage is undefined/absent (the synthetic sentinel path)', () => {
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
        id: 'region_score',
        label: 'Region detection score',
        status: 'stable',
        requires_field: 'region_score',
        field_coverage: null,
      },
    ];
    expect(selectable(sorts)).toEqual(['region_score']);
  });
});
