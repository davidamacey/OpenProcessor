import { describe, expect, it } from 'vitest';
import {
  FALLBACK_METHODS,
  isDiverseOverlayAvailable,
  isEmbeddingVizAvailable,
  isEmbeddingVizBannerRequired,
  normalizeMethodStatus,
  parseKbMethodsResponse,
} from './strategies';
import type { OverlayInfo } from './strategies';

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
        { id: 'hdbscan', axis: 'cluster', label: 'HDBSCAN (dormant)', status: 'disabled' },
        { id: 'default', axis: 'sort', label: 'Recent first', status: 'stable', default: true },
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
        { id: 'viz_projection', axis: 'overlay', label: 'UMAP scatter', status: 'experimental' },
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
      });
    }
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
        { id: 'mistakenness', axis: 'score', label: 'Mistakenness', status: 'experimental' },
        { id: 'mistakenness', axis: 'sort', label: 'Mistakenness', status: 'experimental' },
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
      isDiverseOverlayAvailable([{ id: 'diverse', label: 'Diversity', status: 'shadow' }]),
    ).toBe(false);
  });

  it('is false when diverse is reported but disabled (OP_SELECT_DIVERSE_ENABLED off)', () => {
    expect(
      isDiverseOverlayAvailable([{ id: 'diverse', label: 'Diversity', status: 'disabled' }]),
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
      isDiverseOverlayAvailable([{ id: 'diverse', label: 'Diversity', status: 'stable' }]),
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
      { id: 'default', label: 'Recent first', status: 'stable', default: true },
    ]);
    expect(FALLBACK_METHODS.overlays).toEqual([]);
    expect(FALLBACK_METHODS.scores).toEqual([]);
  });

  it('never contains an experimental/shadow/disabled entry', () => {
    const all = [
      ...FALLBACK_METHODS.cluster_methods,
      ...FALLBACK_METHODS.review_sorts,
      ...FALLBACK_METHODS.overlays,
      ...FALLBACK_METHODS.scores,
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
