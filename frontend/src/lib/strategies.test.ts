import { describe, expect, it } from 'vitest';
import {
  FALLBACK_METHODS,
  isDiverseOverlayAvailable,
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

describe('parseKbMethodsResponse', () => {
  it('parses a well-formed full payload into typed lists', () => {
    const raw = {
      cluster_methods: [
        {
          id: 'ivf',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
        { id: 'hdbscan', label: 'HDBSCAN (dormant)', status: 'disabled' },
      ],
      review_sorts: [
        { id: 'default', label: 'Recent first', status: 'stable', default: true },
        {
          id: 'uncertainty',
          label: 'Uncertainty margin',
          status: 'experimental',
          requires_field: 'probe_pred_margin',
          field_coverage: 0.42,
        },
      ],
      overlays: [{ id: 'umap_viz', label: 'UMAP scatter', status: 'shadow' }],
      scores: [
        {
          id: 'uniqueness',
          label: 'Uniqueness (kNN)',
          status: 'experimental',
          version: '1',
          field_coverage: 0.9,
        },
      ],
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
      id: 'uncertainty',
      requires_field: 'probe_pred_margin',
      field_coverage: 0.42,
    });
    expect(parsed.overlays).toHaveLength(1);
    expect(parsed.overlays[0]?.status).toBe('shadow');
    expect(parsed.scores[0]).toMatchObject({ id: 'uniqueness', version: '1' });
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

  it('defaults a missing/non-array registry key to an empty list without throwing', () => {
    const parsed = parseKbMethodsResponse({
      cluster_methods: [{ id: 'ivf', label: 'IVF', status: 'stable' }],
      review_sorts: 'not-an-array',
      // overlays omitted entirely
      scores: null,
    });
    expect(parsed.cluster_methods).toHaveLength(1);
    expect(parsed.review_sorts).toEqual([]);
    expect(parsed.overlays).toEqual([]);
    expect(parsed.scores).toEqual([]);
  });

  it('drops entries missing a usable id or label instead of crashing the whole parse', () => {
    const parsed = parseKbMethodsResponse({
      cluster_methods: [
        { id: 'ivf', label: 'FAISS IVF-512', status: 'stable' },
        { label: 'no id' },
        { id: 'no-label' },
        { id: '', label: 'empty id' },
        { id: 123, label: 'non-string id' },
        null,
        'garbage',
        42,
      ],
    });
    expect(parsed.cluster_methods).toEqual([
      { id: 'ivf', label: 'FAISS IVF-512', status: 'stable', default: undefined },
    ]);
  });

  it('carries an unrecognized-but-well-formed id through untouched (forward-tolerant)', () => {
    const parsed = parseKbMethodsResponse({
      cluster_methods: [
        { id: 'some_future_method_v9', label: 'Future Method', status: 'experimental' },
      ],
    });
    expect(parsed.cluster_methods[0]?.id).toBe('some_future_method_v9');
    expect(parsed.cluster_methods[0]?.status).toBe('experimental');
  });

  it('normalizes an unrecognized status on a real entry to disabled rather than throwing', () => {
    const parsed = parseKbMethodsResponse({
      review_sorts: [{ id: 'mistakenness', label: 'Mistakenness', status: 'beta_v2' }],
    });
    expect(parsed.review_sorts).toHaveLength(1);
    expect(parsed.review_sorts[0]?.status).toBe('disabled');
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
      { id: 'umap_viz', label: 'UMAP scatter', status: 'experimental' },
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

  it('round-trips through parseKbMethodsResponse unchanged', () => {
    expect(parseKbMethodsResponse(FALLBACK_METHODS)).toEqual(FALLBACK_METHODS);
  });
});
