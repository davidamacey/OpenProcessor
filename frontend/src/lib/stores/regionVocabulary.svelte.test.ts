/**
 * regionVocabularyStore — loads GET {API_PREFIX}/regions/vocabulary once,
 * degrading to empty lists (never a throw) on failure, so ProvenanceChip
 * and SlotGallery's detector filter always have a defined input
 * (W0 naming-sweep finding m9). The store is a module singleton, so every
 * test resets it first.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { regionVocabularyStore } from './regionVocabulary.svelte';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function resetStore(): void {
  regionVocabularyStore.detectors = [];
  regionVocabularyStore.regionSources = [];
  regionVocabularyStore.chainActors = [];
  regionVocabularyStore.rejectionReasons = [];
  regionVocabularyStore.loaded = false;
}

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const PAYLOAD = {
  detectors: [
    { id: 'tag_detector_v1', label: 'Tag detector', role: 'detector', filterable: true },
    { id: 'sam3', label: 'SAM3', role: 'segmenter', filterable: true },
    { id: 'human', label: 'Human', role: 'human', filterable: false },
  ],
  region_sources: [
    { id: 'tag_holdout_sample', label: 'Tag holdout sample', role: 'human' },
  ],
  chain_actors: [{ id: 'gemma-4-e4b', label: 'Gemma', role: 'verifier' }],
};

describe('regionVocabularyStore.init', () => {
  it('populates detectors/region_sources/chain_actors from the response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await regionVocabularyStore.init();

    expect(regionVocabularyStore.detectors).toEqual(PAYLOAD.detectors);
    expect(regionVocabularyStore.regionSources).toEqual(PAYLOAD.region_sources);
    expect(regionVocabularyStore.chainActors).toEqual(PAYLOAD.chain_actors);
    expect(regionVocabularyStore.loaded).toBe(true);
  });

  it('filterableDetectors is only the detectors with filterable=true', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await regionVocabularyStore.init();

    expect(regionVocabularyStore.filterableDetectors.map((d) => d.id)).toEqual([
      'tag_detector_v1',
      'sam3',
    ]);
  });

  it('labelFor/roleFor resolve across all three lists, and unknown ids render verbatim', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await regionVocabularyStore.init();

    expect(regionVocabularyStore.labelFor('tag_detector_v1')).toBe('Tag detector');
    expect(regionVocabularyStore.roleFor('tag_detector_v1')).toBe('detector');
    // chain_actors entry
    expect(regionVocabularyStore.labelFor('gemma-4-e4b')).toBe('Gemma');
    expect(regionVocabularyStore.roleFor('gemma-4-e4b')).toBe('verifier');
    // unknown id — verbatim, neutral (null role)
    expect(regionVocabularyStore.labelFor('some_unknown_id')).toBe('some_unknown_id');
    expect(regionVocabularyStore.roleFor('some_unknown_id')).toBeNull();
    // absent id
    expect(regionVocabularyStore.labelFor(null)).toBe('—');
    expect(regionVocabularyStore.roleFor(null)).toBeNull();
  });

  it('degrades to empty lists (not a throw) when the endpoint fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(500, { detail: 'boom' })),
    );

    await expect(regionVocabularyStore.init()).resolves.toBeUndefined();

    expect(regionVocabularyStore.detectors).toEqual([]);
    expect(regionVocabularyStore.filterableDetectors).toEqual([]);
    expect(regionVocabularyStore.loaded).toBe(true);
  });

  it('a second call is a no-op once loaded (fetch called exactly once)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD));
    vi.stubGlobal('fetch', fetchMock);

    await regionVocabularyStore.init();
    await regionVocabularyStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

// openprocessor fix #29 / 840beb8 adoption: region_rejection_reason gets a
// real labeled vocabulary — exact matches, prefix matches (with
// {detail} substitution), and an unmatched value rendering verbatim
// (never titlecased — that placeholder only ever covered text_choices/
// invalid_reasons, which still have no served labels).
const REJECTION_PAYLOAD = {
  ...PAYLOAD,
  rejection_reasons: [
    {
      id: 'region_visible_elsewhere',
      label: 'Verifier: the box is wrong (region is elsewhere)',
      kind: 'model_verdict',
      match: 'exact',
      label_template: null,
    },
    {
      id: 'sanity_reject:',
      label: 'Box failed the geometry check',
      kind: 'automatic',
      match: 'prefix',
      label_template: 'Box failed the geometry check ({detail})',
    },
    {
      id: 'verifier_no_verdict',
      label: 'Verifier gave no verdict — needs human review',
      kind: 'needs_human',
      match: 'exact',
      label_template: null,
    },
  ],
};

describe('regionVocabularyStore.rejectionReasonLabel / rejectionReasonKind', () => {
  it('resolves an exact match to its served label and kind', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, REJECTION_PAYLOAD)),
    );
    await regionVocabularyStore.init();

    expect(regionVocabularyStore.rejectionReasonLabel('region_visible_elsewhere')).toBe(
      'Verifier: the box is wrong (region is elsewhere)',
    );
    expect(regionVocabularyStore.rejectionReasonKind('region_visible_elsewhere')).toBe(
      'model_verdict',
    );
  });

  it('resolves a prefix match, filling {detail} from the rest of the stored value', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, REJECTION_PAYLOAD)),
    );
    await regionVocabularyStore.init();

    expect(
      regionVocabularyStore.rejectionReasonLabel('sanity_reject:degenerate_zero_size'),
    ).toBe('Box failed the geometry check (degenerate_zero_size)');
    expect(
      regionVocabularyStore.rejectionReasonKind('sanity_reject:degenerate_zero_size'),
    ).toBe('automatic');
  });

  it('a needs_human reason resolves without ever being worded as a rejection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, REJECTION_PAYLOAD)),
    );
    await regionVocabularyStore.init();

    const label = regionVocabularyStore.rejectionReasonLabel('verifier_no_verdict');
    expect(label).toBe('Verifier gave no verdict — needs human review');
    expect(label.toLowerCase()).not.toContain('rejected');
    expect(regionVocabularyStore.rejectionReasonKind('verifier_no_verdict')).toBe(
      'needs_human',
    );
  });

  it('an unmatched value renders verbatim (not titlecased) and kind is null', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, REJECTION_PAYLOAD)),
    );
    await regionVocabularyStore.init();

    // Older free-text human reason, and the documented bare-gate-name
    // case (an older row with no `sanity_reject:` prefix at all).
    expect(regionVocabularyStore.rejectionReasonLabel('degenerate_zero_size')).toBe(
      'degenerate_zero_size',
    );
    expect(regionVocabularyStore.rejectionReasonKind('degenerate_zero_size')).toBeNull();
  });

  it('absent id renders "—" for the label and null for the kind', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, REJECTION_PAYLOAD)),
    );
    await regionVocabularyStore.init();

    expect(regionVocabularyStore.rejectionReasonLabel(null)).toBe('—');
    expect(regionVocabularyStore.rejectionReasonKind(null)).toBeNull();
    expect(regionVocabularyStore.rejectionReasonKind(undefined)).toBeNull();
  });
});
