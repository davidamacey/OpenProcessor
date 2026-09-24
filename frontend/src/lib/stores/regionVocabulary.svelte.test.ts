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
    { id: 'lpr_nanov11_640', label: 'LPR', role: 'detector', filterable: true },
    { id: 'sam3', label: 'SAM3', role: 'segmenter', filterable: true },
    { id: 'human', label: 'Human', role: 'human', filterable: false },
  ],
  region_sources: [
    { id: 'lpr_frozen_test_sample', label: 'LPR frozen test', role: 'human' },
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
      'lpr_nanov11_640',
      'sam3',
    ]);
  });

  it('labelFor/roleFor resolve across all three lists, and unknown ids render verbatim', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await regionVocabularyStore.init();

    expect(regionVocabularyStore.labelFor('lpr_nanov11_640')).toBe('LPR');
    expect(regionVocabularyStore.roleFor('lpr_nanov11_640')).toBe('detector');
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
