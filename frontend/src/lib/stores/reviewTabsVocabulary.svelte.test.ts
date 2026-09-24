/**
 * reviewTabsVocabularyStore — loads GET {API_PREFIX}/review/tabs once,
 * degrading to an empty map (never a throw) on failure so every caller's
 * `labelFor(id, fallback)` falls back to the tab's own static label
 * (W0 naming-sweep finding m9). The store is a module singleton, so every
 * test resets it first.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { reviewTabsVocabularyStore } from './reviewTabsVocabulary.svelte';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function resetStore(): void {
  reviewTabsVocabularyStore.list = [];
  reviewTabsVocabularyStore.loaded = false;
}

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const PAYLOAD = {
  tabs: [
    { id: 'all', label: 'All crops', description: 'Every crop in the pool' },
    { id: 'regions', label: 'License plates', description: 'Plate review queue' },
    { id: 'mismatches', label: 'VLM disagreements' },
  ],
};

describe('reviewTabsVocabularyStore.init', () => {
  it('populates the list from the response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.list).toEqual(PAYLOAD.tabs);
    expect(reviewTabsVocabularyStore.loaded).toBe(true);
  });

  it('labelFor returns the served label when present, falling back otherwise', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.labelFor('all', 'All')).toBe('All crops');
    expect(reviewTabsVocabularyStore.labelFor('regions', 'Plates')).toBe(
      'License plates',
    );
    // Not present in the served list at all — static fallback.
    expect(
      reviewTabsVocabularyStore.labelFor('coco_blind_spots', 'COCO Blind Spots'),
    ).toBe('COCO Blind Spots');
  });

  it('descriptionFor returns the served description or null', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.descriptionFor('all')).toBe(
      'Every crop in the pool',
    );
    // Present but no description field.
    expect(reviewTabsVocabularyStore.descriptionFor('mismatches')).toBeNull();
    expect(reviewTabsVocabularyStore.descriptionFor('uncertainty')).toBeNull();
  });

  it('degrades to an empty list (all lookups fall back) when the endpoint fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(500, { detail: 'boom' })),
    );

    await expect(reviewTabsVocabularyStore.init()).resolves.toBeUndefined();

    expect(reviewTabsVocabularyStore.list).toEqual([]);
    expect(reviewTabsVocabularyStore.labelFor('all', 'All')).toBe('All');
    expect(reviewTabsVocabularyStore.loaded).toBe(true);
  });

  it('a second call is a no-op once loaded (fetch called exactly once)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD));
    vi.stubGlobal('fetch', fetchMock);

    await reviewTabsVocabularyStore.init();
    await reviewTabsVocabularyStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
