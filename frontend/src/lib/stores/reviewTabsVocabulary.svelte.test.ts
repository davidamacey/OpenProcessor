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

    expect(reviewTabsVocabularyStore.list).toEqual(
      PAYLOAD.tabs.map((t) => ({ ...t, filter_specs: [] })),
    );
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

// dq-queues cutover (2026-09-24): GET {API_PREFIX}/review/tabs now serves
// per-tab `filters`/`filter_defaults` — the /review filter bar renders
// only the controls a tab's served `filters` lists, seeded from
// `filter_defaults`.
const FILTERS_PAYLOAD = {
  tabs: [
    {
      id: 'primary_low_conf',
      label: 'Primary low-conf',
      filters: ['class_id', 'source', 'max_rank'],
      filter_defaults: { max_rank: 2 },
    },
    {
      id: 'regions',
      label: 'Plates',
      filters: ['class_id', 'source', 'text'],
      filter_defaults: {},
    },
    // No `filters` at all — an older backend response shape.
    { id: 'all', label: 'All' },
  ],
};

describe('reviewTabsVocabularyStore.filterSupported / filtersFor / filterDefault', () => {
  it('filtersFor returns the served list for a tab that has one', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, FILTERS_PAYLOAD)));
    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.filtersFor('primary_low_conf')).toEqual([
      'class_id',
      'source',
      'max_rank',
    ]);
  });

  it('filterSupported is true only for a param in the served list', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, FILTERS_PAYLOAD)));
    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.filterSupported('regions', 'text')).toBe(true);
    expect(reviewTabsVocabularyStore.filterSupported('regions', 'max_rank')).toBe(false);
  });

  it('filterSupported defaults to true (unknown) when the tab has no served filters list', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, FILTERS_PAYLOAD)));
    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.filterSupported('all', 'max_rank')).toBe(true);
    expect(reviewTabsVocabularyStore.filterSupported('nonexistent_tab', 'max_rank')).toBe(
      true,
    );
  });

  it('filterDefault returns the served default value, null when absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, FILTERS_PAYLOAD)));
    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.filterDefault('primary_low_conf', 'max_rank')).toBe(
      2,
    );
    expect(reviewTabsVocabularyStore.filterDefault('regions', 'max_rank')).toBeNull();
  });
});

// 840beb8 adoption: GET {API_PREFIX}/review/tabs now also serves each tab's
// self-describing enum filter_specs (e.g. Plates' region_status) — the
// generic served-enum filter bar renders one <select> per entry with zero
// param-specific code.
const FILTER_SPECS_PAYLOAD = {
  tabs: [
    {
      id: 'regions',
      label: 'Plates',
      filters: ['text', 'region_status'],
      filter_defaults: { region_status: 'all' },
      filter_specs: [
        {
          param: 'region_status',
          kind: 'enum',
          label: 'Status',
          options: [
            { value: 'all', label: 'All (accepted + rejected candidates)' },
            { value: 'detected', label: 'Detected only' },
            { value: 'verify_rejected', label: 'Verifier-rejected candidates only' },
          ],
        },
      ],
    },
    // No filter_specs at all — every other tab today.
    { id: 'all', label: 'All' },
  ],
};

describe('reviewTabsVocabularyStore.filterSpecsFor', () => {
  it('returns the served specs for a tab that has one', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, FILTER_SPECS_PAYLOAD)),
    );
    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.filterSpecsFor('regions')).toEqual(
      FILTER_SPECS_PAYLOAD.tabs[0].filter_specs,
    );
  });

  it('returns [] for a tab with none, an unknown tab, or before load', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, FILTER_SPECS_PAYLOAD)),
    );

    expect(reviewTabsVocabularyStore.filterSpecsFor('regions')).toEqual([]);

    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.filterSpecsFor('all')).toEqual([]);
    expect(reviewTabsVocabularyStore.filterSpecsFor('nonexistent_tab')).toEqual([]);
  });
});
