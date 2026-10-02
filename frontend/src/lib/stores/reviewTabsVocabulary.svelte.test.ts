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
  reviewTabsVocabularyStore.emptyState = null;
}

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A full served `ReviewTab` entry. */
function tab(id: string, label: string, over: Record<string, unknown> = {}) {
  return {
    id,
    label,
    description: '',
    filters: [],
    filter_defaults: {},
    filter_specs: [],
    ...over,
  };
}

const EMPTY_STATE = {
  has_probe_predictions: false,
  has_item_scores: true,
  has_imported_labels: false,
};

const PAYLOAD = {
  tabs: [
    tab('all', 'All crops', { description: 'Every crop in the pool' }),
    tab('regions', 'Widget tags', { description: 'Region review queue' }),
    tab('mismatches', 'VLM disagreements'),
  ],
  empty_state: EMPTY_STATE,
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
    expect(reviewTabsVocabularyStore.labelFor('regions', 'Regions')).toBe('Widget tags');
    // Not present in the served list at all — static fallback.
    expect(
      reviewTabsVocabularyStore.labelFor(
        'classifier_blind_spots',
        'Classifier Blind Spots',
      ),
    ).toBe('Classifier Blind Spots');
  });

  it('descriptionFor returns the served description or null', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.descriptionFor('all')).toBe(
      'Every crop in the pool',
    );
    // Present with an empty description.
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

  it('a second call is a no-op once loaded (no further fetches)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD));
    vi.stubGlobal('fetch', fetchMock);

    await reviewTabsVocabularyStore.init();
    const callsAfterFirstInit = fetchMock.mock.calls.length;
    await reviewTabsVocabularyStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(callsAfterFirstInit);
  });
});

// #36 item 9: GET {API_PREFIX}/review/tabs also carries a top-level
// empty_state.
describe('reviewTabsVocabularyStore.emptyState', () => {
  it('populates emptyState from the served empty_state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));

    await reviewTabsVocabularyStore.init();

    expect(reviewTabsVocabularyStore.emptyState).toEqual(EMPTY_STATE);
  });
});

describe('reviewTabsVocabularyStore.hasEntry (W10: the imported tab is served-only)', () => {
  it('is true only for an endpoint id the served vocabulary lists', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          tabs: [tab('all', 'All'), tab('imported', 'Imported labels')],
          empty_state: EMPTY_STATE,
        }),
      ),
    );
    expect(reviewTabsVocabularyStore.hasEntry('imported')).toBe(false);
    await reviewTabsVocabularyStore.init();
    expect(reviewTabsVocabularyStore.hasEntry('imported')).toBe(true);
    expect(reviewTabsVocabularyStore.hasEntry('all')).toBe(true);
    expect(reviewTabsVocabularyStore.hasEntry('nope')).toBe(false);
  });

  it('is false for imported when the vocabulary does not list it, and after a failed load', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, PAYLOAD)));
    await reviewTabsVocabularyStore.init();
    expect(reviewTabsVocabularyStore.hasEntry('imported')).toBe(false);
    resetStore();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(500, { detail: 'x' })));
    await reviewTabsVocabularyStore.init();
    expect(reviewTabsVocabularyStore.hasEntry('imported')).toBe(false);
  });
});

// dq-queues cutover (2026-09-24): GET {API_PREFIX}/review/tabs now serves
// per-tab `filters`/`filter_defaults` — the /review filter bar renders
// only the controls a tab's served `filters` lists, seeded from
// `filter_defaults`.
const FILTERS_PAYLOAD = {
  tabs: [
    tab('primary_low_conf', 'Primary low-conf', {
      filters: ['class_id', 'source', 'max_rank'],
      filter_defaults: { max_rank: 2 },
    }),
    tab('regions', 'Regions', { filters: ['class_id', 'source', 'text'] }),
  ],
  empty_state: EMPTY_STATE,
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

  it('filterSupported defaults to true (unknown) for a tab with no served entry', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, FILTERS_PAYLOAD)));
    await reviewTabsVocabularyStore.init();

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

// 3f1a11e adoption: GET {API_PREFIX}/review/tabs now also serves each tab's
// self-describing enum filter_specs (e.g. Regions' region_status) — the
// generic served-enum filter bar renders one <select> per entry with zero
// param-specific code.
const FILTER_SPECS_PAYLOAD = {
  tabs: [
    tab('regions', 'Regions', {
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
    }),
    // No filter_specs — every other tab today.
    tab('all', 'All'),
  ],
  empty_state: EMPTY_STATE,
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
