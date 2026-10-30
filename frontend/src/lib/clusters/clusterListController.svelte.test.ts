/**
 * The /clusters card grid sends the served sort and filters unchanged,
 * renders the served order verbatim, and asks for thumbnails through the
 * representatives window of the same list call (no per-card requests).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

vi.mock('$lib/api_clusters', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_clusters')>('$lib/api_clusters');
  return { ...actual, getClusters: vi.fn() };
});
import { getClusters, type ClustersResponse } from '$lib/api_clusters';
import { createClusterListController } from './clusterListController.svelte';
import { EMPTY_CLUSTER_FILTER } from './clusterListFilter';
import { filterPersistKey } from './persistedFilter';
import { CLUSTER_SORT_IDS } from '$lib/types_clusterList';
import type { Cluster } from '$lib/types';

function card(id: number, extra: Partial<Cluster> = {}): Cluster {
  return {
    id,
    cluster_kind: 'candidate',
    size: 10,
    validated_count: 0,
    dominant_class_id: null,
    dominant_class_name: null,
    dominant_pct: null,
    purity: 0.5,
    purity_tier: 'mixed',
    promotable: false,
    core_similarity_min: null,
    is_unlabeled: true,
    representative_crop_ids: [],
    has_subclusters: false,
    n_subclusters: 0,
    subcluster_ids: [],
    updated_at: null,
    ...extra,
  };
}

function response(
  items: Cluster[],
  extra: Partial<ClustersResponse> = {},
): ClustersResponse {
  return {
    items,
    total: items.length,
    page: 1,
    page_size: items.length,
    purity_thresholds: null,
    representatives_offset: 0,
    representatives_limit: 24,
    sort: 'size_desc',
    sort_options: [{ id: 'size_desc', label: 'Largest first' }],
    flag_vocabulary: [{ id: 'unvalidated', label: 'Unvalidated', description: 'd' }],
    truncated: false,
    ...extra,
  };
}

const withReps = (c: Cluster): Cluster => ({
  ...c,
  representative_crop_ids: [`crop-${c.id}`],
});

let cleanup: (() => void) | undefined;
let slotCards: Cluster[] = [];

function wire(opts: { params?: string; writeUrl?: (...a: unknown[]) => void } = {}) {
  let list!: ReturnType<typeof createClusterListController>;
  cleanup = $effect.root(() => {
    list = createClusterListController({
      classFilter: null,
      isSlotFilter: false,
      gridFilterKey: '{}',
      slotInventory: {
        get slotInventoryCards() {
          return slotCards;
        },
        loadSlotInventoryCards: async () => {},
      } as never,
      gridItemFilter: () => ({}),
      initialUrlParams: new URLSearchParams(opts.params ?? ''),
      writeUrl: opts.writeUrl ?? (() => {}),
    });
  });
  flushSync();
  return list;
}

async function settle() {
  for (let i = 0; i < 5; i++) {
    await Promise.resolve();
    flushSync();
  }
}

const lastQuery = () => vi.mocked(getClusters).mock.calls.at(-1)![0]!;

beforeEach(() => {
  sessionStorage.clear();
  slotCards = [];
  vi.mocked(getClusters).mockResolvedValue(response([]));
});
afterEach(() => {
  cleanup?.();
  cleanup = undefined;
  vi.mocked(getClusters).mockReset();
});

describe('cluster list requests', () => {
  it('loads the first screen with one request that also carries its thumbnails', async () => {
    const all = Array.from({ length: 512 }, (_, i) => card(i + 1));
    vi.mocked(getClusters).mockResolvedValue(
      response(
        all.map((c, i) => (i < 24 ? withReps(c) : c)),
        { total: 512 },
      ),
    );
    const list = wire();
    await settle();
    await settle();
    expect(getClusters).toHaveBeenCalledTimes(1);
    expect(lastQuery()).toMatchObject({
      representatives_offset: 0,
      representatives_limit: 24,
    });
    expect(list.gridItems).toHaveLength(24);
    expect(list.gridItems.every((c) => c.representative_crop_ids.length === 1)).toBe(
      true,
    );
  });

  it('sends no sort until one is chosen, then sends each served sort unchanged', async () => {
    const list = wire();
    await settle();
    expect(lastQuery().sort).toBeUndefined();
    for (const id of CLUSTER_SORT_IDS) {
      vi.mocked(getClusters).mockClear();
      list.sort = id;
      flushSync();
      await settle();
      expect(getClusters, id).toHaveBeenCalledTimes(1);
      expect(lastQuery().sort).toBe(id);
    }
  });

  it('sends every card filter with the sort in one reload', async () => {
    const list = wire();
    await settle();
    vi.mocked(getClusters).mockClear();
    list.setFilter({
      dominant_class: ['class_a'],
      detector_class: ['widget_x', 'widget_y'],
      min_size: 3,
      max_size: 50,
      validation_state: 'none',
      flag: ['unvalidated'],
      needs_review: true,
    });
    flushSync();
    await settle();
    expect(getClusters).toHaveBeenCalledTimes(1);
    expect(lastQuery()).toMatchObject({
      dominant_class: ['class_a'],
      detector_class: ['widget_x', 'widget_y'],
      min_size: 3,
      max_size: 50,
      validation_state: 'none',
      flag: ['unvalidated'],
      needs_review: true,
    });
  });

  it('keeps the filter on the request that fetches later thumbnails', async () => {
    const all = Array.from({ length: 60 }, (_, i) => card(i + 1));
    vi.mocked(getClusters).mockResolvedValue(response(all));
    const list = wire();
    list.setFilter({ ...EMPTY_CLUSTER_FILTER, flag: ['unvalidated'] });
    await settle();
    await settle();
    vi.mocked(getClusters).mockClear();
    list.loadMore();
    await settle();
    expect(getClusters).toHaveBeenCalledTimes(1);
    expect(lastQuery()).toMatchObject({
      flag: ['unvalidated'],
      representatives_offset: 24,
      representatives_limit: 24,
    });
  });
});

describe('cluster list rendering', () => {
  it('keeps the served order verbatim (a shuffled fixture stays shuffled)', async () => {
    const served = [5, 1, 9, 3, 7, 2].map((id) => card(id, { size: id % 4 }));
    vi.mocked(getClusters).mockResolvedValue(response(served));
    const list = wire();
    await settle();
    await settle();
    expect(list.gridItems.map((c) => c.id)).toEqual([5, 1, 9, 3, 7, 2]);
  });

  it('shows the next window and merges its thumbnails by cluster id', async () => {
    const all = Array.from({ length: 60 }, (_, i) => card(i + 1));
    vi.mocked(getClusters).mockResolvedValueOnce(
      response(all.map((c, i) => (i < 24 ? withReps(c) : c))),
    );
    const list = wire();
    await settle();
    await settle();
    expect(list.hasMoreCards).toBe(true);
    vi.mocked(getClusters).mockResolvedValueOnce(
      response(
        all.map((c, i) => (i >= 24 && i < 48 ? withReps(c) : c)),
        {
          representatives_offset: 24,
        },
      ),
    );
    list.loadMore();
    await settle();
    expect(list.gridItems).toHaveLength(48);
    expect(list.gridItems[30]!.representative_crop_ids).toEqual(['crop-31']);
    expect(list.gridItems[3]!.representative_crop_ids).toEqual(['crop-4']);
  });

  it('exposes the served sort, options, flag vocabulary, truncation and total', async () => {
    vi.mocked(getClusters).mockResolvedValue(
      response([card(1)], { truncated: true, total: 2000, sort: 'purity_asc' }),
    );
    const list = wire();
    await settle();
    expect(list.appliedSort).toBe('purity_asc');
    expect(list.sortOptions).toEqual([{ id: 'size_desc', label: 'Largest first' }]);
    expect(list.flagVocabulary.map((f) => f.id)).toEqual(['unvalidated']);
    expect(list.truncated).toBe(true);
    expect(list.total).toBe(2000);
  });

  it('keeps detector class options that a filter later hides', async () => {
    vi.mocked(getClusters).mockResolvedValueOnce(
      response([
        card(1, { detector_top_class: 'widget_x' }),
        card(2, { detector_top_class: 'widget_y' }),
      ]),
    );
    const list = wire();
    await settle();
    vi.mocked(getClusters).mockResolvedValueOnce(
      response([card(1, { detector_top_class: 'widget_x' })]),
    );
    list.setFilter({ ...EMPTY_CLUSTER_FILTER, detector_class: ['widget_x'] });
    flushSync();
    await settle();
    expect(list.detectorClassOptions).toEqual(['widget_x', 'widget_y']);
  });

  it('pins the slot inventory cards first only when no card filter is set', async () => {
    slotCards = [{ ...card(900), isSlotCard: true }];
    vi.mocked(getClusters).mockResolvedValue(response([card(1)]));
    const list = wire();
    await settle();
    expect(list.gridItems.map((c) => c.id)).toEqual([900, 1]);
    list.setFilter({ ...EMPTY_CLUSTER_FILTER, needs_review: true });
    flushSync();
    await settle();
    expect(list.gridItems.map((c) => c.id)).toEqual([1]);
  });
});

describe('persistence', () => {
  it('writes the choice to sessionStorage and to the URL', async () => {
    const writeUrl = vi.fn();
    const list = wire({ writeUrl });
    await settle();
    list.sort = 'size_asc';
    list.setFilter({ ...EMPTY_CLUSTER_FILTER, min_size: 4 });
    flushSync();
    await settle();
    const stored = JSON.parse(sessionStorage.getItem(filterPersistKey('default'))!);
    expect(stored.sort).toBe('size_asc');
    expect(stored.filter.min_size).toBe(4);
    expect(writeUrl).toHaveBeenLastCalledWith(
      'size_asc',
      expect.objectContaining({ min_size: 4 }),
    );
  });

  it('starts from the URL when it names a filter, over sessionStorage', async () => {
    sessionStorage.setItem(
      filterPersistKey('default'),
      JSON.stringify({ sort: 'size_asc', filter: { min_size: 9 } }),
    );
    wire({ params: 'sort=purity_desc&flag=unvalidated' });
    await settle();
    expect(lastQuery()).toMatchObject({ sort: 'purity_desc', flag: ['unvalidated'] });
    expect(lastQuery().min_size).toBeUndefined();
  });

  it('starts from sessionStorage when the URL names nothing', async () => {
    sessionStorage.setItem(
      filterPersistKey('default'),
      JSON.stringify({ sort: 'size_asc', filter: { min_size: 9 } }),
    );
    wire();
    await settle();
    expect(lastQuery()).toMatchObject({ sort: 'size_asc', min_size: 9 });
  });
});
