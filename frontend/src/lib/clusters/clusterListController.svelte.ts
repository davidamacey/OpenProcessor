/**
 * The /clusters card grid. The server sorts and filters the cards and serves
 * their quality fields, so this controller sends the choice as query
 * parameters and renders the served order as it comes. It owns the paging of
 * the one served list: how many cards are on screen, and the representatives
 * window that fetches thumbnails for the cards about to appear.
 */
import { untrack } from 'svelte';
import { getClusters } from '$lib/api_clusters';
import {
  EMPTY_CLUSTER_FILTER,
  clusterFilterFromUrl,
  clusterFilterToQuery,
  isClusterFilterActive,
} from '$lib/clusters/clusterListFilter';
import { filterPersistKey, parsePersistedFilter } from '$lib/clusters/persistedFilter';
import type { SlotInventory } from '$lib/clusters/slotInventory.svelte';
import { createPager } from '$lib/pager.svelte';
import type { Cluster, ClusterFilter } from '$lib/types';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type {
  ClusterFlag,
  ClusterListFilter,
  ClusterSortId,
  ClusterSortOption,
} from '$lib/types_clusterList';
import { projectsStore } from '$stores/projects.svelte';
import { toastStore } from '$stores/toast.svelte';

/** The page state the grid reads. */
export interface ClusterListDeps {
  readonly classFilter: number | null;
  readonly isSlotFilter: boolean;
  readonly gridFilterKey: string;
  readonly slotInventory: SlotInventory;
  gridItemFilter(): ItemFilterQuery;
  /** The query string the page was opened with (read once). */
  readonly initialUrlParams: Pick<URLSearchParams, 'get' | 'getAll' | 'has'>;
  /** Mirrors the sort and card filter into the page URL. */
  writeUrl(sort: ClusterSortId | null, filter: ClusterListFilter): void;
}

/** Cards shown per step, and the size of the representatives window. */
const PAGE_SIZE = 24;

export function createClusterListController(deps: ClusterListDeps) {
  const { slotInventory } = deps;

  // The served purity thresholds, so the border legend renders the real
  // pure_min/mixed_min. null until the first page resolves.
  let purityThresholds = $state<{ pure_min: number; mixed_min: number } | null>(null);
  let appliedSort = $state<string | null>(null);
  let sortOptions = $state<ClusterSortOption[]>([]);
  let flagVocabulary = $state<ClusterFlag[]>([]);
  let truncated = $state(false);
  let total = $state(0);
  // Detector classes seen on served cards, in served order. Kept across
  // filters so a picker does not lose the values a filter just hid.
  let detectorClassOptions = $state<string[]>([]);
  let optionsSlug = projectsStore.current?.slug ?? '';

  // How many served cards are on screen, and how far the representatives
  // window has been fetched. A generation counter drops a window response
  // that a reload has outrun.
  let shown = $state(PAGE_SIZE);
  let repsCoveredTo = PAGE_SIZE;
  let repsLoading = $state(false);
  let generation = 0;

  // One params builder feeds the first request and every window request, so
  // a filter cannot be sent on one and dropped on the next.
  const clusterPager = createPager<Cluster>({
    fetchPage: async () => {
      const res = await getClusters(clusterQuery(0));
      if (res.purity_thresholds) purityThresholds = res.purity_thresholds;
      appliedSort = res.sort;
      sortOptions = res.sort_options;
      flagVocabulary = res.flag_vocabulary;
      truncated = res.truncated;
      total = res.total;
      rememberDetectorClasses(res.items);
      return res;
    },
    keyOf: (c) => String(c.id),
  });

  function rememberDetectorClasses(items: Cluster[]): void {
    const slug = projectsStore.current?.slug ?? '';
    if (slug !== optionsSlug) {
      optionsSlug = slug;
      detectorClassOptions = [];
    }
    const next = [...detectorClassOptions];
    for (const c of items) {
      const name = c.detector_top_class;
      if (name && !next.includes(name)) next.push(name);
    }
    if (next.length !== detectorClassOptions.length) detectorClassOptions = next;
  }

  // The choice starts from the URL when it names one, else from this tab's
  // session storage, so a trip into a cluster and back keeps the view.
  const persistKey = filterPersistKey(projectsStore.current?.slug ?? '');
  const fromUrl = clusterFilterFromUrl(deps.initialUrlParams);
  const start =
    fromUrl ??
    parsePersistedFilter(
      typeof sessionStorage === 'undefined' ? null : sessionStorage.getItem(persistKey),
    );
  let sort = $state<ClusterSortId | null>(start.sort);
  let filter = $state<ClusterListFilter>({ ...EMPTY_CLUSTER_FILTER, ...start.filter });
  const filterActive = $derived(isClusterFilterActive(filter));
  const choiceKey = $derived(JSON.stringify([sort, filter]));
  // What the URL currently says; the effect below writes when state differs.
  let urlKey = JSON.stringify([
    fromUrl?.sort ?? null,
    fromUrl?.filter ?? EMPTY_CLUSTER_FILTER,
  ]);

  const pageSize = PAGE_SIZE;

  // Primary-subject grid filters: card stats reflect only crops that pass.
  // subjectScope 0=all, 1=largest, 2=largest+2nd → max_rank. Clarity slider
  // commits on release. Lets the operator scope the grid to the largest,
  // clear crops (incl. the review-tab blind-spot cohorts) for drag-drop +
  // AHC refine.
  let subjectScope = $state<0 | 1 | 2>(0);
  const maxRank = $derived<number | null>(subjectScope === 0 ? null : subjectScope);
  const BLUR_MAX = 2;
  let blurSlider = $state<number>(0);
  let minBlurRatio = $state<number | null>(null);
  function commitClusterBlur(): void {
    minBlurRatio = blurSlider > 0 ? blurSlider : null;
  }

  $effect(() => {
    const key = choiceKey;
    if (typeof sessionStorage !== 'undefined') {
      sessionStorage.setItem(persistKey, JSON.stringify({ sort, filter }));
    }
    if (key !== urlKey) {
      urlKey = key;
      untrack(() => deps.writeUrl(sort, filter));
    }
  });

  function clusterQuery(representativesOffset: number): ClusterFilter {
    return {
      ...deps.gridItemFilter(),
      ...clusterFilterToQuery(filter),
      ...(sort ? { sort } : {}),
      max_rank: maxRank,
      min_blur_ratio: minBlurRatio,
      representatives_offset: representativesOffset,
      representatives_limit: PAGE_SIZE,
    };
  }

  // The cards on screen: the served list cut at `shown`, with the slot
  // inventory entry points pinned first while no card filter is set.
  const gridItems = $derived.by<Cluster[]>(() => {
    const cards = clusterPager.items.slice(0, shown);
    if (
      deps.classFilter == null &&
      slotInventory.slotInventoryCards.length > 0 &&
      !filterActive
    ) {
      return [...slotInventory.slotInventoryCards, ...cards];
    }
    return cards;
  });
  const hasMoreCards = $derived(shown < clusterPager.items.length);

  // Thumbnails for the next window come from the same list call, asked with
  // that window's offset: the server fills `representatives` for exactly
  // those cards, in the order it served them.
  async function loadRepresentatives(windowStart: number): Promise<void> {
    if (windowStart < repsCoveredTo) return;
    repsCoveredTo = windowStart + PAGE_SIZE;
    const myGeneration = generation;
    repsLoading = true;
    try {
      const res = await getClusters(clusterQuery(windowStart));
      if (myGeneration !== generation) return;
      const repsById: Record<number, string[]> = {};
      for (const c of res.items) {
        if (c.representative_crop_ids.length > 0)
          repsById[c.id] = c.representative_crop_ids;
      }
      // Merged in place by id so selection, drag and scroll state survive.
      for (const item of clusterPager.items) {
        const reps = repsById[item.id];
        if (reps) item.representative_crop_ids = reps;
      }
    } catch {
      toastStore.warn('Could not load the next cluster thumbnails.');
    } finally {
      repsLoading = false;
    }
  }

  async function loadFirst(): Promise<void> {
    generation += 1;
    shown = PAGE_SIZE;
    repsCoveredTo = PAGE_SIZE;
    await clusterPager.loadFirst();
    if (clusterPager.error == null) await slotInventory.loadSlotInventoryCards();
  }

  function loadMore(): void {
    if (repsLoading || !hasMoreCards) return;
    const windowStart = shown;
    shown = Math.min(shown + PAGE_SIZE, clusterPager.items.length);
    void loadRepresentatives(windowStart);
  }

  function setFilter(next: ClusterListFilter): void {
    filter = { ...next };
  }

  function clearFilter(): void {
    filter = { ...EMPTY_CLUSTER_FILTER };
  }

  // Reload when anything the server filters or orders by changes. loadFirst's
  // synchronous prefix reads query state and writes the pager, so it runs
  // untracked (otherwise the effect would depend on what it writes).
  $effect(() => {
    void deps.classFilter;
    void maxRank;
    void minBlurRatio;
    void deps.gridFilterKey;
    void choiceKey;
    if (!deps.isSlotFilter) untrack(() => void loadFirst());
  });

  return {
    get purityThresholds() {
      return purityThresholds;
    },
    get clusterPager() {
      return clusterPager;
    },
    get sort() {
      return sort;
    },
    set sort(v: ClusterSortId | null) {
      sort = v;
    },
    get appliedSort() {
      return appliedSort;
    },
    get sortOptions() {
      return sortOptions;
    },
    get flagVocabulary() {
      return flagVocabulary;
    },
    get truncated() {
      return truncated;
    },
    get total() {
      return total;
    },
    get detectorClassOptions() {
      return detectorClassOptions;
    },
    get filter() {
      return filter;
    },
    get filterActive() {
      return filterActive;
    },
    setFilter,
    clearFilter,
    get subjectScope() {
      return subjectScope;
    },
    set subjectScope(v: 0 | 1 | 2) {
      subjectScope = v;
    },
    get maxRank() {
      return maxRank;
    },
    get blurSlider() {
      return blurSlider;
    },
    set blurSlider(v: number) {
      blurSlider = v;
    },
    get minBlurRatio() {
      return minBlurRatio;
    },
    get gridItems() {
      return gridItems;
    },
    get hasMoreCards() {
      return hasMoreCards;
    },
    get repsLoading() {
      return repsLoading;
    },
    pageSize,
    BLUR_MAX,
    commitClusterBlur,
    loadFirst,
    loadMore,
  };
}

export type ClusterListController = ReturnType<typeof createClusterListController>;
