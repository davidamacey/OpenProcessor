/**
 * The /clusters mode swaps over the card grid: the dataset-wide semantic
 * search (with its own label, ignore and undo keys), the Ignored bucket and
 * the literal item-text search, plus the cluster metadata the result
 * badges read.
 */
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { page } from '$app/state';
import { SvelteMap } from 'svelte/reactivity';
import { getClusters } from '$lib/api_clusters';
import {
  getCrops,
  searchImageCrops,
  searchSimilarCrops,
  type ImageSearchQuery,
} from '$lib/api_cropRead';
import { bulkLabel, excludeCrops, unexcludeCrops } from '$lib/api_cropWrites';
import { ApiError, apiErrorText } from '$lib/api_errors';
import type { Pager } from '$lib/pager.svelte';
import { projectHref } from '$lib/projectPaths';
import { createSelection } from '$lib/selection.svelte';
import { isSemanticSearchAvailable } from '$lib/strategies';
import type { Cluster, Crop } from '$lib/types';
import type { SearchCrop } from '$lib/types_review';
import type { RegistryClass } from '$lib/types_classes';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import { dropOnClassStore } from '$stores/dropOnClass.svelte';
import { keyboardStore } from '$stores/keyboard.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';

/** The page state the modes read and set. */
export interface ClusterSearchModesState {
  readonly clusterPager: Pager<Cluster>;
  showEmbeddingViz: boolean;
  /** The grid's item filter, which "More like this" is scoped by as text search is. */
  gridItemFilter(): ItemFilterQuery;
}

export function createClusterSearchModes(s: ClusterSearchModesState) {
  // ---------------- global dataset-wide search ----------------
  // Semantic text search across the WHOLE dataset (not scoped to a
  // cluster/tab, unlike the existing /clusters/[id] and /review
  // integrations) — a mode-switch over the card grid, not a new route,
  // per ClassSidebar's routing gate (src/routes/+layout.svelte:
  // `showSidebar` only mounts on `/clusters` or `/clusters/[id]`, so a
  // dedicated `/search` route would silently lose drag-to-label).
  const semanticSearchAvailable = $derived(
    isSemanticSearchAvailable(strategiesStore.methods.overlays),
  );
  let searchModeActive = $state(false);
  let searchQuery = $state<string>('');
  let searchResults = $state<Crop[]>([]);
  let searchTotal = $state(0);
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale on every search, never mutated in place
  let searchScores = $state(new Map<string, number>());
  const searchSel = createSelection({ plainClick: 'replace' });
  // "More like this": the source crop of the results on show, and the served refusal.
  let similarTo = $state<string | null>(null);
  let similarError = $state<string | null>(null);
  let searchUnembedded = $state<number | null>(null);
  // What the results rank: the crop vectors, or the per-box vectors (each result then names the
  // region that matched). Shared by text, "More like this" and picture search.
  let searchTarget = $state<'item' | 'box'>('item');
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale on every search, never mutated in place
  let searchMatches = $state(new Map<string, { boxId: string; stale: boolean }>());
  // Query by picture: the picture on show (kept so a target switch can re-run it).
  let imageQuery = $state<ImageSearchQuery | null>(null);

  /** The grid filter plus the target, as every search route takes them. */
  function searchFilter(): Record<string, unknown> {
    return {
      ...s.gridItemFilter(),
      ...(searchTarget === 'box' ? { target: 'box' } : {}),
    };
  }

  /** Record which region matched each result (`target=box` pages only). */
  function noteMatches(items: SearchCrop[]): void {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale, never mutated in place
    searchMatches = new Map(
      items
        .filter((it) => it.matched_box_id)
        .map((it) => [
          it.id,
          { boxId: it.matched_box_id as string, stale: it.matched_box_stale === true },
        ]),
    );
  }

  function matchOf(id: string): { boxId: string; stale: boolean } | null {
    return searchMatches.get(id) ?? null;
  }

  async function showResults(
    res: Awaited<ReturnType<typeof searchSimilarCrops>>,
  ): Promise<void> {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale, never mutated in place
    searchScores = new Map(res.items.map((it) => [it.id, it.similarity_score]));
    noteMatches(res.items);
    searchResults = res.items;
    searchTotal = res.total;
    searchUnembedded = res.unembedded_in_scope ?? null;
    void ensureClusterMeta([
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local dedupe consumed synchronously
      ...new Set(
        res.items.map((it) => it.cluster_id).filter((id): id is number => id != null),
      ),
    ]);
  }

  function refusal(e: unknown, what: string): void {
    if (e instanceof ApiError && [400, 404, 409, 413, 503].includes(e.status)) {
      similarError = apiErrorText(e);
    } else {
      toastStore.error(`${what} failed: ${apiErrorText(e)}`);
    }
  }

  async function runSimilar(cropId: string): Promise<void> {
    searchModeActive = true;
    similarTo = cropId;
    imageQuery = null;
    similarError = null;
    searchQuery = '';
    searchResults = [];
    try {
      await showResults(await searchSimilarCrops(cropId, 1, 200, searchFilter()));
    } catch (e) {
      refusal(e, 'Similar search');
    }
  }

  async function runImage(query: ImageSearchQuery): Promise<void> {
    searchModeActive = true;
    imageQuery = query;
    similarTo = null;
    similarError = null;
    searchQuery = '';
    searchResults = [];
    try {
      await showResults(await searchImageCrops(query, 1, 200, searchFilter()));
    } catch (e) {
      refusal(e, 'Picture search');
    }
  }

  /** Switch Crops / Regions; a picture or "More like this" search re-runs from page one (a text
   *  search is re-run by its box, which the toolbar re-mounts on the change). */
  async function setSearchTarget(target: 'item' | 'box'): Promise<void> {
    if (target === searchTarget) return;
    searchTarget = target;
    if (imageQuery) await runImage(imageQuery);
    else if (similarTo) await runSimilar(similarTo);
  }

  // The `?similar_to=` link (a card's or the review pane's "More like this") runs the search.
  // Tracks the URL's own value (not `similarTo`) so leaving the mode, while the URL is still
  // being rewritten, cannot start the same search again.
  let handledSimilar: string | null = null;
  $effect(() => {
    const id = page.url.searchParams.get('similar_to');
    if (!id) {
      handledSimilar = null;
      return;
    }
    if (semanticSearchAvailable && id !== handledSimilar) {
      handledSimilar = id;
      void runSimilar(id);
    }
  });

  // ---------------- Ignored bucket (G7) ----------------
  // Excluded crops (`class_excluded=true`, cluster_id=-2) — a mode swap
  // over the card grid, same pattern as searchModeActive above. The
  // default crop browse filters class_excluded out, so `include_excluded`
  // is required to see them at all.
  let ignoredModeActive = $state(false);
  let ignoredItems = $state<Crop[]>([]);
  let ignoredTotal = $state(0);
  let ignoredLoading = $state(false);
  const ignoredSel = createSelection({ plainClick: 'replace' });

  async function loadIgnored(): Promise<void> {
    ignoredModeActive = true;
    ignoredLoading = true;
    try {
      const res = await getCrops({
        cluster_id: -2,
        include_excluded: true,
        limit: 200,
      });
      ignoredItems = res.items;
      ignoredTotal = res.total;
    } catch (e) {
      toastStore.error(`Could not load ignored crops: ${apiErrorText(e)}`);
    } finally {
      ignoredLoading = false;
    }
  }

  function exitIgnoredMode(): void {
    ignoredModeActive = false;
    ignoredItems = [];
    ignoredSel.clear();
  }

  async function restoreIgnored(): Promise<void> {
    const ids = [...ignoredSel.ids];
    if (ids.length === 0) {
      toastStore.info('Select crops first to restore.');
      return;
    }
    try {
      const res = await unexcludeCrops(ids);
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local lookup consumed synchronously
      const idSet = new Set(ids);
      ignoredItems = ignoredItems.filter((c) => !idSet.has(c.id));
      ignoredTotal = Math.max(0, ignoredTotal - ids.length);
      ignoredSel.clear();
      toastStore.success(`Restored ${res.unexcluded}.`);
    } catch (e) {
      toastStore.error(`Restore failed: ${apiErrorText(e)}`);
    }
  }

  // ---------------- Item-text search (G8) ----------------
  // A literal OCR-text search over item_text_lines
  // (GET {API_PREFIX}/crops?item_text=), independent of the semantic
  // (embedding) search box above.
  let itemTextQuery = $state('');
  let itemTextModeActive = $state(false);
  let itemTextItems = $state<Crop[]>([]);
  let itemTextTotal = $state(0);
  let itemTextLoading = $state(false);
  let itemTextError = $state<string | null>(null);
  const itemTextSel = createSelection({ plainClick: 'replace' });

  async function runItemTextSearch(): Promise<void> {
    const q = itemTextQuery.trim();
    if (!q) return;
    itemTextModeActive = true;
    itemTextLoading = true;
    itemTextError = null;
    try {
      const res = await getCrops({ item_text: q, limit: 200 });
      itemTextItems = res.items;
      itemTextTotal = res.total;
    } catch (e) {
      // 400 = "item_text must contain a letter or digit" — an inline
      // hint, not a toast (the operator is mid-keystroke, not facing an
      // outage).
      if (e instanceof ApiError && e.status === 400) {
        itemTextError = e.detail ?? 'That search needs a letter or digit.';
        itemTextItems = [];
        itemTextTotal = 0;
      } else {
        toastStore.error(`Item-text search failed: ${apiErrorText(e)}`);
      }
    } finally {
      itemTextLoading = false;
    }
  }

  function exitItemTextMode(): void {
    itemTextModeActive = false;
    itemTextItems = [];
    itemTextError = null;
    itemTextSel.clear();
  }

  // Cluster-origin badges: build a Map from whatever's already loaded for
  // the card grid (getClusters({}) returns every cluster, up to 2000, in
  // one call with dominant_class_name precomputed server-side — never
  // recompute dominant class client-side, per CLAUDE.md). If a search
  // result's cluster_id isn't in that map (e.g. the grid was itself
  // filtered by class), fall back to one additional unfiltered call
  // rather than showing a blank badge.
  const clusterMetaMap = new SvelteMap<number, Cluster>();
  $effect(() => {
    clusterMetaMap.clear();
    for (const c of s.clusterPager.items) clusterMetaMap.set(c.id, c);
  });
  async function ensureClusterMeta(ids: number[]): Promise<void> {
    const missing = ids.filter((id) => !clusterMetaMap.has(id));
    if (missing.length === 0) return;
    try {
      // Badge lookup only reads dominant_class_name/purity/etc, so skip the
      // representatives (`limit` has a minimum of 1; `per_cluster` may be 0).
      const res = await getClusters({ per_cluster: 0 });
      for (const c of res.items) clusterMetaMap.set(c.id, c);
    } catch (e) {
      toastStore.warn(`Could not load cluster info for badges: ${apiErrorText(e)}`);
    }
  }

  function syncSearchUrl(q: string | null): void {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- a throwaway copy used to build the next URL
    const url = new URL(page.url.href);
    if (q) url.searchParams.set('q', q);
    else url.searchParams.delete('q');
    url.searchParams.delete('similar_to');
    void goto(resolve(projectHref(`/clusters${url.search}`)), {
      replace: true,
      reset: false,
    });
  }

  function exitSearchMode(): void {
    searchModeActive = false;
    searchResults = [];
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale, never mutated in place
    searchScores = new Map();
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale, never mutated in place
    searchMatches = new Map();
    imageQuery = null;
    searchSel.clear();
    similarTo = null;
    similarError = null;
    searchUnembedded = null;
    syncSearchUrl(null);
  }

  function applyLocalSearchLabel(
    id: string,
    classId: number,
    className: string | null,
  ): void {
    searchResults = searchResults.map((c) =>
      c.id === id
        ? {
            ...c,
            class_id: classId,
            class_name: className,
            label_validated: true,
            class_validated: true,
            label_source: 'human_confirmed',
          }
        : c,
    );
  }

  /** Roll back an optimistic label the server rejected, or render the
   *  crop an undo restored. */
  function replaceSearchCrop(crop: Crop): void {
    searchResults = searchResults.map((c) => (c.id === crop.id ? crop : c));
  }

  // Label actions in search mode: only A (select all) / Z (undo) / X
  // (ignore) / Esc (clear selection). M (move-to-cluster) and AHC
  // sub-cluster grouping are deliberately not offered — search results
  // aren't a single cluster, those actions don't make sense here.
  $effect(() => {
    if (!searchModeActive) return;
    // Force the embedding plot off — same guard pattern the slot-gallery
    // view already uses (the page's isSlotFilter effect): the plot
    // colors by cluster_id over the card grid's own crops, which search
    // mode replaces entirely.
    if (s.showEmbeddingViz) s.showEmbeddingViz = false;

    const offDrop = dropOnClassStore.register(
      async (cls: RegistryClass, droppedIds: string[]) => {
        const ids = droppedIds.length > 0 ? droppedIds : [...searchSel.ids];
        if (ids.length === 0) {
          toastStore.warn('Select or drag crops first, then press a class hotkey.');
          return;
        }
        const priors = searchResults.filter((c) => ids.includes(c.id));
        for (const id of ids) applyLocalSearchLabel(id, cls.id, cls.name);
        searchSel.clear();
        try {
          const res = await bulkLabel(ids, cls.id);
          undoStore.recordWrites(res.updated_ids);
          const conflicts = res.conflicts?.length ?? 0;
          if (conflicts > 0) {
            toastStore.warn(
              `Labeled ${res.updated} of ${ids.length} → ${cls.name} (${conflicts} blocked by worker).`,
            );
          } else {
            toastStore.success(`Labeled ${res.updated ?? ids.length} → ${cls.name}.`);
          }
        } catch (e) {
          toastStore.error(`Label failed: ${apiErrorText(e)}`);
          for (const prior of priors) replaceSearchCrop(prior);
        }
      },
    );

    const offKeys: Array<() => void> = [];
    const reg = (actionId: string, fn: () => void | Promise<void>) =>
      offKeys.push(keyboardStore.registerAction(actionId, () => void fn(), 'clusters'));

    reg('clusters_search.select_all', () =>
      searchSel.selectAll(searchResults.map((c) => c.id)),
    );
    reg('clusters_search.cancel', () => searchSel.clear());
    reg('clusters_search.ignore', async () => {
      const ids = [...searchSel.ids];
      if (ids.length === 0) {
        toastStore.info('Select crops first to ignore.');
        return;
      }
      try {
        const res = await excludeCrops(ids, 'ignore');
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local lookup consumed synchronously
        const idSet = new Set(ids);
        searchResults = searchResults.filter((c) => !idSet.has(c.id));
        searchTotal = Math.max(0, searchTotal - ids.length);
        searchSel.clear();
        toastStore.success(`Ignored ${res.excluded}.`);
      } catch (e) {
        toastStore.error(`Ignore failed: ${apiErrorText(e)}`);
      }
    });
    reg('clusters_search.undo', async () => {
      const crops = await undoStore.undoLast();
      for (const crop of crops) replaceSearchCrop(crop);
    });

    // Unregister the drop handler + keys when search mode ends. Forgetting
    // this leaves /clusters eating drops after the user backs out of
    // search (flagged explicitly in the implementation plan).
    return () => {
      offDrop();
      offKeys.forEach((off) => off());
    };
  });

  return {
    get semanticSearchAvailable() {
      return semanticSearchAvailable;
    },
    get searchModeActive() {
      return searchModeActive;
    },
    set searchModeActive(v: boolean) {
      searchModeActive = v;
    },
    get searchQuery() {
      return searchQuery;
    },
    set searchQuery(v: string) {
      searchQuery = v;
    },
    get searchResults() {
      return searchResults;
    },
    set searchResults(v: Crop[]) {
      searchResults = v;
    },
    get searchTotal() {
      return searchTotal;
    },
    set searchTotal(v: number) {
      searchTotal = v;
    },
    get searchScores() {
      return searchScores;
    },
    set searchScores(v: Map<string, number>) {
      searchScores = v;
    },
    get similarTo() {
      return similarTo;
    },
    get searchTarget() {
      return searchTarget;
    },
    set searchTarget(v: 'item' | 'box') {
      searchTarget = v;
    },
    get imageQueryName() {
      return imageQuery?.file?.name ?? imageQuery?.imageId ?? null;
    },
    get similarError() {
      return similarError;
    },
    get searchUnembedded() {
      return searchUnembedded;
    },
    get searchSel() {
      return searchSel;
    },
    get ignoredModeActive() {
      return ignoredModeActive;
    },
    get ignoredItems() {
      return ignoredItems;
    },
    get ignoredTotal() {
      return ignoredTotal;
    },
    get ignoredLoading() {
      return ignoredLoading;
    },
    get ignoredSel() {
      return ignoredSel;
    },
    get itemTextQuery() {
      return itemTextQuery;
    },
    set itemTextQuery(v: string) {
      itemTextQuery = v;
    },
    get itemTextModeActive() {
      return itemTextModeActive;
    },
    get itemTextItems() {
      return itemTextItems;
    },
    get itemTextTotal() {
      return itemTextTotal;
    },
    get itemTextLoading() {
      return itemTextLoading;
    },
    get itemTextError() {
      return itemTextError;
    },
    get itemTextSel() {
      return itemTextSel;
    },
    get clusterMetaMap() {
      return clusterMetaMap;
    },
    loadIgnored,
    runSimilar,
    runImage,
    setSearchTarget,
    noteMatches,
    matchOf,
    exitIgnoredMode,
    restoreIgnored,
    runItemTextSearch,
    exitItemTextMode,
    ensureClusterMeta,
    syncSearchUrl,
    exitSearchMode,
  };
}

export type ClusterSearchModes = ReturnType<typeof createClusterSearchModes>;
