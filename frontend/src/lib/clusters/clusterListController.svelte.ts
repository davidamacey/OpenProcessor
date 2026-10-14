/**
 * The /clusters card grid: the cluster pager, the persisted sort and
 * unlabeled-only filter, the subject and clarity gates, the client-side
 * display order, and the representatives fetched in that display order.
 */
import { untrack } from 'svelte';
import { getClusters } from '$lib/api';
import { idsNeedingRepresentatives } from '$lib/clusters/displayOrderRepresentatives';
import {
  filterPersistKey,
  parsePersistedFilter,
  type PersistedClusterFilter,
} from '$lib/clusters/persistedFilter';
import type { SlotInventory } from '$lib/clusters/slotInventory.svelte';
import { createPager } from '$lib/pager.svelte';
import type { Cluster, ClusterFilter } from '$lib/types';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import { projectsStore } from '$stores/projects.svelte';
import { toastStore } from '$stores/toast.svelte';

/** The page state the grid reads. */
export interface ClusterListDeps {
  readonly classFilter: number | null;
  readonly isSlotFilter: boolean;
  readonly gridFilterKey: string;
  readonly slotInventory: SlotInventory;
  gridItemFilter(): ItemFilterQuery;
}

export function createClusterListController(deps: ClusterListDeps) {
  const { slotInventory } = deps;

  // m7 (2026-09-24 interactive pass): the served purity_thresholds this
  // response carries, so the border-color legend can render the real
  // pure_min/mixed_min instead of a hardcoded "≥80% / ≥60%". null until
  // the first page resolves.
  let purityThresholds = $state<{ pure_min: number; mixed_min: number } | null>(null);

  // Cluster grid pager. One params builder (clusterQuery) feeds page 1 and
  // every later page, so a filter can't be sent on the first request and
  // silently dropped on the next.
  const clusterPager = createPager<Cluster>({
    fetchPage: async (page) => {
      const res = await getClusters(clusterQuery(page));
      if (res.purity_thresholds) purityThresholds = res.purity_thresholds;
      return res;
    },
    keyOf: (c) => String(c.id),
  });

  // Persist the cluster-list filter (sort + unlabeled-only) across
  // navigation so going into a cluster and back keeps the operator's
  // last view — they shouldn't have to re-click "Unlabeled only" every
  // time. sessionStorage survives back-nav + refresh within the session
  // regardless of how the user returns (back button, link, etc.).
  const filterPersistKeyForProject = filterPersistKey(projectsStore.current?.slug ?? '');
  function loadPersistedFilter(): PersistedClusterFilter {
    if (typeof sessionStorage === 'undefined') return parsePersistedFilter(null);
    return parsePersistedFilter(sessionStorage.getItem(filterPersistKeyForProject));
  }
  const _persistedFilter = loadPersistedFilter();

  let sort = $state<NonNullable<ClusterFilter['sort']>>(
    _persistedFilter.sort ?? 'purity_asc',
  );
  let unlabeledOnly = $state<boolean>(_persistedFilter.unlabeledOnly);
  const pageSize = 24;

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

  // Write the filter back whenever it changes. Catches every mutation
  // site (toggle button, sort dropdown) without per-handler bookkeeping.
  $effect(() => {
    if (typeof sessionStorage === 'undefined') return;
    sessionStorage.setItem(
      filterPersistKeyForProject,
      JSON.stringify({ sort, unlabeledOnly }),
    );
  });

  // One params builder for both pages of the cluster grid. loadMore used
  // to omit max_rank / min_blur_ratio, so scrolling past page 1 appended
  // unfiltered clusters over a filtered page 1.
  function clusterQuery(page: number): ClusterFilter {
    return {
      ...deps.gridItemFilter(),
      sort,
      page,
      page_size: pageSize,
      max_rank: maxRank,
      min_blur_ratio: minBlurRatio,
      // D-4: the card list itself always comes back in full in one call
      // (the endpoint ignores page/page_size for that). DQ-M4
      // (docs/design/data-quality-pass-2026-09-24.md): representatives are
      // no longer requested here at all — the endpoint's `offset`/`limit`
      // window only ever covers its own size-desc server order, which
      // doesn't line up with the client-side sort (`sortClusters()`)
      // actually shown, so representatives are fetched separately, in
      // DISPLAY order, by loadMoreRepresentatives() below.
      //
      // Live-verified follow-up: `limit: 0` 422s ("Input should be
      // greater than or equal to 1") — the backend's offset/limit window
      // has a real minimum of 1, unlike `per_cluster`, whose minimum is
      // genuinely 0. `per_cluster: 0` achieves the same "skip
      // representative computation" goal without an invalid limit, so
      // offset/limit are left unset (backend default) rather than forced
      // to an invalid or wastefully-real window.
      per_cluster: 0,
    };
  }

  // DQ-M4's loadFirst()/loadMoreRepresentatives() are declared below
  // gridItems (further down this file) since they read it — kept as a
  // forward function reference here would need `gridItems` in the
  // temporal dead zone otherwise (`const`, not hoisted like `function`).

  // Sort the loaded clusters client-side. The {API_PREFIX}/clusters endpoint only
  // returns size-descending (it's a terms agg, not a sortable query), and
  // every cluster comes back in one call — so sorting here is both
  // correct and complete. Without this the sort dropdown did nothing.
  function sortClusters(list: Cluster[], mode: typeof sort): Cluster[] {
    const out = [...list];
    const purity = (c: Cluster) =>
      c.purity == null ? Number.POSITIVE_INFINITY : c.purity;
    switch (mode) {
      case 'size_desc':
        out.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
        break;
      case 'size_asc':
        out.sort((a, b) => (a.size ?? 0) - (b.size ?? 0));
        break;
      case 'purity_desc':
        // null purity (no labelled members) sorts last on a desc view too.
        out.sort((a, b) => {
          const pa = a.purity ?? -1;
          const pb = b.purity ?? -1;
          return pb - pa;
        });
        break;
      case 'purity_asc':
        out.sort((a, b) => purity(a) - purity(b));
        break;
      case 'dominant_class':
        out.sort((a, b) =>
          (a.dominant_class_name ?? '￿').localeCompare(b.dominant_class_name ?? '￿'),
        );
        break;
    }
    return out;
  }

  const gridItems = $derived.by<Cluster[]>(() => {
    const filtered = unlabeledOnly
      ? clusterPager.items.filter((c) => c.cluster_kind !== 'class')
      : clusterPager.items;
    const sorted = sortClusters(filtered, sort);
    // Keep the synthetic slot cards pinned first (entry points to the
    // slot inventories), unaffected by sort, only on the unfiltered
    // labelled view. M4: this used to drop the real class-kind cluster
    // sharing the slot class's id (cluster_id === class_id for class-kind
    // clusters) on the theory that slotInventoryCard replaces it — but that cluster
    // (e.g. #80, size 1) is a real, independently-reachable cluster with
    // its own crops, and hiding it made it permanently unreachable from
    // this grid. The two now render side by side; the #each key below is
    // keyed off `isSlotCard` so the synthetic entry never collides with
    // the real cluster's id.
    if (
      deps.classFilter == null &&
      slotInventory.slotInventoryCards.length > 0 &&
      !unlabeledOnly
    ) {
      return [...slotInventory.slotInventoryCards, ...sorted];
    }
    return sorted;
  });

  // DQ-M4 (docs/design/data-quality-pass-2026-09-24.md): representatives
  // fetched per-card, in the operator's actual DISPLAY order (gridItems,
  // just above), not the backend's fixed size-desc order. The `/clusters`
  // endpoint has no batch-by-id representatives param (checked against
  // contracts/openapi/curation.json) — only a single
  // `cluster_id` filter — so each missing card in the window is fetched
  // individually via that filter, in parallel. See
  // src/lib/clusters/displayOrderRepresentatives.ts for the full
  // rationale and the pure id-selection logic.
  //
  // `dispOffset` tracks how far into `gridItems` a fetch attempt has been
  // made; reset to 0 whenever the display order can change (loadFirst, or
  // the sort/filter effect below) so a resort re-covers its new first
  // window instead of only ever advancing forward.
  let dispOffset = $state(0);
  let repsLoading = $state(false);
  const repsHasMore = $derived(dispOffset < gridItems.length);

  async function loadMoreRepresentatives(): Promise<void> {
    if (repsLoading || !repsHasMore) return;
    repsLoading = true;
    const windowStart = dispOffset;
    try {
      const ids = idsNeedingRepresentatives(gridItems, windowStart, pageSize);
      if (ids.length > 0) {
        const results = await Promise.allSettled(
          ids.map((id) => getClusters({ cluster_id: id, representatives_limit: 1 })),
        );
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local merge lookup consumed synchronously within this call, never stored in reactive state
        const byId = new Map<number, Cluster>();
        for (const r of results) {
          if (r.status === 'fulfilled') {
            const c = r.value.items[0];
            if (c) byId.set(c.id, c);
          }
        }
        // Merge into the already-loaded cards by id. Mutating in place
        // (not replacing clusterPager.items) keeps every other bit of
        // component state (selection, drag, scroll position) untouched.
        for (const item of clusterPager.items) {
          const upd = byId.get(item.id);
          if (upd && upd.representative_crop_ids.length > 0) {
            item.representative_crop_ids = upd.representative_crop_ids;
          }
        }
        const failed = results.filter((r) => r.status === 'rejected').length;
        if (failed > 0) {
          toastStore.warn(
            `Could not load ${failed} cluster thumbnail${failed === 1 ? '' : 's'}.`,
          );
        }
      }
    } finally {
      dispOffset = windowStart + pageSize;
      repsLoading = false;
    }
  }

  async function loadFirst(): Promise<void> {
    dispOffset = 0;
    await clusterPager.loadFirst();
    if (clusterPager.error == null) {
      await slotInventory.loadSlotInventoryCards();
      // DQ-M4: fetch the first screenful's representatives in DISPLAY
      // order right away — this is the fix for cards rendering blank on
      // first paint. loadMoreRepresentatives() below (sentinel-triggered)
      // continues the same windowing as the operator scrolls.
      await loadMoreRepresentatives();
    }
  }

  const unlabeledCount = $derived(
    clusterPager.items.filter((c) => c.cluster_kind !== 'class').length,
  );

  const loadMore = () => {
    void clusterPager.loadMore();
    void loadMoreRepresentatives();
  };

  // Re-load on class-filter change — but only for the cluster view.
  // Sort is applied client-side (sortClusters) over the single loaded
  // batch, so changing it must NOT refetch (the endpoint returns the
  // same size-ordered data regardless).
  //
  // DQ-M4 follow-up: loadFirst()'s synchronous prefix (before its first
  // await) reads clusterQuery()'s `sort`/`class_id`/etc and WRITES
  // clusterPager.items — both wrapped in untrack() below. Without it,
  // this effect's dependency set would pick up every state
  // loadFirst()/loadMoreRepresentatives() touch (the same trap
  // _seedViewBox() in review/+page.svelte documents for the identical
  // reason), and the write-then-reread cycle between this effect and the
  // sort/unlabeledOnly effect below tripped Svelte's
  // effect_update_depth_exceeded guard on every /clusters mount.
  $effect(() => {
    void deps.classFilter;
    void maxRank;
    void minBlurRatio;
    void deps.gridFilterKey;
    if (!deps.isSlotFilter) untrack(() => void loadFirst());
  });

  // DQ-M4: sort/unlabeledOnly reshuffle DISPLAY order (gridItems) without
  // a refetch of the card list itself (see the comment above — the
  // endpoint isn't sortable, sortClusters() is purely client-side). That
  // reshuffle can put a card that was never in an earlier display window
  // first, so the representatives window has to restart from 0 and
  // re-cover the new first screenful — cheap, since
  // idsNeedingRepresentatives() skips every card that already has
  // representatives from a prior window. untrack() for the same reason
  // as the effect above: loadMoreRepresentatives() reads gridItems and
  // writes clusterPager.items — read that synchronously inside THIS
  // effect (instead of untracked) and every representative fetch it
  // triggers would re-fire this same effect, in an unbounded loop.
  $effect(() => {
    void sort;
    void unlabeledOnly;
    dispOffset = 0;
    if (!deps.isSlotFilter) untrack(() => void loadMoreRepresentatives());
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
    set sort(v: NonNullable<ClusterFilter['sort']>) {
      sort = v;
    },
    get unlabeledOnly() {
      return unlabeledOnly;
    },
    set unlabeledOnly(v: boolean) {
      unlabeledOnly = v;
    },
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
    get repsLoading() {
      return repsLoading;
    },
    get repsHasMore() {
      return repsHasMore;
    },
    get unlabeledCount() {
      return unlabeledCount;
    },
    BLUR_MAX,
    commitClusterBlur,
    loadFirst,
    loadMore,
  };
}

export type ClusterListController = ReturnType<typeof createClusterListController>;
