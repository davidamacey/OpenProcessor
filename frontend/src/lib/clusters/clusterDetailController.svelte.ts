/**
 * The Workspace cluster scope's state (`uid` reads the scope's cluster uid): the
 * cluster card, the crop pager and its race guard, the source / subject /
 * clarity / order filters that feed one crop query, and selection.
 */
import { getCluster, getClusterFacets } from '$lib/api_clusters';
import { createExclusionGuard } from '$lib/clusters/clusterController.svelte';
import { SLOW_LOAD_MS } from '$lib/clusters/firstLoad';
import { createGridGroups } from '$lib/gridGroups.svelte';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import { isDiverseOverlayAvailable, isSemanticSearchAvailable } from '$lib/strategies';
import { createStrategyBar } from '$lib/strategyBar.svelte';
import type { Cluster, Crop } from '$lib/types';
import type { ReprocessRequest } from '$lib/types_import';
import type { ClusterFacets, ConfidenceBin } from '$lib/types_clusterList';
import { apiErrorText } from '$lib/api_errors';
import { classesStore } from '$stores/classes.svelte';
import { classSourcesStore } from '$stores/classSources.svelte';
import { keyboardStore } from '$stores/keyboard.svelte';
import { strategiesStore } from '$stores/strategies.svelte';

export function createClusterDetailController(uid: () => string) {
  const clusterUid = $derived(uid());

  let cluster = $state<Cluster | null>(null);
  // Set when the cluster card read failed but the crops loaded: the header
  // would otherwise show a fabricated unlabeled identity with no hint why.
  let cardError = $state<string | null>(null);

  // The registry class whose name is the card's served dominant class; drives the
  // validated / labeled / cluster-total banner in the header. Class identity is the name.
  const clsForCluster = $derived(
    cluster?.dominant_class_name != null
      ? (classesStore.classes.find((c) => c.name === cluster!.dominant_class_name) ??
          null)
      : null,
  );

  // Provenance echoed back by a pool-scale overlay ordering (Phase 4 —
  // order=diverse). Captured here (not by cropPager, which only retains
  // items/total) the same way `cluster` is captured out-of-band from the
  // fetchPage closure below. null whenever not in diverse mode or the
  // backend didn't send it.
  let orderMeta = $state<{
    method: string | null;
    version: string | null;
    n_pool: number | null;
  } | null>(null);

  // v0.4.0: an ordered view (outliers, diverse, core-first) cannot rank
  // items without a vector; the server serves how many and, when there are
  // any, the request that would embed them.
  let unembedded = $state<{
    count: number | null;
    request: ReprocessRequest | null;
  } | null>(null);

  // Human-readable cluster name for the header: the served
  // `dominant_class_name` (class-id-display-audit-2026-09-26), falling back
  // to a registry lookup only when the cluster card itself failed to load
  // (getCluster's null-identity stub).
  const clusterName = $derived(
    cluster?.dominant_class_name ?? clsForCluster?.name ?? null,
  );

  const pageSize = 60;

  // Crop ids this page has just optimistically moved/labeled/discarded/
  // ignored out of the cluster. A GET for this cluster's crops can be
  // in flight (or triggered fresh, e.g. by the SSE live-refresh effect
  // below) at the exact moment a human drag-drop or hotkey move fires;
  // if that GET's snapshot predates our write, its response would
  // otherwise silently overwrite the optimistic removal and the crop
  // would flicker back in and *stay* until a hard reload — the crop
  // really did move, the grid just re-showed stale data. `exclusionGuard`
  // re-applies this exclusion to every fetchPage result (loadFirst
  // included) so a stale response can never resurrect a crop we already
  // know left. Created here (rather than inside the action controller
  // below) so it can be wired into `cropPager`'s `accept` option without
  // a forward-declared `let controller` — see `createExclusionGuard`'s
  // doc comment in clusterController.svelte.ts. Cleared per-id when the
  // corresponding action is undone or reverted on failure, so a crop
  // that never actually left is never hidden.
  const exclusionGuard = createExclusionGuard();

  // Crop pager. cropQuery() feeds page 1 and every later page, so a filter
  // can't be applied to the first request and silently dropped on the next.
  const cropPager = createPager<Crop>({
    fetchPage: async (cursor) => {
      const query = cropQuery();
      const res = await getCluster(clusterUid, cursor, pageSize, undefined, query);
      cluster = res.cluster;
      cardError = res.cardError;
      orderMeta =
        orderMode === 'diverse'
          ? {
              method: res.crops.order_method ?? null,
              version: res.crops.order_version ?? null,
              n_pool: res.crops.n_pool ?? null,
            }
          : null;
      unembedded = {
        count: res.crops.n_unembedded ?? null,
        request: res.crops.suggested_reprocess ?? null,
      };
      return res.crops;
    },
    keyOf: (c) => c.id,
    accept: (c) => exclusionGuard.accept(c),
    onLoadFirstError: () => {
      cluster = null;
    },
  });

  // Selection set (crop_id) + shift-range anchor.
  const sel = createSelection({ plainClick: 'replace' });

  // class_source filter (null = all). Drives the chip-group in the
  // header and propagates to {API_PREFIX}/crops?class_source=... so the grid
  // shows only crops from one source bucket. Cluster card stats in
  // the header are NOT recomputed by this filter — the operator sees
  // the filter against the whole-cluster totals on purpose.
  let classSourceFilter = $state<string | null>(null);
  // Every class_source this deployment can write — the backend's catalog
  // (GET /class_sources). No catalog, no source filter.
  const classSourceOptions = $derived(classSourcesStore.list);

  // Primary-subject scope: 0 = all crops, 1 = largest only, 2 = largest + 2nd.
  // Maps to the {API_PREFIX}/crops?max_rank= filter (the "biggest subject in frame" the
  // business sorts on). null = no rank filter.
  let subjectScope = $state<0 | 1 | 2>(0);
  const maxRank = $derived<number | null>(subjectScope === 0 ? null : subjectScope);

  // Order strategy (curation-strategy plan Phase 3/4 — generalizes the old
  // outliersFirst boolean into an id string so the fuller StrategyBar
  // selector and the "Outliers first" shortcut button share one source
  // of truth). `strategyBar.sort` IS the order id here — 'default' means
  // newest-first (today's behavior), unchanged from before Phase 3.
  // 'default'/'outliers' are always offered (see the `allowedOrderIds`
  // passed to <StrategyBar> below): `{API_PREFIX}/crops?order=` only special-cases
  // those server-side per the curation strategy plan §1, so
  // any other id would silently do nothing were it offered here — except
  // 'diverse' (Phase 4), which is *conditionally* offered, gated purely on
  // {API_PREFIX}/methods reporting it (mirrors the mistakenness-filter gating
  // pattern in StrategyBar.svelte — see isDiverseOverlayAvailable).
  const strategyBar = createStrategyBar({ defaultId: 'default' });
  const orderMode = $derived(strategyBar.sort);
  // Outliers-first: rank members by distance from the cluster centroid (most
  // atypical first) so mislabels / junk in this cluster float to the top.
  // Computed on-the-fly + cached server-side. Off = newest-first.
  const outliersFirst = $derived(orderMode === 'outliers');

  // 'diverse' (Phase 4, core-set / k-center-greedy pool selection) is only
  // ever offered when {API_PREFIX}/methods reports it at stable/experimental status
  // — a pre-Phase-4 backend, or OP_SELECT_DIVERSE_ENABLED off, means the
  // 'diverse' id is passed through but StrategyBar's own gate (same
  // predicate) never actually surfaces it in the <select>, so this page
  // never ends up requesting an order the backend doesn't support.
  const diverseAvailable = $derived(
    isDiverseOverlayAvailable(strategiesStore.methods.overlays),
  );
  const allowedOrderIds = $derived(
    diverseAvailable ? ['default', 'outliers', 'diverse'] : ['default', 'outliers'],
  );

  // P2-14 semantic text search. Gated behind isSemanticSearchAvailable
  // exactly like isDiverseOverlayAvailable above — an old/flag-off
  // backend hides <SemanticSearchBox> entirely. Search results are fed
  // straight into cropPager's settable items/total (searchScores keyed
  // by crop id, for the similarity badge) so the existing CropCard grid,
  // selection, DnD, and label hotkeys below keep working completely
  // unchanged — this is not a parallel rendering path. While a search is
  // active, "load more" is disabled (see hasMore guard in the pager
  // section below): {API_PREFIX}/search/text pagination isn't wired to this
  // cluster page's infinite-scroll trigger, and re-paging into
  // getCluster would silently overwrite the search results.
  const semanticSearchAvailable = $derived(
    isSemanticSearchAvailable(strategiesStore.methods.overlays),
  );
  let searchModeActive = $state(false);
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale on every search, never mutated in place
  let searchScores = $state(new Map<string, number>());
  // "How many diverse crops?" — defaults to the page's own pageSize
  // (60) until the operator overrides it via the StrategyBar stepper.
  // Only forwarded to the API while actually in diverse mode.
  const diverseK = $derived(orderMode === 'diverse' ? (strategyBar.k ?? pageSize) : null);

  // Clarity slider. `blurSlider` is the live drag value; `minBlurRatio` only
  // commits on release (change, not input) so dragging doesn't spam the API.
  // 0 = show everything (null sent). v1.1.9 sale-quality stops: 1.1/1.3/1.4 —
  // training tolerance sits lower, so the slider spans well below them.
  const BLUR_MAX = 2;
  let blurSlider = $state<number>(0);
  let minBlurRatio = $state<number | null>(null);
  function commitBlur(): void {
    minBlurRatio = blurSlider > 0 ? blurSlider : null;
  }

  // Facet filters: one choice per group, each sent as the matching
  // `GET /crops` parameter. The counts on the chips are the served facets.
  let facets = $state<ClusterFacets | null>(null);
  let facetsError = $state<string | null>(null);
  let facetDetectorClass = $state<string | null>(null);
  let facetBand = $state<{ min: number; max: number | null } | null>(null);
  let facetReviewTab = $state<string | null>(null);
  let facetValidated = $state<boolean | null>(null);
  // 'default' leaves the order to the core-first view; the others rank by
  // the detector's confidence on the server.
  let detectorSort = $state<'default' | 'desc' | 'asc'>('default');
  const facetFilterActive = $derived(
    facetDetectorClass != null ||
      facetBand != null ||
      facetReviewTab != null ||
      facetValidated != null ||
      detectorSort !== 'default',
  );

  function pick<T>(current: T | null, next: T): T | null {
    return current === next ? null : next;
  }
  function toggleDetectorClass(value: string): void {
    facetDetectorClass = pick(facetDetectorClass, value);
  }
  function toggleBand(bin: ConfidenceBin): void {
    const same = facetBand?.min === bin.min && facetBand?.max === bin.max;
    facetBand = same ? null : { min: bin.min, max: bin.max };
  }
  function toggleReviewTab(tab: string): void {
    facetReviewTab = pick(facetReviewTab, tab);
  }
  function toggleValidated(value: boolean): void {
    facetValidated = pick(facetValidated, value);
  }
  function resetFacetFilters(): void {
    facetDetectorClass = null;
    facetBand = null;
    facetReviewTab = null;
    facetValidated = null;
    detectorSort = 'default';
  }

  async function loadFacets(): Promise<void> {
    const uid = clusterUid;
    try {
      const res = await getClusterFacets(uid);
      if (uid !== clusterUid) return;
      facets = res;
      facetsError = null;
    } catch (e) {
      if (uid !== clusterUid) return;
      facets = null;
      facetsError = apiErrorText(e);
    }
  }

  function cropQuery() {
    return {
      detectorClassName: facetDetectorClass ? [facetDetectorClass] : [],
      detectorConfMin: facetBand?.min ?? null,
      detectorConfMax: facetBand?.max ?? null,
      reviewTab: facetReviewTab,
      labelValidated: facetValidated,
      sort: detectorSort === 'default' ? null : `detector_confidence:${detectorSort}`,
      classSource: classSourceFilter,
      maxRank,
      minBlurRatio,
      // Unless the operator picked an explicit order, ask for the server's
      // core-first order: best-scored against the partition first.
      order:
        orderMode === 'default'
          ? detectorSort === 'default'
            ? 'core_first'
            : 'default'
          : orderMode,
      k: diverseK,
    };
  }

  async function loadFirst(): Promise<void> {
    if (!clusterUid) return;
    await Promise.all([cropPager.loadFirst(), loadFacets()]);
  }

  async function loadMore(): Promise<void> {
    if (!clusterUid) return;
    await cropPager.loadMore();
  }

  // The first page of a large cluster can take seconds; say so after SLOW_LOAD_MS.
  let slowFirstLoad = $state(false);
  $effect(() => {
    const firstLoad = cropPager.loading && cropPager.items.length === 0;
    if (!firstLoad) {
      slowFirstLoad = false;
      return;
    }
    const timer = setTimeout(() => (slowFirstLoad = true), SLOW_LOAD_MS);
    return () => clearTimeout(timer);
  });
  const pagerState = $derived({
    loading: cropPager.loading,
    loadingMore: cropPager.loadingMore,
    hasMore: cropPager.hasMore,
    itemCount: cropPager.items.length,
    total: cropPager.total,
  });

  $effect(() => {
    keyboardStore.setScope('cluster');
    // Track every filter input so a change triggers a fresh load.
    void clusterUid;
    void classSourceFilter;
    void maxRank;
    void minBlurRatio;
    void orderMode;
    void diverseK;
    void facetDetectorClass;
    void facetBand;
    void facetReviewTab;
    void facetValidated;
    void detectorSort;
    void loadFirst();
  });

  // The crops on screen: the loaded pages, filtered by the server.
  const filteredCrops = $derived(cropPager.items);

  // The rendered grid. `groups` is a real $derived of filteredCrops, so it can
  // never fall behind cropPager.items; the dnd handlers below install a
  // *transient*, id-reconciled override for the duration of a drag and drop it
  // again on finalize. See src/lib/gridGroups.svelte.ts.
  const grid = createGridGroups<Crop>({
    source: () => filteredCrops,
    keyOf: (c) => c.id,
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- a throwaway id lookup the grid reads synchronously
    liveIds: () => new Set(cropPager.items.map((c) => c.id)),
  });
  const gridGroups = $derived(grid.groups);

  // totalPages was used by the Next/Prev buttons — gone now that infinite scroll
  // owns the pagination. Server-side pageSize stays at 60 per request, but the
  // user just keeps scrolling.

  // ---------------- selection ----------------

  // Range selects span [anchor .. clicked] in the visible order, so the
  // ordered id list is handed to the selection helper per click.
  function clickSelect(id: string, e?: MouseEvent): void {
    sel.click(
      id,
      e,
      filteredCrops.map((c) => c.id),
    );
  }

  function selectAllPage(): void {
    sel.selectAll(filteredCrops.map((c) => c.id));
  }

  function deselectAll(): void {
    sel.clear();
  }

  return {
    get clusterUid() {
      return clusterUid;
    },
    get cluster() {
      return cluster;
    },
    get cardError() {
      return cardError;
    },
    get clsForCluster() {
      return clsForCluster;
    },
    get orderMeta() {
      return orderMeta;
    },
    get unembedded() {
      return unembedded;
    },
    get clusterName() {
      return clusterName;
    },
    get exclusionGuard() {
      return exclusionGuard;
    },
    get cropPager() {
      return cropPager;
    },
    get sel() {
      return sel;
    },
    get facets() {
      return facets;
    },
    get facetsError() {
      return facetsError;
    },
    get facetDetectorClass() {
      return facetDetectorClass;
    },
    get facetBand() {
      return facetBand;
    },
    get facetReviewTab() {
      return facetReviewTab;
    },
    get facetValidated() {
      return facetValidated;
    },
    get detectorSort() {
      return detectorSort;
    },
    set detectorSort(v: 'default' | 'desc' | 'asc') {
      detectorSort = v;
    },
    get facetFilterActive() {
      return facetFilterActive;
    },
    toggleDetectorClass,
    toggleBand,
    toggleReviewTab,
    toggleValidated,
    resetFacetFilters,
    get classSourceFilter() {
      return classSourceFilter;
    },
    set classSourceFilter(v: string | null) {
      classSourceFilter = v;
    },
    get classSourceOptions() {
      return classSourceOptions;
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
    get strategyBar() {
      return strategyBar;
    },
    get orderMode() {
      return orderMode;
    },
    get outliersFirst() {
      return outliersFirst;
    },
    get diverseAvailable() {
      return diverseAvailable;
    },
    get allowedOrderIds() {
      return allowedOrderIds;
    },
    get semanticSearchAvailable() {
      return semanticSearchAvailable;
    },
    get searchModeActive() {
      return searchModeActive;
    },
    set searchModeActive(v: boolean) {
      searchModeActive = v;
    },
    get searchScores() {
      return searchScores;
    },
    set searchScores(v: Map<string, number>) {
      searchScores = v;
    },
    get diverseK() {
      return diverseK;
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
    set minBlurRatio(v: number | null) {
      minBlurRatio = v;
    },
    get slowFirstLoad() {
      return slowFirstLoad;
    },
    get pagerState() {
      return pagerState;
    },
    get filteredCrops() {
      return filteredCrops;
    },
    get grid() {
      return grid;
    },
    get gridGroups() {
      return gridGroups;
    },
    pageSize,
    BLUR_MAX,
    commitBlur,
    loadFirst,
    loadMore,
    clickSelect,
    selectAllPage,
    deselectAll,
  };
}

export type ClusterDetailController = ReturnType<typeof createClusterDetailController>;
