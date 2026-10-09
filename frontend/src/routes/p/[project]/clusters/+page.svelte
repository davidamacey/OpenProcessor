<script lang="ts">
  import { apiErrorText } from '$lib/api';
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { goto } from '$app/navigation';
  import { untrack } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import { page } from '$app/state';
  import {
    ApiError,
    bulkLabel,
    excludeCrops,
    getClusters,
    getCrops,
    getRegions,
    getThumbUrl,
    resolveApiUrl,
    unexcludeCrops,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { slotForClassName, registeredSlots } from '$lib/annotations/registeredSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import { displayBoxOf } from '$lib/annotations/rowBox';
  import { createPager } from '$lib/pager.svelte';
  import { idsNeedingRepresentatives } from '$lib/clusters/displayOrderRepresentatives';
  import {
    createSlotGalleryController,
    type SlotGalleryController,
  } from './slotGalleryController.svelte';
  import { createSelection } from '$lib/selection.svelte';
  import {
    isEmbeddingVizAvailable,
    isEmbeddingVizBannerRequired,
    isSemanticSearchAvailable,
  } from '$lib/strategies';
  import { findActiveClassByName } from '$lib/classNameKey';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import ClusterBadge from '$lib/components/ClusterBadge.svelte';
  import CropDetailModal from '$lib/components/CropDetailModal.svelte';
  import CropResultGrid from '$lib/components/CropResultGrid.svelte';
  import ItemFilterBar from '$lib/components/itemFilter/ItemFilterBar.svelte';
  import MatchingItemsView from '$lib/components/itemFilter/MatchingItemsView.svelte';
  import {
    ItemFilterState,
    withoutOpenVocab,
  } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';
  import EmbeddingPlot from '$lib/components/EmbeddingPlot.svelte';
  import SlotGallery from '$lib/components/slots/SlotGallery.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import type { ClusterFilter, RegistryClass, Cluster, Crop } from '$lib/types';
  import {
    filterPersistKey,
    parsePersistedFilter,
    type PersistedClusterFilter,
  } from '$lib/clusters/persistedFilter';
  import { projectsStore } from '$stores/projects.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import {
    dominantShareText,
    dominantShareTitle,
    cohesionText,
    COHESION_TOOLTIP,
  } from '$lib/clusters/clusterCardText';

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

  // Synthetic slot inventory cards. A slot's regions are sub-bboxes on
  // other items, not cluster docs, so the cluster grid never produces a
  // card for them. We surface one per slot from its browse endpoint so the
  // operator can click into the slot's inventory the same way they click
  // into any other class cluster. Empty until the first browse call
  // resolves; the cluster grid shows none during that window.
  let slotInventoryCards = $state<Cluster[]>([]);

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

  // --- Slot gallery (shown instead of the cluster grid when the class
  //     filter is a slot-bound class). State/logic live in
  //     slotGalleryController.svelte.ts and SlotGallery.svelte renders it;
  //     this route owns the class-filter routing decision and the
  //     URL/filter-driven reload effects below.
  // One controller per slot, kept for the page's lifetime so a slot's
  // gallery filters survive switching to another class and back.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- memoization cache of controller instances, kept for the page's lifetime but never read reactively by a template/derived
  const galleriesBySlot = new Map<string, SlotGalleryController>();
  function galleryFor(slot: SlotSpec): SlotGalleryController {
    let g = galleriesBySlot.get(slot.key);
    if (!g) {
      g = untrack(() => createSlotGalleryController(slot));
      galleriesBySlot.set(slot.key, g);
    }
    return g;
  }
  // Leaving the page (or switching project) stops any region-clustering poll.
  $effect(() => () => {
    for (const g of galleriesBySlot.values()) g.dispose();
  });

  const classFilter = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    if (v == null) return null;
    if (Number.isFinite(+v)) return +v;
    // m22: a name-form deep link (`?class=<name>`) used to be silently
    // ignored — only a numeric class id worked, so a name link to a
    // slot-bound class rendered the unfiltered grid instead of routing to
    // the slot gallery. Resolve the name against
    // the loaded registry, same lookup `open()` already does in the
    // opposite direction (cluster -> class name -> id -> slot route).
    return findActiveClassByName(classesStore.classes, v)?.id ?? null;
  });

  // The shared item filter (class by name, confidence / area band, origin,
  // embedding and review state), seeded from and written back to the URL.
  // The largest-N toggle below keeps `max_rank`, so the bar never draws it.
  const itemFilter = new ItemFilterState();
  itemFilter.fromUrl(page.url.searchParams);
  let matchingModeActive = $state(
    page.url.searchParams.get('mode') === 'matching' ||
      !!itemFilter.openVocabSet ||
      !!itemFilter.sourcePrompt,
  );
  const classFilterName = $derived(
    classFilter == null
      ? null
      : (classesStore.classes.find((c) => c.id === classFilter)?.name ?? null),
  );
  // The sidebar's `?class=` scopes the grid by the served class name, unless
  // the bar already names classes.
  function withSidebarClass<T extends { class_name?: string[]; class_names?: string[] }>(
    f: T,
    key: 'class_name' | 'class_names',
  ): T {
    const named = f[key];
    if (named && named.length > 0) return f;
    return classFilterName ? { ...f, [key]: [classFilterName] } : f;
  }
  function gridItemFilter(): ItemFilterQuery {
    return withSidebarClass(
      itemFilter.toQuery((p) => withoutOpenVocab(p) && p !== 'max_rank'),
      'class_name',
    );
  }
  function matchingQuery(): ItemFilterQuery {
    return withSidebarClass(itemFilter.toQuery(), 'class_name');
  }
  function matchingBody(): ItemFilter {
    return withSidebarClass(itemFilter.toBody(), 'class_names');
  }
  const gridFilterKey = $derived(JSON.stringify(gridItemFilter()));
  function syncFilterUrl(): void {
    const url = new URL(page.url.href);
    itemFilter.toUrl(url.searchParams);
    if (matchingModeActive) url.searchParams.set('mode', 'matching');
    else url.searchParams.delete('mode');
    void goto(resolve(projectHref(`/clusters${url.search}`)), {
      replace: true,
      reset: false,
    });
  }
  function enterMatchingMode(): void {
    if (searchModeActive) exitSearchMode();
    if (ignoredModeActive) exitIgnoredMode();
    if (itemTextModeActive) exitItemTextMode();
    matchingModeActive = true;
    syncFilterUrl();
  }
  function exitMatchingMode(): void {
    matchingModeActive = false;
    // The open-vocabulary pair only exists in this view.
    itemFilter.openVocabSet = null;
    itemFilter.sourcePrompt = null;
    syncFilterUrl();
  }

  // The backend stores a slot's regions as a *sub-bbox* on each item
  // (`region_boxes`), NOT as standalone docs in the cluster index. So
  // filtering this page by a slot-bound class always returns 0 /
  // unlabeled clusters. Detect that case and show the slot's browse
  // gallery instead, which is the actual home for that slot's labeling.
  const gallerySlot = $derived.by<SlotSpec | null>(() => {
    if (classFilter == null) return null;
    const cls = classesStore.classes.find((c) => c.id === classFilter);
    const slot = slotForClassName(cls?.name);
    return slot?.capabilities.queue?.browsePath ? slot : null;
  });
  const isSlotFilter = $derived(gallerySlot != null);
  const slotGallery = $derived(gallerySlot ? galleryFor(gallerySlot) : null);

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
  let searchScores = $state(new Map<string, number>());
  const searchSel = createSelection({ plainClick: 'replace' });
  let detailSearchCrop = $state<Crop | null>(null);

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
    for (const c of clusterPager.items) clusterMetaMap.set(c.id, c);
  });
  async function ensureClusterMeta(ids: number[]): Promise<void> {
    const missing = ids.filter((id) => !clusterMetaMap.has(id));
    if (missing.length === 0) return;
    try {
      // Badge lookup only reads dominant_class_name/purity/etc — no
      // representatives needed, so skip that window entirely (D-4).
      const res = await getClusters({ representatives_limit: 0 });
      for (const c of res.items) clusterMetaMap.set(c.id, c);
    } catch (e) {
      toastStore.warn(`Could not load cluster info for badges: ${apiErrorText(e)}`);
    }
  }

  function syncSearchUrl(q: string | null): void {
    const url = new URL(page.url.href);
    if (q) url.searchParams.set('q', q);
    else url.searchParams.delete('q');
    void goto(resolve(projectHref(`/clusters${url.search}`)), {
      replace: true,
      reset: false,
    });
  }

  function exitSearchMode(): void {
    searchModeActive = false;
    searchResults = [];
    searchScores = new Map();
    searchSel.clear();
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
    // view already uses (isSlotFilter effect above): the plot
    // colors by cluster_id over the card grid's own crops, which search
    // mode replaces entirely.
    if (showEmbeddingViz) showEmbeddingViz = false;

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

  // Embedding-plot overlay (curation-strategy plan Phase 5 —
  // docs/curation-strategy-plan-2026-09.md §2.7/§5.6). Off by default,
  // lazily mounted: <EmbeddingPlot> only appears in the template inside
  // the {#if showEmbeddingViz} block below, so it never instantiates
  // (never calls getVizProjection) until the operator explicitly toggles
  // it on. Mirrors isDiverseOverlayAvailable's gating pattern exactly —
  // the toggle button itself is absent (not just disabled) unless
  // {API_PREFIX}/methods reports the overlay at stable/experimental.
  $effect(() => {
    void strategiesStore.init();
  });
  const embeddingVizAvailable = $derived(
    isEmbeddingVizAvailable(strategiesStore.methods.overlays),
  );
  const embeddingVizBannerRequired = $derived(
    isEmbeddingVizBannerRequired(strategiesStore.methods.overlays),
  );
  // m20 (2026-09-24 interactive pass): served field_coverage_total for
  // the viz_projection overlay, so EmbeddingPlot can note when it's
  // showing fewer points than the pool the fit was computed over.
  const embeddingVizCoverageTotal = $derived(
    strategiesStore.methods.overlays.find((o) => o.id === 'viz_projection')
      ?.field_coverage_total ?? null,
  );
  let showEmbeddingViz = $state<boolean>(false);
  // The slot-gallery view (isSlotFilter) has its own grid + bulk-triage
  // toolbar; the embedding plot projects items with a real cluster_id,
  // which regions (sub-bboxes, not their own cluster docs) never have. Force the toggle off rather than
  // leaving a stale plot mounted over a view it doesn't apply to.
  $effect(() => {
    if (isSlotFilter && showEmbeddingViz) showEmbeddingViz = false;
  });

  // One params builder for both pages of the cluster grid. loadMore used
  // to omit max_rank / min_blur_ratio, so scrolling past page 1 appended
  // unfiltered clusters over a filtered page 1.
  function clusterQuery(page: number): ClusterFilter {
    return {
      ...gridItemFilter(),
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

  // Build one synthetic inventory card per registered slot that has a
  // browse endpoint and whose bound class is in the registry. A slot's
  // regions live as sub-boxes on items of other classes (not cluster
  // docs), so the cluster grid never includes them. Each card is built
  // from its own slot's `queue.browsePath` (total + the first 4 boxed
  // items as thumbnails) and bound to its own class; a slot with no
  // browse endpoint gets no card. The grid renders the cards first on
  // the unfiltered view.
  async function loadSlotInventoryCards(): Promise<void> {
    const cards: Cluster[] = [];
    for (const slot of registeredSlots) {
      const card = await buildSlotInventoryCard(slot);
      if (card) cards.push(card);
    }
    if (cards.length > 0 || slotInventoryCards.length > 0) slotInventoryCards = cards;
  }

  async function buildSlotInventoryCard(slot: SlotSpec): Promise<Cluster | null> {
    const browsePath = slot.capabilities.queue?.browsePath;
    const className = slot.bind.className;
    if (!browsePath || !className) return null;
    const cls = findActiveClassByName(classesStore.classes, className);
    if (!cls) return null;
    try {
      // Pull a slightly larger window than 4 so we can drop items
      // missing a sub-box without falling below the tile count.
      const res = await getRegions(browsePath, {
        page: 1,
        page_size: 12,
      });
      // A tile is one box's close-up, so only rows with a served box
      // thumbnail qualify.
      const reps = res.items
        .map((p) => ({ p, box: displayBoxOf(p.slots?.[slot.key], p.region_box_id) }))
        .filter((r) => r.box?.thumbnailUrl != null)
        .slice(0, 4);
      return {
        id: cls.id,
        // Not a real cluster: cluster_kind/purity_tier/promotable/etc.
        // have no server-served value for a slot-inventory entry, so
        // they're left at honest defaults rather than invented. See
        // `isSlotCard` on the type and its use in purityBadge/borderColor
        // below, which skip the purity badge entirely for this card.
        cluster_kind: 'unassigned',
        validated_count: 0,
        size: res.total,
        purity: null,
        purity_tier: null,
        promotable: false,
        core_similarity_min: null,
        is_unlabeled: false,
        dominant_class_id: cls.id,
        dominant_class_name: cls.name,
        dominant_pct: null,
        n_subclusters: 0,
        has_subclusters: false,
        representative_crop_ids: reps.map((r) => r.p.crop_id),
        // Region close-ups, not parent-item thumbnails: the card is the
        // entry point to the slot's own inventory.
        representative_thumb_urls: reps.map((r) => resolveApiUrl(r.box!.thumbnailUrl!)),
        updated_at: null,
        isSlotCard: true,
        slotDisplayName: slot.capabilities.queue?.tabLabel ?? slot.label.plural,
      };
    } catch {
      return null;
    }
  }

  // M4: `loadSlotInventoryCards` used to run exactly once, right after the
  // first `loadFirst()` — if the root layout's own `classesStore.acquire()`
  // fetch hadn't resolved yet at that moment, `classesStore.classes` was
  // still empty, the slot class lookup failed, and the card never
  // retried (observed live: ~1 render in 8). Re-running whenever the
  // classes list changes (classesStore's own 30s poll, or a slower first
  // load) makes the card deterministic instead of a load-order race.
  $effect(() => {
    if (
      slotInventoryCards.length === 0 &&
      classesStore.classes.length > 0 &&
      clusterPager.error == null &&
      !isSlotFilter
    ) {
      untrack(() => void loadSlotInventoryCards());
    }
  });

  // Items rendered in the unfiltered cluster grid: synthetic slot
  // inventory cards prepended (when present) so the operator always has a
  // visible entry point to each slot's inventory. With a class filter
  // active we hand the user to the slot gallery already, so skip the
  // prepend there.
  // gridItems = (synthetic slot cards if unfiltered) + clusters,
  // optionally narrowed to only the "Unlabeled" group when the toggle is on.
  // "Unlabeled" = cluster_kind !== 'class', i.e. the candidate (IVF/AHC)
  // and unassigned buckets the operator still needs to sort. Keying on
  // cluster_kind (not dominant_class_name) is the fix for "only 16
  // showed": candidate clusters dominated by VLM-unmatched crops DO
  // carry a dominant_class_name, so the old !dominant_class_name test
  // wrongly excluded them.
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
    if (classFilter == null && slotInventoryCards.length > 0 && !unlabeledOnly) {
      return [...slotInventoryCards, ...sorted];
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
      await loadSlotInventoryCards();
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

  $effect(() => {
    keyboardStore.setScope('clusters');
  });

  // M6: Z on the slot gallery reverses the most recent region write
  // (bulk status change or a single bbox edit) — same key, same
  // undoStore, as the card-grid view's label-undo Z above; see
  // slotGalleryController's undoLastAction doc comment.
  $effect(() => {
    const gallery = slotGallery;
    if (!gallery) return;
    const off = keyboardStore.registerAction(
      'region_gallery.undo',
      () => void gallery.undoLastAction(),
      'clusters',
      { labelVars: { region: gallery.slot.label.singular } },
    );
    return off;
  });

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
    void classFilter;
    void maxRank;
    void minBlurRatio;
    void gridFilterKey;
    if (!isSlotFilter) untrack(() => void loadFirst());
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
    if (!isSlotFilter) untrack(() => void loadMoreRepresentatives());
  });

  // Re-load the gallery whenever a filter, the top-N rank gate, or the
  // selected region cluster changes (typed filters debounce, the cluster
  // grid reloads only for the rank gate or a closed bucket; see
  // `reloadOnFilterChange`).
  $effect(() => {
    slotGallery?.reloadOnFilterChange();
  });

  // Purity banding is served (`purity_tier`, `{API_PREFIX}/clusters`'
  // `purity_thresholds`) — no client 0.8/0.6 threshold. `has_subclusters`
  // is a separate, unrelated signal (AHC sub-clustering ran) and still
  // wins the border color outright.
  function borderColor(c: Cluster): string {
    // M4: a distinct, neutral border for the non-cluster inventory card —
    // the red "noisy" border would otherwise falsely imply a bad cluster.
    if (c.isSlotCard) return 'border-purple-500/60';
    if (c.has_subclusters) return 'border-blue-500/60';
    if (c.purity_tier === 'pure') return 'border-green-500/60';
    if (c.purity_tier === 'mixed') return 'border-orange-500/60';
    return 'border-red-500/60';
  }

  function purityBadge(c: Cluster): { color: string; text: string } | null {
    // M4: a synthetic slot inventory card is not a cluster —
    // it has no purity, so it gets no purity badge at all rather than
    // falling through to an invented "noisy 0%".
    if (c.isSlotCard) return null;
    if (c.purity_tier === 'pure')
      return { color: 'bg-green-500/20 text-green-300', text: 'pure' };
    if (c.purity_tier === 'mixed')
      return { color: 'bg-orange-500/20 text-orange-200', text: 'mixed' };
    return { color: 'bg-red-500/20 text-red-200', text: 'noisy' };
  }

  function open(c: Cluster): void {
    // Special-case: clicking a cluster whose dominant class is bound to
    // a registered slot should jump to that slot's browse view (which
    // surfaces every crop with the slot's sub-bbox), not the
    // single-cluster crop grid. Slot sub-bboxes live on other items, so
    // that class's cluster only contains the rare crops that were labeled
    // with it as their PRIMARY class — usually 1-2 mis-labels. The
    // operator's intent is "show me all the slot's items", so route them
    // to that inventory instead.
    //
    // P2.11 fix: resolve the CLICKED cluster's own dominant class, not
    // "the first slot-bound class" — the latter silently routed every
    // slot-bound cluster to the first registered slot's class filter,
    // which breaks the instant a second capable slot exists.
    if (slotForClassName(c.dominant_class_name) != null && c.dominant_class_id != null) {
      void goto(resolve(projectHref(`/clusters?class=${c.dominant_class_id}`)));
      return;
    }
    void goto(resolve(projectHref(`/clusters/${c.id}`)));
  }

  // Infinite scroll owns pagination — totalPages no longer needed.
</script>

<div class="flex h-full flex-col">
  <!-- Toolbar -->
  <div class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
    <h1 class="text-lg font-semibold">Clusters</h1>
    <ShortcutsButton />

    {#if classFilter != null}
      <!-- C2 (visual audit 2026-09-24): name the class, not its id. -->
      <span
        class="rounded-md border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-xs text-blue-200"
        title="class id {classFilter}"
        data-testid="class-filter-chip"
      >
        class: {classesStore.classes.find((c) => c.id === classFilter)?.name ??
          `#${classFilter}`}
      </span>
    {/if}

    <!-- Dataset-wide semantic search — left-aligned, first control after
         the title, so it reads as the primary way in rather than a
         control squeezed between unrelated toggles. Deliberately no
         cluster_id/tab scope in `filter` — unscoped-across-the-whole-dataset
         is the entire point of this control, unlike the cluster_id-scoped
         SemanticSearchBox on /clusters/[id]. -->
    {#if semanticSearchAvailable && !isSlotFilter}
      <SemanticSearchBox
        pageSize={200}
        filter={{ ...gridItemFilter() }}
        initialQuery={page.url.searchParams.get('q')}
        onQueryChange={(q) => (searchQuery = q)}
        onResults={(res) => {
          searchModeActive = true;
          searchScores = new Map(res.items.map((it) => [it.id, it.similarity_score]));
          searchResults = res.items;
          searchTotal = res.total;
          syncSearchUrl(searchQuery);
          void ensureClusterMeta([
            ...new Set(
              res.items
                .map((it) => it.cluster_id)
                .filter((id): id is number => id != null),
            ),
          ]);
        }}
        onClear={exitSearchMode}
      />
    {/if}

    <!-- Item-text search (G8) — literal OCR text search over
         item_text_lines, distinct from the semantic search box above. -->
    <form
      class="flex items-center gap-1"
      onsubmit={(e) => {
        e.preventDefault();
        void runItemTextSearch();
      }}
    >
      <input
        type="text"
        bind:value={itemTextQuery}
        placeholder="Item text…"
        class="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
      />
      <button type="submit" class="btn-sm border border-zinc-700 bg-zinc-900 text-xs">
        Search text
      </button>
    </form>
    {#if itemTextError}
      <span class="text-[11px] text-amber-300">{itemTextError}</span>
    {/if}

    <!-- Ignored bucket (G7) -->
    <button
      type="button"
      onclick={() => {
        if (ignoredModeActive) exitIgnoredMode();
        else void loadIgnored();
      }}
      class="btn-sm border text-xs transition-colors {ignoredModeActive
        ? 'border-amber-500/60 bg-amber-500/20 text-amber-200'
        : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-amber-500/40'}"
      title="Crops excluded from training + clustering"
    >
      {ignoredModeActive ? '✓ ' : ''}Ignored
    </button>

    <!-- Every item the shared filter matches, with run-on-selection actions. -->
    <button
      type="button"
      onclick={() => (matchingModeActive ? exitMatchingMode() : enterMatchingMode())}
      class="btn-sm border text-xs transition-colors {matchingModeActive
        ? 'border-blue-500/60 bg-blue-500/20 text-blue-200'
        : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-blue-500/40'}"
      data-testid="matching-mode-toggle"
      title="List every item the filter below matches, and act on all of them"
    >
      {matchingModeActive ? '✓ ' : ''}Matching items
    </button>

    <span class="grow"></span>

    <!-- C5 (visual audit 2026-09-24): the cluster-grid controls below
         (purity legend, unlabeled filter, embedding plot, sort, subject,
         clarity) don't apply to the Ignored bucket's crop list, so they
         are hidden in that mode rather than left live and inert. -->
    {#if !ignoredModeActive}
      <!-- Color legend for the card border. The cluster grid uses border
         color to encode purity at a glance; without this strip the user
         has to mouse over each card to figure out what the colors mean. -->
      <div
        class="flex items-center gap-2 text-[10px] text-zinc-500"
        title="Card border color encodes cluster cohesion (the served bands): share of measured members whose nearest cluster centre is this one — not label agreement. See each card's cohesion chip tooltip for n, label agreement and labelled share."
      >
        <span class="flex items-center gap-1">
          <span class="inline-block h-2 w-3 rounded-sm border-2 border-green-500/60"
          ></span>
          {purityThresholds ? `≥${Math.round(purityThresholds.pure_min * 100)}%` : 'pure'}
        </span>
        <span class="flex items-center gap-1">
          <span class="inline-block h-2 w-3 rounded-sm border-2 border-orange-500/60"
          ></span>
          {purityThresholds
            ? `≥${Math.round(purityThresholds.mixed_min * 100)}%`
            : 'mixed'}
        </span>
        <span class="flex items-center gap-1">
          <span class="inline-block h-2 w-3 rounded-sm border-2 border-red-500/60"></span>
          {purityThresholds
            ? `<${Math.round(purityThresholds.mixed_min * 100)}%`
            : 'noisy'}
        </span>
        <span class="flex items-center gap-1">
          <span class="inline-block h-2 w-3 rounded-sm border-2 border-blue-500/60"
          ></span>
          sub-clustered
        </span>
      </div>

      <!-- Unlabeled-only filter: AHC clusters that didn't reach class-majority
         consensus surface as "Unlabeled #N". This toggle narrows the grid
         to just those, so operators can drain the unlabeled backlog in
         one pass (drag a cluster's reps into a class on the sidebar). -->
      <button
        type="button"
        onclick={() => {
          unlabeledOnly = !unlabeledOnly;
          if (unlabeledOnly) sort = 'size_desc';
        }}
        class="btn-sm border text-xs transition-colors {unlabeledOnly
          ? 'border-amber-500/60 bg-amber-500/20 text-amber-200'
          : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-amber-500/40'}"
        title="Show only clusters without a dominant class (need labeling)"
      >
        {unlabeledOnly ? '✓ ' : ''}Unlabeled only
        <span class="ml-1 font-mono text-[10px] text-zinc-500">({unlabeledCount})</span>
      </button>

      <!-- Embedding-plot toggle (curation-strategy plan §5.6): fully absent
         unless {API_PREFIX}/methods actually reports the overlay, same convention
         as the diverse overlay in <StrategyBar>. Replaces the card grid
         when active (never overlays it) — see the {#if showEmbeddingViz}
         branch below. Hidden on the slot-gallery view, which has its own
         grid + toolbar and no per-crop cluster_id to color by. -->
      {#if embeddingVizAvailable && !isSlotFilter}
        <button
          type="button"
          onclick={() => (showEmbeddingViz = !showEmbeddingViz)}
          class="btn-sm border text-xs transition-colors {showEmbeddingViz
            ? 'border-blue-500/60 bg-blue-500/20 text-blue-200'
            : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-blue-500/40'}"
          title="Toggle a 2-d embedding-projection scatter plot (replaces the grid); lasso-select feeds the same label/move actions as the grid"
        >
          {showEmbeddingViz ? '✓ ' : ''}Embedding plot
        </button>
      {/if}

      <label class="flex items-center gap-2 text-xs text-zinc-400">
        Sort
        <select bind:value={sort} class="select-sm">
          <option value="purity_asc">cohesion asc</option>
          <option value="purity_desc">cohesion desc</option>
          <option value="size_desc">size desc</option>
          <option value="size_asc">size asc</option>
          <option value="dominant_class">dominant class</option>
        </select>
      </label>

      <!-- Primary-subject grid filters: scope cards to the largest / clear
         crops. Card size + reps reflect only passing crops, so a filtered
         grid is ready to drag-drop + AHC-refine on the subjects that matter. -->
      <div class="text-xs">
        <SubjectScopeToggle bind:value={subjectScope} />
      </div>
      <div class="text-xs">
        <BlurSlider
          bind:value={blurSlider}
          oncommit={commitClusterBlur}
          max={BLUR_MAX}
          width="w-28"
          title="Hide crops blurrier than this (blur_lap_ratio)"
        />
      </div>
    {/if}
  </div>

  <!-- Shared item filter: scopes the cluster grid, or the Matching items list. -->
  {#if !isSlotFilter && !ignoredModeActive && !searchModeActive && !itemTextModeActive}
    <div
      class="border-b border-zinc-800 bg-zinc-900/40 px-4 py-2"
      data-testid="clusters-filter-bar"
    >
      <ItemFilterBar
        state={itemFilter}
        visible={(p) => p !== 'max_rank'}
        showOpenVocab={matchingModeActive}
        onchange={syncFilterUrl}
      />
    </div>
  {/if}

  <!-- Grid -->
  <div class="flex-1 overflow-auto p-4">
    {#if searchModeActive}
      <!-- Global dataset-wide search results — a mode swap over the card
           grid, not a new page (mirrors showEmbeddingViz's own swap
           above). Infinite scroll is not offered: {API_PREFIX}/search/text isn't
           paginated the way the cluster grid's sentinel expects. -->
      <div class="mb-3 flex items-center gap-2 text-xs">
        <span class="text-zinc-300">
          <strong class="text-zinc-100">{searchTotal.toLocaleString()}</strong>
          result{searchTotal === 1 ? '' : 's'} for
          <span class="font-medium text-blue-200">"{searchQuery}"</span>
          across the dataset
        </span>
        <span class="grow"></span>
        <button
          type="button"
          class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
          onclick={exitSearchMode}
        >
          ← Back to clusters
        </button>
      </div>
      {#if searchResults.length === 0}
        <p class="text-sm text-zinc-500">No crops matched that search.</p>
      {:else}
        <CropResultGrid
          items={searchResults}
          sel={searchSel}
          scoreOf={(c) => searchScores.get(c.id) ?? null}
          ondetail={(c) => (detailSearchCrop = c)}
        >
          {#snippet cornerBadge(crop)}
            {#if crop.cluster_id != null}
              <ClusterBadge
                clusterId={crop.cluster_id}
                dominantClassName={clusterMetaMap.get(crop.cluster_id)
                  ?.dominant_class_name ?? null}
              />
            {/if}
          {/snippet}
        </CropResultGrid>
      {/if}
    {:else if ignoredModeActive}
      <!-- Ignored bucket (G7): excluded crops, restorable via
           batch_unexclude. Same mode-swap pattern as search above. -->
      <div class="mb-3 flex items-center gap-2 text-xs">
        <span class="text-zinc-300">
          <strong class="text-zinc-100">{ignoredTotal.toLocaleString()}</strong>
          ignored crop{ignoredTotal === 1 ? '' : 's'}
        </span>
        <span class="grow"></span>
        <button
          type="button"
          class="btn-sm border border-green-500/50 bg-green-500/10 text-green-200 hover:bg-green-500/20"
          onclick={() => void restoreIgnored()}
          disabled={ignoredSel.size === 0}
        >
          Restore selected ({ignoredSel.size})
        </button>
        <button
          type="button"
          class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
          onclick={exitIgnoredMode}
        >
          ← Back to clusters
        </button>
      </div>
      {#if ignoredLoading && ignoredItems.length === 0}
        <p class="text-sm text-zinc-500">Loading...</p>
      {:else if ignoredItems.length === 0}
        <p class="text-sm text-zinc-500">Nothing ignored.</p>
      {:else}
        <CropResultGrid
          items={ignoredItems}
          sel={ignoredSel}
          ondetail={(c) => (detailSearchCrop = c)}
        >
          {#snippet cornerBadge(crop)}
            {#if crop.excluded_reason}
              <span
                class="rounded border border-amber-500/50 bg-amber-500/15 px-1 py-0.5 text-[9px] text-amber-200"
              >
                {crop.excluded_reason}
              </span>
            {/if}
          {/snippet}
        </CropResultGrid>
      {/if}
    {:else if itemTextModeActive}
      <!-- Item-text search results (G8). -->
      <div class="mb-3 flex items-center gap-2 text-xs">
        <span class="text-zinc-300">
          <strong class="text-zinc-100">{itemTextTotal.toLocaleString()}</strong>
          result{itemTextTotal === 1 ? '' : 's'} for
          <span class="font-medium text-blue-200">"{itemTextQuery}"</span>
        </span>
        <span class="grow"></span>
        <button
          type="button"
          class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
          onclick={exitItemTextMode}
        >
          ← Back to clusters
        </button>
      </div>
      {#if itemTextLoading && itemTextItems.length === 0}
        <p class="text-sm text-zinc-500">Loading...</p>
      {:else if itemTextError}
        <!-- m18: the 400 detail is already shown next to the input above
             (line ~762) — showing the empty-state copy here too read as
             two contradictory messages ("here's why it failed" AND
             "0 results… no crops matched"). Show only the error. -->
      {:else if itemTextItems.length === 0}
        <p class="text-sm text-zinc-500">No crops matched that text.</p>
      {:else}
        <CropResultGrid
          items={itemTextItems}
          sel={itemTextSel}
          ondetail={(c) => (detailSearchCrop = c)}
        />
      {/if}
    {:else if matchingModeActive}
      <MatchingItemsView
        query={matchingQuery}
        body={matchingBody}
        onexit={exitMatchingMode}
        ondetail={(c) => (detailSearchCrop = c)}
      />
    {:else if showEmbeddingViz}
      <!-- Replaces the card grid entirely (plan §5.6 — no layout thrash
           from showing both at once). Lazily mounted: this is the only
           place <EmbeddingPlot> appears, so it never instantiates (never
           fetches) while the toggle is off. -->
      <EmbeddingPlot
        classId={classFilter}
        bannerRequired={embeddingVizBannerRequired}
        coveragePoolTotal={embeddingVizCoverageTotal}
      />
    {:else if slotGallery}
      <!-- Slot gallery: SlotGallery.svelte, backed by
           slotGalleryController.svelte.ts. -->
      <SlotGallery gallery={slotGallery} />
    {:else if clusterPager.loading && gridItems.length === 0}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if clusterPager.error}
      <p class="text-sm text-red-300">API unavailable: {clusterPager.error}</p>
    {:else if gridItems.length === 0}
      <p class="text-sm text-zinc-500">
        No clusters yet — ingest some images and run the auto-label pipeline.
      </p>
    {:else}
      <ul class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {#each gridItems as c (c.isSlotCard ? `slot-${c.id}` : c.id)}
          {@const pb = purityBadge(c)}
          <li style="content-visibility:auto;contain-intrinsic-size:auto 280px">
            <button
              type="button"
              class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {borderColor(
                c,
              )}"
              onclick={() => open(c)}
            >
              <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
                {#each c.representative_crop_ids?.slice(0, 4) ?? [] as cropId, i (i)}
                  <img
                    src={c.representative_thumb_urls?.[i]
                      ? resolveApiUrl(c.representative_thumb_urls[i])
                      : getThumbUrl(cropId)}
                    alt="thumb"
                    loading="lazy"
                    class="aspect-square w-full bg-zinc-950 object-contain"
                  />
                {/each}
                {#each Array(Math.max(0, 4 - (c.representative_crop_ids?.length ?? 0))) as _, i (i)}
                  <div class="aspect-square w-full bg-zinc-900"></div>
                {/each}
              </div>
              <div class="p-3">
                <div class="mb-1 flex items-center gap-2">
                  {#if c.isSlotCard}
                    <span class="text-sm font-semibold" data-testid="slot-card-title"
                      >{c.slotDisplayName ?? c.dominant_class_name}</span
                    >
                    <span
                      class="rounded px-1.5 py-0.5 text-[10px] font-medium bg-purple-500/20 text-purple-200"
                    >
                      inventory
                    </span>
                  {:else}
                    <span class="text-sm font-semibold">#{c.id}</span>
                    {#if pb}
                      <!-- DQ-M2 fix (dq-queues cutover, 2026-09-24): purity is now
                           nearest-centroid geometry purity, not the tautological
                           label-based number (always 1.0 for a class cluster) —
                           shown with its basis and n (it's noisy at low n), plus
                           label_purity/labelled_share in the tooltip so an
                           operator can tell the two apart. -->
                      <span
                        class="rounded px-1.5 py-0.5 text-[10px] font-medium {pb.color}"
                        data-testid="cluster-cohesion"
                        title="{COHESION_TOOLTIP} · label agreement {c.label_agreement !=
                        null
                          ? `${(c.label_agreement * 100).toFixed(0)}%`
                          : '—'} · labelled share {c.labelled_share != null
                          ? `${(c.labelled_share * 100).toFixed(0)}%`
                          : '—'}"
                      >
                        {pb.text} · {cohesionText(c) ?? 'cohesion —'}
                      </span>
                    {/if}
                  {/if}
                  {#if c.promotable}
                    <span
                      class="rounded border border-emerald-500/40 bg-emerald-500/20 px-1.5 py-0.5 text-[10px] text-emerald-200"
                      title="Meets the server's auto-promote gate (purity + member count + labelled share)"
                    >
                      promotable
                    </span>
                  {/if}
                  {#if c.has_subclusters}
                    <span
                      class="rounded border border-blue-500/40 bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-200"
                    >
                      AHC
                    </span>
                  {/if}
                  <span class="grow"></span>
                  {#if c.isSlotCard}
                    <!-- C4: the inventory count's scope, see SlotGallery. -->
                    <span
                      class="font-mono text-xs text-zinc-400"
                      title="Regions the browse endpoint lists: test-holdout items and active filters excluded. The sidebar's class count and the dashboard total include test-holdout items, and the review queue applies its own queue filters, so those can differ."
                      >{c.size.toLocaleString()} listed</span
                    >
                  {:else}
                    <span class="font-mono text-xs text-zinc-400">{c.size}</span>
                  {/if}
                </div>
                <div
                  class="truncate text-sm text-zinc-300"
                  title={unlabeledOnly
                    ? `Unlabeled cluster #${c.id}`
                    : (c.dominant_class_name ?? `Unlabeled cluster #${c.id}`)}
                >
                  {#if c.isSlotCard}
                    <!-- M4: dominant_pct is meaningless for the inventory
                         card too (there's no "dominant" anything — every
                         item IS the slot's class) — no invented "· 0%".
                         F8 D7: the title carries the served display name,
                         so this line names the class once, as a class. -->
                    <span class="text-zinc-500">class</span>
                    <span class="font-mono">{c.dominant_class_name}</span>
                  {:else if c.dominant_class_name && !unlabeledOnly}
                    {c.dominant_class_name}
                    {#if dominantShareText(c)}
                      <span class="text-zinc-500" title={dominantShareTitle(c)}>
                        · {dominantShareText(c)}
                      </span>
                    {/if}
                  {:else}
                    <span class="text-amber-300">Unlabeled #{c.id}</span>
                    <span class="text-zinc-500">· needs label</span>
                  {/if}
                </div>
              </div>
            </button>
          </li>
        {/each}
      </ul>
      <!-- Sentinel MUST live inside the scroll container so the IntersectionObserver
           can root itself on the right element. Outside the overflow-auto div the
           observer falls back to viewport and either never fires or fires forever. -->
      <div
        use:infiniteScroll={{
          onload: loadMore,
          disabled:
            (clusterPager.loadingMore || !clusterPager.hasMore || clusterPager.loading) &&
            (repsLoading || !repsHasMore),
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
    {/if}
  </div>

  <!-- Status bar (no scroll sentinel here — see above).
       DQ-p2 (docs/design/data-quality-pass-2026-09-24.md): this always
       read off clusterPager (the cluster-grid pager) — "102 / 102 all
       loaded" under a much larger slot-gallery grid, because the gallery
       view (isSlotFilter) renders slotGallery.pager.items,
       a completely different pager, but this footer never switched to
       match. -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    {#if matchingModeActive}
      <span class="font-mono text-xs text-zinc-400">matching items</span>
    {:else if ignoredModeActive}
      <span class="font-mono text-xs text-zinc-500" data-testid="clusters-footer-count">
        {ignoredItems.length} / {ignoredTotal} ignored crops
      </span>
      <span class="font-mono text-xs text-zinc-400">
        {ignoredLoading ? 'loading…' : 'ignored bucket'}
      </span>
    {:else if slotGallery}
      <span class="font-mono text-xs text-zinc-500">
        {slotGallery.pager.items.length} / {slotGallery.pager.total}
      </span>
      <span class="font-mono text-xs text-zinc-400">
        {#if slotGallery.pager.loadingMore}loading more…{:else if slotGallery.pager.hasMore}scroll
          for more{:else}all loaded{/if}
      </span>
    {:else}
      <span class="font-mono text-xs text-zinc-500" data-testid="clusters-footer-count">
        {gridItems.length} / {clusterPager.total +
          (classFilter == null ? slotInventoryCards.length : 0)}
      </span>
      <span class="font-mono text-xs text-zinc-400">
        {#if clusterPager.loadingMore}loading more…{:else if clusterPager.hasMore}scroll
          for more{:else}all loaded{/if}
      </span>
    {/if}
  </div>
</div>

{#if detailSearchCrop}
  <CropDetailModal crop={detailSearchCrop} onclose={() => (detailSearchCrop = null)} />
{/if}
