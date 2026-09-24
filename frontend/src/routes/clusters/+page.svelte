<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import {
    bulkLabel,
    excludeCrops,
    getClusters,
    getPlates,
    getRegionThumbUrl,
    getThumbUrl,
    resolveApiUrl,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';
  import { createPager } from '$lib/pager.svelte';
  import { createPlateGalleryController } from './plateGalleryController.svelte';
  import { createSelection } from '$lib/selection.svelte';
  import {
    isEmbeddingVizAvailable,
    isEmbeddingVizBannerRequired,
    isSemanticSearchAvailable,
  } from '$lib/strategies';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import ClusterBadge from '$lib/components/ClusterBadge.svelte';
  import CropDetailModal from '$lib/components/CropDetailModal.svelte';
  import CropResultGrid from '$lib/components/CropResultGrid.svelte';
  import EmbeddingPlot from '$lib/components/EmbeddingPlot.svelte';
  import SlotGallery from '$lib/components/slots/SlotGallery.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import type { ClusterFilter, RegistryClass, Cluster, Crop } from '$lib/types';
  import { toastStore } from '$stores/toast.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { undoStore } from '$stores/undo.svelte';

  // Cluster grid pager. One params builder (clusterQuery) feeds page 1 and
  // every later page, so a filter can't be sent on the first request and
  // silently dropped on the next.
  const clusterPager = createPager<Cluster>({
    fetchPage: async (page) => await getClusters(clusterQuery(page)),
    keyOf: (c) => String(c.id),
  });

  // Synthetic license_plate gallery card. Plates are sub-bboxes on
  // vehicle crops, not FAISS docs, so the cluster grid never produces
  // a card for them. We surface one explicitly using {API_PREFIX}/regions so the
  // operator can click into the plate inventory the same way they click
  // into any other class cluster. Card is null until the first {API_PREFIX}/regions
  // call resolves; the cluster grid hides it during that window.
  let lpCard = $state<Cluster | null>(null);

  // Persist the cluster-list filter (sort + unlabeled-only) across
  // navigation so going into a cluster and back keeps the operator's
  // last view — they shouldn't have to re-click "Unlabeled only" every
  // time. sessionStorage survives back-nav + refresh within the session
  // regardless of how the user returns (back button, link, etc.).
  const FILTER_PERSIST_KEY = 'clusters_filter_v1';
  function loadPersistedFilter(): { sort?: string; unlabeledOnly?: boolean } | null {
    if (typeof sessionStorage === 'undefined') return null;
    try {
      return JSON.parse(sessionStorage.getItem(FILTER_PERSIST_KEY) ?? 'null');
    } catch {
      return null;
    }
  }
  const _persistedFilter = loadPersistedFilter();

  let sort = $state<NonNullable<ClusterFilter['sort']>>(
    (_persistedFilter?.sort as NonNullable<ClusterFilter['sort']>) ?? 'purity_asc',
  );
  let unlabeledOnly = $state<boolean>(_persistedFilter?.unlabeledOnly ?? false);
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
    sessionStorage.setItem(FILTER_PERSIST_KEY, JSON.stringify({ sort, unlabeledOnly }));
  });

  // --- Plate browse (replaces the "License plates aren't clustered" placeholder
  //     when the operator selects the license_plate class filter).
  // State/logic extracted to plateGalleryController.svelte.ts (P2.6,
  // docs/genericization-plan-2026-09-13.md §3.4/§5a) -- SlotGallery.svelte
  // owns rendering, this route owns the class-filter routing decision and
  // the URL/filter-driven reload effects below.
  const plateGallery = createPlateGalleryController();

  const classFilter = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    return v == null ? null : Number.isFinite(+v) ? +v : null;
  });

  // This backend stores plates as a *sub-bbox* on each vehicle
  // crop (`region_bbox_norm`), NOT as standalone docs in the cluster
  // index. So filtering this page by a slot-bound class (e.g.
  // license_plate) always returns 0 / unlabeled clusters — confusing
  // operators who expect to see plate clusters here. Detect that case
  // and route the user to the slot's browse gallery instead, which is
  // the actual home for that slot's labeling. Driven by registeredSlots
  // (P2.10) instead of a hardcoded license_plate string literal.
  const isLicensePlateFilter = $derived.by<boolean>(() => {
    if (classFilter == null) return false;
    const cls = classesStore.classes.find((c) => c.id === classFilter);
    return slotForClassName(cls?.name) != null;
  });

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

  // Cluster-origin badges: build a Map from whatever's already loaded for
  // the card grid (getClusters({}) returns every cluster, up to 2000, in
  // one call with dominant_class_name precomputed server-side — never
  // recompute dominant class client-side, per CLAUDE.md). If a search
  // result's cluster_id isn't in that map (e.g. the grid was itself
  // filtered by class), fall back to one additional unfiltered call
  // rather than showing a blank badge.
  let clusterMetaMap = $state(new Map<number, Cluster>());
  $effect(() => {
    const m = new Map<number, Cluster>();
    for (const c of clusterPager.items) m.set(c.id, c);
    clusterMetaMap = m;
  });
  async function ensureClusterMeta(ids: number[]): Promise<void> {
    const missing = ids.filter((id) => !clusterMetaMap.has(id));
    if (missing.length === 0) return;
    try {
      const res = await getClusters({});
      const m = new Map(clusterMetaMap);
      for (const c of res.items) m.set(c.id, c);
      clusterMetaMap = m;
    } catch (e) {
      toastStore.warn(`Could not load cluster info for badges: ${(e as Error).message}`);
    }
  }

  function syncSearchUrl(q: string | null): void {
    const url = new URL(page.url);
    if (q) url.searchParams.set('q', q);
    else url.searchParams.delete('q');
    void goto(`${url.pathname}${url.search}`, { replaceState: true, keepFocus: true });
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
    // Force the embedding plot off — same guard pattern the plate-browse
    // view already uses (isLicensePlateFilter effect above): the plot
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
          toastStore.error(`Label failed: ${(e as Error).message}`);
          for (const prior of priors) replaceSearchCrop(prior);
        }
      },
    );

    const offKeys: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offKeys.push(keyboardStore.register(combo, () => void fn(), 'clusters', desc));

    reg(
      'a',
      () => searchSel.selectAll(searchResults.map((c) => c.id)),
      'Select all results',
    );
    reg('escape', () => searchSel.clear(), 'Clear selection');
    reg(
      'x',
      async () => {
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
          toastStore.error(`Ignore failed: ${(e as Error).message}`);
        }
      },
      'Ignore selected (exclude from training)',
    );
    reg(
      'z',
      async () => {
        const crop = await undoStore.undoLast();
        if (crop) replaceSearchCrop(crop);
      },
      'Undo last action',
    );

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
  let showEmbeddingViz = $state<boolean>(false);
  // The synthetic plate-browse view (isLicensePlateFilter) has its own
  // grid + bulk-triage toolbar; the embedding plot projects vehicle
  // crops with a real cluster_id, which plates (sub-bboxes, not their
  // own cluster docs) never have. Force the toggle off rather than
  // leaving a stale plot mounted over a view it doesn't apply to.
  $effect(() => {
    if (isLicensePlateFilter && showEmbeddingViz) showEmbeddingViz = false;
  });

  // One params builder for both pages of the cluster grid. loadMore used
  // to omit max_rank / min_blur_ratio, so scrolling past page 1 appended
  // unfiltered clusters over a filtered page 1.
  function clusterQuery(page: number): ClusterFilter {
    return {
      class_id: classFilter ?? undefined,
      sort,
      page,
      page_size: pageSize,
      max_rank: maxRank,
      min_blur_ratio: minBlurRatio,
    };
  }

  async function loadFirst(): Promise<void> {
    await clusterPager.loadFirst();
    if (clusterPager.error == null) await loadLicensePlateCard();
  }

  // Build the synthetic license_plate gallery card. Plates live as
  // sub-bboxes on vehicle crops (not FAISS docs) so the cluster grid
  // never includes them. We query {API_PREFIX}/regions for the total inventory
  // and use the first 4 plate-bearing crops as thumbnails. Card is
  // null until this resolves; the grid renders it as the first item
  // when the unfiltered view is active.
  //
  // Deliberately keyed to `licensePlateSlot` specifically, NOT "the
  // first slot-bound class" (P2.11, docs/genericization-plan-2026-09-13.md
  // §9.1 finding): every line below calls the plate-specific
  // getPlates()/getRegionThumbUrl() endpoints, so silently aliasing to
  // whichever slot-bound class happened to be first would build a
  // plate card labeled with a DIFFERENT slot's class the moment a
  // second capable slot is registered. A generic per-slot synthetic
  // card is P2.7's SlotGallery parameterization (already scheduled,
  // not re-planned here) — until that lands, a slot with no
  // plate-shaped browse endpoint correctly gets no pinned card at all,
  // rather than an incorrect one.
  async function loadLicensePlateCard(): Promise<void> {
    const lp = classesStore.classes.find(
      (c) => c.name.toLowerCase() === licensePlateSlot.bind.className?.toLowerCase(),
    );
    if (!lp) {
      lpCard = null;
      return;
    }
    try {
      // Pull a slightly larger window than 4 so we can drop items
      // missing a plate sub-bbox without falling below the tile count.
      const res = await getPlates(licensePlateSlot.capabilities.queue!.browsePath, {
        page: 1,
        page_size: 12,
      });
      const withPlateBox = res.items.filter(
        (p) => Array.isArray(p.region_bbox_norm) && p.region_bbox_norm.length === 4,
      );
      const reps = withPlateBox.slice(0, 4);
      lpCard = {
        id: lp.id,
        size: res.total,
        // Purity badge is meaningless for a non-cluster — leave null.
        purity: null,
        dominant_class_id: lp.id,
        dominant_class_name: lp.name,
        dominant_pct: null,
        sub_clusters: 0,
        has_subclusters: false,
        representative_crop_ids: reps.map((p) => p.crop_id),
        // Show plate close-ups, not vehicle thumbnails — the whole
        // point of this card is that the operator is browsing plates.
        // API_PREFIX-relative /crops/{id}/region_thumbnail returns the plate
        // sub-bbox rendered to a 160px tile.
        representative_thumb_urls: reps.map((p) => getRegionThumbUrl(p.crop_id, 160)),
        updated_at: null,
      } as Cluster;
    } catch {
      lpCard = null;
    }
  }

  // Items rendered in the unfiltered cluster grid: synthetic LP card
  // prepended (when present) so the operator always has a visible
  // entry point to the plate inventory. With class filter active we
  // hand the user to the dedicated plate-browse branch already, so
  // skip the prepend there.
  // gridItems = (synthetic license_plate card if unfiltered) + clusters,
  // optionally narrowed to only the "Unlabeled" group when the toggle is on.
  // "Unlabeled" = cluster_kind !== 'class', i.e. the candidate (IVF/AHC)
  // and unassigned buckets the operator still needs to sort. Keying on
  // cluster_kind (not dominant_class_name) is the fix for "only 16
  // showed": candidate clusters dominated by gemma_unmatched crops DO
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
    // Keep the synthetic license_plate card pinned first (entry point to
    // the plate inventory), unaffected by sort, only on the unfiltered
    // labelled view. The real class-kind cluster for license_plate shares
    // its `id` with `lpCard` (cluster_id === class_id for class-kind
    // clusters) — drop it so the keyed #each below never sees a duplicate
    // key; lpCard is its replacement entry point, not an addition to it.
    if (classFilter == null && lpCard != null && !unlabeledOnly) {
      return [lpCard, ...sorted.filter((c) => c.id !== lpCard!.id)];
    }
    return sorted;
  });
  const unlabeledCount = $derived(
    clusterPager.items.filter((c) => c.cluster_kind !== 'class').length,
  );

  const loadMore = () => clusterPager.loadMore();

  $effect(() => {
    keyboardStore.setScope('clusters');
  });

  // Re-load on class-filter change — but only for the cluster view.
  // Sort is applied client-side (sortClusters) over the single loaded
  // batch, so changing it must NOT refetch (the endpoint returns the
  // same size-ordered data regardless).
  $effect(() => {
    void classFilter;
    void maxRank;
    void minBlurRatio;
    if (!isLicensePlateFilter) void loadFirst();
  });

  // Re-load plates whenever a filter, the top-N rank gate, or the selected
  // plate cluster changes. When no cluster is selected, also refresh the
  // cluster-card grid so it reflects the current rank gate.
  $effect(() => {
    void plateGallery.plateDetectorFilter;
    void plateGallery.plateVerifiedOnly;
    void plateGallery.plateMinScore;
    void plateGallery.plateTextQuery;
    void plateGallery.plateMaxRank;
    void plateGallery.selectedPlateCluster;
    if (isLicensePlateFilter) {
      void plateGallery.loadPlatesFirst();
      if (plateGallery.selectedPlateCluster == null)
        void plateGallery.loadPlateClusters();
    }
  });

  // Purity banding is served (`purity_tier`, `{API_PREFIX}/clusters`'
  // `purity_thresholds`) — no client 0.8/0.6 threshold. `has_subclusters`
  // is a separate, unrelated signal (AHC sub-clustering ran) and still
  // wins the border color outright.
  function borderColor(c: Cluster): string {
    if (c.has_subclusters) return 'border-blue-500/60';
    if (c.purity_tier === 'pure') return 'border-green-500/60';
    if (c.purity_tier === 'mixed') return 'border-orange-500/60';
    return 'border-red-500/60';
  }

  function purityBadge(c: Cluster): { color: string; text: string } {
    if (c.purity_tier === 'pure')
      return { color: 'bg-green-500/20 text-green-300', text: 'pure' };
    if (c.purity_tier === 'mixed')
      return { color: 'bg-orange-500/20 text-orange-200', text: 'mixed' };
    return { color: 'bg-red-500/20 text-red-200', text: 'noisy' };
  }

  function open(c: Cluster): void {
    // Special-case: clicking a cluster whose dominant class is bound to
    // a registered slot (e.g. license_plate) should jump to that slot's
    // browse view (which surfaces every crop with the slot's sub-bbox),
    // not the single-cluster crop grid. Slot sub-bboxes live on vehicle
    // crops so that class's cluster only contains the rare crops that
    // were labeled with it as their PRIMARY class — usually 1-2
    // mis-labels. The operator's intent is "show me all the slot's
    // items", so route them to that inventory instead. Driven by
    // registeredSlots (P2.10) instead of a hardcoded license_plate
    // string literal.
    //
    // P2.11 fix: resolve the CLICKED cluster's own dominant class, not
    // "the first slot-bound class" — the latter silently routed every
    // slot-bound cluster to the first registered slot's class filter,
    // which breaks the instant a second capable slot exists.
    if (slotForClassName(c.dominant_class_name) != null) {
      const target = (c.dominant_class_name ?? '').toLowerCase();
      const cls = classesStore.classes.find((k) => k.name.toLowerCase() === target);
      if (cls) {
        void goto(`/clusters?class=${cls.id}`);
        return;
      }
    }
    void goto(`/clusters/${c.id}`);
  }

  // Infinite scroll owns pagination — totalPages no longer needed.
</script>

<div class="flex h-full flex-col">
  <!-- Toolbar -->
  <div class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
    <h1 class="text-lg font-semibold">Clusters</h1>
    <ShortcutsButton />

    {#if classFilter != null}
      <span
        class="rounded-md border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-xs text-blue-200"
      >
        class filter: #{classFilter}
      </span>
    {/if}

    <!-- Dataset-wide semantic search — left-aligned, first control after
         the title, so it reads as the primary way in rather than a
         control squeezed between unrelated toggles. Deliberately no
         cluster_id/tab scope in `filter` — unscoped-across-the-whole-dataset
         is the entire point of this control, unlike the cluster_id-scoped
         SemanticSearchBox on /clusters/[id]. -->
    {#if semanticSearchAvailable && !isLicensePlateFilter}
      <SemanticSearchBox
        pageSize={200}
        filter={classFilter != null ? { class_id: classFilter } : {}}
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

    <span class="grow"></span>

    <!-- Color legend for the card border. The cluster grid uses border
         color to encode purity at a glance; without this strip the user
         has to mouse over each card to figure out what the colors mean. -->
    <div
      class="flex items-center gap-2 text-[10px] text-zinc-500"
      title="Card border color encodes cluster purity"
    >
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-green-500/60"></span>
        ≥80%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-orange-500/60"
        ></span>
        ≥60%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-red-500/60"></span>
        &lt;60%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-blue-500/60"></span>
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
         branch below. Hidden on the plate-browse view, which has its own
         grid + toolbar and no per-crop cluster_id to color by. -->
    {#if embeddingVizAvailable && !isLicensePlateFilter}
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
        <option value="purity_asc">purity asc</option>
        <option value="purity_desc">purity desc</option>
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
  </div>

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
    {:else if showEmbeddingViz}
      <!-- Replaces the card grid entirely (plan §5.6 — no layout thrash
           from showing both at once). Lazily mounted: this is the only
           place <EmbeddingPlot> appears, so it never instantiates (never
           fetches) while the toggle is off. -->
      <EmbeddingPlot classId={classFilter} bannerRequired={embeddingVizBannerRequired} />
    {:else if isLicensePlateFilter}
      <!-- Plates list view -- extracted to SlotGallery.svelte (P2.6),
           backed by plateGalleryController.svelte.ts. -->
      <SlotGallery gallery={plateGallery} />
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
        {#each gridItems as c (c.id)}
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
                {#each c.representative_crop_ids?.slice(0, 4) ?? [] as cropId, i (cropId)}
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
                  <span class="text-sm font-semibold">#{c.id}</span>
                  <span class="rounded px-1.5 py-0.5 text-[10px] font-medium {pb.color}">
                    {pb.text}
                    {((c.purity ?? 0) * 100).toFixed(0)}
                  </span>
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
                  <span class="font-mono text-xs text-zinc-400">{c.size}</span>
                </div>
                <div
                  class="truncate text-sm text-zinc-300"
                  title={unlabeledOnly
                    ? `Unlabeled cluster #${c.id}`
                    : (c.dominant_class_name ?? `Unlabeled cluster #${c.id}`)}
                >
                  {#if c.dominant_class_name && !unlabeledOnly}
                    {c.dominant_class_name}
                    <span class="text-zinc-500">
                      · {((c.dominant_pct ?? 0) * 100).toFixed(0)}%
                    </span>
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
            clusterPager.loadingMore || !clusterPager.hasMore || clusterPager.loading,
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
    {/if}
  </div>

  <!-- Status bar (no scroll sentinel here — see above) -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="font-mono text-xs text-zinc-500">
      {gridItems.length} / {clusterPager.total +
        (classFilter == null && lpCard != null ? 1 : 0)}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if clusterPager.loadingMore}loading more…{:else if clusterPager.hasMore}scroll for
        more{:else}all loaded{/if}
    </span>
  </div>
</div>

{#if detailSearchCrop}
  <CropDetailModal crop={detailSearchCrop} onclose={() => (detailSearchCrop = null)} />
{/if}
