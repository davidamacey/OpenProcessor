<script lang="ts">
  import { goto } from '$app/navigation';
  import { untrack } from 'svelte';
  import { page } from '$app/state';
  import {
    ApiError,
    bulkLabel,
    excludeCrops,
    getClusters,
    getCrops,
    getPlates,
    getRegionThumbUrl,
    getThumbUrl,
    resolveApiUrl,
    unexcludeCrops,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';
  import { createPager } from '$lib/pager.svelte';
  import { idsNeedingRepresentatives } from '$lib/clusters/displayOrderRepresentatives';
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
    if (v == null) return null;
    if (Number.isFinite(+v)) return +v;
    // m22: a name-form deep link (`?class=license_plate`) used to be
    // silently ignored — only a numeric class id worked, so
    // `/clusters?class=license_plate` rendered the unfiltered grid
    // instead of routing to the plate gallery. Resolve the name against
    // the loaded registry, same lookup `open()` already does in the
    // opposite direction (cluster -> class name -> id -> slot route).
    const byName = classesStore.classes.find(
      (c) => c.name.toLowerCase() === v.toLowerCase(),
    );
    return byName?.id ?? null;
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
      toastStore.error(`Could not load ignored crops: ${(e as Error).message}`);
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
      toastStore.error(`Restore failed: ${(e as Error).message}`);
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
        toastStore.error(`Item-text search failed: ${(e as Error).message}`);
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
      // Badge lookup only reads dominant_class_name/purity/etc — no
      // representatives needed, so skip that window entirely (D-4).
      const res = await getClusters({ representatives_limit: 0 });
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
        const crops = await undoStore.undoLast();
        for (const crop of crops) replaceSearchCrop(crop);
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
  // m20 (2026-09-24 interactive pass): served field_coverage_total for
  // the viz_projection overlay, so EmbeddingPlot can note when it's
  // showing fewer points than the pool the fit was computed over.
  const embeddingVizCoverageTotal = $derived(
    strategiesStore.methods.overlays.find((o) => o.id === 'viz_projection')
      ?.field_coverage_total ?? null,
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
        dominant_class_id: lp.id,
        dominant_class_name: lp.name,
        dominant_pct: null,
        n_subclusters: 0,
        sub_clusters: 0,
        has_subclusters: false,
        representative_crop_ids: reps.map((p) => p.crop_id),
        // Show plate close-ups, not vehicle thumbnails — the whole
        // point of this card is that the operator is browsing plates.
        // API_PREFIX-relative /crops/{id}/region_thumbnail returns the plate
        // sub-bbox rendered to a 160px tile.
        representative_thumb_urls: reps.map((p) => getRegionThumbUrl(p.crop_id, 160)),
        updated_at: null,
        isSlotCard: true,
      };
    } catch {
      lpCard = null;
    }
  }

  // M4: `loadLicensePlateCard` used to run exactly once, right after the
  // first `loadFirst()` — if the root layout's own `classesStore.acquire()`
  // fetch hadn't resolved yet at that moment, `classesStore.classes` was
  // still empty, the license_plate class lookup failed, and the card never
  // retried (observed live: ~1 render in 8). Re-running whenever the
  // classes list changes (classesStore's own 30s poll, or a slower first
  // load) makes the card deterministic instead of a load-order race.
  $effect(() => {
    if (
      lpCard == null &&
      classesStore.classes.length > 0 &&
      clusterPager.error == null &&
      !isLicensePlateFilter
    ) {
      void loadLicensePlateCard();
    }
  });

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
    // labelled view. M4: this used to drop the real class-kind cluster
    // sharing license_plate's id (cluster_id === class_id for class-kind
    // clusters) on the theory that lpCard replaces it — but that cluster
    // (e.g. #80, size 1) is a real, independently-reachable cluster with
    // its own crops, and hiding it made it permanently unreachable from
    // this grid. The two now render side by side; the #each key below is
    // keyed off `isSlotCard` so the synthetic entry never collides with
    // the real cluster's id.
    if (classFilter == null && lpCard != null && !unlabeledOnly) {
      return [lpCard, ...sorted];
    }
    return sorted;
  });

  // DQ-M4 (docs/design/data-quality-pass-2026-09-24.md): representatives
  // fetched per-card, in the operator's actual DISPLAY order (gridItems,
  // just above), not the backend's fixed size-desc order. The `/clusters`
  // endpoint has no batch-by-id representatives param (checked against
  // contracts/openprocessor/openapi/curation.json) — only a single
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
      await loadLicensePlateCard();
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

  // M6: Z on the plate gallery reverses the most recent region write
  // (bulk status change or a single bbox edit) — same key, same
  // undoStore, as the card-grid view's label-undo Z above; see
  // plateGalleryController's undoLastPlateAction doc comment.
  $effect(() => {
    if (!isLicensePlateFilter) return;
    const off = keyboardStore.register(
      'z',
      () => void plateGallery.undoLastPlateAction(),
      'clusters',
      'Undo last plate action',
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
    if (!isLicensePlateFilter) untrack(() => void loadFirst());
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
    if (!isLicensePlateFilter) untrack(() => void loadMoreRepresentatives());
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
    // M4: a distinct, neutral border for the non-cluster inventory card —
    // the red "noisy" border would otherwise falsely imply a bad cluster.
    if (c.isSlotCard) return 'border-purple-500/60';
    if (c.has_subclusters) return 'border-blue-500/60';
    if (c.purity_tier === 'pure') return 'border-green-500/60';
    if (c.purity_tier === 'mixed') return 'border-orange-500/60';
    return 'border-red-500/60';
  }

  function purityBadge(c: Cluster): { color: string; text: string } | null {
    // M4: the synthetic license_plate inventory card is not a cluster —
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

    <span class="grow"></span>

    <!-- Color legend for the card border. The cluster grid uses border
         color to encode purity at a glance; without this strip the user
         has to mouse over each card to figure out what the colors mean. -->
    <div
      class="flex items-center gap-2 text-[10px] text-zinc-500"
      title="Card border color encodes cluster purity — nearest-centroid geometry purity (DQ-M2), not label agreement. See each card's purity chip tooltip for n and label_purity/labelled_share."
    >
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-green-500/60"></span>
        {purityThresholds ? `≥${Math.round(purityThresholds.pure_min * 100)}%` : 'pure'}
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-orange-500/60"
        ></span>
        {purityThresholds ? `≥${Math.round(purityThresholds.mixed_min * 100)}%` : 'mixed'}
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-red-500/60"></span>
        {purityThresholds ? `<${Math.round(purityThresholds.mixed_min * 100)}%` : 'noisy'}
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
                  {#if c.isSlotCard}
                    <span class="text-sm font-semibold">{c.dominant_class_name}</span>
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
                        title="{c.purity_basis ??
                          'nearest-centroid'} purity, n={c.purity_n ??
                          '—'} · label purity {c.label_purity != null
                          ? `${(c.label_purity * 100).toFixed(0)}%`
                          : '—'} · labelled share {c.labelled_share != null
                          ? `${(c.labelled_share * 100).toFixed(0)}%`
                          : '—'}"
                      >
                        {pb.text}
                        {((c.purity ?? 0) * 100).toFixed(0)}
                        {#if c.purity_n != null}
                          <span class="opacity-70">· n={c.purity_n}</span>
                        {/if}
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
                  <span class="font-mono text-xs text-zinc-400">{c.size}</span>
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
                         item IS the slot's class) — no invented "· 0%". -->
                    {c.dominant_class_name}
                  {:else if c.dominant_class_name && !unlabeledOnly}
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
       loaded" under a 60 / 1,000 plate grid, because the plate-gallery
       view (isLicensePlateFilter) renders plateGallery.platePager.items,
       a completely different pager, but this footer never switched to
       match. -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    {#if isLicensePlateFilter}
      <span class="font-mono text-xs text-zinc-500">
        {plateGallery.platePager.items.length} / {plateGallery.platePager.total}
      </span>
      <span class="font-mono text-xs text-zinc-400">
        {#if plateGallery.platePager.loadingMore}loading more…{:else if plateGallery.platePager.hasMore}scroll
          for more{:else}all loaded{/if}
      </span>
    {:else}
      <span class="font-mono text-xs text-zinc-500">
        {gridItems.length} / {clusterPager.total +
          (classFilter == null && lpCard != null ? 1 : 0)}
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
