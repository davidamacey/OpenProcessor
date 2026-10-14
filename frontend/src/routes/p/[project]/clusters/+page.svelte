<script lang="ts">
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { goto } from '$app/navigation';
  import { untrack } from 'svelte';
  import { page } from '$app/state';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import {
    createSlotGalleryController,
    type SlotGalleryController,
  } from '$lib/slots/slotGalleryController.svelte';
  import { isEmbeddingVizAvailable, isEmbeddingVizBannerRequired } from '$lib/strategies';
  import { findActiveClassByName } from '$lib/classNameKey';
  import { createClusterListController } from '$lib/clusters/clusterListController.svelte';
  import { createClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import { createSlotInventory } from '$lib/clusters/slotInventory.svelte';
  import ClustersGrid from '$lib/components/clusters/ClustersGrid.svelte';
  import ClustersStatusBar from '$lib/components/clusters/ClustersStatusBar.svelte';
  import ClustersToolbar from '$lib/components/clusters/ClustersToolbar.svelte';
  import CropDetailModal from '$lib/components/CropDetailModal.svelte';
  import ItemFilterBar from '$lib/components/itemFilter/ItemFilterBar.svelte';
  import {
    ItemFilterState,
    withoutOpenVocab,
  } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';
  import type { Crop } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';

  const slotInventory = createSlotInventory();
  const list = createClusterListController({
    get classFilter() {
      return classFilter;
    },
    get isSlotFilter() {
      return isSlotFilter;
    },
    get gridFilterKey() {
      return gridFilterKey;
    },
    slotInventory,
    gridItemFilter,
  });
  const modes = createClusterSearchModes({
    clusterPager: list.clusterPager,
    get showEmbeddingViz() {
      return showEmbeddingViz;
    },
    set showEmbeddingViz(v: boolean) {
      showEmbeddingViz = v;
    },
  });
  let detailSearchCrop = $state<Crop | null>(null);

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
    if (modes.searchModeActive) modes.exitSearchMode();
    if (modes.ignoredModeActive) modes.exitIgnoredMode();
    if (modes.itemTextModeActive) modes.exitItemTextMode();
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

  // Embedding-plot overlay (curation-strategy plan Phase 5 —
  // docs/curation-strategy-plan-2026-09.md §2.7/§5.6). Off by default,
  // lazily mounted: <EmbeddingPlot> only appears in ClustersGrid inside
  // its showEmbeddingViz branch, so it never instantiates
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

  // M4: `loadSlotInventoryCards` used to run exactly once, right after the
  // first `loadFirst()` — if the root layout's own `classesStore.acquire()`
  // fetch hadn't resolved yet at that moment, `classesStore.classes` was
  // still empty, the slot class lookup failed, and the card never
  // retried (observed live: ~1 render in 8). Re-running whenever the
  // classes list changes (classesStore's own 30s poll, or a slower first
  // load) makes the card deterministic instead of a load-order race.
  $effect(() => {
    if (
      slotInventory.slotInventoryCards.length === 0 &&
      classesStore.classes.length > 0 &&
      list.clusterPager.error == null &&
      !isSlotFilter
    ) {
      untrack(() => void slotInventory.loadSlotInventoryCards());
    }
  });

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

  // Re-load the gallery whenever a filter, the top-N rank gate, or the
  // selected region cluster changes (typed filters debounce, the cluster
  // grid reloads only for the rank gate or a closed bucket; see
  // `reloadOnFilterChange`).
  $effect(() => {
    slotGallery?.reloadOnFilterChange();
  });
</script>

<div class="flex h-full flex-col">
  <ClustersToolbar
    {list}
    {modes}
    {classFilter}
    {isSlotFilter}
    {gridItemFilter}
    {matchingModeActive}
    {enterMatchingMode}
    {exitMatchingMode}
    {embeddingVizAvailable}
    bind:showEmbeddingViz
  />

  <!-- Shared item filter: scopes the cluster grid, or the Matching items list. -->
  {#if !isSlotFilter && !modes.ignoredModeActive && !modes.searchModeActive && !modes.itemTextModeActive}
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

  <ClustersGrid
    {list}
    {modes}
    {classFilter}
    {matchingModeActive}
    {matchingQuery}
    {matchingBody}
    {exitMatchingMode}
    {showEmbeddingViz}
    {embeddingVizBannerRequired}
    {embeddingVizCoverageTotal}
    {slotGallery}
    ondetail={(c) => (detailSearchCrop = c)}
  />

  <ClustersStatusBar
    {list}
    {modes}
    {slotInventory}
    {classFilter}
    {matchingModeActive}
    {slotGallery}
  />
</div>

{#if detailSearchCrop}
  <CropDetailModal crop={detailSearchCrop} onclose={() => (detailSearchCrop = null)} />
{/if}
