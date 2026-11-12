<script lang="ts">
  import { untrack } from 'svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import {
    createSlotGalleryController,
    type SlotGalleryController,
  } from '$lib/slots/slotGalleryController.svelte';
  import { findActiveClassByName } from '$lib/classNameKey';
  import { createClusterListController } from '$lib/clusters/clusterListController.svelte';
  import { createClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import { createSlotInventory } from '$lib/clusters/slotInventory.svelte';
  import { clusterFilterToUrl } from '$lib/clusters/clusterListFilter';
  import ClusterListFilterBar from '$lib/components/clusters/ClusterListFilterBar.svelte';
  import ClustersGrid from '$lib/components/clusters/ClustersGrid.svelte';
  import ClustersStatusBar from '$lib/components/clusters/ClustersStatusBar.svelte';
  import ClustersToolbar from '$lib/components/clusters/ClustersToolbar.svelte';
  import CropDetailModal from '$lib/components/CropDetailModal.svelte';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';
  import type { Crop } from '$lib/types';
  import type { ClusterListFilter, ClusterSortId } from '$lib/types_clusterList';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const slotInventory = createSlotInventory();
  // The controller is created once per mount and never swapped.
  // svelte-ignore state_referenced_locally
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
    initialUrlParams: ws.searchParams,
    writeUrl: syncClusterListUrl,
  });
  const modes = createClusterSearchModes({
    clusterPager: list.clusterPager,
    gridItemFilter,
  });
  let detailSearchCrop = $state<Crop | null>(null);
  // The filter row's "N matching · M hidden" note: the served cluster count and hidden counts.
  $effect(() => {
    if (list.clusterPager.loading || list.clusterPager.error) return;
    ws.matching = { total: list.total, hidden: list.hidden };
    return () => {
      ws.matching = null;
    };
  });
  // Select-all applies where cards are items: the search results. The cluster cards are opened,
  // not selected, so the key keeps its browser meaning there.
  $effect(() => {
    if (!modes.searchModeActive) return;
    return ws.registerPage({
      kind: 'items',
      ids: () => modes.searchResults.map((c) => c.id),
      total: () => modes.searchTotal,
      apply: (ids) => modes.searchSel.selectAll(ids),
    });
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

  // A single class in the filter routes a slot-bound class to its gallery. The
  // name is resolved against the loaded registry (class identity is the name).
  const classFilterName = $derived(
    ws.itemQuery().class_name?.length === 1 ? ws.itemQuery().class_name![0]! : null,
  );
  const classFilter = $derived(
    classFilterName
      ? (findActiveClassByName(classesStore.classes, classFilterName)?.id ?? null)
      : null,
  );

  // The shared item filter lives in the Workspace controller. The largest-N
  // toggle below keeps `max_rank`, so the grid leaves it out of the list query.
  const matchingModeActive = $derived(ws.matchingMode);
  function gridItemFilter(): ItemFilterQuery {
    return ws.itemQuery((p) => withoutOpenVocab(p) && p !== 'max_rank');
  }
  function matchingQuery(): ItemFilterQuery {
    return ws.itemQuery();
  }
  function matchingBody(): ItemFilter {
    return ws.itemBody();
  }
  const gridFilterKey = $derived(JSON.stringify(gridItemFilter()));
  // Mirrors the cluster list's sort and card filter into the URL, beside the
  // item filter's own keys.
  function syncClusterListUrl(sort: ClusterSortId | null, f: ClusterListFilter): void {
    ws.patchUrl((params) => clusterFilterToUrl(params, sort, f));
  }
  function enterMatchingMode(): void {
    if (modes.searchModeActive) modes.exitSearchMode();
    if (modes.ignoredModeActive) modes.exitIgnoredMode();
    if (modes.itemTextModeActive) modes.exitItemTextMode();
    ws.setMatchingMode(true);
  }
  function exitMatchingMode(): void {
    ws.setMatchingMode(false);
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
      'cluster',
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
  />

  {#if !isSlotFilter && !matchingModeActive && !modes.ignoredModeActive && !modes.searchModeActive && !modes.itemTextModeActive}
    <ClusterListFilterBar {list} />
  {/if}

  <ClustersGrid
    {list}
    {modes}
    {matchingModeActive}
    {matchingQuery}
    {matchingBody}
    {exitMatchingMode}
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
