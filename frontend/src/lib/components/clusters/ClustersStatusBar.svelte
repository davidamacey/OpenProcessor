<script lang="ts">
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import type { SlotInventory } from '$lib/clusters/slotInventory.svelte';
  import type { SlotGalleryController } from '$lib/slots/slotGalleryController.svelte';

  let {
    list,
    modes,
    slotInventory,
    classFilter,
    matchingModeActive,
    slotGallery,
  }: {
    list: ClusterListController;
    modes: ClusterSearchModes;
    slotInventory: SlotInventory;
    classFilter: number | null;
    matchingModeActive: boolean;
    slotGallery: SlotGalleryController | null;
  } = $props();
</script>

<!-- Status bar (no scroll sentinel here — see ClustersGrid).
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
  {:else if modes.ignoredModeActive}
    <span class="font-mono text-xs text-zinc-500" data-testid="clusters-footer-count">
      {modes.ignoredItems.length} / {modes.ignoredTotal} ignored crops
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {modes.ignoredLoading ? 'loading…' : 'ignored bucket'}
    </span>
  {:else if slotGallery}
    <span class="font-mono text-xs text-zinc-500">
      {slotGallery.pager.items.length} / {slotGallery.pager.total}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if slotGallery.pager.loadingMore}loading more…{:else if slotGallery.pager.hasMore}scroll
        for more{:else}all loaded{/if}
    </span>
  {:else if list.clusterPager.error}
    <!-- A failed load has no honest count; the grid shows the error. -->
  {:else}
    <span class="font-mono text-xs text-zinc-500" data-testid="clusters-footer-count">
      {list.gridItems.length} / {list.clusterPager.total +
        (classFilter == null ? slotInventory.slotInventoryCards.length : 0)}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if list.clusterPager.loadingMore}loading more…{:else if list.clusterPager.hasMore}scroll
        for more{:else}all loaded{/if}
    </span>
  {/if}
</div>
