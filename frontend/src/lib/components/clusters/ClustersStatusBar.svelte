<script lang="ts">
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import type { SlotGalleryController } from '$lib/slots/slotGalleryController.svelte';

  let {
    list,
    modes,
    matchingModeActive,
    slotGallery,
  }: {
    list: ClusterListController;
    modes: ClusterSearchModes;
    matchingModeActive: boolean;
    slotGallery: SlotGalleryController | null;
  } = $props();
</script>

<!-- Status bar (no scroll sentinel here — see ClustersGrid).
     DQ-p2: this always
     read off clusterPager (the cluster-grid pager) — "102 / 102 all
     loaded" under a much larger slot-gallery grid, because the gallery
     view (isSlotFilter) renders slotGallery.pager.items,
     a completely different pager, but this footer never switched to
     match. -->
<div
  class="flex items-center justify-between gap-3 border-t border-line px-4 py-2 text-body"
>
  {#if matchingModeActive}
    <span class="text-meta text-fg-2">matching items</span>
  {:else if modes.ignoredModeActive}
    <span class="text-meta text-fg-3 tabular-nums" data-testid="clusters-footer-count">
      {modes.ignoredItems.length.toLocaleString()} of {modes.ignoredTotal.toLocaleString()}
      ignored crops
    </span>
    <span class="text-meta text-fg-2">
      {modes.ignoredLoading ? 'loading…' : 'ignored bucket'}
    </span>
  {:else if slotGallery}
    <span class="text-meta text-fg-3 tabular-nums">
      {slotGallery.pager.items.length.toLocaleString()} of {slotGallery.pager.total.toLocaleString()}
    </span>
    <span class="text-meta text-fg-2">
      {#if slotGallery.pager.loadingMore}loading more…{:else if slotGallery.pager.hasMore}scroll
        for more{:else}all loaded{/if}
    </span>
  {:else if list.clusterPager.error}
    <!-- A failed load has no honest count; the grid shows the error. -->
  {:else}
    <span class="text-meta text-fg-3 tabular-nums" data-testid="clusters-footer-count">
      {list.gridItems.length.toLocaleString()} of {list.total.toLocaleString()}
    </span>
    <span class="text-meta text-fg-2">
      {#if list.repsLoading}loading more…{:else if list.hasMoreCards}scroll for more{:else}all
        shown{/if}
    </span>
  {/if}
</div>
