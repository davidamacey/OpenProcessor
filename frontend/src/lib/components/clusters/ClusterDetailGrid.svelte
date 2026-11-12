<script lang="ts">
  import { dndzone } from 'svelte-dnd-action';
  import CropCard from '$components/CropCard.svelte';
  import ScoreChip from '$components/ScoreChip.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { firstLoadHint, listedText, statusBarText } from '$lib/clusters/firstLoad';
  import { similarAction } from '$lib/clusters/similarAction';
  import type { Crop } from '$lib/types';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';
  import type { ClusterDetailLive } from '$lib/clusters/clusterDetailLive.svelte';

  // The crop grid (one dndzone), its live-refresh pill
  // and infinite-scroll sentinel, and the status bar.
  let {
    view,
    actions,
    live,
    ondetail,
  }: {
    view: ClusterDetailController;
    actions: ClusterDetailActions;
    live: ClusterDetailLive;
    ondetail: (c: Crop) => void;
  } = $props();

  const onsimilar = $derived(similarAction());
</script>

<!-- Grid container. The class-list left sidebar is the layout-level
     ClassSidebar — it auto-receives drops via dropOnClassStore. -->
<div class="flex min-h-0 flex-1 overflow-hidden">
  <div
    bind:this={live.scrollEl}
    onscroll={live.onScroll}
    class="relative min-w-0 flex-1 overflow-auto p-4"
  >
    {#if live.liveNewCount > 0 && live.scrolledPastFirst20}
      <button
        type="button"
        class="sticky top-2 z-10 mx-auto block animate-pulse rounded-full border border-accent/60 bg-accent/20 px-3 py-1 text-xs text-accent shadow-lg backdrop-blur hover:bg-accent/30"
        onclick={live.refreshFromLive}
        title="Scroll to top and reload with the latest crops"
      >
        {live.liveNewCount} new crop{live.liveNewCount === 1 ? '' : 's'} · click to refresh
      </button>
    {/if}
    {#if view.cropPager.loading && view.cropPager.items.length === 0}
      <p class="text-sm text-fg-3" data-testid="cluster-first-load">
        Loading...
        {#if view.slowFirstLoad}<span data-testid="cluster-slow-load-hint"
            >{firstLoadHint(view.cluster?.size ?? null)}</span
          >{/if}
      </p>
    {:else if view.cropPager.error}
      <p class="text-sm text-danger">API unavailable: {view.cropPager.error}</p>
    {:else if view.filteredCrops.length === 0}
      <p class="text-sm text-fg-3">No crops in this cluster yet.</p>
    {:else}
      <!-- One dndzone. A card drags out to the ClassSidebar exactly as
           before (type='crop-card'). -->
      {#each view.gridGroups as group (group.key)}
        <div
          class="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-3"
          use:dndzone={{
            items: group.items,
            type: 'crop-card',
            flipDurationMs: 150,
            dropTargetStyle: { outline: '2px dashed rgb(59 130 246 / 0.6)' },
            dragDisabled: false,
            // Reject foreign items so a drop meant for the ClassSidebar is not
            // stolen by the grid itself; drag-out to the sidebar is unaffected.
            dropFromOthersDisabled: true,
          }}
          onconsider={(e) => actions.onGroupConsider(group.key, e)}
          onfinalize={(e) => actions.onGroupFinalize(group.key, e)}
        >
          {#each group.items as crop (crop.id)}
            <div class="relative">
              <CropCard
                {crop}
                selected={view.sel.has(crop.id)}
                onclick={(c, e) => view.clickSelect(c.id, e)}
                onacceptVlm={(c) => void actions.controller.acceptVlmForCrop(c)}
                onrejectVlm={(c) => void actions.controller.rejectVlmForCrop(c)}
                {ondetail}
                {onsimilar}
                onreprocessed={(items) => actions.controller.adoptItems(items)}
              />
              {#if view.searchModeActive && view.searchScores.has(crop.id)}
                <div class="pointer-events-none absolute left-1 top-1 z-10">
                  <ScoreChip
                    label="match"
                    value={view.searchScores.get(crop.id) ?? 0}
                    size="sm"
                  />
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/each}
      <!-- Sentinel inside the scroll container so IntersectionObserver
           roots on the right element (the overflow-auto parent). -->
      <div
        use:infiniteScroll={{
          onload: view.loadMore,
          disabled:
            view.searchModeActive ||
            view.cropPager.loadingMore ||
            !view.cropPager.hasMore ||
            view.cropPager.loading,
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
    {/if}
  </div>

  <!-- Right move-target rail removed — class assignment moved to the
       left ClassFolderSidebar. The 'Move to cluster' affordance is
       still available via M-key (picking a destination cluster). -->
</div>

<!-- Status bar (sentinel lives inside the scroll container, see above) -->
<div
  class="flex items-center justify-between gap-2 border-t border-line px-4 py-2 text-sm"
  data-testid="cluster-status-bar"
>
  <span class="font-mono text-xs text-fg-3">
    <span
      title="Loaded / reviewable members matching the current filters. Test-holdout and ignored crops are never listed, so this can be lower than the header's cluster size."
      >{listedText(view.pagerState)}</span
    >
    {#if view.sel.size > 0}<span class="ml-2 text-accent">· {view.sel.size} selected</span
      >{/if}
  </span>
  <span class="font-mono text-xs text-fg-2">
    {statusBarText(view.pagerState)}
  </span>
</div>
