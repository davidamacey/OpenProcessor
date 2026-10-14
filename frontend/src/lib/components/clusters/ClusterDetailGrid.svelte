<script lang="ts">
  import { dndzone } from 'svelte-dnd-action';
  import CropCard from '$components/CropCard.svelte';
  import CutLine from '$components/CutLine.svelte';
  import ScoreChip from '$components/ScoreChip.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { firstLoadHint, listedText, statusBarText } from '$lib/clusters/firstLoad';
  import type { Crop } from '$lib/types';
  import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
  import type { ClusterDetailActions } from '$lib/clusters/clusterDetailActions.svelte';
  import type { ClusterDetailLive } from '$lib/clusters/clusterDetailLive.svelte';

  // The crop grid (one dndzone per sub-cluster group), its live-refresh pill
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
        class="sticky top-2 z-10 mx-auto block animate-pulse rounded-full border border-blue-500/60 bg-blue-500/20 px-3 py-1 text-xs text-blue-100 shadow-lg backdrop-blur hover:bg-blue-500/30"
        onclick={live.refreshFromLive}
        title="Scroll to top and reload with the latest crops"
      >
        {live.liveNewCount} new crop{live.liveNewCount === 1 ? '' : 's'} · click to refresh
      </button>
    {/if}
    {#if view.cropPager.loading && view.cropPager.items.length === 0}
      <p class="text-sm text-zinc-500" data-testid="cluster-first-load">
        Loading...
        {#if view.slowFirstLoad}<span data-testid="cluster-slow-load-hint"
            >{firstLoadHint(view.cluster?.size ?? null)}</span
          >{/if}
      </p>
    {:else if view.cropPager.error}
      <p class="text-sm text-red-300">API unavailable: {view.cropPager.error}</p>
    {:else if view.filteredCrops.length === 0}
      <p class="text-sm text-zinc-500">No crops in this cluster yet.</p>
    {:else}
      <!-- One dndzone per sub-cluster group. Headers sit BETWEEN zones
           (not inside any) so each zone's children map 1:1 to its
           items — keeping drag-and-drop intact. All zones share
           type='crop-card' so a card drags out to the ClassSidebar (or
           across groups) exactly as before. -->
      {#each view.gridGroups as group (group.key)}
        {#if view.groupBySubcluster}
          <div
            class="mt-3 flex items-center gap-2 pt-1 text-xs font-medium text-zinc-300 first:mt-0"
          >
            <span class="rounded bg-zinc-800 px-2 py-0.5 text-zinc-100"
              >{group.label}</span
            >
            <span class="text-zinc-500">
              {group.items.length} crop{group.items.length === 1 ? '' : 's'}
            </span>
            <span class="h-px flex-1 bg-zinc-800"></span>
          </div>
        {/if}
        <div
          class="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-3 {view.groupBySubcluster
            ? 'mt-2'
            : ''}"
          use:dndzone={{
            items: group.items,
            type: 'crop-card',
            flipDurationMs: 150,
            dropTargetStyle: { outline: '2px dashed rgb(59 130 246 / 0.6)' },
            dragDisabled: false,
            // Reject foreign items. After AHC refine the grid is split
            // into one zone per sub-cluster; without this a sibling
            // sub-cluster zone steals a drop meant for the ClassSidebar
            // (it's geometrically larger than the narrow sidebar rows),
            // so the card snaps back into the AHC grouping instead of
            // being labeled. Cross-group grid drops have no meaning
            // anyway (onGroupFinalize rebuilds them), so disabling
            // foreign drops only removes the drop-stealing — drag-out to
            // the sidebar is unaffected.
            dropFromOthersDisabled: true,
          }}
          onconsider={(e) => actions.onGroupConsider(group.key, e)}
          onfinalize={(e) => actions.onGroupFinalize(group.key, e)}
        >
          {#each group.items as crop, i (crop.id)}
            {#if !view.groupBySubcluster && view.cutLineVisible && i === view.cutLineIndex}
              <CutLine />
            {/if}
            <div class="relative">
              <CropCard
                {crop}
                selected={view.sel.has(crop.id)}
                onclick={(c, e) => view.clickSelect(c.id, e)}
                onacceptVlm={(c) => void actions.controller.acceptVlmForCrop(c)}
                onrejectVlm={(c) => void actions.controller.rejectVlmForCrop(c)}
                {ondetail}
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
       left ClassFolderSidebar. The 'Move to cluster #N' affordance is
       still available via M-key (typing a destination cluster id). -->
</div>

<!-- Status bar (sentinel lives inside the scroll container, see above) -->
<div
  class="flex items-center justify-between gap-2 border-t border-zinc-800 px-4 py-2 text-sm"
  data-testid="cluster-status-bar"
>
  <span class="font-mono text-xs text-zinc-500">
    <span
      title="Loaded / reviewable members matching the current filters. Test-holdout and ignored crops are never listed, so this can be lower than the header's cluster size."
      >{listedText(view.pagerState)}</span
    >
    {#if view.sel.size > 0}<span class="ml-2 text-blue-300"
        >· {view.sel.size} selected</span
      >{/if}
  </span>
  <span class="font-mono text-xs text-zinc-400">
    {statusBarText(view.pagerState)}
  </span>
</div>
