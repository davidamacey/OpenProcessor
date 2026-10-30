<script lang="ts">
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import ClusterBadge from '$lib/components/ClusterBadge.svelte';
  import ClusterGridCard from '$lib/components/clusters/ClusterGridCard.svelte';
  import CropResultGrid from '$lib/components/CropResultGrid.svelte';
  import EmbeddingPlot from '$lib/components/EmbeddingPlot.svelte';
  import MatchingItemsView from '$lib/components/itemFilter/MatchingItemsView.svelte';
  import SlotGallery from '$lib/components/slots/SlotGallery.svelte';
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import type { SlotGalleryController } from '$lib/slots/slotGalleryController.svelte';
  import type { Crop } from '$lib/types';
  import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';

  // The /clusters content area: one of the mode swaps (search, ignored,
  // item text, matching items, embedding plot, slot gallery) or the card
  // grid with its infinite-scroll sentinel.
  let {
    list,
    modes,
    classFilter,
    matchingModeActive,
    matchingQuery,
    matchingBody,
    exitMatchingMode,
    showEmbeddingViz,
    embeddingVizBannerRequired,
    embeddingVizCoverageTotal,
    slotGallery,
    ondetail,
  }: {
    list: ClusterListController;
    modes: ClusterSearchModes;
    classFilter: number | null;
    matchingModeActive: boolean;
    matchingQuery: () => ItemFilterQuery;
    matchingBody: () => ItemFilter;
    exitMatchingMode: () => void;
    showEmbeddingViz: boolean;
    embeddingVizBannerRequired: boolean;
    embeddingVizCoverageTotal: number | null;
    slotGallery: SlotGalleryController | null;
    ondetail: (c: Crop) => void;
  } = $props();
</script>

<!-- Grid -->
<div class="flex-1 overflow-auto p-4">
  {#if modes.searchModeActive}
    <!-- Global dataset-wide search results — a mode swap over the card
         grid, not a new page (mirrors showEmbeddingViz's own swap
         above). Infinite scroll is not offered: {API_PREFIX}/search/text isn't
         paginated the way the cluster grid's sentinel expects. -->
    <div class="mb-3 flex items-center gap-2 text-xs">
      <span class="text-zinc-300">
        <strong class="text-zinc-100">{modes.searchTotal.toLocaleString()}</strong>
        result{modes.searchTotal === 1 ? '' : 's'} for
        <span class="font-medium text-blue-200">"{modes.searchQuery}"</span>
        across the dataset
      </span>
      <span class="grow"></span>
      <button
        type="button"
        class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
        onclick={modes.exitSearchMode}
      >
        ← Back to clusters
      </button>
    </div>
    {#if modes.searchResults.length === 0}
      <p class="text-sm text-zinc-500">No crops matched that search.</p>
    {:else}
      <CropResultGrid
        items={modes.searchResults}
        sel={modes.searchSel}
        scoreOf={(c) => modes.searchScores.get(c.id) ?? null}
        {ondetail}
      >
        {#snippet cornerBadge(crop)}
          {#if crop.cluster_id != null}
            <ClusterBadge
              clusterId={crop.cluster_id}
              dominantClassName={modes.clusterMetaMap.get(crop.cluster_id)
                ?.dominant_class_name ?? null}
            />
          {/if}
        {/snippet}
      </CropResultGrid>
    {/if}
  {:else if modes.ignoredModeActive}
    <!-- Ignored bucket (G7): excluded crops, restorable via
         batch_unexclude. Same mode-swap pattern as search above. -->
    <div class="mb-3 flex items-center gap-2 text-xs">
      <span class="text-zinc-300">
        <strong class="text-zinc-100">{modes.ignoredTotal.toLocaleString()}</strong>
        ignored crop{modes.ignoredTotal === 1 ? '' : 's'}
      </span>
      <span class="grow"></span>
      <button
        type="button"
        class="btn-sm border border-green-500/50 bg-green-500/10 text-green-200 hover:bg-green-500/20"
        onclick={() => void modes.restoreIgnored()}
        disabled={modes.ignoredSel.size === 0}
      >
        Restore selected ({modes.ignoredSel.size})
      </button>
      <button
        type="button"
        class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
        onclick={modes.exitIgnoredMode}
      >
        ← Back to clusters
      </button>
    </div>
    {#if modes.ignoredLoading && modes.ignoredItems.length === 0}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if modes.ignoredItems.length === 0}
      <p class="text-sm text-zinc-500">Nothing ignored.</p>
    {:else}
      <CropResultGrid items={modes.ignoredItems} sel={modes.ignoredSel} {ondetail}>
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
  {:else if modes.itemTextModeActive}
    <!-- Item-text search results (G8). -->
    <div class="mb-3 flex items-center gap-2 text-xs">
      <span class="text-zinc-300">
        <strong class="text-zinc-100">{modes.itemTextTotal.toLocaleString()}</strong>
        result{modes.itemTextTotal === 1 ? '' : 's'} for
        <span class="font-medium text-blue-200">"{modes.itemTextQuery}"</span>
      </span>
      <span class="grow"></span>
      <button
        type="button"
        class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
        onclick={modes.exitItemTextMode}
      >
        ← Back to clusters
      </button>
    </div>
    {#if modes.itemTextLoading && modes.itemTextItems.length === 0}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if modes.itemTextError}
      <!-- m18: the 400 detail is already shown next to the input in
           ClustersToolbar — showing the empty-state copy here too read as
           two contradictory messages ("here's why it failed" AND
           "0 results… no crops matched"). Show only the error. -->
    {:else if modes.itemTextItems.length === 0}
      <p class="text-sm text-zinc-500">No crops matched that text.</p>
    {:else}
      <CropResultGrid items={modes.itemTextItems} sel={modes.itemTextSel} {ondetail} />
    {/if}
  {:else if matchingModeActive}
    <MatchingItemsView
      query={matchingQuery}
      body={matchingBody}
      onexit={exitMatchingMode}
      {ondetail}
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
  {:else if list.clusterPager.loading && list.gridItems.length === 0}
    <p class="text-sm text-zinc-500" role="status">Loading clusters…</p>
  {:else if list.clusterPager.error}
    <div class="text-sm text-red-300" role="alert" data-testid="clusters-error">
      <p>Could not load clusters: {list.clusterPager.error}</p>
      <button
        type="button"
        class="btn-sm mt-2 border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
        onclick={() => void list.loadFirst()}
      >
        Retry
      </button>
    </div>
  {:else if list.gridItems.length === 0}
    {#if list.filterActive}
      <div class="text-sm text-zinc-400" data-testid="clusters-filtered-empty">
        <p>No clusters match these filters.</p>
        <button
          type="button"
          class="btn-sm mt-2 border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
          onclick={() => list.clearFilter()}
        >
          Clear filters
        </button>
      </div>
    {:else}
      <p class="text-sm text-zinc-500">
        No clusters yet — ingest some images and run the auto-label pipeline.
      </p>
    {/if}
  {:else}
    {#if list.truncated}
      <p
        class="mb-3 rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
        role="status"
        data-testid="clusters-truncated"
      >
        The server stopped at {list.total.toLocaleString()} clusters, so this list may be incomplete.
        Narrow the filters to see the rest.
      </p>
    {/if}
    <ul class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {#each list.gridItems as c (c.isSlotCard ? `slot-${c.id}` : c.id)}
        <ClusterGridCard {c} flagVocabulary={list.flagVocabulary} />
      {/each}
    </ul>
    <!-- Sentinel MUST live inside the scroll container so the IntersectionObserver
         can root itself on the right element. Outside the overflow-auto div the
         observer falls back to viewport and either never fires or fires forever. -->
    <div
      use:infiniteScroll={{
        onload: list.loadMore,
        disabled: !list.hasMoreCards || list.repsLoading,
      }}
      class="mt-4 h-1"
      aria-hidden="true"
    ></div>
  {/if}
</div>
