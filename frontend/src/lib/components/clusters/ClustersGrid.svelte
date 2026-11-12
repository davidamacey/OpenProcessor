<script lang="ts">
  import EmptyState from '$components/ui/EmptyState.svelte';
  import Icon from '$components/ui/Icon.svelte';
  import EmptyNextSteps from '$components/workspace/EmptyNextSteps.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import ClusterBadge from '$lib/components/ClusterBadge.svelte';
  import ClusterRunPanel from '$lib/components/clusters/ClusterRunPanel.svelte';
  import ClusterGridCard from '$lib/components/clusters/ClusterGridCard.svelte';
  import CropResultGrid from '$lib/components/CropResultGrid.svelte';
  import MatchingItemsView from '$lib/components/itemFilter/MatchingItemsView.svelte';
  import SlotGallery from '$lib/components/slots/SlotGallery.svelte';
  import UnembeddedBanner from '$components/embedding/UnembeddedBanner.svelte';
  import { getThumbUrl } from '$lib/api_cropRead';
  import { similarAction } from '$lib/clusters/similarAction';
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import type { SlotGalleryController } from '$lib/slots/slotGalleryController.svelte';
  import type { Crop } from '$lib/types';
  import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';

  // The /clusters content area: one of the mode swaps (search, ignored,
  // item text, matching items, slot gallery) or the card
  // grid with its infinite-scroll sentinel.
  let {
    list,
    modes,
    matchingModeActive,
    matchingQuery,
    matchingBody,
    exitMatchingMode,
    slotGallery,
    ondetail,
  }: {
    list: ClusterListController;
    modes: ClusterSearchModes;
    matchingModeActive: boolean;
    matchingQuery: () => ItemFilterQuery;
    matchingBody: () => ItemFilter;
    exitMatchingMode: () => void;
    slotGallery: SlotGalleryController | null;
    ondetail: (c: Crop) => void;
  } = $props();

  const onsimilar = $derived(similarAction());
</script>

<!-- Grid -->
<div class="flex-1 overflow-auto p-4">
  {#if modes.searchModeActive}
    <!-- Global dataset-wide search results — a mode swap over the card
         grid, not a new page. Infinite scroll is not offered: {API_PREFIX}/search/text isn't
         paginated the way the cluster grid's sentinel expects. -->
    <div class="mb-3 flex flex-wrap items-center gap-2 text-meta">
      <span class="flex items-center gap-2 text-fg-2" data-testid="search-summary">
        <strong class="text-fg-1">{modes.searchTotal.toLocaleString()}</strong>
        {#if modes.similarTo}
          result{modes.searchTotal === 1 ? '' : 's'} like
          <img
            src={getThumbUrl(modes.similarTo, 64)}
            alt="The source crop"
            class="h-8 w-8 rounded-sm border border-line-strong object-cover"
          />
          <span class="font-medium text-accent-text">{modes.similarTo}</span>
        {:else if modes.imageQueryName}
          result{modes.searchTotal === 1 ? '' : 's'} like the picture
          <span class="font-medium text-accent-text">{modes.imageQueryName}</span>
        {:else}
          result{modes.searchTotal === 1 ? '' : 's'} for
          <span class="font-medium text-accent-text">"{modes.searchQuery}"</span>
          across the dataset
        {/if}
      </span>
      {#if modes.similarTo || modes.imageQueryName}
        <UnembeddedBanner count={modes.searchUnembedded} context="search" />
      {/if}
      <span class="grow"></span>
      <button
        type="button"
        class="btn-sm border border-line-strong bg-surface-1 text-fg-2 hover:bg-surface-2"
        onclick={modes.exitSearchMode}
      >
        <Icon name="arrow-left" size={14} class="mr-1 -mt-0.5 inline-block" />Back to
        clusters
      </button>
    </div>
    {#if modes.similarError}
      <p class="text-body text-warn-text" role="status" data-testid="similar-error">
        {modes.similarError}
      </p>
    {:else if modes.searchResults.length === 0}
      <p class="text-body text-fg-3">
        {modes.similarTo || modes.imageQueryName
          ? 'No similar crops in this scope.'
          : 'No crops matched that search.'}
      </p>
    {:else}
      <CropResultGrid
        items={modes.searchResults}
        sel={modes.searchSel}
        scoreOf={(c) => modes.searchScores.get(c.id) ?? null}
        {ondetail}
        {onsimilar}
      >
        {#snippet cornerBadge(crop)}
          {#if modes.matchOf(crop.id)}
            {@const match = modes.matchOf(crop.id)}
            <span
              class="rounded-sm border border-accent/40 bg-accent/10 px-1 py-0.5 font-mono text-caption text-accent-text"
              title={match?.stale
                ? 'Matched region; its vector predates a move of the box'
                : 'Matched region'}
              data-testid="matched-box"
            >
              region {match?.boxId}{match?.stale ? ' (stale)' : ''}
            </span>
          {/if}
          {#if crop.cluster?.uid}
            <ClusterBadge
              clusterUid={crop.cluster.uid}
              displayNo={crop.cluster.display_no ?? null}
              displayName={modes.clusterMetaMap.get(crop.cluster.uid)?.display_name ??
                null}
              displayNameKind={modes.clusterMetaMap.get(crop.cluster.uid)
                ?.display_name_kind ?? 'none'}
            />
          {/if}
        {/snippet}
      </CropResultGrid>
    {/if}
  {:else if modes.ignoredModeActive}
    <!-- Ignored bucket (G7): excluded crops, restorable via
         batch_unexclude. Same mode-swap pattern as search above. -->
    <div class="mb-3 flex flex-wrap items-center gap-2 text-meta">
      <span class="text-fg-2">
        <strong class="text-fg-1">{modes.ignoredTotal.toLocaleString()}</strong>
        ignored crop{modes.ignoredTotal === 1 ? '' : 's'}
      </span>
      <span class="grow"></span>
      <button
        type="button"
        class="btn-sm border border-success/50 bg-success/10 text-success-text hover:bg-success/20"
        onclick={() => void modes.restoreIgnored()}
        disabled={modes.ignoredSel.size === 0}
      >
        Restore selected ({modes.ignoredSel.size})
      </button>
      <button
        type="button"
        class="btn-sm border border-line-strong bg-surface-1 text-fg-2 hover:bg-surface-2"
        onclick={modes.exitIgnoredMode}
      >
        <Icon name="arrow-left" size={14} class="mr-1 -mt-0.5 inline-block" />Back to
        clusters
      </button>
    </div>
    {#if modes.ignoredLoading && modes.ignoredItems.length === 0}
      <p class="text-body text-fg-3">Loading...</p>
    {:else if modes.ignoredItems.length === 0}
      <p class="text-body text-fg-3">Nothing ignored.</p>
    {:else}
      <CropResultGrid
        items={modes.ignoredItems}
        sel={modes.ignoredSel}
        {ondetail}
        {onsimilar}
      >
        {#snippet cornerBadge(crop)}
          {#if crop.excluded_reason}
            <span
              class="rounded border border-warn/50 bg-warn/15 px-1 py-0.5 text-caption text-warn-text"
            >
              {crop.excluded_reason}
            </span>
          {/if}
        {/snippet}
      </CropResultGrid>
    {/if}
  {:else if modes.itemTextModeActive}
    <!-- Item-text search results (G8). -->
    <div class="mb-3 flex flex-wrap items-center gap-2 text-meta">
      <span class="text-fg-2">
        <strong class="text-fg-1">{modes.itemTextTotal.toLocaleString()}</strong>
        result{modes.itemTextTotal === 1 ? '' : 's'} for
        <span class="font-medium text-accent-text">"{modes.itemTextQuery}"</span>
      </span>
      <span class="grow"></span>
      <button
        type="button"
        class="btn-sm border border-line-strong bg-surface-1 text-fg-2 hover:bg-surface-2"
        onclick={modes.exitItemTextMode}
      >
        <Icon name="arrow-left" size={14} class="mr-1 -mt-0.5 inline-block" />Back to
        clusters
      </button>
    </div>
    {#if modes.itemTextLoading && modes.itemTextItems.length === 0}
      <p class="text-body text-fg-3">Loading...</p>
    {:else if modes.itemTextError}
      <!-- m18: the 400 detail is already shown next to the input in
           ClustersToolbar — showing the empty-state copy here too read as
           two contradictory messages ("here's why it failed" AND
           "0 results… no crops matched"). Show only the error. -->
    {:else if modes.itemTextItems.length === 0}
      <p class="text-body text-fg-3">No crops matched that text.</p>
    {:else}
      <CropResultGrid
        items={modes.itemTextItems}
        sel={modes.itemTextSel}
        {ondetail}
        {onsimilar}
      />
    {/if}
  {:else if matchingModeActive}
    <MatchingItemsView
      query={matchingQuery}
      body={matchingBody}
      onexit={exitMatchingMode}
      {ondetail}
    />
  {:else if slotGallery}
    <!-- Slot gallery: SlotGallery.svelte, backed by
         slotGalleryController.svelte.ts. -->
    <SlotGallery gallery={slotGallery} />
  {:else if list.clusterPager.loading && list.gridItems.length === 0}
    <p class="text-body text-fg-3" role="status">Loading clusters…</p>
  {:else if list.clusterPager.error}
    <div class="text-body text-danger-text" role="alert" data-testid="clusters-error">
      <p>Could not load clusters: {list.clusterPager.error}</p>
      <button
        type="button"
        class="btn-sm mt-2 border border-line-strong bg-surface-1 text-fg-1 hover:bg-surface-2"
        onclick={() => void list.loadFirst()}
      >
        Retry
      </button>
    </div>
  {:else if list.gridItems.length === 0}
    {#if list.filterActive}
      <div class="text-body text-fg-2" data-testid="clusters-filtered-empty">
        <p>No clusters match these filters.</p>
        <button
          type="button"
          class="btn-sm mt-2 border border-line-strong bg-surface-1 text-fg-1 hover:bg-surface-2"
          onclick={() => list.clearFilter()}
        >
          Clear filters
        </button>
      </div>
    {:else}
      <div data-testid="clusters-empty">
        <EmptyState
          icon="layers-2"
          title="No clusters yet"
          description={list.emptyMessage ?? 'No clusters to show.'}
        >
          {#snippet action()}
            <!-- One way forward: the run panel when a run is what is missing, else the served loop. -->
            {#if list.emptyReason === 'no_partition' || list.emptyReason === 'pending_activation'}
              <ClusterRunPanel onserving={() => void list.loadFirst()} />
            {:else}
              <EmptyNextSteps />
            {/if}
          {/snippet}
        </EmptyState>
      </div>
    {/if}
  {:else}
    {#if list.truncated}
      <p
        class="mb-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-meta text-warn-text"
        role="status"
        data-testid="clusters-truncated"
      >
        The server stopped at {list.total.toLocaleString()} clusters, so this list may be incomplete.
        Narrow the filters to see the rest.
      </p>
    {/if}
    <ul class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {#each list.gridItems as c (c.isSlotCard ? `slot-${c.uid}` : c.uid)}
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
