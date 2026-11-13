<script lang="ts">
  /**
   * The cluster list's header: the search, one Refine button and the one Sort, the chips of the
   * refinements in force, the Region boxes line and the view's one primary (open the first
   * cluster). Everything the old two toolbars held is in the Refine popover.
   */
  import { page } from '$app/state';
  import Icon from '$components/ui/Icon.svelte';
  import ImageSearchPicker from '$components/clusters/ImageSearchPicker.svelte';
  import ClusterRefinePanel from '$components/clusters/ClusterRefinePanel.svelte';
  import RefinePopover from '$components/workspace/RefinePopover.svelte';
  import SortSelect from '$components/workspace/SortSelect.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import { asSortId } from '$lib/clusters/clusterListFilter';
  import { refinementChips } from '$lib/clusters/refinementChips';
  import type { ClusterListController } from '$lib/clusters/clusterListController.svelte';
  import type { ClusterSearchModes } from '$lib/clusters/clusterSearchModes.svelte';
  import { registeredSlots } from '$lib/annotations/registeredSlots';
  import type { ItemFilterQuery } from '$lib/types_itemFilter';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  let {
    ws,
    list,
    modes,
    classFilter,
    isSlotFilter,
    gridItemFilter,
    matchingModeActive,
    enterMatchingMode,
    exitMatchingMode,
    openFirst,
  }: {
    ws: WorkspaceController;
    list: ClusterListController;
    modes: ClusterSearchModes;
    classFilter: number | null;
    isSlotFilter: boolean;
    gridItemFilter: () => ItemFilterQuery;
    matchingModeActive: boolean;
    enterMatchingMode: () => void;
    exitMatchingMode: () => void;
    openFirst: () => void;
  } = $props();

  const searchAvailable = $derived(modes.semanticSearchAvailable && !isSlotFilter);
  const chips = $derived(
    refinementChips({
      filter: list.filter,
      flagVocabulary: list.flagVocabulary,
      setFilter: (f) => list.setFilter(f),
      subjectScope: list.subjectScope,
      setSubjectScope: (v) => (list.subjectScope = v),
      minBlurRatio: list.minBlurRatio,
      clearClarity: () => {
        list.blurSlider = 0;
        list.commitClusterBlur();
      },
    }),
  );
  const gridMode = $derived(
    !isSlotFilter &&
      !matchingModeActive &&
      !modes.ignoredModeActive &&
      !modes.searchModeActive &&
      !modes.itemTextModeActive,
  );

  // The region queue of an installed slot, as served: its label and its count under the filter.
  const regionQueue = $derived.by(() => {
    for (const slot of registeredSlots) {
      const id = slot.capabilities.queue?.endpointId;
      if (!id || !reviewTabsVocabularyStore.hasEntry(id)) continue;
      const count = reviewTabsVocabularyStore.countFor(id);
      if (count == null || count <= 0) continue;
      return {
        id,
        count,
        label: reviewTabsVocabularyStore.labelFor(
          id,
          slot.capabilities.queue?.tabLabel ?? slot.label.plural,
        ),
      };
    }
    return null;
  });

  function openRegionQueue(id: string): void {
    ws.setQueue(id);
    ws.setView('review');
  }
</script>

<div class="border-b border-line px-4 py-2.5" data-testid="cluster-list-header">
  <h1 class="sr-only">Clusters</h1>
  <div class="flex flex-wrap items-center gap-2">
    {#if classFilter != null}
      <span
        class="chip chip-accent"
        title="class id {classFilter}"
        data-testid="class-filter-chip"
      >
        class: {classesStore.classes.find((c) => c.id === classFilter)?.name ??
          `#${classFilter}`}
      </span>
    {/if}
    {#if searchAvailable}
      <!-- Re-mounted on a target switch: the box re-runs the typed query with the new target.
           No cluster scope: unscoped across the dataset is the point of this control. -->
      {#key modes.searchTarget}
        <SemanticSearchBox
          pageSize={200}
          placeholder="Search clusters and crops, e.g. blurry closeup"
          filter={{
            ...gridItemFilter(),
            ...(modes.searchTarget === 'box' ? { target: 'box' } : {}),
          }}
          initialQuery={modes.similarTo ? null : page.url.searchParams.get('q')}
          onQueryChange={(q) => (modes.searchQuery = q)}
          onResults={(res) => {
            modes.searchModeActive = true;
            modes.searchScores = new Map(
              res.items.map((it) => [it.id, it.similarity_score]),
            );
            modes.noteMatches(res.items);
            modes.searchResults = res.items;
            modes.searchTotal = res.total;
            modes.syncSearchUrl(modes.searchQuery);
            void modes.ensureClusterMeta([
              ...new Set(
                res.items
                  .map((it) => it.cluster?.uid)
                  .filter((uid): uid is string => uid != null),
              ),
            ]);
          }}
          onClear={modes.exitSearchMode}
        />
      {/key}
      <ImageSearchPicker onSearch={(file, box) => void modes.runImage({ file, box })} />
    {/if}
    <span class="grow"></span>
    {#if gridMode && list.sortOptions.length > 0}
      <SortSelect
        testid="cluster-sort"
        options={list.sortOptions}
        applied={list.appliedSort ?? ''}
        onchange={(id) => (list.sort = asSortId(id))}
      />
    {/if}
    {#if !isSlotFilter}
      <RefinePopover count={chips.length} testid="cluster-refine">
        {#snippet children(close)}
          <ClusterRefinePanel
            {list}
            {modes}
            {searchAvailable}
            {matchingModeActive}
            {enterMatchingMode}
            {exitMatchingMode}
            {close}
          />
        {/snippet}
      </RefinePopover>
    {/if}
    {#if gridMode && list.gridItems.length > 0}
      <button
        type="button"
        class="btn btn-primary"
        data-testid="open-first-cluster"
        title="Open the first cluster to label it"
        onclick={openFirst}
      >
        Label the first cluster
      </button>
    {/if}
  </div>
  {#if chips.length > 0}
    <div class="mt-2 flex flex-wrap items-center gap-1.5" data-testid="refinement-chips">
      {#each chips as chip (chip.id)}
        <button
          type="button"
          class="chip chip-accent"
          title="Remove this refinement"
          onclick={chip.clear}
        >
          {chip.label}<Icon name="x" size={12} class="ml-1 inline-block" />
        </button>
      {/each}
      <button
        type="button"
        class="btn-sm btn-ghost"
        data-testid="cluster-filters-clear"
        onclick={() => {
          list.clearFilter();
          list.subjectScope = 0;
          list.blurSlider = 0;
          list.commitClusterBlur();
        }}
      >
        Clear all
      </button>
    </div>
  {/if}
  {#if regionQueue && gridMode}
    <p class="mt-2 text-meta text-fg-2" data-testid="region-boxes-line">
      Region boxes:
      <span class="tabular-nums">{regionQueue.count.toLocaleString()}</span>
      {regionQueue.label} to confirm
      <button
        type="button"
        class="ml-1 text-accent-text underline underline-offset-2"
        onclick={() => openRegionQueue(regionQueue.id)}
      >
        Open the queue
      </button>
    </p>
  {/if}
</div>
