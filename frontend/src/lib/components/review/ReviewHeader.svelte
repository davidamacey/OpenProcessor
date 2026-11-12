<!--
  The Review view's header: the queue it reads, where the focus sits, what the queue hides, the
  live "N new" pill, the Dismissed panel toggle and the key hints for the active kind of queue.
-->
<script lang="ts">
  import HiddenItemsNote from '$lib/components/itemFilter/HiddenItemsNote.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SortSelect from '$lib/components/workspace/SortSelect.svelte';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import { kg } from '$lib/review/reviewKeys';
  import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import { rejectKeyGlyph } from '$lib/review/slotKeymap';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { regionProfileStore } from '$stores/regionProfile.svelte';

  interface Props {
    ws: WorkspaceController;
    tabs: ReviewTabsController;
    loader: ReviewQueueLoader;
    liveNewCount: number;
    refreshFromLive: () => void;
    toggleDismissedPanel: () => void;
    isMultiBoxSlot: boolean;
    editMode: boolean;
  }

  let {
    ws,
    tabs,
    loader,
    liveNewCount,
    refreshFromLive,
    toggleDismissedPanel,
    isMultiBoxSlot,
    editMode,
  }: Props = $props();

  const queue = $derived(loader.queue);
  const activeSlot = $derived(tabs.activeSlot);

  // The search runs over the Workspace query: its filter, the open queue (as `tab`; the region
  // queue is no `review_tab` filter value) and the queue's own filters.
  const searchFilter = $derived.by(() => {
    const { review_tab: _tab, ...rest } = ws.itemQuery(withoutOpenVocab) as Record<
      string,
      unknown
    >;
    return {
      ...rest,
      ...tabs.queueFilterValues,
      ...(tabs.queue ? { tab: tabs.queue } : {}),
    };
  });
  let typed = '';
  function syncSearchUrl(q: string): void {
    ws.patchUrl((params) => {
      if (q) params.set('q', q);
      else params.delete('q');
    });
  }
</script>

<div
  class="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-line px-4 py-2"
  data-testid="review-header"
>
  <h1 class="text-body font-medium text-fg-1" data-testid="review-queue-label">
    {tabs.activeQueueLabel}
  </h1>
  {#if regionProfileStore.unknown}
    <span class="text-caption text-fg-3" data-testid="region-profile-loading"
      >loading region profile…</span
    >
  {/if}
  <span data-testid="queue-counter" class="font-mono text-meta text-fg-3">
    {queue.items.length > 0 ? `#${loader.currentPosition}` : '—'} · {queue.items.length} loaded
    · {queue.total} total
  </span>
  <HiddenItemsNote hidden={loader.hidden} />
  {#if loader.sortOptions.length > 0 && !loader.searchActive}
    <SortSelect
      testid="review-sort"
      options={loader.sortOptions}
      applied={loader.sortApplied ?? ''}
      fallbackReason={loader.sortFallbackReason}
      onchange={(id) => ws.setSort(tabs.queue ? 'review' : 'detections', id)}
    />
  {/if}
  {#if loader.semanticSearchAvailable}
    <!-- Re-mounted when the queue, filter or sort changes, so the typed query runs again under it. -->
    {#key loader.sourceKey}
      <SemanticSearchBox
        pageSize={200}
        filter={searchFilter}
        initialQuery={ws.url.q}
        onQueryChange={(q) => {
          typed = q;
          if (!q) syncSearchUrl('');
        }}
        onResults={(res) => {
          loader.applySearchResults(res);
          syncSearchUrl(typed);
        }}
        onClear={() => {
          loader.clearSearch();
          syncSearchUrl('');
        }}
      />
    {/key}
  {/if}
  <span class="grow"></span>
  <span class="hidden text-caption text-fg-3 lg:inline">
    {#if isMultiBoxSlot && activeSlot}
      <kbd>{kg('review.region.confirm')}</kbd> confirm proposed ·
      <kbd>{kg('review.region.accept_box')}</kbd> accept box ·
      <kbd>{kg('review.region.reject_box')}</kbd> reject box ·
      <kbd>{kg('box_edit.next_box')}</kbd> next box
      {#if editMode}
        · <kbd>{kg('box_edit.delete_box')}</kbd> delete box · drag empty area to add
      {:else}
        · <kbd>{kg('review.region.edit_box')}</kbd> edit
      {/if}
      · <kbd>{kg('review.skip')}</kbd> skip
    {:else if activeSlot}
      <kbd>{kg('review.region.confirm')}</kbd> confirm ·
      <kbd>{rejectKeyGlyph(activeSlot)}</kbd> reject
      {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
        · <kbd>{kg('review.region.false_positive')}</kbd> false-pos
      {/if}
      · <kbd>{kg('review.skip')}</kbd> skip · <kbd>{kg('review.region.back')}</kbd> back
    {:else}
      per-class letter assigns · <kbd>{kg('review.queue.class_picker')}</kbd> search all
      classes · <kbd>{kg('review.queue.confirm')}</kbd> confirm ·
      <kbd>{kg('review.skip')}</kbd> skip · <kbd>{kg('review.queue.discard')}</kbd>
      discard · <kbd>{kg('review.undo')}</kbd> undo
    {/if}
    · <kbd>{kg('workspace.escape')}</kbd> back to the grid
  </span>
  <ShortcutsButton />
  <button
    type="button"
    class="shrink-0 rounded border border-line-strong px-2 py-1 text-caption text-fg-2 hover:border-fg-3 hover:text-fg-1"
    onclick={toggleDismissedPanel}
    title="Crops permanently dismissed from review — restore one back into the queue"
  >
    Dismissed
  </button>
  {#if liveNewCount > 0}
    <button
      type="button"
      class="shrink-0 animate-pulse rounded-full border border-accent/60 bg-accent/15 px-2.5 py-1 text-caption text-accent-text hover:bg-accent/25"
      onclick={refreshFromLive}
      title="Reload the queue with the latest crops"
    >
      {liveNewCount} new · refresh
    </button>
  {/if}
</div>
