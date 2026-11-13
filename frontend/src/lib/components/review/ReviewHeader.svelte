<!--
  The Review view's header: the queue it reads, where the focus sits, what the queue hides, the
  live "N new" pill, the Dismissed panel toggle and the key hints for the active kind of queue.
-->
<script lang="ts">
  import Menu, { type MenuItem } from '$components/ui/Menu.svelte';
  import AcceptConfidentDialog from '$components/review/AcceptConfidentDialog.svelte';
  import HiddenItemsNote from '$lib/components/itemFilter/HiddenItemsNote.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SortSelect from '$lib/components/workspace/SortSelect.svelte';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
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
  }

  let { ws, tabs, loader, liveNewCount, refreshFromLive, toggleDismissedPanel }: Props =
    $props();

  const queue = $derived(loader.queue);
  let acceptOpen = $state(false);
  const moreItems: MenuItem[] = [
    {
      id: 'dismissed',
      label: 'Dismissed items…',
      testid: 'review-dismissed-open',
      onselect: () => toggleDismissedPanel(),
    },
  ];

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
  <h1 class="sr-only" data-testid="review-queue-label">{tabs.activeQueueLabel}</h1>
  <span class="text-body font-medium text-fg-1" aria-hidden="true">
    {tabs.activeQueueLabel}
  </span>
  {#if regionProfileStore.unknown}
    <span class="text-caption text-fg-3" data-testid="region-profile-loading"
      >loading region profile…</span
    >
  {/if}
  <span
    data-testid="queue-counter"
    class="text-meta text-fg-2 tabular-nums"
    title="{queue.items.length.toLocaleString()} loaded of {queue.total.toLocaleString()}"
  >
    {queue.items.length > 0
      ? `#${loader.currentPosition.toLocaleString()} of ${queue.total.toLocaleString()}`
      : '—'}
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
  {#if tabs.queue === 'check_machine'}
    <button
      type="button"
      class="btn"
      data-testid="accept-confident-open"
      title="Validate the machine labels above a confidence bar in one undoable operation"
      onclick={() => (acceptOpen = true)}
    >
      Accept confident labels…
    </button>
  {/if}
  <Menu
    label="More"
    items={moreItems}
    align="end"
    triggerClass="btn"
    triggerTestid="review-more"
  />
  <ShortcutsButton />
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

{#if acceptOpen}
  <AcceptConfidentDialog
    filter={() => ws.itemBody()}
    onclose={() => (acceptOpen = false)}
    onapplied={() => void loader.loadFirst()}
  />
{/if}
