<script lang="ts">
  /**
   * Detections: every item the filter, queue and scope match, as a flat grid
   * of crops. The list, its total and what it hides are the server's.
   */
  import { untrack } from 'svelte';
  import CropResultGrid from '$components/CropResultGrid.svelte';
  import HiddenItemsNote from '$components/itemFilter/HiddenItemsNote.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import EmptyNextSteps from '$components/workspace/EmptyNextSteps.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { createMatchingItems } from '$lib/itemFilter/matchingItems.svelte';
  import { createSelection } from '$lib/selection.svelte';
  import { NO_SELECTION } from '$lib/workspace/selection';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const items = createMatchingItems(() => ws.itemQuery());
  const pager = items.pager;
  const sel = createSelection({ plainClick: 'replace' });

  // The selection survives a view switch: seed the grid's own from it, then mirror it back.
  // `ws` is fixed for the life of the view, so reading it once is right.
  // svelte-ignore state_referenced_locally
  if (ws.selection.kind === 'items') sel.ids = new Set(ws.selection.cropIds);
  $effect(() => {
    const ids = [...sel.ids];
    ws.select(ids.length > 0 ? { kind: 'items', cropIds: ids } : NO_SELECTION);
  });

  $effect(() => {
    if (!pager.loading) ws.matching = { total: pager.total, hidden: items.hidden };
    return () => {
      ws.matching = null;
    };
  });

  const queryKey = $derived(JSON.stringify(ws.itemQuery()));
  $effect(() => {
    void queryKey;
    untrack(() => void pager.loadFirst());
  });

  const status = $derived(
    pager.error
      ? 'error'
      : pager.loading && pager.items.length === 0
        ? 'loading'
        : pager.items.length === 0
          ? 'empty'
          : 'ready',
  );
</script>

<div class="flex min-h-0 flex-1 flex-col" data-testid="detections-view">
  <h1 class="sr-only">Detections</h1>
  <BulkBar {ws} onapplied={() => void pager.loadFirst()} />
  <div class="min-h-0 flex-1 overflow-auto p-4">
    <p class="mb-3 text-xs text-zinc-300" data-testid="detections-header">
      <strong class="text-zinc-100" data-testid="detections-total"
        >{pager.total.toLocaleString()}</strong
      >
      detection{pager.total === 1 ? '' : 's'}
      <HiddenItemsNote hidden={items.hidden} />
    </p>
    <ViewState
      {status}
      message={pager.error ?? 'No detections match this filter.'}
      onretry={() => void pager.loadFirst()}
    >
      <CropResultGrid
        items={pager.items}
        {sel}
        ondetail={(c) => ws.setFocus({ kind: 'item', id: c.id })}
      />
      <div
        use:infiniteScroll={{
          onload: () => pager.loadMore(),
          disabled: !pager.hasMore || pager.loading || pager.loadingMore,
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
      <p class="mt-2 font-mono text-xs text-zinc-500" data-testid="detections-footer">
        {pager.items.length} / {pager.total}
      </p>
    </ViewState>
    {#if status === 'empty'}
      <EmptyNextSteps />
    {/if}
  </div>
</div>
