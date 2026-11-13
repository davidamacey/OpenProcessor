<script lang="ts">
  /**
   * Detections: every item the filter, queue and scope match, as a flat grid
   * of crops. The list, its total and what it hides are the server's.
   */
  import { tick, untrack } from 'svelte';
  import { locateCrop } from '$lib/api_cropRead';
  import CropResultGrid from '$components/CropResultGrid.svelte';
  import HiddenItemsNote from '$components/itemFilter/HiddenItemsNote.svelte';
  import ViewMoreMenu from '$components/workspace/ViewMoreMenu.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import EmptyNextSteps from '$components/workspace/EmptyNextSteps.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import {
    createMatchingItems,
    MATCHING_PAGE_SIZE,
  } from '$lib/itemFilter/matchingItems.svelte';
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

  $effect(() =>
    ws.registerPage({
      kind: 'items',
      ids: () => pager.items.map((c) => c.id),
      total: () => pager.total,
      apply: (ids) => sel.selectAll(ids),
    }),
  );

  $effect(() => {
    if (!pager.loading) ws.matching = { total: pager.total, hidden: items.hidden };
    return () => {
      ws.matching = null;
    };
  });

  // Coming back from Review lands on the card Review stopped on: one locate request finds it, and
  // the grid loads from there when it sits deeper than the first page.
  let focusRestored = false;
  async function load(): Promise<void> {
    const focus = ws.url.focus?.kind === 'item' ? ws.url.focus.id : null;
    const restore = focus !== null && !focusRestored;
    focusRestored = true;
    if (restore) {
      try {
        const at = await locateCrop(focus, ws.itemQuery());
        if (at.in_view && at.index !== null && at.index >= MATCHING_PAGE_SIZE) {
          await pager.loadAt(at.cursor_before, at.index);
        } else {
          await pager.loadFirst();
        }
      } catch {
        await pager.loadFirst();
      }
      await tick();
      const card = document.querySelector<HTMLElement>(
        `[data-crop-id="${CSS.escape(focus)}"]`,
      );
      card?.scrollIntoView({ block: 'center' });
      card?.focus();
      return;
    }
    await pager.loadFirst();
  }

  const queryKey = $derived(JSON.stringify(ws.itemQuery()));
  $effect(() => {
    void queryKey;
    untrack(() => void load());
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
  <!-- The controls sit above the scroller: a menu that opens from them is not clipped by it. -->
  <div
    class="flex shrink-0 flex-wrap items-center gap-3 px-4 pt-4 pb-3 text-meta text-fg-2"
  >
    <p data-testid="detections-header">
      <strong class="text-fg-1" data-testid="detections-total"
        >{pager.total.toLocaleString()}</strong
      >
      detection{pager.total === 1 ? '' : 's'}
      <HiddenItemsNote hidden={items.hidden} />
    </p>
    {#if ws.url.scope?.kind === 'image'}
      <span class="chip chip-accent" data-testid="detections-image-scope"
        >One image's detections</span
      >
    {/if}
    <span class="grow"></span>
    <button
      type="button"
      class="btn btn-primary"
      data-testid="detections-review"
      title="Review these items one by one"
      onclick={() => ws.setView('review')}
    >
      Review these
    </button>
    <ViewMoreMenu {ws} />
  </div>
  <div class="min-h-0 flex-1 overflow-auto px-4 pb-4">
    <ViewState
      {status}
      message={pager.error ?? 'No detections match this filter.'}
      onretry={() => void pager.loadFirst()}
    >
      <CropResultGrid
        items={pager.items}
        {sel}
        ondetail={(c) => ws.setFocus({ kind: 'item', id: c.id })}
        onfocuscrop={(c) => ws.setFocus({ kind: 'item', id: c.id })}
      />
      <div
        use:infiniteScroll={{
          onload: () => pager.loadMore(),
          disabled: !pager.hasMore || pager.loading || pager.loadingMore,
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
      <p class="mt-2 text-meta text-fg-3 tabular-nums" data-testid="detections-footer">
        {pager.items.length.toLocaleString()} of {pager.total.toLocaleString()}
      </p>
    </ViewState>
    {#if status === 'empty'}
      <EmptyNextSteps />
    {/if}
  </div>
</div>
