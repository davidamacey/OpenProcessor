<script lang="ts">
  /**
   * Gallery: whole images with the boxes, counts and near-duplicate badge the server
   * serves, in the served order. Selecting images acts on their matching items.
   */
  import { untrack } from 'svelte';
  import HiddenItemsNote from '$components/itemFilter/HiddenItemsNote.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import SortSelect from '$components/workspace/SortSelect.svelte';
  import EmptyNextSteps from '$components/workspace/EmptyNextSteps.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import ImageGrid from '$components/images/ImageGrid.svelte';
  import { createImageGalleryController } from '$lib/images/imageGalleryController.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const gallery = createImageGalleryController({
    itemQuery: () => ws.itemQuery(),
    sortId: () => ws.url.sort.gallery ?? null,
  });

  // The selection survives a view switch: seed the grid's own from it, then mirror it back.
  // `ws` is fixed for the life of the view, so reading it once is right.
  // svelte-ignore state_referenced_locally
  if (ws.selection.kind === 'images') gallery.replaceSelection(ws.selection.imageIds);
  $effect(() => {
    ws.select(gallery.workspaceSelection);
  });

  $effect(() =>
    ws.registerPage({
      kind: 'images',
      ids: () => gallery.cards.map((c) => c.image_id),
      total: () => gallery.total,
      apply: (ids) => gallery.replaceSelection(ids),
    }),
  );

  $effect(() => {
    if (gallery.status !== 'loading') {
      ws.matching = { total: gallery.total, hidden: gallery.hidden };
    }
    return () => {
      ws.matching = null;
    };
  });

  $effect(() => {
    void gallery.queryKey;
    untrack(() => void gallery.loadFirst());
  });

  const focusedId = $derived(ws.url.focus?.kind === 'image' ? ws.url.focus.id : null);

  function open(imageId: string): void {
    ws.setFocus({ kind: 'image', id: imageId });
    if (!ws.detailsOpen) ws.toggleDetails();
  }
</script>

<div class="flex min-h-0 flex-1 flex-col" data-testid="gallery-view">
  <h1 class="sr-only">Gallery</h1>
  <BulkBar {ws} onapplied={() => void gallery.loadFirst()} />
  <div class="min-h-0 flex-1 overflow-auto p-4">
    <div class="mb-3 flex flex-wrap items-center gap-3 text-xs text-zinc-300">
      <p data-testid="gallery-header">
        <strong class="text-zinc-100" data-testid="gallery-total"
          >{gallery.total.toLocaleString()}</strong
        >
        image{gallery.total === 1 ? '' : 's'}
        <HiddenItemsNote hidden={gallery.hidden} />
      </p>
      <button
        type="button"
        class="chip needs-work"
        class:active={gallery.unlabeledOnly}
        aria-pressed={gallery.unlabeledOnly}
        data-testid="chip-has-unlabeled"
        onclick={() => (gallery.unlabeledOnly = !gallery.unlabeledOnly)}
      >
        Has unlabelled objects
      </button>
      {#if gallery.sortOptions.length > 0}
        <SortSelect
          testid="gallery-sort"
          options={gallery.sortOptions}
          applied={gallery.sortApplied ?? ''}
          onchange={(id) => ws.setSort('gallery', id)}
        />
      {/if}
      {#if gallery.joinTruncated}
        <span class="text-zinc-500" data-testid="gallery-join-truncated"
          >The item filter matched more images than can be listed; narrow it.</span
        >
      {/if}
    </div>
    <ViewState
      status={gallery.status}
      message={gallery.error ?? gallery.emptyMessage}
      onretry={() => void gallery.loadFirst()}
    >
      <ImageGrid
        cards={gallery.cards}
        thumb={ws.url.thumb}
        isSelected={gallery.isSelected}
        {focusedId}
        hasMore={gallery.hasMore}
        loadingMore={gallery.loadingMore}
        onselect={(id, e) => gallery.select(id, e)}
        onopen={open}
        onloadmore={() => void gallery.loadMore()}
      />
      <p class="mt-2 font-mono text-xs text-zinc-500" data-testid="gallery-footer">
        {gallery.cards.length} / {gallery.total}
      </p>
    </ViewState>
    {#if gallery.status === 'empty'}
      <EmptyNextSteps />
    {/if}
  </div>
</div>

<style>
  .needs-work.active {
    border-color: var(--color-warn);
    background-color: var(--color-warn-soft);
    color: var(--color-warn);
  }
</style>
