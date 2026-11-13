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
  import DuplicatesControl from '$components/workspace/DuplicatesControl.svelte';
  import ViewMoreMenu from '$components/workspace/ViewMoreMenu.svelte';
  import ImageGrid from '$components/images/ImageGrid.svelte';
  import { createImageGalleryController } from '$lib/images/imageGalleryController.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  const gallery = createImageGalleryController({
    itemQuery: () => ws.itemQuery(),
    sortId: () => ws.url.sort.gallery ?? null,
    nearDup: () => ws.url.nearDup,
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
  <!-- The controls sit above the scroller: a popover that opens from them is not clipped by it. -->
  <div
    class="flex shrink-0 flex-wrap items-center gap-3 px-4 pt-4 pb-3 text-meta text-fg-2"
  >
    <p data-testid="gallery-header">
      <strong class="text-fg-1" data-testid="gallery-total"
        >{gallery.total.toLocaleString()}</strong
      >
      photo{gallery.total === 1 ? '' : 's'}
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
    <DuplicatesControl value={ws.url.nearDup} onchange={(v) => ws.setNearDup(v)} />
    {#if gallery.sortOptions.length > 0}
      <SortSelect
        testid="gallery-sort"
        options={gallery.sortOptions}
        applied={gallery.sortApplied ?? ''}
        onchange={(id) => ws.setSort('gallery', id)}
      />
    {/if}
    <span class="grow"></span>
    <button
      type="button"
      class="btn btn-primary"
      data-testid="gallery-review"
      title="Review these items one by one"
      onclick={() => ws.setView('review')}
    >
      Review these
    </button>
    <ViewMoreMenu {ws} />
    {#if gallery.joinTruncated}
      <span class="text-fg-3" data-testid="gallery-join-truncated"
        >The item filter matched more images than can be listed; narrow it.</span
      >
    {/if}
  </div>
  <div class="min-h-0 flex-1 overflow-auto px-4 pb-4">
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
        onlightbox={(card) => void ws.openLightbox(card.image_id, card.file_name)}
        onloadmore={() => void gallery.loadMore()}
      />
      <p class="mt-2 text-meta text-fg-3 tabular-nums" data-testid="gallery-footer">
        {gallery.cards.length.toLocaleString()} of {gallery.total.toLocaleString()}
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
