<script lang="ts">
  /** The focused image: its facts, its served counts and a way into the editor. */
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import Icon from '$components/ui/Icon.svelte';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import { getImage } from '$lib/api_images';
  import { apiErrorText } from '$lib/api_errors';
  import { formatTimestamp } from '$lib/format/date';
  import { activeMap } from '$lib/map/activeMap.svelte';
  import { editorHref } from '$lib/editor/editorUrl';
  import NearDupGroupModal from '$components/nearDup/NearDupGroupModal.svelte';
  import type { ImageDetailResponse } from '$lib/types_images';
  import { keymapStore } from '$stores/keymap.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws, id }: { ws: WorkspaceController; id: string | null } = $props();

  let detail = $state<ImageDetailResponse | null>(null);
  let error = $state<string | null>(null);
  let groupOpen = $state(false);

  $effect(() => {
    const imageId = id;
    detail = null;
    error = null;
    groupOpen = false;
    if (!imageId) return;
    const ctrl = new AbortController();
    getImage(imageId, ctrl.signal)
      .then((d) => (detail = d))
      .catch((e: unknown) => {
        if (!ctrl.signal.aborted) error = apiErrorText(e);
      });
    return () => ctrl.abort();
  });
  // The photos of the cell the user stepped out of, as the cell's served summary counts them.
  const cellPhotos = $derived(activeMap.ctl?.summary?.images ?? null);
  const status = $derived(error ? 'error' : detail ? 'ready' : id ? 'loading' : 'empty');
</script>

<ViewState {status} message={error ?? 'Pick an image to see its details.'}>
  {#if detail}
    {@const image = detail.image}
    {@const counts = detail.counts}
    <section class="space-y-3 text-body" data-testid="image-details">
      {#if ws.previousFocus?.kind === 'cell'}
        <button
          type="button"
          class="btn btn-ghost btn-sm"
          data-testid="back-to-cell"
          onclick={() => ws.setFocus(ws.previousFocus)}
        >
          <Icon name="arrow-left" size={14} />Back to the cell{cellPhotos != null
            ? ` (${cellPhotos.toLocaleString()} photo${cellPhotos === 1 ? '' : 's'})`
            : ''}
        </button>
      {/if}
      <div
        class="relative overflow-hidden rounded border border-line bg-surface-0"
        data-testid="image-preview"
      >
        <!-- A mouse shortcut for the View full size button below, which is the keyboard and
             screen-reader path; a sibling of the image, so no control nests in another. -->
        <button
          type="button"
          tabindex="-1"
          aria-hidden="true"
          class="absolute inset-0 z-10 cursor-zoom-in"
          title="View full size"
          data-testid="image-preview-open"
          onclick={() => void ws.openLightbox(image.image_id, image.file_name)}
        ></button>
        <SourceImageOverlay imageId={image.image_id} context={detail} maxDim={640} />
      </div>
      <h2 class="truncate text-title font-semibold text-fg-1" title={image.file_name}>
        {image.file_name}
      </h2>
      <dl
        class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-meta text-fg-2 tabular-nums"
      >
        {#if image.width != null && image.height != null}
          <dt class="text-fg-3">Size</dt>
          <dd>{image.width}×{image.height}</dd>
        {/if}
        {#if image.source}
          <dt class="text-fg-3">Source</dt>
          <dd>{image.source}</dd>
        {/if}
        {#if image.taken_at}
          <dt class="text-fg-3">Taken</dt>
          <dd>{formatTimestamp(image.taken_at)}</dd>
        {/if}
        {#if image.indexed_at}
          <dt class="text-fg-3">Indexed</dt>
          <dd>{formatTimestamp(image.indexed_at)}</dd>
        {/if}
      </dl>
      <p class="text-meta text-fg-1 tabular-nums" data-testid="image-details-items">
        {counts.items.toLocaleString()} detection{counts.items === 1 ? '' : 's'} · {counts.labeled.toLocaleString()}
        labelled{#if counts.in_holdout > 0}
          · {counts.in_holdout.toLocaleString()} in the test holdout (not listed){/if}
      </p>
      <div class="flex flex-wrap items-center gap-2">
        <a
          class="btn"
          data-testid="open-image"
          title="Edit boxes ({keymapStore.glyph('workspace.open_editor')})"
          href={resolve(editorHref(image.image_id, page.url.search))}>Edit boxes</a
        >
        <button
          type="button"
          class="btn btn-icon"
          aria-label="View full size"
          title="View full size ({keymapStore.glyph('workspace.lightbox')})"
          data-testid="lightbox-image"
          onclick={() => void ws.openLightbox(image.image_id, image.file_name)}
        >
          <Icon name="maximize" size={16} />
        </button>
        {#if ws.url.scope?.kind === 'image' && ws.url.scope.id === image.image_id}
          <button
            type="button"
            class="chip chip-accent"
            data-testid="scope-image"
            title="Leave this image"
            onclick={() => ws.setScope(null)}
          >
            Its {counts.items} detections<Icon
              name="x"
              size={12}
              class="ml-1 inline-block"
            />
          </button>
        {:else}
          <button
            type="button"
            class="btn"
            data-testid="scope-image"
            onclick={() => ws.showImageItems(image.image_id)}
            >Show its {counts.items} detection{counts.items === 1 ? '' : 's'}</button
          >
        {/if}
      </div>
      {#if image.near_dup}
        <p class="text-meta text-fg-2" data-testid="image-near-dup">
          In a group of {image.near_dup.size ?? '?'} ·
          <button
            type="button"
            class="text-accent-text underline underline-offset-2"
            data-testid="decide-near-dup"
            onclick={() => (groupOpen = true)}>Review this group</button
          >
          {#if image.near_dup.is_representative}
            ·
            <button
              type="button"
              class="text-accent-text underline underline-offset-2"
              data-testid="scope-near-dup"
              onclick={() =>
                ws.setScope({ kind: 'near_dup', id: image.near_dup?.group_id ?? '' })}
              >Show its duplicate group</button
            >
          {/if}
        </p>
      {/if}
    </section>
    {#if groupOpen && image.near_dup}
      <NearDupGroupModal
        level="image"
        group={image.near_dup}
        onclose={() => (groupOpen = false)}
      />
    {/if}
  {/if}
</ViewState>
