<script lang="ts">
  /** The focused image: its facts, its served counts and a way into the editor. */
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import ViewState from '$components/state/ViewState.svelte';
  import { getImage } from '$lib/api_images';
  import { apiErrorText } from '$lib/api_errors';
  import { formatTimestamp } from '$lib/format/date';
  import { editorHref } from '$lib/editor/editorUrl';
  import type { ImageDetailResponse } from '$lib/types_images';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws, id }: { ws: WorkspaceController; id: string | null } = $props();

  let detail = $state<ImageDetailResponse | null>(null);
  let error = $state<string | null>(null);

  $effect(() => {
    const imageId = id;
    detail = null;
    error = null;
    if (!imageId) return;
    const ctrl = new AbortController();
    getImage(imageId, ctrl.signal)
      .then((d) => (detail = d))
      .catch((e: unknown) => {
        if (!ctrl.signal.aborted) error = apiErrorText(e);
      });
    return () => ctrl.abort();
  });
  const status = $derived(error ? 'error' : detail ? 'ready' : id ? 'loading' : 'empty');
</script>

<ViewState {status} message={error ?? 'Pick an image to see its details.'}>
  {#if detail}
    {@const image = detail.image}
    {@const counts = detail.counts}
    <section class="space-y-3 text-sm" data-testid="image-details">
      <h2 class="truncate text-xs font-semibold tracking-wide text-zinc-400 uppercase">
        {image.file_name}
      </h2>
      <dl class="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-zinc-300">
        {#if image.width != null && image.height != null}
          <dt class="text-zinc-500">size</dt>
          <dd>{image.width}×{image.height}</dd>
        {/if}
        {#if image.source}
          <dt class="text-zinc-500">source</dt>
          <dd>{image.source}</dd>
        {/if}
        {#if image.taken_at}
          <dt class="text-zinc-500">taken</dt>
          <dd>{formatTimestamp(image.taken_at)}</dd>
        {/if}
        {#if image.indexed_at}
          <dt class="text-zinc-500">indexed</dt>
          <dd>{formatTimestamp(image.indexed_at)}</dd>
        {/if}
        <dt class="text-zinc-500">items</dt>
        <dd data-testid="image-details-items">{counts.items}</dd>
        <dt class="text-zinc-500">labelled</dt>
        <dd>{counts.labeled}</dd>
        <dt class="text-zinc-500">unlabelled</dt>
        <dd>{counts.unlabeled}</dd>
        {#if counts.in_holdout > 0}
          <dt class="text-zinc-500">in test holdout</dt>
          <dd>{counts.in_holdout}</dd>
        {/if}
        {#if image.near_dup}
          <dt class="text-zinc-500">near-dup group</dt>
          <dd>
            {image.near_dup.purpose}{image.near_dup.size
              ? ` · ${image.near_dup.size}`
              : ''}
          </dd>
        {/if}
      </dl>
      <div class="flex flex-wrap gap-2">
        <a
          class="btn btn-primary"
          data-testid="open-image"
          href={resolve(editorHref(image.image_id, page.url.search))}>Open image</a
        >
        <button
          type="button"
          class="btn"
          data-testid="scope-image"
          onclick={() => ws.setScope({ kind: 'image', id: image.image_id })}
          >Show its items</button
        >
        {#if image.near_dup?.is_representative}
          <button
            type="button"
            class="btn"
            data-testid="scope-near-dup"
            onclick={() =>
              ws.setScope({ kind: 'near_dup', id: image.near_dup?.group_id ?? '' })}
            >Show its duplicate group</button
          >
        {/if}
      </div>
    </section>
  {/if}
</ViewState>
