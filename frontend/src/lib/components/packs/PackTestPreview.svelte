<script lang="ts">
  /**
   * A test result's `preview_item` (the item as the write would leave it,
   * §5.1) drawn through the same box path as a stored item: the crop's
   * source-image context with the tested item replaced by the preview,
   * rendered by `SourceImageOverlay` (§7.6 item 3). Nothing is written.
   */
  import { getCropContext, packErrorText } from '$lib/api';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import type { Crop, CropContextResponse } from '$lib/types';

  interface Props {
    preview: Crop;
  }

  let { preview }: Props = $props();

  let context = $state<CropContextResponse | null>(null);
  let error = $state<string | null>(null);

  $effect(() => {
    const p = preview;
    const ctl = new AbortController();
    context = null;
    error = null;
    getCropContext(p.id, ctl.signal)
      .then((ctx) => {
        const has = ctx.items.some((i) => i.id === p.id);
        context = {
          ...ctx,
          items: has ? ctx.items.map((i) => (i.id === p.id ? p : i)) : [...ctx.items, p],
        };
      })
      .catch((e) => {
        if ((e as Error)?.name === 'AbortError') return;
        error = packErrorText(e);
      });
    return () => ctl.abort();
  });
</script>

<div data-testid="pack-test-preview">
  {#if error}
    <p class="text-xs text-red-300">Could not load the source image: {error}</p>
  {:else if context}
    <SourceImageOverlay
      cropId={preview.id}
      {context}
      maxDim={800}
      class="max-h-80 w-full"
    />
  {:else}
    <p class="text-xs text-zinc-500">Loading the source image…</p>
  {/if}
</div>
