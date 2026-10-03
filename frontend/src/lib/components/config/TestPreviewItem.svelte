<script lang="ts">
  /**
   * A test result's `preview_item` (the item as the write would leave it,
   * §5.1, §7.7) drawn through the same box path as a stored item: the
   * crop's source-image context with the tested item replaced by the
   * preview, rendered by `SourceImageOverlay`. Shared by the pack and
   * region-profile test panels. Nothing is written. `extraShapes` adds
   * overlay shapes (profile-test candidates) in the source-image frame.
   */
  import { getCropContext, apiErrorText } from '$lib/api';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import type { OverlayShape } from '$lib/configTest/overlayShapes';
  import type { Crop, CropContextResponse } from '$lib/types';

  interface Props {
    preview: Crop;
    extraShapes?: OverlayShape[];
  }

  let { preview, extraShapes = [] }: Props = $props();

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
        error = apiErrorText(e);
      });
    return () => ctl.abort();
  });
</script>

<!-- A definite height: SourceImageOverlay sizes its image with percentages,
     which collapse to the image's natural size under an auto-height parent. -->
<div class="h-72 w-full max-w-xl" data-testid="test-preview-item">
  {#if error}
    <p class="text-xs text-red-300">Could not load the source image: {error}</p>
  {:else if context}
    <SourceImageOverlay
      cropId={preview.id}
      {context}
      maxDim={800}
      {extraShapes}
      class="h-full w-full"
    />
  {:else}
    <p class="text-xs text-zinc-500">Loading the source image…</p>
  {/if}
</div>
