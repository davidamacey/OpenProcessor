<script lang="ts">
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { getSourceImageWithBbox, getThumbUrl } from '$lib/api';
  import type { Crop } from '$lib/types';
  import CropMetaPanel from './CropMetaPanel.svelte';

  interface Props {
    crop: Crop;
    onclose: () => void;
  }

  let { crop, onclose }: Props = $props();

  // Close on Esc — matches the existing CropCard expanded-image modal so
  // muscle memory carries over.
  function onWindowKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      onclose();
    }
  }
</script>

<svelte:window onkeydown={onWindowKeydown} />

<div
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
  role="dialog"
  aria-modal="true"
  aria-label="Crop details"
  use:focusOnMount
  onclick={(e) => {
    // Backdrop only: a click that bubbled up from the panel is not a
    // dismiss gesture.
    if (e.target === e.currentTarget) onclose();
  }}
  onkeydown={(e) => {
    if (e.key === 'Escape') onclose();
  }}
  tabindex="-1"
>
  <div
    class="relative grid max-h-[92vh] w-full max-w-6xl grid-cols-1 gap-4 overflow-hidden rounded-lg border border-zinc-700 bg-zinc-950 p-4 md:grid-cols-[1fr_320px]"
  >
    <!-- Source image with burned-in bbox. Same endpoint as the review
         page so the rendering matches across surfaces. -->
    <div class="flex min-h-0 flex-col gap-2">
      <div class="text-xs uppercase tracking-wider text-zinc-500">Source</div>
      <div class="flex min-h-0 flex-1 items-center justify-center bg-black">
        <img
          src={getSourceImageWithBbox(crop.id)}
          alt="source"
          loading="eager"
          decoding="async"
          class="max-h-[78vh] max-w-full object-contain"
        />
      </div>
      <div class="flex items-center gap-3">
        <div class="text-xs text-zinc-500">Crop</div>
        <img
          src={getThumbUrl(crop.id, 192)}
          alt="crop"
          loading="lazy"
          class="h-24 w-24 rounded border border-zinc-800 bg-zinc-900 object-contain"
        />
      </div>
    </div>

    <div class="min-h-0 overflow-y-auto">
      <CropMetaPanel {crop} />
    </div>

    <button
      type="button"
      class="absolute -top-3 -right-3 rounded-full border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-white shadow-lg hover:bg-zinc-800"
      onclick={onclose}
      aria-label="Close"
    >
      ×
    </button>
  </div>
</div>
