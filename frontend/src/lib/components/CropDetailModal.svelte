<script lang="ts">
  import { getThumbUrl } from '$lib/api_cropRead';
  import type { Crop } from '$lib/types';
  import CropMetaPanel from './CropMetaPanel.svelte';
  import SourceImageOverlay from './SourceImageOverlay.svelte';
  import Modal from '$components/Modal.svelte';

  interface Props {
    crop: Crop;
    onclose: () => void;
  }

  let { crop: given, onclose }: Props = $props();
  // A Reprocess answers with the served post-write crop; show it in place
  // of the one the modal was opened with.
  let adopted = $state<Crop | null>(null);
  // A sibling opened from the panel replaces the crop shown until the modal closes.
  let opened = $state<Crop | null>(null);
  const base = $derived(opened ?? given);
  const crop = $derived(adopted?.id === base.id ? adopted : base);
</script>

<Modal
  label="Crop details"
  {onclose}
  layer="top"
  scrim="bg-black/80"
  panelClass="relative grid max-h-[92vh] w-full max-w-6xl grid-cols-1 gap-4 overflow-hidden rounded-lg border border-line-strong bg-surface-0 p-4 lg:grid-cols-[1fr_320px]"
>
  <!-- K5 (visual audit 2026-09-24): the close button sits inside the
         panel's padding (it was half outside the top-right corner) and the
         meta column is min-w-0 so its chips wrap instead of clipping. -->
  <!-- K6: boxes/labels drawn client-side from the crop's context — same
         component the review page and the crop-card lightbox use, so the
         rendering matches across surfaces. -->
  <div class="flex min-h-0 flex-col gap-2">
    <div class="text-xs uppercase tracking-wider text-fg-3">Source</div>
    <!-- A definite height: SourceImageOverlay sizes its image with
           percentages, which collapse under an auto-height parent and left the
           boxes off the picture. No stretched background either — a flex-1 box
           taller than the image read as black bars under it. -->
    <div class="h-[50vh] w-full lg:h-[62vh]" data-testid="source-box">
      {#if crop.image_id}
        <SourceImageOverlay
          imageId={crop.image_id}
          selectedCropId={crop.id}
          align="start"
          class="h-full w-full"
        />
      {/if}
    </div>
    <div class="flex items-center gap-3">
      <div class="text-xs text-fg-3">Crop</div>
      <img
        src={getThumbUrl(crop.id, 192)}
        alt="crop"
        loading="lazy"
        class="h-24 w-24 rounded border border-line bg-surface-1 object-contain"
      />
    </div>
  </div>

  <div class="min-h-0 min-w-0 overflow-y-auto pr-6">
    <CropMetaPanel
      {crop}
      onreprocessed={(c) => (adopted = c)}
      onopencrop={(c) => (opened = c)}
    />
  </div>

  <button
    type="button"
    class="absolute top-2 right-2 z-10 rounded-full border border-line-strong bg-surface-1 px-2 py-1 text-sm text-white shadow-lg hover:bg-surface-2"
    onclick={onclose}
    aria-label="Close"
  >
    ×
  </button>
</Modal>
