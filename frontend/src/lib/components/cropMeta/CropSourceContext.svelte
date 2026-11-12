<script lang="ts">
  import type { Crop } from '$lib/types';
  import { getThumbUrl } from '$lib/api_cropRead';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import type { OverlayShape } from '$lib/configTest/overlayShapes';
  import type { CropMetaLoader } from '$lib/cropMeta/cropMetaLoader.svelte';
  import { formatTimestamp } from '$lib/format/date';
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';

  let {
    crop,
    loader,
    onopencrop,
    showImage = true,
  }: {
    crop: Crop;
    loader: CropMetaLoader;
    /** Opens a sibling item in the host; absent, siblings are plain thumbnails. */
    onopencrop?: (crop: Crop) => void;
    /** False beside a host that already draws the source image. */
    showImage?: boolean;
  } = $props();

  const imageMeta = $derived(loader.imageMeta);
  const siblings = $derived(loader.siblings);
  const imageError = $derived(loader.imageError);
  const imageLoading = $derived(loader.imageLoading);
  const contextRes = $derived(loader.contextRes);

  // v0.4.0: the served mask outline (already image-normalised), drawn as
  // served. Only a crop read singly carries it; a list-served one is null.
  const maskShapes = $derived<OverlayShape[]>(
    crop.mask_polygon != null && crop.mask_polygon.length >= 3
      ? [
          {
            key: 'mask',
            kind: 'polygon',
            dimmed: false,
            label: 'mask',
            title: 'mask',
            points: crop.mask_polygon
              .filter((p) => p.length >= 2)
              .map((p) => [p[0]!, p[1]!] as [number, number]),
          },
        ]
      : [],
  );
</script>

<div class="mt-4 border-t border-line pt-3">
  <div class="mb-1.5 text-caption uppercase tracking-wider text-fg-3">Source image</div>
  {#if imageLoading}
    <p class="text-caption text-fg-3">Loading…</p>
  {:else if imageError}
    <p class="text-caption text-danger-text">Source image unavailable: {imageError}</p>
  {:else if imageMeta}
    {#if contextRes && showImage}
      <div class="mb-2 h-40 w-full overflow-hidden rounded bg-surface-0">
        <SourceImageOverlay
          imageId={contextRes.image.image_id}
          selectedCropId={crop.id}
          context={contextRes}
          extraShapes={maskShapes}
        />
      </div>
    {/if}
    <a
      class="text-caption text-info-text underline hover:text-info-text"
      data-testid="open-image"
      href={resolve(projectHref(`/images/${encodeURIComponent(imageMeta.image_id)}`))}
      >Open image</a
    >
    {#if siblings && siblings.length > 0}
      <div class="mt-2 text-caption uppercase tracking-wider text-fg-3">
        Siblings ({siblings.length})
      </div>
      <div class="mt-1 flex flex-wrap gap-1.5">
        {#each siblings as sib (sib.id)}
          {#snippet thumb()}
            <img
              src={getThumbUrl(sib.id, 64)}
              alt="sibling crop"
              class="h-12 w-12 rounded border border-line bg-surface-1 object-contain"
            />
          {/snippet}
          {#if onopencrop}
            <button
              type="button"
              data-testid="sibling-crop"
              title={sib.class_name ?? sib.id}
              aria-label="Open sibling {sib.class_name ?? sib.id}"
              class="rounded hover:ring-2 hover:ring-fg-3"
              onclick={() => onopencrop(sib)}
            >
              {@render thumb()}
            </button>
          {:else}
            <span title={sib.class_name ?? sib.id}>{@render thumb()}</span>
          {/if}
        {/each}
      </div>
    {/if}
  {/if}
</div>

{#if crop.source_image_path || crop.updated_at}
  <div class="mt-4 border-t border-line pt-3 text-caption text-fg-3">
    {#if crop.source_image_path}
      <div class="truncate" title={crop.source_image_path}>
        <span class="text-fg-3">src:</span>
        {crop.source_image_path}
      </div>
    {/if}
    {#if crop.updated_at}
      <div>
        <span class="text-fg-3">updated:</span>
        <span class="font-mono" title={crop.updated_at}
          >{formatTimestamp(crop.updated_at)}</span
        >
      </div>
    {/if}
    <div>
      <span class="text-fg-3">id:</span>
      <span class="font-mono">{crop.id}</span>
    </div>
  </div>
{/if}
