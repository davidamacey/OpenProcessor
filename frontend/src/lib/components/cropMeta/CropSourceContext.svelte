<script lang="ts">
  import type { Crop } from '$lib/types';
  import { getThumbUrl } from '$lib/api';
  import SourceImageOverlay from '$components/SourceImageOverlay.svelte';
  import type { OverlayShape } from '$lib/configTest/overlayShapes';
  import type { CropMetaLoader } from '$lib/cropMeta/cropMetaLoader.svelte';
  import { formatTimestamp } from '$lib/format/date';

  let { crop, loader }: { crop: Crop; loader: CropMetaLoader } = $props();

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

<div class="mt-4 border-t border-zinc-800 pt-3">
  <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">
    Source image
  </div>
  {#if imageLoading}
    <p class="text-[11px] text-zinc-500">Loading…</p>
  {:else if imageError}
    <p class="text-[11px] text-red-300">Source image unavailable: {imageError}</p>
  {:else if imageMeta}
    {#if contextRes}
      <div class="mb-2 h-40 w-full overflow-hidden rounded bg-zinc-950">
        <SourceImageOverlay
          cropId={crop.id}
          context={contextRes}
          extraShapes={maskShapes}
        />
      </div>
    {/if}
    <dl class="grid grid-cols-2 gap-y-1 text-[11px] text-zinc-400">
      {#if imageMeta.width != null && imageMeta.height != null}
        <dt class="text-zinc-500">Size</dt>
        <dd class="font-mono">{imageMeta.width}×{imageMeta.height}</dd>
      {/if}
      {#if imageMeta.source}
        <dt class="text-zinc-500">Source</dt>
        <dd>{imageMeta.source}</dd>
      {/if}
      {#if imageMeta.indexed_at}
        <dt class="text-zinc-500">Indexed</dt>
        <dd class="font-mono" title={imageMeta.indexed_at}>
          {formatTimestamp(imageMeta.indexed_at)}
        </dd>
      {/if}
    </dl>
    {#if siblings && siblings.length > 0}
      <div class="mt-2 text-[10px] uppercase tracking-wider text-zinc-500">
        Siblings ({siblings.length})
      </div>
      <div class="mt-1 flex flex-wrap gap-1.5">
        {#each siblings as sib (sib.id)}
          <img
            src={getThumbUrl(sib.id, 64)}
            alt="sibling crop"
            title={sib.class_name ?? sib.id}
            class="h-12 w-12 rounded border border-zinc-800 bg-zinc-900 object-contain"
          />
        {/each}
      </div>
    {/if}
  {/if}
</div>

{#if crop.source_image_path || crop.updated_at}
  <div class="mt-4 border-t border-zinc-800 pt-3 text-[11px] text-zinc-500">
    {#if crop.source_image_path}
      <div class="truncate" title={crop.source_image_path}>
        <span class="text-zinc-600">src:</span>
        {crop.source_image_path}
      </div>
    {/if}
    {#if crop.updated_at}
      <div>
        <span class="text-zinc-600">updated:</span>
        <span class="font-mono" title={crop.updated_at}
          >{formatTimestamp(crop.updated_at)}</span
        >
      </div>
    {/if}
    <div>
      <span class="text-zinc-600">id:</span>
      <span class="font-mono">{crop.id}</span>
    </div>
  </div>
{/if}
