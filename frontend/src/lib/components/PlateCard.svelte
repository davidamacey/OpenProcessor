<script lang="ts">
  /**
   * Compact card for one plate detection.
   *
   * Used on the /clusters page when class=license_plate, and on the
   * /train page's training-cohort sanity preview.
   *
   * Renders a plate thumbnail (cropped server-side to the
   * plate_bbox_norm region) with the parent vehicle class, plate
   * score, detector provenance chip, and a ⚠ shape warning when
   * the bbox shape envelope fails. Clicking the card emits an
   * `onclick` event so the parent can navigate to the review queue.
   */
  import DetectorChip from './DetectorChip.svelte';
  import type { PlateBrowseItem } from '$lib/api';

  interface Props {
    crop: PlateBrowseItem;
    onclick?: (crop: PlateBrowseItem) => void;
    /** Compact mode hides the chain strip for dense grids. */
    compact?: boolean;
  }

  let { crop, onclick, compact = false }: Props = $props();

  // Client-side shape envelope check — mirrors the server-side
  // is_plausible_plate_bbox in openprocessor so a row that slips past
  // the worker's gate still gets a UI warning chip.
  function shapeWarning(): boolean {
    if (!crop.plate_bbox_norm || crop.plate_bbox_norm.length !== 4) return false;
    if (!crop.bbox_norm || crop.bbox_norm.length !== 4) return false;
    const [vx1 = 0, vy1 = 0, vx2 = 0, vy2 = 0] = crop.bbox_norm;
    const vw = vx2 - vx1;
    const vh = vy2 - vy1;
    if (vw <= 1e-9 || vh <= 1e-9) return false;
    const [px1 = 0, py1 = 0, px2 = 0, py2 = 0] = crop.plate_bbox_norm;
    const w = (px2 - px1) / vw;
    const h = (py2 - py1) / vh;
    if (w <= 0 || h <= 0) return true;
    const aspect = w / h;
    if (aspect < 1.2 || aspect > 8.0) return true;
    if (w > 0.5) return true;
    if (w * h > 0.15) return true;
    return false;
  }

  const warn = $derived(shapeWarning());
  const thumbUrl = $derived(crop.plate_thumbnail_url ?? `/curation/crops/${crop.crop_id}/plate_thumbnail`);

  function handleClick(): void {
    onclick?.(crop);
  }

  function handleKey(e: KeyboardEvent): void {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onclick?.(crop);
    }
  }
</script>

<button
  type="button"
  class="group flex flex-col items-stretch overflow-hidden rounded-md border border-zinc-800 bg-zinc-950 text-left transition-colors hover:border-blue-500/50"
  onclick={handleClick}
  onkeydown={handleKey}
  title={`Open ${crop.crop_id} in plates review queue`}
>
  <div class="relative aspect-[2/1] w-full bg-zinc-900">
    <img
      src={thumbUrl}
      alt="plate"
      loading="lazy"
      decoding="async"
      class="h-full w-full object-contain"
    />
    {#if warn}
      <span
        class="absolute top-1 right-1 rounded border border-yellow-500/60 bg-yellow-500/85 px-1 py-0.5 text-[9px] font-semibold text-yellow-950"
        title="Bbox shape fails the plate envelope — flag for re-detection"
      >
        ⚠
      </span>
    {/if}
  </div>
  <div class="flex flex-col gap-1 p-2 text-[11px]">
    <div class="flex items-center justify-between gap-1 font-mono">
      <span class="truncate text-zinc-300">
        {crop.plate_text ?? '—'}
      </span>
      <span class="text-zinc-500">
        {crop.plate_score != null ? `${(crop.plate_score * 100).toFixed(0)}%` : '—'}
      </span>
    </div>
    <div class="flex flex-wrap items-center gap-1">
      <DetectorChip detector={crop.plate_detector} size="sm" />
      {#if crop.plate_verifier}
        <DetectorChip detector={crop.plate_verifier} tag="verify" size="sm" />
      {/if}
      {#if crop.class_name}
        <span class="rounded border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-300">
          {crop.class_name}
        </span>
      {/if}
    </div>
    {#if !compact && crop.plate_detector_chain && crop.plate_detector_chain.length > 0}
      <div class="flex flex-wrap gap-0.5">
        {#each crop.plate_detector_chain as entry (entry)}
          <DetectorChip raw={entry} size="sm" />
        {/each}
      </div>
    {/if}
  </div>
</button>
