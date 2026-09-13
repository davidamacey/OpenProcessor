<script lang="ts">
  /**
   * Compact card for one slot detection. Renamed from PlateCard.svelte
   * (P2.4, docs/genericization-plan-2026-09-13.md §3.1) — still reads
   * `PlateBrowseItem`'s hardcoded `plate_*` fields directly rather than
   * a generic `slots[key]` lookup, since that requires `getPlates`/
   * `PlateBrowseItem` to route through the slot adapter (`readSlot`),
   * which hasn't happened yet (tracked alongside the SlotGallery
   * extraction). This rename + the shape-gate dedup are what's real
   * today; full field parameterization is follow-up work.
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
  import ProvenanceChip from './ProvenanceChip.svelte';
  import { getPlateThumbUrl, resolveApiUrl, type PlateBrowseItem } from '$lib/api';
  import { evaluateShapeGate, PLATE_SHAPE_ENVELOPE } from '$lib/shapeGate';

  interface Props {
    crop: PlateBrowseItem;
    onclick?: (crop: PlateBrowseItem, e: MouseEvent) => void;
    /** Edit affordance (✎): parent opens the sub-bbox editor for this plate. */
    onedit?: (crop: PlateBrowseItem) => void;
    /** Quick false-positive (✗): parent marks this plate false_positive. */
    onmarkfp?: (crop: PlateBrowseItem) => void;
    /** Selection state for multi-select bulk actions. */
    selected?: boolean;
    /** Compact mode hides the chain strip for dense grids. */
    compact?: boolean;
  }

  let {
    crop,
    onclick,
    onedit,
    onmarkfp,
    selected = false,
    compact = false,
  }: Props = $props();

  // false_positive plates stay visible (kept as hard negatives) but are
  // dimmed + badged so the operator sees the triage state at a glance.
  const isFalsePositive = $derived(crop.plate_status === 'false_positive');

  // Shared shape envelope check — mirrors the server-side
  // is_plausible_plate_bbox in openprocessor so a row that slips past the
  // worker's gate still gets a UI warning chip. Delegates to
  // `evaluateShapeGate` (shapeGate.ts) so this agrees with the
  // /clusters (`getPlates`) and /review surfaces on non-finite input —
  // this card used to have its own inline copy with no `Number.isFinite`
  // guard, so a corrupt row warned in /review but not here.
  const warn = $derived(
    evaluateShapeGate(crop.plate_bbox_norm, crop.bbox_norm, PLATE_SHAPE_ENVELOPE),
  );
  const thumbUrl = $derived(
    crop.plate_thumbnail_url
      ? resolveApiUrl(crop.plate_thumbnail_url)
      : getPlateThumbUrl(crop.crop_id),
  );

  function handleClick(e: MouseEvent): void {
    onclick?.(crop, e);
  }

  function handleKey(e: KeyboardEvent): void {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onclick?.(crop, e as unknown as MouseEvent);
    }
  }
</script>

<button
  type="button"
  style="content-visibility:auto;contain-intrinsic-size:auto 130px"
  class="group relative flex flex-col items-stretch overflow-hidden rounded-md border bg-zinc-950 text-left transition-colors {selected
    ? 'border-blue-500 ring-2 ring-blue-500/40'
    : 'border-zinc-800 hover:border-blue-500/50'} {isFalsePositive ? 'opacity-50' : ''}"
  onclick={handleClick}
  onkeydown={handleKey}
  title={`${crop.crop_id} — click to select, ✎ to edit`}
>
  <div class="relative aspect-[2/1] w-full bg-zinc-900">
    <img
      src={thumbUrl}
      alt="plate"
      loading="lazy"
      decoding="async"
      class="h-full w-full object-contain"
    />
    <!-- Selection checkbox (top-left). -->
    <span
      class="absolute top-1 left-1 flex h-4 w-4 items-center justify-center rounded-sm border text-[10px] {selected
        ? 'border-blue-400 bg-blue-500 text-white'
        : 'border-zinc-500 bg-black/50 text-transparent group-hover:text-zinc-400'}"
      aria-hidden="true"
    >
      ✓
    </span>
    <!-- Edit + quick false-positive (hover). Span (not button) to stay
         valid inside the outer button; stopPropagation so they don't
         trigger select. -->
    {#if onedit}
      <span
        role="button"
        tabindex="0"
        class="absolute right-1 bottom-1 cursor-pointer rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
        onclick={(e) => {
          e.stopPropagation();
          onedit?.(crop);
        }}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.stopPropagation();
            onedit?.(crop);
          }
        }}
        title="Edit plate bbox / status"
      >
        ✎
      </span>
    {/if}
    {#if onmarkfp && !isFalsePositive}
      <span
        role="button"
        tabindex="0"
        class="absolute top-1 right-1 cursor-pointer rounded-sm border border-red-500/60 bg-red-500/80 px-1 py-0.5 text-[10px] font-semibold text-white opacity-0 transition group-hover:opacity-100"
        onclick={(e) => {
          e.stopPropagation();
          onmarkfp?.(crop);
        }}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.stopPropagation();
            onmarkfp?.(crop);
          }
        }}
        title="Mark false positive"
      >
        ✗ FP
      </span>
    {/if}
    {#if isFalsePositive}
      <span
        class="absolute top-1 right-1 rounded border border-red-500/60 bg-red-600/85 px-1 py-0.5 text-[9px] font-semibold text-white"
      >
        false pos
      </span>
    {/if}
    {#if warn}
      <span
        class="absolute bottom-1 left-1 rounded border border-yellow-500/60 bg-yellow-500/85 px-1 py-0.5 text-[9px] font-semibold text-yellow-950"
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
      <ProvenanceChip detector={crop.plate_detector} size="sm" />
      {#if crop.plate_verifier}
        <ProvenanceChip detector={crop.plate_verifier} tag="verify" size="sm" />
      {/if}
      {#if crop.class_name}
        <span
          class="rounded border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-300"
        >
          {crop.class_name}
        </span>
      {/if}
    </div>
    {#if !compact && crop.plate_detector_chain && crop.plate_detector_chain.length > 0}
      <div class="flex flex-wrap gap-0.5">
        {#each crop.plate_detector_chain as entry (entry)}
          <ProvenanceChip raw={entry} size="sm" />
        {/each}
      </div>
    {/if}
  </div>
</button>
