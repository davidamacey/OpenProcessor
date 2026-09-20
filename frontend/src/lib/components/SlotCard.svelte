<script lang="ts">
  /**
   * Compact card for one slot detection. Renamed from PlateCard.svelte
   * (P2.4) and parameterized (P2.7,
   * docs/genericization-plan-2026-09-13.md §3.1/§5a) to read through the
   * `readSlot` adapter instead of `PlateBrowseItem`'s hardcoded `plate_*`
   * fields directly. Rather than routing `getPlates`/`PlateBrowseItem`
   * through the adapter server-side (a bigger, riskier change to
   * api.ts's untested mapping path), this card calls `readSlot()` itself
   * on the raw crop object — `PlateBrowseItem`'s flat `plate_*`
   * properties already match `licensePlateSlot`'s wire-field names
   * exactly, so this is a safe, local parameterization: swap the `slot`
   * prop and the card renders a completely different capability set
   * (see `docs/genericization-plan-2026-09-13.md`'s §5.4 example slots)
   * with zero further code change.
   *
   * Used on the /clusters page when class=license_plate, and on the
   * /train page's training-cohort sanity preview.
   *
   * Renders a sub-bbox thumbnail (cropped server-side to the child
   * bbox region) with the parent vehicle class, detector score, a
   * provenance chip strip, and a ⚠ shape warning when the bbox shape
   * envelope fails. Clicking the card emits an `onclick` event so the
   * parent can navigate to the review queue.
   */
  import ProvenanceChip from './ProvenanceChip.svelte';
  import { getRegionThumbUrl, resolveApiUrl, type PlateBrowseItem } from '$lib/api';
  import { readSlot } from '$lib/annotations/readSlot';
  import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';
  import type { SlotSpec, XYXY } from '$lib/annotations/types';

  interface Props {
    crop: PlateBrowseItem;
    /** Which slot's capabilities to render this card with. Defaults to
     *  the legacy license_plate profile — the only configured
     *  instance today — but any `SlotSpec` whose wire field names
     *  match this crop's properties works unchanged. */
    slot?: SlotSpec;
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
    slot = licensePlateSlot,
    onclick,
    onedit,
    onmarkfp,
    selected = false,
    compact = false,
  }: Props = $props();

  // Single adapter read drives every field below — swap `slot` and every
  // derived value here follows, with no other line in this file changing.
  const data = $derived(
    readSlot(
      crop as unknown as Record<string, unknown>,
      slot,
      (crop.bbox_norm ?? [0, 0, 0, 0]) as XYXY,
    ),
  );

  // false_positive plates stay visible (kept as hard negatives) but are
  // dimmed + badged so the operator sees the triage state at a glance.
  const isFalsePositive = $derived(
    slot.capabilities.lifecycle?.falsePositiveState != null &&
      data.lifecycle?.status === slot.capabilities.lifecycle.falsePositiveState,
  );

  // Shape envelope check — delegates to the shared evaluateShapeGate via
  // readSlot, so this agrees with the /review surface on non-finite
  // input (Finding C.1).
  const warn = $derived(data.subBox?.shapeWarning ?? false);

  const thumbUrl = $derived(
    crop.plate_thumbnail_url
      ? resolveApiUrl(crop.plate_thumbnail_url)
      : getRegionThumbUrl(crop.crop_id),
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
        {data.text?.value ?? '—'}
      </span>
      <span class="text-zinc-500">
        {data.subBox?.score != null ? `${(data.subBox.score * 100).toFixed(0)}%` : '—'}
      </span>
    </div>
    <div class="flex flex-wrap items-center gap-1">
      <ProvenanceChip detector={data.provenance?.detector ?? null} size="sm" />
      {#if data.provenance?.verifier}
        <ProvenanceChip detector={data.provenance.verifier} tag="verify" size="sm" />
      {/if}
      {#if crop.class_name}
        <span
          class="rounded border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-300"
        >
          {crop.class_name}
        </span>
      {/if}
    </div>
    {#if !compact && slot.capabilities.provenance?.showChainOnCard && data.provenance?.chain && data.provenance.chain.length > 0}
      <div class="flex flex-wrap gap-0.5">
        {#each data.provenance.chain as entry (entry)}
          <ProvenanceChip raw={entry} size="sm" />
        {/each}
      </div>
    {/if}
  </div>
</button>
