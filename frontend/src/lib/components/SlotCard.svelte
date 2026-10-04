<script lang="ts">
  /**
   * Compact card for one slot detection, read through the `readSlot`
   * adapter against the given `slot` (docs/genericization-plan-2026-09-13.md
   * §3.1/§5a): swap the `slot` prop and the card renders a completely
   * different capability set with zero further code change.
   *
   * `getRegions` (`api.ts`) now maps every row's `slots` server-side
   * (C2, docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §4) —
   * this card prefers that pre-computed `crop.slots[slot.key]` and only
   * falls back to calling `readSlot()` itself when the raw row never
   * went through `getRegions` (e.g. a locally-constructed fixture).
   *
   * Used on the /clusters slot gallery, and on the /train page's
   * training-cohort sanity preview.
   *
   * Renders a sub-bbox thumbnail (cropped server-side to the child
   * bbox region, per the active slot's own `subBox.thumbnail`
   * path/aspect — not a hardcoded URL/ratio) with the parent item
   * class, detector score, and a provenance chip strip.
   * Clicking the card emits an `onclick` event so the parent can
   * navigate to the review queue.
   */
  import ProvenanceChip from './ProvenanceChip.svelte';
  import ReprocessControl from './datasets/ReprocessControl.svelte';
  import { getThumbUrl, resolveApiUrl, scoped, type RegionBrowseItem } from '$lib/api';
  import { readSlot } from '$lib/annotations/readSlot';
  import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
  import { displayBoxOf, rowBoxOf } from '$lib/annotations/rowBox';
  import type { SlotSpec, XYXY } from '$lib/annotations/types';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import type { Crop } from '$lib/types';

  interface Props {
    crop: RegionBrowseItem;
    /** Which slot's capabilities to render this card with. Required: the
     *  caller always knows the slot it is browsing. */
    slot: SlotSpec;
    onclick?: (crop: RegionBrowseItem, e: MouseEvent) => void;
    /** Edit affordance (✎): parent opens the sub-bbox editor for this slot. */
    onedit?: (crop: RegionBrowseItem) => void;
    /** Quick false-positive (✗): parent marks this slot false_positive. */
    onmarkfp?: (crop: RegionBrowseItem) => void;
    /** Selection state for multi-select bulk actions. */
    selected?: boolean;
    /** Compact mode hides the chain strip for dense grids. */
    compact?: boolean;
    /** The served post-write items after an image Reprocess from this
     *  card; the host adopts them. */
    onreprocessed?: (items: Crop[]) => void;
  }

  let {
    crop,
    slot,
    onclick,
    onedit,
    onmarkfp,
    selected = false,
    compact = false,
    onreprocessed,
  }: Props = $props();

  // Prefer the pre-computed slots map (getRegions already ran
  // mapCropSlots server-side); fall back to calling readSlot() directly
  // for a raw row that never went through that path.
  const data = $derived(
    crop.slots?.[slot.key] ??
      readSlot(
        crop as unknown as Record<string, unknown>,
        slot,
        (crop.bbox_norm ?? [0, 0, 0, 0]) as XYXY,
      ),
  );

  // The box this card describes: the row's own (region_box_id) or, for
  // an item-level row, the item's first box. Every per-box value below
  // reads it — the item carries no per-box scalars.
  const boxCount = $derived(data.subBoxes?.length ?? null);
  const rowBox = $derived(rowBoxOf(data, crop.region_box_id));
  const box = $derived(displayBoxOf(data, crop.region_box_id));
  $effect(() => {
    if (box?.locked) void reprocessVocabularyStore.init();
  });

  // false_positive boxes stay visible (kept as hard negatives) but are
  // dimmed + badged so the operator sees the triage state at a glance.
  const isFalsePositive = $derived(box?.state === 'false_positive');
  // A box the verifier (or a geometry gate) rejected, kept for human
  // review/reversal.
  const isRejectedBox = $derived(box?.state === 'rejected');

  const thumbCap = $derived(slot.capabilities.subBox?.thumbnail);
  // The served per-box thumbnail wins; otherwise build the slot's own
  // thumbnail path for the box id. An item with no box has nothing to
  // crop, so it shows the item thumbnail. The item's served
  // `region_revision` rides along as `v=` so an edited box is re-cropped
  // rather than served from the browser's image cache.
  const revision = $derived(data.boxSet?.revision ?? null);
  const thumbUrl = $derived.by(() => {
    let url: string;
    if (box?.thumbnailUrl) {
      url = resolveApiUrl(box.thumbnailUrl);
    } else if (box?.boxId && thumbCap) {
      url = resolveApiUrl(
        `${scoped()}${thumbCap.path(crop.crop_id, box.boxId, thumbCap.defaultSize)}`,
      );
    } else {
      return getThumbUrl(crop.crop_id);
    }
    return revision == null ? url : `${url}${url.includes('?') ? '&' : '?'}v=${revision}`;
  });
  const thumbAspect = $derived(thumbCap?.aspect ?? '2 / 1');

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

<!-- The wrapper owns hover (`group`) and hosts the Reprocess control, which
     cannot sit inside the card's own <button>. -->
<div class="group relative">
  <button
    type="button"
    class="relative flex w-full flex-col items-stretch overflow-hidden rounded-md border bg-zinc-950 text-left transition-colors {selected
      ? 'border-blue-500 ring-2 ring-blue-500/40'
      : 'border-zinc-800 hover:border-blue-500/50'} {isFalsePositive ? 'opacity-50' : ''}"
    onclick={handleClick}
    onkeydown={handleKey}
    title={`${crop.crop_id} — click to select, ✎ to edit`}
  >
    <div class="relative w-full bg-zinc-900" style="aspect-ratio: {thumbAspect}">
      <img
        src={thumbUrl}
        alt={slot.label.singular}
        loading="lazy"
        decoding="async"
        class="h-full w-full object-contain"
      />
      {#if box?.locked}
        <span
          class="absolute top-1 left-7 flex items-center rounded-sm bg-black/60 px-1 py-0.5 text-zinc-200"
          title={reprocessVocabularyStore.lockText()}
          data-testid="box-locked"
        >
          <svg viewBox="0 0 16 16" fill="currentColor" class="h-3 w-3" aria-hidden="true">
            <path
              d="M8 1a3.5 3.5 0 0 0-3.5 3.5V6H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1h-.5V4.5A3.5 3.5 0 0 0 8 1Zm2 5H6V4.5a2 2 0 1 1 4 0V6Z"
            />
          </svg>
        </span>
      {/if}
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
          title="Edit {slot.label.singular} bbox / status"
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
      {:else if isRejectedBox}
        {@const rejectionKind = regionVocabularyStore.rejectionReasonKind(
          box?.rejectionReason,
        )}
        <!-- 3f1a11e adoption: badge color follows the served kind —
           model_verdict (verifier rejected) reads red, needs_human (no
           verdict) reads neutral zinc rather than a rejection color,
           automatic (geometry gate) keeps the original amber. Tooltip is
           the served label, not the raw id. -->
        <span
          class={`absolute top-1 right-1 rounded border px-1 py-0.5 text-[9px] font-semibold text-white ${
            rejectionKind === 'model_verdict'
              ? 'border-red-500/60 bg-red-600/85'
              : rejectionKind === 'needs_human'
                ? 'border-zinc-500/60 bg-zinc-600/85'
                : 'border-amber-500/60 bg-amber-600/85'
          }`}
          title={box?.rejectionReason
            ? regionVocabularyStore.rejectionReasonLabel(box.rejectionReason)
            : undefined}
        >
          candidate
        </span>
      {:else if boxCount != null && boxCount > 0}
        <!-- W8 multi-box: box count (+ this row's own box state, when the
           browse route selected a specific box via region_box_id). -->
        <span
          class="absolute top-1 right-1 rounded border border-zinc-600/60 bg-zinc-800/85 px-1 py-0.5 text-[9px] font-semibold text-zinc-200"
          title={rowBox ? `box state: ${rowBox.state}` : `${boxCount} box(es)`}
        >
          {boxCount}&nbsp;box{boxCount === 1 ? '' : 'es'}{#if rowBox}
            · {rowBox.state}{/if}
        </span>
      {/if}
    </div>
    <div class="flex min-w-0 flex-col gap-1 p-2 text-[11px]">
      <!-- C3 (visual audit 2026-09-24): the text value keeps its own
         width; the disagree badge wraps under it instead of squeezing the
         value to "6…" on a narrow card. C6: plain text, no emoji glyph. -->
      <div class="flex min-w-0 flex-wrap items-center justify-between gap-1 font-mono">
        <!-- OpenProcessor W1 (text-free regions): a slot with no text
           capability at all never renders a text value/row, not even a
           "—" placeholder — showing one implied the profile reads text
           when it structurally doesn't. -->
        {#if slot.capabilities.text}
          <span class="min-w-0 truncate text-zinc-300" data-testid="slot-text-value"
            >{box?.text ?? '—'}</span
          >
        {/if}
        <span class="shrink-0 text-zinc-500">
          {box?.score != null ? `${(box.score * 100).toFixed(0)}%` : '—'}
        </span>
        {#if box?.textDisagreement}
          <span
            class="shrink-0 rounded border border-orange-500/40 bg-orange-500/15 px-1 text-[9px] font-sans text-orange-200"
            title="vlm: {box.textVlm ?? '∅'} · ocr: {box.textOcr ?? '∅'}"
          >
            readers disagree
          </span>
        {/if}
      </div>
      <div class="flex min-w-0 flex-wrap items-center gap-1">
        <ProvenanceChip
          detector={box?.detector ?? data.provenance?.detector ?? null}
          version={box?.detectorVersion ?? null}
          size="sm"
        />
        {#if data.provenance?.verifier}
          <ProvenanceChip detector={data.provenance.verifier} tag="verify" size="sm" />
        {/if}
        <!-- dq-region (2026-09-24): human-validated vs auto-confirmed
           (machine-accepted, unreviewed) — region_verified alone no
           longer distinguishes the two. -->
        {#if data.lifecycle?.validated}
          <span
            class="rounded border border-emerald-500/40 bg-emerald-500/15 px-1 text-[9px] text-emerald-200"
            title="A human confirmed/drew/rejected this region"
          >
            human
          </span>
        {:else if data.lifecycle?.autoConfirmed}
          <span
            class="rounded border border-blue-500/40 bg-blue-500/15 px-1 text-[9px] text-blue-200"
            title="Auto-confirmed by the worker's policy — accepted but not yet reviewed by a human"
          >
            auto
          </span>
        {/if}
        {#if crop.class_name}
          <span
            class="max-w-full truncate rounded border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-300"
            title={crop.class_name}
          >
            {crop.class_name}
          </span>
        {/if}
      </div>
      {#if !compact && slot.capabilities.provenance?.showChainOnCard && data.provenance?.chain && data.provenance.chain.length > 0}
        <div class="flex min-w-0 flex-wrap gap-0.5">
          {#each data.provenance.chain as entry, i (i)}
            <ProvenanceChip raw={entry} size="sm" />
          {/each}
        </div>
      {/if}
    </div>
  </button>
  {#if crop.image_id}
    <div
      class="absolute bottom-1 left-1 opacity-0 transition group-focus-within:opacity-100 group-hover:opacity-100"
      data-testid="card-reprocess-image"
    >
      <ReprocessControl
        target={{ kind: 'image', imageId: crop.image_id }}
        buttonLabel="Reprocess image…"
        buttonClass="btn btn-sm text-[10px]"
        onadopt={(items) => onreprocessed?.(items)}
      />
    </div>
  {/if}
</div>
