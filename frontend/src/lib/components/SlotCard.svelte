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
  import {
    API_PREFIX,
    getRegionThumbUrl,
    resolveApiUrl,
    type RegionBrowseItem,
  } from '$lib/api';
  import { readSlot } from '$lib/annotations/readSlot';
  import type { SlotSpec, XYXY } from '$lib/annotations/types';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';

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
  }

  let {
    crop,
    slot,
    onclick,
    onedit,
    onmarkfp,
    selected = false,
    compact = false,
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

  // false_positive regions stay visible (kept as hard negatives) but are
  // dimmed + badged so the operator sees the triage state at a glance.
  const isFalsePositive = $derived(
    slot.capabilities.lifecycle?.falsePositiveState != null &&
      data.lifecycle?.status === slot.capabilities.lifecycle.falsePositiveState,
  );

  // dq-region (2026-09-24): a verifier-rejected candidate box, kept for
  // human review/reversal — distinct from a false positive (which keeps
  // the promoted region box; this has no region box at all yet).
  const isRejectedCandidate = $derived(
    data.subBox?.rawXyxy == null && data.subBox?.candidate != null,
  );

  const thumbCap = $derived(slot.capabilities.subBox?.thumbnail);
  // crop.region_thumbnail_url, when present, is a server-provided,
  // cache-busted URL (see api.ts's region_thumbnail_url doc comment) —
  // it wins over a freshly-built one. Otherwise build from the active
  // slot's own thumbnail.path/defaultSize; a slot with no subBox
  // capability at all falls back to the generic region-thumbnail helper.
  const thumbUrl = $derived(
    crop.region_thumbnail_url
      ? resolveApiUrl(crop.region_thumbnail_url)
      : thumbCap
        ? resolveApiUrl(
            `${API_PREFIX}${thumbCap.path(crop.crop_id, thumbCap.defaultSize)}`,
          )
        : getRegionThumbUrl(crop.crop_id),
  );
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
  <div class="relative w-full bg-zinc-900" style="aspect-ratio: {thumbAspect}">
    <img
      src={thumbUrl}
      alt={slot.label.singular}
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
    {:else if isRejectedCandidate}
      {@const rejectionKind = regionVocabularyStore.rejectionReasonKind(
        data.lifecycle?.rejectionReason,
      )}
      <!-- 840beb8 adoption: badge color follows the served kind —
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
        title={data.lifecycle?.rejectionReason
          ? regionVocabularyStore.rejectionReasonLabel(data.lifecycle.rejectionReason)
          : undefined}
      >
        candidate
      </span>
    {/if}
  </div>
  <div class="flex flex-col gap-1 p-2 text-[11px]">
    <div class="flex items-center justify-between gap-1 font-mono">
      <span class="flex min-w-0 items-center gap-1 truncate text-zinc-300">
        <span class="truncate">{data.text?.value ?? '—'}</span>
        {#if data.text?.disagreement}
          <span
            class="shrink-0 rounded border border-orange-500/40 bg-orange-500/15 px-1 text-[9px] font-sans text-orange-200"
            title="vlm: {data.text.vlmValue ?? '∅'} · ocr: {data.text.ocrValue ?? '∅'}"
          >
            ⚠ disagree
          </span>
        {/if}
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
