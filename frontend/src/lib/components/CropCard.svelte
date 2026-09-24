<script lang="ts">
  import { sourceBadge } from '$lib/sourceBadge';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { getThumbUrl, getSourceImageWithBbox } from '$lib/api';
  import type { BBoxNorm, Crop } from '$lib/types';
  import { slotOf } from '$lib/annotations/cropSlots';
  import { slotForClassName } from '$lib/annotations/registeredSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import SlotBboxEditor from './SlotBboxEditor.svelte';

  interface Props {
    crop: Crop;
    /** Slot whose sub-box/ring this card overlays. Defaults to whatever
     *  slot is bound to the crop's own class — every current call site
     *  relies on that default rather than passing one explicitly. */
    slot?: SlotSpec;
    selected?: boolean;
    onclick?: (crop: Crop, e: MouseEvent) => void;
    onacceptVlm?: (crop: Crop) => void;
    onrejectVlm?: (crop: Crop) => void;
    /**
     * Optional callback fired after the slot editor saves a new
     * source-frame sub-box (or null for "not visible"). Lets the page
     * update local state without a full reload.
     */
    onslotsaved?: (cropId: string, boxSrc: BBoxNorm | null) => void;
    /**
     * Optional "open details" hook — wires the per-card "ⓘ" affordance
     * to a parent-owned CropDetailModal so the cluster grid can show the
     * same provenance metadata the review page does.
     */
    ondetail?: (crop: Crop) => void;
  }

  let {
    crop,
    slot,
    selected = false,
    onclick,
    onacceptVlm,
    onrejectVlm,
    onslotsaved,
    ondetail,
  }: Props = $props();

  const activeSlot = $derived(slot ?? slotForClassName(crop.class_name));

  let expanded = $state<boolean>(false);
  let plateEditorOpen = $state<boolean>(false);

  // Track the natural pixel size of the rendered thumbnail so we can
  // letterbox-compensate the plate-ring overlay. The thumbnail is
  // served as a *non-square* JPEG (PIL `crop.thumbnail((size, size))`
  // preserves aspect ratio), but we render it inside an aspect-square
  // container with `object-contain`. That means a 200x600 motorcycle
  // crop sits in a vertical band centered in a square cell — and a
  // ring positioned as a percentage of the *container* lands in the
  // wrong spot. We measure the natural size on load, then place the
  // ring relative to the actual rendered image rect.
  let imgNaturalW = $state<number>(0);
  let imgNaturalH = $state<number>(0);
  function onImgLoad(e: Event): void {
    const img = e.currentTarget as HTMLImageElement;
    imgNaturalW = img.naturalWidth || 0;
    imgNaturalH = img.naturalHeight || 0;
  }

  // Slot data for the active slot, read off the already-mapped crop —
  // never re-derived via readSlot() (§2.4 of the plan: a Crop's boxes
  // are BBoxNorm, not the XYXY readSlot expects).
  const slotData = $derived(slotOf(crop, activeSlot));

  // The predicate for "a human has confirmed this box" is the boolean
  // `verified` field, not a status string (see Finding C.3,
  // docs/genericization-plan-2026-09-13.md §2.7 / §3.8).
  const slotVerified = $derived<boolean>(!!slotData?.lifecycle?.verified);

  const noSlot = $derived(
    activeSlot?.capabilities.lifecycle != null &&
      slotData?.lifecycle?.status === activeSlot.capabilities.lifecycle.rejectState,
  );

  // The sub-box, already projected into the parent-crop frame by
  // readSlot's own projection at mapping time — no second hand-rolled
  // projection here.
  const plateInCrop = $derived<BBoxNorm | null>(
    noSlot ? null : (slotData?.subBox?.parent ?? null),
  );

  // Letterbox-compensated ring rectangle (percent of the aspect-square
  // container). When natural dims aren't known yet (still loading), fall
  // back to naive container-relative placement so first paint isn't blank.
  const ringRectPct = $derived.by<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(() => {
    if (!plateInCrop) return null;
    const x1 = plateInCrop.cx - plateInCrop.w / 2;
    const y1 = plateInCrop.cy - plateInCrop.h / 2;
    const w = plateInCrop.w;
    const h = plateInCrop.h;
    if (imgNaturalW <= 0 || imgNaturalH <= 0) {
      return { left: x1 * 100, top: y1 * 100, width: w * 100, height: h * 100 };
    }
    const aspect = imgNaturalW / imgNaturalH;
    let dispW = 1;
    let dispH = 1;
    let offX = 0;
    let offY = 0;
    if (aspect >= 1) {
      dispH = 1 / aspect;
      offY = (1 - dispH) / 2;
    } else {
      dispW = aspect;
      offX = (1 - dispW) / 2;
    }
    return {
      left: (offX + x1 * dispW) * 100,
      top: (offY + y1 * dispH) * 100,
      width: w * dispW * 100,
      height: h * dispH * 100,
    };
  });

  // Ring color: green when a human has confirmed the sub-box (verified
  // === true), yellow for unverified machine-suggested boxes. Mirrors
  // §11.3 of the design doc, and Finding C.3
  // (docs/genericization-plan-2026-09-13.md §2.7): `verified` — not any
  // status value — is the correct predicate for the confirmed ring.
  const plateRingColorClass = $derived(
    slotVerified
      ? (activeSlot?.capabilities.subBox?.ring.confirmed ??
          'border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]')
      : (activeSlot?.capabilities.subBox?.ring.proposed ??
          'border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]'),
  );

  // Badge color + text reflect the ACTUAL source of the validated label.
  // Previously every validated crop showed a green 'human' chip — but
  // most crops in the ensemble pipeline are auto-validated by Gemma,
  // ensemble consensus, or cluster propagation; only true human labels
  // (label_source='human') get the green chip.
  const badge = $derived(
    sourceBadge(
      crop.label_source,
      crop.label_validated,
      classSourcesStore.roleFor(crop.label_source),
      classSourcesStore.labelFor(crop.label_source),
    ),
  );

  const conf = $derived(
    crop.label_confidence != null ? `${(crop.label_confidence * 100).toFixed(0)}%` : null,
  );

  // Primary-subject rank + blur chip — tells the operator why a crop is in
  // or out of the size/clarity filters. rank 1 = largest in its photo.
  const rank = $derived<number | null>(crop.crop_rank_in_image ?? null);
  const blurRatio = $derived<number | null>(crop.blur_lap_ratio ?? null);
  const rankBlurLabel = $derived.by<string | null>(() => {
    const parts: string[] = [];
    if (rank != null) parts.push(rank === 1 ? '★1' : `#${rank}`);
    if (blurRatio != null) parts.push(`b${blurRatio.toFixed(2)}`);
    return parts.length ? parts.join(' ') : null;
  });
</script>

<div
  role="button"
  tabindex="0"
  style="content-visibility:auto;contain-intrinsic-size:auto 260px"
  class="group relative flex flex-col rounded-md border bg-zinc-900 text-left transition focus:outline-none {selected
    ? 'border-blue-500 ring-2 ring-blue-500/40'
    : 'border-zinc-800 hover:border-zinc-600'}"
  onclick={(e) => onclick?.(crop, e)}
  onkeydown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onclick?.(crop, e as unknown as MouseEvent);
    }
  }}
  aria-pressed={selected}
>
  <div class="relative aspect-square w-full overflow-hidden rounded-t-md bg-zinc-950">
    <img
      src={getThumbUrl(crop.id)}
      alt="crop {crop.id}"
      loading="lazy"
      draggable="false"
      class="h-full w-full object-contain [-webkit-user-drag:none]"
      onload={onImgLoad}
      onerror={(e) => {
        const t = e.currentTarget as HTMLImageElement;
        t.style.opacity = '0.2';
      }}
    />

    {#if noSlot}
      <span
        class="absolute top-1 left-1 rounded-sm border border-zinc-500/60 bg-zinc-700/70 px-1 py-0.5 font-mono text-[10px] text-zinc-200"
        title="{activeSlot?.label.title ??
          'Slot'} marked as not visible by a human reviewer"
      >
        no {activeSlot?.label.singular ?? 'box'}
      </span>
    {:else if ringRectPct}
      <!--
        Plate ring overlaid on the thumbnail. The thumbnail is served as
        a non-square JPEG (aspect-preserved) and rendered with
        object-contain inside an aspect-square container. We have to
        letterbox-compensate the ring placement so it lands on the
        rendered image rect, not the empty letterbox bands. Math runs
        in CropCard once `<img onload>` has populated naturalWidth/Height.
        Pointer-events disabled so the ring never swallows card clicks.
      -->
      <div
        class="pointer-events-none absolute rounded-[2px] border {plateRingColorClass}"
        style:left="{ringRectPct.left}%"
        style:top="{ringRectPct.top}%"
        style:width="{ringRectPct.width}%"
        style:height="{ringRectPct.height}%"
        aria-hidden="true"
      ></div>
      <span
        class="absolute top-1 left-1 rounded-sm border border-blue-400/60 bg-blue-500/30 px-1 py-0.5 font-mono text-[10px] text-white"
      >
        {activeSlot?.label.singular ?? 'box'}
      </span>
    {/if}

    <button
      type="button"
      class="absolute top-1 right-7 rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
      onclick={(e) => {
        e.stopPropagation();
        plateEditorOpen = true;
      }}
      aria-label="Edit {activeSlot?.label.singular ?? 'box'}"
      title="Edit {activeSlot?.label.singular ?? 'box'} (✎)"
    >
      ✎
    </button>

    {#if ondetail}
      <button
        type="button"
        class="absolute top-1 right-14 rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
        onclick={(e) => {
          e.stopPropagation();
          ondetail?.(crop);
        }}
        aria-label="Show crop details"
        title="Details (provenance + plate metadata)"
      >
        ⓘ
      </button>
    {/if}

    {#if conf}
      <span
        class="absolute right-1 bottom-1 rounded-sm bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white"
      >
        {conf}
      </span>
    {/if}

    {#if rankBlurLabel}
      <span
        class="absolute bottom-1 left-1 rounded-sm bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-zinc-200"
        title="rank in photo (★1 = largest) · blur_lap_ratio (higher = clearer)"
      >
        {rankBlurLabel}
      </span>
    {/if}

    <button
      type="button"
      class="absolute top-1 right-1 rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
      onclick={(e) => {
        e.stopPropagation();
        expanded = true;
      }}
      aria-label="Expand"
    >
      view
    </button>
  </div>

  <div class="flex items-center gap-1 px-2 py-1.5">
    <span
      class="truncate rounded-sm border px-1 py-0.5 text-[10px] font-medium {badge.cls}"
      title={crop.class_name ?? 'unlabeled'}
    >
      {badge.text}
    </span>
    <span class="grow truncate text-xs text-zinc-300" title={crop.class_name ?? ''}>
      {crop.class_name ?? '—'}
    </span>
  </div>

  {#if crop.vlm_suggested_class_id != null && !crop.label_validated}
    <div
      class="flex items-center gap-1 border-t border-zinc-800 bg-yellow-500/5 px-2 py-1 text-xs"
    >
      <span
        class="grow truncate text-yellow-200"
        title={crop.vlm_suggested_class_name ?? ''}
      >
        VLM: {crop.vlm_suggested_class_name ?? '—'}
        {#if crop.vlm_confidence}
          <span class="ml-1 text-yellow-400/70">{crop.vlm_confidence}</span>
        {/if}
      </span>
      <button
        type="button"
        class="rounded border border-green-500/40 bg-green-500/20 px-1 text-green-200 hover:bg-green-500/30"
        onclick={(e) => {
          e.stopPropagation();
          onacceptVlm?.(crop);
        }}
        aria-label="Accept VLM suggestion"
      >
        ✓
      </button>
      <button
        type="button"
        class="rounded border border-red-500/40 bg-red-500/20 px-1 text-red-200 hover:bg-red-500/30"
        onclick={(e) => {
          e.stopPropagation();
          onrejectVlm?.(crop);
        }}
        aria-label="Reject VLM suggestion"
      >
        ×
      </button>
    </div>
  {:else if crop.vlm_suggested_class_name && !crop.label_validated}
    <div
      class="border-t border-zinc-800 bg-yellow-500/5 px-2 py-1 text-xs text-yellow-200"
      title="The VLM proposed a class that isn't in the registry yet"
    >
      VLM suggests new class: {crop.vlm_suggested_class_name}
    </div>
  {/if}
</div>

{#if plateEditorOpen}
  <SlotBboxEditor
    {crop}
    slot={activeSlot}
    onclose={() => (plateEditorOpen = false)}
    onsave={(plateSrc) => {
      plateEditorOpen = false;
      onslotsaved?.(crop.id, plateSrc);
    }}
  />
{/if}

{#if expanded}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Source image"
  >
    <div class="relative max-h-full max-w-6xl">
      <img
        src={getSourceImageWithBbox(crop.id)}
        alt="source"
        class="max-h-[85vh] max-w-full rounded-md border border-zinc-700"
      />
      <button
        type="button"
        class="absolute -top-3 -right-3 rounded-full border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-white"
        onclick={() => (expanded = false)}
        aria-label="Close"
      >
        ×
      </button>
    </div>
  </div>
{/if}
