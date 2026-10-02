<script lang="ts">
  import { sourceBadge } from '$lib/sourceBadge';
  import { sourceShortCode, vlmEmptyReasonText } from '$lib/cropCardText';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { getThumbUrl } from '$lib/api';
  import type { BBoxNorm, Crop } from '$lib/types';
  import { slotOf, subBoxSlotFor } from '$lib/annotations/cropSlots';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import { regionStatusesStore, toneBorderClass } from '$stores/regionStatuses.svelte';
  import SlotBboxEditor from './SlotBboxEditor.svelte';
  import SourceImageOverlay from './SourceImageOverlay.svelte';

  interface Props {
    crop: Crop;
    /** Slot whose sub-box/ring this card overlays and edits. Defaults to
     *  `subBoxSlotFor` over the registered slots; with no sub-box slot the
     *  card shows no ring and no ✎. */
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
    onslotsaved?: (cropId: string, item: Crop) => void;
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

  const activeSlot = $derived(slot ?? subBoxSlotFor(crop, slotRegistry.all));

  let expanded = $state<boolean>(false);
  let editorOpen = $state<boolean>(false);

  // Track the natural pixel size of the rendered thumbnail so we can
  // letterbox-compensate the region-ring overlay. The thumbnail is
  // served as a *non-square* JPEG (PIL `crop.thumbnail((size, size))`
  // preserves aspect ratio), but we render it inside an aspect-square
  // container with `object-contain`. That means a 200x600 tall, narrow
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

  // Every box to draw, already in the parent-crop frame (the server's own
  // projection, `SlotBox.parent`; a scalar-box slot's is projected once by
  // readSlot) — no second hand-rolled projection here. A multi-box slot
  // draws all of its boxes, each ringed by its served box-state tone.
  interface RingSource {
    box: BBoxNorm;
    colorClass: string;
    dashed: boolean;
  }
  const ringSources = $derived.by<RingSource[]>(() => {
    if (noSlot || !slotData) return [];
    if (slotData.subBoxes) {
      return slotData.subBoxes.flatMap((b) =>
        b.parent
          ? [
              {
                box: b.parent,
                colorClass: toneBorderClass(regionStatusesStore.boxStateTone(b.state)),
                dashed:
                  regionStatusesStore.boxStateInfo(b.state)?.dashed ??
                  (b.state === 'rejected' || b.state === 'false_positive'),
              },
            ]
          : [],
      );
    }
    return slotData.subBox?.parent
      ? [{ box: slotData.subBox.parent, colorClass: scalarRingColorClass, dashed: false }]
      : [];
  });

  // Letterbox-compensated ring rectangles (percent of the aspect-square
  // container). When natural dims aren't known yet (still loading), fall
  // back to naive container-relative placement so first paint isn't blank.
  const ringRects = $derived.by(() =>
    ringSources.map((src) => {
      const box = src.box;
      const x1 = box.cx - box.w / 2;
      const y1 = box.cy - box.h / 2;
      const w = box.w;
      const h = box.h;
      if (imgNaturalW <= 0 || imgNaturalH <= 0) {
        return {
          left: x1 * 100,
          top: y1 * 100,
          width: w * 100,
          height: h * 100,
          colorClass: src.colorClass,
          dashed: src.dashed,
        };
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
        colorClass: src.colorClass,
        dashed: src.dashed,
      };
    }),
  );

  // Scalar-box slot ring color: green when a human has confirmed the box
  // (verified === true), yellow for unverified machine-suggested boxes
  // (Finding C.3, docs/genericization-plan-2026-09-13.md §2.7: `verified`
  // — not any status value — is the confirmed-ring predicate).
  const scalarRingColorClass = $derived(
    slotVerified
      ? (activeSlot?.capabilities.subBox?.ring.confirmed ??
          'border-green-400 shadow-[0_0_0_1px_rgba(34,197,94,0.45)]')
      : (activeSlot?.capabilities.subBox?.ring.proposed ??
          'border-yellow-400 shadow-[0_0_0_1px_rgba(250,204,21,0.45)]'),
  );

  // Badge color + text reflect the ACTUAL source of the validated label.
  // Previously every validated crop showed a green 'human' chip — but
  // most crops in the ensemble pipeline are auto-validated by the VLM,
  // ensemble consensus, or cluster propagation; only true human labels
  // (label_source='human') get the green chip.
  const badge = $derived(
    sourceBadge(
      crop.label_source,
      // G2: class_validated, not label_validated — this badge is about
      // the class label, and label_validated also flips true on a
      // region-only validation.
      crop.class_validated,
      classSourcesStore.roleFor(crop.label_source),
      classSourcesStore.labelFor(crop.label_source),
    ),
  );

  const sourceRole = $derived(classSourcesStore.roleFor(crop.label_source));
  const hasClass = $derived(crop.class_id != null || !!crop.class_name);

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
    {:else if ringRects.length > 0}
      <!--
        Region rings overlaid on the thumbnail. The thumbnail is served as
        a non-square JPEG (aspect-preserved) and rendered with
        object-contain inside an aspect-square container. We have to
        letterbox-compensate the ring placement so it lands on the
        rendered image rect, not the empty letterbox bands. Math runs
        in CropCard once `<img onload>` has populated naturalWidth/Height.
        Pointer-events disabled so the ring never swallows card clicks.
      -->
      {#each ringRects as ring, i (i)}
        <div
          class="pointer-events-none absolute rounded-[2px] border {ring.colorClass} {ring.dashed
            ? 'border-dashed'
            : ''}"
          style:left="{ring.left}%"
          style:top="{ring.top}%"
          style:width="{ring.width}%"
          style:height="{ring.height}%"
          aria-hidden="true"
        ></div>
      {/each}
      <span
        class="absolute top-1 left-1 rounded-sm border border-blue-400/60 bg-blue-500/30 px-1 py-0.5 font-mono text-[10px] text-white"
      >
        {activeSlot?.label.singular ?? 'box'}
      </span>
    {/if}

    {#if activeSlot?.capabilities.subBox?.listField != null}
      <button
        type="button"
        class="absolute top-1 right-7 rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
        onclick={(e) => {
          e.stopPropagation();
          editorOpen = true;
        }}
        aria-label="Edit {activeSlot.label.singular}"
        title="Edit {activeSlot.label.singular} (✎)"
      >
        ✎
      </button>
    {/if}

    {#if ondetail}
      <button
        type="button"
        class="absolute top-1 right-14 rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
        onclick={(e) => {
          e.stopPropagation();
          ondetail?.(crop);
        }}
        aria-label="Show crop details"
        title="Details (provenance + metadata)"
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
        title="★1 = largest subject in its photo (#2 = second largest) · b = clarity score (higher is sharper)"
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

  <!-- K2/K3 (visual audit 2026-09-24): the class name gets the row's
       width — the source chip is a short role code with the served label
       in its tooltip, instead of "Labeled by the VLM" squeezing the class
       to "spor…". A crop with no class_id is shown as Unlabeled with no
       "labeled by" chip at all: the backend can leave label_source set
       (e.g. an unmatched VLM answer) on a crop that has no class. -->
  <div class="flex min-w-0 items-center gap-1 px-2 py-1.5">
    {#if hasClass}
      <span
        class="shrink-0 rounded-sm border px-1 py-0.5 font-mono text-[10px] font-medium {badge.cls}"
        title="Label source: {badge.text.replace(/ ·$/, '')}{badge.unvalidated
          ? ' — not yet validated'
          : ''}"
        data-testid="source-chip"
      >
        {sourceShortCode(sourceRole, badge.text.replace(/ ·$/, ''))}{badge.unvalidated
          ? '·'
          : ''}
      </span>
      <span
        class="min-w-0 grow truncate text-xs text-zinc-200"
        title={crop.class_name ?? ''}
        data-testid="class-name"
      >
        {crop.class_name ?? `class #${crop.class_id}`}
      </span>
    {:else}
      <span
        class="min-w-0 grow truncate text-xs text-amber-300/90"
        data-testid="class-name"
      >
        Unlabeled
      </span>
    {/if}
    {#if hasClass && crop.class_confidence != null}
      <span
        class="shrink-0 font-mono text-[10px] text-zinc-500"
        title="{crop.class_confidence_source === 'vlm'
          ? 'VLM'
          : crop.class_confidence_source === 'model'
            ? 'Model'
            : ''} label confidence"
      >
        {(crop.class_confidence * 100).toFixed(0)}%
      </span>
    {/if}
  </div>

  {#if crop.vlm_class_empty_reason}
    <div
      class="border-t border-zinc-800 bg-orange-500/5 px-2 py-1 text-[10px] text-orange-300"
    >
      {vlmEmptyReasonText(crop.vlm_class_empty_reason)}
    </div>
  {:else if crop.vlm_raw_class && crop.vlm_raw_class !== crop.class_name}
    <div
      class="border-t border-zinc-800 px-2 py-1 text-[10px] text-zinc-500"
      title={crop.vlm_raw_class}
    >
      VLM said: {crop.vlm_raw_class}
    </div>
  {/if}

  {#if crop.vlm_suggested_class_id != null && !crop.class_validated}
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
  {:else if crop.vlm_suggested_class_name && !crop.class_validated}
    <div
      class="border-t border-zinc-800 bg-yellow-500/5 px-2 py-1 text-xs text-yellow-200"
      title="The VLM proposed a class that isn't in the registry yet"
    >
      VLM suggests new class: {crop.vlm_suggested_class_name}
    </div>
  {/if}
</div>

{#if editorOpen && activeSlot}
  <SlotBboxEditor
    {crop}
    slot={activeSlot}
    onclose={() => (editorOpen = false)}
    onsave={(item) => {
      editorOpen = false;
      onslotsaved?.(crop.id, item);
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
      <SourceImageOverlay
        cropId={crop.id}
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
