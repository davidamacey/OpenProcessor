<!--
  The /review item panes: the source image, the crop (or the multi-box
  canvas with its box chips), and the scrolling metadata pane under it with
  the item rows, the slot panel or the queue actions, the class strip and
  the Details disclosure.
-->
<script lang="ts">
  import MultiBoxCanvas from '$lib/components/MultiBoxCanvas.svelte';
  import SourceImageOverlay from '$lib/components/SourceImageOverlay.svelte';
  import ReviewClassButtons from '$lib/components/review/ReviewClassButtons.svelte';
  import ReviewContextPanel from '$lib/components/review/ReviewContextPanel.svelte';
  import ReviewItemMeta from '$lib/components/review/ReviewItemMeta.svelte';
  import ReviewSlotPanel from '$lib/components/review/ReviewSlotPanel.svelte';
  import VectorRefreshNotice from '$lib/components/review/VectorRefreshNotice.svelte';
  import { getThumbUrl } from '$lib/api_cropRead';
  import { slotOf } from '$lib/annotations/cropSlots';
  import type { ReviewClassPicker } from '$lib/review/reviewClassPicker.svelte';
  import type { ReviewItemActions } from '$lib/review/reviewItemActions.svelte';
  import { kg } from '$lib/review/reviewKeys';
  import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
  import type { ReviewSlotActions } from '$lib/review/reviewSlotActions.svelte';
  import type { ReviewSlotEditController } from '$lib/review/reviewSlotEditController.svelte';
  import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
  import { isSlotTab } from '$lib/reviewTabs';
  import type { ReviewItem } from '$lib/types_review';

  interface Props {
    current: ReviewItem;
    tabs: ReviewTabsController;
    loader: ReviewQueueLoader;
    actions: ReviewItemActions;
    picker: ReviewClassPicker;
    edit: ReviewSlotEditController;
    slot: ReviewSlotActions;
    detailsOpen: boolean;
  }

  let {
    current,
    tabs,
    loader,
    actions,
    picker,
    edit,
    slot,
    detailsOpen = $bindable(),
  }: Props = $props();

  const tab = $derived(tabs.tab);
  const activeSlot = $derived(tabs.activeSlot);
  const queue = $derived(loader.queue);
  const multiBox = $derived(edit.multiBox);
  const isMultiBoxSlot = $derived(edit.isMultiBoxSlot);
  const editMode = $derived(edit.editMode);
  const slotLabels = $derived(edit.slotLabels);
  const cropDisplayStyle = $derived(edit.cropDisplayStyle);
  const multiBoxRingColor = $derived(edit.multiBoxRingColor);
  const multiBoxDashed = $derived(edit.multiBoxDashed);
  const multiBoxStateLabel = $derived(edit.multiBoxStateLabel);
  const canConfirm = $derived(actions.canConfirm);
  const confirmAndAdvance = $derived(actions.confirmAndAdvance);
  const skip = $derived(actions.skip);
  const discard = $derived(actions.discard);
  const undoLast = $derived(actions.undoLast);

  // the served per-box `locked` flag, by box id, for the canvas's
  // lock glyph (the editable working set carries no `locked`).
  const lockedBoxIds = $derived<Set<string>>(
    new Set(
      (activeSlot && current ? (slotOf(current, activeSlot)?.subBoxes ?? []) : [])
        .filter((b) => b.locked === true && b.boxId != null)
        .map((b) => b.boxId as string),
    ),
  );
</script>

<!-- Source image with bbox -->
<div
  class="flex h-[35vh] flex-col surface p-2 lg:h-auto lg:min-h-0"
  data-testid="review-source-panel"
>
  <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
    <span>source</span>
    <span class="grow"></span>
    <span class="font-mono">{current.source ?? ''}</span>
  </div>
  <!-- V-3: top-aligned, so on a tall pane the image sits at the top
       rather than mid-way down an empty black panel. -->
  <div class="flex min-h-0 flex-1 items-start justify-center bg-zinc-950">
    <!-- K6: boxes/labels are drawn client-side from
         GET {API_PREFIX}/crops/{id}/context — the server no longer
         burns an overlay into this image. -->
    <SourceImageOverlay cropId={current.id} maxDim={1280} align="start" />
  </div>
</div>

<!-- Crop + meta -->
<div class="flex flex-col surface p-2 lg:min-h-0" data-testid="review-crop-panel">
  <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
    <span>crop</span>
    <span class="grow"></span>
    <span class="font-mono">{current.id.slice(0, 12)}…</span>
  </div>
  <!-- M2/M12 (2026-09-24 interactive pass): shrink-0 + a floor
       height so this panel never collapses toward 0px when the
       content below it (Details, slot fields) grows past the
       column's fixed height — that content scrolls in its own
       region instead of squeezing the image.
       DQ-M5: `max-h-[40%]` is the other half of the fix — a ceiling
       on top of that floor, so a tall/tiny crop's `h-full` fill (the
       phase-A p9 "upscale to fit" behavior) can no longer consume
       the whole panel and push Reason/Proposed/Confirm-Skip-Discard
       off-screen. The floor itself came down from M2/M12's original
       300px to 210px in the same pass: at a real 1280×720 viewport
       the 300px floor alone (before any tall crop even enters the
       picture) already pushed the action buttons 3px past the
       bottom edge — e2e/stubbed/test_review_crop_viewport.py
       pins this exact budget. 210px keeps a region sub-box legible
       (the box canvas is still square-aspect within it) while fitting.
       Verified at 1280×720, 1600×1000 and 1920×1080 — see
       artifacts_local/cw-live/phase-b-fixes/. -->
  <!-- F8 D4: overflow-hidden so a region canvas never paints over the
       first metadata row (the Reason row read half-clipped). -->
  <div
    class="flex h-[210px] shrink-0 items-center justify-center overflow-hidden bg-zinc-950 lg:h-auto lg:max-h-[40%] lg:min-h-[210px]"
  >
    {#if isMultiBoxSlot && activeSlot}
      <!-- Multi-box:
           every box drawn at once, numbered by list position. Same
           select/add/delete/Tab interaction in scan and edit mode;
           only edit mode allows geometry changes (readonly canvas
           in scan). -->
      <MultiBoxCanvas
        bind:this={edit.slotCanvas}
        cropId={current.id}
        boxes={multiBox.boxes
          .filter((b) => b.box != null)
          .map((b) => ({
            box: b.box!,
            state: b.state,
            label: `${activeSlot!.label.title} (${multiBoxStateLabel(b.state)})`,
            locked: b.boxId != null && lockedBoxIds.has(b.boxId),
          }))}
        selectedIndex={multiBox.selectedIndex}
        busy={multiBox.busy}
        readonly={!editMode}
        maxBoxes={multiBox.maxBoxes}
        ringColorFor={multiBoxRingColor}
        dashedFor={multiBoxDashed}
        onselect={(i) => multiBox.select(i)}
        onnext={() => multiBox.next()}
        onmove={(_i, box) => multiBox.moveSelected(box)}
        onadd={(box) => multiBox.addBox(box)}
        ondelete={() => multiBox.deleteSelected()}
        class="aspect-square w-auto h-full max-h-full min-w-0 max-w-full"
      />
    {:else}
      <!-- p9 (2026-09-24 interactive pass): `max-h-full max-w-full`
           only ever shrinks — a thumbnail smaller than its
           container (common at 1920, where this panel stretches
           to ~900px tall but the served thumb is a few hundred px)
           rendered at its tiny natural size instead of upscaling
           to fill the space. `h-full w-full` + object-contain fills
           the container either direction, still preserving aspect
           ratio.
           DQ-M5: unbounded, that fill upscaled a 98×106 crop to
           598-918px tall depending on viewport, pushing the actions
           below it off-screen — this is a regression from p9, which
           fit width but not height. The container above is now
           height-capped (`max-h-[40%]`); `cropDisplayStyle` (inline,
           from capCropDisplayStyle()) is the second half — it bounds
           the *rendered* size to at most CROP_UPSCALE_CAP× the
           crop's own natural pixels, read back via
           bind:naturalWidth/naturalHeight, so a tiny crop no longer
           blows up to fill whatever room the container has even
           when that room is generous. Empty until the image has
           loaded (natural size unknown), during which `h-full
           w-full` still applies as the pre-p9 fallback. -->
      <img
        src={getThumbUrl(current.id, 384)}
        alt="crop"
        loading="lazy"
        decoding="async"
        bind:naturalWidth={edit.cropNaturalWidth}
        bind:naturalHeight={edit.cropNaturalHeight}
        style={cropDisplayStyle}
        class="h-full w-full object-contain"
      />
    {/if}
  </div>

  {#if isMultiBoxSlot && activeSlot}
    <!-- One state chip per box, click to select (mirrors
         MultiBoxCanvas's numbered rings). The served
         region_set_complete === false means the VLM reported
         visible regions that are missing from the list. -->
    <div class="mt-2 flex flex-wrap gap-1" data-testid="multibox-chips">
      {#each multiBox.boxes as b, i (b.boxId ?? `new-${i}`)}
        <button
          type="button"
          class="chip {i === multiBox.selectedIndex
            ? 'border-sky-500/60 bg-sky-500/15 text-sky-100'
            : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
          onclick={() => multiBox.select(i)}
        >
          #{i + 1}
          {multiBoxStateLabel(b.state)}
        </button>
      {/each}
      {#if multiBox.boxes.length === 0}
        <span class="text-[11px] text-zinc-500">no boxes</span>
      {/if}
      {#if current && slotOf(current, activeSlot)?.boxSet?.setComplete === false}
        <span
          class="chip border-amber-500/40 bg-amber-500/15 text-amber-200"
          title="The VLM reported visible regions that are missing from this list"
          data-testid="set-incomplete"
        >
          set incomplete
        </span>
      {/if}
      {#if multiBox.maxBoxes != null}
        <!-- The served region_profile.limits.max_boxes_per_write —
             never a client-guessed cap. -->
        <span
          class="text-[11px] text-zinc-500"
          title="Served limit on boxes per write (region_profile.limits.max_boxes_per_write)"
        >
          {multiBox.boxes.length} / {multiBox.maxBoxes} max
        </span>
      {/if}
    </div>
    {#if current}
      <VectorRefreshNotice cropId={current.id} refresh={multiBox.vectorRefresh} />
    {/if}
  {/if}

  <!-- Everything below the image scrolls in its own region — the
       image above keeps its floor height regardless of how much
       metadata/Details content is open. -->
  <div
    class="mt-3 pr-1 lg:min-h-0 lg:flex-1 lg:overflow-y-auto"
    data-testid="review-meta-pane"
  >
    <ReviewItemMeta
      {current}
      {activeSlot}
      {isMultiBoxSlot}
      opinion={actions.opinion}
      searchScores={loader.searchScores}
      acceptModelClass={actions.acceptModelClass}
    />

    {#if activeSlot && slotLabels}
      <ReviewSlotPanel {current} {activeSlot} {slotLabels} {edit} {slot} {skip} />
    {:else}
      <div
        class="sticky bottom-0 z-10 mt-3 flex flex-wrap gap-2 bg-[rgb(var(--bg-elevated))] py-1"
        data-testid="review-actions"
      >
        <button
          class="btn btn-primary"
          type="button"
          onclick={confirmAndAdvance}
          disabled={!canConfirm}
          title={canConfirm
            ? undefined
            : `No proposed class on this item — press ${kg('review.queue.class_picker')} or ${kg('review.queue.confirm')} to search.`}
        >
          Confirm
        </button>
        <button class="btn" type="button" onclick={skip}>Skip</button>
        <button class="btn btn-danger" type="button" onclick={discard}>Discard</button>
        <button class="btn" type="button" onclick={undoLast}>Undo</button>
      </div>
    {/if}

    <!-- Most-validated classes — click to label OR press the per-class
       hotkey configured on /classes. Hotkey badges only show for
       classes the user has explicitly bound (otherwise the strip is
       still clickable, just no kbd hint). The class strip is hidden
       on the regions tab; class assignment isn't relevant there. -->
    {#if !isSlotTab(tab)}
      <ReviewClassButtons
        topClasses={actions.topClasses}
        assign={actions.assign}
        openPicker={picker.openPicker}
      />
    {/if}

    <ReviewContextPanel {current} {queue} bind:detailsOpen />
  </div>
</div>
