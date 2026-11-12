<script lang="ts">
  /**
   * Single-crop box-set editor (modal) for the given region slot
   * (w8 multibox frontend plan).
   *
   * Opens from CropCard's pencil button. Lets a curator add, move and
   * delete any number of boxes on top of the parent crop thumbnail (the
   * same `MultiBoxCanvas`/`multiBoxRegionController` the review page's
   * region tab uses), then saves the whole list with one
   * `PUT /crops/{id}/regions` (`frame: 'parent'`: the server does the
   * projection into the source frame, so this component never does).
   *
   * Hotkeys (focus inside the modal): the `box_edit` keymap actions via
   * the canvas, plus Enter to save and Escape to close without saving.
   *
   * `onsave` fires AFTER this component has already performed the write —
   * "notify", not "perform the save". It passes the server's own returned
   * item, so the caller renders what was actually persisted rather than
   * re-deriving it.
   */
  import type { SlotSpec } from '$lib/annotations/types';
  import { toastStore } from '$stores/toast.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import type { Crop } from '$lib/types';
  import { slotOf } from '$lib/annotations/cropSlots';
  import { toCanvasBoxes } from '$lib/annotations/multiBox';
  import MultiBoxCanvas from './MultiBoxCanvas.svelte';
  import { createMultiBoxRegionController } from '$lib/review/multiBoxRegionController.svelte';
  import { regionStatusesStore, toneRingRgb } from '$stores/regionStatuses.svelte';
  import Modal from '$components/Modal.svelte';

  const kg = (id: string) => keymapStore.compactGlyph(id);

  interface Props {
    crop: Crop;
    /** Slot whose sub-box this modal edits. */
    slot: SlotSpec;
    /** Called after a successful save (or clear) with the server's
     *  returned item. */
    onsave?: (item: Crop) => void;
    /** Called when the user dismisses without saving. */
    onclose: () => void;
    /** Optional thumbnail size override (px). Defaults to the active
     *  slot's own `capabilities.subBox.editor.thumbSize` (e.g. 512) —
     *  large enough for accurate hand-drawing. */
    thumbSize?: number;
  }

  let { crop, slot, onsave, onclose, thumbSize }: Props = $props();

  const activeSlot = $derived(slot);
  const editorThumbSize = $derived(
    thumbSize ?? activeSlot?.capabilities.subBox?.editor.thumbSize ?? 512,
  );

  // The served per-box `locked` flag, by box id, for the canvas's lock
  // glyph (the editable working set carries no `locked`).
  const lockedBoxIds = $derived<Set<string>>(
    new Set(
      (slotOf(crop, activeSlot)?.subBoxes ?? [])
        .filter((b) => b.locked === true && b.boxId != null)
        .map((b) => b.boxId as string),
    ),
  );

  const multiBox = createMultiBoxRegionController(() => activeSlot);
  $effect(() => {
    multiBox.seedFrom(crop);
  });

  function multiBoxRingColor(state: string): string {
    return toneRingRgb(regionStatusesStore.boxStateTone(state));
  }
  function multiBoxDashed(state: string): boolean {
    return (
      regionStatusesStore.boxStateInfo(state)?.dashed ??
      (state === 'rejected' || state === 'false_positive')
    );
  }
  function multiBoxStateLabel(state: string): string {
    return regionStatusesStore.boxStateInfo(state)?.label ?? state;
  }

  async function saveMultiBox(): Promise<void> {
    const { ok, item } = await multiBox.saveEdits(crop.id);
    if (ok && item) {
      toastStore.success(`${activeSlot.label.title} saved.`);
      onsave?.(item);
    }
  }

  let multiBoxCanvasEl = $state<MultiBoxCanvas | null>(null);

  function onKeyDown(e: KeyboardEvent): void {
    if (multiBox.busy) return;
    // Reuse the review page's own key handling (Tab/Backspace/arrows);
    // Enter/Escape map to Save/Cancel here, not the review queue's
    // "confirm" semantics.
    if (multiBoxCanvasEl?.handleKey(e)) {
      e.preventDefault();
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      void saveMultiBox();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onclose();
    }
  }
</script>

<svelte:window onkeydown={onKeyDown} />

<Modal
  label="Edit {activeSlot?.label.title ?? 'box'} bounding box"
  {onclose}
  layer="top"
  scrim="bg-black/85"
  panelClass="flex w-full max-w-3xl flex-col gap-3 rounded-lg border border-line bg-surface-0 p-4 shadow-2xl"
>
  <header class="flex items-baseline justify-between">
    <h3 class="text-base font-semibold text-fg-1">
      Edit {activeSlot?.label.singular ?? 'box'}
    </h3>
    <span class="font-mono text-caption text-fg-3">{crop.id}</span>
  </header>

  <MultiBoxCanvas
    bind:this={multiBoxCanvasEl}
    cropId={crop.id}
    boxes={toCanvasBoxes(multiBox.boxes, activeSlot.label.title, lockedBoxIds)}
    selectedIndex={multiBox.selectedIndex}
    busy={multiBox.busy}
    maxBoxes={multiBox.maxBoxes}
    ringColorFor={multiBoxRingColor}
    dashedFor={multiBoxDashed}
    thumbSize={editorThumbSize}
    onselect={(i) => multiBox.select(i)}
    onnext={() => multiBox.next()}
    onmove={(_i, box) => multiBox.moveSelected(box)}
    onadd={(box) => multiBox.addBox(box)}
    ondelete={() => multiBox.deleteSelected()}
  />
  <div class="mt-1 flex flex-wrap gap-1">
    {#each multiBox.boxes as b, i (b.boxId ?? `new-${i}`)}
      <button
        type="button"
        class="rounded border px-1.5 py-0.5 text-caption {i === multiBox.selectedIndex
          ? 'border-info/60 bg-info/15 text-info-text'
          : 'border-line-strong bg-surface-1 text-fg-2 hover:bg-surface-2'}"
        onclick={() => multiBox.select(i)}
      >
        #{i + 1}
        {multiBoxStateLabel(b.state)}
      </button>
    {/each}
    {#if multiBox.boxes.length === 0}
      <span class="text-caption text-fg-3">no boxes — drag to draw one</span>
    {/if}
    {#if multiBox.maxBoxes != null}
      <span class="text-caption text-fg-3"
        >{multiBox.boxes.length} / {multiBox.maxBoxes} max</span
      >
    {/if}
  </div>
  <!-- Footer: hotkey reference. -->
  <footer class="flex flex-col gap-1 text-caption text-fg-2">
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono">
      <span
        ><kbd class="rounded bg-surface-2 px-1">{kg('box_edit.next_box')}</kbd> next box</span
      >
      <span
        ><kbd class="rounded bg-surface-2 px-1">{kg('box_edit.delete_box')}</kbd> delete selected</span
      >
      <span
        ><kbd class="rounded bg-surface-2 px-1"
          >{kg('box_edit.nudge_left')}{kg('box_edit.nudge_up')}{kg(
            'box_edit.nudge_down',
          )}{kg('box_edit.nudge_right')}</kbd
        > nudge selected</span
      >
      <span><kbd class="rounded bg-surface-2 px-1">{kg('box_edit.save')}</kbd> save</span>
      <span
        ><kbd class="rounded bg-surface-2 px-1">{kg('box_edit.cancel')}</kbd> cancel</span
      >
    </div>
  </footer>
  <div class="flex items-center justify-end gap-2">
    <button
      type="button"
      class="rounded-md border border-line-strong px-3 py-1.5 text-sm text-fg-1 hover:bg-surface-2"
      onclick={onclose}
      disabled={multiBox.busy}
    >
      Cancel
    </button>
    <button
      type="button"
      class="rounded-md border border-accent/60 bg-accent/20 px-3 py-1.5 text-sm font-medium text-accent-text hover:bg-accent/30 disabled:opacity-50"
      onclick={saveMultiBox}
      disabled={multiBox.busy}
    >
      {multiBox.busy ? 'Saving…' : 'Save'}
    </button>
  </div>
</Modal>
