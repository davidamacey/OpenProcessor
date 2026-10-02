<script lang="ts">
  import { focusOnMount } from '$lib/actions/focusOnMount';
  /**
   * Single-crop box-set editor (modal) for the given region slot
   * (docs/design/w8-multibox-frontend-plan-2026-09-26.md).
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
  import MultiBoxCanvas from './MultiBoxCanvas.svelte';
  import { createMultiBoxRegionController } from '$lib/review/multiBoxRegionController.svelte';
  import { regionStatusesStore, toneRingRgb } from '$stores/regionStatuses.svelte';

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

<div
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
  role="dialog"
  aria-modal="true"
  aria-label="Edit {activeSlot?.label.title ?? 'box'} bounding box"
  use:focusOnMount
  tabindex="-1"
  onclick={(e) => {
    // Backdrop only: a click that bubbled up from the panel is not a
    // dismiss gesture.
    if (e.target === e.currentTarget) onclose();
  }}
  onkeydown={(e) => e.key === 'Escape' && onclose()}
>
  <div
    class="flex w-full max-w-3xl flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-950 p-4 shadow-2xl"
  >
    <header class="flex items-baseline justify-between">
      <h3 class="text-base font-semibold text-zinc-100">
        Edit {activeSlot?.label.singular ?? 'box'}
      </h3>
      <span class="font-mono text-[11px] text-zinc-500">{crop.id}</span>
    </header>

    <MultiBoxCanvas
      bind:this={multiBoxCanvasEl}
      cropId={crop.id}
      boxes={multiBox.boxes
        .filter((b) => b.box != null)
        .map((b) => ({
          box: b.box!,
          state: b.state,
          label: `${activeSlot.label.title} (${multiBoxStateLabel(b.state)})`,
        }))}
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
          class="rounded border px-1.5 py-0.5 text-[10px] {i === multiBox.selectedIndex
            ? 'border-sky-500/60 bg-sky-500/15 text-sky-100'
            : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
          onclick={() => multiBox.select(i)}
        >
          #{i + 1}
          {multiBoxStateLabel(b.state)}
        </button>
      {/each}
      {#if multiBox.boxes.length === 0}
        <span class="text-[11px] text-zinc-500">no boxes — drag to draw one</span>
      {/if}
      {#if multiBox.maxBoxes != null}
        <span class="text-[11px] text-zinc-500"
          >{multiBox.boxes.length} / {multiBox.maxBoxes} max</span
        >
      {/if}
    </div>
    <!-- Footer: hotkey reference. -->
    <footer class="flex flex-col gap-1 text-[11px] text-zinc-400">
      <div class="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono">
        <span
          ><kbd class="rounded bg-zinc-800 px-1">{kg('box_edit.next_box')}</kbd> next box</span
        >
        <span
          ><kbd class="rounded bg-zinc-800 px-1">{kg('box_edit.delete_box')}</kbd> delete selected</span
        >
        <span
          ><kbd class="rounded bg-zinc-800 px-1"
            >{kg('box_edit.nudge_left')}{kg('box_edit.nudge_up')}{kg(
              'box_edit.nudge_down',
            )}{kg('box_edit.nudge_right')}</kbd
          > nudge selected</span
        >
        <span><kbd class="rounded bg-zinc-800 px-1">{kg('box_edit.save')}</kbd> save</span
        >
        <span
          ><kbd class="rounded bg-zinc-800 px-1">{kg('box_edit.cancel')}</kbd> cancel</span
        >
      </div>
    </footer>
    <div class="flex items-center justify-end gap-2">
      <button
        type="button"
        class="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800"
        onclick={onclose}
        disabled={multiBox.busy}
      >
        Cancel
      </button>
      <button
        type="button"
        class="rounded-md border border-blue-500/60 bg-blue-500/20 px-3 py-1.5 text-sm font-medium text-blue-100 hover:bg-blue-500/30 disabled:opacity-50"
        onclick={saveMultiBox}
        disabled={multiBox.busy}
      >
        {multiBox.busy ? 'Saving…' : 'Save'}
      </button>
    </div>
  </div>
</div>
