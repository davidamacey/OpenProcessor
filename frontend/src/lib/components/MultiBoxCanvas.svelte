<script lang="ts">
  /**
   * Multi-box editing canvas for region slots (docs/design/
   * w8-multibox-frontend-plan-2026-09-26.md). Operates in the crop-local
   * (parent) frame. Renders
   * every box in `boxes`, numbered by its current list position (spec:
   * "not a stable id — use box_id", W8.7). Click selects a box; dragging
   * empty canvas draws a new one; Backspace/Delete removes the selected
   * box; Tab cycles the selection; arrow keys nudge the selected box.
   *
   * No client-GUESSED box cap (owner decision) — `boxes` is unbounded by
   * default. The only real limit is the served
   * `region_profile.limits.max_boxes_per_write` (`maxBoxes` prop, W8.8);
   * when set, drag-create is disabled once `boxes.length >= maxBoxes`
   * (never on a pre-W8.8 backend, which passes `null`). `onadd` /
   * `ondelete` / `onmove` / `onselect` let the parent own the actual
   * `EditableBox[]` state (via `annotations/multiBox.ts`'s pure helpers)
   * so this component stays presentation-only.
   */
  import { getThumbUrl } from '$lib/api';
  import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
  import type { BBoxNormLike } from '$lib/annotations/types';
  import { normalize as normalizeKey } from '$stores/keyboard.svelte';
  import { keymapStore } from '$stores/keymap.svelte';

  export interface CanvasBox {
    box: BBoxNormLike;
    state: string;
    label: string;
    /** The served per-box `locked` flag: a human-locked box the pipeline
     *  must not overwrite. Drawn as a lock glyph; no reason text (the
     *  backend serves none). */
    locked?: boolean;
  }

  interface Props {
    cropId: string;
    boxes: CanvasBox[];
    selectedIndex: number | null;
    thumbSize?: number;
    class?: string;
    busy?: boolean;
    /** Disables drag-create/drag-move (scan mode) — click-select and
     *  keyboard actions (Tab/y/r) still work. */
    readonly?: boolean;
    /** Served `region_profile.limits.max_boxes_per_write`, or `null`/
     *  `undefined` when unknown (no client-guessed default). */
    maxBoxes?: number | null;
    ringColorFor?: (state: string) => string;
    dashedFor?: (state: string) => boolean;
    onselect?: (index: number) => void;
    onmove?: (index: number, box: BBoxNormLike) => void;
    onadd?: (box: BBoxNormLike) => void;
    ondelete?: (index: number) => void;
    onnext?: () => void;
  }

  let {
    cropId,
    boxes,
    selectedIndex,
    thumbSize = 512,
    class: containerClass = 'aspect-square w-full',
    busy = false,
    readonly = false,
    maxBoxes = null,
    ringColorFor = () => 'rgb(80, 200, 255)',
    dashedFor = () => false,
    onselect,
    onmove,
    onadd,
    ondelete,
    onnext,
  }: Props = $props();

  $effect(() => {
    if (boxes.some((b) => b.locked)) void reprocessVocabularyStore.init();
  });

  let canvasEl = $state<HTMLDivElement | null>(null);
  // The thumbnail keeps its aspect ratio, so the image is letterboxed inside
  // the container. Boxes and pointer mapping live in a frame sized to the
  // image itself (1:1 until the image reports its natural size).
  let natural = $state<{ w: number; h: number } | null>(null);
  function onImgLoad(e: Event): void {
    const img = e.currentTarget as HTMLImageElement;
    natural =
      img.naturalWidth > 0 && img.naturalHeight > 0
        ? { w: img.naturalWidth, h: img.naturalHeight }
        : null;
  }
  const frameStyle = $derived(
    natural
      ? `aspect-ratio:${natural.w} / ${natural.h};${natural.w >= natural.h ? 'width:100%' : 'height:100%'}`
      : 'aspect-ratio:1 / 1;height:100%',
  );

  interface DragState {
    mode: 'create' | 'move';
    startX: number;
    startY: number;
    initial: BBoxNormLike | null;
  }
  let drag = $state<DragState | null>(null);

  function clamp01(x: number): number {
    return Math.min(1, Math.max(0, x));
  }

  function clientToNorm(e: PointerEvent): { x: number; y: number } {
    if (!canvasEl) return { x: 0, y: 0 };
    const rect = canvasEl.getBoundingClientRect();
    return {
      x: clamp01((e.clientX - rect.left) / (rect.width || 1)),
      y: clamp01((e.clientY - rect.top) / (rect.height || 1)),
    };
  }

  function normalize(b: BBoxNormLike): BBoxNormLike {
    const x1 = clamp01(b.cx - b.w / 2);
    const y1 = clamp01(b.cy - b.h / 2);
    const x2 = clamp01(b.cx + b.w / 2);
    const y2 = clamp01(b.cy + b.h / 2);
    return {
      cx: (x1 + x2) / 2,
      cy: (y1 + y2) / 2,
      w: Math.max(0, x2 - x1),
      h: Math.max(0, y2 - y1),
    };
  }

  const atBoxCap = $derived(maxBoxes != null && boxes.length >= maxBoxes);

  function onPointerDownCanvas(e: PointerEvent): void {
    if (busy || readonly || atBoxCap) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = clientToNorm(e);
    drag = { mode: 'create', startX: p.x, startY: p.y, initial: null };
  }

  function onPointerDownBox(e: PointerEvent, index: number): void {
    if (busy) return;
    e.preventDefault();
    e.stopPropagation();
    onselect?.(index);
    if (readonly) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = clientToNorm(e);
    drag = { mode: 'move', startX: p.x, startY: p.y, initial: { ...boxes[index].box } };
  }

  function onPointerMove(e: PointerEvent): void {
    if (!drag) return;
    const p = clientToNorm(e);
    if (drag.mode === 'create') {
      const x1 = Math.min(drag.startX, p.x);
      const y1 = Math.min(drag.startY, p.y);
      const x2 = Math.max(drag.startX, p.x);
      const y2 = Math.max(drag.startY, p.y);
      pendingCreate = normalize({
        cx: (x1 + x2) / 2,
        cy: (y1 + y2) / 2,
        w: x2 - x1,
        h: y2 - y1,
      });
      return;
    }
    if (drag.mode === 'move' && drag.initial != null && selectedIndex != null) {
      const dx = p.x - drag.startX;
      const dy = p.y - drag.startY;
      const moved = normalize({
        cx: drag.initial.cx + dx,
        cy: drag.initial.cy + dy,
        w: drag.initial.w,
        h: drag.initial.h,
      });
      onmove?.(selectedIndex, moved);
    }
  }

  let pendingCreate = $state<BBoxNormLike | null>(null);

  function onPointerUp(): void {
    if (
      drag?.mode === 'create' &&
      pendingCreate &&
      pendingCreate.w > 1e-6 &&
      pendingCreate.h > 1e-6
    ) {
      onadd?.(pendingCreate);
    }
    pendingCreate = null;
    drag = null;
  }

  const pxStep = $derived(1 / thumbSize);

  /** Nudges the selected box by one step in the given direction — the
   *  `box_edit.nudge_*` actions forward here (readonly/scan mode ignores
   *  this). */
  function nudgeSelected(dx: number, dy: number): void {
    if (readonly || busy || selectedIndex == null) return;
    const b = boxes[selectedIndex].box;
    onmove?.(selectedIndex, { cx: b.cx + dx, cy: b.cy + dy, w: b.w, h: b.h });
  }

  /** The `box_edit` action a keypress resolves to through the keymap, so a
   *  rebind applies here exactly as the printed hint says. The bare key is
   *  matched too: Shift or Ctrl held down never stopped a nudge. */
  function actionFor(e: KeyboardEvent): string | null {
    return (
      keymapStore.actionFor('box_edit', normalizeKey(e)) ??
      keymapStore.actionFor('box_edit', e.key.toLowerCase())
    );
  }

  /** Runs the `box_edit` action `e` resolves to; true when it was consumed. */
  export function handleKey(e: KeyboardEvent): boolean {
    if (busy) return false;
    switch (actionFor(e)) {
      case 'box_edit.next_box':
        onnext?.();
        return true;
      case 'box_edit.delete_box':
        if (selectedIndex == null) return false;
        ondelete?.(selectedIndex);
        return true;
      case 'box_edit.nudge_up':
        return nudgeKey(0, -pxStep);
      case 'box_edit.nudge_down':
        return nudgeKey(0, pxStep);
      case 'box_edit.nudge_left':
        return nudgeKey(-pxStep, 0);
      case 'box_edit.nudge_right':
        return nudgeKey(pxStep, 0);
      case 'box_edit.shrink_right':
        return moveRightEdge(-pxStep);
      case 'box_edit.grow_right':
        return moveRightEdge(pxStep);
      default:
        return false;
    }
  }

  /** Moves the selected box's right edge by `dw`, keeping the left edge
   *  fixed; the width never drops below one step. */
  function moveRightEdge(dw: number): boolean {
    if (readonly || selectedIndex == null) return false;
    const b = boxes[selectedIndex].box;
    const w = Math.max(pxStep, b.w + dw);
    onmove?.(selectedIndex, { cx: b.cx + (w - b.w) / 2, cy: b.cy, w, h: b.h });
    return true;
  }

  function nudgeKey(dx: number, dy: number): boolean {
    if (readonly || selectedIndex == null) return false;
    nudgeSelected(dx, dy);
    return true;
  }
</script>

<div
  class="relative flex touch-none items-center justify-center overflow-hidden rounded-md border border-zinc-800 bg-zinc-900 select-none {containerClass}"
>
  <div
    bind:this={canvasEl}
    class="relative max-h-full max-w-full"
    style={frameStyle}
    onpointerdown={onPointerDownCanvas}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
    role="application"
    aria-label="multi-box canvas"
    data-testid="multibox-canvas"
  >
    <img
      src={getThumbUrl(cropId, thumbSize)}
      alt="crop preview"
      draggable="false"
      class="pointer-events-none block h-full w-full object-fill"
      onload={onImgLoad}
    />

    {#each boxes as b, i (b.box.cx + ':' + b.box.cy + ':' + i)}
      {@const x1 = (b.box.cx - b.box.w / 2) * 100}
      {@const y1 = (b.box.cy - b.box.h / 2) * 100}
      <div
        class="absolute border-2"
        data-box-index={i}
        data-selected={selectedIndex === i}
        style="left:{x1}%;top:{y1}%;width:{b.box.w * 100}%;height:{b.box.h *
          100}%;border-color:{ringColorFor(b.state)};border-style:{dashedFor(b.state)
          ? 'dashed'
          : 'solid'};"
        onpointerdown={(e) => onPointerDownBox(e, i)}
        role="button"
        tabindex="-1"
        aria-label={b.label}
      >
        <span
          class="absolute -top-2 -left-2 flex h-4 w-4 items-center justify-center rounded-full bg-zinc-900 text-[10px] text-zinc-100"
          style="border:1px solid {ringColorFor(b.state)}"
        >
          {i + 1}
        </span>
        {#if b.locked}
          <span
            class="absolute -top-2 -right-2 flex h-4 w-4 items-center justify-center rounded-full bg-zinc-900 text-zinc-100"
            style="border:1px solid {ringColorFor(b.state)}"
            title={reprocessVocabularyStore.lockText()}
            data-testid="box-locked"
          >
            <svg
              viewBox="0 0 16 16"
              fill="currentColor"
              class="h-2.5 w-2.5"
              aria-hidden="true"
            >
              <path
                d="M8 1a3.5 3.5 0 0 0-3.5 3.5V6H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1h-.5V4.5A3.5 3.5 0 0 0 8 1Zm2 5H6V4.5a2 2 0 1 1 4 0V6Z"
              />
            </svg>
          </span>
        {/if}
      </div>
    {/each}

    {#if pendingCreate}
      <div
        class="pointer-events-none absolute border-2 border-dashed border-sky-400"
        style="left:{(pendingCreate.cx - pendingCreate.w / 2) *
          100}%;top:{(pendingCreate.cy - pendingCreate.h / 2) *
          100}%;width:{pendingCreate.w * 100}%;height:{pendingCreate.h * 100}%"
      ></div>
    {/if}

    {#if boxes.length === 0 && !pendingCreate}
      <span
        class="absolute top-2 left-2 rounded-sm border border-zinc-700 bg-zinc-900/80 px-1.5 py-0.5 text-[11px] text-zinc-300"
      >
        drag to draw a box
      </span>
    {/if}
  </div>
</div>
