<script lang="ts">
  /**
   * Single-crop plate editor (modal).
   *
   * Opens from CropCard's pencil button. Lets a curator draw, drag,
   * resize, and clear the plate sub-bbox on top of the vehicle crop
   * thumbnail, then saves it back to the API.
   *
   * Internally we work in the **crop's local frame** (normalized [0, 1]
   * inside the vehicle box) so that pointer math is independent of the
   * source image. On save we reconstruct the source-frame box via
   * `cropToSourceFrame` and PUT it as `[x1, y1, x2, y2]`.
   *
   * Hotkeys (focus inside the modal):
   *   [ / ]    nudge right edge in / out by 1 crop-pixel
   *   ↑↓←→     move whole box by 1 crop-pixel
   *   Backspace clear the box (saves as plate_status='no_plate_visible')
   *   Enter    save & advance
   *   Escape   close without saving
   */
  import { getThumbUrl, setCropPlate } from '$lib/api';
  import { bboxNormToXYXY, cropToSourceFrame, sourceToCropFrame } from '$lib/plate_geometry';
  import { toastStore } from '$stores/toast.svelte';
  import type { BBoxNorm, OpCrop } from '$lib/types';

  interface Props {
    crop: OpCrop;
    /** Called after a successful save (or clear). Passes the new
     *  source-frame plate bbox, or `null` if cleared. */
    onsave?: (plateBboxSrc: BBoxNorm | null) => void;
    /** Called when the user dismisses without saving. */
    onclose: () => void;
    /** Optional thumbnail size override (px). Default 512 — large enough
     *  for accurate hand-drawing on plates. */
    thumbSize?: number;
  }

  let { crop, onsave, onclose, thumbSize = 512 }: Props = $props();

  // -- state ------------------------------------------------------------
  // Plate box in the crop's local frame ([0, 1]^4). null means "no box".
  // We seed from the existing source-frame plate by projecting it into
  // crop frame; null seed is fine ("no plate yet").
  function seedPlate(): BBoxNorm | null {
    if (!crop.plate_bbox_norm || !crop.bbox_norm) return null;
    return sourceToCropFrame(crop.plate_bbox_norm, crop.bbox_norm);
  }

  let plateLocal = $state<BBoxNorm | null>(seedPlate());
  let busy = $state<boolean>(false);
  let errorText = $state<string | null>(null);

  // The DOM container we track pointer events on; pointer-x/y are
  // normalized against this rect. The image is rendered with object-
  // contain inside it so the crop fills the full square.
  let canvasEl = $state<HTMLDivElement | null>(null);

  // Natural dims of the thumbnail JPEG, captured on <img onload>. The
  // backend serves aspect-preserved JPEGs, so object-contain inside the
  // aspect-square canvas letterboxes non-square crops. Pointer math and
  // the ring must compensate or the box lands in the wrong spot (it
  // rendered too low for wide vehicle crops). baseDisp is the actual
  // image rect inside the unit-square canvas: {offX, offY, w, h} ∈ [0,1].
  // Mirrors PlateBboxCanvas.svelte's baseDisp.
  let imgNaturalW = $state<number>(0);
  let imgNaturalH = $state<number>(0);
  function onImgLoad(e: Event): void {
    const img = e.currentTarget as HTMLImageElement;
    imgNaturalW = img.naturalWidth || 0;
    imgNaturalH = img.naturalHeight || 0;
  }
  const baseDisp = $derived.by(() => {
    if (imgNaturalW <= 0 || imgNaturalH <= 0) {
      return { offX: 0, offY: 0, w: 1, h: 1 };
    }
    const aspect = imgNaturalW / imgNaturalH;
    if (aspect >= 1) {
      const h = 1 / aspect;
      return { offX: 0, offY: (1 - h) / 2, w: 1, h };
    }
    const w = aspect;
    return { offX: (1 - w) / 2, offY: 0, w, h: 1 };
  });

  // Drag state ---------------------------------------------------------
  type DragMode =
    | 'create'      // user dragging from empty canvas — paint a fresh box
    | 'move'        // dragging the whole box body
    | 'n' | 's' | 'e' | 'w'
    | 'ne' | 'nw' | 'se' | 'sw';

  interface DragState {
    mode: DragMode;
    startX: number;       // normalized [0,1] start point
    startY: number;
    initialBox: BBoxNorm | null; // box at drag-start (for move / resize math)
  }

  let drag = $state<DragState | null>(null);

  // Footer footer-text shows current source-frame coords for sanity.
  const sourceFrameSummary = $derived.by<string>(() => {
    if (plateLocal == null) return 'no plate';
    if (!crop.bbox_norm) return '(missing parent vehicle box)';
    const src = cropToSourceFrame(plateLocal, crop.bbox_norm);
    const [x1, y1, x2, y2] = bboxNormToXYXY(src);
    return `src [x1=${x1.toFixed(4)}, y1=${y1.toFixed(4)}, x2=${x2.toFixed(4)}, y2=${y2.toFixed(4)}]`;
  });

  const cropFrameSummary = $derived.by<string>(() => {
    if (plateLocal == null) return '';
    const [x1, y1, x2, y2] = bboxNormToXYXY(plateLocal);
    return `crop [x1=${x1.toFixed(4)}, y1=${y1.toFixed(4)}, x2=${x2.toFixed(4)}, y2=${y2.toFixed(4)}]`;
  });

  // 1 crop-pixel = 1 / displayed-pixel-width, normalized. We don't have
  // direct access to the crop's true pixel dimensions client-side, so
  // approximate via the thumbnail size (close enough for hotkey nudges).
  const pxStep = $derived(1 / thumbSize);

  // -- pointer math -----------------------------------------------------

  function clientToNorm(e: PointerEvent | MouseEvent): { x: number; y: number } {
    if (!canvasEl) return { x: 0, y: 0 };
    const rect = canvasEl.getBoundingClientRect();
    const w = rect.width || 1;
    const h = rect.height || 1;
    // Container-fraction of the pointer, then map onto the letterboxed
    // image rect so the stored box is in crop-local (image) frame.
    const cxf = Math.min(1, Math.max(0, (e.clientX - rect.left) / w));
    const cyf = Math.min(1, Math.max(0, (e.clientY - rect.top) / h));
    const { offX, offY, w: dW, h: dH } = baseDisp;
    if (dW <= 0 || dH <= 0) return { x: cxf, y: cyf };
    return {
      x: clamp01((cxf - offX) / dW),
      y: clamp01((cyf - offY) / dH),
    };
  }

  function clamp01(x: number): number {
    return Math.min(1, Math.max(0, x));
  }

  function normalizeBox(b: BBoxNorm): BBoxNorm {
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

  // -- pointer handlers -------------------------------------------------

  function onPointerDownCanvas(e: PointerEvent): void {
    if (busy) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = clientToNorm(e);
    if (plateLocal == null) {
      // Start painting a new box from this point.
      plateLocal = { cx: p.x, cy: p.y, w: 0, h: 0 };
      drag = { mode: 'create', startX: p.x, startY: p.y, initialBox: null };
    } else {
      // Click-on-body to drag-move.
      drag = {
        mode: 'move',
        startX: p.x,
        startY: p.y,
        initialBox: { ...plateLocal },
      };
    }
  }

  function onPointerDownHandle(e: PointerEvent, mode: DragMode): void {
    if (busy || plateLocal == null) return;
    e.preventDefault();
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = clientToNorm(e);
    drag = { mode, startX: p.x, startY: p.y, initialBox: { ...plateLocal } };
  }

  function onPointerMove(e: PointerEvent): void {
    if (!drag) return;
    const p = clientToNorm(e);
    const dx = p.x - drag.startX;
    const dy = p.y - drag.startY;
    if (drag.mode === 'create') {
      const x1 = Math.min(drag.startX, p.x);
      const y1 = Math.min(drag.startY, p.y);
      const x2 = Math.max(drag.startX, p.x);
      const y2 = Math.max(drag.startY, p.y);
      plateLocal = normalizeBox({
        cx: (x1 + x2) / 2,
        cy: (y1 + y2) / 2,
        w: x2 - x1,
        h: y2 - y1,
      });
      return;
    }
    if (!drag.initialBox) return;
    const ib = drag.initialBox;
    if (drag.mode === 'move') {
      plateLocal = normalizeBox({
        cx: ib.cx + dx,
        cy: ib.cy + dy,
        w: ib.w,
        h: ib.h,
      });
      return;
    }
    // Resize via edge / corner. Convert initial box to corners, then
    // shift the affected sides.
    let x1 = ib.cx - ib.w / 2;
    let y1 = ib.cy - ib.h / 2;
    let x2 = ib.cx + ib.w / 2;
    let y2 = ib.cy + ib.h / 2;
    if (drag.mode.includes('n')) y1 = clamp01(y1 + dy);
    if (drag.mode.includes('s')) y2 = clamp01(y2 + dy);
    if (drag.mode.includes('w')) x1 = clamp01(x1 + dx);
    if (drag.mode.includes('e')) x2 = clamp01(x2 + dx);
    // Keep min < max even if the user crosses over (negative size).
    const nx1 = Math.min(x1, x2);
    const nx2 = Math.max(x1, x2);
    const ny1 = Math.min(y1, y2);
    const ny2 = Math.max(y1, y2);
    plateLocal = normalizeBox({
      cx: (nx1 + nx2) / 2,
      cy: (ny1 + ny2) / 2,
      w: nx2 - nx1,
      h: ny2 - ny1,
    });
  }

  function onPointerUp(): void {
    if (!drag) return;
    // If the user clicked-without-drag on an empty canvas, plateLocal
    // ends up as zero-size — drop it so we don't "save" an invisible box.
    if (plateLocal && (plateLocal.w < 1e-6 || plateLocal.h < 1e-6)) {
      plateLocal = null;
    }
    drag = null;
  }

  // -- keyboard ---------------------------------------------------------

  function nudgeBox(dx: number, dy: number): void {
    if (!plateLocal) return;
    plateLocal = normalizeBox({
      cx: plateLocal.cx + dx,
      cy: plateLocal.cy + dy,
      w: plateLocal.w,
      h: plateLocal.h,
    });
  }

  function nudgeRightEdge(dx: number): void {
    if (!plateLocal) return;
    let x1 = plateLocal.cx - plateLocal.w / 2;
    let x2 = clamp01(plateLocal.cx + plateLocal.w / 2 + dx);
    if (x2 < x1) {
      const t = x1;
      x1 = x2;
      x2 = t;
    }
    plateLocal = normalizeBox({
      cx: (x1 + x2) / 2,
      cy: plateLocal.cy,
      w: x2 - x1,
      h: plateLocal.h,
    });
  }

  async function onKeyDown(e: KeyboardEvent): Promise<void> {
    if (busy) return;
    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        onclose();
        return;
      case 'Enter':
        e.preventDefault();
        await save();
        return;
      case 'Backspace':
        e.preventDefault();
        plateLocal = null;
        return;
      case 'ArrowUp':
        e.preventDefault();
        nudgeBox(0, -pxStep);
        return;
      case 'ArrowDown':
        e.preventDefault();
        nudgeBox(0, pxStep);
        return;
      case 'ArrowLeft':
        e.preventDefault();
        nudgeBox(-pxStep, 0);
        return;
      case 'ArrowRight':
        e.preventDefault();
        nudgeBox(pxStep, 0);
        return;
      case '[':
        e.preventDefault();
        nudgeRightEdge(-pxStep);
        return;
      case ']':
        e.preventDefault();
        nudgeRightEdge(pxStep);
        return;
    }
  }

  // -- save -------------------------------------------------------------

  async function save(): Promise<void> {
    if (busy) return;
    errorText = null;
    busy = true;
    try {
      // Clear: PUT null → backend writes plate_status='no_plate_visible'.
      if (plateLocal == null) {
        await setCropPlate(crop.id, null);
        toastStore.success('Plate cleared.');
        onsave?.(null);
        return;
      }
      if (!crop.bbox_norm) {
        throw new Error('Cannot save plate: parent vehicle bbox is missing.');
      }
      const sourceBox = cropToSourceFrame(plateLocal, crop.bbox_norm);
      const tuple = bboxNormToXYXY(sourceBox);
      await setCropPlate(crop.id, tuple);
      toastStore.success('Plate saved.');
      onsave?.(sourceBox);
    } catch (e) {
      errorText = (e as Error).message;
      toastStore.error(errorText ?? 'Plate save failed.');
    } finally {
      busy = false;
    }
  }

  // Derived overlay rectangle in % of the canvas.
  const ringStyle = $derived.by<string>(() => {
    if (!plateLocal) return 'display:none';
    // Place the ring inside the letterboxed image rect (inverse of the
    // map clientToNorm applies on input) so it lines up with the crop.
    const { offX, offY, w: dW, h: dH } = baseDisp;
    const x1 = (offX + (plateLocal.cx - plateLocal.w / 2) * dW) * 100;
    const y1 = (offY + (plateLocal.cy - plateLocal.h / 2) * dH) * 100;
    const w = plateLocal.w * dW * 100;
    const h = plateLocal.h * dH * 100;
    return `left:${x1}%;top:${y1}%;width:${w}%;height:${h}%`;
  });
</script>

<svelte:window onkeydown={onKeyDown} />

<div
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
  role="dialog"
  aria-modal="true"
  aria-label="Edit plate bounding box"
  tabindex="-1"
  onclick={onclose}
  onkeydown={(e) => e.key === 'Escape' && onclose()}
>
  <div
    class="flex w-full max-w-3xl flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-950 p-4 shadow-2xl"
    role="document"
    tabindex="-1"
    onclick={(e) => e.stopPropagation()}
    onkeydown={(e) => e.stopPropagation()}
  >
    <header class="flex items-baseline justify-between">
      <h3 class="text-base font-semibold text-zinc-100">Edit plate</h3>
      <span class="font-mono text-[11px] text-zinc-500">{crop.id}</span>
    </header>

    <!-- Canvas -->
    <div
      bind:this={canvasEl}
      class="relative aspect-square w-full overflow-hidden rounded-md border border-zinc-800 bg-zinc-900 select-none touch-none"
      onpointerdown={onPointerDownCanvas}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerUp}
      role="application"
      aria-label="Plate bbox canvas"
    >
      <img
        src={getThumbUrl(crop.id, thumbSize)}
        alt="crop preview"
        draggable="false"
        onload={onImgLoad}
        class="pointer-events-none h-full w-full object-contain"
      />

      {#if plateLocal}
        <!-- Plate ring + drag handles -->
        <div
          class="absolute border-2 border-yellow-400 bg-yellow-400/10"
          style={ringStyle}
        >
          <!-- body grab area: covers the full ring interior so onpointerdown on the box body initiates a move -->
          <div
            class="absolute inset-0 cursor-move"
            onpointerdown={(e) => onPointerDownHandle(e, 'move')}
            role="presentation"
          ></div>
          <!-- 4 corner handles -->
          <div
            class="absolute -top-1.5 -left-1.5 h-3 w-3 cursor-nwse-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'nw')}
            role="presentation"
          ></div>
          <div
            class="absolute -top-1.5 -right-1.5 h-3 w-3 cursor-nesw-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'ne')}
            role="presentation"
          ></div>
          <div
            class="absolute -bottom-1.5 -left-1.5 h-3 w-3 cursor-nesw-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'sw')}
            role="presentation"
          ></div>
          <div
            class="absolute -right-1.5 -bottom-1.5 h-3 w-3 cursor-nwse-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'se')}
            role="presentation"
          ></div>
          <!-- 4 edge handles -->
          <div
            class="absolute -top-1.5 left-1/2 h-3 w-3 -translate-x-1/2 cursor-ns-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'n')}
            role="presentation"
          ></div>
          <div
            class="absolute -bottom-1.5 left-1/2 h-3 w-3 -translate-x-1/2 cursor-ns-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 's')}
            role="presentation"
          ></div>
          <div
            class="absolute top-1/2 -left-1.5 h-3 w-3 -translate-y-1/2 cursor-ew-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'w')}
            role="presentation"
          ></div>
          <div
            class="absolute top-1/2 -right-1.5 h-3 w-3 -translate-y-1/2 cursor-ew-resize rounded-sm border border-yellow-300 bg-yellow-500"
            onpointerdown={(e) => onPointerDownHandle(e, 'e')}
            role="presentation"
          ></div>
        </div>
      {:else}
        <span
          class="absolute top-2 left-2 rounded-sm border border-zinc-700 bg-zinc-900/80 px-1.5 py-0.5 text-[11px] text-zinc-300"
        >
          drag to draw a plate box
        </span>
      {/if}
    </div>

    <!-- Footer: hotkey reference + coord summary -->
    <footer class="flex flex-col gap-1 text-[11px] text-zinc-400">
      <div class="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono">
        <span><kbd class="rounded bg-zinc-800 px-1">[</kbd>/<kbd class="rounded bg-zinc-800 px-1">]</kbd> right edge</span>
        <span><kbd class="rounded bg-zinc-800 px-1">←↑↓→</kbd> move</span>
        <span><kbd class="rounded bg-zinc-800 px-1">⌫</kbd> clear</span>
        <span><kbd class="rounded bg-zinc-800 px-1">↵</kbd> save</span>
        <span><kbd class="rounded bg-zinc-800 px-1">Esc</kbd> cancel</span>
      </div>
      <div class="font-mono text-[11px] text-zinc-500">
        {sourceFrameSummary}
      </div>
      {#if cropFrameSummary}
        <div class="font-mono text-[11px] text-zinc-600">{cropFrameSummary}</div>
      {/if}
      {#if errorText}
        <div class="text-red-300">{errorText}</div>
      {/if}
    </footer>

    <div class="flex items-center justify-end gap-2">
      <button
        type="button"
        class="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800"
        onclick={() => (plateLocal = null)}
        disabled={busy || plateLocal == null}
      >
        Clear
      </button>
      <button
        type="button"
        class="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800"
        onclick={onclose}
        disabled={busy}
      >
        Cancel
      </button>
      <button
        type="button"
        class="rounded-md border border-blue-500/60 bg-blue-500/20 px-3 py-1.5 text-sm font-medium text-blue-100 hover:bg-blue-500/30 disabled:opacity-50"
        onclick={save}
        disabled={busy}
      >
        {busy ? 'Saving…' : 'Save'}
      </button>
    </div>
  </div>
</div>
