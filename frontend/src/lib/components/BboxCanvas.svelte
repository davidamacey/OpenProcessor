<script lang="ts">
  /**
   * Reusable sub-bbox canvas — image + draggable/resizable ring for a
   * child bbox living inside a parent crop (e.g. a text region on an
   * item). No domain-specific logic lives here — this file is
   * domain-agnostic by construction (see
   * docs/genericization-plan-2026-09-13.md §3.1).
   *
   * Ring color defaults to sky-blue (RGB 80,200,255); override via
   * `ringColor` for a different slot's color scheme. K6 (docs/design/
   * k6-frontend-overlay-plan-2026-09-24.md): the backend no longer
   * burns a box overlay into {API_PREFIX}/crops/{id}/image — the
   * left-pane source preview now draws its own boxes client-side
   * (`SourceImageOverlay.svelte`, colored from the slot's own
   * `capabilities.subBox.ring`). This canvas only ever operates on the
   * crop's own thumbnail (`getThumbUrl`), never the full source image,
   * so it never needed the burned-in overlay itself — this comment
   * previously described color-matching against it, which is now moot.
   *
   * Operates in **crop-local frame** ([0, 1]^4 normalized inside the
   * parent bbox). Used both by:
   *   - SlotBboxEditor (modal, with Save/Clear/Cancel chrome)
   *   - /review's slot tab (inline, parent owns Confirm)
   *
   * Parent passes `bbox` (BBoxNorm | null) and gets `bbox` back via
   * bind:bbox. No save/network logic lives here.
   *
   * Pointer behavior:
   *   - empty canvas + drag → paint a fresh box
   *   - drag the box body → move
   *   - drag a corner / edge handle → resize
   *
   * Hotkeys (when the parent forwards them via the exported handler):
   *   ↑↓←→  move whole box by 1 px
   *   [ / ] nudge right edge in / out
   *   Backspace clear
   */
  import { getThumbUrl } from '$lib/api';
  import type { BBoxNorm } from '$lib/types';

  interface Props {
    /** Crop id — used to fetch the thumbnail image. */
    cropId: string;
    /** Sub-bbox in crop-local frame. Bind two-way. null = no box. */
    bbox: BBoxNorm | null;
    /** Optional thumbnail size override (px). */
    thumbSize?: number;
    /** Optional class for the outer container (sizing / aspect). */
    class?: string;
    /** Disable interaction (during a save). */
    busy?: boolean;
    /** Ring border color. Default matches the server-rendered region
     *  overlay (sky-blue, RGB 80,200,255) — override for another slot. */
    ringColor?: string;
    /**
     * Read-only display mode. Hides resize handles, ignores pointer
     * events, and uses a thinner ring. Used by /review's slot tab as
     * its default view so the bbox is shown but the canvas doesn't
     * sit on top of the crop with grabbable handles.
     */
    readonly?: boolean;
    /**
     * Optional viewport rectangle in **crop-local frame** (the same
     * space `bbox` lives in). When provided, the canvas zooms in on
     * this sub-region of the item crop so a small region fills the
     * visible area. `bbox` continues to be stored in crop-local frame
     * — saves and downstream consumers are unaffected. Pass `null` or
     * omit to render the full crop.
     */
    viewBox?: BBoxNorm | null;
    /** Human-facing name for the thing this box outlines — drives the
     *  aria-label and the empty-canvas hint. Every current caller passes
     *  the active slot's own label.title, so the default is only ever
     *  seen by a caller that hasn't been updated yet. */
    label?: string;
    /**
     * Draws the ring dashed instead of solid (dq-region, 2026-09-24) —
     * used for a verifier-rejected candidate box (the box hasn't been
     * accepted as the region yet; a human confirming it promotes it
     * server-side) so it reads as visually distinct from a confirmed or
     * proposed box without a whole second ring-color scheme.
     */
    dashed?: boolean;
  }

  let {
    cropId,
    bbox = $bindable(),
    thumbSize = 512,
    class: containerClass = 'aspect-square w-full',
    busy = false,
    readonly = false,
    ringColor = 'rgb(80, 200, 255)',
    viewBox = null,
    label = 'box',
    dashed = false,
  }: Props = $props();

  type DragMode = 'create' | 'move' | 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

  interface DragState {
    mode: DragMode;
    startX: number;
    startY: number;
    initialBox: BBoxNorm | null;
  }

  let canvasEl = $state<HTMLDivElement | null>(null);
  let drag = $state<DragState | null>(null);
  const pxStep = $derived(1 / thumbSize);

  // Natural dims of the thumbnail JPEG, captured on <img onload>. The
  // backend serves non-square JPEGs (aspect-preserved), so the
  // object-contain'd image inside an aspect-square container is
  // letterboxed. Pointer math + ring placement must compensate or the
  // bbox lands in the wrong spot for non-square crops (tall-and-narrow,
  // wide-and-short). dispRect describes the actual image rect inside the
  // unit-square canvas: {offX, offY, w, h} all in [0, 1].
  let imgNaturalW = $state<number>(0);
  let imgNaturalH = $state<number>(0);
  function onImgLoad(e: Event): void {
    const img = e.currentTarget as HTMLImageElement;
    imgNaturalW = img.naturalWidth || 0;
    imgNaturalH = img.naturalHeight || 0;
  }
  // baseDisp: where the IMG content naturally sits inside the unit-
  // square container with object-contain (image-aspect letterbox).
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

  // dispRect: where the IMG content visually maps in the container,
  // after applying the optional viewBox zoom. The combined transform
  // imgFrac → containerFrac is:
  //   containerFrac = (baseDisp.offset - viewBox.tl * scale) + imgFrac * scale
  //                    where scale = baseDisp.size / viewBox.size
  // When viewBox is null we fall straight through to baseDisp so the
  // existing non-zoomed callers (SlotBboxEditor modal) are unaffected.
  const dispRect = $derived.by(() => {
    if (!viewBox || viewBox.w <= 0 || viewBox.h <= 0) return baseDisp;
    const vx1 = viewBox.cx - viewBox.w / 2;
    const vy1 = viewBox.cy - viewBox.h / 2;
    const sx = baseDisp.w / viewBox.w;
    const sy = baseDisp.h / viewBox.h;
    return {
      offX: baseDisp.offX - vx1 * sx,
      offY: baseDisp.offY - vy1 * sy,
      w: sx,
      h: sy,
    };
  });

  // CSS transform string applied to the IMG element so the visible
  // image region matches `dispRect`. The IMG fills the container with
  // object-contain, so its natural image-content area sits at
  // baseDisp. Scaling around baseDisp's top-left and translating by
  // the offset delta yields dispRect.
  const imgTransform = $derived.by<string>(() => {
    if (!viewBox || viewBox.w <= 0 || viewBox.h <= 0) return 'none';
    const sx = dispRect.w / baseDisp.w;
    const sy = dispRect.h / baseDisp.h;
    const dx = (dispRect.offX - baseDisp.offX) * 100;
    const dy = (dispRect.offY - baseDisp.offY) * 100;
    return `translate(${dx}%, ${dy}%) scale(${sx}, ${sy})`;
  });
  const imgOrigin = $derived.by<string>(
    () => `${baseDisp.offX * 100}% ${baseDisp.offY * 100}%`,
  );

  function clamp01(x: number): number {
    return Math.min(1, Math.max(0, x));
  }

  /** Convert a container-fraction point to an image-fraction point. */
  function containerToImage(p: { x: number; y: number }): { x: number; y: number } {
    const { offX, offY, w, h } = dispRect;
    if (w <= 0 || h <= 0) return p;
    return {
      x: clamp01((p.x - offX) / w),
      y: clamp01((p.y - offY) / h),
    };
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

  function clientToNorm(e: PointerEvent | MouseEvent): { x: number; y: number } {
    if (!canvasEl) return { x: 0, y: 0 };
    const rect = canvasEl.getBoundingClientRect();
    const w = rect.width || 1;
    const h = rect.height || 1;
    // Pointer position in container-fraction, then mapped onto the
    // letterboxed image rect so the bbox we store is in image-fraction
    // (i.e. the same crop-local frame the parent expects).
    return containerToImage({
      x: clamp01((e.clientX - rect.left) / w),
      y: clamp01((e.clientY - rect.top) / h),
    });
  }

  function onPointerDownCanvas(e: PointerEvent): void {
    if (busy || readonly) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = clientToNorm(e);
    if (bbox == null) {
      bbox = { cx: p.x, cy: p.y, w: 0, h: 0 };
      drag = { mode: 'create', startX: p.x, startY: p.y, initialBox: null };
    } else {
      drag = {
        mode: 'move',
        startX: p.x,
        startY: p.y,
        initialBox: { ...bbox },
      };
    }
  }

  function onPointerDownHandle(e: PointerEvent, mode: DragMode): void {
    if (busy || readonly || bbox == null) return;
    e.preventDefault();
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = clientToNorm(e);
    drag = { mode, startX: p.x, startY: p.y, initialBox: { ...bbox } };
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
      bbox = normalizeBox({
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
      bbox = normalizeBox({ cx: ib.cx + dx, cy: ib.cy + dy, w: ib.w, h: ib.h });
      return;
    }
    let x1 = ib.cx - ib.w / 2;
    let y1 = ib.cy - ib.h / 2;
    let x2 = ib.cx + ib.w / 2;
    let y2 = ib.cy + ib.h / 2;
    if (drag.mode.includes('n')) y1 = clamp01(y1 + dy);
    if (drag.mode.includes('s')) y2 = clamp01(y2 + dy);
    if (drag.mode.includes('w')) x1 = clamp01(x1 + dx);
    if (drag.mode.includes('e')) x2 = clamp01(x2 + dx);
    const nx1 = Math.min(x1, x2);
    const nx2 = Math.max(x1, x2);
    const ny1 = Math.min(y1, y2);
    const ny2 = Math.max(y1, y2);
    bbox = normalizeBox({
      cx: (nx1 + nx2) / 2,
      cy: (ny1 + ny2) / 2,
      w: nx2 - nx1,
      h: ny2 - ny1,
    });
  }

  function onPointerUp(): void {
    if (!drag) return;
    if (bbox && (bbox.w < 1e-6 || bbox.h < 1e-6)) {
      bbox = null;
    }
    drag = null;
  }

  /** Public hotkey dispatcher — parent forwards keydown events here. */
  export function handleKey(e: KeyboardEvent): boolean {
    if (busy || readonly) return false;
    switch (e.key) {
      case 'Backspace':
        bbox = null;
        return true;
      case 'ArrowUp':
        nudgeBox(0, -pxStep);
        return true;
      case 'ArrowDown':
        nudgeBox(0, pxStep);
        return true;
      case 'ArrowLeft':
        nudgeBox(-pxStep, 0);
        return true;
      case 'ArrowRight':
        nudgeBox(pxStep, 0);
        return true;
      case '[':
        nudgeRightEdge(-pxStep);
        return true;
      case ']':
        nudgeRightEdge(pxStep);
        return true;
    }
    return false;
  }

  function nudgeBox(dx: number, dy: number): void {
    if (!bbox) return;
    bbox = normalizeBox({ cx: bbox.cx + dx, cy: bbox.cy + dy, w: bbox.w, h: bbox.h });
  }

  function nudgeRightEdge(dx: number): void {
    if (!bbox) return;
    let x1 = bbox.cx - bbox.w / 2;
    let x2 = clamp01(bbox.cx + bbox.w / 2 + dx);
    if (x2 < x1) {
      const t = x1;
      x1 = x2;
      x2 = t;
    }
    bbox = normalizeBox({
      cx: (x1 + x2) / 2,
      cy: bbox.cy,
      w: x2 - x1,
      h: bbox.h,
    });
  }

  const ringStyle = $derived.by<string>(() => {
    if (!bbox) return 'display:none';
    // Place the ring in container-fraction = dispRect.off + bbox * dispRect.size,
    // matching the inverse transform clientToNorm performs on input.
    const { offX, offY, w: dW, h: dH } = dispRect;
    const x1 = (offX + (bbox.cx - bbox.w / 2) * dW) * 100;
    const y1 = (offY + (bbox.cy - bbox.h / 2) * dH) * 100;
    const w = bbox.w * dW * 100;
    const h = bbox.h * dH * 100;
    return `left:${x1}%;top:${y1}%;width:${w}%;height:${h}%`;
  });
</script>

<div
  bind:this={canvasEl}
  class="relative overflow-hidden rounded-md border border-zinc-800 bg-zinc-900 select-none {readonly
    ? 'pointer-events-none'
    : 'touch-none'} {containerClass}"
  onpointerdown={onPointerDownCanvas}
  onpointermove={onPointerMove}
  onpointerup={onPointerUp}
  onpointercancel={onPointerUp}
  role={readonly ? 'img' : 'application'}
  aria-label={readonly ? `${label} bounding box (read-only)` : `${label} bbox canvas`}
>
  <img
    src={getThumbUrl(cropId, thumbSize)}
    alt="crop preview"
    draggable="false"
    onload={onImgLoad}
    class="pointer-events-none h-full w-full object-contain"
    style="transform: {imgTransform}; transform-origin: {imgOrigin};"
  />

  {#if bbox}
    <div
      class="absolute border-2"
      style="{ringStyle}; border-color: {ringColor}; border-style: {dashed
        ? 'dashed'
        : 'solid'}; background-color: {readonly
        ? 'transparent'
        : `color-mix(in srgb, ${ringColor} 12%, transparent)`};"
    >
      {#if !readonly}
        <div
          class="absolute inset-0 cursor-move"
          onpointerdown={(e) => onPointerDownHandle(e, 'move')}
          role="presentation"
        ></div>
        <div
          class="absolute -top-1 -left-1 h-2 w-2 cursor-nwse-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'nw')}
          role="presentation"
        ></div>
        <div
          class="absolute -top-1 -right-1 h-2 w-2 cursor-nesw-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'ne')}
          role="presentation"
        ></div>
        <div
          class="absolute -bottom-1 -left-1 h-2 w-2 cursor-nesw-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'sw')}
          role="presentation"
        ></div>
        <div
          class="absolute -right-1 -bottom-1 h-2 w-2 cursor-nwse-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'se')}
          role="presentation"
        ></div>
        <div
          class="absolute -top-1 left-1/2 h-2 w-2 -translate-x-1/2 cursor-ns-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'n')}
          role="presentation"
        ></div>
        <div
          class="absolute -bottom-1 left-1/2 h-2 w-2 -translate-x-1/2 cursor-ns-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 's')}
          role="presentation"
        ></div>
        <div
          class="absolute top-1/2 -left-1 h-2 w-2 -translate-y-1/2 cursor-ew-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'w')}
          role="presentation"
        ></div>
        <div
          class="absolute top-1/2 -right-1 h-2 w-2 -translate-y-1/2 cursor-ew-resize rounded-sm border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.7)]"
          style="background-color: {ringColor};"
          onpointerdown={(e) => onPointerDownHandle(e, 'e')}
          role="presentation"
        ></div>
      {/if}
    </div>
  {:else if !readonly}
    <span
      class="absolute top-2 left-2 rounded-sm border border-zinc-700 bg-zinc-900/80 px-1.5 py-0.5 text-[11px] text-zinc-300"
    >
      drag to draw a {label} box
    </span>
  {/if}
</div>
