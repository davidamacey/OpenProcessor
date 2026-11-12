<script lang="ts">
  /**
   * The picture and its boxes. Draws the boxes the backend served (`BoxLayer`) and reports finished
   * gestures: a drawn box (`ondraw`), a moved or resized one (`onmove`), a selection (`onselect`).
   * One gesture is one report: nothing is sent mid-drag. Pointer events are captured on the
   * canvas, so there is no handler on `window`; keys come from the keymap (`editorKeys`), and the
   * wheel zooms at the cursor.
   */
  import { untrack } from 'svelte';
  import BoxLayer from '$components/boxes/BoxLayer.svelte';
  import type { ImageBox } from '$lib/types_images';
  import type { XYXY } from '$lib/annotations/types';
  import {
    isDrawable,
    pointToNorm,
    RESIZE_HANDLES,
    resizeXyxy,
    translateXyxy,
    xyxyFromPoints,
    type ResizeHandle,
  } from '$lib/annotations/boxGeometry';
  import {
    actualSize,
    FIT,
    centerOn,
    panBy,
    viewportRect,
    visibleFraction,
    zoomAt,
    type Viewport,
  } from '$lib/editor/viewport';

  export type EditorTool = 'select' | 'draw' | 'pan';

  interface Props {
    /** The picture at about this many pixels on the long side. */
    imageUrl: (maxDim: number) => string;
    /** The served pixel size; absent until the browser reports the picture's own. */
    width: number | null;
    height: number | null;
    boxes: ImageBox[];
    selectedCropId: string | null;
    /** Items whose box a person has locked, with the served words for the lock. */
    locks: Readonly<Record<string, string | null>>;
    tool: EditorTool;
    showLabels: boolean;
    busy?: boolean;
    /** A drawn box waiting for its class: kept on screen while the picker is open. */
    pending?: XYXY | null;
    ondraw: (box: XYXY) => void;
    onmove: (cropId: string, box: XYXY) => void;
    onselect: (cropId: string | null) => void;
    /** True while a draw or a drag is under way (Escape cancels it first). */
    ongesture?: (active: boolean) => void;
    /** The zoom as a percentage of the picture's own pixels. */
    onzoom?: (percent: number) => void;
  }

  let {
    imageUrl,
    width,
    height,
    boxes,
    selectedCropId,
    locks,
    tool,
    showLabels,
    busy = false,
    pending = null,
    ondraw,
    onmove,
    onselect,
    ongesture,
    onzoom,
  }: Props = $props();

  const LEVELS = [1280, 2560, 4096, 8192];

  let container = $state<HTMLDivElement | null>(null);
  let frame = $state<HTMLDivElement | null>(null);
  let box = $state({ w: 0, h: 0 });
  let natural = $state<{ w: number; h: number } | null>(null);
  let vp = $state<Viewport>(FIT);
  let spaceHeld = $state(false);
  let loadedLevel = $state(0);

  const pixels = $derived(
    width && height ? { w: width, h: height } : (natural ?? { w: 4, h: 3 }),
  );
  const ratio = $derived(pixels.w / pixels.h);
  // The picture fitted inside the container, before any zoom.
  const fit = $derived.by(() => {
    const w = Math.min(box.w, box.h * ratio);
    return { width: w, height: w / ratio };
  });
  const rect = $derived(viewportRect(vp, fit, { width: box.w, height: box.h }));
  const percent = $derived(
    pixels.w > 0 && fit.width > 0 ? ((fit.width * vp.scale) / pixels.w) * 100 : 100,
  );
  // A photo more than four viewports large gets a small map to find one's way around it.
  const MINIMAP_AREA_FACTOR = 4;
  const showMinimap = $derived(
    box.w > 0 && pixels.w * pixels.h > MINIMAP_AREA_FACTOR * box.w * box.h,
  );
  const view = $derived(visibleFraction(rect, { width: box.w, height: box.h }));
  // Pull in a sharper copy once the picture is drawn larger than the one on screen; never a smaller one.
  const wantLevel = $derived.by(() => {
    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
    const need = Math.max(rect.width, rect.height) * dpr;
    const i = LEVELS.findIndex((l) => l >= need);
    return i === -1 ? LEVELS.length - 1 : i;
  });
  $effect(() => {
    const want = wantLevel;
    untrack(() => {
      if (want > loadedLevel) loadedLevel = want;
    });
  });
  $effect(() => {
    onzoom?.(percent);
  });

  function measure(): void {
    if (!container) return;
    box = { w: container.clientWidth, h: container.clientHeight };
  }
  $effect(() => {
    if (!container) return;
    measure();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null;
    ro?.observe(container);
    return () => ro?.disconnect();
  });

  // The wheel needs `preventDefault`, so it is attached non-passive on the canvas itself.
  function wheel(node: HTMLElement): () => void {
    function onwheel(e: WheelEvent): void {
      e.preventDefault();
      const lines = e.deltaMode === 1 ? 16 : 1;
      zoomWith(Math.exp(-e.deltaY * lines * 0.0015), cursorInContainer(e));
    }
    node.addEventListener('wheel', onwheel, { passive: false });
    return () => node.removeEventListener('wheel', onwheel);
  }

  function cursorInContainer(e: { clientX: number; clientY: number }): {
    x: number;
    y: number;
  } {
    const r = container?.getBoundingClientRect();
    const left = (box.w - fit.width) / 2;
    const top = (box.h - fit.height) / 2;
    return { x: e.clientX - (r?.left ?? 0) - left, y: e.clientY - (r?.top ?? 0) - top };
  }

  function panFromMinimap(e: PointerEvent): void {
    e.stopPropagation();
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const at = {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
    vp = centerOn(vp, at, fit, { width: box.w, height: box.h });
  }

  function zoomWith(factor: number, at: { x: number; y: number }): void {
    vp = zoomAt(vp, factor, at, fit);
  }

  /** Zooms by `factor` about the middle of the picture on screen. */
  export function zoomBy(factor: number): void {
    zoomWith(factor, { x: fit.width / 2, y: fit.height / 2 });
  }
  export function zoomFit(): void {
    vp = FIT;
  }
  export function zoomActual(): void {
    vp = actualSize(vp, pixels.w, { x: fit.width / 2, y: fit.height / 2 }, fit);
  }
  /** One screen pixel as a fraction of the picture, for nudging by pixels. */
  export function pixelStep(): { x: number; y: number } {
    return { x: 1 / Math.max(1, rect.width), y: 1 / Math.max(1, rect.height) };
  }

  // -- gestures ------------------------------------------------------------

  type Drag =
    | { kind: 'draw'; start: { x: number; y: number } }
    | { kind: 'move'; cropId: string; start: { x: number; y: number }; initial: XYXY }
    | {
        kind: 'resize';
        cropId: string;
        handle: ResizeHandle;
        start: { x: number; y: number };
        initial: XYXY;
      }
    | { kind: 'pan'; lastX: number; lastY: number };

  let drag = $state<Drag | null>(null);
  // The first corner of a box drawn with two clicks.
  let anchor = $state<{ x: number; y: number } | null>(null);
  let draft = $state<XYXY | null>(null);
  let live = $state<{ cropId: string; box: XYXY } | null>(null);

  $effect(() => {
    ongesture?.(drag !== null || anchor !== null);
  });
  // A half-drawn box belongs to the draw tool: leaving it drops the box.
  $effect(() => {
    if (tool !== 'draw') untrack(cancelGesture);
  });

  /** Drops a draw or drag under way without sending it. True when there was one. */
  export function cancelGesture(): boolean {
    const was = drag !== null || anchor !== null;
    drag = null;
    anchor = null;
    draft = null;
    live = null;
    return was;
  }

  function norm(e: PointerEvent): { x: number; y: number } {
    if (!frame) return { x: 0, y: 0 };
    return pointToNorm(e.clientX, e.clientY, frame.getBoundingClientRect());
  }

  const selectedBox = $derived(
    boxes.find((b) => b.kind === 'item' && b.crop_id === selectedCropId) ?? null,
  );
  const selectedLocked = $derived(selectedCropId !== null && selectedCropId in locks);
  const overlay = $derived(
    live && live.cropId === selectedCropId ? live.box : (selectedBox?.bbox_norm ?? null),
  );

  function onpointerdown(e: PointerEvent): void {
    if (busy || e.button > 0) return;
    if (tool === 'pan' || spaceHeld) {
      begin(e, { kind: 'pan', lastX: e.clientX, lastY: e.clientY });
    } else if (tool === 'draw') {
      const p = norm(e);
      if (anchor) finishDraw(xyxyFromPoints(anchor, p));
      else begin(e, { kind: 'draw', start: p });
    } else {
      selectDown(e, e.target as HTMLElement);
    }
  }

  /** A press with the select tool: a handle resizes, the box body moves, the empty picture deselects. */
  function selectDown(e: PointerEvent, target: HTMLElement): void {
    const onBox = target.closest('[data-move], [data-testid="overlay-box"]') !== null;
    if (selectedCropId === null || selectedBox === null || selectedLocked) {
      if (!onBox) onselect(null);
      return;
    }
    const grab = {
      cropId: selectedCropId,
      start: norm(e),
      initial: selectedBox.bbox_norm,
    };
    const handle = target.closest<HTMLElement>('[data-handle]')?.dataset.handle as
      ResizeHandle | undefined;
    if (handle) begin(e, { kind: 'resize', handle, ...grab });
    else if (target.closest('[data-move]')) begin(e, { kind: 'move', ...grab });
    else if (!onBox) onselect(null);
  }

  function begin(e: PointerEvent, next: Drag): void {
    container?.setPointerCapture(e.pointerId);
    container?.focus({ preventScroll: true });
    drag = next;
  }

  function onpointermove(e: PointerEvent): void {
    if (drag?.kind === 'pan') {
      vp = panBy(vp, e.clientX - drag.lastX, e.clientY - drag.lastY, fit);
      drag = { ...drag, lastX: e.clientX, lastY: e.clientY };
    } else if (drag?.kind === 'draw') {
      draft = xyxyFromPoints(drag.start, norm(e));
    } else if (drag?.kind === 'move') {
      const p = norm(e);
      live = {
        cropId: drag.cropId,
        box: translateXyxy(drag.initial, p.x - drag.start.x, p.y - drag.start.y),
      };
    } else if (drag?.kind === 'resize') {
      const p = norm(e);
      live = {
        cropId: drag.cropId,
        box: resizeXyxy(
          drag.initial,
          drag.handle,
          p.x - drag.start.x,
          p.y - drag.start.y,
        ),
      };
    } else if (anchor) {
      draft = xyxyFromPoints(anchor, norm(e));
    }
  }

  function finishDraw(box: XYXY): void {
    anchor = null;
    draft = null;
    drag = null;
    if (isDrawable(box)) ondraw(box);
  }

  function onpointerup(e: PointerEvent): void {
    const d = drag;
    drag = null;
    if (d?.kind === 'draw') {
      const box = xyxyFromPoints(d.start, norm(e));
      if (isDrawable(box)) {
        finishDraw(box);
      } else {
        // A click, not a drag: the second click finishes the box.
        anchor = d.start;
        draft = null;
      }
    } else if (d?.kind === 'move' || d?.kind === 'resize') {
      const done = live;
      live = null;
      const changed = done && done.box.some((v, i) => v !== d.initial[i]);
      if (done && changed) onmove(done.cropId, done.box);
    }
  }

  function onpointercancel(): void {
    drag = null;
    live = null;
    draft = null;
  }

  // Space pans while held and the canvas has focus; it is pointer state, not a keymap action.
  function onkeydown(e: KeyboardEvent): void {
    if (e.code === 'Space' || e.key === ' ') {
      spaceHeld = true;
      e.preventDefault();
    }
  }
  function onkeyup(e: KeyboardEvent): void {
    if (e.code === 'Space' || e.key === ' ') spaceHeld = false;
  }

  const cursor = $derived(
    tool === 'pan' || spaceHeld
      ? drag?.kind === 'pan'
        ? 'grabbing'
        : 'grab'
      : tool === 'draw'
        ? 'crosshair'
        : 'default',
  );

  function pct(n: number): string {
    return `${(n * 100).toFixed(3)}%`;
  }
  function boxStyle(b: XYXY): string {
    return `left:${pct(b[0])};top:${pct(b[1])};width:${pct(b[2] - b[0])};height:${pct(b[3] - b[1])};`;
  }
  function onImgLoad(e: Event): void {
    const img = e.currentTarget as HTMLImageElement;
    if (img.naturalWidth > 0 && img.naturalHeight > 0)
      natural = { w: img.naturalWidth, h: img.naturalHeight };
  }
</script>

<!-- A custom drawing widget: it takes focus so Space can pan, and every key action comes from the keymap. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
<div
  bind:this={container}
  {@attach wheel}
  class="relative h-full min-h-0 w-full touch-none overflow-hidden bg-surface-0 select-none focus-visible:ring-2 focus-visible:ring-info focus-visible:outline-none focus-visible:ring-inset"
  style="cursor:{cursor}"
  role="application"
  aria-label="Image editor canvas"
  tabindex="0"
  data-testid="editor-canvas"
  data-tool={tool}
  {onpointerdown}
  {onpointermove}
  {onpointerup}
  {onpointercancel}
  {onkeydown}
  {onkeyup}
>
  <div
    bind:this={frame}
    class="absolute"
    style="left:{rect.left}px;top:{rect.top}px;width:{rect.width}px;height:{rect.height}px"
    data-testid="editor-frame"
  >
    <img
      src={imageUrl(LEVELS[loadedLevel])}
      alt=""
      draggable="false"
      class="pointer-events-none block h-full w-full object-fill"
      onload={onImgLoad}
    />
    <BoxLayer
      {boxes}
      {selectedCropId}
      {showLabels}
      onselect={tool === 'select' && !busy ? (id) => onselect(id) : undefined}
    />

    {#if overlay && selectedCropId !== null && tool === 'select'}
      <div
        class="absolute z-20 border-2 border-dashed border-info {selectedLocked
          ? 'cursor-not-allowed'
          : 'cursor-move'}"
        style={boxStyle(overlay)}
        data-move
        data-testid="selected-box"
        title={selectedLocked ? (locks[selectedCropId] ?? 'Locked') : undefined}
      >
        {#if !selectedLocked}
          {#each Object.entries(RESIZE_HANDLES) as [name, h] (name)}
            <span
              class="absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-[1px] border border-line bg-info [@media(pointer:coarse)]:h-4 [@media(pointer:coarse)]:w-4"
              style="left:{h.x * 100}%;top:{h.y * 100}%;cursor:{name}-resize"
              data-handle={name}
              data-testid="resize-handle"
            ></span>
          {/each}
        {/if}
      </div>
    {/if}

    {#if pending}
      <div
        class="pointer-events-none absolute z-20 border-2 border-dashed border-info bg-info/20"
        style={boxStyle(pending)}
        data-testid="pending-box"
      ></div>
    {/if}

    {#if draft}
      <div
        class="pointer-events-none absolute z-20 border-2 border-dashed border-info bg-info/10"
        style={boxStyle(draft)}
        data-testid="draft-box"
      ></div>
    {/if}
  </div>

  {#if showMinimap}
    <!-- Pointer-only aid: panning has its own keys and the pan tool. -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="absolute right-2 bottom-2 z-30 w-40 cursor-pointer overflow-hidden rounded border border-line-strong bg-surface-1 shadow"
      style="aspect-ratio:{ratio}"
      data-testid="minimap"
      onpointerdown={panFromMinimap}
    >
      <img
        src={imageUrl(LEVELS[0])}
        alt=""
        draggable="false"
        class="pointer-events-none block h-full w-full object-fill"
      />
      <div class="pointer-events-none absolute inset-0">
        <BoxLayer {boxes} {selectedCropId} showLabels={false} />
      </div>
      <div
        class="pointer-events-none absolute border-2 border-info bg-info/25"
        style="left:{view.x * 100}%;top:{view.y * 100}%;width:{view.width *
          100}%;height:{view.height * 100}%"
        data-testid="minimap-view"
      ></div>
    </div>
  {/if}
</div>
