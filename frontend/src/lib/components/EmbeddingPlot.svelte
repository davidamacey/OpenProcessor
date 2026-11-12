<script lang="ts">
  /**
   * The 2-d embedding scatter of the Workspace's Embedding view. It draws what
   * `GET /viz/projection` serves: every projected point, dimmed when the shared filter leaves
   * it out, coloured by the served class legend. It decorates the data and never decides a
   * class or cluster; coordinates are only ever read (the fit is a background job).
   *
   * Pointer and canvas only, no window/document key listeners:
   * - drag: lasso-select the matching points (the selection is the Workspace's)
   * - Shift+drag, or the "Draw a box" button then a drag: a box that becomes a `viz_box` filter
   * - click a dot: focus its item (the details panel shows it)
   */
  import type { Snippet } from 'svelte';
  import { untrack } from 'svelte';
  import { createVizRebuildJob } from '$lib/vizRebuildJob.svelte';
  import { apiErrorText } from '$lib/api_errors';
  import { getVizProjection, type VizLegendEntry, type VizPoint } from '$lib/api_viz';
  import {
    colourForIndex,
    computeScale,
    dragToBox,
    embeddingCoverageSuffix,
    nearestPointId,
    pointAlpha,
    selectIdsInLasso,
  } from '$lib/embeddingPlot';
  import type { ScreenPoint } from '$lib/embeddingPlot';
  import type { VizBox } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ItemFilterQuery } from '$lib/types_itemFilter';
  import EmbeddingSelectedPreview from './EmbeddingSelectedPreview.svelte';

  interface Props {
    /** The shared filter, queue and scope as the Workspace spells them. */
    query: ItemFilterQuery;
    maxPoints?: number;
    /** The applied box filter, drawn on the plot. */
    vizBox: VizBox | null;
    /** The crop ids the Workspace has selected. */
    selectedIds: readonly string[];
    /** True when `/methods` reports this overlay at the banner-required purity tier. */
    bannerRequired?: boolean;
    /** The pool size the fit was computed over (`/methods` `field_coverage_total`). */
    coveragePoolTotal?: number | null;
    onselect: (cropIds: string[]) => void;
    onbox: (box: VizBox) => void;
    onfocus: (cropId: string) => void;
    /** Shown when a projection exists but no item has a point. */
    empty?: Snippet;
  }

  let {
    query,
    maxPoints = 4000,
    vizBox,
    selectedIds,
    bannerRequired = false,
    coveragePoolTotal = null,
    onselect,
    onbox,
    onfocus,
    empty,
  }: Props = $props();

  const PLOT_HEIGHT = 520;
  const POINT_RADIUS = 3;
  const SELECTED_RADIUS = 4.5;
  const CLICK_RADIUS = 8;
  /** A box drag shorter than this (px) is a click, not a box. */
  const MIN_BOX_PX = 6;

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let containerWidth = $state<number>(800);

  let points = $state<VizPoint[]>([]);
  let legend = $state<VizLegendEntry[]>([]);
  let built = $state<boolean>(true);
  let loaded = $state<boolean>(false);
  let loading = $state<boolean>(true);
  let loadError = $state<string | null>(null);

  let dragging = $state<boolean>(false);
  let dragMode = $state<'lasso' | 'box'>('lasso');
  let boxArmed = $state<boolean>(false);
  let lassoPath = $state<ScreenPoint[]>([]);
  let dragStart: ScreenPoint | null = null;
  let dragEnd = $state<ScreenPoint | null>(null);

  let seq = 0;
  let inflight: AbortController | null = null;

  async function load(): Promise<void> {
    const mine = ++seq;
    inflight?.abort();
    const ctrl = new AbortController();
    inflight = ctrl;
    loading = true;
    loadError = null;
    try {
      const res = await getVizProjection(query, maxPoints, ctrl.signal);
      if (mine !== seq) return;
      points = res.points;
      legend = res.legend;
      built = res.built;
    } catch (e) {
      if (mine !== seq) return;
      points = [];
      legend = [];
      loadError = apiErrorText(e);
    } finally {
      if (mine === seq) {
        loading = false;
        loaded = true;
      }
    }
  }

  /** Reads the projection again (after a write changed classes, or a rebuild finished). */
  export function reload(): Promise<void> {
    return load();
  }

  const queryKey = $derived(JSON.stringify(query));
  $effect(() => {
    void queryKey;
    void maxPoints;
    untrack(() => void load());
    return () => inflight?.abort();
  });

  const selected = $derived(new Set(selectedIds));

  const scale = $derived.by(() =>
    computeScale(points, Math.max(1, containerWidth), PLOT_HEIGHT, 20),
  );

  const scaledPoints = $derived.by(() =>
    points.map((p) => ({ ...scale.toScreen(p.x, p.y), id: p.crop_id })),
  );
  const matchedIds = $derived(
    new Set(points.filter((p) => p.matched).map((p) => p.crop_id)),
  );
  const selectedPoints = $derived(points.filter((p) => selected.has(p.crop_id)));

  /** The pointer position in canvas pixels (the canvas has no CSS scaling). */
  function canvasPoint(e: PointerEvent): ScreenPoint {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function onPointerDown(e: PointerEvent): void {
    if (e.button !== 0) return;
    const at = canvasPoint(e);
    dragging = true;
    dragMode = e.shiftKey || boxArmed ? 'box' : 'lasso';
    dragStart = at;
    dragEnd = at;
    lassoPath = [at];
  }

  function onPointerMove(e: PointerEvent): void {
    if (!dragging) return;
    const at = canvasPoint(e);
    if (dragMode === 'box') dragEnd = at;
    else lassoPath = [...lassoPath, at];
  }

  function finishDrag(): void {
    if (!dragging) return;
    dragging = false;
    const start = dragStart;
    const end = dragEnd ?? start;
    if (dragMode === 'box') {
      if (start && end && Math.hypot(end.x - start.x, end.y - start.y) >= MIN_BOX_PX) {
        boxArmed = false;
        onbox(dragToBox(scale, start, end));
      }
    } else if (lassoPath.length > 2) {
      const inside = selectIdsInLasso(scaledPoints, lassoPath);
      onselect(inside.filter((id) => matchedIds.has(id)));
    } else if (start) {
      const id = nearestPointId(scaledPoints, start, CLICK_RADIUS);
      if (id) onfocus(id);
    }
    lassoPath = [];
    dragStart = null;
    dragEnd = null;
  }

  function draw(): void {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.fillStyle = '#09090b'; // zinc-950, the app's dark base
    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);

    const anySelected = selected.size > 0;
    for (const [i, p] of scaledPoints.entries()) {
      const src = points[i]!;
      const isSelected = selected.has(p.id);
      ctx.beginPath();
      ctx.arc(p.x, p.y, isSelected ? SELECTED_RADIUS : POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = colourForIndex(src.colour_index);
      ctx.globalAlpha = pointAlpha(src.matched, isSelected, anySelected);
      ctx.fill();
      if (isSelected) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#f4f4f5'; // zinc-100 ring
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    if (vizBox) {
      const a = scale.toScreen(vizBox[0], vizBox[3]);
      const b = scale.toScreen(vizBox[2], vizBox[1]);
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.95)'; // yellow-400
      ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.setLineDash([]);
    }

    if (dragging && dragMode === 'lasso' && lassoPath.length > 1) {
      ctx.beginPath();
      ctx.moveTo(lassoPath[0]!.x, lassoPath[0]!.y);
      for (const pt of lassoPath.slice(1)) ctx.lineTo(pt.x, pt.y);
      ctx.strokeStyle = 'rgba(96, 165, 250, 0.9)'; // blue-400
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = 'rgba(96, 165, 250, 0.12)';
      ctx.fill();
    }
    if (dragging && dragMode === 'box' && dragStart && dragEnd) {
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.9)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        dragStart.x,
        dragStart.y,
        dragEnd.x - dragStart.x,
        dragEnd.y - dragStart.y,
      );
      ctx.setLineDash([]);
    }
  }

  $effect(() => {
    void scaledPoints;
    void selected;
    void lassoPath;
    void dragEnd;
    void vizBox;
    void containerWidth;
    if (canvasEl) {
      canvasEl.width = Math.max(1, Math.round(containerWidth));
      canvasEl.height = PLOT_HEIGHT;
    }
    draw();
  });

  const rebuild = createVizRebuildJob(load);
</script>

<div class="flex flex-col gap-2" data-testid="embedding-plot">
  <div class="flex flex-wrap items-center gap-2 text-meta text-fg-2">
    {#if built && points.length > 0}
      <button
        type="button"
        class="btn-sm"
        aria-pressed={boxArmed}
        data-testid="embedding-box-tool"
        title="Drag a box on the plot to filter every view to the items inside it (Shift+drag does the same)"
        onclick={() => (boxArmed = !boxArmed)}
      >
        Draw a box
      </button>
      <span class="text-fg-3"
        >Drag to select, Shift+drag for a box, click a dot to open it.</span
      >
    {/if}
    <span class="grow"></span>
    {#if rebuild.job}
      <span>
        Rebuilding projection{rebuild.job.n_pool
          ? ` — ${rebuild.job.n_written.toLocaleString()} / ${rebuild.job.n_pool.toLocaleString()} points`
          : '…'}
      </span>
      <button type="button" class="btn-sm" onclick={() => void rebuild.cancel()}>
        Cancel
      </button>
    {:else if built}
      <button
        type="button"
        disabled={rebuild.rebuilding}
        class="btn-sm"
        title="Refit the projection over the current pool (background job)"
        onclick={() => void rebuild.trigger()}
      >
        {rebuild.rebuilding ? 'Requesting…' : 'Rebuild'}
      </button>
    {/if}
  </div>

  {#if bannerRequired}
    <div
      class="rounded border border-warn/50 bg-warn/10 px-3 py-1.5 text-meta text-warn-text"
    >
      This projection is approximate — nearby points on screen are not guaranteed to share
      a cluster. Use it to spot rough neighborhoods, not to make final labeling decisions.
    </div>
  {/if}

  <div class="flex min-h-0 flex-1 flex-col gap-2 md:flex-row">
    <div
      bind:clientWidth={containerWidth}
      class="relative min-h-0 min-w-0 flex-1 overflow-hidden rounded-md border border-line"
    >
      {#if !loaded}
        <p class="p-4 text-body text-fg-3" role="status">Loading embedding projection…</p>
      {:else if loadError}
        <p class="p-4 text-body text-danger-text" role="alert">{loadError}</p>
      {:else if !built}
        <div
          class="flex h-full flex-col items-center justify-center gap-3 p-6 text-center"
        >
          <p class="text-body text-fg-2">
            No projection has been built yet. Building one runs as a background job (it
            never fits on the request path), so this can take a while on a large pool.
          </p>
          <button
            type="button"
            disabled={rebuild.rebuilding || rebuild.job !== null}
            class="btn btn-primary"
            onclick={() => void rebuild.trigger()}
          >
            {rebuild.rebuilding
              ? 'Requesting…'
              : rebuild.job
                ? 'Building…'
                : 'Build projection'}
          </button>
        </div>
      {:else if points.length === 0}
        <p class="p-4 text-body text-fg-2">No item has a point on the projection yet.</p>
        {@render empty?.()}
      {:else}
        <canvas
          bind:this={canvasEl}
          width={containerWidth}
          height={PLOT_HEIGHT}
          class="block cursor-crosshair touch-none"
          aria-label="Embedding projection scatter plot. Drag to select points, Shift+drag to draw a box."
          data-testid="embedding-canvas"
          onpointerdown={onPointerDown}
          onpointermove={onPointerMove}
          onpointerup={finishDrag}
          onpointercancel={finishDrag}
          onpointerleave={finishDrag}
        ></canvas>
        <p
          class="pointer-events-none absolute bottom-1.5 right-2 font-mono text-caption text-fg-3"
          data-testid="embedding-count"
        >
          {points.length.toLocaleString()}
          {embeddingCoverageSuffix(points.length, coveragePoolTotal)}
          {#if loading}· updating…{/if}
        </p>
      {/if}
    </div>
    {#if selectedPoints.length > 0}
      <EmbeddingSelectedPreview
        points={selectedPoints}
        count={selectedIds.length}
        height={PLOT_HEIGHT}
      />
    {/if}
  </div>

  {#if points.length > 0}
    <!-- The served class legend: the server decides the classes, their slots and counts. -->
    <ul
      class="flex max-h-24 flex-wrap items-center gap-x-3 gap-y-1 overflow-y-auto text-caption text-fg-2"
      data-testid="embedding-legend"
      aria-label="Colour legend: one colour per class"
    >
      {#each legend as e (e.colour_index)}
        <li class="flex items-center gap-1" data-testid="embedding-legend-entry">
          <span
            class="inline-block h-2.5 w-2.5 rounded-full"
            style:background-color={colourForIndex(e.colour_index)}
          ></span>
          {e.class_name ?? 'no class'}
          <span class="text-fg-3">({e.count.toLocaleString()})</span>
        </li>
      {/each}
    </ul>
  {/if}
</div>
