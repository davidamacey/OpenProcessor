<script lang="ts">
  /**
   * 2-d embedding-projection scatter plot (curation-strategy plan Phase 5 —
   * docs/curation-strategy-plan-2026-09.md §2.7/§5.6). UMAP-as-visualization
   * -only overlay: it decorates the existing FAISS-IVF `cluster_id`
   * assignment, it never decides one, and it is never fit on a request
   * path — `getVizProjection` only ever serves whatever the last backend
   * rebuild job produced.
   *
   * Hard constraints from the plan + CLAUDE.md's Keyboard shortcuts
   * section (mirrors StrategyBar.svelte's header comment, and is covered
   * by the same kind of static-source-scan regression test in
   * EmbeddingPlot.test.ts since this repo has no component-mount
   * harness):
   *
   * - Lazily mounted: the caller (`/clusters`) only renders this
   *   component inside an `{#if}` block gated on an explicit operator
   *   toggle — never as part of the page's initial render. Off by
   *   default.
   * - Canvas + pointer events ONLY. Zero window/document keydown
   *   listeners — the lasso-select interaction is click-drag on the
   *   canvas element itself (native `onpointerdown`/`onpointermove`/
   *   `onpointerup` element attributes), never a global key binding.
   * - Lasso-select feeds the EXISTING `bulkLabel`/`moveCropsToCluster`
   *   API paths (same functions the cluster-detail page's class-drop and
   *   move-picker already call) — no new mutation endpoint.
   * - Point color is derived from `cluster_id` only (`colorForCluster` in
   *   `$lib/embeddingPlot.ts`); this component never computes or
   *   assigns a cluster.
   */

  import {
    bulkLabel,
    getThumbUrl,
    getVizProjection,
    moveCropsToCluster,
    rebuildVizProjection,
    type VizPoint,
  } from '$lib/api';
  import { colorForCluster, computeScale, selectIdsInLasso } from '$lib/embeddingPlot';
  import type { ScreenPoint } from '$lib/embeddingPlot';
  import { classesStore } from '$stores/classes.svelte';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    /** Scope the projection to one cluster. `null` = pool-wide (today's
     *  only caller, `/clusters`, always passes `null`). */
    clusterId?: number | null;
    /** Scope the projection to one class (mirrors `/clusters`'s
     *  `?class=` filter). */
    classId?: number | null;
    maxPoints?: number;
    /** True when `/curation/methods` reports this overlay at the
     *  banner-required purity tier (see `isEmbeddingVizBannerRequired`
     *  in `$lib/strategies.ts`) — renders a small persistent note. */
    bannerRequired?: boolean;
  }

  let {
    clusterId = null,
    classId = null,
    maxPoints = 4000,
    bannerRequired = false,
  }: Props = $props();

  const PLOT_HEIGHT = 520;
  const POINT_RADIUS = 3;
  const SELECTED_RADIUS = 4.5;

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let containerWidth = $state<number>(800);

  let points = $state<VizPoint[]>([]);
  // Optimistic true so the very first render (before the first load()
  // resolves) shows "Loading…" rather than a flash of the pending state.
  let built = $state<boolean>(true);
  let loading = $state<boolean>(true);

  let selectedIds = $state<Set<string>>(new Set());
  let dragging = $state<boolean>(false);
  let lassoPath = $state<ScreenPoint[]>([]);

  let assignClassId = $state<number | null>(null);
  let moveTargetInput = $state<string>('');
  let busy = $state<boolean>(false);
  let rebuilding = $state<boolean>(false);
  // Which selected-preview thumbnail (if any) is shown enlarged. The strip
  // thumbnails are 56px -- too small to actually judge a crop by, per live
  // feedback ("I need to be able to individually select and click on them
  // to see in the larger mode as they are small").
  let expandedPreviewId = $state<string | null>(null);

  async function load(): Promise<void> {
    loading = true;
    // getVizProjection() never rejects (except a caller abort, which this
    // component never issues) — a 404/network failure degrades to the
    // empty/pending fallback, so no try/catch is needed here.
    const res = await getVizProjection({
      cluster_id: clusterId ?? undefined,
      class_id: classId ?? undefined,
      max_points: maxPoints,
    });
    points = res.points;
    built = res.built;
    loading = false;
  }

  $effect(() => {
    void clusterId;
    void classId;
    void maxPoints;
    void load();
  });

  const scale = $derived.by(() =>
    computeScale(points, Math.max(1, containerWidth), PLOT_HEIGHT, 20),
  );

  const scaledPoints = $derived.by(() =>
    points.map((p) => {
      const s = scale.toScreen(p.x, p.y);
      return { ...s, id: p.crop_id, cluster_id: p.cluster_id };
    }),
  );

  // The lasso only ever selects by id -- this recovers the full crop data
  // (thumbnail + current class) for the preview strip below, so an
  // operator can see what they're about to bulk-assign/move instead of
  // acting on bare dots. Capped so a large lasso doesn't render thousands
  // of <img> tags; the strip is a sanity check, not a full review grid.
  const SELECTED_PREVIEW_CAP = 60;
  const selectedPreview = $derived.by(() =>
    points.filter((p) => selectedIds.has(p.crop_id)).slice(0, SELECTED_PREVIEW_CAP),
  );

  function canvasPoint(e: PointerEvent): ScreenPoint {
    // offsetX/offsetY are relative to the target element (the canvas
    // itself), which lines up 1:1 with computeScale's pixel space
    // because the canvas's pixel width/height are kept in sync with
    // containerWidth/PLOT_HEIGHT below (no CSS-only scaling).
    return { x: e.offsetX, y: e.offsetY };
  }

  function onPointerDown(e: PointerEvent): void {
    if (e.button !== 0) return;
    dragging = true;
    lassoPath = [canvasPoint(e)];
  }

  function onPointerMove(e: PointerEvent): void {
    if (!dragging) return;
    lassoPath = [...lassoPath, canvasPoint(e)];
  }

  function finishLasso(): void {
    if (!dragging) return;
    dragging = false;
    if (lassoPath.length > 2) {
      selectedIds = new Set(selectIdsInLasso(scaledPoints, lassoPath));
    }
    lassoPath = [];
  }

  function onPointerUp(): void {
    finishLasso();
  }

  function onPointerLeave(): void {
    // Pointer left the canvas mid-drag: finish with whatever was traced
    // rather than leaving `dragging` stuck true (there is no keyboard
    // Escape path for this — canvas/pointer only, per the header note).
    finishLasso();
  }

  function clearSelection(): void {
    selectedIds = new Set();
  }

  function draw(): void {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.fillStyle = '#09090b'; // zinc-950 — matches the app's dark base
    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);

    for (const p of scaledPoints) {
      const selected = selectedIds.has(p.id);
      ctx.beginPath();
      ctx.arc(p.x, p.y, selected ? SELECTED_RADIUS : POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = colorForCluster(p.cluster_id);
      ctx.globalAlpha = selected || selectedIds.size === 0 ? 0.9 : 0.25;
      ctx.fill();
      if (selected) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#f4f4f5'; // zinc-100 ring around a selected point
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    if (dragging && lassoPath.length > 1) {
      ctx.beginPath();
      ctx.moveTo(lassoPath[0]!.x, lassoPath[0]!.y);
      for (const pt of lassoPath.slice(1)) ctx.lineTo(pt.x, pt.y);
      ctx.strokeStyle = 'rgba(96, 165, 250, 0.9)'; // blue-400
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = 'rgba(96, 165, 250, 0.12)';
      ctx.fill();
    }
  }

  $effect(() => {
    // Tracked deps: redraw whenever any of these change. containerWidth
    // drives the canvas element's own width/height attributes below, so
    // this effect also needs to re-run once those attributes actually
    // change (next microtask after containerWidth updates), which the
    // `canvasEl.width`/`canvasEl.height` reads below accomplish.
    void scaledPoints;
    void selectedIds;
    void lassoPath;
    void containerWidth;
    if (canvasEl) {
      canvasEl.width = Math.max(1, Math.round(containerWidth));
      canvasEl.height = PLOT_HEIGHT;
    }
    draw();
  });

  async function assignSelectedToClass(): Promise<void> {
    if (assignClassId == null || selectedIds.size === 0 || busy) return;
    const ids = [...selectedIds];
    const cls = classesStore.classes.find((c) => c.id === assignClassId);
    busy = true;
    try {
      const res = await bulkLabel(ids, assignClassId);
      toastStore.success(
        `Labeled ${res.updated ?? ids.length} → ${cls?.name ?? assignClassId}.`,
      );
      // bulkLabel doesn't move cluster_id, so the points' positions/colors
      // are unaffected — just clear the selection, no reload needed.
      clearSelection();
    } catch (e) {
      toastStore.error(`Label failed: ${(e as Error).message}`);
    } finally {
      busy = false;
    }
  }

  async function moveSelectedToCluster(): Promise<void> {
    const target = Number(moveTargetInput);
    if (!Number.isFinite(target) || target < 0 || selectedIds.size === 0 || busy) return;
    const ids = [...selectedIds];
    busy = true;
    try {
      const res = await moveCropsToCluster(ids, target);
      toastStore.success(`Moved ${res.updated ?? ids.length} → cluster #${target}.`);
      clearSelection();
      moveTargetInput = '';
      // Moving changes cluster_id, which this plot colors by — reload so
      // the moved points render with their new color/bucket.
      await load();
    } catch (e) {
      toastStore.error(`Move failed: ${(e as Error).message}`);
    } finally {
      busy = false;
    }
  }

  async function triggerRebuild(): Promise<void> {
    if (rebuilding) return;
    rebuilding = true;
    try {
      await rebuildVizProjection();
      toastStore.info(
        'Rebuilding embedding projection… this runs as a background job and can take a while.',
      );
    } catch (e) {
      toastStore.error(`Rebuild failed: ${(e as Error).message}`);
    } finally {
      rebuilding = false;
    }
  }
</script>

<div class="flex flex-col gap-2">
  {#if bannerRequired}
    <div
      class="rounded border border-amber-500/50 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-200"
    >
      This projection is approximate — nearby points on screen are not guaranteed to share
      a cluster. Use it to spot rough neighborhoods, not to make final labeling decisions.
    </div>
  {/if}

  {#if selectedIds.size > 0}
    <div
      class="flex flex-wrap items-center gap-2 rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs"
    >
      <span class="font-medium text-blue-200">{selectedIds.size} selected</span>
      <span class="grow"></span>
      <select
        bind:value={assignClassId}
        class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
      >
        <option value={null}>— class —</option>
        {#each classesStore.classes as cls (cls.id)}
          <option value={cls.id}>{cls.name}</option>
        {/each}
      </select>
      <button
        type="button"
        disabled={busy || assignClassId == null}
        class="rounded border border-green-500/50 bg-green-500/20 px-2 py-1 text-green-100 hover:bg-green-500/30 disabled:opacity-50"
        onclick={() => void assignSelectedToClass()}
      >
        Assign
      </button>
      <span class="mx-1 h-4 w-px bg-zinc-700"></span>
      <label class="flex items-center gap-1 text-zinc-400">
        move to cluster
        <input
          type="number"
          min="0"
          step="1"
          bind:value={moveTargetInput}
          class="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100"
          placeholder="id"
        />
      </label>
      <button
        type="button"
        disabled={busy || moveTargetInput === ''}
        class="rounded border border-zinc-600 bg-zinc-800 px-2 py-1 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
        onclick={() => void moveSelectedToCluster()}
      >
        Move
      </button>
      <button
        type="button"
        class="rounded border border-zinc-700 px-2 py-1 text-zinc-400 hover:bg-zinc-800"
        onclick={clearSelection}
      >
        Clear
      </button>
    </div>
  {/if}

  <!-- Plot + selected-crop preview column live side by side so a large
       lasso's thumbnail strip never pushes the plot down the page --
       it scrolls in its own fixed-width column instead. Assign/Move act
       on bare dots from a projection this component's own banner admits
       is approximate -- an operator needs to actually SEE what they're
       about to bulk-relabel before committing, not just trust proximity
       on a 2-d scatter. -->
  <div class="flex min-h-0 flex-1 gap-2">
    <div
      bind:clientWidth={containerWidth}
      class="relative min-h-0 min-w-0 flex-1 overflow-hidden rounded-md border border-zinc-800"
    >
      {#if loading}
      <p class="p-4 text-sm text-zinc-500">Loading embedding projection…</p>
    {:else if !built}
      <div class="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
        <p class="text-sm text-zinc-400">
          No projection has been built yet. Building one runs as a background job (it
          never fits on the request path), so this can take a while on a large pool.
        </p>
        <button
          type="button"
          disabled={rebuilding}
          class="rounded border border-blue-500/50 bg-blue-500/20 px-3 py-1.5 text-xs text-blue-100 hover:bg-blue-500/30 disabled:opacity-50"
          onclick={() => void triggerRebuild()}
        >
          {rebuilding ? 'Requesting…' : 'Build projection'}
        </button>
      </div>
    {:else if points.length === 0}
      <p class="p-4 text-sm text-zinc-500">
        No points match the current filter — try a broader class/cluster scope.
      </p>
    {:else}
      <canvas
        bind:this={canvasEl}
        width={containerWidth}
        height={PLOT_HEIGHT}
        class="block cursor-crosshair touch-none"
        aria-label="Embedding projection scatter plot — click and drag to lasso-select points"
        onpointerdown={onPointerDown}
        onpointermove={onPointerMove}
        onpointerup={onPointerUp}
        onpointercancel={onPointerLeave}
        onpointerleave={onPointerLeave}
      ></canvas>
      <p
        class="pointer-events-none absolute bottom-1.5 right-2 font-mono text-[10px] text-zinc-500"
      >
        {points.length.toLocaleString()} points · click-drag to lasso-select
      </p>
    {/if}
    </div>

    {#if selectedIds.size > 0}
      <!-- Capped (SELECTED_PREVIEW_CAP) so a huge lasso doesn't render
           thousands of <img> tags -- this is a sanity check, not a full
           review grid. Individually clickable: each thumbnail is 56px,
           too small to actually judge a crop by, so clicking one opens
           it enlarged (expandedPreviewId below). -->
      <div
        class="flex w-40 shrink-0 flex-col gap-1 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950 p-2"
        style:height="{PLOT_HEIGHT}px"
      >
        <p class="text-[10px] text-zinc-500">{selectedIds.size} selected — click to enlarge</p>
        <div class="grid grid-cols-2 gap-1">
          {#each selectedPreview as p (p.crop_id)}
            <button
              type="button"
              class="group relative aspect-square overflow-hidden rounded border border-zinc-700 hover:border-blue-400"
              title={p.class_name ?? 'unlabeled'}
              onclick={() => (expandedPreviewId = p.crop_id)}
            >
              <img
                src={getThumbUrl(p.crop_id, 64)}
                alt="crop {p.crop_id}"
                draggable="false"
                class="h-full w-full object-cover [-webkit-user-drag:none]"
              />
              {#if p.class_name}
                <span
                  class="absolute inset-x-0 bottom-0 truncate bg-black/70 px-1 text-center text-[9px] text-zinc-200"
                >
                  {p.class_name}
                </span>
              {/if}
            </button>
          {/each}
        </div>
        {#if selectedIds.size > SELECTED_PREVIEW_CAP}
          <p class="text-center text-[10px] text-zinc-500">
            +{selectedIds.size - SELECTED_PREVIEW_CAP} more
          </p>
        {/if}
      </div>
    {/if}
  </div>
</div>

{#if expandedPreviewId}
  {@const expandedPoint = points.find((p) => p.crop_id === expandedPreviewId)}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Selected crop preview"
  >
    <div class="relative max-h-full max-w-2xl">
      <img
        src={getThumbUrl(expandedPreviewId, 512)}
        alt="crop {expandedPreviewId}"
        draggable="false"
        class="max-h-[80vh] max-w-full rounded-md border border-zinc-700 [-webkit-user-drag:none]"
      />
      {#if expandedPoint?.class_name}
        <p class="absolute inset-x-0 bottom-0 bg-black/70 px-2 py-1 text-center text-sm text-zinc-200">
          {expandedPoint.class_name}
        </p>
      {/if}
      <button
        type="button"
        class="absolute -top-3 -right-3 rounded-full border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-white"
        onclick={(e) => {
          e.stopPropagation();
          expandedPreviewId = null;
        }}
        aria-label="Close"
      >
        ×
      </button>
    </div>
  </div>
{/if}
