<script module lang="ts">
  import { apiErrorText } from '$lib/api_errors';
  import type { ImageDetailResponse } from '$lib/types_images';
  import { activeProjectKey } from '$lib/api';
  import { onProjectChange } from '$lib/projectChange';

  // Per-(project, imageId) cache, shared across every mounted instance:
  // several call sites (review, cluster modal, lightbox) can open the
  // same image within one session; avoid refetching a 500-item payload
  // each time. Image ids are content-derived (the same image gets the
  // same id in every project — projects_plan §2.1), so the cache key is
  // namespaced by the active project too, or switching projects would
  // show another project's cached detail. Not evicted — bounded by
  // "images a human actually opened this session", which is small.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- module-level promise cache, never read reactively by a template/derived; plain Map avoids needless per-entry proxy overhead
  const contextCache = new Map<string, Promise<ImageDetailResponse>>();

  function cacheKey(imageId: string): string {
    return `${activeProjectKey()}:${imageId}`;
  }

  /** Clears every cached context. Registered below with the
   *  project-change registry so a project switch never shows stale,
   *  cross-project data — see `resetForProjectChange()` on
   *  `$lib/stores/undo.svelte`'s `undoStore` for the sibling reset on
   *  the undo ring buffer. */
  export function resetForProjectChange(): void {
    contextCache.clear();
  }
  onProjectChange(resetForProjectChange);
</script>

<script lang="ts">
  /**
   * Draws the boxes `GET {API_PREFIX}/images/{id}` serves (`boxes`: coordinates, label, tone)
   * over the clean pixels of `GET {API_PREFIX}/images/{id}/file`; nothing is burned in and no
   * label or colour is derived here (see `BoxLayer`).
   */
  import { getImage, getImageFileUrl } from '$lib/api_images';
  import { fetchSourceMissing, type SourceMissing } from '$lib/sourceMissing';
  import SourceMissingBadge from './SourceMissingBadge.svelte';
  import type { OverlayShape } from '$lib/configTest/overlayShapes';
  import BoxLayer from './boxes/BoxLayer.svelte';

  interface Props {
    /** Which image (its pixels and every item detected on it) to draw. */
    imageId: string;
    /** Item highlighted with a thicker ring; none when absent. */
    selectedCropId?: string | null;
    /** Passed to the image URL — the review page only needs the boxes to
     *  be readable, not pixel-perfect. */
    maxDim?: number;
    /** Fires when a non-selected item's box is clicked. Absent = boxes
     *  aren't clickable (still hoverable for the tooltip). */
    onselect?: (cropId: string) => void;
    /** Outer wrapper class — sizing/layout stays the caller's job. */
    class?: string;
    /** Vertical placement of the image in a taller box (V-3: `/review`
     *  top-aligns so the image doesn't float mid-way down a tall pane). */
    align?: 'center' | 'start';
    /** Pre-fetched context (e.g. a caller that already loaded it for its
     *  own purposes) — skips this component's own fetch entirely. */
    context?: ImageDetailResponse | null;
    /** Extra shapes in the source-image frame (test-on-crop candidates):
     *  boxes and mask outlines, `dimmed` ones drawn faint. */
    extraShapes?: OverlayShape[];
  }

  let {
    imageId,
    selectedCropId = null,
    maxDim = 1600,
    onselect,
    class: className = '',
    align = 'center',
    context: providedContext = null,
    extraShapes = [],
  }: Props = $props();

  let fetchedContext = $state<ImageDetailResponse | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);
  let showBoxes = $state(true);
  let allRegions = $state(false);
  let sourceMissing = $state<SourceMissing | null>(null);
  let naturalW = $state<number | null>(null);
  let naturalH = $state<number | null>(null);

  const context = $derived(providedContext ?? fetchedContext);

  $effect(() => {
    if (providedContext) return;
    const id = imageId;
    fetchedContext = null;
    sourceMissing = null;
    error = null;
    loading = true;
    const controller = new AbortController();
    const key = cacheKey(id);
    let cached = contextCache.get(key);
    if (!cached) {
      cached = getImage(id, controller.signal);
      contextCache.set(key, cached);
    }
    cached
      .then((res) => {
        fetchedContext = res;
      })
      .catch((e: unknown) => {
        contextCache.delete(key);
        if ((e as Error)?.name === 'AbortError') return;
        error = apiErrorText(e);
      })
      .finally(() => {
        loading = false;
      });
    return () => controller.abort();
  });

  const aspectRatio = $derived.by(() => {
    const w = context?.image?.width ?? naturalW;
    const h = context?.image?.height ?? naturalH;
    return w && h ? `${w} / ${h}` : null;
  });

  function pct(n: number): string {
    return `${(n * 100).toFixed(3)}%`;
  }
</script>

<div
  class="relative flex h-full w-full justify-center {align === 'start'
    ? 'items-start'
    : 'items-center'} {className}"
  data-align={align}
>
  {#if loading && !context}
    <p class="text-xs text-fg-3">Loading…</p>
  {:else if error}
    <p class="text-xs text-danger-text">Source image unavailable: {error}</p>
  {:else if sourceMissing}
    <div class="relative h-full min-h-40 w-full">
      <SourceMissingBadge missing={sourceMissing} />
    </div>
  {:else}
    <div
      class="relative max-h-full max-w-full"
      style={aspectRatio ? `aspect-ratio: ${aspectRatio};` : ''}
    >
      <img
        src={getImageFileUrl(imageId, maxDim)}
        alt="source"
        loading="eager"
        decoding="async"
        class="block h-full max-h-full w-full max-w-full object-contain"
        onload={(e) => {
          const el = e.currentTarget as HTMLImageElement;
          naturalW = el.naturalWidth || naturalW;
          naturalH = el.naturalHeight || naturalH;
        }}
        onerror={(e) => {
          void fetchSourceMissing((e.currentTarget as HTMLImageElement).src).then(
            (m) => (sourceMissing = m),
          );
        }}
      />
      {#if showBoxes && context}
        <div class="pointer-events-none absolute inset-0" data-testid="overlay-layer">
          <BoxLayer boxes={context.boxes} {selectedCropId} {onselect} {allRegions} />
          {#each extraShapes as s (s.key)}
            {#if s.kind === 'box'}
              {@const [x1, y1, x2, y2] = s.box}
              <div
                class="absolute border-2 border-info {s.dimmed
                  ? 'border-dashed opacity-40'
                  : 'z-10'}"
                style="left:{pct(x1)}; top:{pct(y1)}; width:{pct(x2 - x1)}; height:{pct(
                  y2 - y1,
                )};"
                data-testid="overlay-extra-box"
                data-dimmed={s.dimmed}
                title={s.title}
              >
                <span
                  class="absolute -top-4 left-0 rounded-sm bg-surface-0/90 px-1 py-0.5 text-caption leading-none whitespace-nowrap text-fg-1"
                  >{s.label}</span
                >
              </div>
            {:else}
              <svg
                class="absolute inset-0 h-full w-full {s.dimmed ? 'opacity-40' : ''}"
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <polygon
                  points={s.points.map((q) => `${q[0]},${q[1]}`).join(' ')}
                  fill="rgba(56, 189, 248, 0.15)"
                  stroke="rgb(56, 189, 248)"
                  stroke-width="2"
                  stroke-dasharray={s.dimmed ? '4 3' : undefined}
                  vector-effect="non-scaling-stroke"
                  data-testid="overlay-extra-polygon"
                  data-dimmed={s.dimmed}
                >
                  <title>{s.title}</title>
                </polygon>
              </svg>
            {/if}
          {/each}
        </div>
      {/if}
      <div class="pointer-events-auto absolute top-1 right-1 z-20 flex gap-1">
        <button
          type="button"
          class="rounded border border-line-strong bg-surface-0/80 px-1.5 py-0.5 text-caption whitespace-nowrap text-fg-2 hover:bg-surface-1"
          onclick={() => (showBoxes = !showBoxes)}
        >
          {showBoxes ? 'hide boxes' : 'show boxes'}
        </button>
        {#if showBoxes}
          <button
            type="button"
            class="rounded border border-line-strong bg-surface-0/80 px-1.5 py-0.5 text-caption whitespace-nowrap text-fg-2 hover:bg-surface-1"
            aria-pressed={allRegions}
            data-testid="overlay-all-regions"
            onclick={() => (allRegions = !allRegions)}
          >
            All regions
          </button>
        {/if}
      </div>
    </div>
  {/if}
</div>
