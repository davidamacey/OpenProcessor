<script module lang="ts">
  import type { CropContextResponse } from '$lib/types';
  import { activeProjectKey } from '$lib/api';
  import { onProjectChange } from '$lib/projectChange';

  // Per-(project, cropId) cache, shared across every mounted instance:
  // several call sites (review, cluster modal, lightbox) can open the
  // same crop within one session; avoid refetching a 500-item context
  // payload each time. Crop ids are content-derived (the same image
  // gets the same crop_id in every project — projects_plan §2.1), so
  // the cache key is namespaced by the active project too, or switching
  // projects would show another project's cached context. Not evicted —
  // bounded by "crops a human actually opened this session", which is
  // small.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- module-level promise cache, never read reactively by a template/derived; plain Map avoids needless per-entry proxy overhead
  const contextCache = new Map<string, Promise<CropContextResponse>>();

  function cacheKey(cropId: string): string {
    return `${activeProjectKey()}:${cropId}`;
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
   * Draws every box/label a source image needs, client-side, from
   * `GET {API_PREFIX}/crops/{id}/context` metadata — replaces the
   * server-side burn-in OpenProcessor is removing from
   * `GET {API_PREFIX}/crops/{id}/image` (K6,
   * docs/design/k6-frontend-overlay-plan-2026-09-24.md). Domain-neutral
   * by construction: every region box/color/label comes from the
   * matching registered `SlotSpec`, never a hardcoded noun.
   */
  import { getCropContext, getSourceImageScaled } from '$lib/api';
  import type { Crop } from '$lib/types';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import { slotOf, subBoxSlotFor } from '$lib/annotations/cropSlots';
  import type { XYXY } from '$lib/annotations/types';
  import { regionStatusesStore, toneBorderClass } from '$stores/regionStatuses.svelte';

  interface Props {
    /** Which crop's context (source image + every item cropped from it) to draw. */
    cropId: string;
    /** Item highlighted with a thicker ring; defaults to `cropId` itself. */
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
    context?: CropContextResponse | null;
  }

  let {
    cropId,
    selectedCropId = null,
    maxDim = 1600,
    onselect,
    class: className = '',
    align = 'center',
    context: providedContext = null,
  }: Props = $props();

  let fetchedContext = $state<CropContextResponse | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);
  let showBoxes = $state(true);
  let naturalW = $state<number | null>(null);
  let naturalH = $state<number | null>(null);

  const context = $derived(providedContext ?? fetchedContext);

  $effect(() => {
    if (providedContext) return;
    const id = cropId;
    fetchedContext = null;
    error = null;
    loading = true;
    const controller = new AbortController();
    const key = cacheKey(id);
    let cached = contextCache.get(key);
    if (!cached) {
      cached = getCropContext(id, controller.signal);
      contextCache.set(key, cached);
    }
    cached
      .then((res) => {
        fetchedContext = res;
      })
      .catch((e: unknown) => {
        contextCache.delete(key);
        if ((e as Error)?.name === 'AbortError') return;
        error = (e as Error).message;
      })
      .finally(() => {
        loading = false;
      });
    return () => controller.abort();
  });

  const effectiveSelected = $derived(selectedCropId ?? cropId);

  const aspectRatio = $derived.by(() => {
    const w = context?.image?.width ?? naturalW;
    const h = context?.image?.height ?? naturalH;
    return w && h ? `${w} / ${h}` : null;
  });

  /** A box stored relative to its parent crop, as source-image xyxy. */
  function parentToSource(box: XYXY, parent: XYXY): XYXY {
    const pw = parent[2] - parent[0];
    const ph = parent[3] - parent[1];
    return [
      parent[0] + box[0] * pw,
      parent[1] + box[1] * ph,
      parent[0] + box[2] * pw,
      parent[1] + box[3] * ph,
    ];
  }

  function bboxToXyxy(b: { cx: number; cy: number; w: number; h: number }): XYXY {
    return [b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2];
  }

  /** W8 multi-box (docs/design/w8-multibox-frontend-plan-2026-09-26.md):
   *  per-box state -> ring color/dash. The ring color reads the served
   *  box_states `tone` (backend follow-up to W8.7) via
   *  `toneBorderClass(boxStateTone(state))` — 'neutral' on a pre-tone
   *  backend or an unrecognized state, matching `+page.svelte`'s
   *  `multiBoxRingColor`. */
  function multiBoxRingColorClass(state: string): string {
    return toneBorderClass(regionStatusesStore.boxStateTone(state));
  }
  function multiBoxDashed(state: string): boolean {
    return (
      regionStatusesStore.boxStateInfo(state)?.dashed ??
      (state === 'rejected' || state === 'false_positive')
    );
  }

  interface DrawBox {
    kind: 'item' | 'region' | 'region-box';
    cropId: string;
    xyxy: XYXY;
    dashed: boolean;
    colorClass: string;
    label: string;
    tooltip: string;
    selected: boolean;
    clickable: boolean;
  }

  function itemLabelInfo(item: Crop): { label: string; colorClass: string } {
    if (item.class_name) {
      return { label: item.class_name, colorClass: 'border-emerald-400' };
    }
    if (item.proposed_class_name) {
      return {
        label: `${item.proposed_class_name} (proposed)`,
        colorClass: 'border-amber-400',
      };
    }
    return { label: 'unlabeled', colorClass: 'border-zinc-500' };
  }

  const boxes = $derived.by<DrawBox[]>(() => {
    const items = context?.items ?? [];
    const out: DrawBox[] = [];
    for (const item of items) {
      const selected = item.id === effectiveSelected;
      const { label, colorClass } = itemLabelInfo(item);
      const itemXyxy = bboxToXyxy(item.bbox_norm);
      out.push({
        kind: 'item',
        cropId: item.id,
        xyxy: itemXyxy,
        dashed: false,
        colorClass,
        label,
        tooltip: `${label}${item.id === effectiveSelected ? ' (this crop)' : ''}`,
        selected,
        clickable: !selected && !!onselect,
      });

      const slot = subBoxSlotFor(item, slotRegistry.all);
      if (!slot?.capabilities.subBox) continue;
      const data = slotOf(item, slot);
      const sub = data?.subBox;
      const boxList = data?.subBoxes ?? [];
      const ring = slot.capabilities.subBox.ring;

      // A read-only scalar-box slot (tier 2): one box, stored in either
      // frame.
      if (sub?.rawXyxy) {
        out.push({
          kind: 'region',
          cropId: item.id,
          xyxy:
            sub.frame === 'source' ? sub.rawXyxy : parentToSource(sub.rawXyxy, itemXyxy),
          dashed: false,
          colorClass: ring.confirmed,
          label: slot.label.title,
          tooltip: `${slot.label.title}${sub.score != null ? ` · ${(sub.score * 100).toFixed(0)}%` : ''}`,
          selected: false,
          clickable: false,
        });
      }

      // Multi-box slot: every served box, in the source image's frame
      // already (`bbox_norm`), numbered by position.
      boxList.forEach((b, i) => {
        if (!b.rawXyxy) return;
        out.push({
          kind: 'region-box',
          cropId: item.id,
          xyxy: b.rawXyxy,
          dashed: multiBoxDashed(b.state),
          colorClass: multiBoxRingColorClass(b.state),
          label: `${slot.label.title} ${i + 1}`,
          tooltip: `${slot.label.title} ${i + 1} · ${b.state}${
            b.score != null ? ` · ${(b.score * 100).toFixed(0)}%` : ''
          }`,
          selected: false,
          clickable: false,
        });
      });
    }
    return out;
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
    <p class="text-xs text-zinc-500">Loading…</p>
  {:else if error}
    <p class="text-xs text-red-300">Source image unavailable: {error}</p>
  {:else}
    <div
      class="relative max-h-full max-w-full"
      style={aspectRatio ? `aspect-ratio: ${aspectRatio};` : ''}
    >
      <img
        src={getSourceImageScaled(cropId, maxDim)}
        alt="source"
        loading="eager"
        decoding="async"
        class="block h-full max-h-full w-full max-w-full object-contain"
        onload={(e) => {
          const el = e.currentTarget as HTMLImageElement;
          naturalW = el.naturalWidth || naturalW;
          naturalH = el.naturalHeight || naturalH;
        }}
      />
      {#if showBoxes && context}
        {#snippet boxLabel(b: DrawBox)}
          {#if b.selected || b.kind !== 'item'}
            <span
              class="pointer-events-none absolute -top-4 left-0 whitespace-nowrap rounded-sm bg-zinc-950/90 px-1 py-0.5 text-[9px] leading-none text-zinc-100"
            >
              {b.label}
            </span>
          {/if}
        {/snippet}
        <div class="pointer-events-none absolute inset-0" data-testid="overlay-layer">
          {#each boxes as b, i (`${b.kind}-${b.cropId}-${i}`)}
            {@const [x1, y1, x2, y2] = b.xyxy}
            {@const boxClass = `absolute border-2 ${b.colorClass} ${b.dashed ? 'border-dashed' : ''} ${
              b.selected
                ? 'z-10 ring-2 ring-white/80'
                : b.kind === 'item'
                  ? 'opacity-60'
                  : ''
            }`}
            {@const boxStyle = `left:${pct(x1)}; top:${pct(y1)}; width:${pct(x2 - x1)}; height:${pct(y2 - y1)};`}
            {#if b.clickable}
              <button
                type="button"
                class="pointer-events-auto cursor-pointer {boxClass}"
                style={boxStyle}
                data-testid="overlay-box"
                data-kind={b.kind}
                data-crop-id={b.cropId}
                title={b.tooltip}
                onclick={() => onselect?.(b.cropId)}
              >
                {@render boxLabel(b)}
              </button>
            {:else}
              <div
                class={boxClass}
                style={boxStyle}
                data-testid="overlay-box"
                data-kind={b.kind}
                data-crop-id={b.cropId}
                title={b.tooltip}
              >
                {@render boxLabel(b)}
              </div>
            {/if}
          {/each}
        </div>
      {/if}
      <button
        type="button"
        class="pointer-events-auto absolute top-1 right-1 z-20 rounded border border-zinc-700 bg-zinc-950/80 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-900"
        onclick={() => (showBoxes = !showBoxes)}
      >
        {showBoxes ? 'hide boxes' : 'show boxes'}
      </button>
    </div>
  {/if}
</div>
