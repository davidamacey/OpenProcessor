<script lang="ts">
  /**
   * Shared flat crop grid: dndzone + `createSelection`-based click/
   * multi-select + `<CropCard>` with absolute-positioned overlay chips.
   * Extracted from `/clusters/[id]`'s single-zone rendering pattern
   * (src/routes/clusters/[id]/+page.svelte) for reuse by the global
   * search-result grid on `/clusters` (see `CropResultGrid` usage in
   * `src/routes/clusters/+page.svelte`).
   *
   * Deliberately flat — one dndzone, no sub-cluster grouping, no cut-line.
   * Global search results aren't scoped to a single cluster, so AHC
   * sub-clustering and the outliers cut-line don't apply here; a caller
   * that needs those (i.e. `/clusters/[id]` itself) still owns its own
   * per-group dndzone loop today. This component only extracts the
   * flat-list case; retro-fitting `/clusters/[id]` onto it is a
   * follow-up, not done in this change (see report for why).
   *
   * The grid is a draggable-only zone: dragging a card out (onto the
   * layout's ClassSidebar) removes it locally; the real label RPC is the
   * caller's job via `dropOnClassStore`, mirroring `/clusters/[id]`.
   */
  import { dndzone, SOURCES, TRIGGERS } from 'svelte-dnd-action';
  import CropCard from '$components/CropCard.svelte';
  import ScoreChip from '$components/ScoreChip.svelte';
  import type { Selection } from '$lib/selection.svelte';
  import type { OpCrop } from '$lib/types';

  interface Props {
    items: OpCrop[];
    sel: Selection;
    /** Match-score badge value (top-left), e.g. search similarity. Return
     *  null/undefined to omit the chip for that crop. */
    scoreOf?: (crop: OpCrop) => number | null | undefined;
    /** Extra overlay rendered top-right of each card (e.g. cluster-origin
     *  badge) — a snippet so the caller controls its own data lookups. */
    cornerBadge?: import('svelte').Snippet<[OpCrop]>;
    ondetail?: (crop: OpCrop) => void;
    onacceptGemma?: (crop: OpCrop) => void;
    onrejectGemma?: (crop: OpCrop) => void;
    /** Fired whenever the in-flight drag's captured id set changes
     *  (drag start / drag end) — mirrors `/clusters/[id]`'s `dragIds`. */
    ondragidschange?: (ids: string[]) => void;
  }

  let {
    items,
    sel,
    scoreOf,
    cornerBadge,
    ondetail,
    onacceptGemma,
    onrejectGemma,
    ondragidschange,
  }: Props = $props();

  // Drag-local override of the rendered list, mirroring gridGroups.svelte's
  // reconciliation: reset on every finalize so a stale dnd snapshot can
  // never paint back an item the caller already removed from `items`.
  let dragItems = $state<OpCrop[] | null>(null);
  const renderItems = $derived(dragItems ?? items);

  let dragIds = $state<string[]>([]);

  function clickSelect(id: string, e?: MouseEvent): void {
    sel.click(
      id,
      e,
      items.map((c) => c.id),
    );
  }

  function onConsider(
    e: CustomEvent<{
      items: OpCrop[];
      info: { id: string; trigger: TRIGGERS; source: SOURCES };
    }>,
  ): void {
    const draggedId = e.detail.info?.id;
    if (draggedId && !dragIds.includes(draggedId)) {
      if (sel.has(draggedId) && sel.size > 1) {
        dragIds = [...sel.ids];
      } else {
        dragIds = [draggedId];
        sel.ids = new Set([draggedId]);
      }
      ondragidschange?.(dragIds);
    }
    dragItems = e.detail.items;
  }

  function onFinalize(e: CustomEvent<{ items: OpCrop[]; info: { trigger: TRIGGERS } }>): void {
    dragItems = null;
    if (e.detail.info.trigger === TRIGGERS.DROPPED_INTO_ANOTHER) {
      setTimeout(() => {
        dragIds = [];
        ondragidschange?.(dragIds);
      }, 0);
    } else {
      dragIds = [];
      ondragidschange?.(dragIds);
    }
  }
</script>

<div
  class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8"
  use:dndzone={{
    items: renderItems,
    type: 'op-crop',
    flipDurationMs: 150,
    dropTargetStyle: { outline: '2px dashed rgb(59 130 246 / 0.6)' },
    dragDisabled: false,
    dropFromOthersDisabled: true,
  }}
  onconsider={onConsider}
  onfinalize={onFinalize}
>
  {#each renderItems as crop (crop.id)}
    <div class="relative">
      <CropCard
        {crop}
        selected={sel.has(crop.id)}
        onclick={(c, e) => clickSelect(c.id, e)}
        onacceptGemma={(c) => onacceptGemma?.(c)}
        onrejectGemma={(c) => onrejectGemma?.(c)}
        ondetail={(c) => ondetail?.(c)}
      />
      {#if scoreOf}
        {@const score = scoreOf(crop)}
        {#if score != null}
          <div class="pointer-events-none absolute left-1 top-1 z-10">
            <ScoreChip label="match" value={score} size="sm" />
          </div>
        {/if}
      {/if}
      {#if cornerBadge}
        <div class="pointer-events-none absolute right-1 top-1 z-10">
          {@render cornerBadge(crop)}
        </div>
      {/if}
    </div>
  {/each}
</div>
