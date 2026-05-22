<script lang="ts">
  import { dndzone } from 'svelte-dnd-action';
  import AddClassModal from './AddClassModal.svelte';
  import { adequacyChipClass, adequacyTooltip } from '$lib/adequacy';
  import { classesStore } from '$stores/classes.svelte';
  import type { OpClass } from '$lib/types';

  interface Props {
    selectedId: number | null;
    onselect: (cls: OpClass | null) => void;
    /**
     * Optional drop handler — when crops are dragged from a grid onto a
     * class row, this fires with the destination class. Caller is
     * responsible for assigning the currently-selected crops to that class.
     * When omitted, drop targets are disabled (sidebar is filter-only).
     */
    ondrop?: (cls: OpClass, droppedIds: string[]) => void | Promise<void>;
  }

  let { selectedId = null, onselect, ondrop }: Props = $props();

  // hoveredClassId tracks which class row currently sits under the dragged
  // crop. svelte-dnd-action fires `consider` events whenever the active
  // drop zone changes; we use that to highlight ONE row clearly so the
  // user always knows where the drop will land.
  let hoveredClassId = $state<number | null>(null);
  // dragActive — true while any consider event is in flight on any row.
  // Drives the sticky drop-banner across the whole sidebar.
  let dragActive = $state<boolean>(false);
  let dragClearTimer: ReturnType<typeof setTimeout> | null = null;

  function markDragActive(): void {
    dragActive = true;
    if (dragClearTimer) clearTimeout(dragClearTimer);
  }

  function scheduleDragClear(): void {
    if (dragClearTimer) clearTimeout(dragClearTimer);
    // svelte-dnd-action fires consider with items=[] when leaving a zone.
    // Other zones may still receive the drag; defer the clear so we don't
    // flicker the banner off between rows.
    dragClearTimer = setTimeout(() => {
      dragActive = false;
      hoveredClassId = null;
      dragClearTimer = null;
    }, 80);
  }

  // svelte-dnd-action drop-only zones use ``items: []`` + ``dragDisabled``.
  // The library's consider events DO carry the dragged crop ids (we see
  // them while hovering), but its finalize events arrive with an empty
  // items array because we never persist the shadow item into the
  // zone's state. So we capture the dragged ids during consider and
  // use that captured snapshot on finalize.
  //
  // Why not just track the source zone's `dragIds` instead?  The source
  // (the cluster grid) is on a different page; this sidebar lives in
  // the layout. The consider/finalize pair on each row is the only
  // signal we have here.
  let pendingDroppedIds: string[] = $state([]);

  function makeFinalize(cls: OpClass) {
    return (e: CustomEvent): void => {
      const { items } = e.detail as { items: Array<{ id: string }> };
      hoveredClassId = null;
      dragActive = false;
      if (dragClearTimer) {
        clearTimeout(dragClearTimer);
        dragClearTimer = null;
      }
      // Prefer the live items array if the library populated it; fall
      // back to the snapshot we captured during the consider phase.
      const liveIds = (items ?? [])
        .map((it) => it.id)
        .filter((id) => typeof id === 'string');
      const droppedIds = liveIds.length > 0 ? liveIds : pendingDroppedIds;
      pendingDroppedIds = [];
      if (!ondrop) return;
      if (droppedIds.length === 0) return;
      void ondrop(cls, droppedIds);
    };
  }

  function makeConsider(cls: OpClass) {
    return (e: CustomEvent): void => {
      const { items } = e.detail as { items: Array<{ id: string }> };
      if (items.length > 0) {
        hoveredClassId = cls.id;
        // Snapshot the dragged crop ids — finalize will receive items=[]
        // because the zone never accepts the shadow item permanently.
        pendingDroppedIds = items
          .map((it) => it.id)
          .filter((id) => typeof id === 'string');
        markDragActive();
      } else if (hoveredClassId === cls.id) {
        hoveredClassId = null;
        scheduleDragClear();
      }
    };
  }

  let query = $state<string>('');
  let modalOpen = $state<boolean>(false);

  const hoveredClass = $derived(
    hoveredClassId == null ? null : classesStore.byId(hoveredClassId),
  );

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? classesStore.classes.filter(
          (c) =>
            c.name.toLowerCase().includes(q) ||
            (c.group ?? '').toLowerCase().includes(q),
        )
      : classesStore.classes;
    // Sort by cluster bucket size desc — matches what the chip shows so
    // operators can scan top-down to find the biggest backlog.
    return [...list].sort((a, b) => (b.cluster_size ?? 0) - (a.cluster_size ?? 0));
  });

  // Chip color/threshold logic lives in $lib/adequacy so /classes,
  // /clusters, the dashboard, and this sidebar all paint the same
  // count the same way.
</script>

<aside class="flex h-full w-64 flex-col border-r border-zinc-800 bg-zinc-950">
  <div class="border-b border-zinc-800 p-3">
    <h2 class="mb-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">Classes</h2>
    <input
      type="search"
      bind:value={query}
      placeholder="Filter..."
      class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-blue-500 focus:outline-none"
    />
  </div>

  {#if dragActive && ondrop}
    <div class="border-b border-blue-500/30 bg-blue-500/10 px-3 py-2 text-center text-[11px] font-medium text-blue-200">
      {#if hoveredClass}
        Drop on <span class="font-semibold text-blue-100">{hoveredClass.name}</span>
      {:else}
        Drag onto a class row to label
      {/if}
    </div>
  {/if}

  <div class="flex-1 overflow-y-auto">
    {#if classesStore.loading && classesStore.classes.length === 0}
      <div class="p-4 text-sm text-zinc-500">Loading classes...</div>
    {:else if classesStore.error}
      <div class="p-4 text-sm text-red-300">API unavailable</div>
    {:else if filtered.length === 0}
      <div class="p-4 text-sm text-zinc-500">No classes</div>
    {:else}
      <ul class="py-1">
        <li>
          <button
            type="button"
            class="flex w-full items-center justify-between px-3 py-1.5 text-left text-sm hover:bg-zinc-900 {selectedId ===
            null
              ? 'bg-zinc-800 text-white'
              : 'text-zinc-300'}"
            onclick={() => onselect(null)}
          >
            <span>All classes</span>
          </button>
        </li>
        {#each filtered as cls (cls.id)}
          {@const isHover = hoveredClassId === cls.id}
          <li>
            <div
              class="flex w-full items-center justify-between gap-2 transition-colors
                {isHover
                ? 'bg-blue-500/30 ring-2 ring-inset ring-blue-400'
                : selectedId === cls.id
                  ? 'bg-zinc-800'
                  : 'hover:bg-zinc-900'}"
              use:dndzone={{
                items: [],
                type: 'op-crop',
                flipDurationMs: 0,
                morphDisabled: true,
                // centreDraggedOnCursor placed the shadow on top of the
                // narrow row, masking the underlying hit target — drops
                // and hover-highlight became flaky. Letting the shadow
                // trail the cursor keeps each row fully hittable.
                dropTargetStyle: { outline: 'none' },
                dropFromOthersDisabled: !ondrop,
                dragDisabled: true,
              }}
              onconsider={makeConsider(cls)}
              onfinalize={makeFinalize(cls)}
            >
              <button
                type="button"
                class="flex min-h-9 grow items-center justify-between gap-2 px-3 py-2 text-left text-sm {selectedId ===
                cls.id
                  ? 'text-white'
                  : 'text-zinc-300'}"
                onclick={() => onselect(cls)}
                title={cls.group ? `${cls.group} / ${cls.name}` : cls.name}
              >
                <span class="flex grow items-center gap-1.5 truncate">
                  {#if cls.hotkey_letter}
                    <kbd
                      class="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase
                             text-blue-300"
                      title="Press '{cls.hotkey_letter}' to assign selected crops to {cls.name}"
                    >
                      {cls.hotkey_letter}
                    </kbd>
                  {/if}
                  <span class="truncate">{cls.name}</span>
                </span>
                <span
                  class="rounded-md border px-1.5 py-0.5 font-mono text-xs {adequacyChipClass(
                    cls.validated_count ?? 0,
                  )}"
                  title="Cluster bucket size — total crops on /clusters/{cls.id}.&#10;{cls.validated_count ?? 0} of {cls.count ?? 0} labeled crops are human-validated.&#10;Chip color reflects validated-count adequacy."
                >
                  {(cls.cluster_size ?? 0).toLocaleString()}
                </span>
              </button>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <div class="border-t border-zinc-800 p-2">
    <button type="button" class="btn w-full justify-center" onclick={() => (modalOpen = true)}>
      + Add Class
    </button>
  </div>
</aside>

<AddClassModal open={modalOpen} onclose={() => (modalOpen = false)} />
