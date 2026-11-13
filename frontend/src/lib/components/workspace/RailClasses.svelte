<script lang="ts">
  import { dndzone } from 'svelte-dnd-action';
  import ClassPicker from '$components/ClassPicker.svelte';
  import { adequacyChipClass } from '$lib/adequacy';
  import { isAssignableClass } from '$lib/classVisibility';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import type { RegistryClass } from '$lib/types_classes';

  // The classes half of the rail: the served registry in the served order, with
  // its served counts. A click filters the Workspace by the class name; a card
  // dragged onto a row assigns it.
  let { ws }: { ws: WorkspaceController } = $props();

  const selectedName = $derived(
    ws.filter.classNames.length === 1 ? ws.filter.classNames[0]! : null,
  );

  function onselect(cls: RegistryClass | null): void {
    ws.filter.classNames = cls ? [cls.name] : [];
    ws.applyFilter();
  }

  function ondrop(cls: RegistryClass, droppedIds: string[]): void {
    void dropOnClassStore.dispatch(cls, droppedIds);
  }

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

  function makeFinalize(cls: RegistryClass) {
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
      if (droppedIds.length === 0) return;
      void ondrop(cls, droppedIds);
    };
  }

  function makeConsider(cls: RegistryClass) {
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

  let createOpen = $state<boolean>(false);

  const hoveredClass = $derived(
    hoveredClassId == null ? null : classesStore.byId(hoveredClassId),
  );

  const classes = $derived(classesStore.classes.filter(isAssignableClass));
</script>

<!-- Narrow screens: a short strip above the content, so the page keeps the full width.
     md and up: the fixed 16rem column beside it. -->
<section class="relative flex min-h-0 flex-1 flex-col" aria-label="Classes">
  <h2 class="p-3 pb-1 text-meta font-semibold text-fg-2">Classes</h2>
  {#if classesStore.thresholds.warn_below > 0}
    <!-- The count chips use the served adequacy tiers; this says what each colour means. -->
    <p class="px-3 pb-2 text-caption text-fg-2" data-testid="class-count-legend">
      Count colour: validated examples.
      <span class="text-danger-text">Red</span> under {classesStore.thresholds
        .block_below},
      <span class="text-warn-text">amber</span> under {classesStore.thresholds
        .warn_below},
      <span class="text-success-text">green</span> from there.
    </p>
  {/if}

  <!-- Drop hint is an ABSOLUTE overlay (pointer-events-none) so it can't
       reflow the class list mid-drag. Previously it was in flow and its
       appearance pushed every folder row down, so the drop landed on the
       wrong row / above the name. Overlaying keeps every row's hit target
       fixed under the cursor. -->
  {#if dragActive}
    <div
      class="pointer-events-none absolute inset-x-0 top-[60px] z-20 border-y border-accent/40 bg-accent-fill/90 px-3 py-1.5 text-center text-caption font-medium text-accent-text shadow-lg backdrop-blur"
    >
      {#if hoveredClass}
        Drop on <span class="font-semibold text-white">{hoveredClass.name}</span>
      {:else}
        Drag onto a class row to label
      {/if}
    </div>
  {/if}

  <div class="flex-1 overflow-y-auto">
    {#if classesStore.loading && classesStore.classes.length === 0}
      <div class="p-4 text-body text-fg-3">Loading classes...</div>
    {:else if classesStore.error}
      <div class="p-4 text-body text-danger-text">API unavailable</div>
    {:else if classes.length === 0}
      <div class="p-4 text-body text-fg-3">No classes</div>
    {:else}
      <ul class="py-1">
        <li>
          <button
            type="button"
            class="flex w-full items-center justify-between px-3 py-1.5 text-left text-body hover:bg-surface-1 {selectedName ===
            null
              ? 'bg-surface-2 text-white'
              : 'text-fg-2'}"
            onclick={() => onselect(null)}
          >
            <span>All classes</span>
          </button>
        </li>
        {#each classes as cls (cls.id)}
          {@const isHover = hoveredClassId === cls.id}
          <li>
            <div
              class="crop-drop-row flex w-full items-center justify-between gap-2 transition-colors
                {isHover
                ? 'bg-accent/30 ring-2 ring-inset ring-accent'
                : selectedName === cls.name
                  ? 'bg-surface-2'
                  : 'hover:bg-surface-1'}"
              use:dndzone={{
                items: [],
                type: 'crop-card',
                flipDurationMs: 0,
                morphDisabled: true,
                // centreDraggedOnCursor placed the shadow on top of the
                // narrow row, masking the underlying hit target — drops
                // and hover-highlight became flaky. Letting the shadow
                // trail the cursor keeps each row fully hittable.
                dropTargetStyle: { outline: 'none' },
                dropFromOthersDisabled: false,
                dragDisabled: true,
              }}
              onconsider={makeConsider(cls)}
              onfinalize={makeFinalize(cls)}
            >
              <!-- The dnd action stamps role="listitem" on the zone's child; a wrapper
                   keeps that role off the button. -->
              <div class="flex min-w-0 grow">
                <button
                  type="button"
                  class="flex min-h-9 grow items-center justify-between gap-2 px-3 py-2 text-left text-body {selectedName ===
                  cls.name
                    ? 'text-white'
                    : 'text-fg-2'}"
                  onclick={() => onselect(cls)}
                  title={cls.group ? `${cls.group} / ${cls.name}` : cls.name}
                >
                  <span class="flex grow items-center gap-1.5 truncate">
                    {#if cls.hotkey_letter}
                      <kbd
                        class="rounded bg-surface-2 px-1 py-0.5 font-mono text-caption text-accent-text"
                        title="Press '{cls.hotkey_letter}' to assign selected crops to {cls.name}"
                      >
                        {cls.hotkey_letter}
                      </kbd>
                    {/if}
                    <span class="truncate">{cls.name}</span>
                  </span>
                  <span
                    class="rounded-md border px-1.5 py-0.5 font-mono text-meta {adequacyChipClass(
                      cls.adequacy,
                    )}"
                    title="{cls.count.toLocaleString()} items carry this class.&#10;{cls.validated_count ??
                      0} of {cls.count ??
                      0} are human-validated.&#10;Chip color reflects the server's adequacy tier ({cls.adequacy ??
                      'unknown'})."
                  >
                    {cls.count.toLocaleString()}
                  </span>
                </button>
              </div>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <div class="border-t border-line p-2">
    <button
      type="button"
      class="btn w-full justify-center"
      onclick={() => (createOpen = true)}
    >
      + New class
    </button>
  </div>
</section>

<ClassPicker
  open={createOpen}
  mode="create"
  onpick={() => (createOpen = false)}
  onclose={() => (createOpen = false)}
/>

<style>
  /* svelte-dnd-action injects a shadow placeholder element into the
     hovered drop zone. In these narrow flex class-rows it became a
     flex child and pushed the name + count down, shifting the visible
     drop target away from the cursor (you had to aim above the name).
     Take the shadow out of flow: overlay it on the row instead so the
     row never reflows and the whole row stays a stable hit target. */
  .crop-drop-row {
    position: relative;
  }
  .crop-drop-row :global([data-is-dnd-shadow-item-hint]) {
    position: absolute;
    inset: 0;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    pointer-events: none;
    opacity: 0;
  }
</style>
