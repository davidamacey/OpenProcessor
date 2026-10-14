/**
 * The /clusters/[id] actions: class assignment and advance, the M-key move
 * picker, flag / ignore / refine / Run VLM, the drag-and-drop handlers and
 * every keyboard action. Optimistic writes go through the shared
 * `createClusterActionController`; the move-picker state stays together
 * here with the drag set it races against.
 */
import { SOURCES, TRIGGERS } from 'svelte-dnd-action';
import {
  apiErrorText,
  flagNeedsNewClass,
  pollAutoLabelJob,
  refineCluster,
  runVlmOnCluster,
  type AutoLabelJobState,
  type ExcludeReason,
} from '$lib/api';
import { createClusterActionController } from '$lib/clusters/clusterController.svelte';
import type { ClusterDetailController } from '$lib/clusters/clusterDetailController.svelte';
import type { Crop } from '$lib/types';
import { vlmRunErrorText } from '$lib/vlm/runErrors';
import { dropOnClassStore } from '$stores/dropOnClass.svelte';
import { keyboardStore } from '$stores/keyboard.svelte';
import { keymapStore } from '$stores/keymap.svelte';
import { toastStore } from '$stores/toast.svelte';

export function createClusterDetailActions(view: ClusterDetailController) {
  const { cropPager, sel, exclusionGuard, grid, loadFirst, loadMore, selectAllPage } =
    view;

  // Class dropdown
  let confirmClassId = $state<number | null>(null);

  // ---- Move/DnD state ----------------------------------------------------
  // Recent target cluster ids the labeler has typed in this session (LRU 8).
  let recentTargets = $state<number[]>([]);
  // The grid is rendered as one dndzone PER sub-cluster group so the
  // refine (AHC) delineation can show headers between groups without
  // putting non-item elements inside a dndzone (svelte-dnd-action maps
  // each zone's direct children 1:1 to its items array — interleaved
  // separators break that and the drag). When not grouping, there's a
  // single '__all__' group == the whole cluster. Each group.items is a
  // separate mutable array the dnd action shuffles during a drag.
  //
  // The grouping itself is DERIVED from the pager (see gridGroups below), not
  // a snapshot the dnd handlers own — see gridGroups.svelte.ts for why that
  // distinction is the whole fix for "moved crops flicker back and stay".
  // Declared after filteredCrops/groupBySubcluster, which it reads.
  // Tracks the in-flight drag's payload (one or many crops). Set on dragStart.
  let dragIds = $state<string[]>([]);
  // Inline cluster-picker (opened by M-key) state.
  let movePickerOpen = $state<boolean>(false);
  let movePickerValue = $state<string>('');
  let movePickerInput = $state<HTMLInputElement | null>(null);

  // Action controller (see clusterController.svelte.ts) — owns every
  // optimistic mutation below; shares `exclusionGuard` (declared next to
  // `cropPager` above) rather than a second copy of the race guard.
  // filteredCrops/rememberTarget/loadFirst are captured by closure and
  // resolved lazily, so this can sit ahead of their declarations further
  // down the script.
  const controller = createClusterActionController({
    cropPager,
    sel,
    exclusionGuard,
    resetGrid: () => grid.reset(),
    getDragIds: () => dragIds,
    setDragIds: (ids) => {
      dragIds = ids;
    },
    getVisibleCrops: () => view.filteredCrops,
    getClusterId: () => view.clusterId,
    rememberTarget: (id) => rememberTarget(id),
    loadFirst,
  });

  // Wire the layout-level ClassSidebar's drop targets to bulk-label the
  // currently-selected crops. Registered on mount, unregistered on
  // teardown so other pages don't accidentally receive cluster-page
  // drop dispatches.
  $effect(() => {
    // Body (Finder-pattern drag-set resolution, optimistic removal,
    // excludedCropIds claim/release, conflict resync) now lives in
    // clusterController.svelte.ts's handleClassDrop.
    const off = dropOnClassStore.register((cls, droppedIds) =>
      controller.handleClassDrop(cls, droppedIds),
    );
    return off;
  });

  // ---------------- mutations ----------------

  /**
   * Flag the selection for curator review as needing a class the registry
   * doesn't have yet. Shared by the Shift+N binding and the toolbar chip.
   */
  async function flagSelectedForNewClass(): Promise<void> {
    const ids = [...sel.ids];
    if (ids.length === 0) {
      toastStore.info('Select crops first to flag for new class.');
      return;
    }
    // Note is optional; skip the blocking prompt and flag immediately.
    // Curator can add notes later via /review when triaging the flagged
    // queue, where it doesn't interrupt the labeling cadence.
    try {
      const res = await flagNeedsNewClass(ids, '');
      toastStore.success(
        `${res.flagged} flagged for new-class review${res.errors ? ` (${res.errors} errors)` : ''}`,
      );
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
      sel.ids = new Set();
    } catch (e) {
      toastStore.error(`Flag failed: ${apiErrorText(e)}`);
    }
  }

  // In-flight guards so the operator gets feedback and can't double-fire
  // these long-running cluster ops.
  let refining = $state<boolean>(false);
  let vlmRunning = $state<boolean>(false);
  // Live status while POST /vlm/label_cluster/{id}'s job runs, polled via
  // pollAutoLabelJob — rendered as a compact inline stage/progress string
  // next to the Run VLM button.
  let vlmJob = $state<AutoLabelJobState | null>(null);
  // the endpoint this one run uses (`null` = the project default) and
  // the acknowledgement for an external one.
  let vlmChoice = $state<string | null>(null);
  let vlmAck = $state<boolean>(false);

  async function runVlm(): Promise<void> {
    if (vlmRunning) return;
    // p2/M7: a nag-confirm, unlike the one-keystroke label/move/discard
    // actions above (which Z undoes instantly) — this kicks off a
    // background VLM sweep over every unvalidated member of the cluster
    // with no undo, so a misclick isn't free.
    if (
      !window.confirm(
        `Run the VLM over every unvalidated crop in cluster #${view.clusterId}?`,
      )
    ) {
      return;
    }
    vlmRunning = true;
    vlmJob = null;
    try {
      vlmJob = await runVlmOnCluster(view.clusterId, {
        vlm: vlmChoice,
        acknowledgeExternal: vlmAck,
      });
      // M7: poll only the job we just started — see pollAutoLabelJob's
      // `expectedJobId` doc comment for why this matters.
      const final = await pollAutoLabelJob(
        (j) => (vlmJob = j),
        undefined,
        undefined,
        vlmJob.job_id,
      );
      const stages = (final.result?.stages ?? {}) as Record<
        string,
        Record<string, unknown>
      >;
      const vlm = stages.vlm ?? {};
      if (final.status === 'failed') {
        toastStore.error(`VLM run failed: ${final.error ?? 'unknown error'}`);
      } else {
        toastStore.success(
          `VLM labeled ${Number(vlm.predicted ?? 0)} crops (${Number(vlm.updated ?? 0)} updated).`,
        );
      }
    } catch (e) {
      toastStore.error(`VLM run failed: ${vlmRunErrorText(e)}`);
    } finally {
      vlmRunning = false;
    }
  }

  // -- Ignore / exclude --------------------------------------------------
  // Excluded crops drop out of training + clustering (reversible). Actual
  // request + excludedCropIds claim/release + last-batch bookkeeping now
  // live in clusterController.svelte.ts (ignoreSelected/undoIgnore);
  // ignoreMenuOpen/EXCLUDE_REASONS stay here — they're pure toolbar UI
  // state, not action logic.
  let ignoreMenuOpen = $state<boolean>(false);
  const EXCLUDE_REASONS: { value: ExcludeReason; label: string }[] = [
    { value: 'ignore', label: 'Ignore (generic)' },
    { value: 'blurry', label: 'Blurry' },
    { value: 'unidentifiable', label: 'Unidentifiable' },
    { value: 'not_the_subject', label: 'Not the subject' },
    { value: 'partial_crop', label: 'Partial crop' },
  ];

  async function ignoreSelected(reason: ExcludeReason = 'ignore'): Promise<void> {
    if (sel.size > 0) ignoreMenuOpen = false;
    await controller.ignoreSelected(reason);
  }

  async function refine(): Promise<void> {
    if (refining) return;
    refining = true;
    try {
      const res = await refineCluster(view.clusterId);
      toastStore.success(`Refine produced ${res.n_subclusters ?? 0} sub-clusters.`);
      await loadFirst();
    } catch (e) {
      toastStore.error(`Refine failed: ${apiErrorText(e)}`);
    } finally {
      refining = false;
    }
  }

  async function confirmSelected(): Promise<void> {
    if (confirmClassId == null) {
      toastStore.warn('Pick a class first.');
      return;
    }
    await controller.assignClassToSelected(confirmClassId);
    void advance();
  }

  async function advance(): Promise<void> {
    // Advance: jump to next unvalidated crop. If we're at the end of what's
    // loaded but more pages exist, fetch them; otherwise tell the user.
    const next = view.filteredCrops.find((c) => !c.class_validated && !sel.has(c.id));
    if (next) {
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
      sel.ids = new Set([next.id]);
    } else if (cropPager.hasMore) {
      await loadMore();
      const nextAfterLoad = view.filteredCrops.find(
        (c) => !c.class_validated && !sel.has(c.id),
      );
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
      if (nextAfterLoad) sel.ids = new Set([nextAfterLoad.id]);
    } else {
      toastStore.info('End of cluster.');
    }
  }

  // ---------------- move (DnD + hotkey) ----------------

  function rememberTarget(id: number): void {
    const next = [id, ...recentTargets.filter((x) => x !== id)].slice(0, 8);
    recentTargets = next;
  }

  // moveCropIds (the excludedCropIds claim/release + conflict resync +
  // revert-on-failure) now lives in clusterController.svelte.ts.

  function openMovePicker(): void {
    if (sel.size === 0) {
      toastStore.warn('Select crops to move first.');
      return;
    }
    movePickerValue = '';
    movePickerOpen = true;
    queueMicrotask(() => movePickerInput?.focus());
  }

  function cancelMovePicker(): void {
    movePickerOpen = false;
    movePickerValue = '';
  }

  async function confirmMovePicker(): Promise<void> {
    const id = Number(movePickerValue);
    // A negative (or otherwise invalid-as-a-cluster-id) value is no
    // longer rejected client-side — the server owns that rule and
    // returns its own 400 detail (see moveCropIds/M10), which the
    // toast then shows verbatim. Only a genuinely non-numeric entry is
    // caught here, since NaN can't even be sent as a JSON int.
    if (!Number.isFinite(id)) {
      toastStore.error('Cluster id must be a number.');
      return;
    }
    movePickerOpen = false;
    await controller.moveCropIds([...sel.ids], id);
  }

  /**
   * dnd-action handlers. The grid is a draggable-only zone: removing items
   * is fine (they animate out), adding is rejected. The real label RPC
   * fires from the layout ClassSidebar's own finalize, routed back here
   * through dropOnClassStore.
   */
  function onGroupConsider(
    key: string,
    e: CustomEvent<{
      items: Crop[];
      info: { id: string; trigger: TRIGGERS; source: SOURCES };
    }>,
  ): void {
    // The `!dragIds.includes` guard makes this block run once per drag
    // (consider fires repeatedly). Finder pattern: grabbing any selected
    // card drags the whole selection; grabbing an unselected card
    // replaces the selection with just that one. This is the only place
    // the multi-drag set is captured — the ClassSidebar's own dnd events
    // only see the one shadow item being hovered.
    const draggedId = e.detail.info?.id;
    if (draggedId && !dragIds.includes(draggedId)) {
      if (sel.has(draggedId) && sel.size > 1) {
        dragIds = [...sel.ids];
      } else {
        dragIds = [draggedId];
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
        sel.ids = new Set([draggedId]);
      }
    }
    grid.setZoneItems(key, e.detail.items);
  }

  function onGroupFinalize(
    _key: string,
    e: CustomEvent<{ items: Crop[]; info: { trigger: TRIGGERS } }>,
  ): void {
    // Drop the drag-local override outright rather than adopting
    // `e.detail.items`. That payload is the dnd library's own pre-drop
    // snapshot of the zone — instrumented drops show it arriving with one
    // more crop than cropPager.items already holds, because the ClassSidebar
    // finalize (which fires first) has already run the optimistic removal.
    // Adopting it is exactly how a labelled-away crop got painted back into
    // its old slot. Nothing here is worth keeping either way: real label RPCs
    // fire from the sidebar's finalize, and dropping into another grid
    // sub-group has no semantic meaning (all groups share cluster_id).
    grid.reset();
    if (e.detail.info.trigger === TRIGGERS.DROPPED_INTO_ANOTHER) {
      // The crop landed in the sidebar zone. The ClassSidebar finalize
      // that dispatches this drop reads dragIds synchronously (before its
      // first await) as the authoritative multi-drag set, and the drop
      // handler clears it after use — so clearing here synchronously
      // would strand the sidebar with only its single shadow item. The
      // deferred clear covers the no-dispatch edge, where the sidebar
      // finalize early-returns because both its live items array and its
      // pendingDroppedIds snapshot are empty.
      setTimeout(() => {
        dragIds = [];
      }, 0);
    } else {
      // DROPPED_INTO_ZONE (in-grid reorder — no sidebar dispatch follows),
      // DROPPED_OUTSIDE_OF_ANY, DRAG_STOPPED.
      dragIds = [];
    }
  }

  // ---------------- shortcuts ----------------

  // Every key this page prints comes from the keymap (never a literal).
  const kg = (actionId: string) => keymapStore.glyph(actionId);
  const kc = (actionId: string) => keymapStore.compactGlyph(actionId);

  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (actionId: string, fn: () => void | Promise<void>) =>
      offs.push(keyboardStore.registerAction(actionId, () => void fn(), 'cluster'));

    // Per-class letter hotkeys (configured on /classes) are routed
    // through dropOnClassStore by the layout-level keydown listener;
    // see src/routes/+layout.svelte. The legacy 1..0 top-N scheme has
    // been removed — one binding scheme means no "what does this key
    // do here?" friction.

    reg('cluster.confirm', confirmSelected);
    reg('cluster.accept_all_vlm', controller.acceptAllVlmOnPage);
    reg('cluster.accept_vlm', async () => {
      const ids = [...sel.ids];
      for (const id of ids) {
        const c = cropPager.items.find((cc) => cc.id === id);
        if (c) await controller.acceptVlmForCrop(c);
      }
    });
    reg('cluster.skip', async () => {
      toastStore.info('Skipped.');
      await advance();
    });
    // Its default is 'shift+n' (keymapFallback.ts), not 'N': combos are
    // lowercased, so 'N' would collapse onto the skip binding above.
    reg('cluster.flag_new_class', flagSelectedForNewClass);
    reg('cluster.discard', controller.discardSelected);
    reg('cluster.undo', controller.undoLast);
    reg('cluster.ignore', () => void ignoreSelected('ignore'));
    reg('cluster.unignore', controller.undoIgnore);
    reg('cluster.select_all', selectAllPage);
    // Arrow keys navigate within the loaded grid. With infinite scroll the
    // next-page concept is gone — left/right move selection by one position
    // in the visible filtered list.
    reg('cluster.prev', () => {
      const ids = view.filteredCrops.map((c) => c.id);
      if (ids.length === 0) return;
      const cur = ids.findIndex((id) => sel.has(id));
      const prev = cur <= 0 ? ids.length - 1 : cur - 1;
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
      sel.ids = new Set([ids[prev]!]);
      sel.anchorId = ids[prev]!;
    });
    reg('cluster.next', () => {
      const ids = view.filteredCrops.map((c) => c.id);
      if (ids.length === 0) return;
      const cur = ids.findIndex((id) => sel.has(id));
      const next = cur < 0 || cur >= ids.length - 1 ? 0 : cur + 1;
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
      sel.ids = new Set([ids[next]!]);
      sel.anchorId = ids[next]!;
      if (next === ids.length - 1 && cropPager.hasMore) void loadMore();
    });
    reg('cluster.move', openMovePicker);
    reg('cluster.cancel', () => {
      if (movePickerOpen) {
        cancelMovePicker();
        return;
      }
      if (dragIds.length > 0) {
        // svelte-dnd-action can't cancel an in-progress POINTER drag —
        // its only Escape handling is gated on the keyboard-drag (aria)
        // module's isDragging flag. So all Escape can do here is discard
        // the captured multi-drag set and restore the grid layout; the
        // drag itself ends when the user releases the pointer. Do NOT
        // re-dispatch a synthetic Escape on window: dispatchEvent is
        // synchronous, so it re-enters this same handler with dragIds
        // still populated and recurses until the stack blows.
        dragIds = [];
        grid.reset();
        return;
      }
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- copied into the selection's own SvelteSet by its setter
      sel.ids = new Set();
    });

    return () => offs.forEach((off) => off());
  });

  return {
    get confirmClassId() {
      return confirmClassId;
    },
    set confirmClassId(v: number | null) {
      confirmClassId = v;
    },
    get dragIds() {
      return dragIds;
    },
    get movePickerOpen() {
      return movePickerOpen;
    },
    get movePickerValue() {
      return movePickerValue;
    },
    set movePickerValue(v: string) {
      movePickerValue = v;
    },
    get movePickerInput() {
      return movePickerInput;
    },
    set movePickerInput(v: HTMLInputElement | null) {
      movePickerInput = v;
    },
    get refining() {
      return refining;
    },
    get vlmRunning() {
      return vlmRunning;
    },
    get vlmJob() {
      return vlmJob;
    },
    get vlmChoice() {
      return vlmChoice;
    },
    set vlmChoice(v: string | null) {
      vlmChoice = v;
    },
    get vlmAck() {
      return vlmAck;
    },
    set vlmAck(v: boolean) {
      vlmAck = v;
    },
    get ignoreMenuOpen() {
      return ignoreMenuOpen;
    },
    set ignoreMenuOpen(v: boolean) {
      ignoreMenuOpen = v;
    },
    controller,
    EXCLUDE_REASONS,
    kg,
    kc,
    flagSelectedForNewClass,
    runVlm,
    ignoreSelected,
    refine,
    confirmSelected,
    openMovePicker,
    cancelMovePicker,
    confirmMovePicker,
    onGroupConsider,
    onGroupFinalize,
  };
}

export type ClusterDetailActions = ReturnType<typeof createClusterDetailActions>;
