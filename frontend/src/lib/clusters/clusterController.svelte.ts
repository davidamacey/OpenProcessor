/**
 * `/clusters/[id]` action controller — assign / drop-on-class / VLM accept-
 * reject / discard / undo / ignore / move, extracted out of
 * `clusters/[id]/+page.svelte` (docs/design/test-audit-2026-09-24.md P1-4,
 * second pass) so this hot path's optimistic-mutation-then-
 * rollback-or-confirm logic is unit-testable without mounting the page —
 * same motivation, and the same factory-function convention (state via
 * closures, not a class), as `$lib/review/reviewController.svelte.ts`.
 *
 * The page still owns `cropPager` / `sel` (crop selection) / `dragIds` /
 * the grid's drag-local override, and hands them in by reference/accessor
 * so there is exactly one copy of each — the same reasoning
 * reviewController's header comment gives for `queue`/`cursor`.
 *
 * `createExclusionGuard()` below is the page-local set (now
 * `ExclusionGuard`) that closes the stale-fetch race (a same-cluster GET
 * already in flight, or triggered fresh by the SSE live-refresh effect,
 * must not resurrect a crop this controller already knows left): every
 * mutation that removes a crop from the grid claims its ids before the
 * awaited request settles, and releases them on revert/undo/conflict.
 * It's its own factory, not folded into `createClusterActionController`,
 * so the page can build it — and wire `cropPager`'s `accept` option to
 * it — before the rest of the controller exists (see its own doc
 * comment for why).
 */

import {
  ApiError,
  bulkLabel,
  discardCrop,
  discardCropsBatch,
  excludeCrops,
  moveCropsToCluster,
  putCropLabel,
  unexcludeCrops,
  vlmDismissCrop,
  type ExcludeReason,
} from '$lib/api';
import type { Pager } from '$lib/pager.svelte';
import type { Selection } from '$lib/selection.svelte';
import type { Crop, RegistryClass } from '$lib/types';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';

export interface ExclusionGuard {
  /** Wire straight into `cropPager`'s `accept` pager option. */
  accept(crop: Crop): boolean;
  claim(ids: Iterable<string>): void;
  release(ids: Iterable<string>): void;
}

/**
 * The stale-fetch-race guard, split out of the controller as its own
 * factory so the page can create it — and wire `cropPager`'s `accept`
 * option to it — *before* the rest of the controller exists (which
 * itself needs `cropPager` as an input). Without this split the page
 * would need a temporal-dead-zone forward reference (`let controller`,
 * assigned later) just to close the circular dependency, which
 * `svelte-check` flags as a non-reactive `$state` update.
 */
export function createExclusionGuard(): ExclusionGuard {
  const ids = new Set<string>();
  return {
    accept: (crop) => !ids.has(crop.id),
    claim: (idsToClaim) => {
      for (const id of idsToClaim) ids.add(id);
    },
    release: (idsToRelease) => {
      for (const id of idsToRelease) ids.delete(id);
    },
  };
}

export interface ClusterActionControllerOptions {
  /** The page's crop pager — items/total are mutated in place. */
  cropPager: Pager<Crop>;
  /** The page's crop-grid selection. */
  sel: Selection;
  /** Shared with `cropPager`'s `accept` option — see `createExclusionGuard`. */
  exclusionGuard: ExclusionGuard;
  /** Drop the dnd grid's drag-local override so it repaints from the
   *  (already-corrected) pager instead of a stale dnd snapshot. */
  resetGrid: () => void;
  getDragIds: () => string[];
  setDragIds: (ids: string[]) => void;
  /** Currently-rendered crops (post sub-cluster/filter), for picking
   *  accept-all-on-page targets. */
  getVisibleCrops: () => Crop[];
  getClusterId: () => number;
  /** LRU-remember a move target for the move-picker's recents row. */
  rememberTarget: (id: number) => void;
  /** Re-fetch page 1 — used to resync after a worker-conflict. */
  loadFirst: () => Promise<void>;
}

export function createClusterActionController(opts: ClusterActionControllerOptions) {
  const {
    cropPager,
    sel,
    exclusionGuard,
    resetGrid,
    getDragIds,
    setDragIds,
    getVisibleCrops,
    getClusterId,
    rememberTarget,
    loadFirst,
  } = opts;

  // Self-contained ignore/un-ignore history (self-contained: does not use
  // the label-revert undoStore/Z, which only handles class-label writes).
  let lastExcludedIds: string[] = [];

  function applyLocalLabel(id: string, classId: number, className: string | null): void {
    cropPager.items = cropPager.items.map((c) =>
      c.id === id
        ? {
            ...c,
            class_id: classId,
            class_name: className,
            label_validated: true,
            class_validated: true,
            label_source: 'human_confirmed',
          }
        : c,
    );
  }

  /** Roll back an optimistic label that the server rejected. */
  function revertLocalLabel(prior: Crop): void {
    cropPager.items = cropPager.items.map((c) => (c.id === prior.id ? prior : c));
  }

  async function assignClassToSelected(classId: number): Promise<void> {
    const ids = [...sel.ids];
    if (ids.length === 0) {
      toastStore.warn('Nothing selected.');
      return;
    }
    const cls = classesStore.byId(classId);
    if (!cls) {
      toastStore.error('Unknown class id ' + classId);
      return;
    }
    // No nag-confirm — undo is one keystroke (Z), so any mistake is
    // instantly reversible. The prior crops are kept only to roll back
    // the optimistic label if the write fails.
    const priors = cropPager.items.filter((c) => ids.includes(c.id));
    for (const id of ids) applyLocalLabel(id, classId, cls.name);
    try {
      if (ids.length === 1) {
        await putCropLabel(ids[0]!, classId);
        undoStore.recordWrites(ids);
      } else {
        const res = await bulkLabel(ids, classId);
        undoStore.recordWrites(res.updated_ids);
      }
      toastStore.success(`Labeled ${ids.length} crop${ids.length === 1 ? '' : 's'}.`);
      sel.ids = new Set();
    } catch (e) {
      toastStore.error(`Label failed: ${(e as Error).message}`);
      for (const prior of priors) revertLocalLabel(prior);
    }
  }

  /**
   * Register with `dropOnClassStore` — the sidebar-drop / class-hotkey
   * path shared by DnD and per-class hotkeys.
   */
  async function handleClassDrop(
    cls: RegistryClass,
    droppedIds: string[],
  ): Promise<void> {
    // Priority order matters. onGroupConsider captures the full drag
    // set at drag-start time (Finder pattern — grab any selected card
    // to drag all selected; grab an unselected card to drag just that
    // one), and the ClassSidebar's own finalize only ever sees the
    // single shadow item — so for a multi-drag drop droppedIds has 1 id
    // while dragIds has N. dragIds must therefore win. The `selected`
    // fallback serves the keyboard path: the layout's class-letter
    // listener dispatches with an empty droppedIds and no drag context.
    const dragIds = getDragIds();
    const ids =
      dragIds.length > 0
        ? [...dragIds]
        : droppedIds.length > 0
          ? droppedIds
          : [...sel.ids];
    // Consume dragIds here — leaving it populated would make the NEXT
    // hotkey press relabel the previously dragged crops.
    setDragIds([]);
    if (ids.length === 0) {
      toastStore.warn('Select or drag crops first, then press a class hotkey.');
      return;
    }
    // Optimistic: remove the dropped crops from the visible grid
    // BEFORE the await, so the labeling feels real-time.
    const snap = cropPager.items;
    const snapTotal = cropPager.total;
    const droppedSet = new Set(ids);
    cropPager.items = cropPager.items.filter((c) => !droppedSet.has(c.id));
    cropPager.total = Math.max(0, cropPager.total - ids.length);
    // Tear down any drag-local grid override *now*, so the grid repaints
    // from the (already-corrected) pager instead of from whatever
    // snapshot the in-flight drag left behind.
    resetGrid();
    sel.ids = new Set();
    // Claim these ids before the await resolves — see excludedCropIds
    // above. A stale/concurrent fetch that lands between now and the
    // await settling must not be allowed to resurrect them.
    exclusionGuard.claim(ids);
    try {
      const res = await bulkLabel(ids, cls.id);
      undoStore.recordWrites(res.updated_ids);
      const conflicts = res.conflicts?.length ?? 0;
      if (conflicts > 0) {
        // A concurrent worker (typically the VLM worker) beat us on
        // some crops. Only the crops that actually moved stay excluded
        // — the conflicted ones never left cluster_id, so they must be
        // allowed back.
        exclusionGuard.release(res.conflicts.map((c) => c.crop_id));
        toastStore.warn(
          `Labeled ${res.updated} of ${ids.length} → ${cls.name} (${conflicts} blocked by worker). Reloading.`,
        );
        void loadFirst();
      } else {
        toastStore.success(`Labeled ${res.updated ?? ids.length} → ${cls.name}.`);
      }
    } catch (e) {
      // Revert the optimistic mutation on hard failure.
      exclusionGuard.release(ids);
      cropPager.items = snap;
      cropPager.total = snapTotal;
      toastStore.error(`Label failed: ${(e as Error).message}`);
    }
  }

  async function acceptVlmForCrop(crop: Crop): Promise<void> {
    if (crop.vlm_suggested_class_id == null) return;
    applyLocalLabel(
      crop.id,
      crop.vlm_suggested_class_id,
      crop.vlm_suggested_class_name ?? null,
    );
    try {
      await putCropLabel(crop.id, crop.vlm_suggested_class_id);
      undoStore.recordWrites([crop.id]);
    } catch (e) {
      toastStore.error(`Accept VLM suggestion failed: ${(e as Error).message}`);
      revertLocalLabel(crop);
    }
  }

  async function rejectVlmForCrop(crop: Crop): Promise<void> {
    // Reject = dismiss the VLM's proposed class on the server
    // (POST {API_PREFIX}/crops/{id}/vlm_dismiss) and render the item it
    // returns. A 409 means there was already nothing to dismiss.
    try {
      const item = await vlmDismissCrop(crop.id);
      cropPager.items = cropPager.items.map((c) => (c.id === crop.id ? item : c));
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        toastStore.info('No VLM suggestion to reject.');
        return;
      }
      toastStore.error(`Reject VLM suggestion failed: ${(e as Error).message}`);
    }
  }

  async function acceptAllVlmOnPage(): Promise<void> {
    const targets = getVisibleCrops().filter(
      // G2: class_validated, not label_validated (which also flips true
      // on a region-only validation and would wrongly hide the accept
      // chip).
      (c) => c.vlm_suggested_class_id != null && !c.class_validated,
    );
    if (targets.length === 0) {
      toastStore.info('No VLM suggestions on this page.');
      return;
    }
    // Shift+Enter is already a deliberate two-finger gesture and Z undoes
    // it, so no nag-confirm. Group by class id for bulk_label.
    const groups = new Map<number, string[]>();
    // Prior crop per id so a failing group can be rolled back precisely —
    // a single try/catch around the whole loop left the failed group and
    // every later group locally green but never sent.
    const priors = new Map<string, Crop>();
    for (const t of targets) {
      const k = t.vlm_suggested_class_id!;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(t.id);
      priors.set(t.id, t);
      applyLocalLabel(t.id, k, t.vlm_suggested_class_name ?? null);
    }
    let ok = 0;
    let lastError: string | null = null;
    const failedIds: string[] = [];
    for (const [k, ids] of groups) {
      try {
        const res = await bulkLabel(ids, k);
        undoStore.recordWrites(res.updated_ids);
        ok += ids.length;
      } catch (e) {
        lastError = (e as Error).message;
        failedIds.push(...ids);
      }
    }
    if (failedIds.length === 0) {
      toastStore.success(`Accepted ${targets.length} suggestions.`);
      return;
    }
    for (const id of failedIds) {
      const prior = priors.get(id);
      if (prior) revertLocalLabel(prior);
    }
    toastStore.error(
      `Accepted ${ok}, failed ${failedIds.length} (reverted)${lastError ? `: ${lastError}` : '.'}`,
    );
  }

  /** D hotkey / toolbar action: discard the current selection. */
  async function discardSelected(): Promise<void> {
    const ids = [...sel.ids];
    if (ids.length === 0) return;
    // Discard is recorded like a label write, so it's reversible via Z
    // (POST {API_PREFIX}/crops/{id}/label/undo) — record undo entries
    // for exactly the ids the server actually discarded.
    let succeededIds: string[] = [];
    let failedCount = 0;
    let lastError: string | null = null;
    try {
      if (ids.length === 1) {
        await discardCrop(ids[0]!);
        succeededIds = [ids[0]!];
      } else {
        const res = await discardCropsBatch(ids);
        succeededIds = res.items.map((c) => c.id);
        failedCount = ids.length - succeededIds.length;
      }
    } catch (e) {
      lastError = (e as Error).message;
      failedCount = ids.length;
    }
    const succeededSet = new Set(succeededIds);
    cropPager.items = cropPager.items.filter((c) => !succeededSet.has(c.id));
    exclusionGuard.claim(succeededIds);
    undoStore.recordWrites(succeededIds);
    // Keep whatever didn't succeed visible and selected so the operator
    // can retry.
    sel.ids = new Set(ids.filter((id) => !succeededSet.has(id)));
    if (succeededIds.length > 0) {
      toastStore.success(`Discarded ${succeededIds.length}. Press Z to undo.`);
    }
    if (failedCount > 0) {
      toastStore.error(
        `${failedCount} discard(s) failed — still selected${lastError ? `: ${lastError}` : '.'}`,
      );
    }
  }

  async function undoLast(): Promise<void> {
    const crops = await undoStore.undoLast();
    if (crops.length === 0) return;
    // Render whatever the backend restored, one crop at a time. Each
    // crop may have left this cluster's grid (sidebar-drop labels
    // remove it), so re-insert it rather than assume it's still
    // present.
    for (const crop of crops) {
      exclusionGuard.release([crop.id]);
      if (cropPager.items.some((c) => c.id === crop.id)) {
        cropPager.items = cropPager.items.map((c) => (c.id === crop.id ? crop : c));
      } else {
        cropPager.items = [crop, ...cropPager.items];
        cropPager.total += 1;
      }
    }
  }

  async function ignoreSelected(reason: ExcludeReason = 'ignore'): Promise<void> {
    const ids = [...sel.ids];
    if (ids.length === 0) {
      toastStore.info('Select crops first to ignore.');
      return;
    }
    try {
      const res = await excludeCrops(ids, reason);
      cropPager.items = cropPager.items.filter((c) => !ids.includes(c.id));
      exclusionGuard.claim(ids);
      sel.ids = new Set();
      lastExcludedIds = ids;
      const tag = reason === 'ignore' ? '' : ` (${reason})`;
      toastStore.success(`Ignored ${res.excluded}${tag}. Press U to undo.`);
    } catch (e) {
      toastStore.error(`Ignore failed: ${(e as Error).message}`);
    }
  }

  async function undoIgnore(): Promise<void> {
    if (lastExcludedIds.length === 0) {
      toastStore.info('Nothing to un-ignore.');
      return;
    }
    const ids = lastExcludedIds;
    try {
      const res = await unexcludeCrops(ids);
      exclusionGuard.release(ids);
      lastExcludedIds = [];
      toastStore.success(`Restored ${res.unexcluded}. Re-cluster to re-sort them.`);
    } catch (e) {
      toastStore.error(`Un-ignore failed: ${(e as Error).message}`);
    }
  }

  /**
   * Issue a move from the current cluster to `targetClusterId`. On
   * success the moved crops disappear from the local grid; on failure
   * the grid is fully reloaded so we can't strand a stale optimistic
   * state.
   */
  async function moveCropIds(ids: string[], targetClusterId: number): Promise<void> {
    if (!Number.isFinite(targetClusterId) || targetClusterId === getClusterId()) {
      toastStore.warn('Pick a different cluster id.');
      return;
    }
    if (ids.length === 0) return;
    // Snapshot for revert: full crops list before mutation.
    const snap = cropPager.items;
    cropPager.items = cropPager.items.filter((c) => !ids.includes(c.id));
    sel.ids = new Set();
    rememberTarget(targetClusterId);
    // Claim these ids immediately — see excludedCropIds above. Without
    // this, a GET for this cluster that was already in flight (or gets
    // triggered by the SSE live-refresh effect) can resolve after this
    // optimistic removal with data snapshotted before this move
    // landed, silently un-removing the crop and leaving it stuck in
    // the grid until a hard reload.
    exclusionGuard.claim(ids);
    try {
      const res = await moveCropsToCluster(ids, targetClusterId);
      const moved = res.updated ?? ids.length;
      const conflicts = res.conflicts?.length ?? 0;
      if (conflicts > 0) {
        // These specific ids never actually left the source cluster —
        // let them back in once the reload below re-syncs.
        exclusionGuard.release(res.conflicts.map((c) => c.crop_id));
        toastStore.warn(
          `Moved ${moved} of ${ids.length} crop${ids.length === 1 ? '' : 's'} (${conflicts} blocked by worker). Reloading.`,
        );
        void loadFirst();
      } else {
        toastStore.success(
          `Moved ${moved} crop${moved === 1 ? '' : 's'} → cluster #${targetClusterId}.`,
        );
      }
    } catch (e) {
      exclusionGuard.release(ids);
      cropPager.items = snap;
      toastStore.error(`Move failed: ${(e as Error).message}`);
    }
  }

  return {
    assignClassToSelected,
    handleClassDrop,
    acceptVlmForCrop,
    rejectVlmForCrop,
    acceptAllVlmOnPage,
    discardSelected,
    undoLast,
    ignoreSelected,
    undoIgnore,
    moveCropIds,
  };
}

export type ClusterActionController = ReturnType<typeof createClusterActionController>;
