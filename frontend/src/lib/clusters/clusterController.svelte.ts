/**
 * `/clusters/[id]` action controller — assign / drop-on-class / VLM accept-
 * reject / discard / undo / ignore / move, extracted out of
 * `clusters/[id]/+page.svelte` (test audit P1-4,
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

import { moveCropsToCluster } from '$lib/api_clusters';
import { runLabelOp } from '$lib/labelOps/runLabelOp';
import {
  bulkLabel,
  discardCrop,
  discardCropsBatch,
  excludeCrops,
  putCropLabel,
  unexcludeCrops,
  vlmDismissCrop,
  type ExcludeReason,
} from '$lib/api_cropWrites';
import { ApiError, apiErrorText } from '$lib/api_errors';
import type { Pager } from '$lib/pager.svelte';
import type { Selection } from '$lib/selection.svelte';
import type { Crop } from '$lib/types';
import type { RegistryClass } from '$lib/types_classes';
import { classesStore } from '$stores/classes.svelte';
import { keymapStore } from '$stores/keymap.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { SvelteSet } from 'svelte/reactivity';

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
  const ids = new SvelteSet<string>();
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
  getClusterUid: () => string;
  /** LRU-remember a move target for the move-picker's recents row. */
  rememberTarget: (uid: string) => void;
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
    getClusterUid,
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
        const out = await runLabelOp(await bulkLabel(ids, classId));
        undoStore.recordLabelOp(out.opId, ids);
        if (out.conflicts > 0) {
          toastStore.warn(
            `Labeled ${out.updated} of ${ids.length} (${out.conflicts} blocked by worker).`,
          );
        }
      }
      toastStore.success(`Labeled ${ids.length} crop${ids.length === 1 ? '' : 's'}.`);
      sel.ids = new SvelteSet();
      // Bug 4 (live smoke: header stayed "0 validated" after 34 crops
      // were labeled through this page, only updating on reload): the
      // header's validated/labeled/cluster-total chip reads
      // `classesStore` (per-class `validated_count`/`count`), which the
      // server — not this optimistic write — owns. Re-fetch it instead
      // of trying to derive the new count client-side.
      void classesStore.refresh();
      // The served first page carries the cluster's real members and count.
      void loadFirst();
    } catch (e) {
      toastStore.error(`Label failed: ${apiErrorText(e)}`);
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
    // A label does not move an item between clusters (the partition is
    // geometry), so the dropped crops stay in the grid and show their new class.
    // Tear down any drag-local grid override *now*, so the grid repaints
    // from the pager instead of from whatever snapshot the in-flight drag left behind.
    resetGrid();
    sel.ids = new SvelteSet();
    const priors = cropPager.items.filter((c) => ids.includes(c.id));
    for (const id of ids) applyLocalLabel(id, cls.id, cls.name);
    try {
      const out = await runLabelOp(await bulkLabel(ids, cls.id));
      undoStore.recordLabelOp(out.opId, ids);
      if (out.conflicts > 0) {
        // A concurrent worker (typically the VLM worker) beat us on
        // some crops: reload so the grid shows what the server holds.
        toastStore.warn(
          `Labeled ${out.updated} of ${ids.length} → ${cls.name} (${out.conflicts} blocked by worker). Reloading.`,
        );
        void loadFirst();
      } else {
        toastStore.success(`Labeled ${out.updated} → ${cls.name}.`);
      }
      // See assignClassToSelected's comment — refresh the server-owned
      // per-class validated/labeled counts the header chip reads.
      void classesStore.refresh();
    } catch (e) {
      for (const prior of priors) revertLocalLabel(prior);
      toastStore.error(`Label failed: ${apiErrorText(e)}`);
    }
  }

  async function acceptVlmForCrop(crop: Crop): Promise<void> {
    if (crop.vlm_suggested_class_id == null) return;
    const targetClassId = crop.vlm_suggested_class_id;
    const targetClassName = crop.vlm_suggested_class_name ?? null;
    applyLocalLabel(crop.id, targetClassId, targetClassName);
    try {
      await putCropLabel(crop.id, targetClassId);
      undoStore.recordWrites([crop.id]);
      void classesStore.refresh();
    } catch (e) {
      toastStore.error(`Accept VLM suggestion failed: ${apiErrorText(e)}`);
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
      // M6/V1: Z reverses a Reject-VLM the same way it reverses a label
      // write or move — see undo.svelte.ts's header comment for why this
      // shares the one undoStore stack (kind: 'vlm_dismiss') rather than
      // its own.
      undoStore.recordVlmDismiss(crop.id);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        toastStore.info('No VLM suggestion to reject.');
        return;
      }
      toastStore.error(`Reject VLM suggestion failed: ${apiErrorText(e)}`);
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
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local lookup, built and consumed synchronously within this function, never stored in reactive state
    const groups = new Map<number, string[]>();
    // Prior crop per id so a failing group can be rolled back precisely —
    // a single try/catch around the whole loop left the failed group and
    // every later group locally green but never sent.
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local lookup, built and consumed synchronously within this function, never stored in reactive state
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
        const out = await runLabelOp(await bulkLabel(ids, k));
        undoStore.recordLabelOp(out.opId, ids);
        ok += ids.length;
      } catch (e) {
        lastError = apiErrorText(e);
        failedIds.push(...ids);
      }
    }
    if (failedIds.length === 0) {
      toastStore.success(`Accepted ${targets.length} suggestions.`);
      void classesStore.refresh();
      return;
    }
    for (const id of failedIds) {
      const prior = priors.get(id);
      if (prior) revertLocalLabel(prior);
    }
    toastStore.error(
      `Accepted ${ok}, failed ${failedIds.length} (reverted)${lastError ? `: ${lastError}` : '.'}`,
    );
    if (ok > 0) void classesStore.refresh();
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
        if (res.failed.length > 0) {
          lastError = `${res.failed.length} could not be written`;
        }
      }
    } catch (e) {
      lastError = apiErrorText(e);
      failedCount = ids.length;
    }
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local lookup, built and consumed synchronously within this function, never stored in reactive state
    const succeededSet = new Set(succeededIds);
    cropPager.items = cropPager.items.filter((c) => !succeededSet.has(c.id));
    exclusionGuard.claim(succeededIds);
    undoStore.recordWrites(succeededIds);
    // Keep whatever didn't succeed visible and selected so the operator
    // can retry.
    sel.ids = new SvelteSet(ids.filter((id) => !succeededSet.has(id)));
    if (succeededIds.length > 0) {
      toastStore.success(
        `Discarded ${succeededIds.length}. Press ${keymapStore.glyph('cluster.undo')} to undo.`,
      );
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
    // Undo reverses a label write like the ones above — refresh the
    // header's server-owned counts here too.
    void classesStore.refresh();
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
      sel.ids = new SvelteSet();
      lastExcludedIds = ids;
      const tag = reason === 'ignore' ? '' : ` (${reason})`;
      toastStore.success(
        `Ignored ${res.excluded}${tag}. Press ${keymapStore.glyph('cluster.unignore')} to undo.`,
      );
    } catch (e) {
      toastStore.error(`Ignore failed: ${apiErrorText(e)}`);
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
      toastStore.success(`Restored ${res.unexcluded}.`);
    } catch (e) {
      toastStore.error(`Un-ignore failed: ${apiErrorText(e)}`);
    }
  }

  /**
   * Issue a move from the current cluster to `targetClusterUid`. On
   * success the moved crops disappear from the local grid; on failure
   * the grid is fully reloaded so we can't strand a stale optimistic
   * state.
   */
  async function moveCropIds(ids: string[], targetClusterUid: string): Promise<void> {
    if (!targetClusterUid || targetClusterUid === getClusterUid()) {
      toastStore.warn('Pick a different cluster.');
      return;
    }
    if (ids.length === 0) return;
    // Snapshot for revert: full crops list before mutation.
    const snap = cropPager.items;
    cropPager.items = cropPager.items.filter((c) => !ids.includes(c.id));
    sel.ids = new SvelteSet();
    rememberTarget(targetClusterUid);
    // Claim these ids immediately — see excludedCropIds above. Without
    // this, a GET for this cluster that was already in flight (or gets
    // triggered by the SSE live-refresh effect) can resolve after this
    // optimistic removal with data snapshotted before this move
    // landed, silently un-removing the crop and leaving it stuck in
    // the grid until a hard reload.
    exclusionGuard.claim(ids);
    try {
      // The move is a recorded operation run by a queued job: follow it, then
      // undo it by operation id (Z).
      const out = await runLabelOp(await moveCropsToCluster(ids, targetClusterUid));
      undoStore.recordLabelOp(out.opId, ids);
      if (out.conflicts > 0 || !out.complete) {
        // Items a worker held (or a job that stopped) stay in this cluster:
        // let them back in once the reload below re-syncs.
        exclusionGuard.release(ids);
        toastStore.warn(
          `Moved ${out.updated} of ${ids.length} crop${ids.length === 1 ? '' : 's'} (${out.conflicts} blocked by worker). Reloading.`,
        );
        void loadFirst();
      } else {
        toastStore.success(
          `Moved ${out.updated} crop${out.updated === 1 ? '' : 's'} → cluster ${targetClusterUid.slice(0, 8)}.`,
        );
      }
    } catch (e) {
      exclusionGuard.release(ids);
      cropPager.items = snap;
      toastStore.error(`Move failed: ${apiErrorText(e)}`);
    }
  }

  /** Adopt the served post-write items of a Reprocess (every item of the
   *  reprocessed image): each replaces the grid's copy with the same
   *  `crop_id`; an id the grid does not hold is ignored (it is not in this
   *  cluster, or not loaded yet). */
  function adoptItems(items: Crop[]): void {
    if (items.length === 0) return;
    const byId: Record<string, Crop> = Object.fromEntries(items.map((i) => [i.id, i]));
    cropPager.items = cropPager.items.map((c) => byId[c.id] ?? c);
    resetGrid();
  }

  return {
    adoptItems,
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
