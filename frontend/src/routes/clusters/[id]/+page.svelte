<script lang="ts">
  import { page } from '$app/state';
  import { dndzone, SOURCES, TRIGGERS } from 'svelte-dnd-action';
  import { goto } from '$app/navigation';
  import {
    bulkLabel,
    deleteCropLabel,
    excludeCrops,
    flagNeedsNewClass,
    getCluster,
    moveCropsToCluster,
    putCropLabel,
    refineCluster,
    runGemmaOnCluster,
    unexcludeCrops,
    type ExcludeReason,
  } from '$lib/api';
  import CropCard from '$components/CropCard.svelte';
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import CutLine from '$components/CutLine.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import type { OpClass, OpCluster, OpCrop, PaginatedResponse, UndoEntry } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import { subscribeKbEvents, type OpEventSubscription } from '$lib/sse';
  import { onMount } from 'svelte';

  const clusterIdParam = $derived(page.params.id);
  const clusterId = $derived(Number(clusterIdParam));
  // In the legacy ensemble cluster_id == class_id, so the class entry
  // for this page is whichever class shares the cluster's numeric id.
  // Drives the validated / labeled / cluster-total banner in the header.
  const clsForCluster = $derived(
    classesStore.classes.find((c) => c.id === clusterId) ?? null,
  );

  let cluster = $state<OpCluster | null>(null);
  let crops = $state<OpCrop[]>([]);
  let total = $state<number>(0);
  let loadedPages = $state<number>(0);
  const pageSize = 60;
  let loading = $state<boolean>(false);
  let loadingMore = $state<boolean>(false);
  let error = $state<string | null>(null);
  const hasMore = $derived(crops.length < total);

  // Crop opened in the read-only details modal (info button on each card).
  let detailCrop = $state<OpCrop | null>(null);

  // Selection set (crop_id)
  let selected = $state<Set<string>>(new Set());

  // Sub-cluster tab (null = all). Backend stores cluster_subid as a
  // keyword string ("47a", "47b", "47aa", ...).
  let subTab = $state<string | null>(null);

  // class_source filter (null = all). Drives the chip-group in the
  // header and propagates to /curation/crops?class_source=... so the grid
  // shows only crops from one source bucket. Cluster card stats in
  // the header are NOT recomputed by this filter — the operator sees
  // the filter against the whole-cluster totals on purpose.
  let classSourceFilter = $state<string | null>(null);
  const CLASS_SOURCE_OPTIONS: { value: string | null; label: string; hint: string }[] = [
    { value: null, label: 'All', hint: 'Every source' },
    { value: 'v6_model', label: 'v6', hint: 'v6 model, ≥0.75 confidence' },
    { value: 'gemma', label: 'Gemma', hint: 'Gemma matched registry class' },
    { value: 'human', label: 'Human', hint: 'Human-validated' },
    { value: 'v6_low_conf', label: 'v6 low', hint: 'v6 below 0.75; demoted' },
    { value: 'gemma_unmatched', label: 'Gemma ?', hint: 'Gemma class not in registry' },
    { value: 'coco_yolo11_proposal', label: 'COCO', hint: 'Raw yolo proposal' },
    { value: 'gemma_new_class_pending', label: 'New cls', hint: 'Gemma proposed new class' },
  ];

  // Class dropdown
  let confirmClassId = $state<number | null>(null);

  // ---- Move/DnD state ----------------------------------------------------
  // Recent target cluster ids the labeler has typed in this session (LRU 8).
  let recentTargets = $state<number[]>([]);
  // The dnd-action items array shown in the grid; mirrors filteredCrops but
  // is what we mutate during a drag so optimistic UI feels native.
  let gridItems = $state<OpCrop[]>([]);
  // Tracks the in-flight drag's payload (one or many crops). Set on dragStart.
  let dragIds = $state<string[]>([]);
  // Inline cluster-picker (opened by M-key) state.
  let movePickerOpen = $state<boolean>(false);
  let movePickerValue = $state<string>('');
  let movePickerInput = $state<HTMLInputElement | null>(null);

  async function loadFirst(): Promise<void> {
    if (!Number.isFinite(clusterId)) return;
    loading = true;
    error = null;
    crops = [];
    loadedPages = 0;
    total = 0;
    try {
      const res = await getCluster(clusterId, 1, pageSize, undefined, {
        classSource: classSourceFilter,
      });
      cluster = res.cluster;
      const list = res.crops as PaginatedResponse<OpCrop>;
      crops = list.items;
      total = list.total ?? list.items.length;
      loadedPages = 1;
    } catch (e) {
      error = (e as Error).message;
      cluster = null;
      crops = [];
      total = 0;
    } finally {
      loading = false;
    }
  }

  async function loadMore(): Promise<void> {
    if (!Number.isFinite(clusterId) || loadingMore || !hasMore) return;
    loadingMore = true;
    try {
      const next = loadedPages + 1;
      const res = await getCluster(clusterId, next, pageSize, undefined, {
        classSource: classSourceFilter,
      });
      const list = res.crops as PaginatedResponse<OpCrop>;
      // Dedup by id in case server returns overlapping pages after a relabel.
      const seen = new Set(crops.map((c) => c.id));
      const fresh = list.items.filter((c) => !seen.has(c.id));
      crops = [...crops, ...fresh];
      total = list.total ?? total;
      loadedPages = next;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loadingMore = false;
    }
  }

  $effect(() => {
    keyboardStore.setScope('cluster');
    // Track both inputs so a filter change triggers a fresh load.
    void clusterId;
    void classSourceFilter;
    void loadFirst();
  });

  // Wire the layout-level ClassSidebar's drop targets to bulk-label the
  // currently-selected crops. Registered on mount, unregistered on
  // teardown so other pages don't accidentally receive cluster-page
  // drop dispatches.
  $effect(() => {
    const off = dropOnClassStore.register(async (cls: OpClass, droppedIds: string[]) => {
      // onGridConsider captures the full drag set at drag-start time
      // (Finder pattern — grab any selected card to drag all selected;
      // grab an unselected card to drag just that one). The
      // ClassSidebar's own consider/finalize events only see the single
      // shadow item, so dragIds is the authoritative source. droppedIds
      // is kept as a defensive fallback.
      const ids = dragIds.length > 0 ? [...dragIds] : droppedIds;
      if (ids.length === 0) {
        toastStore.warn('Drag a crop card onto a class to label it.');
        return;
      }
      for (const id of ids) {
        const c = crops.find((x) => x.id === id);
        if (c) {
          undoStore.push({
            crop_id: c.id,
            prior_class_id: c.class_id,
            prior_label_source: c.label_source,
            prior_validated: c.label_validated,
            at: Date.now(),
          });
        }
      }
      // Optimistic: remove the dropped crops from the visible grid
      // BEFORE the await, so the labeling feels real-time. The dragged
      // selection is the source of truth — if the backend reports
      // conflicts we re-sync, if it errors we restore the snapshot.
      const snap = crops;
      const snapTotal = total;
      const droppedSet = new Set(ids);
      crops = crops.filter((c) => !droppedSet.has(c.id));
      total = Math.max(0, total - ids.length);
      selected = new Set();
      try {
        const res = await bulkLabel(ids, cls.id);
        const conflicts = res.conflicts?.length ?? 0;
        if (conflicts > 0) {
          // A concurrent worker (typically op_gemma_worker) beat us on
          // some crops. The backend kept those crops on their old class;
          // re-fetch so the grid reflects truth.
          toastStore.warn(
            `Labeled ${res.updated} of ${ids.length} → ${cls.name} (${conflicts} blocked by worker). Reloading.`,
          );
          void loadFirst();
        } else {
          toastStore.success(`Labeled ${res.updated ?? ids.length} → ${cls.name}.`);
        }
      } catch (e) {
        // Revert the optimistic mutation on hard failure.
        crops = snap;
        total = snapTotal;
        toastStore.error(`Label failed: ${(e as Error).message}`);
      }
    });
    return off;
  });

  // Sub-cluster filtering (in-memory, after load). cluster_subid is
  // backend-owned (keyword like "47a") — the frontend string-matches.
  const filteredCrops = $derived(
    subTab == null ? crops : crops.filter((c) => c.cluster_subid === subTab),
  );

  const subClusterIds = $derived.by(() => {
    const set = new Set<string>();
    for (const c of crops) {
      if (c.cluster_subid != null) set.add(c.cluster_subid);
    }
    // Lexicographic sort keeps "47a","47b","47aa"... in human-expected order.
    return [...set].sort();
  });

  // When refine has produced sub-clusters and we're viewing "all", group
  // the grid inline by cluster_subid (contiguous groups + a labeled
  // separator before each) so the operator sees what refine found at a
  // glance instead of clicking through sub-cluster tabs one at a time.
  const groupBySubcluster = $derived(subTab == null && subClusterIds.length > 0);

  // Per-subid crop counts for the separator-header labels. '__none__'
  // buckets the crops refine left ungrouped (or pre-refine crops).
  const subCounts = $derived.by(() => {
    const m = new Map<string, number>();
    for (const c of crops) {
      const k = c.cluster_subid ?? '__none__';
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
  });

  // Mirror filteredCrops into gridItems whenever the underlying list changes.
  // svelte-dnd-action mutates its `items` prop in place, so we use a separate
  // array — never feed it `filteredCrops` directly. When grouping, sort so
  // each cluster_subid is contiguous (nulls last) for inline delineation.
  $effect(() => {
    const items = [...filteredCrops];
    if (groupBySubcluster) {
      items.sort((a, b) => {
        const sa = a.cluster_subid ?? '￿';
        const sb = b.cluster_subid ?? '￿';
        return sa < sb ? -1 : sa > sb ? 1 : 0;
      });
    }
    gridItems = items;
  });

  // Cut-line index: crops with similarity > 0.75 come first (already sorted
  // by API). Suppressed while grouping by sub-cluster (subid order wins).
  const cutLineIndex = $derived.by(() => {
    let i = 0;
    for (; i < filteredCrops.length; i++) {
      const s = filteredCrops[i]?.similarity_to_centroid ?? 1;
      if (s <= 0.75) break;
    }
    return i;
  });

  // Header label for the sub-cluster group starting at grid index i, or
  // null if card i isn't the start of a new group. Drives the inline
  // full-width separators in the grid.
  function subHeaderAt(i: number): { label: string; count: number } | null {
    if (!groupBySubcluster) return null;
    const cur = gridItems[i]?.cluster_subid ?? '__none__';
    const prev = i > 0 ? (gridItems[i - 1]?.cluster_subid ?? '__none__') : null;
    if (i !== 0 && cur === prev) return null;
    return {
      label: cur === '__none__' ? 'unrefined' : `sub-cluster ${cur}`,
      count: subCounts.get(cur) ?? 0,
    };
  }

  // totalPages was used by the Next/Prev buttons — gone now that infinite scroll
  // owns the pagination. Server-side pageSize stays at 60 per request, but the
  // user just keeps scrolling.

  // ---------------- selection ----------------

  // Anchor for shift-range selection: the last item clicked without
  // shift (plain or ctrl/cmd). Range selects span [anchor .. clicked]
  // in the visible filteredCrops order.
  let anchorId = $state<string | null>(null);

  function clickSelect(id: string, e?: MouseEvent): void {
    const isToggle = !!(e && (e.ctrlKey || e.metaKey));
    const isRange = !!(e && e.shiftKey);

    if (isRange && anchorId) {
      // Shift+click: select the contiguous range between the anchor and
      // the clicked card (inclusive), unioned with the current
      // selection so shift-after-ctrl extends rather than replaces.
      const ids = filteredCrops.map((c) => c.id);
      const a = ids.indexOf(anchorId);
      const b = ids.indexOf(id);
      if (a !== -1 && b !== -1) {
        const [lo, hi] = a <= b ? [a, b] : [b, a];
        const next = new Set(selected);
        for (let i = lo; i <= hi; i++) next.add(ids[i]!);
        selected = next;
        return;
      }
      // Anchor no longer visible — fall through to single-select.
    }

    if (isToggle) {
      // Ctrl/Cmd+click: add or remove just this card; move the anchor.
      const next = new Set(selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      selected = next;
      anchorId = id;
      return;
    }

    // Plain click: select only this card and set it as the new anchor.
    selected = new Set([id]);
    anchorId = id;
  }

  function selectAllPage(): void {
    selected = new Set(filteredCrops.map((c) => c.id));
  }

  function deselectAll(): void {
    selected = new Set();
    anchorId = null;
  }

  // ---------------- mutations ----------------

  function snapshot(crop: OpCrop): UndoEntry {
    return {
      crop_id: crop.id,
      prior_class_id: crop.class_id,
      prior_label_source: crop.label_source,
      prior_validated: crop.label_validated,
      at: Date.now(),
    };
  }

  function applyLocalLabel(id: string, classId: number, className: string | null): void {
    crops = crops.map((c) =>
      c.id === id
        ? {
            ...c,
            class_id: classId,
            class_name: className,
            label_validated: true,
            label_source: 'human_confirmed',
          }
        : c,
    );
  }

  function revertLocalLabel(prev: UndoEntry, prevName: string | null): void {
    crops = crops.map((c) =>
      c.id === prev.crop_id
        ? {
            ...c,
            class_id: prev.prior_class_id,
            class_name: prevName,
            label_validated: prev.prior_validated,
            label_source: prev.prior_label_source,
          }
        : c,
    );
  }

  async function assignClassToSelected(classId: number): Promise<void> {
    const ids = [...selected];
    if (ids.length === 0) {
      toastStore.warn('Nothing selected.');
      return;
    }
    const cls = classesStore.byId(classId);
    if (!cls) {
      toastStore.error('Unknown class id ' + classId);
      return;
    }
    // No nag-confirm — undo is one keystroke (Z) and the snapshot below
    // captures the prior state, so any mistake is instantly reversible.
    // Snapshot for undo
    for (const id of ids) {
      const prior = crops.find((c) => c.id === id);
      if (prior) undoStore.push(snapshot(prior));
      applyLocalLabel(id, classId, cls.name);
    }
    try {
      if (ids.length === 1) {
        await putCropLabel(ids[0]!, classId);
      } else {
        await bulkLabel(ids, classId);
      }
      toastStore.success(`Labeled ${ids.length} crop${ids.length === 1 ? '' : 's'}.`);
      selected = new Set();
    } catch (e) {
      toastStore.error(`Label failed: ${(e as Error).message}`);
      // Revert: pop snapshots back
      for (let i = 0; i < ids.length; i++) {
        const prev = undoStore.pop();
        if (!prev) break;
        const prevCls = prev.prior_class_id != null ? classesStore.byId(prev.prior_class_id) : null;
        revertLocalLabel(prev, prevCls?.name ?? null);
      }
    }
  }

  async function acceptGemmaForCrop(crop: OpCrop): Promise<void> {
    if (crop.gemma_suggested_class_id == null) return;
    undoStore.push(snapshot(crop));
    applyLocalLabel(
      crop.id,
      crop.gemma_suggested_class_id,
      crop.gemma_suggested_class_name ?? null,
    );
    try {
      await putCropLabel(crop.id, crop.gemma_suggested_class_id);
    } catch (e) {
      toastStore.error(`Accept Gemma failed: ${(e as Error).message}`);
      const prev = undoStore.pop();
      if (prev) {
        const prevCls =
          prev.prior_class_id != null ? classesStore.byId(prev.prior_class_id) : null;
        revertLocalLabel(prev, prevCls?.name ?? null);
      }
    }
  }

  async function rejectGemmaForCrop(crop: OpCrop): Promise<void> {
    // Reject = clear the suggestion locally; the server clears on next batch.
    crops = crops.map((c) =>
      c.id === crop.id
        ? { ...c, gemma_suggested_class_id: null, gemma_suggested_class_name: null }
        : c,
    );
  }

  async function acceptAllGemmaOnPage(): Promise<void> {
    const targets = filteredCrops.filter(
      (c) => c.gemma_suggested_class_id != null && !c.label_validated,
    );
    if (targets.length === 0) {
      toastStore.info('No Gemma suggestions on this page.');
      return;
    }
    // Shift+Enter is already a deliberate two-finger gesture; the snapshots
    // below feed undoStore so Z reverts instantly. No nag-confirm.
    // Group by class id for bulk_label; fall back to per-crop PUT for the long tail.
    const groups = new Map<number, string[]>();
    for (const t of targets) {
      const k = t.gemma_suggested_class_id!;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(t.id);
      undoStore.push(snapshot(t));
      applyLocalLabel(t.id, k, t.gemma_suggested_class_name ?? null);
    }
    try {
      for (const [k, ids] of groups) {
        await bulkLabel(ids, k);
      }
      toastStore.success(`Accepted ${targets.length} suggestions.`);
    } catch (e) {
      toastStore.error(`Bulk accept failed: ${(e as Error).message}`);
    }
  }

  async function undoLast(): Promise<void> {
    const entry = undoStore.pop();
    if (!entry) {
      toastStore.info('Nothing to undo.');
      return;
    }
    const prevCls =
      entry.prior_class_id != null ? classesStore.byId(entry.prior_class_id) : null;
    revertLocalLabel(entry, prevCls?.name ?? null);
    try {
      await deleteCropLabel(entry.crop_id);
      toastStore.success('Reverted.');
    } catch (e) {
      toastStore.error(`Undo failed: ${(e as Error).message}`);
    }
  }

  async function runGemma(): Promise<void> {
    try {
      const res = await runGemmaOnCluster(clusterId);
      toastStore.success(`Gemma labeled ${res.predicted ?? 0} crops (${res.updated ?? 0} updated).`);
    } catch (e) {
      toastStore.error(`Gemma run failed: ${(e as Error).message}`);
    }
  }

  // -- Ignore / exclude --------------------------------------------------
  // Excluded crops drop out of training + clustering (reversible). The
  // backend sets class_excluded=true; we remove them from the grid and
  // keep the last batch so 'U' can undo. Self-contained — does not use
  // the label-revert undoStore (Z), which only handles class labels.
  let lastExcludedIds = $state<string[]>([]);
  let ignoreMenuOpen = $state<boolean>(false);
  const EXCLUDE_REASONS: { value: ExcludeReason; label: string }[] = [
    { value: 'ignore', label: 'Ignore (generic)' },
    { value: 'blurry', label: 'Blurry' },
    { value: 'unidentifiable', label: 'Unidentifiable' },
    { value: 'not_a_vehicle', label: 'Not a vehicle' },
    { value: 'partial_crop', label: 'Partial crop' },
  ];

  async function ignoreSelected(reason: ExcludeReason = 'ignore'): Promise<void> {
    const ids = [...selected];
    if (ids.length === 0) {
      toastStore.info('Select crops first to ignore.');
      return;
    }
    ignoreMenuOpen = false;
    try {
      const res = await excludeCrops(ids, reason);
      crops = crops.filter((c) => !ids.includes(c.id));
      selected = new Set();
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
      lastExcludedIds = [];
      toastStore.success(`Restored ${res.unexcluded}. Re-cluster to re-sort them.`);
    } catch (e) {
      toastStore.error(`Un-ignore failed: ${(e as Error).message}`);
    }
  }

  async function refine(): Promise<void> {
    try {
      const res = await refineCluster(clusterId);
      toastStore.success(`Refine produced ${res.n_subclusters ?? 0} sub-clusters.`);
      void loadFirst();
    } catch (e) {
      toastStore.error(`Refine failed: ${(e as Error).message}`);
    }
  }

  async function confirmSelected(): Promise<void> {
    if (confirmClassId == null) {
      toastStore.warn('Pick a class first.');
      return;
    }
    await assignClassToSelected(confirmClassId);
    void advance();
  }

  async function advance(): Promise<void> {
    // Advance: jump to next unvalidated crop. If we're at the end of what's
    // loaded but more pages exist, fetch them; otherwise tell the user.
    const next = filteredCrops.find((c) => !c.label_validated && !selected.has(c.id));
    if (next) {
      selected = new Set([next.id]);
    } else if (hasMore) {
      await loadMore();
      const nextAfterLoad = filteredCrops.find(
        (c) => !c.label_validated && !selected.has(c.id),
      );
      if (nextAfterLoad) selected = new Set([nextAfterLoad.id]);
    } else {
      toastStore.info('End of cluster.');
    }
  }

  // ---------------- move (DnD + hotkey) ----------------

  function rememberTarget(id: number): void {
    const next = [id, ...recentTargets.filter((x) => x !== id)].slice(0, 8);
    recentTargets = next;
  }

  /**
   * Issue a move from the source cluster to `targetClusterId`. On success
   * the moved crops disappear from the local grid; on failure the grid is
   * fully reloaded so we can't strand a stale optimistic state.
   */
  async function moveCropIds(ids: string[], targetClusterId: number): Promise<void> {
    if (!Number.isFinite(targetClusterId) || targetClusterId === clusterId) {
      toastStore.warn('Pick a different cluster id.');
      return;
    }
    if (ids.length === 0) return;
    // Snapshot for revert: full crops list before mutation.
    const snap = crops;
    crops = crops.filter((c) => !ids.includes(c.id));
    selected = new Set();
    rememberTarget(targetClusterId);
    try {
      const res = await moveCropsToCluster(ids, targetClusterId);
      const moved = res.updated ?? ids.length;
      const conflicts = res.conflicts?.length ?? 0;
      if (conflicts > 0) {
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
      crops = snap;
      toastStore.error(`Move failed: ${(e as Error).message}`);
    }
  }

  function openMovePicker(): void {
    if (selected.size === 0) {
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
    if (!Number.isFinite(id) || id < 0) {
      toastStore.error('Cluster id must be a non-negative integer.');
      return;
    }
    movePickerOpen = false;
    await moveCropIds([...selected], id);
  }

  function jumpToCluster(id: number): void {
    void goto(`/clusters/${id}`);
  }

  /**
   * dnd-action handlers. We treat the source grid as a draggable-only zone:
   * removing items is fine (they animate out), adding is rejected. Each
   * target zone receives the dropped items, fires the move RPC, and then
   * resets its own items array so the visual placeholder doesn't linger.
   */
  function onGridConsider(e: CustomEvent<{ items: OpCrop[]; info: { id: string; trigger: TRIGGERS; source: SOURCES } }>): void {
    // The `!dragIds.includes` guard makes this block run once per drag
    // (consider fires repeatedly). Finder pattern: grabbing any selected
    // card drags the whole selection; grabbing an unselected card
    // replaces the selection with just that one. This is the only place
    // the multi-drag set is captured — the ClassSidebar's own dnd events
    // only see the one shadow item being hovered.
    const draggedId = e.detail.info?.id;
    if (draggedId && !dragIds.includes(draggedId)) {
      if (selected.has(draggedId) && selected.size > 1) {
        dragIds = [...selected];
      } else {
        dragIds = [draggedId];
        selected = new Set([draggedId]);
      }
    }
    gridItems = e.detail.items;
  }

  function onGridFinalize(e: CustomEvent<{ items: OpCrop[]; info: { trigger: TRIGGERS } }>): void {
    // The grid is the source-of-truth zone; if a finalize lands here without
    // a corresponding target drop, revert to filteredCrops to undo any
    // shadow-item shuffling.
    gridItems = e.detail.items;
    if (e.detail.info.trigger === TRIGGERS.DROPPED_INTO_ZONE || e.detail.info.trigger === TRIGGERS.DROPPED_INTO_ANOTHER) {
      // The actual move RPC fires from the target zone's finalize handler.
      // No-op here.
    } else {
      // DROPPED_OUTSIDE_OF_ANY / DRAG_STOPPED → restore.
      gridItems = [...filteredCrops];
      dragIds = [];
    }
  }

  // Per-target consider: we render an empty `items: []` array; dnd-action
  // accepts the shadow item but we never persist it.
  function onTargetConsider(_targetId: number) {
    return (_e: CustomEvent<{ items: OpCrop[] }>): void => {
      // We deliberately do nothing — keeping the visual cue but never
      // mutating any persistent state on hover.
    };
  }

  function onTargetFinalize(targetId: number) {
    return (e: CustomEvent<{ items: OpCrop[]; info: { trigger: TRIGGERS } }>): void => {
      const dropped = e.detail.items.filter((it) => it && typeof it.id === 'string');
      const ids = dropped.length > 0 ? dropped.map((it) => it.id) : dragIds;
      dragIds = [];
      if (ids.length === 0) return;
      void moveCropIds(ids, targetId);
    };
  }

  // ---------------- SSE: live updates for this cluster's class ----
  // The legacy convention is `cluster_id == class_id` once a class
  // has been assigned (see legacy_ingest.py + design doc), so we
  // subscribe with `class_id=clusterId`. Crop.created without a class
  // is hidden from per-class pages by event_hub's filter.
  let liveNewCount = $state<number>(0);
  let scrolledPastFirst20 = $state<boolean>(false);
  let liveSub: OpEventSubscription | null = null;
  let scrollEl = $state<HTMLDivElement | null>(null);

  onMount(() => {
    if (!Number.isFinite(clusterId)) return () => {};
    liveSub = subscribeKbEvents({
      class_id: clusterId,
      onEvent: (ev) => {
        if (ev.type === 'crop.classified' || ev.type === 'crop.created') {
          liveNewCount += 1;
        }
      },
    });
    return () => {
      liveSub?.close();
      liveSub = null;
    };
  });

  // Scroll-aware: only show the pill once the user has scrolled past
  // the first ~20 cards (otherwise just refresh in-place silently).
  function onScroll(): void {
    if (!scrollEl) return;
    scrolledPastFirst20 = scrollEl.scrollTop > 480; // ~3 rows at 8-col grid
  }

  $effect(() => {
    // Auto-refresh while user is at the top — they're not actively
    // labeling far down the list, so prepending new cards is safe.
    if (liveNewCount > 0 && !scrolledPastFirst20 && !loading && !loadingMore) {
      const n = liveNewCount;
      liveNewCount = 0;
      void loadFirst().then(() => {
        // Surface a small toast so the user knows something refreshed.
        toastStore.info(`${n} new crop${n === 1 ? '' : 's'} loaded.`);
      });
    }
  });

  function refreshFromLive(): void {
    liveNewCount = 0;
    if (scrollEl) scrollEl.scrollTop = 0;
    void loadFirst();
  }

  // ---------------- shortcuts ----------------

  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offs.push(keyboardStore.register(combo, () => void fn(), 'cluster', desc));

    // Per-class letter hotkeys (configured on /classes) are routed
    // through dropOnClassStore by the layout-level keydown listener;
    // see src/routes/+layout.svelte. The legacy 1..0 top-N scheme has
    // been removed — one binding scheme means no "what does this key
    // do here?" friction.

    reg('enter', confirmSelected, 'Confirm selected & advance');
    reg(
      'shift+enter',
      acceptAllGemmaOnPage,
      'Confirm all Gemma suggestions on page',
    );
    reg(
      'g',
      async () => {
        const ids = [...selected];
        for (const id of ids) {
          const c = crops.find((cc) => cc.id === id);
          if (c) await acceptGemmaForCrop(c);
        }
      },
      'Accept Gemma for selected',
    );
    reg(
      'n',
      async () => {
        toastStore.info('Skipped.');
        await advance();
      },
      'Skip selected',
    );
    reg(
      'N',
      async () => {
        const ids = [...selected];
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
          selected = new Set();
        } catch (e) {
          toastStore.error(`Flag failed: ${(e as Error).message}`);
        }
      },
      'Flag selected as needing new class (curator review)',
    );
    reg(
      'd',
      async () => {
        const ids = [...selected];
        if (ids.length === 0) return;
        // Snapshot before discard so undo (Z) brings them back.
        for (const id of ids) {
          const c = crops.find((cc) => cc.id === id);
          if (c) undoStore.push(snapshot(c));
        }
        for (const id of ids) {
          try {
            await deleteCropLabel(id);
          } catch (e) {
            toastStore.error(`Discard failed: ${(e as Error).message}`);
          }
        }
        crops = crops.filter((c) => !ids.includes(c.id));
        selected = new Set();
        toastStore.success(`Discarded ${ids.length}. Press Z to undo.`);
      },
      'Discard selected',
    );
    reg('z', undoLast, 'Undo last action');
    reg('x', () => void ignoreSelected('ignore'), 'Ignore selected (exclude from training)');
    reg('u', undoIgnore, 'Undo last ignore');
    reg('a', selectAllPage, 'Select all on page');
    // Arrow keys navigate within the loaded grid. With infinite scroll the
    // next-page concept is gone — left/right move selection by one position
    // in the visible filtered list.
    reg(
      'arrowleft',
      () => {
        const ids = filteredCrops.map((c) => c.id);
        if (ids.length === 0) return;
        const cur = ids.findIndex((id) => selected.has(id));
        const prev = cur <= 0 ? ids.length - 1 : cur - 1;
        selected = new Set([ids[prev]!]);
        anchorId = ids[prev]!;
      },
      'Previous crop',
    );
    reg(
      'arrowright',
      () => {
        const ids = filteredCrops.map((c) => c.id);
        if (ids.length === 0) return;
        const cur = ids.findIndex((id) => selected.has(id));
        const next = cur < 0 || cur >= ids.length - 1 ? 0 : cur + 1;
        selected = new Set([ids[next]!]);
        anchorId = ids[next]!;
        if (next === ids.length - 1 && hasMore) void loadMore();
      },
      'Next crop',
    );
    reg(
      'm',
      openMovePicker,
      'Move selected to cluster…',
    );
    reg(
      'escape',
      () => {
        if (movePickerOpen) {
          cancelMovePicker();
          return;
        }
        if (dragIds.length > 0) {
          // Synthesize a drag-cancel: dispatch a global Escape that
          // svelte-dnd-action listens for to abort the active pointer drag.
          window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
          dragIds = [];
          gridItems = [...filteredCrops];
          return;
        }
        selected = new Set();
      },
      'Cancel drag / picker / clear selection',
    );

    return () => offs.forEach((off) => off());
  });
</script>

<div class="flex h-full flex-col">
  <!-- Toolbar — title + summary + bulk-action buttons + class assignment.
       Drag-and-drop onto the left ClassSidebar is the alternative path. -->
  <div
    class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5"
  >
    <h1 class="text-lg font-semibold">Cluster #{clusterIdParam}</h1>
    {#if cluster}
      <span class="text-xs text-zinc-400">
        size {cluster.size.toLocaleString()} · dominant
        <strong class="text-zinc-200">{cluster.dominant_class_name ?? '—'}</strong>
      </span>
    {/if}
    {#if clsForCluster}
      <!-- Triage progress: how much of this cluster bucket has been
           touched and confirmed. Cluster total = FAISS bucket size
           (what's visible on this page); labeled = crops with class_id
           set; validated = label_validated=true. The sidebar chip
           mirrors the "in cluster" number so the operator's eyes match. -->
      <span
        class="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px] text-zinc-300"
        title="validated · labeled · cluster total"
      >
        <span class="text-emerald-300">{(clsForCluster.validated_count ?? 0).toLocaleString()}</span>
        <span class="text-zinc-500">validated</span>
        <span class="mx-1 text-zinc-600">·</span>
        <span class="text-blue-300">{(clsForCluster.count ?? 0).toLocaleString()}</span>
        <span class="text-zinc-500">labeled</span>
        <span class="mx-1 text-zinc-600">·</span>
        <span class="text-zinc-200">{(clsForCluster.cluster_size ?? 0).toLocaleString()}</span>
        <span class="text-zinc-500">in cluster</span>
      </span>
    {/if}

    <span class="grow"></span>

    <div class="flex items-center gap-2">
      <button class="btn" type="button" onclick={selectAllPage} title="A">
        Select page
      </button>
      <button class="btn" type="button" onclick={deselectAll} title="Esc">
        Deselect
      </button>
      <span class="font-mono text-xs text-zinc-500">{selected.size} selected</span>

      <select
        bind:value={confirmClassId}
        class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm"
      >
        <option value={null}>— class —</option>
        {#each classesStore.classes as cls (cls.id)}
          <option value={cls.id}>{cls.name}</option>
        {/each}
      </select>
      <button
        class="btn btn-primary"
        type="button"
        onclick={confirmSelected}
        disabled={selected.size === 0 || confirmClassId == null}
        title="Enter — confirm selected to chosen class"
      >
        Confirm Selected
      </button>

      <!-- Selection-aware action chips. Always rendered so the user knows
           the actions exist; disabled until a selection is non-empty.
           Each chip shows its hotkey so the keyboard path is discoverable. -->
      <button
        class="btn"
        type="button"
        onclick={openMovePicker}
        disabled={selected.size === 0}
        title="M — move selected to a different cluster"
      >
        Move <kbd class="ml-1 font-mono text-[10px] text-zinc-400">M</kbd>
      </button>
      <button
        class="btn"
        type="button"
        onclick={async () => {
          const ids = [...selected];
          if (ids.length === 0) return;
          try {
            const res = await flagNeedsNewClass(ids, '');
            toastStore.success(
              `${res.flagged} flagged for new-class review${res.errors ? ` (${res.errors} errors)` : ''}`,
            );
            selected = new Set();
          } catch (e) {
            toastStore.error(`Flag failed: ${(e as Error).message}`);
          }
        }}
        disabled={selected.size === 0}
        title="Shift+N — flag selected as needing a new class"
      >
        Flag <kbd class="ml-1 font-mono text-[10px] text-zinc-400">⇧N</kbd>
      </button>
      <button
        class="btn"
        type="button"
        onclick={acceptAllGemmaOnPage}
        title="Shift+Enter — accept all Gemma suggestions on this page"
      >
        Accept Gemma <kbd class="ml-1 font-mono text-[10px] text-zinc-400">⇧↵</kbd>
      </button>

      <span class="mx-1 h-5 w-px bg-zinc-800"></span>

      <button class="btn" type="button" onclick={runGemma}>Run Gemma</button>
      <button class="btn" type="button" onclick={refine}>Refine (AHC)</button>

      <span class="mx-1 h-5 w-px bg-zinc-800"></span>

      <!-- Ignore: one-click default (generic), dropdown for a reason.
           Excludes from training + clustering; reversible (U). -->
      <div class="relative inline-flex">
        <button
          class="btn rounded-r-none"
          type="button"
          title="Ignore selected — exclude from training + clustering (X)"
          onclick={() => void ignoreSelected('ignore')}
        >
          Ignore <kbd class="ml-1 font-mono text-[10px] text-zinc-400">X</kbd>
        </button>
        <button
          class="btn rounded-l-none border-l border-zinc-700 px-1.5"
          type="button"
          aria-label="Choose ignore reason"
          onclick={() => (ignoreMenuOpen = !ignoreMenuOpen)}
        >
          ▾
        </button>
        {#if ignoreMenuOpen}
          <div
            class="absolute right-0 top-full z-20 mt-1 w-44 rounded border border-zinc-700 bg-zinc-900 py-1 shadow-lg"
          >
            {#each EXCLUDE_REASONS as r (r.value)}
              <button
                type="button"
                class="block w-full px-3 py-1 text-left text-xs text-zinc-200 hover:bg-zinc-800"
                onclick={() => void ignoreSelected(r.value)}
              >
                {r.label}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- class_source filter chips. Narrows the grid to one source bucket
       (v6_model / gemma / human / v6_low_conf / ...) without changing
       the cluster card stats. Active filter is reflected in the URL-free
       reactive state so re-mounting the page resets to "all". -->
  <div
    class="flex flex-wrap items-center gap-1.5 border-b border-zinc-800 px-4 py-1.5 text-xs"
  >
    <span class="text-zinc-500">source:</span>
    {#each CLASS_SOURCE_OPTIONS as opt (opt.value ?? '__all__')}
      <button
        type="button"
        title={opt.hint}
        class="rounded px-2 py-0.5 {classSourceFilter === opt.value
          ? 'bg-blue-600 text-white'
          : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
        onclick={() => (classSourceFilter = opt.value)}
      >
        {opt.label}
      </button>
    {/each}
    {#if classSourceFilter !== null}
      <span class="ml-auto text-zinc-500">
        showing {total.toLocaleString()} from {classSourceFilter}
      </span>
    {/if}
  </div>

  <!-- Sub-cluster tabs -->
  {#if subClusterIds.length > 0}
    <div
      class="flex items-center gap-2 border-b border-zinc-800 px-4 py-1.5 text-xs"
    >
      <span class="text-zinc-500">sub-clusters:</span>
      <button
        type="button"
        class="rounded px-2 py-0.5 {subTab === null
          ? 'bg-blue-600 text-white'
          : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
        onclick={() => (subTab = null)}
      >
        all
      </button>
      {#each subClusterIds as sid (sid)}
        <button
          type="button"
          class="rounded px-2 py-0.5 {subTab === sid
            ? 'bg-blue-600 text-white'
            : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
          onclick={() => (subTab = sid)}
        >
          #{sid}
        </button>
      {/each}
    </div>
  {/if}

  <!-- Hotkey legend strip removed — class folders + hotkeys live in the
       left sidebar now (the legacy_sorter UX). -->

  <!-- Grid container. The class-list left sidebar is the layout-level
       ClassSidebar — it auto-receives drops via dropOnClassStore. -->
  <div class="flex min-h-0 flex-1 overflow-hidden">
    <div
      bind:this={scrollEl}
      onscroll={onScroll}
      class="relative min-w-0 flex-1 overflow-auto p-4"
    >
      {#if liveNewCount > 0 && scrolledPastFirst20}
        <button
          type="button"
          class="sticky top-2 z-10 mx-auto block animate-pulse rounded-full border border-blue-500/60 bg-blue-500/20 px-3 py-1 text-xs text-blue-100 shadow-lg backdrop-blur hover:bg-blue-500/30"
          onclick={refreshFromLive}
          title="Scroll to top and reload with the latest crops"
        >
          {liveNewCount} new crop{liveNewCount === 1 ? '' : 's'} · click to refresh
        </button>
      {/if}
      {#if loading && crops.length === 0}
        <p class="text-sm text-zinc-500">Loading...</p>
      {:else if error}
        <p class="text-sm text-red-300">API unavailable: {error}</p>
      {:else if filteredCrops.length === 0}
        <p class="text-sm text-zinc-500">No crops in this cluster yet.</p>
      {:else}
        <div
          class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8"
          use:dndzone={{
            items: gridItems,
            type: 'op-crop',
            flipDurationMs: 150,
            dropTargetStyle: { outline: '2px dashed rgb(59 130 246 / 0.6)' },
            dragDisabled: false,
          }}
          onconsider={onGridConsider}
          onfinalize={onGridFinalize}
        >
          {#each gridItems as crop, i (crop.id)}
            {@const sub = subHeaderAt(i)}
            {#if sub}
              <!-- Inline sub-cluster separator: full-width band that
                   breaks the grid into the groups refine (AHC) found. -->
              <div
                class="col-span-full mt-2 flex items-center gap-2 border-t border-zinc-700 pt-2 text-xs font-medium text-zinc-300 first:mt-0 first:border-t-0 first:pt-0"
              >
                <span class="rounded bg-zinc-800 px-2 py-0.5 text-zinc-100">{sub.label}</span>
                <span class="text-zinc-500">{sub.count} crop{sub.count === 1 ? '' : 's'}</span>
                <span class="h-px flex-1 bg-zinc-800"></span>
              </div>
            {:else if !groupBySubcluster && i === cutLineIndex && cutLineIndex > 0 && cutLineIndex < gridItems.length}
              <CutLine />
            {/if}
            <CropCard
              {crop}
              selected={selected.has(crop.id)}
              onclick={(c, e) => clickSelect(c.id, e)}
              onacceptGemma={(c) => void acceptGemmaForCrop(c)}
              onrejectGemma={(c) => void rejectGemmaForCrop(c)}
              ondetail={(c) => (detailCrop = c)}
            />
          {/each}
        </div>
        <!-- Sentinel inside the scroll container so IntersectionObserver
             roots on the right element (the overflow-auto parent). -->
        <div
          use:infiniteScroll={{
            onload: loadMore,
            disabled: loadingMore || !hasMore || loading,
          }}
          class="mt-4 h-1"
          aria-hidden="true"
        ></div>
      {/if}
    </div>

    <!-- Right move-target rail removed — class assignment moved to the
         left ClassFolderSidebar. The 'Move to cluster #N' affordance is
         still available via M-key (typing a destination cluster id). -->
  </div>

  <!-- Status bar (sentinel lives inside the scroll container, see above) -->
  <div
    class="flex items-center justify-between gap-2 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="font-mono text-xs text-zinc-500">
      {crops.length} / {total}
      {#if selected.size > 0}<span class="ml-2 text-blue-300">· {selected.size} selected</span>{/if}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if loadingMore}loading more…{:else if hasMore}scroll for more{:else}all loaded{/if}
    </span>
  </div>
</div>

{#if movePickerOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Move crops to cluster"
  >
    <div class="w-full max-w-sm rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl">
      <h3 class="mb-2 text-base font-semibold">Move {selected.size} crop{selected.size === 1 ? '' : 's'}</h3>
      <p class="mb-3 text-xs text-zinc-400">
        Move these from cluster #{clusterId} to a target cluster id. The
        operation is reversible per crop via the cluster page.
      </p>
      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Target cluster id</span>
        <input
          type="number"
          min="0"
          step="1"
          bind:this={movePickerInput}
          bind:value={movePickerValue}
          onkeydown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              void confirmMovePicker();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              cancelMovePicker();
            }
          }}
          class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
          placeholder="e.g. 42"
        />
      </label>
      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={cancelMovePicker}>Cancel</button>
        <button type="button" class="btn btn-primary" onclick={() => void confirmMovePicker()}>
          Move
        </button>
      </div>
    </div>
  </div>
{/if}

{#if detailCrop}
  <CropDetailModal crop={detailCrop} onclose={() => (detailCrop = null)} />
{/if}
