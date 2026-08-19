<script lang="ts">
  import { page } from '$app/state';
  import { dndzone, SOURCES, TRIGGERS } from 'svelte-dnd-action';
  import {
    bulkLabel,
    deleteCropLabel,
    excludeCrops,
    flagNeedsNewClass,
    getCluster,
    getCrop,
    moveCropsToCluster,
    putCropLabel,
    refineCluster,
    runGemmaOnCluster,
    unexcludeCrops,
    type ExcludeReason,
  } from '$lib/api';
  import BlurSlider from '$components/BlurSlider.svelte';
  import CropCard from '$components/CropCard.svelte';
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import CutLine from '$components/CutLine.svelte';
  import SubjectScopeToggle from '$components/SubjectScopeToggle.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { createPager } from '$lib/pager.svelte';
  import { createSelection } from '$lib/selection.svelte';
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

  // Human-readable cluster name for the header. Class clusters resolve to
  // the registry class name (cluster_id == class_id), falling back to the
  // backend's dominant_class_name. Candidate clusters (>= 10000) have no
  // name yet — label them so the operator knows it's unlabeled by design,
  // not a bug.
  const clusterName = $derived(
    clsForCluster?.name ?? cluster?.dominant_class_name ?? null,
  );

  const pageSize = 60;
  // Crop pager. cropQuery() feeds page 1 and every later page, so a filter
  // can't be applied to the first request and silently dropped on the next.
  const cropPager = createPager<OpCrop>({
    fetchPage: async (page) => {
      const res = await getCluster(clusterId, page, pageSize, undefined, cropQuery());
      cluster = res.cluster;
      return res.crops as PaginatedResponse<OpCrop>;
    },
    keyOf: (c) => c.id,
    onLoadFirstError: () => {
      cluster = null;
    },
  });

  // Crop opened in the read-only details modal (info button on each card).
  let detailCrop = $state<OpCrop | null>(null);

  // Selection set (crop_id) + shift-range anchor.
  const sel = createSelection({ plainClick: 'replace' });

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

  // Primary-subject scope: 0 = all crops, 1 = largest only, 2 = largest + 2nd.
  // Maps to the /curation/crops?max_rank= filter (the "biggest vehicle in frame" the
  // business sorts on). null = no rank filter.
  let subjectScope = $state<0 | 1 | 2>(0);
  const maxRank = $derived<number | null>(subjectScope === 0 ? null : subjectScope);

  // Outliers-first: rank members by distance from the cluster centroid (most
  // atypical first) so mislabels / junk in this cluster float to the top.
  // Computed on-the-fly + cached server-side. Off = newest-first.
  let outliersFirst = $state<boolean>(false);

  // Clarity slider. `blurSlider` is the live drag value; `minBlurRatio` only
  // commits on release (change, not input) so dragging doesn't spam the API.
  // 0 = show everything (null sent). v1.1.9 sale-quality stops: 1.1/1.3/1.4 —
  // training tolerance sits lower, so the slider spans well below them.
  const BLUR_MAX = 2;
  let blurSlider = $state<number>(0);
  let minBlurRatio = $state<number | null>(null);
  function commitBlur(): void {
    minBlurRatio = blurSlider > 0 ? blurSlider : null;
  }

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
  type GridGroup = { key: string; label: string; items: OpCrop[] };
  let gridGroups = $state<GridGroup[]>([]);
  // Tracks the in-flight drag's payload (one or many crops). Set on dragStart.
  let dragIds = $state<string[]>([]);
  // Inline cluster-picker (opened by M-key) state.
  let movePickerOpen = $state<boolean>(false);
  let movePickerValue = $state<string>('');
  let movePickerInput = $state<HTMLInputElement | null>(null);

  function cropQuery() {
    return {
      classSource: classSourceFilter,
      maxRank,
      minBlurRatio,
      order: outliersFirst ? ('outliers' as const) : null,
    };
  }

  async function loadFirst(): Promise<void> {
    if (!Number.isFinite(clusterId)) return;
    await cropPager.loadFirst();
  }

  async function loadMore(): Promise<void> {
    if (!Number.isFinite(clusterId)) return;
    await cropPager.loadMore();
  }

  $effect(() => {
    keyboardStore.setScope('cluster');
    // Track every filter input so a change triggers a fresh load.
    void clusterId;
    void classSourceFilter;
    void maxRank;
    void minBlurRatio;
    void outliersFirst;
    void loadFirst();
  });

  // Wire the layout-level ClassSidebar's drop targets to bulk-label the
  // currently-selected crops. Registered on mount, unregistered on
  // teardown so other pages don't accidentally receive cluster-page
  // drop dispatches.
  $effect(() => {
    const off = dropOnClassStore.register(async (cls: OpClass, droppedIds: string[]) => {
      // Priority order matters. onGroupConsider captures the full drag
      // set at drag-start time (Finder pattern — grab any selected card
      // to drag all selected; grab an unselected card to drag just that
      // one), and the ClassSidebar's own finalize only ever sees the
      // single shadow item — so for a multi-drag drop droppedIds has 1 id
      // while dragIds has N. dragIds must therefore win. The `selected`
      // fallback serves the keyboard path: the layout's class-letter
      // listener dispatches with an empty droppedIds and no drag context.
      // Consume dragIds here — leaving it populated would make the NEXT
      // hotkey press relabel the previously dragged crops.
      const ids =
        dragIds.length > 0
          ? [...dragIds]
          : droppedIds.length > 0
            ? droppedIds
            : [...sel.ids];
      dragIds = [];
      if (ids.length === 0) {
        toastStore.warn('Select or drag crops first, then press a class hotkey.');
        return;
      }
      for (const id of ids) {
        const c = cropPager.items.find((x) => x.id === id);
        if (c) undoStore.push(undoStore.snapshotOf(c));
      }
      // Optimistic: remove the dropped crops from the visible grid
      // BEFORE the await, so the labeling feels real-time. The dragged
      // selection is the source of truth — if the backend reports
      // conflicts we re-sync, if it errors we restore the snapshot.
      const snap = cropPager.items;
      const snapTotal = cropPager.total;
      const droppedSet = new Set(ids);
      cropPager.items = cropPager.items.filter((c) => !droppedSet.has(c.id));
      cropPager.total = Math.max(0, cropPager.total - ids.length);
      sel.ids = new Set();
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
        cropPager.items = snap;
        cropPager.total = snapTotal;
        toastStore.error(`Label failed: ${(e as Error).message}`);
      }
    });
    return off;
  });

  // Sub-cluster filtering (in-memory, after load). cluster_subid is
  // backend-owned (keyword like "47a") — the frontend string-matches.
  const filteredCrops = $derived(
    subTab == null ? cropPager.items : cropPager.items.filter((c) => c.cluster_subid === subTab),
  );

  const subClusterIds = $derived.by(() => {
    const set = new Set<string>();
    for (const c of cropPager.items) {
      if (c.cluster_subid != null) set.add(c.cluster_subid);
    }
    // Lexicographic sort keeps "47a","47b","47aa"... in human-expected order.
    return [...set].sort();
  });

  // When refine has produced sub-clusters and we're viewing "all", group
  // the grid inline by cluster_subid (contiguous groups + a labeled
  // separator before each) so the operator sees what refine found at a
  // glance instead of clicking through sub-cluster tabs one at a time.
  // Outliers-first wins over sub-cluster grouping: when ranking by centroid
  // distance we want one flat, server-ordered list (most atypical at top),
  // not a regroup by subid.
  const groupBySubcluster = $derived(
    !outliersFirst && subTab == null && subClusterIds.length > 0,
  );

  // Build gridGroups from filteredCrops. When grouping, partition into
  // contiguous sub-cluster groups (sorted by cluster_subid, unrefined
  // last) each with its own header + dndzone. When not grouping, one
  // '__all__' group holding the whole filtered list. Never feed
  // filteredCrops directly to a zone — the dnd action mutates items in
  // place, so each group gets a fresh array copy.
  function buildGroups(source: OpCrop[]): GridGroup[] {
    if (!groupBySubcluster) {
      return [{ key: '__all__', label: '', items: [...source] }];
    }
    const sorted = [...source].sort((a, b) => {
      const sa = a.cluster_subid ?? '￿';
      const sb = b.cluster_subid ?? '￿';
      return sa < sb ? -1 : sa > sb ? 1 : 0;
    });
    const groups: GridGroup[] = [];
    let lastSub: string | null = null;
    for (const c of sorted) {
      const sub = c.cluster_subid ?? '__none__';
      const last = groups[groups.length - 1];
      if (!last || lastSub !== sub) {
        // The key carries the run index, not just the subid: this is a
        // contiguity walk, so a subid that reappears non-contiguously
        // (a transient render over a not-yet-sorted list) would emit the
        // same key twice and crash the keyed {#each} with
        // each_key_duplicate. The sibling grouping on /clusters documents
        // the same crash.
        groups.push({
          key: `${sub}#${groups.length}`,
          label: sub === '__none__' ? 'unrefined' : `sub-cluster ${sub}`,
          items: [c],
        });
        lastSub = sub;
      } else {
        last.items.push(c);
      }
    }
    return groups;
  }

  // Rebuild groups whenever the underlying list (or grouping mode)
  // changes. Tracked deps: filteredCrops + groupBySubcluster. Does NOT
  // read gridGroups, so mid-drag mutations of gridGroups don't retrigger
  // it (which would clobber the in-flight shuffle).
  $effect(() => {
    gridGroups = buildGroups(filteredCrops);
  });

  // Cut-line index: crops with similarity > 0.75 come first (already
  // sorted by API). Only meaningful in the single '__all__' group;
  // suppressed while grouping by sub-cluster (subid order wins).
  const cutLineIndex = $derived.by(() => {
    let i = 0;
    for (; i < filteredCrops.length; i++) {
      const s = filteredCrops[i]?.similarity_to_centroid ?? 1;
      if (s <= 0.75) break;
    }
    return i;
  });

  // totalPages was used by the Next/Prev buttons — gone now that infinite scroll
  // owns the pagination. Server-side pageSize stays at 60 per request, but the
  // user just keeps scrolling.

  // ---------------- selection ----------------

  // Range selects span [anchor .. clicked] in the visible order, so the
  // ordered id list is handed to the selection helper per click.
  function clickSelect(id: string, e?: MouseEvent): void {
    sel.click(
      id,
      e,
      filteredCrops.map((c) => c.id),
    );
  }

  function selectAllPage(): void {
    sel.selectAll(filteredCrops.map((c) => c.id));
  }

  function deselectAll(): void {
    sel.clear();
  }

  // ---------------- mutations ----------------

  function applyLocalLabel(id: string, classId: number, className: string | null): void {
    cropPager.items = cropPager.items.map((c) =>
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
    cropPager.items = cropPager.items.map((c) =>
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
    // No nag-confirm — undo is one keystroke (Z) and the snapshot below
    // captures the prior state, so any mistake is instantly reversible.
    // Keep our own references to the pushed entries: the revert path must
    // drop exactly these, not pop N off a stack the operator may have
    // changed (by pressing Z) while the request was in flight.
    const pushed: UndoEntry[] = [];
    for (const id of ids) {
      const prior = cropPager.items.find((c) => c.id === id);
      if (prior) {
        const entry = undoStore.snapshotOf(prior);
        undoStore.push(entry);
        pushed.push(entry);
      }
      applyLocalLabel(id, classId, cls.name);
    }
    try {
      if (ids.length === 1) {
        await putCropLabel(ids[0]!, classId);
      } else {
        await bulkLabel(ids, classId);
      }
      toastStore.success(`Labeled ${ids.length} crop${ids.length === 1 ? '' : 's'}.`);
      sel.ids = new Set();
    } catch (e) {
      toastStore.error(`Label failed: ${(e as Error).message}`);
      for (const prev of pushed) {
        const prevCls = prev.prior_class_id != null ? classesStore.byId(prev.prior_class_id) : null;
        revertLocalLabel(prev, prevCls?.name ?? null);
      }
      undoStore.remove(pushed);
    }
  }

  async function acceptGemmaForCrop(crop: OpCrop): Promise<void> {
    if (crop.gemma_suggested_class_id == null) return;
    const entry = undoStore.snapshotOf(crop);
    undoStore.push(entry);
    applyLocalLabel(
      crop.id,
      crop.gemma_suggested_class_id,
      crop.gemma_suggested_class_name ?? null,
    );
    try {
      await putCropLabel(crop.id, crop.gemma_suggested_class_id);
    } catch (e) {
      toastStore.error(`Accept Gemma failed: ${(e as Error).message}`);
      const prevCls =
        entry.prior_class_id != null ? classesStore.byId(entry.prior_class_id) : null;
      revertLocalLabel(entry, prevCls?.name ?? null);
      undoStore.remove([entry]);
    }
  }

  async function rejectGemmaForCrop(crop: OpCrop): Promise<void> {
    // Reject = clear the suggestion locally; the server clears on next batch.
    cropPager.items = cropPager.items.map((c) =>
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
    // Snapshot per crop id so a failing group can be reverted precisely —
    // a single try/catch around the whole loop left the failed group and
    // every later group locally green but never sent.
    const snaps = new Map<string, UndoEntry>();
    for (const t of targets) {
      const k = t.gemma_suggested_class_id!;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(t.id);
      const entry = undoStore.snapshotOf(t);
      snaps.set(t.id, entry);
      undoStore.push(entry);
      applyLocalLabel(t.id, k, t.gemma_suggested_class_name ?? null);
    }
    let ok = 0;
    let lastError: string | null = null;
    const failedIds: string[] = [];
    for (const [k, ids] of groups) {
      try {
        await bulkLabel(ids, k);
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
    const stale: UndoEntry[] = [];
    for (const id of failedIds) {
      const s = snaps.get(id);
      if (!s) continue;
      const prevCls = s.prior_class_id != null ? classesStore.byId(s.prior_class_id) : null;
      revertLocalLabel(s, prevCls?.name ?? null);
      stale.push(s);
    }
    undoStore.remove(stale);
    toastStore.error(
      `Accepted ${ok}, failed ${failedIds.length} (reverted)${lastError ? `: ${lastError}` : '.'}`,
    );
  }

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
      sel.ids = new Set();
    } catch (e) {
      toastStore.error(`Flag failed: ${(e as Error).message}`);
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
      if (entry.prior_validated && entry.prior_class_id != null) {
        // The crop carried a human-validated label before the action we're
        // undoing. DELETE would reset it to the model suggestion instead,
        // silently diverging from what the grid shows. putCropLabel sets
        // validated=true server-side, which matches prior_validated; the
        // finer prior_label_source granularity is lost, which is fine.
        await putCropLabel(entry.crop_id, entry.prior_class_id);
      } else {
        await deleteCropLabel(entry.crop_id);
      }
      toastStore.success('Reverted.');
      // Crops labeled via the sidebar-drop path were removed from the
      // grid, so revertLocalLabel above was a no-op for them. Pull the
      // crop back so the operator can see what returned.
      if (!cropPager.items.some((c) => c.id === entry.crop_id)) {
        try {
          const restored = await getCrop(entry.crop_id);
          cropPager.items = [restored, ...cropPager.items];
          cropPager.total += 1;
        } catch (e) {
          toastStore.info(`Reverted, but could not re-fetch the crop: ${(e as Error).message}`);
        }
      }
    } catch (e) {
      toastStore.error(`Undo failed: ${(e as Error).message}`);
      // Put the entry back so Z can be retried; the local revert stands
      // (re-applying the label optimistically would be the bigger lie).
      undoStore.push(entry);
    }
  }

  // In-flight guards so the operator gets feedback and can't double-fire
  // these long-running cluster ops.
  let refining = $state<boolean>(false);
  let gemmaRunning = $state<boolean>(false);

  async function runGemma(): Promise<void> {
    if (gemmaRunning) return;
    gemmaRunning = true;
    try {
      const res = await runGemmaOnCluster(clusterId);
      toastStore.success(`Gemma labeled ${res.predicted ?? 0} crops (${res.updated ?? 0} updated).`);
    } catch (e) {
      toastStore.error(`Gemma run failed: ${(e as Error).message}`);
    } finally {
      gemmaRunning = false;
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
    const ids = [...sel.ids];
    if (ids.length === 0) {
      toastStore.info('Select crops first to ignore.');
      return;
    }
    ignoreMenuOpen = false;
    try {
      const res = await excludeCrops(ids, reason);
      cropPager.items = cropPager.items.filter((c) => !ids.includes(c.id));
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
      lastExcludedIds = [];
      toastStore.success(`Restored ${res.unexcluded}. Re-cluster to re-sort them.`);
    } catch (e) {
      toastStore.error(`Un-ignore failed: ${(e as Error).message}`);
    }
  }

  async function refine(): Promise<void> {
    if (refining) return;
    refining = true;
    try {
      const res = await refineCluster(clusterId);
      toastStore.success(`Refine produced ${res.n_subclusters ?? 0} sub-clusters.`);
      await loadFirst();
    } catch (e) {
      toastStore.error(`Refine failed: ${(e as Error).message}`);
    } finally {
      refining = false;
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
    const next = filteredCrops.find((c) => !c.label_validated && !sel.has(c.id));
    if (next) {
      sel.ids = new Set([next.id]);
    } else if (cropPager.hasMore) {
      await loadMore();
      const nextAfterLoad = filteredCrops.find(
        (c) => !c.label_validated && !sel.has(c.id),
      );
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
    const snap = cropPager.items;
    cropPager.items = cropPager.items.filter((c) => !ids.includes(c.id));
    sel.ids = new Set();
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
      cropPager.items = snap;
      toastStore.error(`Move failed: ${(e as Error).message}`);
    }
  }

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
    if (!Number.isFinite(id) || id < 0) {
      toastStore.error('Cluster id must be a non-negative integer.');
      return;
    }
    movePickerOpen = false;
    await moveCropIds([...sel.ids], id);
  }

  /**
   * dnd-action handlers. The grid is a draggable-only zone: removing items
   * is fine (they animate out), adding is rejected. The real label RPC
   * fires from the layout ClassSidebar's own finalize, routed back here
   * through dropOnClassStore.
   */
  function _setGroupItems(key: string, items: OpCrop[]): void {
    const g = gridGroups.find((x) => x.key === key);
    if (g) g.items = items;
  }

  function onGroupConsider(
    key: string,
    e: CustomEvent<{ items: OpCrop[]; info: { id: string; trigger: TRIGGERS; source: SOURCES } }>,
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
        sel.ids = new Set([draggedId]);
      }
    }
    _setGroupItems(key, e.detail.items);
  }

  function onGroupFinalize(
    key: string,
    e: CustomEvent<{ items: OpCrop[]; info: { trigger: TRIGGERS } }>,
  ): void {
    _setGroupItems(key, e.detail.items);
    // Every branch rebuilds the grid: real label-move RPCs fire from the
    // ClassSidebar's finalize, and dropping into another grid sub-group
    // has no semantic meaning (all groups share cluster_id).
    gridGroups = buildGroups(filteredCrops);
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
    if (liveNewCount > 0 && !scrolledPastFirst20 && !cropPager.loading && !cropPager.loadingMore) {
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
        const ids = [...sel.ids];
        for (const id of ids) {
          const c = cropPager.items.find((cc) => cc.id === id);
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
    // 'shift+n', not 'N': register() lowercases combos, so 'N' would
    // collapse onto the skip binding above and never fire.
    reg(
      'shift+n',
      flagSelectedForNewClass,
      'Flag selected as needing new class (curator review)',
    );
    reg(
      'd',
      async () => {
        const ids = [...sel.ids];
        if (ids.length === 0) return;
        // Snapshot per id BEFORE the delete (the crop must still be in the
        // grid to read its prior label), but only push the snapshots for
        // ids the server actually accepted — an undo entry for a crop that
        // was never unlabeled would clobber its real label on Z.
        const snaps = new Map<string, UndoEntry>();
        for (const id of ids) {
          const c = cropPager.items.find((cc) => cc.id === id);
          if (c) snaps.set(id, undoStore.snapshotOf(c));
        }
        const succeeded: string[] = [];
        const failed: string[] = [];
        let lastError: string | null = null;
        for (const id of ids) {
          try {
            await deleteCropLabel(id);
            succeeded.push(id);
          } catch (e) {
            lastError = (e as Error).message;
            failed.push(id);
          }
        }
        for (const id of succeeded) {
          const s = snaps.get(id);
          if (s) undoStore.push(s);
        }
        const succeededSet = new Set(succeeded);
        cropPager.items = cropPager.items.filter((c) => !succeededSet.has(c.id));
        // Keep the failures visible and selected so the operator can retry.
        sel.ids = new Set(failed);
        if (succeeded.length > 0) {
          toastStore.success(`Discarded ${succeeded.length}. Press Z to undo.`);
        }
        if (failed.length > 0) {
          toastStore.error(
            `${failed.length} discard(s) failed — still selected${lastError ? `: ${lastError}` : '.'}`,
          );
        }
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
        const cur = ids.findIndex((id) => sel.has(id));
        const prev = cur <= 0 ? ids.length - 1 : cur - 1;
        sel.ids = new Set([ids[prev]!]);
        sel.anchorId = ids[prev]!;
      },
      'Previous crop',
    );
    reg(
      'arrowright',
      () => {
        const ids = filteredCrops.map((c) => c.id);
        if (ids.length === 0) return;
        const cur = ids.findIndex((id) => sel.has(id));
        const next = cur < 0 || cur >= ids.length - 1 ? 0 : cur + 1;
        sel.ids = new Set([ids[next]!]);
        sel.anchorId = ids[next]!;
        if (next === ids.length - 1 && cropPager.hasMore) void loadMore();
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
          // svelte-dnd-action can't cancel an in-progress POINTER drag —
          // its only Escape handling is gated on the keyboard-drag (aria)
          // module's isDragging flag. So all Escape can do here is discard
          // the captured multi-drag set and restore the grid layout; the
          // drag itself ends when the user releases the pointer. Do NOT
          // re-dispatch a synthetic Escape on window: dispatchEvent is
          // synchronous, so it re-enters this same handler with dragIds
          // still populated and recurses until the stack blows.
          dragIds = [];
          gridGroups = buildGroups(filteredCrops);
          return;
        }
        sel.ids = new Set();
      },
      'Clear drag capture / close picker / clear selection',
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
    <h1 class="flex items-baseline gap-2 text-lg font-semibold">
      {#if clusterName}
        <span class="capitalize text-zinc-100">{clusterName}</span>
        <span class="text-sm font-normal text-zinc-500">#{clusterIdParam}</span>
      {:else if cluster?.cluster_kind === 'candidate'}
        <span class="text-zinc-100">Cluster #{clusterIdParam}</span>
        <span class="text-sm font-normal text-amber-400/80">unlabeled candidate</span>
      {:else}
        <span class="text-zinc-100">Cluster #{clusterIdParam}</span>
      {/if}
    </h1>
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
      <span class="font-mono text-xs text-zinc-500">{sel.size} selected</span>

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
        disabled={sel.size === 0 || confirmClassId == null}
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
        disabled={sel.size === 0}
        title="M — move selected to a different cluster"
      >
        Move <kbd class="ml-1 font-mono text-[10px] text-zinc-400">M</kbd>
      </button>
      <button
        class="btn"
        type="button"
        onclick={() => void flagSelectedForNewClass()}
        disabled={sel.size === 0}
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

      <button class="btn" type="button" onclick={runGemma} disabled={gemmaRunning}>
        {#if gemmaRunning}
          <span class="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-100 align-[-1px]"></span>
          Running…
        {:else}
          Run Gemma
        {/if}
      </button>
      <button
        class="btn"
        type="button"
        onclick={refine}
        disabled={refining}
        title="Sub-cluster this cluster with AHC"
      >
        {#if refining}
          <span class="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-100 align-[-1px]"></span>
          Refining…
        {:else}
          Refine (AHC)
        {/if}
      </button>

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
        showing {cropPager.total.toLocaleString()} from {classSourceFilter}
      </span>
    {/if}
  </div>

  <!-- Primary-subject controls: focus on the largest vehicle(s) in frame
       (what the business sorts on) and hide too-blurry crops. View-only —
       no data is deleted; the slider commits on release to avoid a reload
       per pixel. -->
  <div
    class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-1.5 text-xs"
  >
    <SubjectScopeToggle
      bind:value={subjectScope}
      labels={['All', 'Largest', 'Largest + 2nd']}
      label="subject:"
      labelClass="text-zinc-500"
      dense
    />

    <BlurSlider
      bind:value={blurSlider}
      oncommit={commitBlur}
      max={BLUR_MAX}
      labelClass="ml-2 text-zinc-500"
      width="w-40"
      stops
      title="Hide crops blurrier than this (blur_lap_ratio). 1.1/1.3/1.4 are the v1.1.9 sale-quality stops; training tolerance is lower."
    />

    <!-- Outliers-first: float the members least like the cluster centroid to
         the top, so wrong/atypical items are easy to cherry-pick out. -->
    <button
      type="button"
      class="rounded border px-2 py-0.5 {outliersFirst
        ? 'border-amber-500/60 bg-amber-500/20 text-amber-100'
        : 'border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
      onclick={() => (outliersFirst = !outliersFirst)}
      title="Sort by distance from the cluster centroid (most atypical first) to spot mislabels/junk"
    >
      {outliersFirst ? '◤ Outliers first' : 'Outliers first'}
    </button>

    {#if subjectScope !== 0 || minBlurRatio !== null || outliersFirst}
      <button
        type="button"
        class="ml-auto rounded bg-zinc-800 px-2 py-0.5 text-zinc-300 hover:bg-zinc-700"
        onclick={() => {
          subjectScope = 0;
          blurSlider = 0;
          minBlurRatio = null;
          outliersFirst = false;
        }}
      >
        reset
      </button>
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
      {#if cropPager.loading && cropPager.items.length === 0}
        <p class="text-sm text-zinc-500">Loading...</p>
      {:else if cropPager.error}
        <p class="text-sm text-red-300">API unavailable: {cropPager.error}</p>
      {:else if filteredCrops.length === 0}
        <p class="text-sm text-zinc-500">No crops in this cluster yet.</p>
      {:else}
        <!-- One dndzone per sub-cluster group. Headers sit BETWEEN zones
             (not inside any) so each zone's children map 1:1 to its
             items — keeping drag-and-drop intact. All zones share
             type='op-crop' so a card drags out to the ClassSidebar (or
             across groups) exactly as before. -->
        {#each gridGroups as group (group.key)}
          {#if groupBySubcluster}
            <div
              class="mt-3 flex items-center gap-2 pt-1 text-xs font-medium text-zinc-300 first:mt-0"
            >
              <span class="rounded bg-zinc-800 px-2 py-0.5 text-zinc-100">{group.label}</span>
              <span class="text-zinc-500">
                {group.items.length} crop{group.items.length === 1 ? '' : 's'}
              </span>
              <span class="h-px flex-1 bg-zinc-800"></span>
            </div>
          {/if}
          <div
            class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 {groupBySubcluster
              ? 'mt-2'
              : ''}"
            use:dndzone={{
              items: group.items,
              type: 'op-crop',
              flipDurationMs: 150,
              dropTargetStyle: { outline: '2px dashed rgb(59 130 246 / 0.6)' },
              dragDisabled: false,
              // Reject foreign items. After AHC refine the grid is split
              // into one zone per sub-cluster; without this a sibling
              // sub-cluster zone steals a drop meant for the ClassSidebar
              // (it's geometrically larger than the narrow sidebar rows),
              // so the card snaps back into the AHC grouping instead of
              // being labeled. Cross-group grid drops have no meaning
              // anyway (onGroupFinalize rebuilds them), so disabling
              // foreign drops only removes the drop-stealing — drag-out to
              // the sidebar is unaffected.
              dropFromOthersDisabled: true,
            }}
            onconsider={(e) => onGroupConsider(group.key, e)}
            onfinalize={(e) => onGroupFinalize(group.key, e)}
          >
            {#each group.items as crop, i (crop.id)}
              {#if !groupBySubcluster && i === cutLineIndex && cutLineIndex > 0 && cutLineIndex < group.items.length}
                <CutLine />
              {/if}
              <CropCard
                {crop}
                selected={sel.has(crop.id)}
                onclick={(c, e) => clickSelect(c.id, e)}
                onacceptGemma={(c) => void acceptGemmaForCrop(c)}
                onrejectGemma={(c) => void rejectGemmaForCrop(c)}
                ondetail={(c) => (detailCrop = c)}
              />
            {/each}
          </div>
        {/each}
        <!-- Sentinel inside the scroll container so IntersectionObserver
             roots on the right element (the overflow-auto parent). -->
        <div
          use:infiniteScroll={{
            onload: loadMore,
            disabled: cropPager.loadingMore || !cropPager.hasMore || cropPager.loading,
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
      {cropPager.items.length} / {cropPager.total}
      {#if sel.size > 0}<span class="ml-2 text-blue-300">· {sel.size} selected</span>{/if}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if cropPager.loadingMore}loading more…{:else if cropPager.hasMore}scroll for more{:else}all loaded{/if}
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
      <h3 class="mb-2 text-base font-semibold">Move {sel.size} crop{sel.size === 1 ? '' : 's'}</h3>
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
