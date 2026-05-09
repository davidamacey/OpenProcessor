<script lang="ts">
  import { page } from '$app/state';
  import { dndzone, SOURCES, TRIGGERS } from 'svelte-dnd-action';
  import { goto } from '$app/navigation';
  import {
    bulkLabel,
    deleteCropLabel,
    flagNeedsNewClass,
    getCluster,
    moveCropsToCluster,
    putCropLabel,
    refineCluster,
    runGemmaOnCluster,
  } from '$lib/api';
  import CropCard from '$components/CropCard.svelte';
  import CutLine from '$components/CutLine.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import type { OpClass, OpCluster, OpCrop, PaginatedResponse, UndoEntry } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';

  const clusterIdParam = $derived(page.params.id);
  const clusterId = $derived(Number(clusterIdParam));

  let cluster = $state<OpCluster | null>(null);
  let crops = $state<OpCrop[]>([]);
  let total = $state<number>(0);
  let loadedPages = $state<number>(0);
  const pageSize = 60;
  let loading = $state<boolean>(false);
  let loadingMore = $state<boolean>(false);
  let error = $state<string | null>(null);
  const hasMore = $derived(crops.length < total);

  // Selection set (crop_id)
  let selected = $state<Set<string>>(new Set());

  // Sub-cluster tab (null = all)
  let subTab = $state<number | null>(null);

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
      const res = await getCluster(clusterId, 1, pageSize);
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
      const res = await getCluster(clusterId, next, pageSize);
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
    void clusterId;
    void loadFirst();
  });

  // Wire the layout-level ClassSidebar's drop targets to bulk-label the
  // currently-selected crops. Registered on mount, unregistered on
  // teardown so other pages don't accidentally receive cluster-page
  // drop dispatches.
  $effect(() => {
    const off = dropOnClassStore.register(async (cls: OpClass) => {
      const ids = [...selected];
      if (ids.length === 0) {
        toastStore.warn('Select crops first, then drag onto a class.');
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
      try {
        const res = await bulkLabel(ids, cls.id);
        toastStore.success(
          `Labeled ${res.affected ?? ids.length} → ${cls.name}.`,
        );
        crops = crops.filter((c) => !selected.has(c.id));
        total = Math.max(0, total - ids.length);
        selected = new Set();
      } catch (e) {
        toastStore.error(`Label failed: ${(e as Error).message}`);
      }
    });
    return off;
  });

  // Sub-cluster filtering (in-memory, after load)
  const filteredCrops = $derived(
    subTab == null ? crops : crops.filter((c) => c.sub_cluster_id === subTab),
  );

  // Mirror filteredCrops into gridItems whenever the underlying list changes.
  // svelte-dnd-action mutates its `items` prop in place, so we use a separate
  // array — never feed it `filteredCrops` directly.
  $effect(() => {
    gridItems = [...filteredCrops];
  });

  // Cut-line index: crops with similarity > 0.75 come first (already sorted by API).
  const cutLineIndex = $derived.by(() => {
    let i = 0;
    for (; i < filteredCrops.length; i++) {
      const s = filteredCrops[i]?.similarity_to_centroid ?? 1;
      if (s <= 0.75) break;
    }
    return i;
  });

  // Top-N classes for hotkeys 1..9, 0
  const topClasses = $derived(classesStore.topNForCluster(clusterId, 10));

  const subClusterIds = $derived.by(() => {
    const set = new Set<number>();
    for (const c of crops) {
      if (c.sub_cluster_id != null) set.add(c.sub_cluster_id);
    }
    return [...set].sort((a, b) => a - b);
  });

  // totalPages was used by the Next/Prev buttons — gone now that infinite scroll
  // owns the pagination. Server-side pageSize stays at 60 per request, but the
  // user just keeps scrolling.

  // ---------------- selection ----------------

  function toggleSelect(id: string, e?: MouseEvent): void {
    const next = new Set(selected);
    if (e?.shiftKey && next.has(id)) {
      next.delete(id);
    } else if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    selected = next;
  }

  function selectAllPage(): void {
    selected = new Set(filteredCrops.map((c) => c.id));
  }

  function deselectAll(): void {
    selected = new Set();
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
    if (ids.length > 1) {
      const ok = window.confirm(`Confirm bulk-label of ${ids.length} crops as "${cls.name}"?`);
      if (!ok) return;
    }
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
    const ok = window.confirm(`Accept ${targets.length} Gemma suggestion${targets.length === 1 ? '' : 's'}?`);
    if (!ok) return;
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

  async function refine(): Promise<void> {
    try {
      const res = await refineCluster(clusterId);
      toastStore.success(`Refine produced ${res.subclusters ?? 0} sub-clusters.`);
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
      const moved = res.moved ?? ids.length;
      const failed = res.failed?.length ?? 0;
      if (failed > 0) {
        toastStore.warn(
          `Moved ${moved} of ${ids.length} crop${ids.length === 1 ? '' : 's'} (${failed} failed). Reloading.`,
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
    // Track payload for hotkey-based cancel/abort flows.
    const draggedId = e.detail.info?.id;
    if (draggedId && !dragIds.includes(draggedId)) {
      // If the dragged crop is part of the selection, drag the whole group.
      if (selected.has(draggedId) && selected.size > 1) {
        dragIds = [...selected];
      } else {
        dragIds = [draggedId];
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

  // ---------------- shortcuts ----------------

  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offs.push(keyboardStore.register(combo, () => void fn(), 'cluster', desc));

    // 1..9 + 0 → top classes
    for (let i = 0; i < 10; i++) {
      const key = i === 9 ? '0' : String(i + 1);
      reg(
        key,
        async () => {
          const cls = topClasses[i];
          if (!cls) {
            toastStore.warn(`No class bound to ${key}`);
            return;
          }
          await assignClassToSelected(cls.id);
        },
        `Assign top-class #${i + 1}`,
      );
    }

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
        const note = window.prompt(
          `Flag ${ids.length} crop${ids.length === 1 ? '' : 's'} as NEEDS NEW CLASS?\n` +
            `Optional note (e.g. proposed class name):`,
          '',
        );
        if (note === null) return;
        try {
          const res = await flagNeedsNewClass(ids, note);
          toastStore.success(
            `${res.flagged} flagged for new-class review${res.errors ? ` (${res.errors} errors)` : ''}`,
          );
          selected.clear();
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
        const ok = window.confirm(`Discard ${ids.length} crop${ids.length === 1 ? '' : 's'}?`);
        if (!ok) return;
        for (const id of ids) {
          try {
            await deleteCropLabel(id);
          } catch (e) {
            toastStore.error(`Discard failed: ${(e as Error).message}`);
          }
        }
        crops = crops.filter((c) => !ids.includes(c.id));
        selected = new Set();
      },
      'Discard selected',
    );
    reg('z', undoLast, 'Undo last action');
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
        size {cluster.size} · dominant
        <strong class="text-zinc-200">{cluster.dominant_class_name ?? '—'}</strong>
      </span>
    {/if}

    <span class="grow"></span>

    <div class="flex items-center gap-2">
      <button class="btn" type="button" onclick={selectAllPage}>Select page</button>
      <button class="btn" type="button" onclick={deselectAll}>Deselect</button>
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
      >
        Confirm Selected
      </button>

      <button class="btn" type="button" onclick={runGemma}>Run Gemma</button>
      <button class="btn" type="button" onclick={refine}>Refine (AHC)</button>
    </div>
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
    <div class="min-w-0 flex-1 overflow-auto p-4">
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
            {#if i === cutLineIndex && cutLineIndex > 0 && cutLineIndex < gridItems.length}
              <CutLine />
            {/if}
            <CropCard
              {crop}
              selected={selected.has(crop.id)}
              onclick={(c, e) => toggleSelect(c.id, e)}
              onacceptGemma={(c) => void acceptGemmaForCrop(c)}
              onrejectGemma={(c) => void rejectGemmaForCrop(c)}
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
    <span class="text-xs text-zinc-500">
      {selected.size} selected · {crops.length} loaded · {total} total
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if loadingMore}loading more…{:else if hasMore}{total - crops.length} more available — scroll to load{:else}all loaded{/if}
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
