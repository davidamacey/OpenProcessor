<script lang="ts">
  import { page } from '$app/state';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { cohesionText, COHESION_TOOLTIP } from '$lib/clusters/clusterCardText';
  import { dndzone, SOURCES, TRIGGERS } from 'svelte-dnd-action';
  import {
    flagNeedsNewClass,
    getCluster,
    pollAutoLabelJob,
    refineCluster,
    runVlmOnCluster,
    type AutoLabelJobState,
    type ExcludeReason,
  } from '$lib/api';
  import {
    createClusterActionController,
    createExclusionGuard,
  } from '$lib/clusters/clusterController.svelte';
  import BlurSlider from '$components/BlurSlider.svelte';
  import CropCard from '$components/CropCard.svelte';
  import CropDetailModal from '$components/CropDetailModal.svelte';
  import CutLine from '$components/CutLine.svelte';
  import ScoreChip from '$components/ScoreChip.svelte';
  import ChevronDownIcon from '$components/ChevronDownIcon.svelte';
  import SemanticSearchBox from '$components/SemanticSearchBox.svelte';
  import ShortcutsButton from '$components/ShortcutsButton.svelte';
  import StrategyBar from '$components/StrategyBar.svelte';
  import SubjectScopeToggle from '$components/SubjectScopeToggle.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { createGridGroups } from '$lib/gridGroups.svelte';
  import { createPager } from '$lib/pager.svelte';
  import { computeCutLine } from '$lib/clusters/cutLine';
  import { createSelection } from '$lib/selection.svelte';
  import { createStrategyBar } from '$lib/strategyBar.svelte';
  import { isDiverseOverlayAvailable, isSemanticSearchAvailable } from '$lib/strategies';
  import { isAssignableClass } from '$lib/classVisibility';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import type { Cluster, Crop, PaginatedResponse } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { subscribeCurationEvents, type CurationEventSubscription } from '$lib/sse';

  const clusterIdParam = $derived(page.params.id);
  const clusterId = $derived(Number(clusterIdParam));

  let cluster = $state<Cluster | null>(null);

  // The served `cluster_kind` (not an id-equality guess) decides whether
  // this cluster has a class at all — 'class' clusters carry their own
  // `dominant_class_id`, and a candidate/unassigned cluster has no class
  // regardless of what its numeric id happens to collide with. Drives the
  // validated / labeled / cluster-total banner in the header.
  const clsForCluster = $derived(
    cluster?.cluster_kind === 'class' && cluster.dominant_class_id != null
      ? (classesStore.classes.find((c) => c.id === cluster!.dominant_class_id) ?? null)
      : null,
  );

  // Provenance echoed back by a pool-scale overlay ordering (Phase 4 —
  // order=diverse). Captured here (not by cropPager, which only retains
  // items/total) the same way `cluster` is captured out-of-band from the
  // fetchPage closure below. null whenever not in diverse mode or the
  // backend didn't send it.
  let orderMeta = $state<{
    method: string | null;
    version: string | null;
    n_pool: number | null;
  } | null>(null);

  // Human-readable cluster name for the header. Class clusters resolve to
  // the registry class name (cluster_id == class_id), falling back to the
  // backend's dominant_class_name. Candidate clusters (>= 10000) have no
  // name yet — label them so the operator knows it's unlabeled by design,
  // not a bug.
  const clusterName = $derived(
    clsForCluster?.name ?? cluster?.dominant_class_name ?? null,
  );

  const pageSize = 60;

  // Crop ids this page has just optimistically moved/labeled/discarded/
  // ignored out of the cluster. A GET for this cluster's crops can be
  // in flight (or triggered fresh, e.g. by the SSE live-refresh effect
  // below) at the exact moment a human drag-drop or hotkey move fires;
  // if that GET's snapshot predates our write, its response would
  // otherwise silently overwrite the optimistic removal and the crop
  // would flicker back in and *stay* until a hard reload — the crop
  // really did move, the grid just re-showed stale data. `exclusionGuard`
  // re-applies this exclusion to every fetchPage result (loadFirst
  // included) so a stale response can never resurrect a crop we already
  // know left. Created here (rather than inside the action controller
  // below) so it can be wired into `cropPager`'s `accept` option without
  // a forward-declared `let controller` — see `createExclusionGuard`'s
  // doc comment in clusterController.svelte.ts. Cleared per-id when the
  // corresponding action is undone or reverted on failure, so a crop
  // that never actually left is never hidden.
  const exclusionGuard = createExclusionGuard();

  // Crop pager. cropQuery() feeds page 1 and every later page, so a filter
  // can't be applied to the first request and silently dropped on the next.
  const cropPager = createPager<Crop>({
    fetchPage: async (page) => {
      const res = await getCluster(clusterId, page, pageSize, undefined, cropQuery());
      cluster = res.cluster;
      orderMeta =
        orderMode === 'diverse'
          ? {
              method: res.crops.order_method ?? null,
              version: res.crops.order_version ?? null,
              n_pool: res.crops.n_pool ?? null,
            }
          : null;
      return res.crops as PaginatedResponse<Crop>;
    },
    keyOf: (c) => c.id,
    accept: (c) => exclusionGuard.accept(c),
    onLoadFirstError: () => {
      cluster = null;
    },
  });

  // Crop opened in the read-only details modal (info button on each card).
  let detailCrop = $state<Crop | null>(null);

  // Selection set (crop_id) + shift-range anchor.
  const sel = createSelection({ plainClick: 'replace' });

  // Sub-cluster tab (null = all). Backend stores cluster_subid as a
  // keyword string ("47a", "47b", "47aa", ...).
  let subTab = $state<string | null>(null);

  // class_source filter (null = all). Drives the chip-group in the
  // header and propagates to {API_PREFIX}/crops?class_source=... so the grid
  // shows only crops from one source bucket. Cluster card stats in
  // the header are NOT recomputed by this filter — the operator sees
  // the filter against the whole-cluster totals on purpose.
  let classSourceFilter = $state<string | null>(null);
  // Every class_source this deployment can write — the backend's catalog
  // (GET /class_sources). No catalog, no source filter.
  const classSourceOptions = $derived(classSourcesStore.list);

  // Primary-subject scope: 0 = all crops, 1 = largest only, 2 = largest + 2nd.
  // Maps to the {API_PREFIX}/crops?max_rank= filter (the "biggest subject in frame" the
  // business sorts on). null = no rank filter.
  let subjectScope = $state<0 | 1 | 2>(0);
  const maxRank = $derived<number | null>(subjectScope === 0 ? null : subjectScope);

  // Order strategy (curation-strategy plan Phase 3/4 — generalizes the old
  // outliersFirst boolean into an id string so the fuller StrategyBar
  // selector and the "Outliers first" shortcut button share one source
  // of truth). `strategyBar.sort` IS the order id here — 'default' means
  // newest-first (today's behavior), unchanged from before Phase 3.
  // 'default'/'outliers' are always offered (see the `allowedOrderIds`
  // passed to <StrategyBar> below): `{API_PREFIX}/crops?order=` only special-cases
  // those server-side per docs/curation-strategy-plan-2026-09.md §1, so
  // any other id would silently do nothing were it offered here — except
  // 'diverse' (Phase 4), which is *conditionally* offered, gated purely on
  // {API_PREFIX}/methods reporting it (mirrors the mistakenness-filter gating
  // pattern in StrategyBar.svelte — see isDiverseOverlayAvailable).
  const strategyBar = createStrategyBar({ defaultId: 'default' });
  const orderMode = $derived(strategyBar.sort);
  // Outliers-first: rank members by distance from the cluster centroid (most
  // atypical first) so mislabels / junk in this cluster float to the top.
  // Computed on-the-fly + cached server-side. Off = newest-first.
  const outliersFirst = $derived(orderMode === 'outliers');

  // 'diverse' (Phase 4, core-set / k-center-greedy pool selection) is only
  // ever offered when {API_PREFIX}/methods reports it at stable/experimental status
  // — a pre-Phase-4 backend, or OP_SELECT_DIVERSE_ENABLED off, means the
  // 'diverse' id is passed through but StrategyBar's own gate (same
  // predicate) never actually surfaces it in the <select>, so this page
  // never ends up requesting an order the backend doesn't support.
  const diverseAvailable = $derived(
    isDiverseOverlayAvailable(strategiesStore.methods.overlays),
  );
  const allowedOrderIds = $derived(
    diverseAvailable ? ['default', 'outliers', 'diverse'] : ['default', 'outliers'],
  );

  // P2-14 semantic text search. Gated behind isSemanticSearchAvailable
  // exactly like isDiverseOverlayAvailable above — an old/flag-off
  // backend hides <SemanticSearchBox> entirely. Search results are fed
  // straight into cropPager's settable items/total (searchScores keyed
  // by crop id, for the similarity badge) so the existing CropCard grid,
  // selection, DnD, and label hotkeys below keep working completely
  // unchanged — this is not a parallel rendering path. While a search is
  // active, "load more" is disabled (see hasMore guard in the pager
  // section below): {API_PREFIX}/search/text pagination isn't wired to this
  // cluster page's infinite-scroll trigger, and re-paging into
  // getCluster would silently overwrite the search results.
  const semanticSearchAvailable = $derived(
    isSemanticSearchAvailable(strategiesStore.methods.overlays),
  );
  let searchModeActive = $state(false);
  let searchScores = $state(new Map<string, number>());
  // "How many diverse crops?" — defaults to the page's own pageSize
  // (60) until the operator overrides it via the StrategyBar stepper.
  // Only forwarded to the API while actually in diverse mode.
  const diverseK = $derived(orderMode === 'diverse' ? (strategyBar.k ?? pageSize) : null);

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

  function cropQuery() {
    return {
      classSource: classSourceFilter,
      maxRank,
      minBlurRatio,
      // DQ-M3 (docs/design/data-quality-pass-2026-09-24.md): backend main
      // 7254ec4 now serves `order=core_first` on `GET {API_PREFIX}/crops`
      // (nearest-to-centroid first, with cluster_distance/
      // cluster_similarity/cluster_is_core recomputed against the live
      // centroid) — exactly what the cut line needs and previously had
      // no way to request, which is why computeCutLine() (cutLine.ts)
      // has to defensively hide the line whenever the loaded order isn't
      // actually core-first-consistent. Requesting it here (whenever the
      // operator hasn't picked their own explicit order) means that
      // defensive check now passes in the common case instead of always
      // falling back to hidden — computeCutLine() is left in place
      // as-is: it still hides the line for any cluster the backend
      // hasn't backfilled cluster_is_core for, or an order override the
      // operator explicitly picked (outliers/diverse) genuinely isn't
      // core-first.
      order: orderMode === 'default' ? 'core_first' : orderMode,
      k: diverseK,
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
    void orderMode;
    void diverseK;
    void loadFirst();
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

  // Sub-cluster filtering (in-memory, after load). cluster_subid is
  // backend-owned (keyword like "47a") — the frontend string-matches.
  const filteredCrops = $derived(
    subTab == null
      ? cropPager.items
      : cropPager.items.filter((c) => c.cluster_subid === subTab),
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
  // Outliers-first (and, Phase 4, diverse mode) wins over sub-cluster
  // grouping: when ranking by centroid distance, or when the server
  // already hand-picked a diverse subset/order, we want one flat,
  // server-ordered list — not a regroup by subid that would scramble it.
  const groupBySubcluster = $derived(
    !outliersFirst &&
      orderMode !== 'diverse' &&
      subTab == null &&
      subClusterIds.length > 0,
  );

  // The rendered grid. `groups` is a real $derived of filteredCrops +
  // groupBySubcluster, so it can never fall behind cropPager.items; the dnd
  // handlers below install a *transient*, id-reconciled override for the
  // duration of a drag and drop it again on finalize. Before this it was a
  // plain $state snapshot that consider/finalize wrote the dnd library's own
  // (pre-drop) list into, with nothing to re-derive it afterwards — which is
  // how a labelled-away crop could get painted back into the grid and stay
  // there until a hard reload. See src/lib/gridGroups.svelte.ts.
  const grid = createGridGroups<Crop>({
    source: () => filteredCrops,
    grouped: () => groupBySubcluster,
    keyOf: (c) => c.id,
    subidOf: (c) => c.cluster_subid ?? null,
    // Truth is the *whole* pager buffer, not filteredCrops: a sub-cluster tab
    // narrows what's rendered but doesn't mean the hidden crops left.
    liveIds: () => new Set(cropPager.items.map((c) => c.id)),
  });
  const gridGroups = $derived(grid.groups);

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
    getVisibleCrops: () => filteredCrops,
    getClusterId: () => clusterId,
    rememberTarget: (id) => rememberTarget(id),
    loadFirst,
  });

  // Cut-line index: the server's own `cluster_is_core` flag (computed
  // against `{API_PREFIX}/clusters`' `core_similarity_min`, echoed onto
  // `cluster.core_similarity_min` — no client 0.75 constant) decides which
  // leading crops are "core". Only meaningful in the single '__all__'
  // group; suppressed while grouping by sub-cluster (subid order wins).
  //
  // DQ-M3 (docs/design/data-quality-pass-2026-09-24.md): this used to
  // assume "crops are already sorted core-first by the API" — false.
  // `cluster_is_core` is null on 1,000/1,000 class-cluster members, and
  // the default order (`sort=updated_at:desc`) isn't core-first at all —
  // live, a class cluster's member #1 usually isn't core (this old logic
  // degenerately produced index 0, effectively already hidden by the
  // `cutLineIndex > 0` template guard below), while a candidate cluster
  // has `cluster_is_core` set on every member but in recency order, so
  // the old "stop at first non-core" logic drew a line at a meaningless
  // boundary with core crops resuming right after it (#10000: break at
  // 181, core again from 182). There's no server-side core-first order to
  // request instead (checked against the vendored OpenAPI contract) — so
  // computeCutLine() verifies the loaded order is actually
  // core-first-consistent (and not mostly null) before trusting any
  // boundary, rather than drawing one on a guess. See
  // src/lib/clusters/cutLine.ts.
  const cutLine = $derived.by(() => computeCutLine(filteredCrops));
  const cutLineIndex = $derived(cutLine.index);
  const cutLineVisible = $derived(cutLine.visible);

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

  // In-flight guards so the operator gets feedback and can't double-fire
  // these long-running cluster ops.
  let refining = $state<boolean>(false);
  let vlmRunning = $state<boolean>(false);
  // Live status while POST /vlm/label_cluster/{id}'s job runs, polled via
  // pollAutoLabelJob — rendered as a compact inline stage/progress string
  // next to the Run VLM button (2026-09-24 logic-moves W3).
  let vlmJob = $state<AutoLabelJobState | null>(null);

  async function runVlm(): Promise<void> {
    if (vlmRunning) return;
    // p2/M7: a nag-confirm, unlike the one-keystroke label/move/discard
    // actions above (which Z undoes instantly) — this kicks off a
    // background VLM sweep over every unvalidated member of the cluster
    // with no undo, so a misclick isn't free.
    if (
      !window.confirm(`Run the VLM over every unvalidated crop in cluster #${clusterId}?`)
    ) {
      return;
    }
    vlmRunning = true;
    vlmJob = null;
    try {
      vlmJob = await runVlmOnCluster(clusterId);
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
      toastStore.error(`VLM run failed: ${(e as Error).message}`);
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
    await controller.assignClassToSelected(confirmClassId);
    void advance();
  }

  async function advance(): Promise<void> {
    // Advance: jump to next unvalidated crop. If we're at the end of what's
    // loaded but more pages exist, fetch them; otherwise tell the user.
    const next = filteredCrops.find((c) => !c.class_validated && !sel.has(c.id));
    if (next) {
      sel.ids = new Set([next.id]);
    } else if (cropPager.hasMore) {
      await loadMore();
      const nextAfterLoad = filteredCrops.find(
        (c) => !c.class_validated && !sel.has(c.id),
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

  // ---------------- SSE: live updates for this cluster's class ----
  // cluster_id == class_id holds for class-kind clusters (see
  // `ClusterKind` in src/lib/types.ts and the scoping note on getCluster
  // in src/lib/api.ts), so we subscribe with `class_id=clusterId`.
  // Crop.created without a class is hidden from per-class pages by
  // event_hub's filter. Candidate clusters have no class, and event_hub
  // filters on exact class_id equality, so a subscription there could
  // never deliver anything — skip it rather than hold an idle stream open.
  let liveNewCount = $state<number>(0);
  let scrolledPastFirst20 = $state<boolean>(false);
  let liveSub: CurationEventSubscription | null = null;
  let scrollEl = $state<HTMLDivElement | null>(null);
  const liveClassId = $derived(clsForCluster?.id ?? null);

  $effect(() => {
    if (liveClassId == null) return;
    liveSub = subscribeCurationEvents({
      class_id: liveClassId,
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
    if (
      liveNewCount > 0 &&
      !scrolledPastFirst20 &&
      !cropPager.loading &&
      !cropPager.loadingMore
    ) {
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
      controller.acceptAllVlmOnPage,
      'Confirm all VLM suggestions on page',
    );
    reg(
      'g',
      async () => {
        const ids = [...sel.ids];
        for (const id of ids) {
          const c = cropPager.items.find((cc) => cc.id === id);
          if (c) await controller.acceptVlmForCrop(c);
        }
      },
      'Accept VLM suggestion for selected',
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
    reg('d', controller.discardSelected, 'Discard selected');
    reg('z', controller.undoLast, 'Undo last action');
    reg(
      'x',
      () => void ignoreSelected('ignore'),
      'Ignore selected (exclude from training)',
    );
    reg('u', controller.undoIgnore, 'Undo last ignore');
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
    reg('m', openMovePicker, 'Move selected to cluster…');
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
          grid.reset();
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
  <div class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
    <a href="/clusters" class="btn shrink-0" title="Back to all clusters">
      ← All clusters
    </a>
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
    <ShortcutsButton />
    {#if cluster}
      <span class="text-xs text-zinc-400">
        size {cluster.size.toLocaleString()} · dominant
        <strong class="text-zinc-200">{cluster.dominant_class_name ?? '—'}</strong>
      </span>
      <!-- DQ-M2 fix (dq-queues cutover, 2026-09-24): purity is now the
           nearest-centroid geometry share — purity_tier stays the badge
           elsewhere (the sidebar/card border), this is the number + its
           basis/n, with label_purity/labelled_share in the tooltip so an
           operator can tell "geometrically coherent" from "the labels we
           have agree with each other". -->
      {#if cluster.purity != null}
        <span
          class="font-mono text-xs text-zinc-500"
          data-testid="cluster-cohesion"
          title="{COHESION_TOOLTIP} · label agreement {cluster.label_purity != null
            ? `${(cluster.label_purity * 100).toFixed(0)}%`
            : '—'} · labelled share {cluster.labelled_share != null
            ? `${(cluster.labelled_share * 100).toFixed(0)}%`
            : '—'}"
        >
          · {cohesionText(cluster)}
        </span>
      {/if}
    {/if}
    {#if cluster}
      <!-- K1 (visual audit 2026-09-24): this chip used to read "35
           validated · 135 labeled · 134 in cluster" — the first two are
           the class registry's CLASS-WIDE counts (every crop of the class,
           test holdout included), the last this cluster's own size, so
           "labeled" could exceed "in cluster". Each number now names its
           scope; the class-wide pair stays on classesStore because it is
           what re-fetches after a label write. -->
      <span
        class="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px] text-zinc-300"
        title="validated · labeled · cluster total — class-wide counts are every crop of this class (test holdout included), from the class registry; 'in this cluster' is this cluster's own member count (test holdout included). The grid lists only reviewable members."
        data-testid="cluster-header-counts"
      >
        {#if clsForCluster}
          <span class="text-zinc-500">class-wide:</span>
          <span class="text-emerald-300"
            >{(clsForCluster.validated_count ?? 0).toLocaleString()}</span
          >
          <span class="text-zinc-500">validated</span>
          <span class="mx-1 text-zinc-600">·</span>
          <span class="text-blue-300">{(clsForCluster.count ?? 0).toLocaleString()}</span>
          <span class="text-zinc-500">labeled</span>
          <span class="mx-1 text-zinc-600">|</span>
        {/if}
        <span class="text-zinc-200">{cluster.size.toLocaleString()}</span>
        <span class="text-zinc-500">in this cluster</span>
      </span>
    {/if}

    <span class="grow"></span>

    <div class="flex flex-wrap items-center gap-x-0.5 gap-y-1">
      <button class="btn" type="button" onclick={selectAllPage} title="A">
        Select page
      </button>
      <button class="btn" type="button" onclick={deselectAll} title="Esc">
        Deselect
      </button>
      <span class="font-mono text-xs text-zinc-500">{sel.size} selected</span>

      <select bind:value={confirmClassId} class="select">
        <option value={null}>— class —</option>
        {#each classesStore.classes.filter(isAssignableClass) as cls (cls.id)}
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
        onclick={controller.acceptAllVlmOnPage}
        title="Shift+Enter — accept all VLM suggestions on this page"
      >
        Accept VLM <kbd class="ml-1 font-mono text-[10px] text-zinc-400">⇧↵</kbd>
      </button>

      <span class="mx-1 h-5 w-px bg-zinc-800"></span>

      <button class="btn" type="button" onclick={runVlm} disabled={vlmRunning}>
        {#if vlmRunning}
          <span
            class="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-100 align-[-1px]"
          ></span>
          Running…
        {:else}
          Run VLM
        {/if}
      </button>
      {#if vlmRunning && vlmJob}
        <span class="text-xs text-zinc-400">
          {vlmJob.stage || 'preparing…'}{vlmJob.total > 0
            ? ` (${vlmJob.processed}/${vlmJob.total})`
            : ''}
        </span>
      {/if}
      <button
        class="btn"
        type="button"
        onclick={refine}
        disabled={refining}
        title="Sub-cluster this cluster with AHC"
      >
        {#if refining}
          <span
            class="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-100 align-[-1px]"
          ></span>
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
          class="btn btn-join-start"
          type="button"
          title="Ignore selected — exclude from training + clustering (X)"
          onclick={() => void ignoreSelected('ignore')}
        >
          Ignore <kbd class="ml-1 font-mono text-[10px] text-zinc-400">X</kbd>
        </button>
        <button
          class="btn btn-icon btn-join-end"
          type="button"
          aria-label="Choose ignore reason"
          onclick={() => (ignoreMenuOpen = !ignoreMenuOpen)}
        >
          <ChevronDownIcon size={18} />
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
    <label class="flex items-center gap-1.5">
      <select
        aria-label="Filter by label source"
        class="select-sm"
        value={classSourceFilter ?? ''}
        onchange={(e) => {
          const v = (e.currentTarget as HTMLSelectElement).value;
          classSourceFilter = v === '' ? null : v;
        }}
      >
        <option value="">All sources</option>
        {#each classSourceOptions as opt (opt.id)}
          <option value={opt.id}>{opt.label}</option>
        {/each}
      </select>
    </label>
    {#if classSourceFilter !== null}
      <span class="ml-auto text-zinc-500">
        showing {cropPager.total.toLocaleString()} from {classSourcesStore.labelFor(
          classSourceFilter,
        )}
      </span>
    {/if}
  </div>

  <!-- Primary-subject controls: focus on the largest subject(s) in frame
       (what the business sorts on) and hide too-blurry crops. View-only —
       no data is deleted; the slider commits on release to avoid a reload
       per pixel. -->
  <div
    class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-1.5 text-xs"
  >
    <!-- Cluster-scoped semantic search — left-aligned, first control in
         the row, so it reads as the primary way in rather than a control
         squeezed wherever there happens to be leftover flex space. -->
    {#if semanticSearchAvailable}
      <SemanticSearchBox
        filter={{ cluster_id: clusterId }}
        pageSize={200}
        onResults={(res) => {
          searchModeActive = true;
          searchScores = new Map(res.items.map((it) => [it.id, it.similarity_score]));
          cropPager.items = res.items;
          cropPager.total = res.total;
        }}
        onClear={() => {
          searchModeActive = false;
          searchScores = new Map();
          void cropPager.loadFirst();
        }}
      />
    {/if}

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
         the top, so wrong/atypical items are easy to cherry-pick out. Kept
         as a direct shortcut into strategyBar.sort — same muscle memory as
         before this phase, just backed by the generalized order id now. -->
    <button
      type="button"
      class="btn-sm {outliersFirst
        ? 'border-amber-500/60 bg-amber-500/20 text-amber-100'
        : 'border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
      onclick={() => (strategyBar.sort = outliersFirst ? 'default' : 'outliers')}
      title="Sort by distance from the cluster centroid (most atypical first) to spot mislabels/junk"
    >
      {outliersFirst ? '◤ Outliers first' : 'Outliers first'}
    </button>

    <!-- Fuller order selector — additive alongside the shortcut above.
         Filters are hidden here: the crop query doesn't forward
         min-mistakenness / hide-near-dup params (Phase 3 review-only).
         'diverse' (Phase 4) is included in allowedOrderIds unconditionally
         — StrategyBar only actually renders it once {API_PREFIX}/methods reports
         the overlay, so this is harmless against a backend that hasn't
         shipped it yet. -->
    <StrategyBar
      bar={strategyBar}
      allowedIds={allowedOrderIds}
      showFilters={false}
      diverseKDefault={pageSize}
      diverseMeta={orderMeta}
    />

    {#if subjectScope !== 0 || minBlurRatio !== null || !strategyBar.isDefault}
      <button
        type="button"
        class="btn-sm ml-auto bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
        onclick={() => {
          subjectScope = 0;
          blurSlider = 0;
          minBlurRatio = null;
          strategyBar.reset();
        }}
      >
        reset
      </button>
    {/if}
  </div>

  <!-- Sub-cluster tabs -->
  {#if subClusterIds.length > 0}
    <div class="flex items-center gap-2 border-b border-zinc-800 px-4 py-1.5 text-xs">
      <span class="text-zinc-500">sub-clusters:</span>
      <button
        type="button"
        class="chip {subTab === null
          ? 'bg-blue-600 text-white'
          : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
        onclick={() => (subTab = null)}
      >
        all
      </button>
      {#each subClusterIds as sid (sid)}
        <button
          type="button"
          class="chip {subTab === sid
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
       left sidebar now (the sorter-app UX). -->

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
             type='crop-card' so a card drags out to the ClassSidebar (or
             across groups) exactly as before. -->
        {#each gridGroups as group (group.key)}
          {#if groupBySubcluster}
            <div
              class="mt-3 flex items-center gap-2 pt-1 text-xs font-medium text-zinc-300 first:mt-0"
            >
              <span class="rounded bg-zinc-800 px-2 py-0.5 text-zinc-100"
                >{group.label}</span
              >
              <span class="text-zinc-500">
                {group.items.length} crop{group.items.length === 1 ? '' : 's'}
              </span>
              <span class="h-px flex-1 bg-zinc-800"></span>
            </div>
          {/if}
          <div
            class="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-3 {groupBySubcluster
              ? 'mt-2'
              : ''}"
            use:dndzone={{
              items: group.items,
              type: 'crop-card',
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
              {#if !groupBySubcluster && cutLineVisible && i === cutLineIndex}
                <CutLine />
              {/if}
              <div class="relative">
                <CropCard
                  {crop}
                  selected={sel.has(crop.id)}
                  onclick={(c, e) => clickSelect(c.id, e)}
                  onacceptVlm={(c) => void controller.acceptVlmForCrop(c)}
                  onrejectVlm={(c) => void controller.rejectVlmForCrop(c)}
                  ondetail={(c) => (detailCrop = c)}
                />
                {#if searchModeActive && searchScores.has(crop.id)}
                  <div class="pointer-events-none absolute left-1 top-1 z-10">
                    <ScoreChip
                      label="match"
                      value={searchScores.get(crop.id) ?? 0}
                      size="sm"
                    />
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/each}
        <!-- Sentinel inside the scroll container so IntersectionObserver
             roots on the right element (the overflow-auto parent). -->
        <div
          use:infiniteScroll={{
            onload: loadMore,
            disabled:
              searchModeActive ||
              cropPager.loadingMore ||
              !cropPager.hasMore ||
              cropPager.loading,
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
      <span
        title="Loaded / reviewable members matching the current filters. Test-holdout and ignored crops are never listed, so this can be lower than the header's cluster size."
        >{cropPager.items.length} / {cropPager.total} listed</span
      >
      {#if sel.size > 0}<span class="ml-2 text-blue-300">· {sel.size} selected</span>{/if}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if cropPager.loadingMore}loading more…{:else if cropPager.hasMore}scroll for more{:else}all
        loaded{/if}
    </span>
  </div>
</div>

{#if movePickerOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Move crops to cluster"
    tabindex="-1"
    use:trapFocus={{ onEscape: cancelMovePicker }}
    onclick={(e) => {
      if (e.target === e.currentTarget) cancelMovePicker();
    }}
  >
    <div
      class="w-full max-w-sm rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-2 text-base font-semibold">
        Move {sel.size} crop{sel.size === 1 ? '' : 's'}
      </h3>
      <p class="mb-3 text-xs text-zinc-400">
        Move these from cluster #{clusterId} to a target cluster id. The operation is reversible
        per crop via the cluster page.
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
        <button
          type="button"
          class="btn btn-primary"
          onclick={() => void confirmMovePicker()}
        >
          Move
        </button>
      </div>
    </div>
  </div>
{/if}

{#if detailCrop}
  <CropDetailModal crop={detailCrop} onclose={() => (detailCrop = null)} />
{/if}
