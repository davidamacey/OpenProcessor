/**
 * Slot-gallery controller — the state + logic behind a region slot's
 * gallery view on /clusters (`SlotGallery.svelte` renders it): the
 * browse pager over the slot's `queue.browsePath`, multi-select, the
 * secondary region-clustering sub-system (clusters, FP centroids,
 * suspected-FP triage), the filter strip, and the
 * bbox-editor modal wiring.
 *
 * One controller per slot: `createSlotGalleryController(slot)` reads every
 * slot-specific value (browse path, lifecycle states, slot key) from the
 * `SlotSpec` it is given, never from a specific profile.
 *
 * Follows this codebase's `createPager`/`createSelection` factory-function
 * convention (a plain object of `$state` fields + closures, not a class).
 */

import { untrack } from 'svelte';
import { getCrop } from '$lib/api_cropRead';
import { apiErrorText } from '$lib/api_errors';
import {
  getRegionClusters,
  getRegions,
  getSuspectedFalsePositives,
  type RegionBrowseItem,
  type SuspectedFpItem,
} from '$lib/api_regions';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Cluster, Crop } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';
import type { SlotSpec } from '$lib/annotations/types';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';
import { createSlotGalleryJobs } from '$lib/slots/slotGalleryJobs';
import { createSlotGalleryWrites } from '$lib/slots/slotGalleryWrites';

const GALLERY_PAGE_SIZE = 60;

export function createSlotGalleryController(slot: SlotSpec) {
  // m9 (2026-09-24 interactive pass): the served `GET
  // {API_PREFIX}/regions/statuses` confirm/reject/false-positive statuses
  // win; the slot's own lifecycle states are the fallback for a
  // missing/pre-rollout endpoint, so the bulk-status buttons never break.
  const confirmState = (): string | undefined =>
    regionStatusesStore.confirmStatus ?? slot.capabilities.lifecycle?.confirmState;
  const rejectState = (): string | undefined =>
    regionStatusesStore.rejectStatus ?? slot.capabilities.lifecycle?.rejectState;
  const falsePositiveState = (): string | undefined =>
    regionStatusesStore.falsePositiveStatus ??
    slot.capabilities.lifecycle?.falsePositiveState;

  // per-box `state` triage, a different vocabulary from the
  // item-level region_status above (`accepted`/`rejected`/
  // `false_positive`/`proposed`, keyed here by the served `box_states`
  // entry's `role`). Falls back to the literal role string — the fixed
  // wire vocabulary, not a client guess — for a backend that
  // hasn't served `box_states` yet.
  const confirmBoxState = (): string =>
    regionStatusesStore.boxStateByRole('accepted') ?? 'accepted';
  const rejectBoxState = (): string =>
    regionStatusesStore.boxStateByRole('rejected') ?? 'rejected';
  const falsePositiveBoxState = (): string =>
    regionStatusesStore.boxStateByRole('false_positive') ?? 'false_positive';

  const browsePath = slot.capabilities.queue?.browsePath;
  // total_rows counts rows (boxes, on a box-selecting request) vs.
  // pager.total's item count — a side channel since createPager is generic
  // and only ever reads `.total`. rows_truncated: an item on the page
  // matched more boxes than the index reports per item.
  let totalRows = $state<number | null>(null);
  let rowsTruncated = $state<boolean>(false);
  const pager = createPager<RegionBrowseItem>({
    fetchPage: async (page) => {
      if (!browsePath) throw new Error(`slot "${slot.key}" declares no browse path`);
      const res = await getRegions(browsePath, browseQuery(page));
      totalRows = res.total_rows ?? null;
      rowsTruncated = res.rows_truncated ?? false;
      return res;
    },
    keyOf: (p) => p.row_key,
  });

  // Region triage: multi-select for bulk actions + the inline bbox editor.
  // Plain click TOGGLES here (accumulating), unlike the crop grid where
  // it replaces — region triage is a bulk-marking flow.
  const sel = createSelection({ plainClick: 'toggle' });
  let editCrop = $state<Crop | null>(null);
  let busy = $state<boolean>(false);

  // Top-N largest-crop gate. The sort is built on the largest 1-3 crops,
  // so this is the key filter for finding the regions that matter.
  // null = all ranks.
  let maxRank = $state<number | null>(null);

  // Region clusters (the serving region-box partition). selectedCluster is a
  // `cluster_uid` and narrows the gallery to that cluster; null shows the card
  // grid (or the flat gallery when no clustering has run).
  let clusters = $state<Cluster[]>([]);
  let selectedCluster = $state<string | null>(null);
  let clusterBusy = $state<boolean>(false);

  // Suspected-FP review: crops the FP centroids flag as likely false
  // positives. Loads into the same `pager.items` so the shift-select +
  // Mark-FP workflow works unchanged; `pager.total` is pinned to the loaded
  // count so the infinite-scroll sentinel never pages in normal regions
  // over the top.
  let suspectedFpView = $state<boolean>(false);
  let suspectedFpThreshold = $state<number>(0.35);

  // `viewingAll` opens the flat gallery with no `cluster_uid` filter, so every
  // region — clustered or not — is browsable. Kept independent of
  // `selectedCluster` so `browseQuery` never needs a sentinel.
  let viewingAll = $state<boolean>(false);

  // Filter sidebar state — only active on the gallery view.
  let detectorFilter = $state<string>('');
  let verifiedOnly = $state<boolean>(false);
  let minScore = $state<number>(0);
  let textQuery = $state<string>('');
  // dq-region (2026-09-24): backed by GET {API_PREFIX}/regions?status=, options
  // are the served region-status vocabulary (regionStatusesStore), not a
  // hardcoded list — includes verify_rejected (candidate-only rows), the
  // auto-confirmed-but-unreviewed 'detected' rows, etc.
  let statusFilter = $state<string>('');
  // Per-box state (`GET /regions/statuses` `box_states` vocabulary): every
  // box filter applies to the same box, so this narrows which boxes the
  // rows are about.
  let boxStateFilter = $state<string>('');

  function browseQuery(page: number): import('$lib/api_regions').RegionsQuery {
    return {
      page,
      page_size: GALLERY_PAGE_SIZE,
      detector: detectorFilter || undefined,
      verified: verifiedOnly || undefined,
      min_score: minScore > 0 ? minScore : undefined,
      text: textQuery || undefined,
      status: statusFilter || undefined,
      box_state: boxStateFilter || undefined,
      max_rank: maxRank ?? undefined,
      cluster_uid: selectedCluster ?? undefined,
    };
  }

  const loadFirst = () => pager.loadFirst();
  const loadMore = () => pager.loadMore();

  async function loadClusters(): Promise<void> {
    try {
      const res = await getRegionClusters({
        maxClusters: 500,
        maxRank: maxRank ?? undefined,
      });
      // Defensive: never render empty buckets (the backend already omits
      // them, but a stale response shouldn't surface a 0-size card).
      clusters = (res.clusters ?? []).filter((c) => c.size > 0);
    } catch (e) {
      toastStore.error(`Load ${slot.label.singular} clusters failed: ${apiErrorText(e)}`);
    }
  }

  async function loadSuspectedFp(): Promise<void> {
    if (clusterBusy) return;
    clusterBusy = true;
    try {
      const res = await getSuspectedFalsePositives({
        threshold: suspectedFpThreshold,
        pageSize: 200,
      });
      selectedCluster = null;
      sel.clear();
      pager.items = res.items as SuspectedFpItem[];
      pager.total = res.items.length;
      suspectedFpView = true;
      if (!res.centroids_built) {
        toastStore.info(res.message ?? 'No FP centroids yet — build them first.');
      } else {
        toastStore.success(
          `${res.total} suspected false positive(s) at ≤ ${suspectedFpThreshold}.`,
        );
      }
    } catch (e) {
      toastStore.error(`Load suspected FPs failed: ${apiErrorText(e)}`);
    } finally {
      clusterBusy = false;
    }
  }

  // Filter-driven reloads, called from one page $effect (so the state it
  // reads is what the effect tracks). Discrete filters reload the browse
  // at once; typed ones (text, min score) wait for a pause, since each
  // reload is a request. The cluster grid depends only on the rank gate and
  // the selected bucket, so a filter change never refetches it.
  const TYPED_FILTER_DEBOUNCE_MS = 250;
  let lastBrowseKey: string | null = null;
  let lastTypedKey: string | null = null;
  let lastClusterKey: string | null = null;
  let typedDebounce: ReturnType<typeof setTimeout> | null = null;

  function reloadOnFilterChange(): void {
    const browseKey = JSON.stringify([
      detectorFilter,
      verifiedOnly,
      statusFilter,
      boxStateFilter,
      maxRank,
      selectedCluster,
    ]);
    const typedKey = JSON.stringify([minScore, textQuery]);
    const clusterKey = JSON.stringify([maxRank, selectedCluster]);
    const clustersNeeded = selectedCluster == null && clusterKey !== lastClusterKey;
    untrack(() => {
      if (browseKey !== lastBrowseKey) {
        if (typedDebounce) clearTimeout(typedDebounce);
        typedDebounce = null;
        void loadFirst();
      } else if (typedKey !== lastTypedKey) {
        if (typedDebounce) clearTimeout(typedDebounce);
        typedDebounce = setTimeout(() => {
          typedDebounce = null;
          if (!disposed) void loadFirst();
        }, TYPED_FILTER_DEBOUNCE_MS);
      }
      if (clustersNeeded) void loadClusters();
    });
    lastBrowseKey = browseKey;
    lastTypedKey = typedKey;
    lastClusterKey = clusterKey;
  }

  // Set by the owning page's teardown: the clustering polls below stop, and
  // finish silently, once nobody is looking at this controller any more.
  let disposed = false;

  const jobs = createSlotGalleryJobs({
    slot,
    get disposed() {
      return disposed;
    },
    get clusterBusy() {
      return clusterBusy;
    },
    set clusterBusy(v: boolean) {
      clusterBusy = v;
    },
    loadClusters,
  });

  function openCluster(uid: string): void {
    suspectedFpView = false;
    viewingAll = false;
    // Clear the previous gallery synchronously so the render between selecting
    // the cluster and its data arriving doesn't show the previous list.
    pager.items = [];
    selectedCluster = uid;
  }

  /** M3: browse every region with no `cluster_uid` filter — the entry point for
   *  regions that aren't in any cluster yet. */
  function openAll(): void {
    suspectedFpView = false;
    pager.items = [];
    selectedCluster = null;
    viewingAll = true;
    void loadFirst();
  }

  function backToClusters(): void {
    selectedCluster = null;
    viewingAll = false;
    sel.clear();
    if (suspectedFpView) {
      // Leaving the suspected-FP view: reload the real region gallery the
      // filter effect would otherwise have populated.
      suspectedFpView = false;
      void loadFirst();
    }
  }

  // Range selects span the currently displayed order, so hand the helper
  // the loaded region ids on each click.
  function toggleSelect(p: RegionBrowseItem, e?: MouseEvent): void {
    sel.click(
      p.row_key,
      e,
      pager.items.map((x) => x.row_key),
    );
  }

  function selectAll(): void {
    sel.selectAll(pager.items.map((p) => p.row_key));
  }

  async function openEditor(p: RegionBrowseItem): Promise<void> {
    try {
      editCrop = await getCrop(p.crop_id);
    } catch (err) {
      toastStore.error(`Could not load ${slot.label.singular}: ${apiErrorText(err)}`);
    }
  }

  const writes = createSlotGalleryWrites({
    slot,
    pager,
    sel,
    get busy() {
      return busy;
    },
    set busy(v: boolean) {
      busy = v;
    },
    get editCrop() {
      return editCrop;
    },
    set editCrop(v: Crop | null) {
      editCrop = v;
    },
  });

  return {
    adoptReprocessed: writes.adoptReprocessed,
    get slot() {
      return slot;
    },
    confirmState,
    rejectState,
    falsePositiveState,
    confirmBoxState,
    rejectBoxState,
    falsePositiveBoxState,
    get pager() {
      return pager;
    },
    /** The served total_rows (see above) — null when absent. */
    get totalRows() {
      return totalRows;
    },
    get rowsTruncated() {
      return rowsTruncated;
    },
    get sel() {
      return sel;
    },
    get editCrop() {
      return editCrop;
    },
    set editCrop(v: Crop | null) {
      editCrop = v;
    },
    get busy() {
      return busy;
    },
    get maxRank() {
      return maxRank;
    },
    set maxRank(v: number | null) {
      maxRank = v;
    },
    get clusters() {
      return clusters;
    },
    get selectedCluster() {
      return selectedCluster;
    },
    get clusterBusy() {
      return clusterBusy;
    },
    get suspectedFpView() {
      return suspectedFpView;
    },
    get viewingAll() {
      return viewingAll;
    },
    get suspectedFpThreshold() {
      return suspectedFpThreshold;
    },
    set suspectedFpThreshold(v: number) {
      suspectedFpThreshold = v;
    },
    get detectorFilter() {
      return detectorFilter;
    },
    set detectorFilter(v: string) {
      detectorFilter = v;
    },
    get verifiedOnly() {
      return verifiedOnly;
    },
    set verifiedOnly(v: boolean) {
      verifiedOnly = v;
    },
    get minScore() {
      return minScore;
    },
    set minScore(v: number) {
      minScore = v;
    },
    get textQuery() {
      return textQuery;
    },
    set textQuery(v: string) {
      textQuery = v;
    },
    get boxStateFilter() {
      return boxStateFilter;
    },
    set boxStateFilter(v: string) {
      boxStateFilter = v;
    },
    get statusFilter() {
      return statusFilter;
    },
    set statusFilter(v: string) {
      statusFilter = v;
    },
    loadFirst,
    loadMore,
    loadClusters,
    loadSuspectedFp,
    runBuildFpCentroids: jobs.runBuildFpCentroids,
    runClustering: jobs.runClustering,
    reloadOnFilterChange,
    dispose(): void {
      disposed = true;
      if (typedDebounce) clearTimeout(typedDebounce);
      typedDebounce = null;
    },
    openCluster,
    openAll,
    backToClusters,
    toggleSelect,
    selectAll,
    openEditor,
    applyStatus: writes.applyStatus,
    cropIdsOfRows: writes.cropIdsOfRows,
    applyBoxState: writes.applyBoxState,
    saveBox: writes.saveBox,
    undoLastAction: writes.undoLastAction,
  };
}

export type SlotGalleryController = ReturnType<typeof createSlotGalleryController>;
