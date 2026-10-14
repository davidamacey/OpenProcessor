/**
 * Slot-gallery controller — the state + logic behind a region slot's
 * gallery view on /clusters (`SlotGallery.svelte` renders it): the
 * browse pager over the slot's `queue.browsePath`, multi-select, the
 * secondary region-clustering sub-system (buckets, sub-cluster refine,
 * FP centroids, suspected-FP triage), the filter strip, and the
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
import {
  getCrop,
  getRegionClusters,
  getRegions,
  getSuspectedFalsePositives,
  type RegionBrowseItem,
  type SuspectedFpItem,
  apiErrorText,
} from '$lib/api';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Cluster, Crop } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';
import type { SlotSpec } from '$lib/annotations/types';
import { rowBoxOf } from '$lib/annotations/rowBox';
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

  // Region clustering (AHC-refinable buckets over the region embeddings).
  // selectedCluster narrows the gallery to one bucket; null shows the
  // bucket grid (or the flat gallery when no clustering has run).
  let clusters = $state<Cluster[]>([]);
  let selectedCluster = $state<number | null>(null);
  let clusterBusy = $state<boolean>(false);

  // The permanent false-positive bucket is identified by the served
  // `cluster_kind === 'false_positive'` on the selected cluster card —
  // never by its id (the backend's `-100` is just today's convention, not
  // a contract). Looked up in the already-loaded `clusters` list rather
  // than carried alongside `selectedCluster`, so a normal cluster that
  // happens to reuse that id is never mistaken for the FP bucket.
  const selectedClusterIsFalsePositive = $derived(
    clusters.find((c) => c.id === selectedCluster)?.cluster_kind === 'false_positive',
  );

  // Sub-cluster delineation inside an open region bucket — mirrors the item
  // cluster detail. null = "all" (the server returns regions ordered by subid so
  // AHC groups are contiguous; we render a labeled separator before each).
  // Selecting a chip filters the gallery to that one sub-cluster.
  let subTab = $state<string | null>(null);
  // Last refine outcome, shown inline next to the button so the result is not
  // just a transient toast (the run is fast and easy to miss).
  let refineMsg = $state<string | null>(null);

  // Suspected-FP review: crops the FP centroids flag as likely false
  // positives. Loads into the same `pager.items` so the shift-select +
  // Mark-FP workflow works unchanged; `pager.total` is pinned to the loaded
  // count so the infinite-scroll sentinel never pages in normal regions
  // over the top.
  let suspectedFpView = $state<boolean>(false);
  let suspectedFpThreshold = $state<number>(0.35);

  // M3 (docs/design/interactive-pass-2026-09-24.md): the bucket-card grid
  // below only ever showed real AHC region clusters — when the only
  // cluster is the permanent false-positive bucket (the common case
  // before "Cluster regions" has been run over the good regions), every
  // other region was unreachable from this view: no "unclustered" entry
  // and no fallback flat grid. `viewingAll` opens the same flat
  // gallery `selectedCluster !== null` already renders, but with no
  // `region_cluster_id` filter, so every region — clustered or not — is
  // browsable. Kept independent of `selectedCluster` (rather than
  // reusing it with a sentinel id) so `browseQuery` never needs to
  // distinguish "filter to real cluster 0" from "no filter".
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

  // A row's sub-cluster lives on its own box (`region_boxes[].cluster_subid`),
  // the one `region_box_id` names.
  function rowSubid(p: RegionBrowseItem): string | null {
    return rowBoxOf(p.slots?.[slot.key], p.region_box_id)?.clusterSubid ?? null;
  }

  // Distinct sub-cluster ids present in the loaded regions, sorted lexically so
  // "9a","9aa","9ab"… land in human-expected order.
  const subclusterIds = $derived.by(() => {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this computation, never stored in reactive state
    const set = new Set<string>();
    for (const p of pager.items) {
      const sub = rowSubid(p);
      if (sub) set.add(sub);
    }
    return [...set].sort();
  });

  // Per-subid counts for the separator-header labels ('__none__' = unrefined).
  const subCounts = $derived.by(() => {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this computation, never stored in reactive state
    const m = new Map<string, number>();
    for (const p of pager.items) {
      const k = rowSubid(p) ?? '__none__';
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
  });

  // Group only when a bucket is open, on the "all" tab, and refine has produced
  // sub-clusters. Otherwise render one flat group (no separators).
  const groupBySubid = $derived(
    selectedCluster != null && subTab == null && subclusterIds.length > 0,
  );

  // Partition loaded regions into one group PER sub-cluster id. Built with a
  // Map (not a contiguity walk) so it is robust to a non-contiguous list —
  // e.g. the transient render right after opening a bucket, when the pager
  // still holds the previous mixed-bucket gallery before the bucket's own
  // (subid-sorted) data arrives. A contiguity walk would emit the same subid
  // as multiple groups there, producing duplicate {#each} keys and a Svelte
  // each_key_duplicate crash that froze the detail view from opening.
  const groups = $derived.by(
    (): { key: string; label: string; items: RegionBrowseItem[] }[] => {
      if (!groupBySubid) return [{ key: '__all__', label: '', items: pager.items }];
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this computation, never stored in reactive state
      const byKey = new Map<string, RegionBrowseItem[]>();
      for (const p of pager.items) {
        const sub = rowSubid(p) ?? '__none__';
        let bucket = byKey.get(sub);
        if (!bucket) {
          bucket = [];
          byKey.set(sub, bucket);
        }
        bucket.push(p);
      }
      // Sort subids lexically; the '__none__' (unrefined) group always last.
      const keys = [...byKey.keys()].sort((a, b) => {
        if (a === '__none__') return 1;
        if (b === '__none__') return -1;
        return a < b ? -1 : a > b ? 1 : 0;
      });
      return keys.map((k) => ({
        key: k,
        label: k === '__none__' ? 'unrefined' : `sub-cluster ${k}`,
        items: byKey.get(k)!,
      }));
    },
  );

  function browseQuery(page: number): import('$lib/api').RegionsQuery {
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
      region_cluster_id: selectedCluster ?? undefined,
      // When a single sub-cluster tab is active, filter to it; otherwise (the
      // "all" tab) ask the server to order by sub-cluster so AHC groups come
      // back contiguous across pages and we can render them with separators.
      region_cluster_subid: subTab ?? undefined,
      sort_by_subid: selectedCluster != null && subTab == null ? true : undefined,
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
    get maxRank() {
      return maxRank;
    },
    get selectedCluster() {
      return selectedCluster;
    },
    get clusterBusy() {
      return clusterBusy;
    },
    set clusterBusy(v: boolean) {
      clusterBusy = v;
    },
    get refineMsg() {
      return refineMsg;
    },
    set refineMsg(v: string | null) {
      refineMsg = v;
    },
    get subTab() {
      return subTab;
    },
    set subTab(v: string | null) {
      subTab = v;
    },
    loadClusters,
    loadFirst,
  });

  function openCluster(id: number): void {
    suspectedFpView = false;
    viewingAll = false;
    subTab = null;
    refineMsg = null;
    // Clear the previous gallery synchronously so the render between selecting
    // the bucket and its data arriving doesn't group a stale mixed-bucket list.
    pager.items = [];
    selectedCluster = id;
  }

  /** M3: browse every region with no `region_cluster_id` filter — the
   *  entry point for regions that aren't in any AHC bucket yet (or when
   *  the only bucket that exists is the false-positive one). */
  function openAll(): void {
    suspectedFpView = false;
    subTab = null;
    refineMsg = null;
    pager.items = [];
    selectedCluster = null;
    viewingAll = true;
    void loadFirst();
  }

  function selectSubTab(sub: string | null): void {
    if (subTab === sub) return;
    subTab = sub;
    void loadFirst();
  }

  function backToClusters(): void {
    selectedCluster = null;
    viewingAll = false;
    subTab = null;
    refineMsg = null;
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
    get selectedClusterIsFalsePositive() {
      return selectedClusterIsFalsePositive;
    },
    get clusterBusy() {
      return clusterBusy;
    },
    get subTab() {
      return subTab;
    },
    get refineMsg() {
      return refineMsg;
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
    get subclusterIds() {
      return subclusterIds;
    },
    get subCounts() {
      return subCounts;
    },
    get groups() {
      return groups;
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
    runRefineCluster: jobs.runRefineCluster,
    openCluster,
    openAll,
    selectSubTab,
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
