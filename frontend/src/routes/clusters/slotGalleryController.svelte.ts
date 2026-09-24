/**
 * Plate-gallery controller — the state + logic behind the license_plate
 * "plates list" view on /clusters, extracted verbatim out of
 * clusters/+page.svelte (P2.6, docs/genericization-plan-2026-09-13.md
 * §3.4/§5a) so `SlotGallery.svelte` can own the rendering while this
 * module owns the ~30-item state/logic surface: the plate pager,
 * multi-select, the secondary AHC plate-clustering sub-system (buckets,
 * sub-cluster refine, FP centroids, suspected-FP triage), the filter
 * strip, and the bbox-editor modal wiring.
 *
 * Follows this codebase's existing `createPager`/`createSelection`
 * factory-function convention (a plain object of `$state` fields +
 * closures, not a class) rather than inventing a new pattern for this
 * extraction.
 *
 * Deliberately NOT parameterized yet (P2.6 is a verbatim move; P2.7
 * parameterizes). Every field/method name here is identical to what
 * `clusters/+page.svelte` used to have inline — this is the "moves
 * everything the plate view needs, changes nothing about it" half of
 * the plan's two-commit split.
 */

import {
  batchRegionStatus,
  buildRegionFpCentroids,
  clusterRegions,
  getCrop,
  getRegionClusters,
  getRegionClusterStatus,
  getRegionFpCentroidStatus,
  getRegions,
  getRegionThumbUrl,
  getSuspectedFalsePositives,
  refineRegionCluster,
  type RegionBrowseItem,
  type SuspectedFpItem,
} from '$lib/api';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Cluster, Crop } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

// m9 (2026-09-24 interactive pass): the review tab reads
// confirm/reject/false_positive status from the served
// `GET {API_PREFIX}/regions/statuses` (`regionStatusesStore`); this gallery
// used to read only the hand-maintained slot-profile literal. Now tries
// the served value first and falls back to the profile — same
// degrade-to-static-default contract `regionStatusesStore`'s own doc
// comment documents, so a missing/pre-rollout endpoint never breaks the
// bulk-status buttons.
export function PLATE_CONFIRM_STATE(): string {
  return (
    regionStatusesStore.confirmStatus ??
    licensePlateSlot.capabilities.lifecycle!.confirmState
  );
}
export function PLATE_REJECT_STATE(): string {
  return (
    regionStatusesStore.rejectStatus ??
    licensePlateSlot.capabilities.lifecycle!.rejectState
  );
}
export function PLATE_FALSE_POSITIVE_STATE(): string {
  return (
    regionStatusesStore.falsePositiveStatus ??
    licensePlateSlot.capabilities.lifecycle!.falsePositiveState!
  );
}

/** Mirrors FALSE_POSITIVE_PLATE_CLUSTER_ID in the API (clustering/orchestrator.py). */
export const FP_PLATE_CLUSTER_ID = -100;

const GALLERY_PAGE_SIZE = 60;

export function createSlotGalleryController() {
  const pager = createPager<RegionBrowseItem>({
    fetchPage: async (page) =>
      await getRegions(
        licensePlateSlot.capabilities.queue!.browsePath,
        browseQuery(page),
      ),
    keyOf: (p) => p.crop_id,
  });

  // Plate triage: multi-select for bulk actions + the inline bbox editor.
  // Plain click TOGGLES here (accumulating), unlike the crop grid where
  // it replaces — plate triage is a bulk-marking flow.
  const sel = createSelection({ plainClick: 'toggle' });
  let editCrop = $state<Crop | null>(null);
  let busy = $state<boolean>(false);

  // Top-N largest-crop gate for plates. The sort is built on the largest
  // 1-3 crops, so this is the key filter for finding the plates that matter.
  // null = all ranks.
  let maxRank = $state<number | null>(null);

  // Plate clustering (AHC-refinable buckets over plate_pe_embedding).
  // selectedCluster narrows the gallery to one bucket; null shows the
  // bucket grid (or the flat gallery when no clustering has run).
  let clusters = $state<Cluster[]>([]);
  let selectedCluster = $state<number | null>(null);
  let clusterBusy = $state<boolean>(false);

  // Sub-cluster delineation inside an open plate bucket — mirrors the vehicle
  // cluster detail. null = "all" (the server returns plates ordered by subid so
  // AHC groups are contiguous; we render a labeled separator before each).
  // Selecting a chip filters the gallery to that one sub-cluster.
  let subTab = $state<string | null>(null);
  // Last refine outcome, shown inline next to the button so the result is not
  // just a transient toast (the run is fast and easy to miss).
  let refineMsg = $state<string | null>(null);

  // Suspected-FP review: crops the FP centroids flag as likely false
  // positives. Loads into the same `plates` array so the beloved
  // shift-select + Mark-FP workflow works unchanged; platesTotal is pinned to
  // the loaded count so the infinite-scroll sentinel never pages in normal
  // plates over the top.
  let suspectedFpView = $state<boolean>(false);
  let suspectedFpThreshold = $state<number>(0.35);

  // M3 (docs/design/interactive-pass-2026-09-24.md): the bucket-card grid
  // below only ever showed real AHC region clusters — when the only
  // cluster is the permanent false-positive bucket (the common case
  // before "Cluster plates" has been run over the good plates), every
  // other plate was unreachable from this view: no "unclustered" entry
  // and no fallback flat grid. `viewingAll` opens the same flat
  // gallery `selectedCluster !== null` already renders, but with no
  // `region_cluster_id` filter, so every plate — clustered or not — is
  // browsable. Kept independent of `selectedCluster` (rather than
  // reusing it with a sentinel id) so `browseQuery` never needs to
  // distinguish "filter to real cluster 0" from "no filter".
  let viewingAll = $state<boolean>(false);

  // Filter sidebar state — only active on the plates view.
  let detectorFilter = $state<string>('');
  let verifiedOnly = $state<boolean>(false);
  let minScore = $state<number>(0);
  let textQuery = $state<string>('');
  // dq-region (2026-09-24): backed by GET {API_PREFIX}/regions?status=, options
  // are the served region-status vocabulary (regionStatusesStore), not a
  // hardcoded list — includes verify_rejected (candidate-only rows), the
  // auto-confirmed-but-unreviewed 'detected' rows, etc.
  let statusFilter = $state<string>('');

  // Distinct sub-cluster ids present in the loaded plates, sorted lexically so
  // "9a","9aa","9ab"… land in human-expected order.
  const subclusterIds = $derived.by(() => {
    const set = new Set<string>();
    for (const p of pager.items)
      if (p.region_cluster_subid) set.add(p.region_cluster_subid);
    return [...set].sort();
  });

  // Per-subid counts for the separator-header labels ('__none__' = unrefined).
  const subCounts = $derived.by(() => {
    const m = new Map<string, number>();
    for (const p of pager.items) {
      const k = p.region_cluster_subid ?? '__none__';
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
  });

  // Group only when a bucket is open, on the "all" tab, and refine has produced
  // sub-clusters. Otherwise render one flat group (no separators).
  const groupBySubid = $derived(
    selectedCluster != null && subTab == null && subclusterIds.length > 0,
  );

  // Partition loaded plates into one group PER sub-cluster id. Built with a
  // Map (not a contiguity walk) so it is robust to a non-contiguous list —
  // e.g. the transient render right after opening a bucket, when `plates`
  // still holds the previous mixed-bucket gallery before the bucket's own
  // (subid-sorted) data arrives. A contiguity walk would emit the same subid
  // as multiple groups there, producing duplicate {#each} keys and a Svelte
  // each_key_duplicate crash that froze the detail view from opening.
  const groups = $derived.by(
    (): { key: string; label: string; items: RegionBrowseItem[] }[] => {
      if (!groupBySubid) return [{ key: '__all__', label: '', items: pager.items }];
      const byKey = new Map<string, RegionBrowseItem[]>();
      for (const p of pager.items) {
        const sub = p.region_cluster_subid ?? '__none__';
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
      toastStore.error(`Load plate clusters failed: ${(e as Error).message}`);
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
      toastStore.error(`Load suspected FPs failed: ${(e as Error).message}`);
    } finally {
      clusterBusy = false;
    }
  }

  async function runBuildFpCentroids(): Promise<void> {
    if (clusterBusy) return;
    clusterBusy = true;
    try {
      await buildRegionFpCentroids();
      toastStore.info('Building FP centroids… sub-typing the false-positive bucket.');
      while (true) {
        await new Promise((r) => setTimeout(r, 3000));
        const job = await getRegionFpCentroidStatus();
        if (job.running) continue;
        if (job.error) {
          toastStore.error(`Build FP centroids failed: ${job.error}`);
        } else if (job.result) {
          toastStore.success(
            `FP centroids built: ${job.result.n_members} members → ${job.result.k} sub-types.`,
          );
          await loadClusters();
        }
        break;
      }
    } catch (e) {
      toastStore.error(`Build FP centroids failed: ${(e as Error).message}`);
    } finally {
      clusterBusy = false;
    }
  }

  async function runClustering(): Promise<void> {
    if (clusterBusy) return;
    clusterBusy = true;
    try {
      // One-click pipeline (rebuild FP centroids → auto-pull tight FPs →
      // re-partition good plates) is a multi-minute background job, so we kick
      // it off and poll for completion instead of holding one request open.
      await clusterRegions(maxRank ?? undefined);
      toastStore.info(
        'Clustering plates… rebuilding FP centroids, pulling FPs, re-bucketing.',
      );
      while (true) {
        await new Promise((r) => setTimeout(r, 3000));
        const job = await getRegionClusterStatus();
        if (job.running) continue;
        if (job.error) {
          toastStore.error(`Cluster plates failed: ${job.error}`);
        } else if (job.result) {
          const r = job.result;
          const moved = r.auto_fp?.n_moved ?? 0;
          if (r.status === 'skipped_repartition_ttl') {
            toastStore.success(
              `Auto-moved ${moved} crop(s) to false positives. Good-plate re-partition skipped to preserve a refine from the last few minutes — re-run shortly to include it.`,
            );
          } else {
            toastStore.success(
              `Clustered ${r.n_regions ?? 0} plates into ${r.n_clusters ?? 0} buckets; auto-moved ${moved} to false positives.`,
            );
          }
          await loadClusters();
        }
        break;
      }
    } catch (e) {
      toastStore.error(`Cluster plates failed: ${(e as Error).message}`);
    } finally {
      clusterBusy = false;
    }
  }

  async function runRefineCluster(): Promise<void> {
    if (selectedCluster == null || clusterBusy) return;
    clusterBusy = true;
    refineMsg = `Refining bucket #${selectedCluster}…`;
    try {
      const res = await refineRegionCluster(selectedCluster);
      const n = res.n_subclusters ?? 0;
      if (n > 0) {
        refineMsg = `Split into ${n} sub-clusters — grouped below.`;
        toastStore.success(`Refine produced ${n} sub-clusters.`);
      } else {
        // Backend skipped it (too small / too large). Surface why.
        const reason = (res as { reason?: string }).reason ?? 'no sub-clusters found';
        refineMsg = `Not refined: ${reason}.`;
        toastStore.info(`Bucket not refined: ${reason}.`);
      }
      // Drop back to the "all" tab so the freshly grouped view shows, then
      // reload (server returns plates ordered by sub-cluster).
      subTab = null;
      await loadFirst();
    } catch (e) {
      refineMsg = `Refine failed: ${(e as Error).message}`;
      toastStore.error(`Refine failed: ${(e as Error).message}`);
    } finally {
      clusterBusy = false;
    }
  }

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

  /** M3: browse every plate with no `region_cluster_id` filter — the
   *  entry point for plates that aren't in any AHC bucket yet (or when
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
      // Leaving the suspected-FP view: reload the real plate gallery the
      // filter effect would otherwise have populated.
      suspectedFpView = false;
      void loadFirst();
    }
  }

  // Range selects span the currently displayed order, so hand the helper
  // the loaded plate ids on each click.
  function toggleSelect(p: RegionBrowseItem, e?: MouseEvent): void {
    sel.click(
      p.crop_id,
      e,
      pager.items.map((x) => x.crop_id),
    );
  }

  function selectAll(): void {
    sel.selectAll(pager.items.map((p) => p.crop_id));
  }

  async function openEditor(p: RegionBrowseItem): Promise<void> {
    try {
      editCrop = await getCrop(p.crop_id);
    } catch (err) {
      toastStore.error(`Could not load plate: ${(err as Error).message}`);
    }
  }

  async function applyStatus(cropIds: string[], status: string): Promise<void> {
    if (cropIds.length === 0 || busy) return;
    busy = true;
    sel.clear();
    try {
      // Callers only ever pass one of the profile's own state values
      // (PLATE_CONFIRM_STATE / PLATE_REJECT_STATE / PLATE_FALSE_POSITIVE_STATE);
      // the cast just satisfies batchRegionStatus's still-literal wire
      // type (that union is api.ts's Wave 2 concern, not this file's).
      // `region_verified` is not sent — the server derives it from
      // `region_status` and ignores the field when present.
      const res = await batchRegionStatus(
        licensePlateSlot,
        cropIds,
        status as 'detected' | 'no_region_visible' | 'verify_rejected' | 'false_positive',
      );
      // Render exactly what the server wrote. `items` covers every crop
      // actually updated; conflicted/invalid ids are left untouched here
      // and reported in the toast below.
      const byId = new Map(res.items.map((p) => [p.crop_id, p]));
      pager.items = pager.items.map((p) => byId.get(p.crop_id) ?? p);
      // M6: Z reverses this bulk status write server-side — the server's
      // own `items` list (not the request's `cropIds`) is what actually
      // got written, same "server's ids, not the request's" rule as
      // every other undoStore.record* call.
      undoStore.recordRegionWrites(res.items.map((p) => p.crop_id));
      const conflictCount = res.conflicts?.length ?? 0;
      const invalid = res.invalid ?? [];
      if (conflictCount > 0 || invalid.length > 0) {
        const parts = [
          conflictCount > 0 ? `${conflictCount} conflicted` : null,
          invalid.length > 0
            ? `${invalid.length} invalid (${invalid.map((i) => i.detail).join('; ')})`
            : null,
        ].filter((s): s is string => s != null);
        toastStore.error(
          `${status.replace('_', ' ')}: ${res.updated} updated, ${parts.join(', ')}`,
        );
      } else {
        toastStore.success(`${status.replace('_', ' ')}: ${res.updated} plate(s)`);
      }
    } catch (err) {
      toastStore.error(`Bulk update failed: ${(err as Error).message}`);
    } finally {
      busy = false;
    }
  }

  /**
   * `onsave` for `SlotBboxEditor` — the editor has ALREADY performed the
   * write via `setSlotBox` by the time this fires, and passes back the
   * server's own returned item. This function only patches the matching
   * card from that item; it must NOT re-PUT the box (a pre-C8 bug — this
   * used to call `setCropPlate` a second time here, redundantly
   * re-sending a box the editor had just saved), and must NOT re-derive
   * confirmed-vs-rejected status client-side — it renders what the
   * server wrote.
   */
  function saveBox(item: Crop): void {
    if (!editCrop) return;
    const cropId = item.id;
    toastStore.success('Plate saved');
    editCrop = null;
    const slotData = item.slots?.[licensePlateSlot.key];
    // Patch just this card in place rather than reloading page 1 (which
    // would wipe the list and reset scroll). The plate thumbnail is a
    // server-rendered URL, so bust its cache to pull the re-cropped box.
    pager.items = pager.items.map((p) =>
      p.crop_id === cropId
        ? {
            ...p,
            region_status: slotData?.lifecycle?.status ?? p.region_status,
            region_verified: slotData?.lifecycle?.verified ?? p.region_verified,
            region_bbox_norm: slotData?.subBox?.rawXyxy ?? null,
            region_bbox_frame: slotData?.subBox?.frame ?? p.region_bbox_frame,
            region_thumbnail_url: getRegionThumbUrl(cropId, 160, Date.now()),
          }
        : p,
    );
    // M6: the editor's write (setSlotBox, already completed by the time
    // this fires — see the doc comment above) is undoable via Z too.
    undoStore.recordRegionWrites([cropId]);
  }

  /**
   * M6: Z on the plate gallery reverses the most recent region write
   * (single-item bbox edit or bulk status change), the same way Z
   * reverses a label write on the card-grid `/clusters` view — see
   * `undo.svelte.ts`'s header comment for why this shares the ONE
   * `undoStore` stack (kind: `'region'`) with every other undo entry
   * rather than its own gallery-local stack.
   */
  function mergeUndoneItems(crops: Crop[]): void {
    if (crops.length === 0) return;
    const byId = new Map(crops.map((c) => [c.id, c]));
    pager.items = pager.items.map((p) => {
      const restored = byId.get(p.crop_id);
      if (!restored) return p;
      const slotData = restored.slots?.[licensePlateSlot.key];
      return {
        ...p,
        region_status: slotData?.lifecycle?.status ?? p.region_status,
        region_verified: slotData?.lifecycle?.verified ?? p.region_verified,
        region_bbox_norm: slotData?.subBox?.rawXyxy ?? null,
        region_bbox_frame: slotData?.subBox?.frame ?? p.region_bbox_frame,
        region_thumbnail_url: getRegionThumbUrl(p.crop_id, 160, Date.now()),
      };
    });
  }

  async function undoLastAction(): Promise<void> {
    const crops = await undoStore.undoLast();
    mergeUndoneItems(crops);
  }

  return {
    get pager() {
      return pager;
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
    runBuildFpCentroids,
    runClustering,
    runRefineCluster,
    openCluster,
    openAll,
    selectSubTab,
    backToClusters,
    toggleSelect,
    selectAll,
    openEditor,
    applyStatus,
    saveBox,
    undoLastAction,
  };
}

export type SlotGalleryController = ReturnType<typeof createSlotGalleryController>;
