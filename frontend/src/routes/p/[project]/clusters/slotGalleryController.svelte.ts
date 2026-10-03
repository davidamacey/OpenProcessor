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

import {
  batchRegionStatus,
  buildRegionFpCentroids,
  clusterRegions,
  getCrop,
  getRegionClusters,
  getRegionClusterStatus,
  getRegionFpCentroidStatus,
  getRegions,
  getSuspectedFalsePositives,
  postBatchBoxState,
  refineRegionCluster,
  type RegionBrowseItem,
  type SuspectedFpItem,
} from '$lib/api';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Cluster, Crop } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import type { SlotSpec } from '$lib/annotations/types';
import { rowBoxOf } from '$lib/annotations/rowBox';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

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

  // W8.7: per-box `state` triage, a different vocabulary from the
  // item-level region_status above (`accepted`/`rejected`/
  // `false_positive`/`proposed`, keyed here by the served `box_states`
  // entry's `role`). Falls back to the literal role string — the fixed
  // W8.7 wire vocabulary, not a client guess — for a pre-W8 backend that
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
    keyOf: (p) => `${p.crop_id}:${p.region_box_id ?? ''}`,
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
      toastStore.error(
        `Load ${slot.label.singular} clusters failed: ${(e as Error).message}`,
      );
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
      // re-partition good regions) is a multi-minute background job, so we kick
      // it off and poll for completion instead of holding one request open.
      await clusterRegions(maxRank ?? undefined);
      toastStore.info(
        `Clustering ${slot.label.plural}… rebuilding FP centroids, pulling FPs, re-bucketing.`,
      );
      while (true) {
        await new Promise((r) => setTimeout(r, 3000));
        const job = await getRegionClusterStatus();
        if (job.running) continue;
        if (job.error) {
          toastStore.error(`Cluster ${slot.label.plural} failed: ${job.error}`);
        } else if (job.result) {
          const r = job.result;
          const moved = r.auto_fp?.n_moved ?? 0;
          if (r.status === 'skipped_repartition_ttl') {
            toastStore.success(
              `Auto-moved ${moved} crop(s) to false positives. Good-${slot.label.singular} re-partition skipped to preserve a refine from the last few minutes — re-run shortly to include it.`,
            );
          } else {
            toastStore.success(
              `Clustered ${r.n_regions ?? 0} ${slot.label.plural} into ${r.n_clusters ?? 0} buckets; auto-moved ${moved} to false positives.`,
            );
          }
          await loadClusters();
        }
        break;
      }
    } catch (e) {
      toastStore.error(`Cluster ${slot.label.plural} failed: ${(e as Error).message}`);
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
      // reload (server returns regions ordered by sub-cluster).
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
      toastStore.error(
        `Could not load ${slot.label.singular}: ${(err as Error).message}`,
      );
    }
  }

  async function applyStatus(
    cropIds: string[],
    status: string | undefined,
  ): Promise<void> {
    if (cropIds.length === 0 || busy || !status) return;
    busy = true;
    sel.clear();
    try {
      // The server validates `status` against its own /regions/statuses.
      // `region_verified` is not sent — the server derives it from
      // `region_status` and ignores the field when present.
      const res = await batchRegionStatus(slot, cropIds, status);
      // Render exactly what the server wrote. `items` covers every crop
      // actually updated; conflicted/invalid ids are left untouched here
      // and reported in the toast below.
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this computation, never stored in reactive state
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
        // W8: a conflict now carries a served `message` (RegionBatchConflict)
        // on a W8 backend — show it verbatim instead of just a count when
        // present, since it names the real reason ("The item changed
        // since it was loaded."). A pre-W8 backend has no `message`, so
        // this falls back to the bare count exactly as before.
        const conflictDetail = res.conflicts?.find((c) => c.message)?.message;
        const parts = [
          conflictCount > 0
            ? `${conflictCount} conflicted${conflictDetail ? ` (${conflictDetail})` : ''}`
            : null,
          invalid.length > 0
            ? `${invalid.length} invalid (${invalid.map((i) => i.detail).join('; ')})`
            : null,
        ].filter((s): s is string => s != null);
        toastStore.error(
          `${status.replace('_', ' ')}: ${res.updated} updated, ${parts.join(', ')}`,
        );
      } else {
        toastStore.success(
          `${status.replace('_', ' ')}: ${res.updated} ${slot.label.singular}(s)`,
        );
      }
    } catch (err) {
      toastStore.error(`Bulk update failed: ${(err as Error).message}`);
    } finally {
      busy = false;
    }
  }

  /**
   * W8: per-box triage over the selected rows (region gallery triage — a
   * cluster is a set of boxes, W8.8/§7.7). Unlike `applyStatus` above
   * (whole-item `region_status`, `PATCH region_meta` / `POST
   * batch_status`), this goes through `POST /regions/batch_box_state`,
   * which flips only the targeted box on each item, never its siblings —
   * exactly the spec's rule for triage from a cluster ("never the
   * item-level batch_status, which would flip every sibling box").
   * Targets are built from the row's own served `region_box_id`
   * (`RegionBrowseItem`, W8.10) — a pre-W8 row without one is skipped
   * rather than silently flipping a whole item.
   */
  async function applyBoxState(cropIds: string[], state: string): Promise<void> {
    if (cropIds.length === 0 || busy) return;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this computation, never stored in reactive state
    const idSet = new Set(cropIds);
    const targets = pager.items
      .filter((p) => idSet.has(p.crop_id) && p.region_box_id != null)
      .map((p) => ({ cropId: p.crop_id, boxId: p.region_box_id! }));
    const skipped = cropIds.length - targets.length;
    if (targets.length === 0) {
      toastStore.error(
        `Cannot triage by box: this row has no served box id (pre-W8 backend?).`,
      );
      return;
    }
    busy = true;
    sel.clear();
    try {
      const res = await postBatchBoxState(targets, state);
      // A box-state write returns full items (Crop), not gallery rows —
      // the row grid re-fetches its own page rather than patching rows
      // in place from a shape the gallery doesn't render (RegionBrowseItem
      // vs Crop): the triaged boxes typically leave the current bucket
      // anyway (their state changed), so a refetch is the correct result,
      // not a shortcut.
      undoStore.recordRegionWrites(res.items.map((c) => c.id));
      const invalid = res.invalid ?? [];
      const conflicts = res.conflicts ?? [];
      if (invalid.length > 0 || conflicts.length > 0) {
        const conflictDetail = conflicts.find((c) => c.message)?.message;
        const parts = [
          conflicts.length > 0
            ? `${conflicts.length} conflicted${conflictDetail ? ` (${conflictDetail})` : ''}`
            : null,
          invalid.length > 0
            ? `${invalid.length} invalid (${invalid.map((i) => i.message).join('; ')})`
            : null,
          skipped > 0 ? `${skipped} skipped (no box id)` : null,
        ].filter((s): s is string => s != null);
        toastStore.error(
          `${state.replace('_', ' ')}: ${res.updated} updated, ${parts.join(', ')}`,
        );
      } else {
        toastStore.success(
          `${state.replace('_', ' ')}: ${res.updated} box(es)${skipped > 0 ? `, ${skipped} skipped (no box id)` : ''}`,
        );
      }
      await pager.loadPage(pager.firstPage);
    } catch (err) {
      toastStore.error(`Bulk box triage failed: ${(err as Error).message}`);
    } finally {
      busy = false;
    }
  }

  /**
   * `onsave` for `SlotBboxEditor` — the editor has ALREADY performed the
   * write (`PUT /crops/{id}/regions`) by the time this fires, and passes back the
   * server's own returned item. This function only patches the matching
   * card from that item; it must NOT re-PUT the box (a pre-C8 bug — this
   * used to write the box a second time here, redundantly re-sending a
   * box the editor had just saved), and must NOT re-derive
   * confirmed-vs-rejected status client-side — it renders what the
   * server wrote.
   */
  function saveBox(item: Crop): void {
    if (!editCrop) return;
    const cropId = item.id;
    toastStore.success(`${slot.label.title} saved`);
    editCrop = null;
    const slotData = item.slots?.[slot.key];
    // Patch just this card in place rather than reloading page 1 (which
    // would wipe the list and reset scroll). The returned item's slot data
    // carries the new boxes and revision, which is also what re-crops the
    // card's thumbnail.
    pager.items = pager.items.map((p) =>
      p.crop_id === cropId
        ? {
            ...p,
            region_status: slotData?.lifecycle?.status ?? p.region_status,
            region_verified: slotData?.lifecycle?.verified ?? p.region_verified,
            slots: { ...p.slots, ...item.slots },
          }
        : p,
    );
    // M6: the editor's write (already completed by the time this fires —
    // see the doc comment above) is undoable via Z too.
    undoStore.recordRegionWrites([cropId]);
  }

  /**
   * M6: Z on the region gallery reverses the most recent region write
   * (single-item bbox edit or bulk status change), the same way Z
   * reverses a label write on the card-grid `/clusters` view — see
   * `undo.svelte.ts`'s header comment for why this shares the ONE
   * `undoStore` stack (kind: `'region'`) with every other undo entry
   * rather than its own gallery-local stack.
   */
  function mergeUndoneItems(crops: Crop[]): void {
    if (crops.length === 0) return;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local, built and consumed synchronously within this computation, never stored in reactive state
    const byId = new Map(crops.map((c) => [c.id, c]));
    pager.items = pager.items.map((p) => {
      const restored = byId.get(p.crop_id);
      if (!restored) return p;
      const slotData = restored.slots?.[slot.key];
      return {
        ...p,
        region_status: slotData?.lifecycle?.status ?? p.region_status,
        region_verified: slotData?.lifecycle?.verified ?? p.region_verified,
        slots: { ...p.slots, ...restored.slots },
      };
    });
  }

  /**
   * `onreprocessed` for a card's image Reprocess: the served items are
   * merged into the cards already loaded (later pages included), then the
   * first page is re-fetched, since re-detection can add, move or drop the
   * regions an image contributes.
   */
  async function adoptReprocessed(items: Crop[]): Promise<void> {
    mergeUndoneItems(items);
    await pager.loadPage(pager.firstPage);
  }

  async function undoLastAction(): Promise<void> {
    const crops = await undoStore.undoLast();
    mergeUndoneItems(crops);
  }

  return {
    adoptReprocessed,
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
    applyBoxState,
    saveBox,
    undoLastAction,
  };
}

export type SlotGalleryController = ReturnType<typeof createSlotGalleryController>;
