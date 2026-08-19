<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import {
    batchPlateStatus,
    buildPlateFpCentroids,
    clusterPlates,
    getClusters,
    getCrop,
    getPlateClusters,
    getPlateClusterStatus,
    getPlateFpCentroidStatus,
    getPlates,
    getSuspectedFalsePositives,
    getThumbUrl,
    refinePlateCluster,
    setCropPlate,
    type PlateBrowseItem,
    type SuspectedFpItem,
  } from '$lib/api';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import { bboxNormToXYXY } from '$lib/plate_geometry';
  import { createPager } from '$lib/pager.svelte';
  import { createSelection } from '$lib/selection.svelte';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import PlateCard from '$lib/components/PlateCard.svelte';
  import PlateEditor from '$lib/components/PlateEditor.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import type { ClusterFilter, OpCluster, OpCrop } from '$lib/types';
  import { toastStore } from '$stores/toast.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  // Cluster grid pager. One params builder (clusterQuery) feeds page 1 and
  // every later page, so a filter can't be sent on the first request and
  // silently dropped on the next.
  const clusterPager = createPager<OpCluster>({
    fetchPage: async (page) => await getClusters(clusterQuery(page)),
    keyOf: (c) => String(c.id),
  });

  // Synthetic license_plate gallery card. Plates are sub-bboxes on
  // vehicle crops, not FAISS docs, so the cluster grid never produces
  // a card for them. We surface one explicitly using /curation/plates so the
  // operator can click into the plate inventory the same way they click
  // into any other class cluster. Card is null until the first /curation/plates
  // call resolves; the cluster grid hides it during that window.
  let lpCard = $state<OpCluster | null>(null);

  // Persist the cluster-list filter (sort + unlabeled-only) across
  // navigation so going into a cluster and back keeps the operator's
  // last view — they shouldn't have to re-click "Unlabeled only" every
  // time. sessionStorage survives back-nav + refresh within the session
  // regardless of how the user returns (back button, link, etc.).
  const FILTER_PERSIST_KEY = 'op_clusters_filter_v1';
  function loadPersistedFilter(): { sort?: string; unlabeledOnly?: boolean } | null {
    if (typeof sessionStorage === 'undefined') return null;
    try {
      return JSON.parse(sessionStorage.getItem(FILTER_PERSIST_KEY) ?? 'null');
    } catch {
      return null;
    }
  }
  const _persistedFilter = loadPersistedFilter();

  let sort = $state<NonNullable<ClusterFilter['sort']>>(
    (_persistedFilter?.sort as NonNullable<ClusterFilter['sort']>) ?? 'purity_asc',
  );
  let unlabeledOnly = $state<boolean>(_persistedFilter?.unlabeledOnly ?? false);
  const pageSize = 24;

  // Primary-subject grid filters: card stats reflect only crops that pass.
  // subjectScope 0=all, 1=largest, 2=largest+2nd → max_rank. Clarity slider
  // commits on release. Lets the operator scope the grid to the largest,
  // clear crops (incl. the review-tab blind-spot cohorts) for drag-drop +
  // AHC refine.
  let subjectScope = $state<0 | 1 | 2>(0);
  const maxRank = $derived<number | null>(subjectScope === 0 ? null : subjectScope);
  const BLUR_MAX = 2;
  let blurSlider = $state<number>(0);
  let minBlurRatio = $state<number | null>(null);
  function commitClusterBlur(): void {
    minBlurRatio = blurSlider > 0 ? blurSlider : null;
  }

  // Write the filter back whenever it changes. Catches every mutation
  // site (toggle button, sort dropdown) without per-handler bookkeeping.
  $effect(() => {
    if (typeof sessionStorage === 'undefined') return;
    sessionStorage.setItem(
      FILTER_PERSIST_KEY,
      JSON.stringify({ sort, unlabeledOnly }),
    );
  });

  // --- Plate browse (replaces the "License plates aren't clustered" placeholder
  //     when the operator selects the license_plate class filter).
  const PLATES_PAGE_SIZE = 60;
  const platePager = createPager<PlateBrowseItem>({
    fetchPage: async (page) => await getPlates(_plateQuery(page)),
    keyOf: (p) => p.crop_id,
  });

  // Plate triage: multi-select for bulk actions + the inline PlateEditor.
  // Plain click TOGGLES here (accumulating), unlike the crop grid where
  // it replaces — plate triage is a bulk-marking flow.
  const plateSel = createSelection({ plainClick: 'toggle' });
  let editPlateCrop = $state<OpCrop | null>(null);
  let plateBusy = $state<boolean>(false);

  // Top-N largest-crop gate for plates. The sort is built on the largest
  // 1-3 crops, so this is the key filter for finding the plates that matter.
  // null = all ranks.
  let plateMaxRank = $state<number | null>(null);

  // Plate clustering (AHC-refinable buckets over plate_pe_embedding).
  // selectedPlateCluster narrows the gallery to one bucket; null shows the
  // bucket grid (or the flat gallery when no clustering has run).
  let plateClusters = $state<OpCluster[]>([]);
  let selectedPlateCluster = $state<number | null>(null);
  let plateClusterBusy = $state<boolean>(false);
  // Mirrors FALSE_POSITIVE_PLATE_CLUSTER_ID in the API (op_clustering.py).
  const FP_PLATE_CLUSTER_ID = -100;

  // Sub-cluster delineation inside an open plate bucket — mirrors the vehicle
  // cluster detail. null = "all" (the server returns plates ordered by subid so
  // AHC groups are contiguous; we render a labeled separator before each).
  // Selecting a chip filters the gallery to that one sub-cluster.
  let plateSubTab = $state<string | null>(null);
  // Last refine outcome, shown inline next to the button so the result is not
  // just a transient toast (the run is fast and easy to miss).
  let plateRefineMsg = $state<string | null>(null);

  // Distinct sub-cluster ids present in the loaded plates, sorted lexically so
  // "9a","9aa","9ab"… land in human-expected order.
  const plateSubclusterIds = $derived.by(() => {
    const set = new Set<string>();
    for (const p of platePager.items) if (p.plate_cluster_subid) set.add(p.plate_cluster_subid);
    return [...set].sort();
  });

  // Per-subid counts for the separator-header labels ('__none__' = unrefined).
  const plateSubCounts = $derived.by(() => {
    const m = new Map<string, number>();
    for (const p of platePager.items) {
      const k = p.plate_cluster_subid ?? '__none__';
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
  });

  // Group only when a bucket is open, on the "all" tab, and refine has produced
  // sub-clusters. Otherwise render one flat group (no separators).
  const groupPlatesBySubid = $derived(
    selectedPlateCluster != null && plateSubTab == null && plateSubclusterIds.length > 0,
  );

  // Partition loaded plates into one group PER sub-cluster id. Built with a
  // Map (not a contiguity walk) so it is robust to a non-contiguous list —
  // e.g. the transient render right after opening a bucket, when `plates`
  // still holds the previous mixed-bucket gallery before the bucket's own
  // (subid-sorted) data arrives. A contiguity walk would emit the same subid
  // as multiple groups there, producing duplicate {#each} keys and a Svelte
  // each_key_duplicate crash that froze the detail view from opening.
  const plateGroups = $derived.by((): { key: string; label: string; items: PlateBrowseItem[] }[] => {
    if (!groupPlatesBySubid) return [{ key: '__all__', label: '', items: platePager.items }];
    const byKey = new Map<string, PlateBrowseItem[]>();
    for (const p of platePager.items) {
      const sub = p.plate_cluster_subid ?? '__none__';
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
  });

  async function loadPlateClusters(): Promise<void> {
    try {
      const res = await getPlateClusters({
        maxClusters: 500,
        maxRank: plateMaxRank ?? undefined,
      });
      // Defensive: never render empty buckets (the backend already omits
      // them, but a stale response shouldn't surface a 0-size card).
      plateClusters = (res.clusters ?? []).filter((c) => c.size > 0);
    } catch (e) {
      toastStore.error(`Load plate clusters failed: ${(e as Error).message}`);
    }
  }

  // Suspected-FP review: crops the FP centroids flag as likely false
  // positives. Loads into the same `plates` array so the beloved
  // shift-select + Mark-FP workflow works unchanged; platesTotal is pinned to
  // the loaded count so the infinite-scroll sentinel never pages in normal
  // plates over the top.
  let suspectedFpView = $state<boolean>(false);
  let suspectedFpThreshold = $state<number>(0.35);

  async function loadSuspectedFp(): Promise<void> {
    if (plateClusterBusy) return;
    plateClusterBusy = true;
    try {
      const res = await getSuspectedFalsePositives({
        threshold: suspectedFpThreshold,
        pageSize: 200,
      });
      selectedPlateCluster = null;
      plateSel.clear();
      platePager.items = res.items as SuspectedFpItem[];
      platePager.total = res.items.length;
      suspectedFpView = true;
      if (!res.centroids_built) {
        toastStore.info(res.message ?? 'No FP centroids yet — build them first.');
      } else {
        toastStore.success(`${res.total} suspected false positive(s) at ≤ ${suspectedFpThreshold}.`);
      }
    } catch (e) {
      toastStore.error(`Load suspected FPs failed: ${(e as Error).message}`);
    } finally {
      plateClusterBusy = false;
    }
  }

  async function runBuildFpCentroids(): Promise<void> {
    if (plateClusterBusy) return;
    plateClusterBusy = true;
    try {
      await buildPlateFpCentroids();
      toastStore.info('Building FP centroids… sub-typing the false-positive bucket.');
      while (true) {
        await new Promise((r) => setTimeout(r, 3000));
        const job = await getPlateFpCentroidStatus();
        if (job.running) continue;
        if (job.error) {
          toastStore.error(`Build FP centroids failed: ${job.error}`);
        } else if (job.result) {
          toastStore.success(
            `FP centroids built: ${job.result.n_members} members → ${job.result.k} sub-types.`,
          );
          await loadPlateClusters();
        }
        break;
      }
    } catch (e) {
      toastStore.error(`Build FP centroids failed: ${(e as Error).message}`);
    } finally {
      plateClusterBusy = false;
    }
  }

  async function runClusterPlates(): Promise<void> {
    if (plateClusterBusy) return;
    plateClusterBusy = true;
    try {
      // One-click pipeline (rebuild FP centroids → auto-pull tight FPs →
      // re-partition good plates) is a multi-minute background job, so we kick
      // it off and poll for completion instead of holding one request open.
      await clusterPlates(plateMaxRank ?? undefined);
      toastStore.info('Clustering plates… rebuilding FP centroids, pulling FPs, re-bucketing.');
      while (true) {
        await new Promise((r) => setTimeout(r, 3000));
        const job = await getPlateClusterStatus();
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
              `Clustered ${r.n_plates ?? 0} plates into ${r.n_clusters ?? 0} buckets; auto-moved ${moved} to false positives.`,
            );
          }
          await loadPlateClusters();
        }
        break;
      }
    } catch (e) {
      toastStore.error(`Cluster plates failed: ${(e as Error).message}`);
    } finally {
      plateClusterBusy = false;
    }
  }

  async function runRefinePlateCluster(): Promise<void> {
    if (selectedPlateCluster == null || plateClusterBusy) return;
    plateClusterBusy = true;
    plateRefineMsg = `Refining bucket #${selectedPlateCluster}…`;
    try {
      const res = await refinePlateCluster(selectedPlateCluster);
      const n = res.n_subclusters ?? 0;
      if (n > 0) {
        plateRefineMsg = `Split into ${n} sub-clusters — grouped below.`;
        toastStore.success(`Refine produced ${n} sub-clusters.`);
      } else {
        // Backend skipped it (too small / too large). Surface why.
        const reason = (res as { reason?: string }).reason ?? 'no sub-clusters found';
        plateRefineMsg = `Not refined: ${reason}.`;
        toastStore.info(`Bucket not refined: ${reason}.`);
      }
      // Drop back to the "all" tab so the freshly grouped view shows, then
      // reload (server returns plates ordered by sub-cluster).
      plateSubTab = null;
      await loadPlatesFirst();
    } catch (e) {
      plateRefineMsg = `Refine failed: ${(e as Error).message}`;
      toastStore.error(`Refine failed: ${(e as Error).message}`);
    } finally {
      plateClusterBusy = false;
    }
  }

  function openPlateCluster(id: number): void {
    suspectedFpView = false;
    plateSubTab = null;
    plateRefineMsg = null;
    // Clear the previous gallery synchronously so the render between selecting
    // the bucket and its data arriving doesn't group a stale mixed-bucket list.
    platePager.items = [];
    selectedPlateCluster = id;
  }

  function selectPlateSubTab(sub: string | null): void {
    if (plateSubTab === sub) return;
    plateSubTab = sub;
    void loadPlatesFirst();
  }

  function backToPlateClusters(): void {
    selectedPlateCluster = null;
    plateSubTab = null;
    plateRefineMsg = null;
    plateSel.clear();
    if (suspectedFpView) {
      // Leaving the suspected-FP view: reload the real plate gallery the
      // filter effect would otherwise have populated.
      suspectedFpView = false;
      void loadPlatesFirst();
    }
  }

  // Range selects span the currently displayed order, so hand the helper
  // the loaded plate ids on each click.
  function togglePlateSelect(p: PlateBrowseItem, e?: MouseEvent): void {
    plateSel.click(
      p.crop_id,
      e,
      platePager.items.map((x) => x.crop_id),
    );
  }

  function selectAllPlates(): void {
    plateSel.selectAll(platePager.items.map((p) => p.crop_id));
  }

  async function openPlateEditor(p: PlateBrowseItem): Promise<void> {
    try {
      editPlateCrop = await getCrop(p.crop_id);
    } catch (err) {
      toastStore.error(`Could not load plate: ${(err as Error).message}`);
    }
  }

  async function applyPlateStatus(
    cropIds: string[],
    status: 'false_positive' | 'no_plate_visible' | 'detected',
  ): Promise<void> {
    if (cropIds.length === 0 || plateBusy) return;
    plateBusy = true;
    // Snapshot for rollback, then update the affected cards IN PLACE. The
    // grid's #each is keyed by crop_id, so patching the array (rather than
    // reloading page 1) reuses the existing DOM nodes and preserves scroll
    // position — critical when the operator is deep in a 15k-item gallery.
    const snap = platePager.items;
    const snapById = new Map(snap.map((p) => [p.crop_id, p]));
    const idSet = new Set(cropIds);
    const verified = status === 'detected' ? true : undefined;
    // Mirror the backend write contract (batch_set_plate_status): a human
    // status change is terminal, so it also flips plate_validated=true. Keep
    // the optimistic patch identical to what OpenSearch persists so the card
    // never diverges from authoritative state.
    const patch = (p: PlateBrowseItem): PlateBrowseItem => ({
      ...p,
      plate_status: status,
      plate_verified: verified ?? p.plate_verified,
      plate_validated: true,
    });
    platePager.items = platePager.items.map((p) => (idSet.has(p.crop_id) ? patch(p) : p));
    plateSel.clear();
    try {
      const res = await batchPlateStatus(cropIds, status, {
        plateVerified: verified,
      });
      // Reconcile with the backend: any crop_id the server reported as a
      // conflict was NOT written, so revert just those cards to their
      // pre-edit state rather than leaving a falsely-applied status.
      const conflictIds = new Set((res.conflicts ?? []).map((c) => c.crop_id));
      if (conflictIds.size > 0) {
        platePager.items = platePager.items.map((p) =>
          conflictIds.has(p.crop_id) ? (snapById.get(p.crop_id) ?? p) : p,
        );
        toastStore.error(
          `${status.replace('_', ' ')}: ${res.updated} updated, ${conflictIds.size} conflicted (reverted)`,
        );
      } else {
        toastStore.success(`${status.replace('_', ' ')}: ${res.updated} plate(s)`);
      }
    } catch (err) {
      platePager.items = snap;
      toastStore.error(`Bulk update failed: ${(err as Error).message}`);
    } finally {
      plateBusy = false;
    }
  }

  // Filter sidebar state — only active on the plates view.
  let plateDetectorFilter = $state<string>('');
  let plateVerifiedOnly = $state<boolean>(false);
  let plateMinScore = $state<number>(0);
  let plateTextQuery = $state<string>('');

  const classFilter = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    return v == null ? null : Number.isFinite(+v) ? +v : null;
  });

  // The legacy ensemble stores plates as a *sub-bbox* on each vehicle
  // crop (`plate_bbox_norm`), NOT as standalone docs in the cluster
  // index. So filtering this page by the `license_plate` class always
  // returns 0 / unlabeled clusters — confusing operators who expect to
  // see plate clusters here. Detect that case and route the user to the
  // plates review queue instead, which is the actual home for plate
  // labeling.
  const isLicensePlateFilter = $derived.by<boolean>(() => {
    if (classFilter == null) return false;
    const cls = classesStore.classes.find((c) => c.id === classFilter);
    return (cls?.name ?? '').toLowerCase() === 'license_plate';
  });

  // One params builder for both pages of the cluster grid. loadMore used
  // to omit max_rank / min_blur_ratio, so scrolling past page 1 appended
  // unfiltered clusters over a filtered page 1.
  function clusterQuery(page: number): ClusterFilter {
    return {
      class_id: classFilter ?? undefined,
      sort,
      page,
      page_size: pageSize,
      max_rank: maxRank,
      min_blur_ratio: minBlurRatio,
    };
  }

  async function loadFirst(): Promise<void> {
    await clusterPager.loadFirst();
    if (clusterPager.error == null) await loadLicensePlateCard();
  }

  // Build the synthetic license_plate gallery card. Plates live as
  // sub-bboxes on vehicle crops (not FAISS docs) so the cluster grid
  // never includes them. We query /curation/plates for the total inventory
  // and use the first 4 plate-bearing crops as thumbnails. Card is
  // null until this resolves; the grid renders it as the first item
  // when the unfiltered view is active.
  async function loadLicensePlateCard(): Promise<void> {
    const lp = classesStore.classes.find(
      (c) => (c.name ?? '').toLowerCase() === 'license_plate',
    );
    if (!lp) {
      lpCard = null;
      return;
    }
    try {
      // Pull a slightly larger window than 4 so we can drop items
      // missing a plate sub-bbox without falling below the tile count.
      const res = await getPlates({ page: 1, page_size: 12 });
      const withPlateBox = res.items.filter(
        (p) => Array.isArray(p.plate_bbox_norm) && p.plate_bbox_norm.length === 4,
      );
      const reps = withPlateBox.slice(0, 4);
      lpCard = {
        id: lp.id,
        size: res.total,
        // Purity badge is meaningless for a non-cluster — leave null.
        purity: null,
        dominant_class_id: lp.id,
        dominant_class_name: 'license_plate',
        dominant_pct: null,
        sub_clusters: 0,
        has_subclusters: false,
        representative_crop_ids: reps.map((p) => p.crop_id),
        // Show plate close-ups, not vehicle thumbnails — the whole
        // point of this card is that the operator is browsing plates.
        // /curation/crops/{id}/plate_thumbnail returns the plate sub-bbox
        // rendered to a 160px tile.
        representative_thumb_urls: reps.map(
          (p) => `/curation/crops/${encodeURIComponent(p.crop_id)}/plate_thumbnail?size=160`,
        ),
        updated_at: null,
      } as OpCluster;
    } catch {
      lpCard = null;
    }
  }

  // Items rendered in the unfiltered cluster grid: synthetic LP card
  // prepended (when present) so the operator always has a visible
  // entry point to the plate inventory. With class filter active we
  // hand the user to the dedicated plate-browse branch already, so
  // skip the prepend there.
  // gridItems = (synthetic license_plate card if unfiltered) + clusters,
  // optionally narrowed to only the "Unlabeled" group when the toggle is on.
  // "Unlabeled" = cluster_kind !== 'class', i.e. the candidate (IVF/AHC)
  // and unassigned buckets the operator still needs to sort. Keying on
  // cluster_kind (not dominant_class_name) is the fix for "only 16
  // showed": candidate clusters dominated by gemma_unmatched crops DO
  // carry a dominant_class_name, so the old !dominant_class_name test
  // wrongly excluded them.
  // Sort the loaded clusters client-side. The /curation/clusters endpoint only
  // returns size-descending (it's a terms agg, not a sortable query), and
  // every cluster comes back in one call — so sorting here is both
  // correct and complete. Without this the sort dropdown did nothing.
  function sortClusters(list: OpCluster[], mode: typeof sort): OpCluster[] {
    const out = [...list];
    const purity = (c: OpCluster) => (c.purity == null ? Number.POSITIVE_INFINITY : c.purity);
    switch (mode) {
      case 'size_desc':
        out.sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
        break;
      case 'size_asc':
        out.sort((a, b) => (a.size ?? 0) - (b.size ?? 0));
        break;
      case 'purity_desc':
        // null purity (no labelled members) sorts last on a desc view too.
        out.sort((a, b) => {
          const pa = a.purity ?? -1;
          const pb = b.purity ?? -1;
          return pb - pa;
        });
        break;
      case 'purity_asc':
        out.sort((a, b) => purity(a) - purity(b));
        break;
      case 'dominant_class':
        out.sort((a, b) =>
          (a.dominant_class_name ?? '￿').localeCompare(b.dominant_class_name ?? '￿'),
        );
        break;
    }
    return out;
  }

  const gridItems = $derived.by<OpCluster[]>(() => {
    const filtered = unlabeledOnly
      ? clusterPager.items.filter((c) => c.cluster_kind !== 'class')
      : clusterPager.items;
    const sorted = sortClusters(filtered, sort);
    // Keep the synthetic license_plate card pinned first (entry point to
    // the plate inventory), unaffected by sort, only on the unfiltered
    // labelled view.
    return classFilter == null && lpCard != null && !unlabeledOnly
      ? [lpCard, ...sorted]
      : sorted;
  });
  const unlabeledCount = $derived(
    clusterPager.items.filter((c) => c.cluster_kind !== 'class').length,
  );

  const loadMore = () => clusterPager.loadMore();

  $effect(() => {
    keyboardStore.setScope('clusters');
  });

  // Re-load on class-filter change — but only for the cluster view.
  // Sort is applied client-side (sortClusters) over the single loaded
  // batch, so changing it must NOT refetch (the endpoint returns the
  // same size-ordered data regardless).
  $effect(() => {
    void classFilter;
    void maxRank;
    void minBlurRatio;
    if (!isLicensePlateFilter) void loadFirst();
  });

  function _plateQuery(page: number): import('$lib/api').PlatesQuery {
    return {
      page,
      page_size: PLATES_PAGE_SIZE,
      detector: plateDetectorFilter || undefined,
      verified: plateVerifiedOnly || undefined,
      min_score: plateMinScore > 0 ? plateMinScore : undefined,
      text: plateTextQuery || undefined,
      max_rank: plateMaxRank ?? undefined,
      plate_cluster_id: selectedPlateCluster ?? undefined,
      // When a single sub-cluster tab is active, filter to it; otherwise (the
      // "all" tab) ask the server to order by sub-cluster so AHC groups come
      // back contiguous across pages and we can render them with separators.
      plate_cluster_subid: plateSubTab ?? undefined,
      sort_by_subid: selectedPlateCluster != null && plateSubTab == null ? true : undefined,
    };
  }

  const loadPlatesFirst = () => platePager.loadFirst();
  const loadPlatesMore = () => platePager.loadMore();

  // Re-load plates whenever a filter, the top-N rank gate, or the selected
  // plate cluster changes. When no cluster is selected, also refresh the
  // cluster-card grid so it reflects the current rank gate.
  $effect(() => {
    void plateDetectorFilter;
    void plateVerifiedOnly;
    void plateMinScore;
    void plateTextQuery;
    void plateMaxRank;
    void selectedPlateCluster;
    if (isLicensePlateFilter) {
      void loadPlatesFirst();
      if (selectedPlateCluster == null) void loadPlateClusters();
    }
  });

  async function savePlateBbox(plateBboxSrc: import('$lib/types').BBoxNorm | null): Promise<void> {
    if (!editPlateCrop) return;
    const cropId = editPlateCrop.id;
    // PlateEditor yields a source-frame BBoxNorm {cx,cy,w,h}; the API takes
    // [x1,y1,x2,y2]. null clears the box (→ no_plate_visible server-side).
    const arr = plateBboxSrc ? bboxNormToXYXY(plateBboxSrc) : null;
    try {
      const res = await setCropPlate(cropId, arr as [number, number, number, number] | null);
      toastStore.success('Plate saved');
      editPlateCrop = null;
      // Patch just this card in place rather than reloading page 1 (which
      // would wipe the list and reset scroll). Use the authoritative
      // plate_status the backend returned (clearing the box → no_plate_visible
      // server-side) instead of guessing. The plate thumbnail is a
      // server-rendered URL, so bust its cache to pull the re-cropped box.
      platePager.items = platePager.items.map((p) =>
        p.crop_id === cropId
          ? {
              ...p,
              plate_status: res.plate_status ?? p.plate_status,
              plate_bbox_norm: arr ?? null,
              plate_thumbnail_url: `/curation/crops/${cropId}/plate_thumbnail?v=${Date.now()}`,
            }
          : p,
      );
    } catch (err) {
      toastStore.error(`Save failed: ${(err as Error).message}`);
    }
  }

  function borderColor(c: OpCluster): string {
    if (c.has_subclusters) return 'border-blue-500/60';
    const p = c.purity ?? 0;
    if (p >= 0.8) return 'border-green-500/60';
    if (p >= 0.6) return 'border-orange-500/60';
    return 'border-red-500/60';
  }

  function purityBadge(c: OpCluster): { color: string; text: string } {
    const p = c.purity ?? 0;
    if (p >= 0.8) return { color: 'bg-green-500/20 text-green-300', text: 'pure' };
    if (p >= 0.6) return { color: 'bg-orange-500/20 text-orange-200', text: 'mixed' };
    return { color: 'bg-red-500/20 text-red-200', text: 'noisy' };
  }

  function open(c: OpCluster): void {
    // Special-case: clicking a cluster whose dominant class is
    // 'license_plate' should jump to the plate browse view (which
    // surfaces every crop with a plate_bbox_norm), not the single-
    // cluster crop grid. Plates live as sub-bboxes on vehicle crops
    // so the "license_plate" cluster only contains the rare crops
    // that were labeled with license_plate as their PRIMARY class —
    // usually 1-2 mis-labels. The operator's intent is "show me all
    // the plate items", so route them to the plates inventory.
    const dom = (c.dominant_class_name ?? '').toLowerCase();
    if (dom === 'license_plate') {
      const cls = classesStore.classes.find(
        (k) => (k.name ?? '').toLowerCase() === 'license_plate',
      );
      if (cls) {
        void goto(`/clusters?class=${cls.id}`);
        return;
      }
    }
    void goto(`/clusters/${c.id}`);
  }

  // Infinite scroll owns pagination — totalPages no longer needed.
</script>

<div class="flex h-full flex-col">
  <!-- Toolbar -->
  <div
    class="flex flex-wrap items-center gap-3 border-b border-zinc-800 px-4 py-2.5"
  >
    <h1 class="text-lg font-semibold">Clusters</h1>

    {#if classFilter != null}
      <span
        class="rounded-md border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-xs text-blue-200"
      >
        class filter: #{classFilter}
      </span>
    {/if}

    <span class="grow"></span>

    <!-- Color legend for the card border. The cluster grid uses border
         color to encode purity at a glance; without this strip the user
         has to mouse over each card to figure out what the colors mean. -->
    <div class="flex items-center gap-2 text-[10px] text-zinc-500" title="Card border color encodes cluster purity">
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-green-500/60"></span>
        ≥80%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-orange-500/60"></span>
        ≥60%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-red-500/60"></span>
        &lt;60%
      </span>
      <span class="flex items-center gap-1">
        <span class="inline-block h-2 w-3 rounded-sm border-2 border-blue-500/60"></span>
        sub-clustered
      </span>
    </div>

    <!-- Unlabeled-only filter: AHC clusters that didn't reach class-majority
         consensus surface as "Unlabeled #N". This toggle narrows the grid
         to just those, so operators can drain the unlabeled backlog in
         one pass (drag a cluster's reps into a class on the sidebar). -->
    <button
      type="button"
      onclick={() => {
        unlabeledOnly = !unlabeledOnly;
        if (unlabeledOnly) sort = 'size_desc';
      }}
      class="rounded border px-2 py-1 text-xs transition-colors {unlabeledOnly
        ? 'border-amber-500/60 bg-amber-500/20 text-amber-200'
        : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-amber-500/40'}"
      title="Show only clusters without a dominant class (need labeling)"
    >
      {unlabeledOnly ? '✓ ' : ''}Unlabeled only
      <span class="ml-1 font-mono text-[10px] text-zinc-500">({unlabeledCount})</span>
    </button>

    <label class="flex items-center gap-2 text-xs text-zinc-400">
      Sort
      <select
        bind:value={sort}
        class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-100"
      >
        <option value="purity_asc">purity asc</option>
        <option value="purity_desc">purity desc</option>
        <option value="size_desc">size desc</option>
        <option value="size_asc">size asc</option>
        <option value="dominant_class">dominant class</option>
      </select>
    </label>

    <!-- Primary-subject grid filters: scope cards to the largest / clear
         crops. Card size + reps reflect only passing crops, so a filtered
         grid is ready to drag-drop + AHC-refine on the subjects that matter. -->
    <div class="text-xs">
      <SubjectScopeToggle bind:value={subjectScope} />
    </div>
    <div class="text-xs">
      <BlurSlider
        bind:value={blurSlider}
        oncommit={commitClusterBlur}
        max={BLUR_MAX}
        width="w-28"
        title="Hide crops blurrier than this (blur_lap_ratio)"
      />
    </div>
  </div>

  <!-- Grid -->
  <div class="flex-1 overflow-auto p-4">
    {#if isLicensePlateFilter}
      <!-- Plates list view — backed by /curation/plates. Plates live as a
           plate_bbox_norm sub-bbox on each vehicle crop (not as their
           own cluster docs), so this view surfaces them directly with
           detector provenance + OCR text chips. -->
      <div class="flex min-h-0 flex-col gap-3">
        <!-- Sticky header: the filter strip + bulk-action toolbar stay pinned
             to the top of the scroll area, so the verify / false-positive /
             no-plate controls remain reachable while scrolling deep into a
             bucket or sub-cluster. -mx-4/-mt-4 cancels the scroll container's
             p-4 so it spans edge-to-edge and pins at the very top. -->
        <div
          class="sticky top-0 z-20 -mx-4 -mt-4 flex flex-col gap-3 border-b border-zinc-800 bg-zinc-950 px-4 pt-4 pb-3"
        >
        <!-- Filter strip: detector / verified / score / text-search. -->
        <div
          class="flex flex-wrap items-center gap-3 rounded-md border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs"
        >
          <label class="flex items-center gap-1.5">
            <span class="text-zinc-400">Detector</span>
            <select
              bind:value={plateDetectorFilter}
              class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
            >
              <option value="">any</option>
              <option value="lpr_nanov11_640">LPR</option>
              <option value="sam3">SAM3</option>
              <option value="paddleocr_det_trt">Paddle det</option>
              <option value="human">Human</option>
            </select>
          </label>
          <label class="flex items-center gap-1.5">
            <input type="checkbox" bind:checked={plateVerifiedOnly} class="accent-blue-500" />
            <span class="text-zinc-400">Verified only</span>
          </label>
          <label class="flex items-center gap-1.5">
            <span class="text-zinc-400">Min score</span>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              bind:value={plateMinScore}
              class="w-16 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
            />
          </label>
          <label class="flex items-center gap-1.5">
            <span class="text-zinc-400">Text</span>
            <input
              type="text"
              bind:value={plateTextQuery}
              placeholder="e.g. S14"
              class="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100 focus:border-blue-500 focus:outline-none"
            />
          </label>

          <!-- Top-N largest-crop gate. The sort runs on the largest 1-3
               crops, so this is the key filter for the plates that matter. -->
          <div class="inline-flex overflow-hidden rounded border border-zinc-700">
            {#each [{ v: null, l: 'All' }, { v: 1, l: 'Largest' }, { v: 2, l: '+2nd' }, { v: 3, l: '+3rd' }] as o (o.l)}
              <button
                type="button"
                class="px-2 py-1 {plateMaxRank === o.v
                  ? 'bg-blue-600 text-white'
                  : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-700'}"
                onclick={() => (plateMaxRank = o.v as number | null)}
              >
                {o.l}
              </button>
            {/each}
          </div>

          {#if suspectedFpView}
            <button
              type="button"
              class="rounded border border-zinc-600 bg-zinc-800 px-2 py-1 text-zinc-200 hover:bg-zinc-700"
              onclick={backToPlateClusters}
            >
              ← Clusters
            </button>
            <span class="font-medium text-red-200">Suspected false positives</span>
            <label class="flex items-center gap-1 text-[11px] text-zinc-400">
              ≤
              <input
                type="number"
                step="0.05"
                min="0"
                max="2"
                bind:value={suspectedFpThreshold}
                class="w-16 rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5 text-zinc-200"
              />
            </label>
            <button
              type="button"
              disabled={plateClusterBusy}
              class="rounded border border-red-500/50 bg-red-500/20 px-2 py-1 text-red-100 hover:bg-red-500/30 disabled:opacity-50"
              onclick={loadSuspectedFp}
            >
              {plateClusterBusy ? 'Loading…' : 'Reload'}
            </button>
          {:else if selectedPlateCluster == null}
            <button
              type="button"
              disabled={plateClusterBusy}
              class="rounded border border-purple-500/50 bg-purple-500/20 px-2 py-1 text-purple-100 hover:bg-purple-500/30 disabled:opacity-50"
              onclick={runClusterPlates}
              title="Group plates by visual similarity so outliers/false-positives surface"
            >
              {plateClusterBusy ? 'Clustering…' : '⟳ Cluster plates'}
            </button>
            <button
              type="button"
              disabled={plateClusterBusy}
              class="rounded border border-red-500/50 bg-red-500/20 px-2 py-1 text-red-100 hover:bg-red-500/30 disabled:opacity-50"
              onclick={loadSuspectedFp}
              title="List plate crops that look like known false positives (needs FP centroids built)"
            >
              Suspected FPs
            </button>
            <button
              type="button"
              disabled={plateClusterBusy}
              class="rounded border border-amber-500/50 bg-amber-500/20 px-2 py-1 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50"
              onclick={runBuildFpCentroids}
              title="Sub-type the false-positive bucket and (re)build its centroids"
            >
              {plateClusterBusy ? 'Building…' : 'Build FP centroids'}
            </button>
          {:else}
            <button
              type="button"
              class="rounded border border-zinc-600 bg-zinc-800 px-2 py-1 text-zinc-200 hover:bg-zinc-700"
              onclick={backToPlateClusters}
            >
              ← Clusters
            </button>
            {#if selectedPlateCluster === FP_PLATE_CLUSTER_ID}
              <span
                class="rounded bg-red-500/25 px-2 py-0.5 text-[11px] font-semibold tracking-wide text-red-200 uppercase"
              >
                ✗ False-positive cluster
              </span>
              <span class="text-[11px] text-zinc-400">not plates — hard negatives for LPR</span>
              <button
                type="button"
                disabled={plateClusterBusy}
                class="rounded border border-amber-500/50 bg-amber-500/20 px-2 py-1 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50"
                onclick={runBuildFpCentroids}
                title="Refine the FP bucket into sub-types and rebuild its centroids"
              >
                {plateClusterBusy ? 'Refining…' : 'Refine FP (build centroids)'}
              </button>
            {:else}
              <span class="font-mono text-[11px] text-zinc-300">bucket #{selectedPlateCluster}</span>
              <button
                type="button"
                disabled={plateClusterBusy}
                class="rounded border border-blue-500/50 bg-blue-500/20 px-2 py-1 text-blue-100 hover:bg-blue-500/30 disabled:opacity-50"
                onclick={runRefinePlateCluster}
                title="AHC-refine this bucket into sub-clusters to isolate outliers"
              >
                {plateClusterBusy ? 'Refining…' : 'Refine AHC'}
              </button>
              {#if plateRefineMsg}
                <span class="text-[11px] text-zinc-400">{plateRefineMsg}</span>
              {/if}
            {/if}
          {/if}
          <span class="grow"></span>
          {#if platePager.items.length > 0}
            <button
              type="button"
              class="rounded border border-zinc-700 bg-zinc-800 px-2 py-1 text-zinc-300 hover:bg-zinc-700"
              onclick={selectAllPlates}
              title="Select all loaded plates (shift-click a card for a range, ctrl/cmd-click to toggle)"
            >
              Select all
            </button>
          {/if}
          <span class="font-mono text-[11px] text-zinc-500">
            {platePager.items.length.toLocaleString()} / {platePager.total.toLocaleString()} platePager.items
          </span>
        </div>

        <!-- Bulk-action toolbar — appears when plates are selected. Triage
             outliers without leaving the gallery (no /review round-trip). -->
        {#if plateSel.size > 0}
          <div
            class="flex flex-wrap items-center gap-2 rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-xs"
          >
            <span class="font-medium text-blue-200">{plateSel.size} selected</span>
            <span class="grow"></span>
            <button
              type="button"
              disabled={plateBusy}
              class="rounded border border-red-500/50 bg-red-500/20 px-2 py-1 text-red-200 hover:bg-red-500/30 disabled:opacity-50"
              onclick={() => applyPlateStatus([...plateSel.ids], 'false_positive')}
            >
              ✗ Mark false positive
            </button>
            <button
              type="button"
              disabled={plateBusy}
              class="rounded border border-zinc-600 bg-zinc-800 px-2 py-1 text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
              onclick={() => applyPlateStatus([...plateSel.ids], 'no_plate_visible')}
            >
              No plate
            </button>
            <button
              type="button"
              disabled={plateBusy}
              class="rounded border border-green-500/50 bg-green-500/20 px-2 py-1 text-green-200 hover:bg-green-500/30 disabled:opacity-50"
              onclick={() => applyPlateStatus([...plateSel.ids], 'detected')}
            >
              ✓ Verify
            </button>
            <button
              type="button"
              class="rounded border border-zinc-700 px-2 py-1 text-zinc-400 hover:bg-zinc-800"
              onclick={() => plateSel.clear()}
            >
              Clear
            </button>
          </div>
        {/if}
        </div>
        <!-- /sticky header -->

        {#if platePager.error}
          <p class="text-sm text-red-300">API unavailable: {platePager.error}</p>
        {:else if !suspectedFpView && selectedPlateCluster == null && plateClusters.length > 0}
          <!-- Plate cluster cards. Click one to open its plates (with the
               bulk toolbar + AHC Refine). Buckets with sub-clusters (refined)
               get a blue border so refined buckets are easy to spot. The
               permanent false-positive bucket gets a red border + label. -->
          <ul class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
            {#each plateClusters as c (c.id)}
              <li style="content-visibility:auto;contain-intrinsic-size:auto 200px">
                <button
                  type="button"
                  class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {c.cluster_kind ===
                  'false_positive'
                    ? 'border-red-500/70'
                    : c.has_subclusters
                      ? 'border-blue-500/60'
                      : 'border-zinc-700'}"
                  onclick={() => openPlateCluster(c.id)}
                >
                  <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
                    {#each c.representative_thumb_urls?.slice(0, 4) ?? [] as url, i (i)}
                      <img
                        src={url}
                        alt="plate"
                        loading="lazy"
                        class="aspect-[2/1] w-full bg-zinc-950 object-contain"
                      />
                    {/each}
                  </div>
                  <div class="flex items-center justify-between gap-1 p-2 text-xs">
                    {#if c.cluster_kind === 'false_positive'}
                      <span
                        class="rounded bg-red-500/25 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-red-200 uppercase"
                        title="Permanent false-positive bucket — these are NOT plates"
                      >
                        ✗ False positives
                      </span>
                    {:else}
                      <span class="font-semibold text-zinc-200">#{c.id}</span>
                    {/if}
                    <span class="text-zinc-400">{c.size.toLocaleString()}</span>
                    {#if c.n_subclusters > 0}
                      <span class="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-200"
                        >{c.n_subclusters} sub</span
                      >
                    {/if}
                  </div>
                </button>
              </li>
            {/each}
          </ul>
        {:else if platePager.loading && platePager.items.length === 0}
          <p class="text-sm text-zinc-500">Loading plates...</p>
        {:else if platePager.items.length === 0}
          <p class="text-sm text-zinc-500">
            No plates match the current filters. The re-detection drain
            may still be populating provenance — fresh rows appear here
            as the worker processes them.
          </p>
        {:else}
          <!-- Sub-cluster tabs: appear once a bucket has been AHC-refined.
               "All" shows the grouped view (separators per sub-cluster);
               clicking a chip filters to that one sub-cluster. -->
          {#if selectedPlateCluster != null && plateSubclusterIds.length > 0}
            <div class="mb-3 flex flex-wrap items-center gap-1.5">
              <span class="text-[11px] text-zinc-500">sub-clusters:</span>
              <button
                type="button"
                class="rounded px-2 py-0.5 text-[11px] {plateSubTab === null
                  ? 'bg-blue-500/30 text-blue-100'
                  : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
                onclick={() => selectPlateSubTab(null)}
              >
                all
              </button>
              {#each plateSubclusterIds as sid (sid)}
                <button
                  type="button"
                  class="rounded px-2 py-0.5 font-mono text-[11px] {plateSubTab === sid
                    ? 'bg-blue-500/30 text-blue-100'
                    : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
                  onclick={() => selectPlateSubTab(sid)}
                >
                  {sid}
                  <span class="text-zinc-500">{plateSubCounts.get(sid) ?? ''}</span>
                </button>
              {/each}
            </div>
          {/if}

          {#each plateGroups as g (g.key)}
            {#if g.label}
              <div class="mt-3 mb-1.5 flex items-center gap-2">
                <span class="font-mono text-[11px] text-zinc-300">{g.label}</span>
                <span class="text-[11px] text-zinc-500">{g.items.length}</span>
                <span class="h-px grow bg-zinc-800"></span>
              </div>
            {/if}
            <div
              class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
            >
              {#each g.items as p (p.crop_id)}
                <PlateCard
                  crop={p}
                  selected={plateSel.has(p.crop_id)}
                  onclick={togglePlateSelect}
                  onedit={openPlateEditor}
                  onmarkfp={(c) => applyPlateStatus([c.crop_id], 'false_positive')}
                />
              {/each}
            </div>
          {/each}
          <!-- Sentinel AFTER the grid (not on it): the observer must root on a
               small element that only intersects once the user scrolls to the
               bottom. Attaching to the tall grid itself keeps it permanently
               intersecting and loads every page at once. -->
          <div
            use:infiniteScroll={{
              onload: loadPlatesMore,
              disabled: platePager.loading || platePager.loadingMore || !platePager.hasMore,
            }}
            class="mt-4 h-1"
            aria-hidden="true"
          ></div>
          {#if platePager.loadingMore}
            <p class="py-2 text-center text-xs text-zinc-500">Loading more…</p>
          {/if}
        {/if}
      </div>
    {:else if clusterPager.loading && gridItems.length === 0}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if clusterPager.error}
      <p class="text-sm text-red-300">API unavailable: {clusterPager.error}</p>
    {:else if gridItems.length === 0}
      <p class="text-sm text-zinc-500">
        No clusters yet — ingest some images and run the auto-label pipeline.
      </p>
    {:else}
      <ul class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {#each gridItems as c (c.id)}
          {@const pb = purityBadge(c)}
          <li style="content-visibility:auto;contain-intrinsic-size:auto 280px">
            <button
              type="button"
              class="flex w-full flex-col rounded-md border-2 bg-zinc-900 text-left transition hover:border-zinc-300 {borderColor(
                c,
              )}"
              onclick={() => open(c)}
            >
              <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t bg-zinc-950">
                {#each c.representative_crop_ids?.slice(0, 4) ?? [] as cropId, i (cropId)}
                  <img
                    src={c.representative_thumb_urls?.[i] ?? getThumbUrl(cropId)}
                    alt="thumb"
                    loading="lazy"
                    class="aspect-square w-full bg-zinc-950 object-contain"
                  />
                {/each}
                {#each Array(Math.max(0, 4 - (c.representative_crop_ids?.length ?? 0))) as _, i (i)}
                  <div class="aspect-square w-full bg-zinc-900"></div>
                {/each}
              </div>
              <div class="p-3">
                <div class="mb-1 flex items-center gap-2">
                  <span class="text-sm font-semibold">#{c.id}</span>
                  <span
                    class="rounded px-1.5 py-0.5 text-[10px] font-medium {pb.color}"
                  >
                    {pb.text} {((c.purity ?? 0) * 100).toFixed(0)}
                  </span>
                  {#if c.has_subclusters}
                    <span
                      class="rounded border border-blue-500/40 bg-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-200"
                    >
                      AHC
                    </span>
                  {/if}
                  <span class="grow"></span>
                  <span class="font-mono text-xs text-zinc-400">{c.size}</span>
                </div>
                <div
                  class="truncate text-sm text-zinc-300"
                  title={unlabeledOnly
                    ? `Unlabeled cluster #${c.id}`
                    : (c.dominant_class_name ?? `Unlabeled cluster #${c.id}`)}
                >
                  {#if c.dominant_class_name && !unlabeledOnly}
                    {c.dominant_class_name}
                    <span class="text-zinc-500">
                      · {((c.dominant_pct ?? 0) * 100).toFixed(0)}%
                    </span>
                  {:else}
                    <span class="text-amber-300">Unlabeled #{c.id}</span>
                    <span class="text-zinc-500">· needs label</span>
                  {/if}
                </div>
              </div>
            </button>
          </li>
        {/each}
      </ul>
      <!-- Sentinel MUST live inside the scroll container so the IntersectionObserver
           can root itself on the right element. Outside the overflow-auto div the
           observer falls back to viewport and either never fires or fires forever. -->
      <div
        use:infiniteScroll={{
          onload: loadMore,
          disabled: clusterPager.loadingMore || !clusterPager.hasMore || clusterPager.loading,
        }}
        class="mt-4 h-1"
        aria-hidden="true"
      ></div>
    {/if}
  </div>

  <!-- Status bar (no scroll sentinel here — see above) -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="font-mono text-xs text-zinc-500">
      {gridItems.length} / {clusterPager.total + (classFilter == null && lpCard != null ? 1 : 0)}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if clusterPager.loadingMore}loading more…{:else if clusterPager.hasMore}scroll for more{:else}all loaded{/if}
    </span>
  </div>
</div>

{#if editPlateCrop}
  <PlateEditor
    crop={editPlateCrop}
    onsave={savePlateBbox}
    onclose={() => (editPlateCrop = null)}
  />
{/if}
