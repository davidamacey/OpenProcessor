/**
 * Cluster cards, one cluster's crops, refine and moving crops between
 * clusters (`/clusters*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import { apiErrorText } from '$lib/api_errors';
import type {
  BulkLabelResult,
  Cluster,
  ClusterFilter,
  Crop,
  PaginatedResponse,
} from '$lib/types';
import type { ReprocessRequest } from '$lib/types_import';
import type {
  ClusterFacets,
  ClusterFlag,
  ClusterSortOption,
  ValidationState,
} from '$lib/types_clusterList';

/** Raw cluster card from `{API_PREFIX}/clusters`. The backend is the single
 *  source of truth for every field — the frontend must never recompute
 *  dominant_class, purity, or is_unlabeled. */
type RawCluster = {
  cluster_id: number;
  cluster_kind: 'class' | 'candidate' | 'unassigned';
  size: number;
  validated_count: number;
  labelled_count: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  dominant_count: number;
  /** DQ-M2 fix (dq-queues cutover, 2026-09-24): the share of the
   *  `purity_n` members the cluster-geometry pass measured for this
   *  cluster whose NEAREST CLUSTER CENTROID is this cluster's own —
   *  independent of the labels that placed them, so a class cluster is
   *  no longer 1.0 by construction. Null when no member has been
   *  measured. */
  purity: number | null;
  /** How many members `purity` was computed over (the geometry pass's
   *  coverage for this cluster) — purity is noisy at low n. */
  purity_n: number | null;
  /** Always `'nearest_centroid'` today; served so the frontend never
   *  hardcodes what `purity` means. */
  purity_basis: string | null;
  /** Server-banded purity (see `purity_thresholds` below) — 'pure' | 'mixed' | 'noisy'. */
  purity_tier: 'pure' | 'mixed' | 'noisy' | null;
  /** The top class's share of the LABELLED members. */
  label_agreement: number | null;
  /** Share of this cluster's members that have any label at all. */
  labelled_share: number | null;
  /** Server's auto-promote eligibility gate for this cluster. */
  promotable: boolean;
  is_unlabeled: boolean;
  n_subclusters: number;
  subcluster_ids: string[];
  updated_at: string | null;
  representatives: Array<{
    crop_id: string;
    cluster_distance: number | null;
    class_name: string | null;
    cluster_subid: string | null;
  }>;
  detector_top_class?: string | null;
  detector_top_share?: number | null;
  detector_conf_mean?: number | null;
  validation_state?: ValidationState;
  review_counts?: Record<string, number>;
  needs_review_count?: number;
  flags?: string[];
};

type RawClustersResp = {
  items: RawCluster[];
  total: number;
  total_class_clusters: number;
  total_candidate_clusters: number;
  cluster_id_offset: number;
  /** Thresholds behind every item's `purity_tier`/`promotable` — informational,
   *  not re-applied client-side. */
  purity_thresholds?: {
    pure_min: number;
    mixed_min: number;
    promote_min_members: number;
    promote_min_labelled_share: number;
  } | null;
  /** Similarity floor behind every crop's `cluster_is_core` — copied onto
   *  each mapped `Cluster` so `/clusters/[id]`'s cut line never hardcodes it. */
  core_similarity_min?: number | null;
  /** D-4: the representatives window this response actually populated —
   *  echoed back so a caller windowing successive calls can advance past
   *  exactly what it got, not what it asked for. */
  representatives_offset?: number;
  representatives_limit?: number;
  sort?: string;
  sort_options?: ClusterSortOption[];
  flag_vocabulary?: ClusterFlag[];
  truncated?: boolean;
};

function _rawClusterToCluster(
  c: RawCluster,
  coreSimilarityMin: number | null = null,
): Cluster {
  const label_agreement = c.label_agreement ?? null;
  return {
    id: c.cluster_id,
    cluster_kind: c.cluster_kind,
    size: c.size,
    validated_count: c.validated_count,
    dominant_class_id: c.dominant_class_id,
    dominant_class_name: c.dominant_class_name,
    // C1 (visual audit 2026-09-24): the subtitle's "dominant share" is
    // the served label-based share (largest class among labelled
    // members), never the nearest-centroid geometry `purity` below —
    // mapping `purity` here rendered "class_b · 3%" for a cluster that
    // is 616/616 class_b.
    dominant_pct: label_agreement,
    dominant_count: c.dominant_count ?? null,
    labelled_count: c.labelled_count ?? null,
    purity: c.purity,
    purity_n: c.purity_n,
    purity_basis: c.purity_basis,
    purity_tier: c.purity_tier ?? null,
    label_agreement,
    labelled_share: c.labelled_share,
    promotable: !!c.promotable,
    core_similarity_min: coreSimilarityMin,
    is_unlabeled: c.is_unlabeled,
    representative_crop_ids: (c.representatives ?? []).map((r) => r.crop_id),
    has_subclusters: c.n_subclusters > 0,
    n_subclusters: c.n_subclusters,
    subcluster_ids: c.subcluster_ids,
    updated_at: c.updated_at,
    detector_top_class: c.detector_top_class ?? null,
    detector_top_share: c.detector_top_share ?? null,
    detector_conf_mean: c.detector_conf_mean ?? null,
    validation_state: c.validation_state,
    review_counts: c.review_counts ?? {},
    needs_review_count: c.needs_review_count ?? 0,
    flags: c.flags ?? [],
  };
}

export interface ClustersResponse extends PaginatedResponse<Cluster> {
  /** D-4: the representatives window this response actually populated
   *  (echoed straight off the raw response) — a caller paging through
   *  windows advances by this, not by what it asked for. */
  representatives_offset: number;
  representatives_limit: number;
  /** The sort the server applied, the sorts it offers, the flag
   *  vocabulary, and whether the list was cut at `max_clusters`. */
  sort: string | null;
  sort_options: ClusterSortOption[];
  flag_vocabulary: ClusterFlag[];
  truncated: boolean;
}

export async function getClusters(
  filter: ClusterFilter = {},
  signal?: AbortSignal,
): Promise<ClustersResponse> {
  // One round-trip returns every card (up to max_clusters) already sorted and
  // filtered by the server, with the quality fields on each card. The
  // frontend only shapes the result into the Cluster type. Representative
  // crops are served for the `offset`/`limit` window of that list only, so
  // a caller asks for the window it is about to render.
  const raw = await apiFetch<RawClustersResp>(
    `${scoped()}/clusters${qs({
      per_cluster: filter.per_cluster ?? 4,
      class_name: filter.class_name,
      exclude_class_name: filter.exclude_class_name,
      conf_min: filter.conf_min ?? undefined,
      conf_max: filter.conf_max ?? undefined,
      min_area: filter.min_area ?? undefined,
      max_area: filter.max_area ?? undefined,
      origin: filter.origin,
      embedding_state: filter.embedding_state,
      review_status: filter.review_status,
      cluster_id: filter.cluster_id ?? undefined,
      // Enough buckets that the 512 IVF candidate clusters and the class
      // clusters all come back in one call; `truncated` says when not.
      max_clusters: 2000,
      sort: filter.sort,
      dominant_class: filter.dominant_class,
      detector_class: filter.detector_class,
      min_size: filter.min_size ?? undefined,
      max_size: filter.max_size ?? undefined,
      validation_state: filter.validation_state ?? undefined,
      flag: filter.flag,
      needs_review: filter.needs_review ? true : undefined,
      // Primary-subject grid filters — card stats reflect only passing crops.
      max_rank: filter.max_rank ?? undefined,
      min_blur_ratio: filter.min_blur_ratio ?? undefined,
      class_source: filter.class_source ?? undefined,
      offset: filter.representatives_offset ?? undefined,
      limit: filter.representatives_limit ?? undefined,
    })}`,
    {},
    signal,
  );
  const coreSimilarityMin = raw.core_similarity_min ?? null;
  const items = (raw.items ?? []).map((c) => _rawClusterToCluster(c, coreSimilarityMin));
  return {
    items,
    total: raw.total ?? items.length,
    page: 1,
    page_size: items.length,
    purity_thresholds: raw.purity_thresholds ?? null,
    sort: raw.sort ?? null,
    sort_options: raw.sort_options ?? [],
    flag_vocabulary: raw.flag_vocabulary ?? [],
    truncated: raw.truncated === true,
    representatives_offset:
      raw.representatives_offset ?? filter.representatives_offset ?? 0,
    representatives_limit:
      raw.representatives_limit ?? filter.representatives_limit ?? 50,
  };
}

/** One cluster's facet counts (`GET {P}/clusters/{id}/facets`). */
export function getClusterFacets(
  id: number,
  signal?: AbortSignal,
): Promise<ClusterFacets> {
  return apiFetch<ClusterFacets>(`${scoped()}/clusters/${id}/facets`, {}, signal);
}

export async function getCluster(
  id: number,
  page = 1,
  pageSize = 60,
  signal?: AbortSignal,
  opts: {
    classSource?: string | null;
    maxRank?: number | null;
    minBlurRatio?: number | null;
    classifierConfLt?: number | null;
    /** One sub-cluster of the cluster, filtered by the server (`/crops?cluster_subid=`). */
    clusterSubid?: string | null;
    /**
     * Forwarded verbatim to `{API_PREFIX}/crops?order=`. Only `'outliers'` is
     * special-cased server-side today (crops.py `order` query param —
     * see docs/curation-strategy-plan-2026-09.md §1); an id the backend
     * doesn't recognize is harmless (qs() still sends it, the server
     * just falls back to its default ordering). Typed as `string` rather
     * than a fixed union so a new `{API_PREFIX}/methods`-reported order id doesn't
     * require touching this signature — callers should still gate which
     * ids they actually offer against what `{API_PREFIX}/methods` reports.
     */
    order?: string | null;
    /**
     * Pool-scale overlay parameter, forwarded to `{API_PREFIX}/crops?k=` only when
     * set (curation-strategy plan Phase 4 — `order: 'diverse'`'s "how
     * many diverse crops" count). Meaningless for every other `order`
     * value; the caller (`/clusters/[id]`) only sets it in diverse mode.
     */
    k?: number | null;
    /** The facet filters, each sent as the served `GET /crops` parameter of
     *  the same name: the detector's class (repeatable), a confidence band
     *  (`detector_conf_min` inclusive, `detector_conf_max` exclusive), a
     *  review tab id, and the validated flag. */
    detectorClassName?: string[];
    detectorConfMin?: number | null;
    detectorConfMax?: number | null;
    reviewTab?: string | null;
    labelValidated?: boolean | null;
    /** `GET /crops` `sort`, e.g. `detector_confidence:desc`. */
    sort?: string | null;
  } = {},
): Promise<{
  cluster: Cluster;
  crops: PaginatedResponse<Crop>;
  /** Why the cluster card could not be read (the crops still render), or null. */
  cardError: string | null;
}> {
  // Two parallel calls: paginated crops + the authoritative cluster
  // card from {API_PREFIX}/clusters (server-computed). The page no longer
  // derives any of the cluster's identity fields client-side.
  //
  // `classSource` narrows the crop grid to one source bucket
  // (model / vlm / human / low_conf / ...) without touching
  // the cluster card stats — the header still shows the whole-cluster
  // totals so the operator sees the filter against the full size.
  type CropPage = {
    total: number;
    page: number;
    page_size: number;
    crops: RawCrop[];
    /** Only present for a pool-scale overlay ordering (e.g.
     *  `order=diverse`) — see `PaginatedResponse.order_method` /
     *  `.order_version` / `.n_pool` in types.ts. Untyped/optional and
     *  parsed tolerantly below: the shape belongs to whichever `order`
     *  overlay is active, not something this function should assume. */
    method?: unknown;
    version?: unknown;
    n_pool?: unknown;
    /** v0.4.0 ordered views: matching items with no vector, and the
     *  request that would embed them (set only when the count is > 0). */
    n_unembedded?: number | null;
    suggested_reprocess?: ReprocessRequest | null;
  };
  const cropQuery: Record<string, unknown> = {
    cluster_id: id,
    page,
    page_size: pageSize,
  };
  if (opts.classSource) cropQuery.class_source = opts.classSource;
  if (opts.maxRank != null) cropQuery.max_rank = opts.maxRank;
  if (opts.minBlurRatio != null) cropQuery.min_blur_ratio = opts.minBlurRatio;
  if (opts.classifierConfLt != null) cropQuery.classifier_conf_lt = opts.classifierConfLt;
  if (opts.clusterSubid) cropQuery.cluster_subid = opts.clusterSubid;
  if (opts.order) cropQuery.order = opts.order;
  if (opts.k != null) cropQuery.k = opts.k;
  if (opts.detectorClassName?.length)
    cropQuery.detector_class_name = opts.detectorClassName;
  if (opts.detectorConfMin != null) cropQuery.detector_conf_min = opts.detectorConfMin;
  if (opts.detectorConfMax != null) cropQuery.detector_conf_max = opts.detectorConfMax;
  if (opts.reviewTab) cropQuery.review_tab = opts.reviewTab;
  if (opts.labelValidated != null) cropQuery.label_validated = opts.labelValidated;
  if (opts.sort) cropQuery.sort = opts.sort;
  let cardError: string | null = null;
  const [cropPage, clustersResp] = await Promise.all([
    apiFetch<CropPage>(`${scoped()}/crops${qs(cropQuery)}`, {}, signal),
    apiFetch<RawClustersResp>(
      // cluster_id (not class_id!) is the correct filter for "fetch this
      // one cluster's card by its own identity" — cluster_id == class_id
      // is only an invariant for 'class' clusters, so filtering by
      // class_id silently returned nothing for every 'candidate' cluster
      // (id >= RESIDUAL_CLUSTER_ID_OFFSET, no matching class exists),
      // which fell back to the null-identity stub below and showed no
      // human-readable name in the header even though {API_PREFIX}/clusters'
      // list view has dominant_class_name for the same cluster.
      `${scoped()}/clusters${qs({ per_cluster: 4, max_clusters: 1, cluster_id: id })}`,
      {},
      signal,
    ).catch((e: unknown) => {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      cardError = apiErrorText(e);
      return null;
    }),
  ]);
  const items = cropPage.crops.map(mapRawCrop);
  const found = clustersResp?.items?.find((c) => c.cluster_id === id) ?? null;
  const cluster: Cluster = found
    ? _rawClusterToCluster(found, clustersResp?.core_similarity_min ?? null)
    : {
        // Fallback only if the cluster card lookup failed — leaves
        // identity fields null but lets the crop grid render.
        id,
        cluster_kind: 'unassigned',
        size: cropPage.total,
        validated_count: 0,
        dominant_class_id: null,
        dominant_class_name: null,
        dominant_pct: null,
        purity: null,
        purity_tier: null,
        promotable: false,
        core_similarity_min: null,
        is_unlabeled: true,
        has_subclusters: false,
        n_subclusters: 0,
        subcluster_ids: [],
        representative_crop_ids: items.slice(0, 4).map((c) => c.id),
        updated_at: null,
      };
  return {
    cluster,
    cardError,
    crops: {
      items,
      total: cropPage.total,
      page: cropPage.page,
      page_size: cropPage.page_size,
      order_method: typeof cropPage.method === 'string' ? cropPage.method : null,
      order_version: typeof cropPage.version === 'string' ? cropPage.version : null,
      n_pool: typeof cropPage.n_pool === 'number' ? cropPage.n_pool : null,
      n_unembedded: cropPage.n_unembedded ?? null,
      suggested_reprocess: cropPage.suggested_reprocess ?? null,
    },
  };
}

export type RefineClusterResponse = {
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  purity: number;
  subcluster_weighted_purity?: number;
  n_updated?: number;
  distance_threshold?: number;
  linkage?: string;
  metric?: string;
  action: 'refined' | 'skipped_too_small' | 'skipped_too_large';
  reason?: string;
};

export function refineCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<RefineClusterResponse> {
  return apiFetch<RefineClusterResponse>(
    `${scoped()}/clusters/refine/${clusterId}`,
    { method: 'POST' },
    signal,
  );
}

// -- DnD: move crops between clusters ------------------------------------

export function moveCropsToCluster(
  cropIds: string[],
  targetClusterId: number,
  signal?: AbortSignal,
): Promise<BulkLabelResult> {
  // Backend returns the same {updated, conflicts: [...]} shape as
  // {API_PREFIX}/crops/batch_label. Reuse the type so both call sites share the
  // conflict-handling code path.
  return apiFetch<BulkLabelResult>(
    `${scoped()}/crops/move`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, cluster_id: targetClusterId }),
    },
    signal,
  );
}
