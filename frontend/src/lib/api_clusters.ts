/**
 * Cluster cards, one cluster's crops and moving crops between clusters (`/clusters*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import { apiErrorText } from '$lib/api_errors';
import type { Cluster, ClusterFilter, Crop, PaginatedResponse } from '$lib/types';
import type { ReprocessRequest } from '$lib/types_import';
import type { LabelOpAccepted } from '$lib/types_labelOps';
import type {
  ClusterActivationAccepted,
  ClusterActivationRequest,
  ClusterFacets,
  ClusterFlag,
  ClusterPartitionList,
  ClusterRunAccepted,
  ClusterTarget,
  ClusterSortOption,
  ValidationState,
} from '$lib/types_clusterList';

/** Raw cluster card from `{API_PREFIX}/clusters` (`ClusterCard`). The backend is the
 *  single source of truth for every field; the frontend never recomputes the
 *  dominant class, the agreement or is_unlabeled. */
type RawCluster = {
  cluster_uid: string;
  display_no: number | null;
  size: number;
  validated_count: number;
  labelled_count: number;
  dominant_class_name: string | null;
  dominant_count: number;
  /** The top class's share of the LABELLED members. */
  label_agreement: number | null;
  /** Share of this cluster's members that have any label at all. */
  labelled_share: number | null;
  is_unlabeled: boolean;
  updated_at: string | null;
  representatives: Array<{
    crop_id: string;
    cluster_score: number | null;
    class_name: string | null;
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

function _rawClusterToCluster(c: RawCluster): Cluster {
  const label_agreement = c.label_agreement ?? null;
  return {
    uid: c.cluster_uid,
    display_no: c.display_no ?? null,
    size: c.size,
    validated_count: c.validated_count,
    dominant_class_name: c.dominant_class_name,
    // The subtitle's "dominant share" is the served label-based share
    // (largest class among labelled members).
    dominant_pct: label_agreement,
    dominant_count: c.dominant_count ?? null,
    labelled_count: c.labelled_count ?? null,
    label_agreement,
    labelled_share: c.labelled_share,
    is_unlabeled: c.is_unlabeled,
    representative_crop_ids: (c.representatives ?? []).map((r) => r.crop_id),
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
      taken_from: filter.taken_from ?? undefined,
      taken_to: filter.taken_to ?? undefined,
      image_id: filter.image_id,
      cluster_uid: filter.cluster_uid ?? undefined,
      // Enough buckets to return every cluster in one call; `truncated` says when not.
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
  const items = (raw.items ?? []).map(_rawClusterToCluster);
  return {
    items,
    total: raw.total ?? items.length,
    page: 1,
    page_size: items.length,
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

/** One cluster's facet counts (`GET {P}/clusters/{cluster_uid}/facets`). */
export function getClusterFacets(
  uid: string,
  signal?: AbortSignal,
): Promise<ClusterFacets> {
  return apiFetch<ClusterFacets>(
    `${scoped()}/clusters/${encodeURIComponent(uid)}/facets`,
    {},
    signal,
  );
}

export async function getCluster(
  uid: string,
  page = 1,
  pageSize = 60,
  signal?: AbortSignal,
  opts: {
    classSource?: string | null;
    maxRank?: number | null;
    minBlurRatio?: number | null;
    classifierConfLt?: number | null;
    /**
     * Forwarded verbatim to `{API_PREFIX}/crops?order=`. Only `'outliers'` is
     * special-cased server-side today (crops.py `order` query param —
     * see the curation strategy plan §1); an id the backend
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
    cluster_uid: uid,
    page,
    page_size: pageSize,
  };
  if (opts.classSource) cropQuery.class_source = opts.classSource;
  if (opts.maxRank != null) cropQuery.max_rank = opts.maxRank;
  if (opts.minBlurRatio != null) cropQuery.min_blur_ratio = opts.minBlurRatio;
  if (opts.classifierConfLt != null) cropQuery.classifier_conf_lt = opts.classifierConfLt;
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
      `${scoped()}/clusters${qs({ per_cluster: 4, max_clusters: 1, cluster_uid: uid })}`,
      {},
      signal,
    ).catch((e: unknown) => {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      cardError = apiErrorText(e);
      return null;
    }),
  ]);
  const items = cropPage.crops.map(mapRawCrop);
  const found = clustersResp?.items?.find((c) => c.cluster_uid === uid) ?? null;
  const cluster: Cluster = found
    ? _rawClusterToCluster(found)
    : {
        // Fallback only if the cluster card lookup failed — leaves
        // identity fields null but lets the crop grid render.
        uid,
        size: cropPage.total,
        validated_count: 0,
        dominant_class_name: null,
        dominant_pct: null,
        is_unlabeled: true,
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

// -- DnD: move crops between clusters ------------------------------------

/** Queue a recorded move into a cluster of the serving partition (202 + job id);
 *  follow it with `runLabelOp`. 400 when the uid is not one of its clusters. */
export function moveCropsToCluster(
  cropIds: string[],
  targetClusterUid: string,
  signal?: AbortSignal,
): Promise<LabelOpAccepted> {
  return apiFetch<LabelOpAccepted>(
    `${scoped()}/crops/move`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, cluster_uid: targetClusterUid }),
    },
    signal,
  );
}

// -- partitions: fit, list, activate ---------------------------------------

/** `POST /clusters/runs`: queue a clustering run for a target (202; 409 `job_busy`
 *  while one for that space is active). Follow `job_id` with `followJob`: its
 *  result says `activated` (the server activates a run that is safe to). */
export function startClusterRun(
  target: ClusterTarget,
  signal?: AbortSignal,
): Promise<ClusterRunAccepted> {
  return apiFetch<ClusterRunAccepted>(
    `${scoped()}/clusters/runs`,
    { method: 'POST', body: JSON.stringify({ target }) },
    signal,
  );
}

/** `GET /clusters/partitions`, newest first. */
export function getClusterPartitions(
  params: { target?: ClusterTarget; limit?: number } = {},
  signal?: AbortSignal,
): Promise<ClusterPartitionList> {
  return apiFetch<ClusterPartitionList>(
    `${scoped()}/clusters/partitions${qs({ target: params.target, limit: params.limit })}`,
    {},
    signal,
  );
}

/** `PUT /clusters/activation`: serve a ready partition (202; 409
 *  `partition_conflict` when the serving partition or revision moved). */
export function activateClusterPartition(
  body: ClusterActivationRequest,
  signal?: AbortSignal,
): Promise<ClusterActivationAccepted> {
  return apiFetch<ClusterActivationAccepted>(
    `${scoped()}/clusters/activation`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
}
