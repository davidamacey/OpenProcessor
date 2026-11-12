/**
 * Cluster cards, one cluster's crops and moving crops between clusters (`/clusters*`).
 */
import { getCrops } from '$lib/api_cropRead';
import type { CursorPage, SortOption } from '$lib/types_cursor';
import { apiFetch, qs, scoped } from '$lib/api';
import { apiErrorText } from '$lib/api_errors';
import type { Cluster, ClusterFilter, Crop, CropFilter, HiddenCounts } from '$lib/types';
import type { LabelOpAccepted } from '$lib/types_labelOps';
import type {
  ClusterActivationAccepted,
  ClusterActivationRequest,
  ClusterFacets,
  ClusterFlag,
  ClusterPartitionList,
  ClusterRunAccepted,
  ClusterTarget,
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

export type ClusterEmptyReason = 'no_partition' | 'pending_activation' | 'filtered_empty';

/** The newest built partition that is not serving yet. */
export interface PendingPartition {
  partition_id: string;
  clusters: number;
  items: number;
}

type RawClustersResp = {
  items: RawCluster[];
  total: number;
  /** D-4: the representatives window this response actually populated —
   *  echoed back so a caller windowing successive calls can advance past
   *  exactly what it got, not what it asked for. */
  representatives_offset?: number;
  representatives_limit?: number;
  sort?: string;
  sort_options?: SortOption[];
  flag_vocabulary?: ClusterFlag[];
  truncated?: boolean;
  empty_reason?: ClusterEmptyReason | null;
  empty_message?: string | null;
  pending_partition?: PendingPartition | null;
  hidden?: HiddenCounts | null;
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

export interface ClustersResponse extends CursorPage<Cluster> {
  /** D-4: the representatives window this response actually populated
   *  (echoed straight off the raw response) — a caller paging through
   *  windows advances by this, not by what it asked for. */
  representatives_offset: number;
  representatives_limit: number;
  /** The sort the server applied, the sorts it offers, the flag
   *  vocabulary, and whether the list was cut at `max_clusters`. */
  sort: string | null;
  sort_options: SortOption[];
  flag_vocabulary: ClusterFlag[];
  truncated: boolean;
  /** Why `items` is empty, with the served text; null when cards are served. */
  empty_reason: ClusterEmptyReason | null;
  empty_message: string | null;
  pending_partition: PendingPartition | null;
  /** What the filter leaves out (the frozen holdout, ignored items), counted by the server. */
  hidden: HiddenCounts | null;
}

function _rawToClustersResponse(
  raw: RawClustersResp,
  filter: ClusterFilter,
): ClustersResponse {
  const items = (raw.items ?? []).map(_rawClusterToCluster);
  return {
    items,
    total: raw.total ?? items.length,
    next_cursor: null,
    limit: items.length,
    sort: raw.sort ?? null,
    sort_options: raw.sort_options ?? [],
    flag_vocabulary: raw.flag_vocabulary ?? [],
    truncated: raw.truncated === true,
    empty_reason: raw.empty_reason ?? null,
    empty_message: raw.empty_message ?? null,
    pending_partition: raw.pending_partition ?? null,
    hidden: raw.hidden ?? null,
    representatives_offset:
      raw.representatives_offset ?? filter.representatives_offset ?? 0,
    representatives_limit:
      raw.representatives_limit ?? filter.representatives_limit ?? 50,
  };
}

/** The cluster, area, location and group keys of the shared filter. */
const areaParams = (f: ClusterFilter) => ({
  cluster_uid: f.cluster_uid,
  geo_bbox: f.geo_bbox,
  geo_polygon: f.geo_polygon,
  has_location: f.has_location,
  near_dup_group: f.near_dup_group,
});

/** Card-level and primary-subject grid filters; card stats reflect only passing crops. */
const cardParams = (f: ClusterFilter) => ({
  min_size: f.min_size ?? undefined,
  max_size: f.max_size ?? undefined,
  validation_state: f.validation_state ?? undefined,
  flag: f.flag,
  needs_review: f.needs_review ? true : undefined,
  max_rank: f.max_rank ?? undefined,
  min_blur_ratio: f.min_blur_ratio ?? undefined,
  class_source: f.class_source ?? undefined,
  offset: f.representatives_offset ?? undefined,
  limit: f.representatives_limit ?? undefined,
});

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
      import_id: filter.import_id,
      taken_from: filter.taken_from ?? undefined,
      taken_to: filter.taken_to ?? undefined,
      image_id: filter.image_id,
      ...areaParams(filter),
      // Enough buckets to return every cluster in one call; `truncated` says when not.
      max_clusters: 2000,
      sort: filter.sort,
      dominant_class: filter.dominant_class,
      detector_class: filter.detector_class,
      ...cardParams(filter),
    })}`,
    {},
    signal,
  );
  return _rawToClustersResponse(raw, filter);
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

export interface GetClusterOpts {
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
}

/** Nullable option values become absent filter keys. */
function clusterCropFilter(uid: string, limit: number, opts: GetClusterOpts): CropFilter {
  const filter = {
    cluster_uid: [uid],
    limit,
    class_source: opts.classSource,
    max_rank: opts.maxRank,
    min_blur_ratio: opts.minBlurRatio,
    classifier_conf_lt: opts.classifierConfLt,
    order: opts.order,
    k: opts.k,
    detector_class_name: opts.detectorClassName?.length
      ? opts.detectorClassName
      : undefined,
    detector_conf_min: opts.detectorConfMin,
    detector_conf_max: opts.detectorConfMax,
    review_tab: opts.reviewTab,
    label_validated: opts.labelValidated,
    sort: opts.sort,
  };
  return Object.fromEntries(
    Object.entries(filter).filter(([, v]) => v != null),
  ) as unknown as CropFilter;
}

export async function getCluster(
  uid: string,
  cursor: string | null = null,
  limit = 60,
  signal?: AbortSignal,
  opts: GetClusterOpts = {},
): Promise<{
  /** The served card (`GET /clusters/{uid}`); null when the server refused it, see `cardError`. */
  cluster: Cluster | null;
  crops: CursorPage<Crop>;
  /** Why the cluster card could not be read (the crops still render), or null. */
  cardError: string | null;
}> {
  // Two parallel calls: one page of crops + the authoritative cluster card
  // from {API_PREFIX}/clusters/{uid} (server-computed). The page no longer
  // derives any of the cluster's identity fields client-side.
  //
  // `classSource` narrows the crop grid to one source bucket
  // (model / vlm / human / low_conf / ...) without touching
  // the cluster card stats — the header still shows the whole-cluster
  // totals so the operator sees the filter against the full size.
  const filter = clusterCropFilter(uid, limit, opts);
  let cardError: string | null = null;
  const [crops, card] = await Promise.all([
    getCrops(filter, cursor, signal),
    apiFetch<RawCluster>(
      `${scoped()}/clusters/${encodeURIComponent(uid)}${qs({ per_cluster: 4 })}`,
      {},
      signal,
    ).catch((e: unknown) => {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      cardError = apiErrorText(e);
      return null;
    }),
  ]);
  return { cluster: card ? _rawClusterToCluster(card) : null, cardError, crops };
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
