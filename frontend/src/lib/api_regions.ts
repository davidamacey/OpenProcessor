/**
 * Region browse, region clustering, FP centroids, training candidates, and
 * the region status / vocabulary reads (`/regions*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { SlotData, SlotKey, XYXY } from '$lib/annotations/types';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import type { Cluster, ServedRegionProfile } from '$lib/types';

// -- region browse / training-cohort selection --------------------------

/**
 * Base path for the region/annotation-slot collection endpoints (the
 * region browse sub-system below — cluster, FP centroids, training
 * candidates, etc). The single place this path lives: every call site
 * below builds on it, and `regionRouteScan.test.ts` fails the build if a
 * bare path literal reappears.
 */
export const REGION_BASE = '/regions';

/** One row of a region browse route (`RegionRow`): the full item plus the
 *  box the row is about. Every per-box value (geometry, score, detector,
 *  text, cluster) is an element of `region_boxes`, read through
 *  `slots[slotKey].subBoxes`; the row's own box is the one whose id equals
 *  `region_box_id`. */
export interface RegionBrowseItem {
  crop_id: string;
  /** Non-empty and unique across the pages of one query
   *  (`<crop_id>#<region_box_id>`, or `<crop_id>#item` for an item-level
   *  row): key every list of these rows on it. */
  row_key: string;
  id: string;
  image_path: string;
  /** The source image's id; targets an image Reprocess. */
  image_id?: string;
  bbox_norm: number[];
  region_status: string | null;
  region_verified: boolean | null;
  region_validated: boolean | null;
  region_detector_chain: string[] | null;
  region_detected_at: string | null;
  region_verifier: string | null;
  region_verifier_version: string | null;
  region_verified_at: string | null;
  region_rejection_reason: string | null;
  region_visible: boolean | null;
  class_id: number | null;
  class_name: string | null;
  cluster_id: number | null;
  /** Parent-crop rank by size in its image (1 = largest). */
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  updated_at: string;
  thumbnail_url?: string;
  selection_reason?: string;
  /** Per-slot capability data — see `Crop.slots` in types.ts. Added by
   *  `getRegions` via `mapCropSlots`. */
  slots?: Record<SlotKey, SlotData>;
  /** The box this row is about; null for an item-level row. */
  region_box_id?: string | null;
}

export interface RegionsPage {
  total: number;
  page: number;
  page_size: number;
  items: RegionBrowseItem[];
  mode?: string;
  selection_reason?: string;
  /** `RegionRowPage.total_rows` — counts ROWS (boxes, when the request
   *  selects boxes), vs. `total` which counts ITEMS (the unit pages
   *  paginate). Shown beside the item count when they differ — see
   *  `SlotGallery.svelte`'s count chip. */
  total_rows?: number;
  /** True when an item on this page matched more boxes than the index
   *  reports per item, so some of its rows are missing (`total_rows`
   *  still counts them). */
  rows_truncated?: boolean;
}

export interface RegionsQuery extends Omit<
  ItemFilterQuery,
  'open_vocab_set' | 'source_prompt'
> {
  page?: number;
  page_size?: number;
  cluster_id?: number;
  /** Region clustering bucket (independent of the item cluster_id). */
  region_cluster_id?: number;
  /** AHC region sub-cluster id (e.g. "17a"). */
  region_cluster_subid?: string;
  /** Order a bucket's regions by sub-cluster so AHC groups come back contiguous. */
  sort_by_subid?: boolean;
  /** Only regions on the top-N largest crops (crop_rank_in_image<=N). */
  max_rank?: number;
  min_score?: number;
  max_score?: number;
  verified?: boolean;
  detector?: string;
  text?: string;
  include_test?: boolean;
  /** Region lifecycle status (`GET {API_PREFIX}/regions/statuses` for the
   *  deployment's vocabulary) — the backend 400s on an unknown value.
   *  Independent of `verified`, which is a boolean, not a status. */
  status?: string;
  /** Per-box state filter (`GET {API_PREFIX}/regions/statuses` `box_states`
   *  serves the vocabulary): every box filter applies to the same box. */
  box_state?: string;
}

/** `browsePath` is the slot's declared browse collection
 *  (`capabilities.queue.browsePath`), so a slot never inherits another
 *  slot's route by accident. */
export async function getRegions(
  browsePath: string,
  params: RegionsQuery = {},
  signal?: AbortSignal,
): Promise<RegionsPage> {
  const page = await apiFetch<RegionsPage>(
    `${scoped()}${browsePath}${qs(params as Record<string, unknown>)}`,
    {},
    signal,
  );
  return {
    ...page,
    items: page.items.map((raw) => ({
      ...raw,
      slots: mapCropSlots(
        raw as unknown as Record<string, unknown>,
        (raw.bbox_norm ?? [0, 0, 0, 0]) as XYXY,
      ),
    })),
  };
}

/** Region-clustering background-job snapshot. The one-click pipeline result also
 *  carries the FP-rebuild + auto-assign sub-steps, and may report a re-partition
 *  that was skipped to protect a fresh manual refine (TTL). */
export interface RegionClusterJob {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result:
    | ({
        status?: string;
        n_regions?: number;
        n_clusters?: number;
        assigned?: number;
        fp_centroids?: { status: string; n_members?: number; k?: number };
        auto_fp?: { status: string; n_moved?: number; threshold?: number };
      } & Record<string, unknown>)
    | null;
  error: string | null;
}

/** Launch the one-click region-clustering pipeline (background job — a large
 *  pool takes minutes): rebuild FP sub-centroids → auto-pull tight FP matches →
 *  re-partition the good regions. Returns immediately; poll getRegionClusterStatus for completion. */
export function clusterRegions(
  maxRank?: number,
  opts: { forceRepartition?: boolean; autoFpThreshold?: number } = {},
  signal?: AbortSignal,
): Promise<RegionClusterJob> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/cluster${qs({
      max_rank: maxRank,
      force_repartition: opts.forceRepartition,
      auto_fp_threshold: opts.autoFpThreshold,
    })}`,
    { method: 'POST' },
    signal,
  );
}

/** Poll the background region-clustering job. */
export function getRegionClusterStatus(signal?: AbortSignal): Promise<RegionClusterJob> {
  return apiFetch(`${scoped()}${REGION_BASE}/cluster/status`, {}, signal);
}

/** Per-bucket AHC refine over the region embeddings; writes region_cluster_subid. */
export function refineRegionCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  action: string;
}> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/clusters/refine/${clusterId}`,
    { method: 'POST' },
    signal,
  );
}

/** Region cluster cards (mirrors getClusters' Cluster shape). */
export function getRegionClusters(
  opts: { maxClusters?: number; perCluster?: number; maxRank?: number } = {},
  signal?: AbortSignal,
): Promise<{ clusters: Cluster[]; count: number }> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/clusters${qs({
      max_clusters: opts.maxClusters,
      per_cluster: opts.perCluster,
      max_rank: opts.maxRank,
    })}`,
    {},
    signal,
  );
}

/** Background FP-centroid build-job snapshot + persisted centroid metadata. */
export interface RegionFpCentroidJob {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result: { status: string; n_members: number; k: number } | null;
  error: string | null;
  centroids: {
    trained_at: string | null;
    k: number | null;
    n_members: number | null;
  } | null;
}

/** (Re)build the FP centroid store — sub-types the FP bucket (background job). */
export function buildRegionFpCentroids(
  signal?: AbortSignal,
): Promise<RegionFpCentroidJob> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/fp_centroids/build`,
    { method: 'POST' },
    signal,
  );
}

/** Poll the FP-centroid build job + read persisted centroid metadata. */
export function getRegionFpCentroidStatus(
  signal?: AbortSignal,
): Promise<RegionFpCentroidJob> {
  return apiFetch(`${scoped()}${REGION_BASE}/fp_centroids/status`, {}, signal);
}

export interface SuspectedFpItem extends RegionBrowseItem {
  suspected_fp_distance: number;
  nearest_fp_subid: string | null;
}

export interface SuspectedFpPage {
  items: SuspectedFpItem[];
  total: number;
  page: number;
  page_size: number;
  threshold?: number;
  centroids_built: boolean;
  trained_at?: string | null;
  message?: string;
}

/** Non-FP region crops ranked by similarity to the known FP centroids. */
export function getSuspectedFalsePositives(
  opts: { threshold?: number; page?: number; pageSize?: number } = {},
  signal?: AbortSignal,
): Promise<SuspectedFpPage> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/suspected_false_positives${qs({
      threshold: opts.threshold,
      page: opts.page,
      page_size: opts.pageSize,
    })}`,
    {},
    signal,
  );
}

// A region training-candidate `?mode=` value. Not a closed union: the
// modes are served (`GET {API_PREFIX}/training_cohorts`), and the server
// rejects one it doesn't know.
export type TrainingCohortMode = string;

export function getTrainingCandidates(
  mode: TrainingCohortMode,
  params: { page?: number; page_size?: number; class_id?: number } = {},
  signal?: AbortSignal,
): Promise<RegionsPage> {
  return apiFetch<RegionsPage>(
    `${scoped()}${REGION_BASE}/training_candidates${qs({ mode, ...params })}`,
    {},
    signal,
  );
}

export interface RegionStatusEntry {
  value: string;
  label: string;
  role: string;
  terminal: boolean;
  human_writable: boolean;
  clears_box: boolean;
  wants_reason: boolean;
}

/** per-box state styling/labels, distinct from the item-level
 *  `RegionStatusEntry` vocabulary above — a box's `state` is
 *  `proposed`/`accepted`/`rejected`/`false_positive`, never one of the
 *  item's `region_status` values. */
export type BoxStateTone = 'accepted' | 'proposed' | 'rejected' | 'neutral';

export interface BoxStateEntry {
  value: string;
  label: string;
  role: string;
  human_writable: boolean;
  exported: boolean;
  dashed: boolean;
  dim: boolean;
  badge: string | null;
  /**
   * Pending backend contract: the backend approved this
   * as a follow-up to the `box_states` vocabulary — the served
   * color/theme mapping for a box state, since `box_states` itself only
   * ever served `dashed`/`dim`/`badge` (styling flags, no color). Not in
   * the vendored OpenAPI snapshot yet (no `box_states` schema exists
   * there at all — `box_states` predates any contract-sync coverage);
   * remove this note (not widen it) once `npm run contract:sync` picks
   * it up. Optional so a pre-tone backend (or one that serves an
   * unrecognized value) renders neutral — see
   * `regionStatusesStore.boxStateTone()`.
   */
  tone?: BoxStateTone;
}

export interface RegionStatusesResponse {
  statuses: RegionStatusEntry[];
  confirm_status: string;
  reject_status: string;
  false_positive_status: string;
  /** served box-state vocabulary (`GET /regions/statuses`), absent
   *  on a backend that predates multi-box. */
  box_states?: BoxStateEntry[];
}

/** The deployment's region-status vocabulary (`GET {API_PREFIX}/regions/statuses`),
 *  meant to be loaded once by a store — see `$stores/regionStatuses.svelte`. */
export function getRegionStatuses(signal?: AbortSignal): Promise<RegionStatusesResponse> {
  return apiFetch<RegionStatusesResponse>(
    `${scoped()}${REGION_BASE}/statuses`,
    {},
    signal,
  );
}

/** What kind of writer a detector/source/chain-actor vocabulary entry names
 *  (`GET {API_PREFIX}/regions/vocabulary`). Forward-tolerant —
 *  an unrecognized role string is still carried through, it just falls
 *  back to the neutral chip color. */
export type RegionVocabularyRole =
  | 'detector'
  | 'segmenter'
  | 'ocr'
  | 'verifier'
  | 'human'
  | 'classifier'
  | 'proposal'
  | (string & {});

export interface RegionVocabularyEntry {
  id: string;
  label: string;
  role: RegionVocabularyRole;
  /** Only present on `detectors` entries — marks the values that can
   *  actually appear in stored `region_detector` (the detector filter's
   *  exact option list). */
  filterable?: boolean;
}

/** The active deployment's region-text validity rules (dq-region,
 *  2026-09-24) — `GET {API_PREFIX}/regions/vocabulary`'s `text_rules`, null
 *  without a region profile. Informational only: the frontend never
 *  re-implements these rules client-side, it just has somewhere to show
 *  them (e.g. next to a `region_text_vlm_invalid` badge). */
export interface RegionTextRules {
  uppercase: boolean;
  charset: string;
  len_min: number;
  len_max: number;
  format: string;
  reject_sequences: boolean;
  placeholders: string[];
  no_reading_words: string[];
  invalid_reasons: string[];
}

/** What kind of thing rejected a region candidate (OpenProcessor 3f1a11e,
 *  2026-09-24) — drives both the label lookup and the styling: a
 *  `model_verdict` is the verifier judging the box wrong, `automatic` is
 *  a geometry sanity gate, and `needs_human` means no verdict was given
 *  at all — it must never be worded as a rejection. */
export type RejectionReasonKind = 'model_verdict' | 'automatic' | 'needs_human';

/** One pipeline-written `region_rejection_reason` value
 *  (`GET {API_PREFIX}/regions/vocabulary`, OpenProcessor 3f1a11e). `match: 'exact'`
 *  entries match the stored value verbatim; `match: 'prefix'` entries
 *  match a stored-value prefix (e.g. `sanity_reject:`) with
 *  `label_template`'s `{detail}` filled from whatever follows the
 *  prefix. A stored value matching nothing here (older free-text human
 *  reasons) renders verbatim — see `regionVocabularyStore.rejectionReasonLabel`. */
export interface RejectionReasonEntry {
  id: string;
  label: string;
  kind: RejectionReasonKind;
  match: 'exact' | 'prefix';
  label_template: string | null;
}

export interface RegionVocabularyResponse {
  detectors: RegionVocabularyEntry[];
  region_sources: RegionVocabularyEntry[];
  chain_actors: RegionVocabularyEntry[];
  /** `region_text_choice` values (dq-region, 2026-09-24) — plain ids,
   *  e.g. `readers_agree` / `vlm_preferred` / `vlm_only` / `ocr_only` /
   *  `ocr_mode` / `vlm_invalid` / `no_valid_reading` / `human`. No
   *  served label yet; the UI titlecases the id as a placeholder (see
   *  `regionVocabularyStore.textChoiceLabel`). */
  text_choices: string[];
  /** The active profile's region-text validity rules, or `null` without
   *  a region profile. */
  text_rules: RegionTextRules | null;
  /** Labeled `region_rejection_reason` vocabulary (OpenProcessor 3f1a11e). */
  rejection_reasons: RejectionReasonEntry[];
  /** The active region profile (same value `/health` serves), or `null`
   *  when none is configured, in which case every list above is empty. */
  region_profile: ServedRegionProfile | null;
}

/** The deployment-configured detector/segmenter/verifier vocabulary
 *  (`GET {API_PREFIX}/regions/vocabulary`), built from the active region
 *  profile / ingest profiles / `OP_VLM_MODEL` — never a hardcoded model id.
 *  Meant to be loaded once by a store — see `$stores/regionVocabulary.svelte`. */
export async function getRegionVocabulary(
  signal?: AbortSignal,
): Promise<RegionVocabularyResponse> {
  const res = await apiFetch<Partial<RegionVocabularyResponse>>(
    `${scoped()}${REGION_BASE}/vocabulary`,
    {},
    signal,
  );
  return {
    detectors: res.detectors ?? [],
    region_sources: res.region_sources ?? [],
    chain_actors: res.chain_actors ?? [],
    text_choices: res.text_choices ?? [],
    text_rules: res.text_rules ?? null,
    rejection_reasons: res.rejection_reasons ?? [],
    region_profile: res.region_profile ?? null,
  };
}
