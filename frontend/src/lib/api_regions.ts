/**
 * Region browse, region cluster cards, the FP centroid build, training
 * candidates, and the region status / vocabulary reads (`/regions*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { ClusterRef } from '$contracts/ts/itemWire';
import type { SlotData, SlotKey, XYXY } from '$lib/annotations/types';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import type { Cluster } from '$lib/types';
import type { CursorPage } from '$lib/types_cursor';
import type { ServedRegionProfile } from '$lib/types_health';

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
  /** The box's cluster in the serving region partition (`ClusterRef`). */
  cluster?: ClusterRef | null;
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

export interface RegionsPage extends Omit<CursorPage<RegionBrowseItem>, 'items'> {
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
  'open_vocab_set' | 'source_prompt' | 'cluster_uid'
> {
  cursor?: string;
  limit?: number;
  /** Rows whose box is in this cluster of the serving region partition. */
  cluster_uid?: string;
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

/** One card of `GET /regions/clusters`: boxes grouped by `cluster_uid`. A card's `size`
 *  is the items with a box in the cluster, `box_count` the boxes. */
interface RawRegionCluster {
  cluster_uid: string;
  size: number;
  box_count: number;
  validated_count: number;
  representative_crop_ids: string[];
  representative_thumb_urls: string[];
}

/** Region cluster cards in the shape the grid renders. */
export async function getRegionClusters(
  opts: { maxClusters?: number; perCluster?: number; maxRank?: number } = {},
  signal?: AbortSignal,
): Promise<{ clusters: Cluster[]; count: number }> {
  const raw = await apiFetch<{ clusters: RawRegionCluster[]; count: number }>(
    `${scoped()}${REGION_BASE}/clusters${qs({
      max_clusters: opts.maxClusters,
      per_cluster: opts.perCluster,
      max_rank: opts.maxRank,
    })}`,
    {},
    signal,
  );
  return {
    count: raw.count,
    clusters: raw.clusters.map((c) => ({
      uid: c.cluster_uid,
      size: c.size,
      box_count: c.box_count,
      validated_count: c.validated_count,
      dominant_class_name: null,
      dominant_pct: null,
      is_unlabeled: true,
      representative_crop_ids: c.representative_crop_ids,
      representative_thumb_urls: c.representative_thumb_urls,
      updated_at: null,
    })),
  };
}

/** (Re)build the FP centroid store, sub-typing the false-positive boxes: queued as
 *  a job (202; 409 `job_busy` while one is active). Follow `job_id` with `followJob`. */
export function buildRegionFpCentroids(
  signal?: AbortSignal,
): Promise<{ job_id: string; kind: string }> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/fp_centroids/build`,
    { method: 'POST' },
    signal,
  );
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
  params: { cursor?: string; limit?: number; class_id?: number } = {},
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
