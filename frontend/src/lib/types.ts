/**
 * TypeScript types mirroring the OpenSearch indexes defined in Wave 1d
 * (the items, classes and clusters indexes) and the OpenProcessor
 * `{API_PREFIX}/...` endpoint responses defined in Phase 2D.
 *
 * These shapes are forward-tolerant: we accept extra fields silently so a
 * server-side schema bump won't break the app.
 */

import type { SlotData, SlotKey } from '$lib/annotations/types';
import type { EmbeddingState, ItemFilterQuery } from '$lib/types_itemFilter';
import type { ReprocessRequest } from '$lib/types_import';
import type { ItemAssist } from '$lib/types_assist';

/** Who wrote a crop's current label. Same vocabulary as `class_source`
 *  (curation_api_contract.md "class_source values"): `human*`, the fixed
 *  VLM writer values, or an ingest detector's config-derived value — so
 *  it stays an open string. */
export type LabelSource =
  'human' | 'human_confirmed' | 'vlm' | 'vlm_human_confirmed' | (string & {});

export interface BBoxNorm {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

/** One OCR text line on the item crop (wire `ItemTextLine` —
 *  `box_norm` is already normalized in the item-crop frame, `rel_height`
 *  is line height / crop height). Populates `Crop.item_text_lines`. */
export interface ItemTextLine {
  text: string | null;
  confidence: number | null;
  box_norm: number[] | null;
  rel_height: number | null;
}

export interface Crop {
  id: string;
  source_image_path: string;
  /** The source image's id (wire `image_id`); undefined on a legacy item
   *  that serves none. Targets an image Reprocess. */
  image_id?: string;
  source_image_sha256?: string;
  bbox_norm: BBoxNorm;
  class_id: number | null;
  class_name: string | null;
  /** The detector's own class for this item, kept apart from `class_*`
   *  (which a VLM or a human may overwrite). Null when the detector gave
   *  none, and on items stored before it was kept. */
  detector_class_name?: string | null;
  detector_class_id?: number | null;
  detector_confidence?: number | null;
  /** Where the class assignment came from — `human*`, the fixed VLM
   *  writer values (`vlm`, `vlm_unmatched`, …), or an ingest detector's
   *  config-derived value (`{primary}_proposal`, `{secondary}_model`, …).
   *  Drives the per-source filter chip in the cluster view. */
  class_source: string | null;
  label_source: LabelSource;
  /** `class_validated OR region_validated` (wire's `label_validated`,
   *  `wire.py:105`). "Anything on this crop was validated" — NOT the
   *  same as the class label being trustworthy. Read `class_validated`
   *  for any class-label display/eligibility check; keep this only for
   *  an intentional "anything validated" meaning (G2). */
  label_validated: boolean;
  /** Whether the *class* assignment specifically was human-validated.
   *  Independent of region_validated — a crop can be
   *  `label_validated=true` (region validated) while `class_validated`
   *  is still false, e.g. a model-proposed class label on a region a human
   *  confirmed has no visible region. */
  class_validated: boolean;
  label_confidence: number | null;
  /** Confidence of whoever set the *class label* (wire `class_confidence`,
   *  dq-queues cutover 2026-09-24) — distinct from `label_confidence`
   *  (the detector/classifier score). VLM high/medium/low map to
   *  0.92/0.70/0.40 server-side; classifier labels carry their own
   *  score; human labels are null. */
  class_confidence?: number | null;
  /** `'vlm' | 'model' | null` — which pipeline produced `class_confidence`. */
  class_confidence_source?: string | null;
  /** The VLM's verbatim class answer, even when unmatched/unapplied. */
  vlm_raw_class?: string | null;
  vlm_class_attempted_at?: string | null;
  /** `no_answer | no_match | invalid_index | unparseable | null`. */
  vlm_class_empty_reason?: string | null;
  /** The VLM's registry-matched class for this crop when it did not
   *  auto-apply it (wire `vlm_proposed_class_id`/`_name`). Drives the
   *  accept-suggestion chip and the `G`/`Shift+Enter` keys. */
  vlm_suggested_class_id?: number | null;
  vlm_suggested_class_name?: string | null;
  /** The VLM's categorical confidence: `high` | `medium` | `low`. */
  vlm_confidence?: string | null;
  /** Cluster + similarity-to-centroid (0..1), served verbatim as
   *  `cluster_similarity` — no client 1−cosine-distance computation. */
  cluster_id: number | null;
  similarity_to_centroid: number | null;
  /** Distance to this crop's cluster centroid, null when the item has
   *  left the cluster it's measured against (dq-queues cutover). */
  cluster_distance?: number | null;
  /** Nearest cluster centroid id, only meaningful together with
   *  cluster_distance/similarity_to_centroid/cluster_is_core. */
  cluster_nearest_id?: number | null;
  /** Server-computed: this crop's `cluster_similarity` is at/above the
   *  cluster response's `core_similarity_min`. Null when the backend
   *  hasn't computed a similarity for this crop yet. Drives the
   *  cluster-detail "core" cut line — never a client-side 0.75 constant. */
  cluster_is_core?: boolean | null;
  /** AHC sub-cluster id (string, e.g. "47a"), populated only after
   *  refine ran. Cleared by the backend whenever cluster_id changes
   *  (move / batch_label / class_merge / residual recluster) — the
   *  subid is meaningful only inside its origin cluster. */
  cluster_subid: string | null;
  /** Per-slot capability data, keyed by SlotKey, produced by mapRawCrop
   *  via mapCropSlots(). Only slots with actual evidence on the row
   *  appear (slotIsPresent). Optional because many test fixtures
   *  construct Crop literals directly; read it through
   *  `slotOf(crop, spec)` (annotations/cropSlots.ts), never `?.[...]`
   *  by hand, so the accessor can be tightened later. */
  slots?: Record<SlotKey, SlotData>;
  // -- Class provenance -----------------------------------------
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  /** Ingest source tag (`GET {API_PREFIX}/review/{tab}?source=` /
   *  `GET {API_PREFIX}/crops?source=`); see `CropFilter.source` below. */
  source?: string | null;
  /** The backend's confirmable suggestion for this crop — what
   *  Enter/Confirm assigns. Served on every crop-shaped item, not just
   *  review-queue rows (undo/restore, `/crops/{id}`, diverse selection
   *  hydration all carry it too), so it lives on `Crop` rather than only
   *  `ReviewItem`. */
  proposed_class_id: number | null;
  proposed_class_name: string | null;
  test_holdout: boolean;
  // -- Exclude / Ignore (G7) ----------------------------------------------
  /** True when the crop is in the excluded/"Ignored" bucket
   *  (`cluster_id=-2`) — set by `POST /crops/batch_exclude`, cleared by
   *  `POST /crops/batch_unexclude`. */
  class_excluded?: boolean;
  /** Free-text tag recorded at exclude time (`'ignore'`, `'blurry'`, …). */
  excluded_reason?: string | null;
  excluded_at?: string | null;
  /** Item-crop-frame OCR text lines (G-series item text; wire
   *  `item_text_lines`, `ItemTextLine[]`). Empty array when the item has
   *  no detected text, never absent. */
  item_text_lines?: ItemTextLine[];
  outlier_flagged?: boolean;
  outlier_score?: number | null;
  // -- Primary-subject rank + blur quality -------------------------------
  /** 1 = largest-area crop in its source photo, 2 = second, … Drives the
   *  "Largest / Largest + 2nd" subject toggle. */
  crop_rank_in_image?: number | null;
  /** Normalized bbox area (w*h in [0,1]). */
  crop_area_norm?: number | null;
  /** Crop/full Laplacian ratio (higher = clearer).
   *  Drives the clarity slider. */
  blur_lap_ratio?: number | null;
  /** Raw detector confidence (recorded even below the 0.75 floor). */
  classifier_raw_confidence?: number | null;
  /** Coarse COCO class hint for coco_yolo11_proposal blind spots. */
  proposal_name?: string | null;
  // -- Curation scores (Phase 3 review-queue strategies, 2026-09) --------
  // Provenance quad mirroring the region_detector/region_detector_version
  // pattern (docs/curation-strategy-plan-2026-09.md §4). `mistakenness`
  // is the only curation score that cleared the full validation gate as
  // of OpenProcessor docs/design/curation_scores.md — representativeness/
  // atypicality/uncertainty_entropy are pre-existing fields (cluster
  // distance, probe entropy) exposed as named sorts, not new score
  // fields, so they don't need their own Crop fields. Everything else
  // in the plan (uniqueness, near-dup) is still shadow/pending and has
  // no field here yet — added when/if it clears validation.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
  /**
   * F8 D1 (OpenProcessor 51b05d7): the probe's opinion on this item.
   * `probe_in_scope`: null = not scored yet, true = the item's class is
   * one the probe knows (a real opinion), false = outside the probe's
   * classes (no opinion; `probe_disagreement` is then null).
   * `probe_disagreement`: true/false only when the probe has an opinion.
   */
  probe_disagreement?: boolean | null;
  probe_in_scope?: boolean | null;
  probe_model_version?: string | null;
  /**
   * OpenProcessor main 8990ede: `true` only when the probe disagrees AND
   * the item is in scope AND `probe_pred_confidence` cleared the
   * backend's own threshold (`OP_PROBE_ACTIONABLE_MIN_CONFIDENCE`,
   * echoed read-only as `ProbeStatusResponse.actionable_min_confidence`
   * — no client-side threshold anywhere). `false` when scored but not
   * actionable (including every non-disagreeing item); `null` when not
   * scored at all.
   */
  probe_actionable?: boolean | null;
  /** the VLM endpoint (`name@revision`), model and prompt pack that
   *  produced this item's VLM answer. Null when none ran. */
  vlm_endpoint?: string | null;
  vlm_model?: string | null;
  vlm_prompt_pack?: string | null;
  /** Suggestions from the newest assist run (attributes, the VLM's class
   *  answer). Never a label: a locked item carries them too. */
  assist?: ItemAssist | null;
  /** a human-locked label the pipeline must not overwrite. */
  label_locked?: boolean;
  /** ids of the dataset imports that carried this item. */
  import_ids?: string[];
  dataset_split?: string | null;
  imported_at?: string | null;
  proposed_by_import?: string | null;
  on_negative_frame?: boolean;
  import_standalone_region?: boolean;
  proposal_chain?: string[];
  /** P4: where a combined item came from (null on a non-combined one). */
  origin_project?: string | null;
  origin_item_id?: string | null;
  origin_image_id?: string | null;
  origin_split?: string | null;
  combine_conflict?: boolean;
  combine_conflict_origins?: string[];
  combine_merged_origins?: string[];
  /** v0.4.0: whether the item has a vector; `null` on an item written
   *  before the field existed. */
  embedding_state?: EmbeddingState | null;
  /** v0.4.0 open vocabulary: the prompt, set and set revision that found
   *  this item, its image-normalised mask outline, and why the region
   *  gate skipped it. `null` when not served. */
  source_prompt?: string | null;
  open_vocab_set?: string | null;
  open_vocab_revision?: number | null;
  mask_polygon?: number[][] | null;
  region_gate_skip?: string | null;
  updated_at: string;
}

/** Class clusters mirror class_id (0..80); candidate clusters land at
 *  10000+ from the residual AHC pass; unassigned is < 0. The region view
 *  also emits "false_positive" for the permanent FP bucket (-100). The
 *  backend derives this from cluster_id; the frontend NEVER recomputes it. */
export type ClusterKind = 'class' | 'candidate' | 'unassigned' | 'false_positive';

/** Backend's purity banding (`purity_thresholds`: pure_min 0.85 / mixed_min
 *  0.6) — served per cluster, never recomputed against a client constant. */
export type PurityTier = 'pure' | 'mixed' | 'noisy';

export interface Cluster {
  id: number;
  /** Backend-derived: "class" | "candidate" | "unassigned". */
  cluster_kind: ClusterKind;
  /** Items with >=1 box in the cluster. For a region cluster, this
   *  is DISTINCT from `box_count` below — an item can have more than one
   *  box in the same cluster. */
  size: number;
  /** boxes
   *  (rows) in the cluster, region clusters only. Absent on an item
   *  cluster or an older backend. */
  box_count?: number | null;
  /** class_validated=true count. */
  validated_count: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  /** Served `label_purity` — the dominant class's share of the
   *  LABELLED members. Not the geometry `purity`. */
  dominant_pct: number | null;
  /** Served member counts behind `dominant_pct` (cluster-scoped, include
   *  any test-holdout members). */
  dominant_count?: number | null;
  labelled_count?: number | null;
  /** DQ-M2 fix (dq-queues cutover, 2026-09-24): nearest-centroid geometry
   *  purity — the share of `purity_n` measured members whose nearest
   *  cluster centroid is this cluster's own. Independent of labels, so a
   *  class cluster is no longer 1.0 by construction. 0..1, null when no
   *  member has been measured. */
  purity: number | null;
  /** How many members `purity` was measured over — show alongside
   *  `purity` (it's noisy at low n). */
  purity_n?: number | null;
  /** Always `'nearest_centroid'` today — served, never hardcoded. */
  purity_basis?: string | null;
  /** Server-banded purity — drives the card's pure/mixed/noisy badge and
   *  border color. Null only for the client-only "cluster card lookup
   *  failed" stub in `getCluster`. */
  purity_tier: PurityTier | null;
  /** Largest-class share among labelled members (the pre-cutover
   *  label-based "purity" — 1.0 for a class cluster by construction).
   *  `promotable` is gated on this, not the geometry-based `purity`. */
  label_purity?: number | null;
  /** #61: served `label_agreement` (same value as `label_purity`) — what
   *  the UI reads for label quality; `purity` stays geometry-only. */
  label_agreement?: number | null;
  /** Share of this cluster's members that carry any label at all. */
  labelled_share?: number | null;
  /** Server's auto-promote eligibility for this cluster (purity +
   *  member-count + labelled-share gate — `purity_thresholds` on the
   *  `{API_PREFIX}/clusters` response). */
  promotable: boolean;
  /** `{API_PREFIX}/clusters`' `core_similarity_min` (0.75), copied onto every
   *  cluster in that response so a consumer doesn't need the raw list
   *  response's top-level field. Null when the lookup that would have
   *  supplied it failed (see `getCluster`'s stub). */
  core_similarity_min: number | null;
  /** True when no member has a class_name — pure candidate. */
  is_unlabeled: boolean;
  representative_crop_ids: string[]; // up to 4
  // Optional explicit thumbnail URLs (one per representative_crop_ids
  // entry, same order). Used by the synthetic region card so its
  // tiles show region close-ups (API_PREFIX-relative /crops/{id}/region_thumbnail) rather
  // than the default item-crop thumbnail. Regular clusters leave
  // this undefined; the grid then falls back to getThumbUrl().
  representative_thumb_urls?: string[];
  has_subclusters: boolean;
  /** Distinct cluster_subid count from the backend. */
  n_subclusters: number;
  centroid_sha?: string;
  updated_at: string | null;
  /** Set only on the client-built region inventory entry pinned
   *  atop `/clusters` (M4, docs/design/interactive-pass-2026-09-24.md) —
   *  it is not a real cluster (no purity, no cluster_kind), so the grid
   *  must not draw a pure/mixed/noisy badge for it or key it against a
   *  real cluster id. Absent (not false) on every server-served Cluster. */
  isSlotCard?: boolean;
  /** The slot's served display name (e.g. its tab label), set only on
   *  that inventory entry (F8 D7): the card is titled by it, not by the
   *  raw class id. */
  slotDisplayName?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  /** Set by `{API_PREFIX}/review/{tab}` when the requested `?sort=` couldn't be
   *  honored (e.g. the field isn't backfilled yet) and the server fell
   *  back to the default ordering. Rendered as an inline note, never a
   *  toast — this isn't an error, just a degraded request. Absent on
   *  every other endpoint and on a request that didn't ask for a sort. */
  sort_fallback_reason?: string | null;
  /** The sort id `{API_PREFIX}/review/{tab}` actually used — the tab's own
   *  default when no `?sort=` was sent, or the requested id when it was
   *  honored. Shown in the StrategyBar summary chip so the operator can
   *  see what's actually ordering the queue, not just what they last
   *  picked. Absent on endpoints that don't
   *  report it. */
  sort_applied?: string | null;
  /** Set by `{API_PREFIX}/review/{tab}` when `total === 0` — the server's
   *  own explanation for why this queue is empty right now (e.g. "no
   *  probe predictions — run a probe"), distinct from and more direct
   *  than `sort_fallback_reason` (OpenProcessor #36 item 9). Absent when
   *  the queue isn't empty. */
  empty_reason?: string | null;
  /** Provenance for a pool-scale overlay ordering (curation-strategy plan
   *  Phase 4 — currently only `{API_PREFIX}/crops?order=diverse`): which
   *  overlay/version produced this selection, and how large the pool it
   *  drew from was. `n_pool` lets the UI note when diverse selection is
   *  sampling a much larger cohort than what's on screen, rather than
   *  silently truncating. All three are optional/tolerant — absent on
   *  every other order value and on any backend that hasn't shipped this
   *  yet (Phase 0/3 backend, or `OP_SELECT_DIVERSE_ENABLED` off). */
  order_method?: string | null;
  order_version?: string | null;
  n_pool?: number | null;
  /** m7 (2026-09-24 interactive pass): `{API_PREFIX}/clusters`' own
   *  `purity_thresholds` — the border-color legend on `/clusters` used
   *  to hardcode "≥80% / ≥60% / <60%", which drifted from the server's
   *  real `pure_min`/`mixed_min` (0.85/0.6). Absent on every other
   *  endpoint. */
  purity_thresholds?: { pure_min: number; mixed_min: number } | null;
  /** v0.4.0 `GET {API_PREFIX}/crops`: matching items with no vector (null
   *  when the route did not compute it). Absent on every other endpoint. */
  n_unembedded?: number | null;
  /** v0.4.0 ordered `GET /crops` views: the request that would embed the
   *  unembedded items, sent only when `n_unembedded` is > 0. */
  suggested_reprocess?: ReprocessRequest | null;
  /** v0.4.0 `GET {API_PREFIX}/search/text`: items in the search scope a
   *  semantic search cannot reach because they have no vector. */
  unembedded_in_scope?: number | null;
}

/** `GET {API_PREFIX}/crops` query: the shared item filter
 *  (`ItemFilterQuery`: class names, confidence and area bands, `max_rank`,
 *  origin, embedding and review state) plus this route's own keys. */
export interface CropFilter extends ItemFilterQuery {
  cluster_id?: number | null;
  label_source?: LabelSource;
  /** Original label source (where the class came from). Matches the
   *  {API_PREFIX}/crops?class_source= query param. */
  class_source?: string;
  label_validated?: boolean;
  /** `GET {API_PREFIX}/crops?source=`: filter by ingest source tag. */
  source?: string;
  sort?: string;
  limit?: number;
  page?: number;
  // -- Primary-subject filters -------------------------------------------
  /** Clarity slider: keep crops with blur_lap_ratio >= this (null-safe). */
  min_blur_ratio?: number | null;
  /** Mine the low-confidence pool: classifier_raw_confidence < this OR no detector box. */
  classifier_conf_lt?: number | null;
  /** Crops permanently dismissed from every /review queue via {API_PREFIX}/crops/{id}/review_dismiss. */
  review_dismissed?: boolean;
  /** Include excluded ("Ignored") crops — off by default server-side, so
   *  the "Ignored" bucket view is the only caller that sets this. */
  include_excluded?: boolean;
  /** Free-text OCR search over `item_text_lines` (G-series item text).
   *  A query with no letter/digit 400s server-side — see `getCrops`. */
  item_text?: string;
}

/** `GET {API_PREFIX}/crops/{id}/history` — the item's class history, oldest
 *  first. Each entry is the item's class state *before* one write, plus
 *  who made it (`writer`) and when (`at`). Untyped on the wire
 *  (`additionalProperties: true`) beyond `writer`/`at`, so every other key
 *  is read tolerantly. */
export interface CropHistoryEntry {
  writer: string | null;
  at: string | null;
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string | null;
  label_source?: string | null;
  confidence?: number | null;
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeler?: string | null;
  class_labeled_at?: string | null;
  class_validated?: boolean | null;
  cluster_id?: number | null;
  cluster_subid?: string | null;
  review_dismissed_at?: string | null;
  review_dismissed_by?: string | null;
}

export interface CropHistoryResponse {
  crop_id: string;
  entries: CropHistoryEntry[];
}

/** `GET {API_PREFIX}/crops/{id}/image` — the shared source image plus every
 *  item cropped from it (siblings, including the requested crop itself). */
export interface CropImageMeta {
  image_id: string;
  image_path: string;
  width: number | null;
  height: number | null;
  source: string | null;
  indexed_at: string | null;
}

export interface CropContextResponse {
  image: CropImageMeta;
  items: Crop[];
}

export interface ClusterFilter extends Omit<
  ItemFilterQuery,
  'open_vocab_set' | 'source_prompt'
> {
  /** DQ-M4 (docs/design/data-quality-pass-2026-09-24.md): restrict to
   *  exactly one cluster, so its representatives can be fetched
   *  individually in the frontend's own (client-sorted) display order —
   *  the server has no `cluster_ids`/batch-by-id param, only a
   *  size-desc-ordered `offset`/`limit` window (see
   *  representatives_offset/_limit below), which is why representatives
   *  used to fill in server order regardless of what sort the operator
   *  picked. */
  cluster_id?: number | null;
  /** DQ-M4 live-verified follow-up: the backend's `offset`/`limit`
   *  representatives window rejects `limit=0` (422, "Input should be
   *  greater than or equal to 1") — confirmed live against real data,
   *  not just the vendored OpenAPI's `minimum: 1`. To genuinely skip
   *  representative computation on the main card-list call (superseded
   *  entirely by per-card display-order fetches, see
   *  displayOrderRepresentatives.ts), set this to 0 instead — `0`
   *  representative crops per card is valid (`minimum: 0`) and just as
   *  cheap, without touching offset/limit at all. */
  per_cluster?: number;
  min_size?: number;
  sort?: 'purity_asc' | 'purity_desc' | 'size_desc' | 'size_asc' | 'dominant_class';
  page?: number;
  page_size?: number;
  // Primary-subject grid filters: card stats reflect only crops that pass.
  max_rank?: number | null;
  min_blur_ratio?: number | null;
  class_source?: string | null;
  /** D-4 (docs/design/curation_query_performance_audit.md): which window
   *  of the (already-fully-returned, size-desc-ordered) card list gets a
   *  populated `representatives` array — cards outside the window come
   *  back with `representatives: []`, not omitted. Independent of
   *  `page`/`page_size` above, which the endpoint ignores entirely (every
   *  card up to `max_clusters` is always returned in one response); the
   *  caller windows representatives to whatever's actually visible. */
  representatives_offset?: number;
  representatives_limit?: number;
}

export interface BulkLabelConflict {
  crop_id: string;
  current_source: string | null;
}

export interface BulkLabelResult {
  // Matches the FastAPI handler at src/routers/curation/crops.py:
  //   batch_label_crops -> { updated: int, updated_ids: [...], conflicts: [...] }.
  // `move` (POST {API_PREFIX}/crops/move) returns the same shape.
  updated: number;
  updated_ids: string[];
  conflicts: BulkLabelConflict[];
}

/** `POST {API_PREFIX}/crops/label/undo_batch` response — batch form of
 *  `undoCropLabel`. Each crop is restored independently to its own state
 *  before its most recent human class write; `items` carries only the
 *  crops actually restored. */
export interface CropUndoBatchResult {
  items: Crop[];
  undone: number;
  nothing_to_undo: string[];
  conflicts: string[];
  not_found: string[];
}

/** `POST {API_PREFIX}/crops/region/undo_batch` response (M6) — same shape
 *  as `CropUndoBatchResult`, kept as its own type since it's a distinct
 *  wire contract (region writes, not class writes), not because the
 *  fields differ. */
export type CropRegionUndoBatchResult = CropUndoBatchResult;

export interface ToastMessage {
  id: string;
  kind: 'info' | 'success' | 'warn' | 'error';
  text: string;
  ttl_ms?: number;
}

export interface KeyboardShortcut {
  /** Every combo bound to this action/registration, in document order. */
  keys: string[];
  scope: string;
  description: string;
}

/** One undoable user action — a single confirmed write, however many
 *  crops it touched. What it restores is the backend's business: Z calls
 *  `POST /crops/{id}/label/undo` (one id) or
 *  `POST /crops/label/undo_batch` (several) and renders the result(s). */
export interface UndoEntry {
  crop_ids: string[];
  at: number;
  /**
   * Which backend undo route this entry reverses (M6,
   * docs/design/interactive-pass-2026-09-24.md). Defaults to `'label'`
   * when absent (every entry pushed before this field existed). Kept as
   * ONE ring buffer with a kind tag, not a separate region/vlm_dismiss
   * stack, so Z on `/clusters/[id]` — where a label write and a
   * Reject-VLM (`vlm_dismiss`) can interleave in the same session —
   * always reverses whatever the operator *actually did last*,
   * chronologically, rather than "the last label write" while silently
   * skipping a more recent dismiss. A page-scoped ignore/un-ignore
   * history (`clusterController`'s `lastExcludedIds`) stays its own
   * thing on purpose: it's bound to its own `X`/`U` keys, never `Z`, so
   * there's no ordering question to get right by sharing a stack.
   */
  kind?: 'label' | 'region' | 'vlm_dismiss' | 'exclude' | 'unexclude';
}
