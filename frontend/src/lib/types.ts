/**
 * TypeScript types mirroring the OpenSearch indexes defined in Wave 1d
 * (the items, classes and clusters indexes) and the openprocessor
 * `{API_PREFIX}/...` endpoint responses defined in Phase 2D.
 *
 * These shapes are forward-tolerant: we accept extra fields silently so a
 * server-side schema bump won't break the app.
 */

import type { SlotKey, SlotData } from './annotations/types';

/** Who wrote a crop's current label. Same vocabulary as `class_source`
 *  (curation_api_contract.md "class_source values"): `human*`, the fixed
 *  VLM writer values, or an ingest detector's config-derived value — so
 *  it stays an open string. */
export type LabelSource =
  | 'human'
  | 'human_confirmed'
  | 'vlm'
  | 'vlm_human_confirmed'
  | (string & {});

export type ClassSource = 'registry' | 'derived' | 'imported';

export interface RegistryClass {
  id: number;
  name: string;
  group: string | null;
  count: number;
  validated_count: number;
  /**
   * Size of the FAISS cluster bucket whose id == this class id. Includes
   * unlabeled candidates the operator hasn't triaged yet — i.e. matches
   * what the operator sees when they open /clusters/{id}. The sidebar
   * chip displays this; the cluster-page banner breaks out validated /
   * labeled / total side-by-side.
   */
  cluster_size: number;
  /** Single-character keyboard shortcut, persisted in the registry. */
  hotkey_letter?: string | null;
  added_at: string;
  /** Hex color hint or null. */
  color?: string | null;
  /** True when the class has been merged into another and should be hidden by default. */
  deprecated?: boolean;
  /** Server-computed adequacy tier from `GET {API_PREFIX}/classes`
   *  (`block` | `warn` | `ok`, against the served `thresholds`). Never
   *  recomputed client-side from `validated_count`. */
  adequacy?: string;
}

/** `thresholds` served on `GET {API_PREFIX}/classes`, `GET {API_PREFIX}/stats/classes`
 *  and `{API_PREFIX}/train/preflight` — the single source of truth for the
 *  adequacy tiers, augmentation target range and test-holdout minimum. */
export interface ClassThresholds {
  block_below: number;
  warn_below: number;
  min_test_per_class: number;
  aug_target_min: number;
  aug_target_max: number;
}

/** `GET {API_PREFIX}/classes` response envelope. */
export interface ClassesResponse {
  classes: RegistryClass[];
  thresholds: ClassThresholds;
  /** Every single-character combo reserved for a labeling action —
   *  core keys plus every registered queue-capable slot's keymap. */
  reserved_hotkeys: string[];
}

/** Payload for `POST {API_PREFIX}/classes`. */
export interface RegistryClassCreate {
  name: string;
  group: string;
  notes?: string;
}

/** Payload for `PUT {API_PREFIX}/classes/{id}`. Any subset of fields may be supplied. */
export interface RegistryClassUpdate {
  name?: string;
  group?: string;
  /** Pass an empty string to clear the binding, or omit to leave unchanged. */
  hotkey_letter?: string;
}

/** Payload for `POST {API_PREFIX}/classes/merge`. */
export interface RegistryClassMerge {
  source_id: number;
  target_id: number;
}

/** `POST {API_PREFIX}/classes/merge?dry_run=true` response — reports
 *  counts and writes nothing. A real merge 409s when `holdout_blocking > 0`. */
export interface ClassMergeDryRun {
  dry_run: true;
  source_id: number;
  target_id: number;
  would_relabel: number;
  would_unvalidate: number;
  holdout_blocking: number;
  blocked: boolean;
}

/** Server response from `GET {API_PREFIX}/export/status`. */
export interface ExportStatus {
  /** 'idle' | 'running' | 'success' | 'failed' | 'unknown'. */
  status: string;
  last_run: string | null;
  /** 0..1 for active jobs. */
  progress?: number;
  job_id?: string | null;
  /** On success: directory the manifest was written to. */
  export_dir?: string | null;
  /** On failure: the error message. */
  error?: string | null;
  /** Optional human-readable detail. */
  message?: string | null;
}

/** Server response from `POST {API_PREFIX}/export/yolo`. */
export interface ExportResult {
  status: string;
  job_id?: string | null;
  message?: string | null;
}

/** Server response from `POST {API_PREFIX}/export/single_class`. */
export interface SingleClassExportResult {
  status: string;
  export_dir: string;
  version_tag?: string | null;
  manifest_path: string;
  data_yaml_path: string;
  dataset_sha: string;
  frozen_test_sha?: string | null;
  split_counts: Record<string, number>;
  image_count: number;
  class_count?: number | null;
  positive_images?: number | null;
  background_images?: number | null;
  positives_zero_warning?: boolean | null;
  current_symlink?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

/** Server response from `GET {API_PREFIX}/export/single_class/status`. */
export interface SingleClassExportStatus {
  status: 'idle' | 'unknown' | 'success' | string;
  profile_name: string;
  last_run: string | null;
  export_dir?: string | null;
  dataset_kind?: string | null;
  dataset_sha?: string | null;
  frozen_test_sha?: string | null;
  class_count?: number | null;
  class_names?: string[] | null;
  image_count?: number | null;
  positive_images?: number | null;
  background_images?: number | null;
  false_positive_background_images?: number | null;
  positives_zero_warning?: boolean | null;
  split_counts?: Record<string, number> | null;
}

/** One materialized dataset version from `GET {API_PREFIX}/export/datasets`. */
export interface ExportDataset {
  /** The `/methods` export-axis id that produced it: `yolo` (multi-class)
   *  or `single_class`. */
  kind: string;
  /** Narrowed exports only — which single-class profile wrote it. */
  profile_name?: string | null;
  export_dir: string;
  version_tag: string;
  image_count?: number | null;
  split_counts?: Record<string, number> | null;
  dataset_sha?: string | null;
  exported_at?: string | null;
  image_mode?: string | null;
  img_max_side?: number | null;
  sampling?: string | null;
  max_positive_images?: number | null;
  max_images?: number | null;
  class_count?: number | null;
  is_current: boolean;
}

/** Server response from `GET {API_PREFIX}/export/datasets`. */
export interface ExportDatasetList {
  datasets: ExportDataset[];
  count: number;
}

/** Server response from `POST {API_PREFIX}/test_holdout/freeze`. */
export interface TestHoldoutFreezeResult {
  n_frozen: number;
  n_classes_covered: number;
  test_holdout_sha: string;
  per_class_counts: Record<string, number>;
}

/** Server response from `GET {API_PREFIX}/test_holdout/stats`. */
export interface TestHoldoutStats {
  total: number;
  by_class: Array<{ key: number; doc_count: number; deficient?: boolean }>;
  /** The same class-adequacy threshold served on `/classes`/`/stats/classes`
   *  — the frontend's "below 5 test crops" copy reads this, never a
   *  hardcoded 5. */
  min_test_per_class?: number;
}

export interface BBoxNorm {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface Crop {
  id: string;
  source_image_path: string;
  source_image_sha256?: string;
  bbox_norm: BBoxNorm;
  class_id: number | null;
  class_name: string | null;
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
   *  is still false, e.g. a v6-model class label on a region a human
   *  confirmed has no visible plate. */
  class_validated: boolean;
  label_confidence: number | null;
  /** The VLM's registry-matched class for this crop when it did not
   *  auto-apply it (wire `vlm_proposed_class_id`/`_name`). Drives the
   *  accept-suggestion chip and the `G`/`Shift+Enter` keys. */
  vlm_suggested_class_id?: number | null;
  vlm_suggested_class_name?: string | null;
  /** The VLM's categorical confidence: `high` | `medium` | `low`. */
  vlm_confidence?: string | null;
  /** Cluster + similarity-to-centroid (0..1). */
  cluster_id: number | null;
  similarity_to_centroid: number | null;
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
  // -- Class provenance (Wave 1) -----------------------------------------
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  hdd_source?: string | null;
  test_holdout: boolean;
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
  /** Raw v6 detection confidence (recorded even below the 0.75 floor). */
  classifier_raw_confidence?: number | null;
  /** Coarse COCO class hint for coco_yolo11_proposal blind spots. */
  proposal_name?: string | null;
  // -- Curation scores (Phase 3 review-queue strategies, 2026-09) --------
  // Provenance quad mirroring the region_detector/region_detector_version
  // pattern (docs/curation-strategy-plan-2026-09.md §4). `mistakenness`
  // is the only curation score that cleared the full validation gate as
  // of openprocessor/docs/design/curation_scores.md — representativeness/
  // atypicality/uncertainty_entropy are pre-existing fields (cluster
  // distance, probe entropy) exposed as named sorts, not new score
  // fields, so they don't need their own Crop fields. Everything else
  // in the plan (uniqueness, near-dup) is still shadow/pending and has
  // no field here yet — added when/if it clears validation.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
  updated_at: string;
}

/** Class clusters mirror class_id (0..80); candidate clusters land at
 *  10000+ from the residual AHC pass; unassigned is < 0. The plate view
 *  also emits "false_positive" for the permanent FP bucket (-100). The
 *  backend derives this from cluster_id; the frontend NEVER recomputes it. */
export type ClusterKind = 'class' | 'candidate' | 'unassigned' | 'false_positive';

export interface Cluster {
  id: number;
  /** Backend-derived: "class" | "candidate" | "unassigned". */
  cluster_kind: ClusterKind;
  size: number;
  /** class_validated=true count. */
  validated_count: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  dominant_pct: number | null;
  purity: number | null; // 0..1, null when no labelled members
  /** True when no member has a class_name — pure candidate. */
  is_unlabeled: boolean;
  representative_crop_ids: string[]; // up to 4
  // Optional explicit thumbnail URLs (one per representative_crop_ids
  // entry, same order). Used by the synthetic license_plate card so its
  // tiles show plate close-ups (API_PREFIX-relative /crops/{id}/region_thumbnail) rather
  // than the default vehicle-crop thumbnail. Regular clusters leave
  // this undefined; the grid then falls back to getThumbUrl().
  representative_thumb_urls?: string[];
  has_subclusters: boolean;
  /** Distinct cluster_subid count from the backend. */
  n_subclusters: number;
  /** Legacy alias for n_subclusters — kept until callers migrate. */
  sub_clusters?: number;
  centroid_sha?: string;
  updated_at: string | null;
}

export interface StatsSummary {
  /** Set when `/stats/dataset` failed; the totals below are then zeros,
   *  not real counts. `per_class` comes from `/stats/classes` and stays. */
  dataset_error: string | null;
  total_crops: number;
  validated_crops: number;
  test_holdout_crops: number;
  ingestion: {
    images_processed: number;
    images_pending: number;
    last_run_at: string | null;
  };
  per_class: Array<{
    class_id: number;
    class_name: string;
    count: number;
    validated_count: number;
    /** Server-computed adequacy tier (`block`/`warn`/`ok`) — see
     *  `RegistryClass.adequacy`. */
    adequacy?: string;
    /** Server-computed YOLO augmentation target for this class. */
    aug_target?: number;
    /** `aug_target - validated_count`, served directly. */
    aug_gap?: number;
  }>;
  /** Served alongside `per_class` on `/stats/classes` — same shape as
   *  `ClassesResponse.thresholds`. */
  thresholds?: ClassThresholds;
}

/** `GET {API_PREFIX}/health`. `degraded` means a non-critical
 *  dependency (e.g. the VLM) is down; labeling still works. */
export interface ApiHealth {
  status: 'ok' | 'degraded' | 'down';
  triton?: { reachable: boolean; detail?: string };
  opensearch?: { reachable: boolean; indexes?: Record<string, boolean> };
  vlm?: { reachable: boolean; model?: string | null };
  registry?: { path?: string; exists?: boolean; mtime?: string | null };
}

// 'outliers' was retired from the UI in the 2026-09 tab consolidation
// (3 live rows, functionally identical to the `atypicality` sort already
// offered via the strategy bar) — its backend {API_PREFIX}/review/outliers query
// is untouched, but nothing in the frontend calls it anymore, so the
// literal is gone from this union too. 'mismatches' / 'vlm_low_conf' /
// 'primary_low_conf' are no longer top-level UI tabs but still real
// values here — they're driven by the All-tab preset chips instead (see
// $lib/reviewTabs.ts's resolveEffectiveTab).
// P2.8b (docs/genericization-plan-2026-09-13.md §9.5): 'plates' left this
// union — a queue-capable slot's tab id is now the structural
// `slot:${SlotKey}` template (SlotReviewTab) instead of a hand-maintained
// literal per slot. 'plates' survives only as the `urlId` bookmark value
// (see `$lib/reviewTabs.ts`'s `tabFromUrlId`), not as an internal id.
export type SlotReviewTab = `slot:${string}`;

export type CoreReviewTab =
  | 'all'
  | 'mismatches'
  | 'vlm_low_conf'
  | 'uncertainty'
  | 'model_disagreements'
  | 'primary_low_conf'
  | 'coco_blind_spots';

export type ReviewTab = CoreReviewTab | SlotReviewTab;

export interface ReviewItem extends Crop {
  reason: string;
  proposed_class_id: number | null;
  proposed_class_name: string | null;
  // Phase 5 — populated for the model_disagreements tab. The new model's
  // prediction for this crop, plus how confident it was. Lets the
  // labeler render "human said X, model said Y" inline.
  probe_pred_class?: string | null;
  probe_pred_entropy?: number | null;
}

/**
 * A crop returned by `GET {API_PREFIX}/search/text` (P2-14 semantic text search).
 * Same shape as `Crop` plus the query-similarity score. Field named
 * `similarity_score` to match this repo's existing `<name>_score`
 * convention for per-crop curation scores (`mistakenness_score`), not a
 * bare `score` — kept distinct from those since it's query-relative, not
 * an absolute crop property.
 */
export interface SearchCrop extends Crop {
  similarity_score: number;
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
}

/**
 * `POST {API_PREFIX}/select/diverse`'s scope object (P2-10, `/review`'s diverse
 * overlay — docs/CLAUDE.md's curation-strategy-selector-bar section).
 * Distinct from `/clusters/[id]`'s `GET {API_PREFIX}/crops?order=diverse&k=N` —
 * that path is a small, synchronous, cluster-scoped selection; this one
 * scopes to a review-tab cohort (`review_tab` reuses the backend's
 * existing tab-query builder) which can be pool-scale (the `all` tab is
 * ~320k crops), hence the job/poll contract below. `filters` only
 * supports term/terms filters server-side (`class_id`, `hdd_source`) —
 * NOT `conf_min`/`conf_max`/`min_blur_ratio`/`max_rank`/plate `text`.
 */
export interface SelectDiverseScope {
  cluster_id?: number | null;
  review_tab?: string | null;
  filters?: Record<string, unknown>;
}

/**
 * Immediate (200) result of `POST {API_PREFIX}/select/diverse` for a small pool.
 * Forward-tolerant per this file's convention — an unrecognized `method`
 * string is still carried through as-is.
 */
export interface DiverseSelection {
  crop_ids: string[];
  method: string;
  version: string;
  n_pool: number;
}

/** Background-job state for a large-pool diverse selection (202 path),
 *  polled via `GET {API_PREFIX}/select/status`. CONFIRMED against the real backend
 *  (`selection/job.py`'s `_JobState`, 2026-09-12): `status` is one of
 *  `'idle' | 'running' | 'completed' | 'failed' | 'cancelled'`, and the
 *  finished selection is nested under `result` (the same
 *  `{crop_ids, method, version, n_pool}` shape the sync 200 path returns
 *  directly) — NOT flattened onto the job status object. `status` values
 *  this build doesn't recognize are carried through as-is rather than
 *  normalized — callers should treat anything other than `'running'` as
 *  "stop polling." */
export interface SelectJobStatus {
  job_id?: string | null;
  status: string;
  result?: DiverseSelection | null;
  error?: string | null;
}

export interface CropFilter {
  class_id?: number | null;
  cluster_id?: number | null;
  label_source?: LabelSource;
  /** Original label source (where the class came from). Matches the
   *  {API_PREFIX}/crops?class_source= query param. */
  class_source?: string;
  label_validated?: boolean;
  hdd_source?: string;
  conf_min?: number;
  conf_max?: number;
  sort?: string;
  limit?: number;
  page?: number;
  // -- Primary-subject filters -------------------------------------------
  /** Keep only crops with crop_rank_in_image <= max_rank (1 or 2). */
  max_rank?: number | null;
  /** Clarity slider: keep crops with blur_lap_ratio >= this (null-safe). */
  min_blur_ratio?: number | null;
  /** Mine the low-confidence pool: classifier_raw_confidence < this OR no v6 box. */
  classifier_conf_lt?: number | null;
  /** Crops permanently dismissed from every /review queue via {API_PREFIX}/crops/{id}/review_dismiss. */
  review_dismissed?: boolean;
}

export interface ClusterFilter {
  class_id?: number | null;
  min_size?: number;
  sort?: 'purity_asc' | 'purity_desc' | 'size_desc' | 'size_asc' | 'dominant_class';
  page?: number;
  page_size?: number;
  // Primary-subject grid filters: card stats reflect only crops that pass.
  max_rank?: number | null;
  min_blur_ratio?: number | null;
  class_source?: string | null;
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

export interface ToastMessage {
  id: string;
  kind: 'info' | 'success' | 'warn' | 'error';
  text: string;
  ttl_ms?: number;
}

export interface KeyboardShortcut {
  key: string;
  scope: string;
  description: string;
}

export type ModelStatus = 'ready' | 'not_ready' | 'unavailable';
export type ModelKind = 'triton' | 'external';

export interface ModelInfo {
  name: string;
  friendly_name: string;
  role: string;
  kind: ModelKind;
  model_type: string;
  status: ModelStatus;
  version: string | null;
  inference_count: number | null;
  exec_count: number | null;
  inference_failed: number | null;
  avg_latency_ms: number | null;
  last_error: string | null;
  endpoint: string | null;
  /**
   * Follow-up gap 2 (docs/design/audit-remediation-plan-2026-09.md
   * Appendix D item 3, 2026-09-11): `DELETE {API_PREFIX}/models/{name}` guard
   * flags, mirrored from the same checks the endpoint enforces
   * server-side so the UI never has to re-derive them (and can't drift
   * out of sync with the real guard).
   */
  /** Region-detection pipeline model (backend `_is_region_protected_model`) — never unloadable through the UI, no override. */
  is_region_protected?: boolean;
  /** ACTIVE_VEHICLE_MODEL or another core pipeline model — unload requires force=true. */
  requires_force_to_unload?: boolean;
  /** Present (with job_id/version) only for models promoted through this pipeline. */
  job_id?: string | null;
  promoted_at?: string | null;
}

export interface ModelsStatus {
  models: ModelInfo[];
}

export interface UnloadModelResponse {
  triton_name: string;
  triton_unloaded: boolean;
  directory_removed: boolean;
  forced: boolean;
  warning: string | null;
}

/** One undoable human class write. What it restores is the backend's
 *  business: Z calls `POST /crops/{id}/label/undo` and renders the result. */
export interface UndoEntry {
  crop_id: string;
  at: number;
}
