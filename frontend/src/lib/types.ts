/**
 * TypeScript types mirroring the OpenSearch indexes defined in Wave 1d
 * (op_classes, op_vehicle_crops, op_clusters, etc.) and the openprocessor
 * `/curation/...` endpoint responses defined in Phase 2D.
 *
 * These shapes are forward-tolerant: we accept extra fields silently so a
 * server-side schema bump won't break the app.
 */

export type LabelSource =
  | 'v6_original_label'
  | 'hdd_user_label'
  | 'model_suggestion'
  | 'gemma_suggestion'
  | 'human'
  | 'human_confirmed'
  | 'cluster_propagation'
  | 'ensemble'
  | 'unknown';

export type ClassSource = 'registry' | 'derived' | 'imported';

export interface OpClass {
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
}

/** Payload for `POST /curation/classes`. */
export interface OpClassCreate {
  name: string;
  group: string;
  notes?: string;
}

/** Payload for `PUT /curation/classes/{id}`. Any subset of fields may be supplied. */
export interface OpClassUpdate {
  name?: string;
  group?: string;
  /** Pass an empty string to clear the binding, or omit to leave unchanged. */
  hotkey_letter?: string;
}

/** Payload for `POST /curation/classes/merge`. */
export interface OpClassMerge {
  source_id: number;
  target_id: number;
}

/** Server response from `GET /curation/export/status`. */
export interface OpExportStatus {
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

/** Server response from `POST /curation/export/yolo`. */
export interface OpExportResult {
  status: string;
  job_id?: string | null;
  message?: string | null;
}

/** Server response from `POST /curation/export/lpr` (single-class plate dataset). */
export interface OpLprExportResult {
  status: string;
  export_dir: string;
  manifest_path: string;
  data_yaml_path: string;
  dataset_sha: string;
  split_counts: Record<string, number>;
  image_count: number;
  positive_images?: number | null;
  background_images?: number | null;
  false_positive_background_images?: number | null;
  positives_zero_warning?: boolean | null;
  dedup?: Record<string, unknown> | null;
  image_mode?: 'whole_frame' | 'vehicle_crop' | null;
  img_max_side?: number | null;
  current_symlink?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

/** Server response from `GET /curation/export/lpr/status`. */
export interface OpLprExportStatus {
  status: string;
  last_run: string | null;
  export_dir?: string | null;
  dataset_sha?: string | null;
  positive_images?: number | null;
  background_images?: number | null;
  false_positive_background_images?: number | null;
  split_counts?: Record<string, number> | null;
}

/** One materialized dataset version from `GET /curation/export/datasets`. */
export interface OpDataset {
  kind: 'lpr' | 'vehicles';
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

/** Server response from `GET /curation/export/datasets`. */
export interface OpDatasetList {
  datasets: OpDataset[];
  count: number;
}

/** Server response from `POST /curation/test_holdout/freeze`. */
export interface OpTestHoldoutFreezeResult {
  n_frozen: number;
  n_classes_covered: number;
  test_holdout_sha: string;
  per_class_counts: Record<string, number>;
}

/** Server response from `GET /curation/test_holdout/stats`. */
export interface OpTestHoldoutStats {
  total: number;
  by_class: Array<{ key: number; doc_count: number }>;
}

export interface BBoxNorm {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface OpCrop {
  id: string;
  source_image_path: string;
  source_image_sha256?: string;
  bbox_norm: BBoxNorm;
  class_id: number | null;
  class_name: string | null;
  /** Where the class assignment came from: 'v6_model' / 'gemma' /
   *  'human' / 'v6_low_conf' / 'gemma_unmatched' /
   *  'coco_yolo11_proposal' / 'gemma_new_class_pending'.
   *  Drives the per-source filter chip in the cluster view. */
  class_source: string | null;
  label_source: LabelSource;
  label_validated: boolean;
  label_confidence: number | null;
  /** Gemma's most-recent suggestion if any. */
  gemma_suggested_class_id?: number | null;
  gemma_suggested_class_name?: string | null;
  gemma_suggested_confidence?: number | null;
  /** Cluster + similarity-to-centroid (0..1). */
  cluster_id: number | null;
  similarity_to_centroid: number | null;
  /** AHC sub-cluster id (string, e.g. "47a"), populated only after
   *  refine ran. Cleared by the backend whenever cluster_id changes
   *  (move / batch_label / class_merge / residual recluster) — the
   *  subid is meaningful only inside its origin cluster. */
  cluster_subid: string | null;
  /** Plate sub-bbox normalized to source image. */
  plate_bbox_norm?: BBoxNorm | null;
  /** Detector confidence for the plate proposal (0..1). */
  plate_score?: number | null;
  /** State machine value: 'detected' | 'no_plate_visible' |
   *  'verify_rejected' | 'no_plate_box' | 'pending_verify' | 'human_confirmed' */
  plate_status?: string | null;
  plate_verified?: boolean | null;
  // -- Plate provenance (Wave 1 of plate-integrity overhaul) -------------
  /** Which detector produced the stored bbox. */
  plate_detector?: string | null;
  plate_detector_version?: string | null;
  /** Every detector attempted on this crop with a hit/miss tag,
   *  e.g. ['lpr_nanov11_640:miss', 'sam3:hit', 'gemma:verify_ok']. */
  plate_detector_chain?: string[] | null;
  /** Frame the bbox is in. Always 'source' on current writes. */
  plate_bbox_frame?: string | null;
  plate_detected_at?: string | null;
  plate_verifier?: string | null;
  plate_verifier_version?: string | null;
  plate_verified_at?: string | null;
  plate_rejection_reason?: string | null;
  plate_visible?: boolean | null;
  // -- Plate OCR (Wave 2b) -----------------------------------------------
  plate_text?: string | null;
  plate_text_raw?: string | null;
  plate_text_source?: string | null;
  plate_text_confidence?: number | null;
  plate_text_engine_version?: string | null;
  // -- Class provenance (Wave 1) -----------------------------------------
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  // -- Client-side derived flag (set in mapRawCrop) ----------------------
  /** True when the projected plate-bbox shape fails the same envelope
   *  the server-side sanity gate uses. Defense in depth. */
  plate_shape_warning?: boolean;
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
  /** legacy_sorter v1.1.9 crop/full Laplacian ratio (higher = clearer).
   *  Drives the clarity slider. */
  blur_lap_ratio?: number | null;
  /** Raw v6 detection confidence (recorded even below the 0.75 floor). */
  v6_raw_confidence?: number | null;
  /** Coarse COCO class hint for coco_yolo11_proposal blind spots. */
  coco_proposal_name?: string | null;
  // -- Curation scores (Phase 3 review-queue strategies, 2026-09) --------
  // Provenance quad mirroring the plate_detector/plate_detector_version
  // pattern (docs/curation-strategy-plan-2026-09.md §4). `mistakenness`
  // is the only curation score that cleared the full validation gate as
  // of openprocessor/docs/design/curation_scores.md — representativeness/
  // atypicality/uncertainty_entropy are pre-existing fields (cluster
  // distance, probe entropy) exposed as named sorts, not new score
  // fields, so they don't need their own OpCrop fields. Everything else
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

export interface OpCluster {
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
  // tiles show plate close-ups (/curation/crops/{id}/plate_thumbnail) rather
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

export interface OpStats {
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
  }>;
}

export interface OpHealth {
  ok: boolean;
  components: {
    triton: 'ok' | 'degraded' | 'down';
    opensearch: 'ok' | 'degraded' | 'down';
    gemma_openwebui: 'ok' | 'degraded' | 'down';
    class_registry_sha: string | null;
  };
  timestamp: string;
}

export type ReviewTab =
  | 'all'
  | 'mismatches'
  | 'gemma_low_conf'
  | 'outliers'
  | 'uncertainty'
  | 'model_disagreements'
  | 'plates'
  | 'primary_low_conf'
  | 'coco_blind_spots';

export interface ReviewItem extends OpCrop {
  reason: string;
  proposed_class_id: number | null;
  proposed_class_name: string | null;
  // Phase 5 — populated for the model_disagreements tab. The new model's
  // prediction for this crop, plus how confident it was. Lets the
  // labeler render "human said X, model said Y" inline.
  probe_pred_class?: string | null;
  probe_pred_entropy?: number | null;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  /** Set by `/curation/review/{tab}` when the requested `?sort=` couldn't be
   *  honored (e.g. the field isn't backfilled yet) and the server fell
   *  back to the default ordering. Rendered as an inline note, never a
   *  toast — this isn't an error, just a degraded request. Absent on
   *  every other endpoint and on a request that didn't ask for a sort. */
  sort_fallback_reason?: string | null;
}

export interface CropFilter {
  class_id?: number | null;
  cluster_id?: number | null;
  label_source?: LabelSource;
  /** Original label source (where the class came from). Matches the
   *  /curation/crops?class_source= query param. */
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
  /** Mine the low-confidence pool: v6_raw_confidence < this OR no v6 box. */
  v6_conf_lt?: number | null;
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
  // Matches the FastAPI handler at src/routers/legacy/op_crops.py:
  //   batch_label_crops -> { updated: int, conflicts: [...] }.
  updated: number;
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

export type OpModelStatus = 'ready' | 'not_ready' | 'unavailable';
export type OpModelKind = 'triton' | 'external';

export interface OpModel {
  name: string;
  friendly_name: string;
  role: string;
  kind: OpModelKind;
  model_type: string;
  status: OpModelStatus;
  version: string | null;
  inference_count: number | null;
  exec_count: number | null;
  inference_failed: number | null;
  avg_latency_ms: number | null;
  last_error: string | null;
  endpoint: string | null;
}

export interface OpModelsStatus {
  models: OpModel[];
}

export interface UndoEntry {
  crop_id: string;
  prior_class_id: number | null;
  prior_label_source: LabelSource;
  prior_validated: boolean;
  at: number;
}
