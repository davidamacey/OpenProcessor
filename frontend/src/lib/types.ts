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
  /** AHC sub-cluster id, populated only after refine ran. */
  sub_cluster_id?: number | null;
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
  updated_at: string;
}

export interface OpCluster {
  id: number;
  size: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  dominant_pct: number | null;
  purity: number | null; // 0..1, null when not yet computed
  representative_crop_ids: string[]; // up to 4
  has_subclusters: boolean;
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
  | 'plates';

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
}

export interface CropFilter {
  class_id?: number | null;
  cluster_id?: number | null;
  label_source?: LabelSource;
  label_validated?: boolean;
  hdd_source?: string;
  conf_min?: number;
  conf_max?: number;
  sort?: string;
  limit?: number;
  page?: number;
}

export interface ClusterFilter {
  class_id?: number | null;
  min_size?: number;
  sort?: 'purity_asc' | 'purity_desc' | 'size_desc' | 'size_asc' | 'dominant_class';
  page?: number;
  page_size?: number;
}

export interface BulkLabelResult {
  affected: number;
  failed: string[];
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
