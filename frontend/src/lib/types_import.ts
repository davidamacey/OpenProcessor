/**
 * Wire types for OpenProcessor W10: labeled-dataset import and Reprocess
 * (`any_domain_plan.md` §7.12, W10.4, W10.11–W10.14). Every route is
 * project-scoped (`{prefix}/datasets/...`, `{prefix}/reprocess...`).
 *
 * Built against the frozen spec before the backend lands W10, so nothing
 * here is pinned to the vendored OpenAPI yet. Where the spec names a model
 * without its fields, the type is the literal reading recorded in
 * docs/design/w10-import-reprocess-ui-plan-2026-09-27.md §8 (question N).
 */

/** `{id, label}` vocabulary entry served by `GET /datasets/formats`. */
export interface LabeledId {
  id: string;
  label: string;
}

/** A vocabulary entry that also carries served description copy. */
export interface DescribedId extends LabeledId {
  description?: string | null;
}

export type DatasetIssueSeverity = 'error' | 'warning' | 'info';

/** One `/datasets/formats` issue-catalog row (W10.4 `ISSUE_CATALOG`). */
export interface DatasetIssueCatalogEntry {
  code: string;
  severity: DatasetIssueSeverity;
  blocking: boolean;
  bypassable: boolean;
  label: string;
}

/** delta 20 (question 2): the served Reprocess vocabulary. */
export interface ReprocessVocabulary {
  scopes: Array<DescribedId & { unit?: string | null }>;
  region_modes: DescribedId[];
  lock_rule: string;
}

/** `GET /datasets/formats` (§7.12 JSON). */
export interface DatasetFormatsResponse {
  formats: LabeledId[];
  processing: DescribedId[];
  parents: DescribedId[];
  label_trust: DescribedId[];
  mapping_actions: LabeledId[];
  match_kinds: LabeledId[];
  issues: DatasetIssueCatalogEntry[];
  upload: { max_bytes: number; max_files: number; accepted: string[] };
  status_labels: Record<string, string>;
  /** delta 20 (question 2); absent ⇒ no Reprocess surface. */
  reprocess?: ReprocessVocabulary | null;
}

export type DatasetFormat = 'auto' | 'yolo' | 'coco' | 'openprocessor_export';
export type MappingAction = 'map' | 'create' | 'skip' | 'region';
export type MapTargetKind = 'item' | 'region' | 'skip';

export interface CocoAnnotationFile {
  path: string;
  images_dir: string;
  split: string | null;
}

export interface DatasetSource {
  path: string;
  format: DatasetFormat | string;
  coco_annotations?: CocoAnnotationFile[];
}

/** One explicit mapping row (W10.5). Names only — never an index. */
export interface ClassMappingEntry {
  dataset_class: string;
  action: MappingAction | string;
  class_id?: number | null;
  new_class_name?: string | null;
  new_class_group?: string | null;
}

/**
 * `DatasetImportOptions` (W10.14). Every field is optional on the wire:
 * the client sends only what the operator set, so the server's default
 * applies to the rest (question 6).
 */
export interface DatasetImportOptions {
  processing?: string;
  label_trust?: string;
  parents?: string;
  freeze_test_split?: boolean | null;
  missing_label?: string;
  region_negatives?: boolean;
  region_containment?: number;
  name?: string;
  source_tag?: string;
  force?: boolean;
}

/** `DatasetPreviewRequest` — no `project` field (delta 15, W10.3). */
export interface DatasetPreviewRequest {
  source: DatasetSource;
  mapping: ClassMappingEntry[];
  accept_suggestions: boolean;
  options: DatasetImportOptions;
}

export interface DatasetImportRequest extends DatasetPreviewRequest {
  expected_import_key?: string | null;
}

export interface DatasetIssueSample {
  file: string;
  line: number | null;
  detail: Record<string, unknown>;
}

export interface DatasetIssue {
  code: string;
  severity: DatasetIssueSeverity;
  blocking: boolean;
  bypassable: boolean;
  message: string;
  count: number;
  samples: DatasetIssueSample[];
}

export interface DatasetSplitRow {
  split: string;
  images: number;
  labeled: number;
  negatives: number;
  unlabeled: number;
  boxes: number;
}

export interface DatasetClassRef {
  class_id: number | null;
  class_name: string | null;
}

export interface MappingSuggestion {
  action: MappingAction | string;
  class_id: number | null;
  class_name: string | null;
  match: string;
}

/** A resolved mapping target (the job's `mapping[]`, the preview's
 *  `classes[].resolved`; question 5). */
export interface ResolvedMapTarget {
  dataset_class?: string;
  kind: MapTargetKind | string;
  class_id: number | null;
  class_name: string | null;
}

export interface DatasetClassRow {
  dataset_class: string;
  /** The dataset's own id — shown as a label only, never compared. */
  dataset_id: number | null;
  boxes: number;
  images: number;
  /** An OP export's source-project registry id — a label only. */
  source_class_id: number | null;
  index_would_have_mapped_to: DatasetClassRef | null;
  suggestion: MappingSuggestion | null;
  resolved: ResolvedMapTarget | null;
}

export interface OpExportInfo {
  dataset_kind: string | null;
  box_source: string | null;
  image_mode: string | null;
  manifest_dataset_sha: string | null;
  recomputed_dataset_sha: string | null;
  frozen_test_sha: string | null;
  test_frozen: { present: boolean; verified: boolean; test_label_sha: string | null };
  stratum_map: { present: boolean; entries: number };
  source_classes?: unknown;
}

export interface DatasetRegionInfo {
  profile: { name: string; revision: number | null };
  region_class_name: string;
  parent_classes: string[];
  parents_mode: string;
  standalone_boxes: number;
}

export interface DatasetPreview {
  project: string;
  format: string;
  root: string;
  source_sha: string;
  import_key: string;
  op_export: OpExportInfo | null;
  splits: DatasetSplitRow[];
  totals: {
    images: number;
    boxes: number;
    images_already_indexed: number;
    images_to_ingest: number;
  };
  classes: DatasetClassRow[];
  region: DatasetRegionInfo | null;
  issues: DatasetIssue[];
  blocking: boolean;
  force_allowed: boolean;
  estimate: { detector_images: number; embeddings: number };
}

export type DatasetImportStatus =
  | 'queued'
  | 'running'
  | 'paused_backpressure'
  | 'completed'
  | 'completed_with_errors'
  | 'failed'
  | 'cancelled'
  | 'interrupted'
  | 'undoing'
  | 'undone';

export interface DatasetImportProgress {
  images_total: number;
  images_done: number;
  images_failed: number;
  chunks_total: number;
  chunks_done: number;
  images_per_s: number | null;
  eta_s: number | null;
}

export interface DatasetImportReport {
  images_created: number;
  images_reused: number;
  items_created: number;
  items_updated: number;
  items_noop: number;
  labels_written: number;
  boxes_written: number;
  standalone_regions: number;
  negatives: number;
  unlabeled: number;
  parents_detected: number;
  proposals_created: number;
  proposals_merged: number;
  holdout_frozen: number;
  label_conflicts_locked: number;
  disagreements: { counts: Record<string, number>; samples: unknown[] };
}

/** W10.11 `next_steps` entry (question 10). */
export interface NextStep {
  action: string;
  method: string;
  path: string;
  reason: string;
}

export interface DatasetUndoReport {
  import_id: string;
  dry_run: boolean;
  items_deleted: number;
  items_restored: number;
  items_kept_human_edited: number;
  class_labels_removed: number;
  boxes_removed: number;
  boxes_kept_human_edited: number;
  proposals_deleted: number;
  holdout_flags_cleared: number;
  images_deleted: number;
  images_kept: number;
  classes_deprecated: string[];
  samples?: Record<string, unknown>;
}

export interface DatasetUndoRequest {
  dry_run: boolean;
  remove_images: boolean;
  deprecate_created_classes: boolean;
}

export interface DatasetImportJob {
  project: string;
  import_id: string;
  import_key: string;
  name: string | null;
  status: DatasetImportStatus | string;
  reused: boolean;
  progress: DatasetImportProgress;
  waiting_for: string | null;
  report: DatasetImportReport;
  mapping: ResolvedMapTarget[];
  options: DatasetImportOptions;
  source: { format: string; root: string; source_sha: string; op_export: unknown };
  issues_summary: DatasetIssue[];
  undo: DatasetUndoReport | null;
  next_steps: NextStep[];
  started_at: string | null;
  updated_at: string | null;
  finished_at: string | null;
  /** Seconds until the next poll; null once terminal. */
  poll_after_s: number | null;
  labels?: { status?: Record<string, string> };
  /** Question 7: the served failure reason on a `failed` job. */
  error?: { code?: string | null; message: string } | null;
}

/** Question 4: a page of `{items, total, page, page_size}`. */
export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export type DatasetImportList = Page<DatasetImportJob>;
export type DatasetIssuePage = Page<DatasetIssue>;

/** Question 4: one ledger line (W10.11). */
export interface DatasetImportEntry {
  source_stem: string;
  rel_path: string;
  image_id: string | null;
  image_created: boolean;
  split: string | null;
  label_state: string;
  status: string;
  error_kind: string | null;
}

export type DatasetImportEntryPage = Page<DatasetImportEntry>;

export interface DatasetUploadResponse {
  upload_id: string;
  dataset_path: string;
  bytes: number;
  files: number;
}

/** `ConfigErrorDetail` (§7.1) with the W10 optional fields (W10.14). */
export interface DatasetErrorDetail {
  error: string;
  message: string;
  issues?: DatasetIssue[] | null;
  unmapped?: string[] | null;
  import_id?: string | null;
  limit?: number | null;
}

// -- Reprocess (W10.13) --------------------------------------------------

export type ReprocessScope = 'detect' | 'region' | 'vlm' | 'embed';

export interface ReprocessTargets {
  image_ids?: string[];
  crop_ids?: string[];
  filter?: Record<string, unknown>;
}

export interface ReprocessRequest {
  targets: ReprocessTargets;
  scopes: string[];
  region_mode?: string;
  dry_run: boolean;
}

export interface ReprocessOneRequest {
  scopes: string[];
  region_mode?: string;
  dry_run?: boolean;
}

export interface ReprocessScopeResult {
  scope: string;
  selected: number;
  locked_skipped: number;
  queued: number;
  breakdown?: Array<{ detector?: string | null; reason?: string | null; count: number }>;
}

/** Question 3. */
export interface ReprocessJob {
  job_id: string;
  status: string;
  scopes?: ReprocessScopeResult[];
  error?: string | null;
  poll_after_s?: number | null;
  labels?: { status?: Record<string, string> };
}

export interface ReprocessResponse<T = unknown> {
  dry_run: boolean;
  scopes: ReprocessScopeResult[];
  job: ReprocessJob | null;
  /** Post-write item docs (single-target routes); `reprocessCrop` maps
   *  them through `mapRawCrop`. */
  items: T[];
  /** delta 20 (question 2): served summary copy. */
  message?: string | null;
}
