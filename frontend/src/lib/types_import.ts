/**
 * Wire types for OpenProcessor W10: labeled-dataset import and Reprocess
 * (`any_domain_plan.md` §7.12, W10.4, W10.11-W10.14). Every route is
 * project-scoped (`{prefix}/datasets/...`, `{prefix}/reprocess...`).
 *
 * Pinned field-for-field to the vendored OpenAPI (OpenProcessor f582aa05)
 * by `contract/datasetsContract.test.ts`.
 */
import type {
  EmbeddingState,
  ItemOrigin,
  ReviewStatus,
  SelectionSample,
} from '$lib/types_itemFilter';

/** `LabeledChoice`: a served vocabulary entry (`value` is the wire id). */
export interface LabeledChoice {
  value: string;
  label: string;
  description?: string;
}

/** `DatasetFormatInfo`: one dataset layout the importer reads. */
export interface DatasetFormatInfo {
  format: string;
  label: string;
}

export type DatasetIssueSeverity = 'error' | 'warning' | 'info';

/** One `/datasets/formats` issue-catalog row (W10.4 `ISSUE_CATALOG`). */
export interface DatasetIssueCatalogEntry {
  code: string;
  severity: string;
  blocking: boolean;
  bypassable: boolean;
  label: string;
}

/** `DatasetUploadLimits`. */
export interface DatasetUploadLimits {
  max_bytes: number;
  max_files: number;
  ttl_hours: number;
  preview_max_files: number;
}

/** `GET /datasets/formats`. */
export interface DatasetFormatsResponse {
  formats: DatasetFormatInfo[];
  issues: DatasetIssueCatalogEntry[];
  mapping_actions: LabeledChoice[];
  match_kinds: LabeledChoice[];
  processing_modes: LabeledChoice[];
  parents_modes: LabeledChoice[];
  trust_levels: LabeledChoice[];
  upload_limits: DatasetUploadLimits;
  status_labels: Record<string, string>;
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
  /** Unique within the response (`code[:subject]`, `#2` on a repeat). */
  id: string;
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

/** `IndexWouldHaveMapped`: what an index-based reader would have used. */
export interface DatasetClassRef {
  class_id: number;
  class_name: string;
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
  dataset_class: string;
  kind: MapTargetKind | string;
  class_id: number | null;
  class_name: string | null;
  /** True when the import created this class. */
  created?: boolean;
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
  /** Dataset class names merged into this row, when the source had case
   *  or spelling variants. */
  merged_from?: string[];
  suggestion: MappingSuggestion;
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
  /** Untyped `object` in the contract; today `{name, revision}`. */
  profile: { name: string; revision: number | null };
  region_class_name: string;
  parent_classes: string[];
  parents_mode: string;
  standalone_boxes: number;
}

export interface DatasetTotals {
  images: number;
  boxes: number;
  images_already_indexed: number;
  images_to_ingest: number;
}

export interface DatasetEstimate {
  detector_images: number;
  embeddings: number;
}

export interface DatasetPreview {
  project: string;
  format: string;
  root: string;
  source_sha: string;
  import_key: string;
  op_export: OpExportInfo | null;
  splits: DatasetSplitRow[];
  totals: DatasetTotals;
  classes: DatasetClassRow[];
  region: DatasetRegionInfo | null;
  issues: DatasetIssue[];
  blocking: boolean;
  force_allowed: boolean;
  estimate: DatasetEstimate;
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
  images_failed: number;
  images_skipped: number;
  items_reconciled_removed: number;
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
  items_kept_shared: number;
  items_reinstated: number;
  class_labels_removed: number;
  boxes_removed: number;
  boxes_kept_human_edited: number;
  proposals_deleted: number;
  holdout_flags_cleared: number;
  images_deleted: number;
  images_kept: number;
  classes_deprecated: string[];
  samples?: Record<string, unknown[]>;
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
  name: string;
  status: DatasetImportStatus | string;
  reused: boolean;
  progress: DatasetImportProgress;
  waiting_for: string | null;
  report: DatasetImportReport;
  mapping: ResolvedMapTarget[];
  options: DatasetImportOptions;
  source: { format?: string | null; root?: string | null; source_sha?: string | null };
  issues_summary: DatasetIssue[];
  undo: DatasetUndoReport | null;
  next_steps: NextStep[];
  started_at: string | null;
  updated_at: string | null;
  finished_at: string | null;
  /** Seconds until the next poll; null once terminal. */
  poll_after_s: number | null;
  labels?: Record<string, Record<string, string>>;
  /** The served failure reason on a `failed` job. */
  error?: string | null;
}

/** Question 4: a page of `{items, total, page, page_size}`. */
export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export type DatasetImportList = Page<DatasetImportJob>;
/** `DatasetIssuePage.items` is an untyped object list in the contract;
 *  the served rows are `DatasetIssue`-shaped. */
export type DatasetIssuePage = Page<DatasetIssue>;

/** Question 4: one ledger line (W10.11). */
export interface DatasetImportEntry {
  source_stem: string;
  rel_path: string;
  image_id?: string | null;
  image_path?: string | null;
  image_created?: boolean | null;
  split?: string | null;
  label_state?: string | null;
  status: string;
  error_kind?: string | null;
  boxes?: Record<string, unknown>[];
  items?: Record<string, unknown>[];
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

/** The ids the contract enumerates (`ReprocessOneRequest.scopes`), in
 *  served order. */
/** An image's `open_vocab_status` (the reprocess filter's enum). */
export type OpenVocabStatus = 'pending' | 'done' | 'skipped_gate' | 'failed';

export type ReprocessScope = 'detect' | 'open_vocab' | 'region' | 'vlm' | 'embed';
export type ReprocessRegionMode = 'redetect' | 'reverify';

/** `ReprocessFilter`: the reprocess-only keys plus the shared item filter
 *  (`ItemFilter`, `types_itemFilter.ts`). */
export interface ReprocessFilter {
  all_images?: boolean;
  class_id?: number | null;
  class_names?: string[];
  class_source?: string | null;
  classifier_conf_lt?: number | null;
  cluster_id?: number | null;
  conf_max?: number | null;
  conf_min?: number | null;
  dataset_split?: string | null;
  detector?: string[];
  embedding_state?: EmbeddingState[];
  exclude_class_names?: string[];
  import_id?: string | null;
  include_detected?: boolean;
  item_text?: string | null;
  label_source?: string | null;
  label_validated?: boolean | null;
  max_area?: number | null;
  max_rank?: number | null;
  min_area?: number | null;
  min_blur_ratio?: number | null;
  missing_provenance?: boolean;
  missing_status?: boolean;
  needs_new_class?: boolean | null;
  on_negative_frame?: boolean | null;
  open_vocab_set?: string | null;
  open_vocab_status?: OpenVocabStatus[];
  origin?: ItemOrigin[];
  profile_not?: string | null;
  profile_revision_below?: number | null;
  proposed_by_import?: boolean | null;
  reason?: string[];
  region_gate_skipped?: boolean | null;
  region_status?: string[];
  review_dismissed?: boolean | null;
  review_status?: ReviewStatus[];
  source?: string | null;
  source_prompt?: string | null;
}

export interface ReprocessTargets {
  image_ids?: string[] | null;
  crop_ids?: string[] | null;
  filter?: ReprocessFilter | null;
  limit?: number | null;
  sample?: SelectionSample | null;
  seed?: number;
}

/** `EmbedOptions`: what the `embed` scope encodes. */
export interface EmbedOptions {
  only_missing?: boolean;
  parts?: ('crop' | 'frame' | 'region')[] | null;
}

export interface ReprocessRequest {
  targets: ReprocessTargets;
  scopes: ReprocessScope[];
  region_mode?: ReprocessRegionMode;
  dry_run?: boolean;
  embed?: EmbedOptions;
}

export interface ReprocessOneRequest {
  scopes: ReprocessScope[];
  region_mode?: ReprocessRegionMode;
  dry_run?: boolean;
}

/** `BreakdownRow`. */
export interface BreakdownRow {
  detector: string;
  reason: string;
  count: number;
}

/** `ReprocessScopeResult`. */
export interface ReprocessScopeResult {
  scope: ReprocessScope;
  selected?: number;
  locked_skipped?: number;
  queued?: number;
  failed?: number;
  not_found?: number;
  breakdown?: BreakdownRow[];
  /** Per-scope served facts (int, float, bool or str); the open-vocabulary
   *  dry run serves a boolean (`segmenter_reachable`) and floats beside the
   *  counts. */
  detail?: Record<string, number | boolean | string>;
}

/** `ReprocessJobInfo`. */
export interface ReprocessJob {
  job_id: string;
  status: string;
  scopes?: ReprocessScope[];
  results?: ReprocessScopeResult[];
  images_total?: number;
  images_done?: number;
  images_failed?: number;
  error?: string | null;
  started_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  poll_after_s?: number | null;
}

/** `ReprocessWireResponse`; `items` are post-write item docs on the
 *  single-target routes (`reprocessCrop` / `reprocessImage` map them
 *  through `mapRawCrop`). */
export interface ReprocessResponse<T = unknown> {
  dry_run: boolean;
  scopes: ReprocessScopeResult[];
  job?: ReprocessJob | null;
  items?: T[];
}
