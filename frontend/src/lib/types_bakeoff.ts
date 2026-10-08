/**
 * Model-comparison (bake-off) wire types, v2 — OpenProcessor #34
 * (`generic_model_comparison_plan.md` §7). Hand-written from the vendored
 * `contracts/openapi/curation.json`; every interface here
 * is pinned to its schema's property set by
 * `src/lib/contract/bakeoffContract.test.ts`.
 */

export type BakeoffJobState = 'queued' | 'running' | 'done' | 'error';
export type BakeoffModelSource = 'run' | 'baseline' | 'custom';
export type BakeoffMode = 'full' | 'crop' | 'both';
export type RankScope = 'common' | 'overall';
export type ClassMappingMethod =
  'explicit' | 'run_class_remap' | 'registry_ids' | 'names' | 'single_class_fallback';

// --- eval datasets (§7.2) ---------------------------------------------------

export interface EvalDatasetClass {
  eval_class_id: number;
  name: string;
  registry_class_id: number | null;
  /**
   * Live registry name for `registry_class_id`, resolved by id lookup
   * server-side — never assumed equal to `name` (the eval dataset's own
   * name). Null when `registry_class_id` is null or no longer present in
   * the registry. OpenProcessor 3cd4ca87. Not rendered anywhere today
   * (`registry_class_id` itself has no display site) — kept paired for
   * whenever one is added, per the thin-frontend "never a client id
   * lookup" rule.
   */
  registry_class_name: string | null;
  n_objects: number;
  n_images: number;
}

export interface EvalDataset {
  id: string;
  source: 'export' | 'external';
  group: string | null;
  name: string;
  path: string;
  is_current: boolean;
  dataset_kind: 'multi_class' | 'single_class' | 'external';
  nc: number;
  classes: EvalDatasetClass[];
  n_images: number;
  n_objects: number;
  n_background_images: number;
  frozen_test_sha: string | null;
  test_label_sha: string | null;
  sha_source: 'manifest' | 'computed';
  dataset_sha: string | null;
  exported_at: string | null;
  unlabeled_items_on_exported_images: number | null;
  frozen_ok: boolean | null;
}

export interface EvalDatasetList {
  datasets: EvalDataset[];
  count: number;
}

// --- trained models (§7.3) --------------------------------------------------

export interface TrainTestOverlap {
  n_images: number;
  fraction: number;
}

export interface TrainedModelForDataset {
  dataset_id: string;
  same_export: boolean;
  same_frozen_test: boolean | null;
  n_classes_mapped: number;
  train_test_overlap: TrainTestOverlap | null;
}

export interface TrainedModel {
  run_id: string;
  display_name: string;
  model_family: string | null;
  model_size: string | null;
  imgsz: number;
  checkpoint_path: string;
  finished_at: string | null;
  campaign_id: string | null;
  train_export_id: string | null;
  dataset_sha: string | null;
  frozen_test_sha: string | null;
  class_names: string[];
  single_cls: boolean;
  /** The trainer's own Ultralytics number — not a comparison metric. */
  trainer_map50: number | null;
  trainer_map50_split: string | null;
  /** Present only when listed with `?dataset_id=`. */
  for_dataset?: TrainedModelForDataset | null;
}

export interface TrainedModelList {
  models: TrainedModel[];
  count: number;
}

// --- profiles + baselines (§7.4, §7.5) --------------------------------------

export interface BakeoffProfile {
  name: string;
  description: string;
  kind: 'registered' | 'configured';
  default: boolean;
  class_filter: string[];
  imgsz: number;
  conf_floor: number;
  nms_iou: number;
  op_conf: number;
  op_iou: number;
  rank_metric: string;
  default_backend: string;
  triton_model: string;
  context_class_ids: number[];
  /**
   * Registry names for `context_class_ids`, same order/length, resolved
   * by id lookup server-side; an entry is the id's string form when it's
   * no longer in the registry. OpenProcessor 3cd4ca87. Not rendered
   * anywhere today — `/bakeoff`'s profile summary shows `class_filter`
   * (already names), not `context_class_ids`.
   */
  context_class_names: string[];
  baselines_path: string;
}

export interface BakeoffProfileList {
  profiles: BakeoffProfile[];
  count: number;
  default_profile: string | null;
  default_error?: string | null;
}

export interface BaselineModel {
  name: string;
  backend: string;
  weights?: string | null;
  imgsz?: number | null;
  mode?: BakeoffMode;
  class_map?: Record<string, string> | null;
  backend_options?: Record<string, unknown>;
  training_data?: string | null;
  triton_model?: string | null;
}

export interface BaselineModelList {
  baselines: BaselineModel[];
  count: number;
}

// --- run request / accepted (§7.7) ------------------------------------------

export interface RunModelRef {
  source: 'run';
  run_id: string;
  display_name?: string | null;
  backend?: 'ultralytics' | 'onnxruntime';
  mode?: BakeoffMode;
}

export interface BaselineModelRef {
  source: 'baseline';
  name: string;
  display_name?: string | null;
}

export interface CustomModelRef {
  source: 'custom';
  name: string;
  backend: string;
  weights?: string | null;
  triton_model?: string | null;
  imgsz?: number | null;
  mode?: BakeoffMode;
  class_map?: Record<string, string> | null;
  backend_options?: Record<string, unknown>;
  display_name?: string | null;
}

export type BakeoffModelRef = RunModelRef | BaselineModelRef | CustomModelRef;

export interface DatasetRef {
  id: string;
}

export interface QuantizeRequest {
  run_id: string;
  formats?: ('fp32_onnx' | 'fp16_onnx' | 'int8_onnx')[];
  n_calib?: number;
  calib_split?: 'train' | 'val';
  throughput?: boolean;
}

export interface BakeoffRunRequest {
  job_id?: string | null;
  profile?: string | null;
  datasets?: DatasetRef[];
  models?: BakeoffModelRef[];
  quantize?: QuantizeRequest | null;
}

export interface NotCoveredClass {
  eval_class_id: number;
  name: string;
}

export interface UnmappedModelClass {
  model_class_id: number;
  name?: string | null;
  n_predictions?: number | null;
}

export interface ClassMapping {
  method: ClassMappingMethod;
  model_to_eval: Record<string, number> | null;
  /**
   * `{"<model class id>": "<eval class name>"}` — same keys as
   * `model_to_eval`, paired by name (never by raw index) so a consumer
   * never has to re-derive the name from the eval class id; null under
   * the same condition as `model_to_eval`. OpenProcessor 3cd4ca87. Not
   * rendered anywhere today — `ComparisonView`'s per-class table already
   * renders served names (`NotCoveredClass`/`UnmappedModelClass`), never
   * `model_to_eval`'s ids.
   */
  model_to_eval_names: Record<string, string> | null;
  unmapped_model_classes: UnmappedModelClass[];
  not_covered_eval_classes: NotCoveredClass[];
  warnings: string[];
}

export interface AcceptedDataset {
  id: string;
  path: string;
  frozen_test_sha: string | null;
  test_label_sha: string | null;
  n_eval_classes: number;
}

export interface AcceptedModel {
  model: string;
  display_name: string;
  source: BakeoffModelSource;
  class_mapping: Record<string, ClassMapping>;
  train_test_overlap: Record<string, TrainTestOverlap | null>;
}

export interface BakeoffRunAccepted {
  status: 'enqueued';
  job_id: string;
  profile: string;
  datasets: AcceptedDataset[];
  models: AcceptedModel[];
  warnings: string[];
}

// --- status + runs (§7.8, §7.9) ---------------------------------------------

export interface JobProgress {
  done: number;
  total: number;
}

export interface CompletedTask {
  dataset: string;
  model: string;
}

export interface FailedTask {
  stage: string | null;
  dataset: string | null;
  model: string | null;
  error: string;
}

export interface BakeoffStatus {
  schema_version: 2;
  job_id: string;
  state: BakeoffJobState;
  profile: string | null;
  datasets: string[];
  models: string[];
  started_at: string | null;
  finished_at: string | null;
  progress: JobProgress;
  completed: CompletedTask[];
  failed: FailedTask[];
  error: string | null;
}

export interface BakeoffRunRow {
  job_id: string;
  state: BakeoffJobState;
  profile: string | null;
  datasets: string[];
  models: string[];
  started_at: string | null;
  finished_at: string | null;
}

export interface BakeoffRunList {
  runs: BakeoffRunRow[];
}

// --- results (§7.10) --------------------------------------------------------

export interface MetricBlock {
  n_classes: number;
  map_50: number | null;
  map_50_95: number | null;
  map_75: number | null;
  ap_small: number | null;
  ap_medium: number | null;
  ap_large: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  mean_iou: number | null;
  tp: number;
  fp: number;
  fn: number;
}

export interface CommonMetricBlock {
  n_classes: number;
  map_50: number | null;
  map_50_95: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  tp: number;
  fp: number;
  fn: number;
}

export interface PerClassRow {
  eval_class_id: number;
  name: string;
  n_gt: number;
  covered: boolean;
  model_class_ids: number[];
  ap50: number | null;
  ap50_95: number | null;
  ap75: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  tp: number | null;
  fp: number | null;
  fn: number | null;
}

export interface Coverage {
  n_eval_classes: number;
  n_covered: number;
  not_covered: NotCoveredClass[];
  unmapped_model_classes: UnmappedModelClass[];
  predictions_outside_scored_classes: number;
}

export interface RowClassMapping {
  method: ClassMappingMethod;
  warnings: string[];
}

export interface LatencyStats {
  mean: number;
  p50: number;
  p90: number;
  p99: number;
}

export interface StratumMetrics {
  n_images: number;
  map_50: number | null;
  precision: number | null;
  recall: number | null;
}

export interface ComparisonRow {
  rank: number | null;
  model: string;
  display_name: string;
  source: BakeoffModelSource;
  run_id: string | null;
  runtime: string;
  imgsz: number | null;
  training_data: string | null;
  overall: MetricBlock;
  common: CommonMetricBlock;
  per_class: PerClassRow[];
  coverage: Coverage;
  class_mapping: RowClassMapping;
  train_test_overlap: TrainTestOverlap | null;
  latency_ms: LatencyStats;
  fps: number;
  size_mb: number | null;
  per_stratum: Record<string, StratumMetrics>;
}

export interface ComparisonDataset {
  id: string;
  frozen_test_sha?: string | null;
  test_label_sha?: string | null;
  n_images?: number | null;
  n_objects?: number | null;
  n_background_images?: number | null;
}

export interface EvalClassGt {
  eval_class_id: number;
  name: string;
  n_gt: number;
}

export interface FailedModel {
  model: string;
  error: string | null;
}

export interface BakeoffComparison {
  schema_version: 2;
  job_id: string | null;
  profile: string | null;
  thresholds: Record<string, number>;
  dataset: ComparisonDataset;
  eval_classes: EvalClassGt[];
  common_classes: number[];
  rank_by: string;
  rank_scope: RankScope;
  models: ComparisonRow[];
  failed: FailedModel[];
  warnings: string[];
  n_models: number;
}

// --- matrix (§7.11) ---------------------------------------------------------

export interface MatrixDataset {
  id: string;
  frozen_test_sha: string | null;
  test_label_sha: string | null;
  rank_scope: RankScope | null;
  n_common_classes: number;
}

export interface MatrixModel {
  model: string;
  display_name: string;
  source: BakeoffModelSource;
}

export interface MatrixCell {
  map_50: number | null;
  map_50_95: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  latency_ms: number | null;
  size_mb: number | null;
  coverage: number | null;
  rank: number | null;
}

export interface BakeoffMatrix {
  schema_version: 2;
  job_id: string | null;
  rank_by: string | null;
  datasets: MatrixDataset[];
  models: MatrixModel[];
  metrics: string[];
  cells: Record<string, Record<string, MatrixCell>>;
  /** Every tied winner per dataset per metric (a list, never one key). */
  best: Record<string, Record<string, string[]>>;
}
