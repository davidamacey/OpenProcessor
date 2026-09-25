/**
 * TypeScript types mirroring the Pydantic models in
 * `OpenProcessor/src/services/training/jobs.py` +
 * `OpenProcessor/src/routers/curation_train.py`.
 *
 * Kept separate from `types.ts` so the (already large) labeler types
 * file isn't dwarfed by the training surface, and so the eventual
 * deletion is a single-file delete.
 *
 * Forward-compatibility: every model is a structural subset — the
 * server can add fields without us re-shipping. Server-side enums
 * (state, profile name, etc.) are typed as string literal unions
 * matching the Pydantic Literal[...] declarations.
 */
export type ModelFamily = 'yolo26';
export type ModelSize = 'n' | 's' | 'm' | 'l' | 'x';

export type ProfileName =
  | 'probe'
  | 'nano'
  | 'small'
  | 'medium'
  | 'large'
  | 'xlarge'
  | 'custom';

export type TrainState =
  | 'queued'
  | 'starting'
  | 'running'
  | 'exporting'
  | 'finished'
  | 'failed'
  | 'cancelled'
  | 'skipped'
  | 'lost';

export type PreflightSeverity = 'ok' | 'warn' | 'block';

/** Augmentation block on TrainJobSpec / TrainCampaignSpec. */
export interface AugmentationSpec {
  enabled?: boolean;
  multiplier?: number;
  preset?: string;
  albumentations?: Record<string, unknown>;
  per_class_multiplier?: Record<string, number>;
  auto_balance?: {
    target_count: number;
    max_multiplier: number;
  };
  [extra: string]: unknown;
}

/** POST {API_PREFIX}/train/start payload. */
export interface TrainJobSpec {
  job_id?: string | null;
  campaign_id?: string | null;
  submitted_by?: string;
  submitted_at?: string | null;

  model_family?: ModelFamily;
  model_size?: ModelSize;
  profile?: ProfileName;

  dataset_export_dir: string;
  include_classes?: number[] | null;
  single_cls?: boolean;

  cuda_visible_devices?: string;

  hyperparameters?: Record<string, unknown>;
  augmentation?: AugmentationSpec | null;

  mlflow_run_name?: string | null;

  // Opt-in: on successful finish, auto-export all deployable formats and
  // benchmark them (drives the trainer's auto_quantize_bakeoff hook).
  auto_quantize_bakeoff?: boolean;
}

export interface CampaignRunSpec {
  profile: ProfileName | string;
  model_size?: ModelSize | null;
  hyperparameters?: Record<string, unknown>;
}

/** POST {API_PREFIX}/train/start_campaign payload. */
export interface TrainCampaignSpec {
  campaign_id?: string | null;
  dataset_export_dir: string;
  include_classes?: number[] | null;
  single_cls?: boolean;
  cuda_visible_devices?: string;
  augmentation?: AugmentationSpec | null;
  runs: CampaignRunSpec[];
  stop_when?: { map50_at_least?: number; [k: string]: number | undefined } | null;
  auto_promote_best?: boolean;
  submitted_by?: string;
}

export interface GpuInfo {
  index?: number;
  util_pct?: number;
  mem_used_mb?: number;
  mem_total_mb?: number;
  [extra: string]: unknown;
}

/**
 * Which pass produced `TrainEval`'s overall figures. Absent on today's
 * backend — see `TrainEval`'s doc comment for what that means. Present
 * once the trainer's train-eval cutover lands (branch `cutover/train-
 * eval`), naming whichever pass actually won.
 */
export type TrainEvalSplit = 'test' | 'val';

/** One row of `TrainEval.per_class`. */
export interface TrainEvalPerClass {
  class_id: number;
  name: string;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  ap50?: number | null;
  support?: number | null;
}

/**
 * `TrainJobStatus.eval` / `TrainManifest.results.eval` — test-split
 * evaluation the trainer writes after the run finishes.
 *
 * **Today's backend (no `split` field):** `map50`/`map50_95` (and any
 * other overall figure) are actually the *last VAL epoch's* numbers,
 * while `per_class` really is computed over the frozen test holdout.
 * Two different passes under one object — render each half labelled by
 * what it actually is (`src/lib/trainResults.ts`'s `evalOverallLabel`/
 * `evalPerClassLabel`), never both as "test".
 *
 * **Upcoming backend** (train-eval cutover): `split` names the pass
 * that produced `map50`/`map50_95`/`precision`/`recall` — `'test'`
 * when a test pass ran, falling back to `'val'` otherwise. Once present,
 * both the overall figures and the per-class table are labelled by this
 * field instead of the pre-cutover guess above.
 *
 * `confusion_matrix_url` is the servable artifact URL
 * (`GET /train/artifacts/{job_id}/{name}`) — render an `<img>` from
 * this only, never from `confusion_matrix_path` (a server filesystem
 * path, text-only).
 */
export interface TrainEval {
  map50?: number | null;
  map50_95?: number | null;
  precision?: number | null;
  recall?: number | null;
  /** TODO(train-eval cutover): absent on today's backend — see doc comment above. */
  split?: TrainEvalSplit | null;
  per_class?: TrainEvalPerClass[] | null;
  /** Server filesystem path — text only, never an `<img src>`. */
  confusion_matrix_path?: string | null;
  /** TODO(train-eval cutover): servable URL once the backend ships it. */
  confusion_matrix_url?: string | null;
  [extra: string]: unknown;
}

/** Status JSON the trainer writes; nullable everywhere except job_id+state. */
export interface TrainJobStatus {
  job_id: string;
  campaign_id?: string | null;
  state: TrainState;
  started_at?: string | null;
  finished_at?: string | null;
  current_epoch?: number | null;
  total_epochs?: number | null;
  epoch_time_s?: number | null;
  best_metric?: {
    map50?: number;
    map50_95?: number;
    [k: string]: number | undefined;
  } | null;
  last_metric?: {
    map50?: number;
    map50_95?: number;
    [k: string]: number | undefined;
  } | null;
  mlflow_run_id?: string | null;
  /**
   * TODO: backend is being asked to serve this as `null` unless
   * `OP_MLFLOW_PUBLIC_URL` is set — never the docker-internal hostname
   * (e.g. `http://op-mlflow:5000/...`). Until that lands, this may be a
   * URL a browser can't reach; we render it as a link whenever it's
   * non-null anyway, per that request — see `RunResults.svelte`.
   */
  mlflow_run_url?: string | null;
  checkpoint_path?: string | null;
  /** Forward-compat: not served on `TrainJobStatus` today (only on the
   *  manifest's `results.checkpoint_sha256`) — kept here too so a future
   *  backend that starts serving it on status needs no frontend change. */
  checkpoint_sha256?: string | null;
  gpu?: GpuInfo[];
  eval?: TrainEval | null;
  error?: string | null;
  heartbeat_at?: string | null;
  /** Forward-compat: extra fields server may add. */
  [extra: string]: unknown;
}

/** `GET {API_PREFIX}/train/manifest/{job_id}` — full lineage envelope:
 *  dataset SHA, class remap, code versions, eval results. 404 means the
 *  run finished before the manifest writer was added. */
export interface TrainManifestClassRemap {
  include_classes?: number[] | null;
  names?: string[] | null;
  /** new class id (string key, index into `names`) -> original registry id */
  new_to_original?: Record<string, number> | null;
  /** original registry id (string key) -> new class id */
  original_to_new?: Record<string, number> | null;
  single_cls?: boolean | null;
}

export interface TrainManifestLineage {
  augmentation_seed?: number | null;
  class_remap?: TrainManifestClassRemap | null;
  dataset_sha?: string | null;
  deterministic?: boolean | null;
  export_dir?: string | null;
  include_classes?: number[] | null;
  registry_sha?: string | null;
  single_cls?: boolean | null;
  training_seed?: number | null;
}

export interface TrainManifestCodeVersions {
  api_sha?: string | null;
  trainer_image?: string | null;
  ultralytics_pkg?: string | null;
  ultralytics_sha?: string | null;
  [extra: string]: unknown;
}

export interface TrainManifestResults {
  best_metric?: {
    map50?: number;
    map50_95?: number;
    [k: string]: number | undefined;
  } | null;
  checkpoint_path?: string | null;
  checkpoint_sha256?: string | null;
  compare?: unknown;
  eval?: TrainEval | null;
  final_state?: TrainState | null;
  mlflow_run_id?: string | null;
  mlflow_run_url?: string | null;
  [extra: string]: unknown;
}

export interface TrainManifest {
  campaign_id?: string | null;
  code_versions?: TrainManifestCodeVersions | null;
  created_at?: string | null;
  job_id: string;
  kind?: string;
  lineage?: TrainManifestLineage | null;
  promoted_to?: string | null;
  results?: TrainManifestResults | null;
  spec?: Record<string, unknown>;
  [extra: string]: unknown;
}

export interface PreflightCheck {
  name: string;
  severity: PreflightSeverity;
  message: string;
  detail?: Record<string, unknown> | null;
}

export interface PreflightReport {
  blocked: boolean;
  checks: PreflightCheck[];
  summary?: string;
}

export interface StartTrainResponse {
  job_id: string;
  preflight: PreflightReport;
}

export interface StartCampaignResponse {
  campaign_id: string;
  job_ids: string[];
  preflight?: PreflightReport;
}

export interface RunsListResponse {
  items: TrainJobStatus[];
  total: number;
}

export interface LogTailResponse {
  job_id: string;
  lines: string[];
}

export interface CancelResponse {
  cancelled: boolean | number;
  job_id?: string;
  campaign_id?: string;
}

export interface Profile {
  name: string;
  description: string;
  defaults: Record<string, unknown>;
}

export interface ProfilesResponse {
  profiles: Profile[];
}

export type PresetSelectorKind = 'all' | 'all_except' | 'names' | 'groups';

export interface ClassSubsetPreset {
  name: string;
  label: string;
  description: string;
  selector: {
    kind: PresetSelectorKind;
    names?: string[];
    groups?: string[];
  };
  single_cls_default?: boolean;
}

export interface PresetsResponse {
  class_subset_presets: ClassSubsetPreset[];
}

/**
 * One selectable `augmentation.preset` — `GET
 * {API_PREFIX}/train/augmentation_presets` (OpenProcessor 6c77deb).
 */
export interface AugmentationPresetOption {
  id: string;
  label: string;
  description: string;
  /** Horizontal flip is disabled for the whole run with this preset. */
  orientation_sensitive: boolean;
}

/** `GET {API_PREFIX}/train/augmentation_presets` response. Absent on a
 *  pre-6c77deb backend (404) — `AugmentationPanel` falls back to a
 *  read-only display of the current value when this fails to load. */
export interface AugmentationPresetsResponse {
  presets: AugmentationPresetOption[];
  /** Preset used when a job omits `augmentation.preset`. */
  default: string;
}

/** POST {API_PREFIX}/train/promote/{job_id} body. */
export interface PromoteRequest {
  triton_name: string;
  max_batch_size?: number;
  input_size?: number;
  fp16?: boolean;
  overwrite?: boolean;
}

export interface PromoteResponse {
  job_id: string;
  triton_name: string;
  onnx_path: string;
  config_path: string;
  labels_path: string;
  triton_loaded: boolean;
}
