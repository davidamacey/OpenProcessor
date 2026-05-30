/**
 * TypeScript types mirroring the Pydantic models in
 * `openprocessor/src/services/legacy/train_jobs.py` +
 * `openprocessor/src/routers/legacy_train.py`.
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

/** POST /curation/train/start payload. */
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

/** POST /curation/train/start_campaign payload. */
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
  best_metric?: { map50?: number; map50_95?: number; [k: string]: number | undefined } | null;
  last_metric?: { map50?: number; map50_95?: number; [k: string]: number | undefined } | null;
  mlflow_run_id?: string | null;
  mlflow_run_url?: string | null;
  checkpoint_path?: string | null;
  gpu?: GpuInfo[];
  eval?: Record<string, unknown> | null;
  error?: string | null;
  heartbeat_at?: string | null;
  /** Forward-compat: extra fields server may add. */
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

/** POST /curation/train/promote/{job_id} body. */
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
