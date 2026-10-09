/**
 * TypeScript types mirroring the Pydantic models in
 * `OpenProcessor/src/services/training/jobs.py` +
 * `OpenProcessor/src/routers/curation_train/` and `src/services/training/`.
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
  'probe' | 'nano' | 'small' | 'medium' | 'large' | 'xlarge' | 'custom';

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

  // Lineage — normally auto-filled server-side at submit time
  // (`write_job`) from the export's manifest.json and this build's own
  // identity; a caller should not set these directly. Surfaced here only
  // so a served `TrainJobSpec` (e.g. `TrainManifest.spec`, or a
  // Reproduce resubmit) round-trips them without loss.
  dataset_sha?: string | null;
  frozen_test_sha?: string | null;
  test_label_sha?: string | null;
  dataset_version_tag?: string | null;
  api_sha?: string | null;
  trainer_image_id?: string | null;
  trainer_image_revision?: string | null;
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
 * Which pass produced `TrainEval`'s overall figures — `'test'` when the
 * post-training re-validation against the frozen holdout succeeded,
 * `'val'` when it didn't run or produced no usable box metrics (falls
 * back to the training-time validation numbers). The trainer writes it
 * on every `eval` block.
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
 * `TrainJobStatus.eval` / `TrainManifest.results.eval` — evaluation the
 * trainer writes after the run finishes: a fresh `model.val(..., split=
 * 'test')` pass against the frozen holdout whenever it succeeds
 * (`split === 'test'`), falling back to the training-time validation
 * numbers when it didn't (`split === 'val'`, no `per_class`). This is
 * the run's headline number — render `map50`/`map50_95` labelled by
 * `split` (`src/lib/trainResults.ts`'s `evalOverallLabel`/
 * `evalPerClassLabel`); never `TrainJobStatus.best_checkpoint_metric`,
 * which is a per-epoch training-time figure, not this holdout pass.
 *
 * `head` (OpenProcessor #34 W1) names the detection head that test-split
 * pass scored, e.g. `'end2end'` when the loaded checkpoint's NMS-free
 * one-to-one head was explicitly forced to match what's actually served
 * — `null`/absent for a model family with no such distinction.
 *
 * `confusion_matrix_url` is the servable artifact URL
 * (`GET /train/artifacts/{job_id}/{name}`) — render an `<img>` from
 * this only. The served `confusion_matrix_path` (a server filesystem
 * path) is never read or shown.
 */
export interface TrainEval {
  map50?: number | null;
  map50_95?: number | null;
  precision?: number | null;
  recall?: number | null;
  split: TrainEvalSplit;
  /** The last validation epoch's numbers, served separately from the
   *  overall figures since 5595474. */
  val_last?: { map50?: number | null; map50_95?: number | null } | null;
  per_class?: TrainEvalPerClass[] | null;
  /** Detection head this eval pass scored (OpenProcessor #34 W1),
   *  e.g. `'end2end'`; `null`/absent when not applicable. */
  head?: string | null;
  /** Servable URL (`GET {API_PREFIX}/train/artifacts/{job_id}/{name}`),
   *  API-prefix-relative; resolve with `resolveApiUrl`. */
  confusion_matrix_url?: string | null;
  [extra: string]: unknown;
}

/** One row of `TrainJobStatus.last_epoch_metric` /
 *  `.best_checkpoint_metric` (OpenProcessor #34 W1) — a single coherent
 *  validation pass (map50 + map50_95 from the SAME pass, never a
 *  per-key running max across different epochs). `epoch` is that pass's
 *  1-indexed training epoch number. `null` on a run whose status.json
 *  predates this field (no incorrect back-fill — render "—"). */
export interface TrainEpochMetric {
  epoch?: number | null;
  map50?: number | null;
  map50_95?: number | null;
  [k: string]: unknown;
}

/** Status JSON the trainer writes; nullable everywhere except job_id+state. */
export interface TrainJobStatus {
  job_id: string;
  /** Latest promote of this run; served by `GET /train/status/{job_id}` only. */
  promote?: PromoteJobStatus | null;
  campaign_id?: string | null;
  state: TrainState;
  started_at?: string | null;
  finished_at?: string | null;
  current_epoch?: number | null;
  total_epochs?: number | null;
  epoch_time_s?: number | null;
  /** The true last TRAINING epoch's own metrics (OpenProcessor #34 W1) —
   *  distinct from `best_checkpoint_metric` because Ultralytics
   *  re-validates the best checkpoint once more after training and that
   *  pass doesn't advance the epoch counter. */
  last_epoch_metric?: TrainEpochMetric | null;
  /** The best checkpoint's (best.pt) own re-validation metrics, as one
   *  coherent row. Never the run's headline number — that's `eval.map50`
   *  labelled by `eval.split`; this is a training-time figure. `null` on
   *  a run whose status.json predates this field. */
  best_checkpoint_metric?: TrainEpochMetric | null;
  mlflow_run_id?: string | null;
  /** Served `null` unless `OP_MLFLOW_PUBLIC_URL` is set, so any non-null
   *  value is browser-reachable; rendered as a link (`RunResults.svelte`). */
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
  /** The export's frozen-test-split hash (OpenProcessor #34 W1) — copied
   *  from the export manifest, or computed from disk when an older
   *  export's manifest lacks it. */
  frozen_test_sha?: string | null;
  /** Hash of just the frozen test split's label files (#34 W1) —
   *  distinct from `frozen_test_sha` (the whole frozen split). */
  test_label_sha?: string | null;
  /** The export's own human-readable version tag (#34 W1), when the
   *  export recorded one. */
  dataset_version_tag?: string | null;
  deterministic?: boolean | null;
  export_dir?: string | null;
  include_classes?: number[] | null;
  registry_sha?: string | null;
  single_cls?: boolean | null;
  training_seed?: number | null;
}

export interface TrainManifestCodeVersions {
  api_sha?: string | null;
  /** The trainer container's own build identity (#34 W1) — its baked
   *  `OP_BUILD_SHA` when present, else the API's stamped
   *  `trainer_image_revision` observed at submit time. */
  trainer_sha?: string | null;
  /** The trainer image's own id (#34 W1), independent of `trainer_sha`'s
   *  revision label. */
  trainer_image_id?: string | null;
  ultralytics_pkg?: string | null;
  ultralytics_sha?: string | null;
  [extra: string]: unknown;
}

export interface TrainManifestResults {
  /** The true last training epoch's metrics (#34 W1) — see
   *  `TrainJobStatus.last_epoch_metric`. */
  last_epoch_metric?: TrainEpochMetric | null;
  /** The best checkpoint's own re-validation metrics (#34 W1) — see
   *  `TrainJobStatus.best_checkpoint_metric`. Never the headline number;
   *  that's `eval.map50` labelled by `eval.split`. */
  best_checkpoint_metric?: TrainEpochMetric | null;
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
 * {API_PREFIX}/train/augmentation_presets` (OpenProcessor df01309).
 */
export interface AugmentationPresetOption {
  id: string;
  label: string;
  description: string;
  /** Horizontal flip is disabled for the whole run with this preset. */
  orientation_sensitive: boolean;
}

/** `GET {API_PREFIX}/train/augmentation_presets` response. */
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
  /** Bypass the promote gate. Offered only when the server's 422 says
   *  `force_allowed` (F-64). */
  force?: boolean;
}

/** `PromoteJobStatus.status`. Active phases run in this order. */
export type PromoteJobPhase =
  'queued' | 'exporting' | 'loading' | 'building' | 'warming' | 'done' | 'failed';

/** `POST {API_PREFIX}/train/promote/{job_id}` 202 (and the 200 for an
 *  already-active job), `GET .../train/promote/{job_id}/jobs/{promote_id}`
 *  and `TrainJobStatus.promote`. */
export interface PromoteJobStatus {
  promote_id: string;
  job_id: string;
  triton_name: string;
  status: PromoteJobPhase;
  error?: string | null;
  /** HTTP status a synchronous promote would have returned. */
  error_status?: number | null;
  /** The synchronous body; only on `done`. */
  result?: PromoteResponse | null;
  started_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  /** Seconds to wait before the next poll; null once terminal. */
  poll_after_s?: number | null;
}

export interface PromoteResponse {
  job_id: string;
  triton_name: string;
  onnx_path: string;
  config_path: string;
  labels_path: string;
  triton_loaded: boolean;
  /** The first inference after a promote builds the TensorRT engine and
   *  is slow. */
  cold_start_expected_on_first_inference: boolean;
}
