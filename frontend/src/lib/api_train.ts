/**
 * Training: served cohorts, preflight, runs, GPUs, presets, promote,
 * and the probe pass.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import type {
  AugmentationPresetsResponse,
  CancelResponse,
  LogTailResponse,
  PreflightReport,
  PresetsResponse,
  ProfilesResponse,
  PromoteJobStatus,
  PromoteRequest,
  RunsListResponse,
  StartCampaignResponse,
  StartTrainResponse,
  TrainCampaignSpec,
  TrainJobSpec,
  TrainJobStatus,
  TrainManifest,
} from '$lib/types_train';

/**
 * `GET {API_PREFIX}/training_cohorts?class_id=` — the deployment's own training-cohort definitions,
 * already resolved for the requested class (params fold `class_id` into
 * every cohort). `row_kind: 'region'` cohorts only appear when the
 * backend has a region profile configured; `class_id` omitted returns
 * the generic (unscoped) definitions. `/train`'s `runCohortQuery`
 * dispatches `endpoint`/`params` verbatim — no client re-derivation.
 */
export interface ServedTrainingCohort {
  id: string;
  label: string;
  description: string;
  cutoffs: Record<string, number>;
  /** Relative to API_PREFIX, e.g. `/crops`, `/regions/training_candidates`. */
  endpoint: string;
  /** Already resolved for the requested class — no `{classId}` templates. */
  params: Record<string, unknown>;
  row_kind: 'crop' | 'region';
}

export function getTrainingCohorts(
  classId?: number | null,
  signal?: AbortSignal,
): Promise<{ cohorts: ServedTrainingCohort[] }> {
  return apiFetch<{ cohorts: ServedTrainingCohort[] }>(
    `${scoped()}/training_cohorts${qs({ class_id: classId ?? undefined })}`,
    {},
    signal,
  );
}

// -- training endpoints --------------------------------------------------
//
// Mirror the FastAPI `{API_PREFIX}/train/*` router. The labeler `/train` page is
// the only consumer; types live in `./types_train.ts` so the existing
// types.ts stays focused on the labeling data model.

/**
 * Run the preflight checks for a candidate spec WITHOUT writing
 * `job.json`. Used for inline form validation.
 *
 * The endpoint never throws on blocking issues — it returns
 * `{blocked, checks, summary}` so the UI can render every row.
 */
export function trainPreflight(
  spec: TrainJobSpec,
  signal?: AbortSignal,
): Promise<PreflightReport> {
  return apiFetch<PreflightReport>(
    `${scoped()}/train/preflight`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/**
 * Start a single training run. Backend returns 422 with the preflight
 * report on blocking checks unless `force=true`. We surface those to
 * the caller as ApiError; the form reads `body.preflight` and renders.
 */
export function trainStart(
  spec: TrainJobSpec,
  force: boolean = false,
  signal?: AbortSignal,
): Promise<StartTrainResponse> {
  return apiFetch<StartTrainResponse>(
    `${scoped()}/train/start${qs({ force: force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/** Submit a multi-size training campaign (queued back-to-back). */
export function trainStartCampaign(
  spec: TrainCampaignSpec,
  force: boolean = false,
  signal?: AbortSignal,
): Promise<StartCampaignResponse> {
  return apiFetch<StartCampaignResponse>(
    `${scoped()}/train/start_campaign${qs({ force: force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/**
 * Most-recent or active job's `status.json`. Returns null when there's
 * never been a run.
 *
 * Pass a `jobId` to fetch a specific run (404 on missing).
 */
export async function getTrainStatus(
  jobId?: string,
  signal?: AbortSignal,
): Promise<TrainJobStatus | null> {
  const path = jobId
    ? `${scoped()}/train/status/${encodeURIComponent(jobId)}`
    : `${scoped()}/train/status`;
  try {
    return await apiFetch<TrainJobStatus | null>(path, {}, signal);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}

export function getTrainRuns(
  limit: number = 50,
  offset: number = 0,
  signal?: AbortSignal,
): Promise<RunsListResponse> {
  return apiFetch<RunsListResponse>(
    `${scoped()}/train/runs${qs({ limit, offset })}`,
    {},
    signal,
  );
}

export function tailTrainLog(
  jobId: string,
  lines: number = 200,
  signal?: AbortSignal,
): Promise<LogTailResponse> {
  return apiFetch<LogTailResponse>(
    `${scoped()}/train/log/tail/${encodeURIComponent(jobId)}${qs({ lines })}`,
    {},
    signal,
  );
}

export function cancelTrainJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `${scoped()}/train/cancel/${encodeURIComponent(jobId)}`,
    { method: 'POST' },
    signal,
  );
}

export function cancelTrainCampaign(
  campaignId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `${scoped()}/train/cancel_campaign/${encodeURIComponent(campaignId)}`,
    { method: 'POST' },
    signal,
  );
}

export function getTrainProfiles(signal?: AbortSignal): Promise<ProfilesResponse> {
  return apiFetch<ProfilesResponse>(`${scoped()}/train/profiles`, {}, signal);
}

/** One GPU claim the backend allows for a training run. */
export interface TrainGpuOption {
  /** The exact `cuda_visible_devices` value to send. */
  value: string;
  gpu_ids: number[];
  label: string;
  /** What the claim does to other services (e.g. which containers stop). */
  advisory: string | null;
  stops_containers: string[];
  default: boolean;
}

/** `GET {API_PREFIX}/train/gpus`. `unrestricted` means no allowlist is
 *  configured, so any `cuda_visible_devices` value may be entered. */
export interface TrainGpuOptionsResponse {
  options: TrainGpuOption[];
  allowed_ids: number[];
  unrestricted: boolean;
}

export function getTrainGpus(signal?: AbortSignal): Promise<TrainGpuOptionsResponse> {
  return apiFetch<TrainGpuOptionsResponse>(`${scoped()}/train/gpus`, {}, signal);
}

/** The served default claim, or '' when none is marked (the backend then
 *  picks from its allowlist when `cuda_visible_devices` is omitted). */
export function defaultGpuValue(res: TrainGpuOptionsResponse): string {
  return res.options.find((o) => o.default)?.value ?? '';
}

export function getTrainPresets(signal?: AbortSignal): Promise<PresetsResponse> {
  return apiFetch<PresetsResponse>(`${scoped()}/train/presets`, {}, signal);
}

/**
 * `GET {API_PREFIX}/train/augmentation_presets` (OpenProcessor df01309) —
 * the trainer's real preset catalog, for `AugmentationPanel`'s picker.
 */
export function getAugmentationPresets(
  signal?: AbortSignal,
): Promise<AugmentationPresetsResponse> {
  return apiFetch<AugmentationPresetsResponse>(
    `${scoped()}/train/augmentation_presets`,
    {},
    signal,
  );
}

/**
 * Start a background promote. The default is 202 with a
 * `PromoteJobStatus`; an identical promote already running answers 200 with
 * that same job. A different `triton_name` while one runs is 409
 * `promote_in_progress` with the active `promote_id` (see
 * `promoteInProgressId` in `$lib/promote`). Synchronous gate / name
 * failures (422, 409, 404) still arrive as `ApiError` before any job exists.
 */
export function promoteTrainJob(
  jobId: string,
  body: PromoteRequest,
  signal?: AbortSignal,
): Promise<PromoteJobStatus> {
  return apiFetch<PromoteJobStatus>(
    `${scoped()}/train/promote/${encodeURIComponent(jobId)}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `GET {API_PREFIX}/train/promote/{job_id}/jobs/{promote_id}`. */
export function getPromoteJob(
  jobId: string,
  promoteId: string,
  signal?: AbortSignal,
): Promise<PromoteJobStatus> {
  return apiFetch<PromoteJobStatus>(
    `${scoped()}/train/promote/${encodeURIComponent(jobId)}/jobs/${encodeURIComponent(promoteId)}`,
    {},
    signal,
  );
}

/**
 * Run manifest (design §15.4) — full lineage envelope: dataset SHA,
 * class remap, code versions, eval results. 404 means the run finished
 * before the manifest writer was added.
 */
export function getTrainManifest(
  jobId: string,
  signal?: AbortSignal,
): Promise<TrainManifest> {
  return apiFetch<TrainManifest>(
    `${scoped()}/train/manifest/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

// -- Probe (#36 item 8) ---------------------------------------------------
// Wraps POST {API_PREFIX}/probe/{run,cancel} / GET {API_PREFIX}/probe/status —
// starts a probe pass from a finished training run's export, populating
// probe_pred_* on items (the Uncertainty/Model-disagreements queues' one
// prerequisite, per item 9's empty_state). Same idempotent job-poll shape
// as scores/embedding-viz jobs: one job at a time, GET /probe/status is
// the single source of truth for what's running.

export interface ProbeRunRequest {
  job_id: string;
  architecture?: string;
  gpu?: string | null;
  resume?: boolean;
}

/** `ProbeStatusResponse` — served verbatim, including `error`, which is
 *  the backend's own message (a GPU-arbiter claim failure, a training job
 *  that isn't finished / has no exported checkpoint, etc.) and is always
 *  rendered as-is, never reworded. */
export interface ProbeStatusResponse {
  status: string;
  job_id?: string | null;
  train_job_id?: string | null;
  gpu?: string | null;
  model_path?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  updated_count?: number | null;
  error?: string | null;
  /** Read-only echo of `OP_PROBE_ACTIONABLE_MIN_CONFIDENCE` (default 0.5)
   *  — the threshold the server applies to compute the item wire's
   *  `probe_actionable`. No client code recomputes actionability against
   *  this; it exists purely to render "threshold: NN%" where a probe
   *  status is already in hand. */
  actionable_min_confidence?: number | null;
}

/** Start a probe pass from `trainJobId`'s finished export. 409 when the
 *  training job isn't finished / has no exported checkpoint, or a probe
 *  is already running — surfaced via `ApiError.detail`, never guessed. */
export function runProbe(
  trainJobId: string,
  opts: { architecture?: string; gpu?: string | null; resume?: boolean } = {},
  signal?: AbortSignal,
): Promise<ProbeStatusResponse> {
  const body: ProbeRunRequest = { job_id: trainJobId, ...opts };
  return apiFetch<ProbeStatusResponse>(
    `${scoped()}/probe/run`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Poll the current/last probe job. */
export function getProbeStatus(signal?: AbortSignal): Promise<ProbeStatusResponse> {
  return apiFetch<ProbeStatusResponse>(`${scoped()}/probe/status`, {}, signal);
}

/** Best-effort cancel of the active probe job. */
export function cancelProbe(signal?: AbortSignal): Promise<ProbeStatusResponse> {
  return apiFetch<ProbeStatusResponse>(
    `${scoped()}/probe/cancel`,
    { method: 'POST' },
    signal,
  );
}
