// Owner: Track B (P4 combine projects).
// Holds the backend wrappers for that feature. Import from `$lib/api_combine`
// directly; never re-exported from `api.ts` (that would make the two
// modules circular).
//
// Every combine route is GLOBAL (`{globalApi()}/projects/combine*`): the
// target is created by the job, so no project scope exists yet. Only the
// two reads that address an EXISTING project (its served mapping
// vocabulary, a finished job's next step) go through that project's own
// served prefix.
import { ApiError, apiFetch, globalApi, projectPrefix } from '$lib/api';
import type {
  CombineFormatsVocabulary,
  CombineJobResponse,
  CombineNextStep,
  CombinePreview,
  CombineRequest,
  CombineStartRequest,
  CombineStartResponse,
} from '$lib/types_combine';

const GLOBAL = { global: true } as const;

/** `POST /projects/combine/preview` — writes nothing. */
export function previewCombine(
  body: CombineRequest,
  signal?: AbortSignal,
): Promise<CombinePreview> {
  return apiFetch<CombinePreview>(
    `${globalApi()}/projects/combine/preview`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

/** `POST /projects/combine` — 202; the target project is created
 *  `building` and the job runs in the background. */
export function startCombine(
  body: CombineStartRequest,
  signal?: AbortSignal,
): Promise<CombineStartResponse> {
  return apiFetch<CombineStartResponse>(
    `${globalApi()}/projects/combine`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

export function getCombineJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<CombineJobResponse> {
  return apiFetch<CombineJobResponse>(
    `${globalApi()}/projects/combine/${encodeURIComponent(jobId)}`,
    {},
    signal,
    GLOBAL,
  );
}

export function cancelCombine(
  jobId: string,
  signal?: AbortSignal,
): Promise<CombineJobResponse> {
  return apiFetch<CombineJobResponse>(
    `${globalApi()}/projects/combine/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' },
    signal,
    GLOBAL,
  );
}

export function resumeCombine(
  jobId: string,
  signal?: AbortSignal,
): Promise<CombineJobResponse> {
  return apiFetch<CombineJobResponse>(
    `${globalApi()}/projects/combine/${encodeURIComponent(jobId)}/resume`,
    { method: 'POST' },
    signal,
    GLOBAL,
  );
}

/** `GET {prefix}/datasets/formats` of one source project — read only for
 *  its served `mapping_actions` labels (combine serves no vocabulary of
 *  its own, plan question P4-7). */
export function getDatasetFormatsFor(
  project: { prefix: string },
  signal?: AbortSignal,
): Promise<CombineFormatsVocabulary> {
  return apiFetch<CombineFormatsVocabulary>(
    `${projectPrefix(project)}/datasets/formats`,
    {},
    signal,
    GLOBAL,
  );
}

/** A finished job's served `next_steps` entry, run as served against the
 *  target project's own prefix: the step's `method` and `path`, no body
 *  (plan question P4-5). */
export function runCombineNextStep(
  project: { prefix: string },
  step: CombineNextStep,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `${projectPrefix(project)}${step.path}`,
    { method: step.method.toUpperCase() },
    signal,
    GLOBAL,
  );
}

/** True for the served `{detail: {error: 'combine_not_found'}}` 404: the
 *  combine router is mounted and answering about a job id. */
export function isCombineNotFound(e: unknown): boolean {
  if (!(e instanceof ApiError) || e.status !== 404) return false;
  const body = e.body as { detail?: unknown } | null;
  const d = body?.detail;
  return (
    !!d &&
    typeof d === 'object' &&
    (d as { error?: unknown }).error === 'combine_not_found'
  );
}
