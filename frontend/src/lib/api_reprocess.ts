/**
 * Reprocess: the served next step, batch / crop / image reprocess and the job.
 */
import { apiFetch, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import type {
  NextStep,
  ReprocessJob,
  ReprocessOneRequest,
  ReprocessRequest,
  ReprocessResponse,
} from '$lib/types_import';
import type { Crop } from '$lib/types';

/** A finished import's served `next_steps` entry, run as served: its
 *  `method` against its `path` under the project's prefix, no body
 *  (plan §8 question 10). */
export function runServedNextStep(
  step: NextStep,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}${step.path}`,
    { method: step.method.toUpperCase() },
    signal,
  );
}

/** Batch Reprocess. `dry_run` defaults to true on the server; the caller
 *  always sends it explicitly. */
export function reprocessBatch(
  body: ReprocessRequest,
  signal?: AbortSignal,
): Promise<ReprocessResponse> {
  return apiFetch<ReprocessResponse>(
    `${scoped()}/reprocess`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Single-item Reprocess; returns the post-write `items` (mapped like
 *  every other crop) for the caller to adopt. */
export async function reprocessCrop(
  cropId: string,
  body: ReprocessOneRequest,
  signal?: AbortSignal,
): Promise<ReprocessResponse<Crop>> {
  const res = await apiFetch<ReprocessResponse<RawCrop>>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/reprocess`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
  return { ...res, items: (res.items ?? []).map(mapRawCrop) };
}

/** Single-image Reprocess (`POST /images/{image_id}/reprocess`); returns
 *  the post-write `items` of that image, mapped like every other crop. */
export async function reprocessImage(
  imageId: string,
  body: ReprocessOneRequest,
  signal?: AbortSignal,
): Promise<ReprocessResponse<Crop>> {
  const res = await apiFetch<ReprocessResponse<RawCrop>>(
    `${scoped()}/images/${encodeURIComponent(imageId)}/reprocess`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
  return { ...res, items: (res.items ?? []).map(mapRawCrop) };
}

export function getReprocessJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<ReprocessJob> {
  return apiFetch<ReprocessJob>(
    `${scoped()}/reprocess/jobs/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

export function cancelReprocessJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<ReprocessJob> {
  return apiFetch<ReprocessJob>(
    `${scoped()}/reprocess/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' },
    signal,
  );
}
