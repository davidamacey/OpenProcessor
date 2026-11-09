/**
 * The job runtime (`{prefix}/jobs`): one job.
 * Every queued start (202 + `job_id`) is followed here. Types live in
 * `$lib/types_jobs`; the polling loop is `$lib/jobs/followJob`.
 */
import { apiFetch, scoped } from '$lib/api';
import type { JobView } from '$lib/types_jobs';

/** `GET /jobs/{job_id}`; 404 `job_not_found` for another project's id. */
export function getJob(jobId: string, signal?: AbortSignal): Promise<JobView> {
  return apiFetch<JobView>(`${scoped()}/jobs/${encodeURIComponent(jobId)}`, {}, signal);
}
