/**
 * The job runtime (`{prefix}/jobs`): one job.
 * Every queued start (202 + `job_id`) is followed here. Types live in
 * `$lib/types_jobs`; the polling loop is `$lib/jobs/followJob`.
 */
import { apiFetch, globalApi, scoped } from '$lib/api';
import type { JobView } from '$lib/types_jobs';

/** `GET /jobs/{job_id}`; 404 `job_not_found` for another project's id. */
export function getJob(jobId: string, signal?: AbortSignal): Promise<JobView> {
  return apiFetch<JobView>(`${scoped()}/jobs/${encodeURIComponent(jobId)}`, {}, signal);
}

/** `GET {globalApi()}/jobs/{job_id}`: a deployment-level job (project create and delete belong to no project). */
export function getDeploymentJob(jobId: string, signal?: AbortSignal): Promise<JobView> {
  return apiFetch<JobView>(
    `${globalApi()}/jobs/${encodeURIComponent(jobId)}`,
    {},
    signal,
    {
      global: true,
    },
  );
}
