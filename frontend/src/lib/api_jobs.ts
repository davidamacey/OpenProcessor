/**
 * The job runtime (`{prefix}/jobs`): one job and the project's job list.
 * Every queued start (202 + `job_id`) is followed here. Types live in
 * `$lib/types_jobs`; the polling loop is `$lib/jobs/followJob`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { JobPage, JobStatus, JobView } from '$lib/types_jobs';

/** `GET /jobs/{job_id}`; 404 `job_not_found` for another project's id. */
export function getJob(jobId: string, signal?: AbortSignal): Promise<JobView> {
  return apiFetch<JobView>(`${scoped()}/jobs/${encodeURIComponent(jobId)}`, {}, signal);
}

/** `GET /jobs`, newest first; `cursor` is the previous page's `next_cursor`. */
export function listJobs(
  params: {
    kind?: string[];
    status?: JobStatus[];
    limit?: number;
    cursor?: string | null;
  } = {},
  signal?: AbortSignal,
): Promise<JobPage> {
  return apiFetch<JobPage>(
    `${scoped()}/jobs${qs({
      kind: params.kind,
      status: params.status,
      limit: params.limit,
      cursor: params.cursor,
    })}`,
    {},
    signal,
  );
}
