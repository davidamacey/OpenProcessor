/**
 * Near-duplicate status and runs (`{prefix}/near_duplicates`). A run start is
 * a 202 with a job id, followed through `$lib/jobs/followJob`. Types in
 * `$lib/types_nearDup`.
 */
import { apiFetch, scoped } from '$lib/api';
import type {
  NearDupRunRequest,
  NearDupRunStarted,
  NearDupStatus,
} from '$lib/types_nearDup';

/** `GET /near_duplicates/status`: per level, the current run and why it is stale. */
export function getNearDupStatus(signal?: AbortSignal): Promise<NearDupStatus> {
  return apiFetch<NearDupStatus>(`${scoped()}/near_duplicates/status`, {}, signal);
}

/** `POST /near_duplicates/runs`: a full build or the incremental pass of one level. */
export function startNearDupRun(
  body: NearDupRunRequest,
  signal?: AbortSignal,
): Promise<NearDupRunStarted> {
  return apiFetch<NearDupRunStarted>(
    `${scoped()}/near_duplicates/runs`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}
