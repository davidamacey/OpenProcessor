/**
 * Near-duplicates (`{prefix}/near_duplicates`, `{prefix}/near_dup_profiles`):
 * status, runs, groups, decisions and the threshold profile. A run start is
 * a 202 with a job id, followed through `$lib/jobs/followJob`. Types in
 * `$lib/types_nearDup`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  NearDupDecisionIn,
  NearDupDecisionPage,
  NearDupDecisionQuery,
  NearDupDecisionView,
  NearDupGroupDetail,
  NearDupGroupPage,
  NearDupGroupQuery,
  NearDupLevel,
  NearDupRunPage,
  NearDupRunRequest,
  NearDupRunStarted,
  NearDupStatus,
} from '$lib/types_nearDup';

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

/** `GET /near_duplicates/status`: per level the current run, the pending count and why it is stale. */
export function getNearDupStatus(signal?: AbortSignal): Promise<NearDupStatus> {
  return apiFetch<NearDupStatus>(`${scoped()}/near_duplicates/status`, {}, signal);
}

/** `GET /near_duplicates/runs`: the run history, newest first. */
export function listNearDupRuns(
  level?: NearDupLevel,
  limit?: number,
  signal?: AbortSignal,
): Promise<NearDupRunPage> {
  return apiFetch<NearDupRunPage>(
    `${scoped()}/near_duplicates/runs${qs({ level, limit })}`,
    {},
    signal,
  );
}

/** `GET /near_duplicates/groups`: a page of the current run's groups, largest first. */
export function listNearDupGroups(
  query: NearDupGroupQuery,
  signal?: AbortSignal,
): Promise<NearDupGroupPage> {
  return apiFetch<NearDupGroupPage>(
    `${scoped()}/near_duplicates/groups${qs({
      level: query.level,
      purpose: query.purpose,
      min_size: query.min_size,
      has_holdout: query.has_holdout,
      cursor: query.cursor,
      limit: query.limit,
    })}`,
    {},
    signal,
  );
}

/** `GET /near_duplicates/groups/{id}`: the members with their picture, class, holdout state and pin. */
export function getNearDupGroup(
  level: NearDupLevel,
  groupId: string,
  signal?: AbortSignal,
): Promise<NearDupGroupDetail> {
  return apiFetch<NearDupGroupDetail>(
    `${scoped()}/near_duplicates/groups/${encodeURIComponent(groupId)}${qs({ level })}`,
    {},
    signal,
  );
}

/** `POST /near_duplicates/decisions`: pin two members together or apart for every later run. */
export function decideNearDup(
  body: NearDupDecisionIn,
  signal?: AbortSignal,
): Promise<NearDupDecisionView> {
  return apiFetch<NearDupDecisionView>(
    `${scoped()}/near_duplicates/decisions`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `GET /near_duplicates/decisions`: the pinned pairs, newest first. */
export function listNearDupDecisions(
  query: NearDupDecisionQuery = {},
  signal?: AbortSignal,
): Promise<NearDupDecisionPage> {
  return apiFetch<NearDupDecisionPage>(
    `${scoped()}/near_duplicates/decisions${qs({
      level: query.level,
      member_id: query.member_id,
      cursor: query.cursor,
      limit: query.limit,
    })}`,
    {},
    signal,
  );
}

/** `DELETE /near_duplicates/decisions/{id}`: take a decision back. */
export async function undecideNearDup(
  decisionId: string,
  signal?: AbortSignal,
): Promise<void> {
  await apiFetch<void>(
    `${scoped()}/near_duplicates/decisions/${encodeURIComponent(decisionId)}`,
    { method: 'DELETE' },
    signal,
  );
}
