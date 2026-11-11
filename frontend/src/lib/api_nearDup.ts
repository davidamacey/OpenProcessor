/**
 * Near-duplicates (`{prefix}/near_duplicates`, `{prefix}/near_dup_profiles`):
 * status, runs, groups, decisions and the threshold profile. A run start is
 * a 202 with a job id, followed through `$lib/jobs/followJob`. Types in
 * `$lib/types_nearDup`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  NearDupActiveProfile,
  NearDupDecisionIn,
  NearDupDecisionView,
  NearDupGroupDetail,
  NearDupGroupPage,
  NearDupGroupQuery,
  NearDupLevel,
  NearDupProfileBody,
  NearDupProfileDoc,
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

/** `GET /near_duplicates/groups/{id}`: the members with their cosine to the representative. */
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

const PROFILES = 'near_dup_profiles';

/** `GET /near_dup_profiles/active`: which profile (and revision) the project groups with. */
export function getActiveNearDupProfile(
  signal?: AbortSignal,
): Promise<NearDupActiveProfile> {
  return apiFetch<NearDupActiveProfile>(`${scoped()}/${PROFILES}/active`, {}, signal);
}

/** `GET /near_dup_profiles/{name}`: one revision (the current one when omitted). */
export function getNearDupProfile(
  name: string,
  revision?: number,
  signal?: AbortSignal,
): Promise<NearDupProfileDoc> {
  return apiFetch<NearDupProfileDoc>(
    `${scoped()}/${PROFILES}/${encodeURIComponent(name)}${qs({ revision })}`,
    {},
    signal,
  );
}

/** `PUT /near_dup_profiles/{name}`: save the next revision (OCC on `expected_revision`). */
export function saveNearDupProfile(
  name: string,
  body: NearDupProfileBody,
  expectedRevision: number,
  signal?: AbortSignal,
): Promise<NearDupProfileDoc> {
  return apiFetch<NearDupProfileDoc>(
    `${scoped()}/${PROFILES}/${encodeURIComponent(name)}`,
    {
      method: 'PUT',
      body: JSON.stringify({ body, expected_revision: expectedRevision }),
    },
    signal,
  );
}

/** `POST /near_dup_profiles/{name}/activate`: make a revision the project's profile (OCC on `expected_active`). */
export function activateNearDupProfile(
  name: string,
  revision: number,
  expectedActive: { name: string | null; revision: number | null } | null,
  signal?: AbortSignal,
): Promise<NearDupActiveProfile> {
  return apiFetch<NearDupActiveProfile>(
    `${scoped()}/${PROFILES}/${encodeURIComponent(name)}/activate`,
    {
      method: 'POST',
      body: JSON.stringify({ revision, expected_active: expectedActive }),
    },
    signal,
  );
}
