/**
 * Project list and lifecycle (global `/projects` routes) and the per-project
 * pipeline pause/resume.
 */
import { apiFetch, globalApi, qs } from '$lib/api';
import { structuredDetail } from '$lib/api_errors';
import type {
  ArchiveRequest,
  CloneSettingsRequest,
  CreateProjectRequest,
  DeleteDryRunResponse,
  PatchProjectRequest,
  PipelinePauseState,
  ProjectErrorDetail,
  ProjectLifecycleResponse,
  ProjectRecordResponse,
  ProjectsResponse,
} from '$lib/types_projects';

/** `GET {globalApi()}/projects` — the switcher vocabulary, global
 *  (unscoped). The one read every project-scoped call depends on: a
 *  project's `prefix` here is what `setScopedPrefix()` is seeded with.
 *  `includeArchived` adds the served `archived` projects (the list
 *  membership per status is the server's, never filtered here). */
export function getProjects(
  signal?: AbortSignal,
  includeArchived = false,
): Promise<ProjectsResponse> {
  return apiFetch<ProjectsResponse>(
    `${globalApi()}/projects${qs({ include_archived: includeArchived ? true : undefined })}`,
    {},
    signal,
    { global: true },
  );
}

// -- project lifecycle (P3; all GLOBAL, never scoped) ---------------------

/** `GET {globalApi()}/projects/{slug}` — the record for a slug the
 *  default list doesn't carry (an archived project's deep link). */
export function getProject(
  slug: string,
  signal?: AbortSignal,
): Promise<ProjectRecordResponse> {
  return apiFetch<ProjectRecordResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}`,
    {},
    signal,
    { global: true },
  );
}

export function createProject(
  body: CreateProjectRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function patchProject(
  slug: string,
  body: PatchProjectRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}`,
    { method: 'PATCH', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function archiveProject(
  slug: string,
  body: ArchiveRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}/archive`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function unarchiveProject(
  slug: string,
  body: ArchiveRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}/unarchive`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function cloneProjectSettings(
  slug: string,
  body: CloneSettingsRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}/clone_settings`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

/** `DELETE …?dry_run=true` — the served report, writes nothing. */
export function deleteProjectDryRun(slug: string): Promise<DeleteDryRunResponse> {
  return apiFetch<DeleteDryRunResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}${qs({ dry_run: true })}`,
    { method: 'DELETE' },
    undefined,
    { global: true },
  );
}

/** `DELETE …?confirm=<slug>` — a real, guarded delete. Answers 202 with
 *  the `deleting` record; the removal finishes in the background. */
export function deleteProject(
  slug: string,
  confirm: string,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}${qs({ confirm })}`,
    { method: 'DELETE' },
    undefined,
    { global: true },
  );
}

/**
 * The structured `{detail: {error, message, ...}}` body every project
 * route answers an error with (`ConfigErrorDetail`), or `null` when the
 * error isn't one (a network failure, a plain-string detail, a pydantic
 * validation list). The UI renders `message` verbatim and branches only
 * on the served `error` code.
 */
export const projectErrorDetail = (e: unknown): ProjectErrorDetail | null =>
  structuredDetail<ProjectErrorDetail>(e);

/**
 * A project's own served API prefix, verbatim. The only way to address a
 * project OTHER than the active one (the `/projects` page acts on each
 * row's own project): the prefix is the served
 * `ProjectSummary.prefix`, never assembled from a slug.
 */
export function projectPrefix(project: { prefix: string }): string {
  return project.prefix;
}

/** `GET {prefix}/pause`. A row action on `/projects`, so `global`: it
 *  belongs to that row's project, not the active one, and is never
 *  dropped as stale when the active project changes. */
export function getProjectPause(
  project: { prefix: string },
  signal?: AbortSignal,
): Promise<PipelinePauseState> {
  return apiFetch<PipelinePauseState>(`${projectPrefix(project)}/pause`, {}, signal, {
    global: true,
  });
}

/** `POST {prefix}/pause` — workers skip the project until resumed. */
export function pauseProject(project: { prefix: string }): Promise<PipelinePauseState> {
  return apiFetch<PipelinePauseState>(
    `${projectPrefix(project)}/pause`,
    { method: 'POST' },
    undefined,
    { global: true },
  );
}

/** `POST {prefix}/resume`. */
export function resumeProject(project: { prefix: string }): Promise<PipelinePauseState> {
  return apiFetch<PipelinePauseState>(
    `${projectPrefix(project)}/resume`,
    { method: 'POST' },
    undefined,
    { global: true },
  );
}
