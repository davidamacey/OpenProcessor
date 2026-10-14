/**
 * Prompt packs: list, schema, validate, revisions, clone, update, delete,
 * activate and roll back (`/config/prompt_packs*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  PackUpdateRequest,
  PackValidateRequest,
  PromptPackDoc,
  PromptPackList,
  PromptPackSchema,
} from '$lib/types_packs';
import type {
  ActivateResponse,
  ActiveConfigResponse,
  ActiveRef,
  ConfigActivateRequest,
  ConfigCloneRequest,
  ConfigRevisionList,
  ValidationReport,
} from '$lib/types_config';

// -- Prompt packs (and test-on-crop) -------------------------------------

/** `GET /prompt_packs`: every pack (builtin, file, stored) plus the
 *  clone-only templates and the active ref. Also the pack gate's probe. */
export function listPromptPacks(signal?: AbortSignal): Promise<PromptPackList> {
  return apiFetch<PromptPackList>(`${scoped()}/prompt_packs`, {}, signal);
}

export function getPromptPackSchema(signal?: AbortSignal): Promise<PromptPackSchema> {
  return apiFetch<PromptPackSchema>(`${scoped()}/prompt_packs/schema`, {}, signal);
}

/** `POST /prompt_packs/validate`: a draft's report. Never writes and
 *  never 422s; a reserved or taken `name` is an issue in the report. */
export function validatePromptPack(
  body: PackValidateRequest,
  signal?: AbortSignal,
): Promise<ValidationReport> {
  return apiFetch<ValidationReport>(
    `${scoped()}/prompt_packs/validate`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function getPromptPack(
  name: string,
  signal?: AbortSignal,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}`,
    {},
    signal,
  );
}

export function getPromptPackRevisions(
  name: string,
  signal?: AbortSignal,
): Promise<ConfigRevisionList> {
  return apiFetch<ConfigRevisionList>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/revisions`,
    {},
    signal,
  );
}

export function getPromptPackRevision(
  name: string,
  revision: number,
  signal?: AbortSignal,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/revisions/${encodeURIComponent(String(revision))}`,
    {},
    signal,
  );
}

/** `POST /prompt_packs/{name}/clone` → 201 the new stored pack. */
export function clonePromptPack(
  name: string,
  body: ConfigCloneRequest,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/clone`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}

/** `PUT /prompt_packs/{name}`: saves a new revision (OCC on
 *  `expected_revision`; 409 `revision_conflict` carries the current one). */
export function updatePromptPack(
  name: string,
  body: PackUpdateRequest,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(`${scoped()}/prompt_packs/${encodeURIComponent(name)}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

/** `DELETE /prompt_packs/{name}?expected_revision=` → 204. */
export function deletePromptPack(name: string, expectedRevision: number): Promise<void> {
  return apiFetch<void>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}${qs({ expected_revision: expectedRevision })}`,
    {
      method: 'DELETE',
    },
  );
}

export function getActivePromptPack(signal?: AbortSignal): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/prompt_packs/active`, {}, signal);
}

/** `POST /prompt_packs/{name}/activate` (OCC on `expected_active`). */
export function activatePromptPack(
  name: string,
  body: ConfigActivateRequest,
): Promise<ActivateResponse> {
  return apiFetch<ActivateResponse>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/activate`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}

/** `POST /prompt_packs/active/rollback`: re-activates the previous pack. */
export function rollbackPromptPack(body: {
  expected_active: ActiveRef;
}): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/prompt_packs/active/rollback`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
