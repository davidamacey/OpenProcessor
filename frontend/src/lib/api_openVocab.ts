/**
 * SAM 3 open-vocabulary set wrappers (`{prefix}/open_vocab*`, OpenProcessor
 * v0.4.0): list, schema, validate, test, active/rollback/deactivate, CRUD,
 * activate, clone and revisions. Every route is project-scoped. Import from
 * `$lib/api_openVocab` directly; never re-exported from `api.ts` (that would
 * make the two modules circular). Callers render `configErrorText`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  ActiveConfigResponse,
  ActiveRef,
  ValidationReport,
} from '$lib/types_config';
import type {
  OpenVocabActivateRequest,
  OpenVocabActivateResponse,
  OpenVocabCloneRequest,
  OpenVocabCreateRequest,
  OpenVocabDoc,
  OpenVocabList,
  OpenVocabRevisionsResponse,
  OpenVocabSaveRequest,
  OpenVocabSchema,
  OpenVocabTestRequest,
  OpenVocabTestResponse,
  OpenVocabValidateRequest,
} from '$lib/types_openVocab';

/** `GET /open_vocab?include_templates=true`: the sets, the clone-only
 *  templates and the active ref. Also the availability probe. */
export function listOpenVocab(signal?: AbortSignal): Promise<OpenVocabList> {
  return apiFetch<OpenVocabList>(
    `${scoped()}/open_vocab${qs({ include_templates: true })}`,
    {},
    signal,
  );
}

export function getOpenVocabSchema(signal?: AbortSignal): Promise<OpenVocabSchema> {
  return apiFetch<OpenVocabSchema>(`${scoped()}/open_vocab/schema`, {}, signal);
}

/** `POST /open_vocab` → 201 the new stored set. */
export function createOpenVocab(
  req: OpenVocabCreateRequest,
  signal?: AbortSignal,
): Promise<OpenVocabDoc> {
  return apiFetch<OpenVocabDoc>(
    `${scoped()}/open_vocab`,
    { method: 'POST', body: JSON.stringify(req) },
    signal,
  );
}

export function getOpenVocab(name: string, signal?: AbortSignal): Promise<OpenVocabDoc> {
  return apiFetch<OpenVocabDoc>(
    `${scoped()}/open_vocab/${encodeURIComponent(name)}`,
    {},
    signal,
  );
}

/** `PUT /open_vocab/{name}`: saves a new revision (OCC on `expected_revision`). */
export function updateOpenVocab(
  name: string,
  req: OpenVocabSaveRequest,
): Promise<OpenVocabDoc> {
  return apiFetch<OpenVocabDoc>(`${scoped()}/open_vocab/${encodeURIComponent(name)}`, {
    method: 'PUT',
    body: JSON.stringify(req),
  });
}

/** `DELETE /open_vocab/{name}?expected_revision=` → 204. */
export function deleteOpenVocab(name: string, expectedRevision: number): Promise<void> {
  return apiFetch<void>(
    `${scoped()}/open_vocab/${encodeURIComponent(name)}${qs({ expected_revision: expectedRevision })}`,
    { method: 'DELETE' },
  );
}

/** `POST /open_vocab/validate`: a draft's report, never a write.
 *  `for_activation` is sent only when true. */
export function validateOpenVocab(
  req: OpenVocabValidateRequest,
  forActivation = false,
  signal?: AbortSignal,
): Promise<ValidationReport> {
  return apiFetch<ValidationReport>(
    `${scoped()}/open_vocab/validate${qs({ for_activation: forActivation || undefined })}`,
    { method: 'POST', body: JSON.stringify(req) },
    signal,
  );
}

/** `POST /open_vocab/test`: one unsaved target on one image; writes nothing. */
export function testOpenVocab(
  req: OpenVocabTestRequest,
  signal?: AbortSignal,
): Promise<OpenVocabTestResponse> {
  return apiFetch<OpenVocabTestResponse>(
    `${scoped()}/open_vocab/test`,
    { method: 'POST', body: JSON.stringify(req) },
    signal,
  );
}

export function getActiveOpenVocab(signal?: AbortSignal): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/open_vocab/active`, {}, signal);
}

/** `POST /open_vocab/{name}/activate` (OCC on `expected_active`). */
export function activateOpenVocab(
  name: string,
  req: OpenVocabActivateRequest,
): Promise<OpenVocabActivateResponse> {
  return apiFetch<OpenVocabActivateResponse>(
    `${scoped()}/open_vocab/${encodeURIComponent(name)}/activate`,
    { method: 'POST', body: JSON.stringify(req) },
  );
}

/** `POST /open_vocab/active/rollback`: re-activates the previous set. */
export function rollbackOpenVocab(req: {
  expected_active: ActiveRef;
}): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/open_vocab/active/rollback`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/** `POST /open_vocab/deactivate`: no set active (OCC). */
export function deactivateOpenVocab(req: {
  expected_active: ActiveRef;
}): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/open_vocab/deactivate`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/** `POST /open_vocab/{name}/clone` → 201 the new stored set. */
export function cloneOpenVocab(
  name: string,
  req: OpenVocabCloneRequest,
): Promise<OpenVocabDoc> {
  return apiFetch<OpenVocabDoc>(
    `${scoped()}/open_vocab/${encodeURIComponent(name)}/clone`,
    { method: 'POST', body: JSON.stringify(req) },
  );
}

export function getOpenVocabRevisions(
  name: string,
  signal?: AbortSignal,
): Promise<OpenVocabRevisionsResponse> {
  return apiFetch<OpenVocabRevisionsResponse>(
    `${scoped()}/open_vocab/${encodeURIComponent(name)}/revisions`,
    {},
    signal,
  );
}

export function getOpenVocabRevision(
  name: string,
  revision: number,
  signal?: AbortSignal,
): Promise<OpenVocabDoc> {
  return apiFetch<OpenVocabDoc>(
    `${scoped()}/open_vocab/${encodeURIComponent(name)}/revisions/${encodeURIComponent(String(revision))}`,
    {},
    signal,
  );
}
