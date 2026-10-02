/**
 * W9 VLM wrappers: the endpoint registry, the local-model catalog and the
 * per-project activation (any_domain_plan.md §7.8; OpenProcessor f582aa05).
 *
 * The registry, schema, validate, probe, catalog and local-model routes
 * are GLOBAL (`globalApi()`: one registry for the deployment); only
 * activation, rollback, deactivate and the active read are per project
 * (`scoped()`). Import from `$lib/api_vlm` directly; never re-exported
 * from `api.ts` (that would make the two modules circular).
 */
import { apiFetch, globalApi, scoped } from '$lib/api';
import type {
  VlmActivateRequest,
  VlmActiveResponse,
  VlmCatalogResponse,
  VlmDeactivateRequest,
  VlmEndpointCloneRequest,
  VlmEndpointCreate,
  VlmEndpointDoc,
  VlmEndpointList,
  VlmEndpointSaveRequest,
  VlmEndpointSchema,
  VlmLocalSelectRequest,
  VlmLocalStatus,
  VlmProbeResult,
  VlmRevisionsResponse,
  VlmRollbackRequest,
  VlmValidateRequest,
  VlmValidateResponse,
} from '$lib/types_vlm';

const GLOBAL = { global: true } as const;

/** `GET /vlm/endpoints`: the registry, the secret refs and the served
 *  labels. Also the availability probe (a 404/501 = no W9 backend). */
export function listVlmEndpoints(signal?: AbortSignal): Promise<VlmEndpointList> {
  return apiFetch<VlmEndpointList>(`${globalApi()}/vlm/endpoints`, {}, signal, GLOBAL);
}

export function getVlmEndpointSchema(signal?: AbortSignal): Promise<VlmEndpointSchema> {
  return apiFetch<VlmEndpointSchema>(
    `${globalApi()}/vlm/endpoints/schema`,
    {},
    signal,
    GLOBAL,
  );
}

/** `POST /vlm/endpoints/validate?probe=`: a draft's report (always 200; a
 *  probe may 429 `probe_busy`). Never writes. */
export function validateVlmEndpoint(
  body: VlmValidateRequest,
  probe = false,
  signal?: AbortSignal,
): Promise<VlmValidateResponse> {
  return apiFetch<VlmValidateResponse>(
    `${globalApi()}/vlm/endpoints/validate?probe=${probe}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

/** `POST /vlm/endpoints` → 201 the new stored endpoint. */
export function createVlmEndpoint(
  body: VlmEndpointCreate,
  signal?: AbortSignal,
): Promise<VlmEndpointDoc> {
  return apiFetch<VlmEndpointDoc>(
    `${globalApi()}/vlm/endpoints`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

export function getVlmEndpoint(
  name: string,
  signal?: AbortSignal,
): Promise<VlmEndpointDoc> {
  return apiFetch<VlmEndpointDoc>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}`,
    {},
    signal,
    GLOBAL,
  );
}

/** `PUT /vlm/endpoints/{name}`: saves a new revision (OCC on
 *  `expected_revision`). */
export function updateVlmEndpoint(
  name: string,
  body: VlmEndpointSaveRequest,
  signal?: AbortSignal,
): Promise<VlmEndpointDoc> {
  return apiFetch<VlmEndpointDoc>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

/** `DELETE /vlm/endpoints/{name}?expected_revision=` → 204. */
export function deleteVlmEndpoint(
  name: string,
  expectedRevision: number,
  signal?: AbortSignal,
): Promise<void> {
  return apiFetch<void>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}?expected_revision=${expectedRevision}`,
    { method: 'DELETE' },
    signal,
    GLOBAL,
  );
}

export function getVlmEndpointRevisions(
  name: string,
  signal?: AbortSignal,
): Promise<VlmRevisionsResponse> {
  return apiFetch<VlmRevisionsResponse>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}/revisions`,
    {},
    signal,
    GLOBAL,
  );
}

export function getVlmEndpointRevision(
  name: string,
  revision: number,
  signal?: AbortSignal,
): Promise<VlmEndpointDoc> {
  return apiFetch<VlmEndpointDoc>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}/revisions/${revision}`,
    {},
    signal,
    GLOBAL,
  );
}

/** `POST /vlm/endpoints/{name}/clone` → 201 the new stored endpoint. */
export function cloneVlmEndpoint(
  name: string,
  body: VlmEndpointCloneRequest,
  signal?: AbortSignal,
): Promise<VlmEndpointDoc> {
  return apiFetch<VlmEndpointDoc>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}/clone`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

/** `POST /vlm/endpoints/{name}/probe`: one live probe of a saved
 *  endpoint (429 `probe_busy` while another runs). */
export function probeVlmEndpoint(
  name: string,
  signal?: AbortSignal,
): Promise<VlmProbeResult> {
  return apiFetch<VlmProbeResult>(
    `${globalApi()}/vlm/endpoints/${encodeURIComponent(name)}/probe`,
    { method: 'POST' },
    signal,
    GLOBAL,
  );
}

export function getVlmCatalog(signal?: AbortSignal): Promise<VlmCatalogResponse> {
  return apiFetch<VlmCatalogResponse>(`${globalApi()}/vlm/catalog`, {}, signal, GLOBAL);
}

export function getLocalVlm(signal?: AbortSignal): Promise<VlmLocalStatus> {
  return apiFetch<VlmLocalStatus>(`${globalApi()}/vlm/local`, {}, signal, GLOBAL);
}

/** `POST /vlm/local/select` → 202: records the wish; the server keeps
 *  serving what it serves until it is restarted. */
export function selectLocalVlm(
  body: VlmLocalSelectRequest,
  signal?: AbortSignal,
): Promise<VlmLocalStatus> {
  return apiFetch<VlmLocalStatus>(
    `${globalApi()}/vlm/local/select`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
    GLOBAL,
  );
}

export function clearLocalVlmSelection(signal?: AbortSignal): Promise<VlmLocalStatus> {
  return apiFetch<VlmLocalStatus>(
    `${globalApi()}/vlm/local/select`,
    { method: 'DELETE' },
    signal,
    GLOBAL,
  );
}

/** `GET {scoped}/vlm/endpoints/active`: this project's active endpoint. */
export function getActiveVlm(signal?: AbortSignal): Promise<VlmActiveResponse> {
  return apiFetch<VlmActiveResponse>(`${scoped()}/vlm/endpoints/active`, {}, signal);
}

/** `POST {scoped}/vlm/endpoints/{name}/activate` (OCC on
 *  `expected_active`). */
export function activateVlm(
  name: string,
  body: VlmActivateRequest,
  signal?: AbortSignal,
): Promise<VlmActiveResponse> {
  return apiFetch<VlmActiveResponse>(
    `${scoped()}/vlm/endpoints/${encodeURIComponent(name)}/activate`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `POST {scoped}/vlm/endpoints/active/rollback`. */
export function rollbackVlm(
  body: VlmRollbackRequest,
  signal?: AbortSignal,
): Promise<VlmActiveResponse> {
  return apiFetch<VlmActiveResponse>(
    `${scoped()}/vlm/endpoints/active/rollback`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `POST {scoped}/vlm/endpoints/deactivate`: this project runs no VLM. */
export function deactivateVlm(
  body: VlmDeactivateRequest,
  signal?: AbortSignal,
): Promise<VlmActiveResponse> {
  return apiFetch<VlmActiveResponse>(
    `${scoped()}/vlm/endpoints/deactivate`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}
