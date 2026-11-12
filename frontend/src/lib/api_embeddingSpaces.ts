/** `{prefix}/embedding/spaces`: the spaces a project holds, and the writes that build, promote and retire them. Types in `$lib/types_embeddingSpaces`. */
import { apiFetch, globalApi, qs, scoped } from '$lib/api';
import type {
  EmbeddingBuildRequest,
  EmbeddingModelEntry,
  EmbeddingModelListResponse,
  EmbeddingPromoteRequest,
  EmbeddingPromoteResponse,
  EmbeddingReindexRun,
  EmbeddingRunCancelResponse,
  EmbeddingSpaceListResponse,
  EmbeddingSpaceRegisterRequest,
  EmbeddingSpaceStartResponse,
  EmbeddingSpaceView,
  EmbeddingTarget,
} from '$lib/types_embeddingSpaces';

/** `GET /embedding/spaces`; `target` narrows to one target (the server default is all). */
export async function getEmbeddingSpaces(
  target?: EmbeddingTarget,
  signal?: AbortSignal,
): Promise<EmbeddingSpaceView[]> {
  const res = await apiFetch<EmbeddingSpaceListResponse>(
    `${scoped()}/embedding/spaces${qs({ target })}`,
    {},
    signal,
  );
  return res.spaces;
}

/** `GET /curation/embedding/models`: the encoders this deployment can serve (not project-scoped). */
export async function getEmbeddingModels(
  signal?: AbortSignal,
): Promise<EmbeddingModelEntry[]> {
  const res = await apiFetch<EmbeddingModelListResponse>(
    `${globalApi()}/embedding/models`,
    {},
    signal,
    { global: true },
  );
  return res.models;
}

/** `POST /embedding/spaces` -> 202. A repeat with the same `Idempotency-Key` answers the first job. */
export function registerEmbeddingSpace(
  req: EmbeddingSpaceRegisterRequest,
  idempotencyKey: string,
): Promise<EmbeddingSpaceStartResponse> {
  return apiFetch<EmbeddingSpaceStartResponse>(`${scoped()}/embedding/spaces`, {
    method: 'POST',
    body: JSON.stringify(req),
    headers: { 'Idempotency-Key': idempotencyKey },
  });
}

/** `POST /embedding/spaces/{key}/build?target=` -> 202: restart a failed or cancelled build, or backfill. */
export function buildEmbeddingSpace(
  key: string,
  target: EmbeddingTarget,
  req: EmbeddingBuildRequest,
): Promise<EmbeddingSpaceStartResponse> {
  return apiFetch<EmbeddingSpaceStartResponse>(
    `${scoped()}/embedding/spaces/${encodeURIComponent(key)}/build${qs({ target })}`,
    {
      method: 'POST',
      body: JSON.stringify(req),
    },
  );
}

/** `POST /embedding/spaces/{key}/promote?target=`: 409 `embedding_space_not_ready` or `embedding_evaluation_missing`. */
export function promoteEmbeddingSpace(
  key: string,
  target: EmbeddingTarget,
  req: EmbeddingPromoteRequest,
): Promise<EmbeddingPromoteResponse> {
  return apiFetch<EmbeddingPromoteResponse>(
    `${scoped()}/embedding/spaces/${encodeURIComponent(key)}/promote${qs({ target })}`,
    {
      method: 'POST',
      body: JSON.stringify(req),
    },
  );
}

/** `POST /embedding/spaces/{key}/retire?target=` -> 202: a non-primary space, with its clear run. */
export function retireEmbeddingSpace(
  key: string,
  target: EmbeddingTarget,
): Promise<EmbeddingSpaceStartResponse> {
  return apiFetch<EmbeddingSpaceStartResponse>(
    `${scoped()}/embedding/spaces/${encodeURIComponent(key)}/retire${qs({ target })}`,
    {
      method: 'POST',
    },
  );
}

/** `GET /embedding/reindex-runs/{run_id}`. */
export function getEmbeddingReindexRun(
  runId: string,
  signal?: AbortSignal,
): Promise<EmbeddingReindexRun> {
  return apiFetch<EmbeddingReindexRun>(
    `${scoped()}/embedding/reindex-runs/${encodeURIComponent(runId)}`,
    {},
    signal,
  );
}

/** `POST /embedding/reindex-runs/{run_id}/cancel?abandon=`. */
export function cancelEmbeddingReindexRun(
  runId: string,
  abandon: boolean,
): Promise<EmbeddingRunCancelResponse> {
  return apiFetch<EmbeddingRunCancelResponse>(
    `${scoped()}/embedding/reindex-runs/${encodeURIComponent(runId)}/cancel${qs({ abandon: abandon || undefined })}`,
    { method: 'POST' },
  );
}
