/**
 * Ingest-source wrappers (`{prefix}/ingest/sources*`): register a folder or S3
 * bucket/prefix with triggers, read its status, run it now, requeue what gave
 * up. Everything is decided and counted by the backend; these only send and
 * read. Import from `$lib/api_ingestSources` directly; never re-exported from
 * `api.ts` (that would make the two modules circular).
 */
import { apiFetch, scoped } from '$lib/api';
import type { BulkIngestRun } from '$lib/types_bulkIngest';
import type {
  IngestSourceList,
  IngestSourcePatch,
  IngestSourceRequest,
  IngestSourceStatus,
  RetryDeadResult,
} from '$lib/types_ingestSources';

const base = () => `${scoped()}/ingest/sources`;
const one = (id: string) => `${base()}/${encodeURIComponent(id)}`;

export function listIngestSources(signal?: AbortSignal): Promise<IngestSourceList> {
  return apiFetch<IngestSourceList>(base(), {}, signal);
}

export function createIngestSource(
  body: IngestSourceRequest,
  signal?: AbortSignal,
): Promise<IngestSourceStatus> {
  return apiFetch<IngestSourceStatus>(
    base(),
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function updateIngestSource(
  id: string,
  body: IngestSourcePatch,
  signal?: AbortSignal,
): Promise<IngestSourceStatus> {
  return apiFetch<IngestSourceStatus>(
    one(id),
    { method: 'PATCH', body: JSON.stringify(body) },
    signal,
  );
}

export function deleteIngestSource(id: string, signal?: AbortSignal): Promise<void> {
  return apiFetch<void>(one(id), { method: 'DELETE' }, signal);
}

/** `POST .../run` (202): scan the source and ingest what is new. 409
 *  `bulk_ingest_busy` while the project has a run live. */
export function runIngestSource(
  id: string,
  signal?: AbortSignal,
): Promise<BulkIngestRun> {
  return apiFetch<BulkIngestRun>(`${one(id)}/run`, { method: 'POST' }, signal);
}

export function retryDeadObjects(
  id: string,
  signal?: AbortSignal,
): Promise<RetryDeadResult> {
  return apiFetch<RetryDeadResult>(`${one(id)}/retry-dead`, { method: 'POST' }, signal);
}
