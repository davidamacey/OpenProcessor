/**
 * Bulk-ingest wrappers (`{prefix}/ingest/bulk*`): start a run over a server
 * folder or manifest, read its progress, cancel it. The run lives in the API;
 * these only start it and read it. Import from `$lib/api_bulkIngest` directly;
 * never re-exported from `api.ts` (that would make the two modules circular).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  BulkIngestConfig,
  BulkIngestFailurePage,
  BulkIngestList,
  BulkIngestRequest,
  BulkIngestRun,
} from '$lib/types_bulkIngest';

export function getBulkIngestConfig(signal?: AbortSignal): Promise<BulkIngestConfig> {
  return apiFetch<BulkIngestConfig>(`${scoped()}/ingest/bulk/config`, {}, signal);
}

/** `POST /ingest/bulk` (202): a new run, or the interrupted run of the same
 *  source continued (`resumed: true`). 409 `bulk_ingest_busy` while one is live. */
export function startBulkIngest(
  body: BulkIngestRequest,
  signal?: AbortSignal,
): Promise<BulkIngestRun> {
  return apiFetch<BulkIngestRun>(
    `${scoped()}/ingest/bulk`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function listBulkIngests(
  params: { page?: number; page_size?: number } = {},
  signal?: AbortSignal,
): Promise<BulkIngestList> {
  return apiFetch<BulkIngestList>(
    `${scoped()}/ingest/bulk${qs({ page: params.page, page_size: params.page_size })}`,
    {},
    signal,
  );
}

export function getBulkIngest(
  runId: string,
  signal?: AbortSignal,
): Promise<BulkIngestRun> {
  return apiFetch<BulkIngestRun>(
    `${scoped()}/ingest/bulk/${encodeURIComponent(runId)}`,
    {},
    signal,
  );
}

export function getBulkIngestFailures(
  runId: string,
  params: { offset?: number; limit?: number } = {},
  signal?: AbortSignal,
): Promise<BulkIngestFailurePage> {
  return apiFetch<BulkIngestFailurePage>(
    `${scoped()}/ingest/bulk/${encodeURIComponent(runId)}/failures${qs({
      offset: params.offset,
      limit: params.limit,
    })}`,
    {},
    signal,
  );
}

/** `POST /ingest/bulk/{run_id}/cancel`: stops after the batches in flight. */
export function cancelBulkIngest(
  runId: string,
  signal?: AbortSignal,
): Promise<BulkIngestRun> {
  return apiFetch<BulkIngestRun>(
    `${scoped()}/ingest/bulk/${encodeURIComponent(runId)}/cancel`,
    { method: 'POST' },
    signal,
  );
}
