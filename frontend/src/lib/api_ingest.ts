/**
 * Ingest status, region drain, path lookup, upload and batch ingest
 * (`/ingest/*`).
 */
import { apiFetch, assertNonEmptyBatch, scoped } from '$lib/api';
import type {
  BatchIngestResponse,
  IngestBatchRequest,
  IngestConfig,
  IngestPathLookupResponse,
  IngestStatus,
  IngestUploadRequest,
  RegionDrain,
} from '$lib/types_ingest';

// ===========================================================================
// Ingest ({API_PREFIX}/ingest/*) — bring images into the pool.
// docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.2.

export function getIngestStatus(signal?: AbortSignal): Promise<IngestStatus> {
  return apiFetch<IngestStatus>(`${scoped()}/ingest/status`, {}, signal);
}

export function getRegionDrain(signal?: AbortSignal): Promise<RegionDrain> {
  return apiFetch<RegionDrain>(`${scoped()}/ingest/region_drain`, {}, signal);
}

export function ingestPathLookup(
  ids: string[],
  signal?: AbortSignal,
): Promise<IngestPathLookupResponse> {
  return apiFetch<IngestPathLookupResponse>(
    `${scoped()}/ingest/path_lookup`,
    { method: 'POST', body: JSON.stringify({ image_paths: ids }) },
    signal,
  );
}

/**
 * `POST {API_PREFIX}/ingest/upload` — multipart. The browser must set its
 * own `Content-Type` boundary; see the `apiFetch` FormData carve-out above.
 */
export function ingestUpload(
  req: IngestUploadRequest,
  signal?: AbortSignal,
): Promise<BatchIngestResponse> {
  const fd = new FormData();
  for (const f of req.files) fd.append('images', f, f.name);
  fd.append('image_paths', JSON.stringify(req.identifiers));
  fd.append('source', req.source);
  return apiFetch<BatchIngestResponse>(
    `${scoped()}/ingest/upload`,
    { method: 'POST', body: fd },
    signal,
  );
}

export async function ingestBatch(
  req: IngestBatchRequest,
  signal?: AbortSignal,
): Promise<BatchIngestResponse> {
  assertNonEmptyBatch('ingest', req.items);
  return apiFetch<BatchIngestResponse>(
    `${scoped()}/ingest/batch`,
    { method: 'POST', body: JSON.stringify(req) },
    signal,
  );
}

/**
 * Typed ingest capability + limits, actually enforced by
 * `/ingest/upload`/`/ingest/batch`/`/ingest/region_drain`.
 */
export function getIngestConfig(signal?: AbortSignal): Promise<IngestConfig> {
  return apiFetch<IngestConfig>(`${scoped()}/ingest/config`, {}, signal);
}
