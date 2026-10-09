/**
 * Pure config resolution for `/ingest`
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2).
 *
 * Every backend limit comes from the served `IngestConfig`
 * (`GET {API_PREFIX}/ingest/config`). `resolveIngestConfig` is the single
 * seam that turns it into the resolved values the rest of `$lib/ingest`
 * uses, combining it with the one deployment-owned limit (the nginx body
 * cap).
 */

import type { IngestConfig } from '$lib/types';

/**
 * `_PathLookupRequest.image_paths.maxItems` — this one IS already
 * declared in the vendored OpenAPI (`ingestContract.test.ts` asserts the
 * two stay equal) rather than served by `/ingest/config`.
 */
export const PATH_LOOKUP_MAX = 10_000;

/**
 * Deployment-owned (not a backend value at all — see A.4): the nginx
 * body-size cap this Cropwright deployment was built/started with.
 * `CROPWRIGHT_INGEST_MAX_REQUEST_MB` (docker-entrypoint.sh) substitutes
 * the runtime value into `window.__CROPWRIGHT_INGEST_MAX_REQUEST_MB__`;
 * unset (local dev, or the literal unresolved placeholder) falls back
 * to nginx.conf's default.
 */
export const DEFAULT_INGEST_MAX_REQUEST_MB = 256;

/** UX hint threshold only — never blocks a selection. */
export const LARGE_SELECTION_HINT = 50_000;

/** Default in-flight upload concurrency (§A.4). */
export const DEFAULT_UPLOAD_CONCURRENCY = 2;

export interface ResolvedIngestConfig {
  uploadEnabled: boolean;
  uploadPersistsBytes: boolean;
  uploadMaxImages: number;
  uploadMaxBytes: number;
  acceptedExtensions: string[];
  pathLookupMax: number;
  batchEnabled: boolean;
  batchMaxItems: number;
  batchSourceRoots: string[];
  regionDrainPollIntervalS: number;
  regionDrainStablePolls: number;
}

function readRuntimeMaxRequestMb(): number {
  if (typeof window === 'undefined') return DEFAULT_INGEST_MAX_REQUEST_MB;
  const raw = (window as unknown as Record<string, unknown>)
    .__CROPWRIGHT_INGEST_MAX_REQUEST_MB__;
  if (typeof raw !== 'string' || !raw || raw.startsWith('__')) {
    return DEFAULT_INGEST_MAX_REQUEST_MB;
  }
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_INGEST_MAX_REQUEST_MB;
}

/** Whether `/ingest` offers server-path ingest: the served `batch.enabled`
 *  AND at least one served `batch.source_roots` entry. */
export function serverPathIngestAvailable(config: ResolvedIngestConfig): boolean {
  return config.batchEnabled && config.batchSourceRoots.length > 0;
}

/** Resolve the served `IngestConfig` into the values every ingest module uses. */
export function resolveIngestConfig(served: IngestConfig): ResolvedIngestConfig {
  const maxRequestMb = readRuntimeMaxRequestMb();
  // Leave headroom under the nginx proxy cap for multipart overhead
  // (boundaries, field names, per-part headers).
  const nginxCapBytes = Math.floor(maxRequestMb * 1024 * 1024 * 0.9);
  return {
    uploadEnabled: served.upload.enabled,
    uploadPersistsBytes: served.upload.persists_bytes,
    uploadMaxImages: served.upload.max_images_per_request,
    // The tighter of the two real limits: the nginx proxy's body-size cap
    // (deployment-owned, §A.4) and the backend's own enforced
    // `upload.max_bytes_per_request` — a chunk that respects only one
    // could still 413 against the other.
    uploadMaxBytes: Math.min(nginxCapBytes, served.upload.max_bytes_per_request),
    acceptedExtensions: served.upload.accepted_extensions,
    pathLookupMax: PATH_LOOKUP_MAX,
    batchEnabled: served.batch.enabled,
    batchMaxItems: served.batch.max_items,
    batchSourceRoots: served.batch.source_roots,
    regionDrainPollIntervalS: served.region_drain.poll_interval_s,
    regionDrainStablePolls: served.region_drain.stable_polls,
  };
}
