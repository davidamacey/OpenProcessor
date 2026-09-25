/**
 * Pure config resolution for `/ingest`
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2).
 *
 * BA-2 (`GET {API_PREFIX}/ingest/config`) landed on OpenProcessor
 * c676d2b — every real limit below now comes from the served
 * `IngestConfig` when present. The constants that remain are the
 * documented fallback for a pre-BA-2 backend only (`served ?? interim`),
 * so `resolveIngestConfig(null)` still behaves exactly as it did before
 * BA-2 shipped. `resolveIngestConfig` is the single seam that turns a
 * served `IngestConfig | null` into the resolved values the rest of
 * `$lib/ingest` uses — nothing else in this module tree should read the
 * interim constants directly.
 */

import type { IngestConfig } from '$lib/types';

/**
 * Pre-BA-2 fallback only. `MAX_UPLOAD_IMAGES` in OpenProcessor's
 * `src/routers/curation/ingest_upload.py` (was `ingest.py:259` before
 * BA-1's upload-route split) and `MAX_BATCH` in
 * `scripts/curation/ingest_upload.py:86`.
 */
export const DOCUMENTED_UPLOAD_MAX_IMAGES = 128;

/**
 * `_PathLookupRequest.image_paths.maxItems` — this one IS already
 * declared in the vendored OpenAPI (`ingestContract.test.ts` asserts the
 * two stay equal), so it isn't a BA-2 dependency, just kept alongside
 * the others for a single ingest-limits home.
 */
export const PATH_LOOKUP_MAX = 10_000;

/**
 * Pre-BA-2 fallback only. OpenAPI `images` part description ("Encoded
 * image files (JPEG/PNG)") and `scripts/curation/ingest_upload.py:84`.
 */
export const DOCUMENTED_ACCEPTED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];

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
  uploadPersistsBytes: boolean | null;
  uploadMaxImages: number;
  uploadMaxBytes: number;
  acceptedExtensions: string[];
  pathLookupMax: number;
  batchEnabled: boolean;
  batchMaxItems: number | null;
  batchSourceRoots: string[];
  regionDrainPollIntervalS: number;
  regionDrainStablePolls: number | null;
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

/**
 * Resolve the served `IngestConfig` (BA-2, `getIngestConfig()` — `null`
 * on a pre-BA-2 backend or a genuinely absent ingest router) into the
 * values every ingest module uses. Every served field wins over its
 * interim constant; a `null` `served` (or a missing field on an older
 * backend's partial response) falls back to the documented interim
 * value.
 */
export function resolveIngestConfig(served: IngestConfig | null): ResolvedIngestConfig {
  const maxRequestMb = readRuntimeMaxRequestMb();
  // Leave headroom under the nginx proxy cap for multipart overhead
  // (boundaries, field names, per-part headers).
  const nginxCapBytes = Math.floor(maxRequestMb * 1024 * 1024 * 0.9);
  const servedMaxBytes = served?.upload?.max_bytes_per_request;
  return {
    uploadEnabled: served?.upload?.enabled ?? true,
    uploadPersistsBytes: served?.upload?.persists_bytes ?? null,
    uploadMaxImages:
      served?.upload?.max_images_per_request ?? DOCUMENTED_UPLOAD_MAX_IMAGES,
    // The tighter of the two real limits: the nginx proxy's body-size cap
    // (deployment-owned, §A.4) and the backend's own enforced
    // `upload.max_bytes_per_request` (BA-2) — a chunk that respects only
    // one could still 413 against the other.
    uploadMaxBytes:
      servedMaxBytes !== undefined
        ? Math.min(nginxCapBytes, servedMaxBytes)
        : nginxCapBytes,
    acceptedExtensions:
      served?.upload?.accepted_extensions ?? DOCUMENTED_ACCEPTED_EXTENSIONS,
    pathLookupMax: PATH_LOOKUP_MAX,
    batchEnabled: served?.batch?.enabled ?? false,
    batchMaxItems: served?.batch?.max_items ?? null,
    batchSourceRoots: served?.batch?.source_roots ?? [],
    regionDrainPollIntervalS: served?.region_drain?.poll_interval_s ?? 10,
    regionDrainStablePolls: served?.region_drain?.stable_polls ?? null,
  };
}
