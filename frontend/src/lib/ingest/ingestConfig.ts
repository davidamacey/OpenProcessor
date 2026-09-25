/**
 * Pure config resolution for `/ingest`
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2).
 *
 * The backend does not yet serve `GET {API_PREFIX}/ingest/config` (BA-2),
 * so every numeric limit below is an INTERIM, documented client
 * constant, each carrying a `TODO(BA-2)` pointing at the backend source
 * line it was read off. `resolveIngestConfig` is the single seam that
 * turns a served `IngestConfig` (once it exists) into the resolved
 * values the rest of `$lib/ingest` uses — nothing else in this module
 * tree should read the interim constants directly.
 */

import type { IngestConfig } from '$lib/types';

/**
 * `MAX_UPLOAD_IMAGES` in OpenProcessor's `src/routers/curation/ingest.py:259`
 * and `MAX_BATCH` in `scripts/curation/ingest_upload.py:86`.
 * TODO(BA-2): delete once `IngestConfig.upload.max_images_per_request` is served.
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
 * OpenAPI `images` part description ("Encoded image files (JPEG/PNG)")
 * and `scripts/curation/ingest_upload.py:84`.
 * TODO(BA-2): delete once `IngestConfig.upload.accepted_extensions` is served.
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
 * Resolve the served `IngestConfig` (BA-2, currently always `null` since
 * no wrapper calls `/ingest/config` yet) into the values every ingest
 * module uses. Every served field wins over its interim constant; an
 * absent field falls back to the documented interim value.
 */
export function resolveIngestConfig(served: IngestConfig | null): ResolvedIngestConfig {
  const maxRequestMb = readRuntimeMaxRequestMb();
  return {
    uploadEnabled: served?.upload?.enabled ?? true,
    uploadPersistsBytes: served?.upload?.persists_bytes ?? null,
    uploadMaxImages:
      served?.upload?.max_images_per_request ?? DOCUMENTED_UPLOAD_MAX_IMAGES,
    // Leave headroom under the nginx cap for multipart overhead
    // (boundaries, field names, per-part headers).
    uploadMaxBytes: Math.floor(maxRequestMb * 1024 * 1024 * 0.9),
    acceptedExtensions:
      served?.upload?.accepted_extensions ?? DOCUMENTED_ACCEPTED_EXTENSIONS,
    pathLookupMax: PATH_LOOKUP_MAX,
    batchEnabled: served?.batch?.enabled ?? false,
    batchMaxItems: served?.batch?.max_items_per_request ?? null,
    batchSourceRoots: served?.batch?.source_roots ?? [],
    regionDrainPollIntervalS: served?.region_drain?.poll_interval_s ?? 10,
  };
}
