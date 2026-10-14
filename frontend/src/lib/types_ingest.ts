/**
 * Ingest wire types (`{API_PREFIX}/ingest/*`).
 */
import type { IngestDetectorInfo, IngestPolicy } from '$lib/types_detector';

// -- ingest --
// docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.1. Wire
// shapes for OpenProcessor's `{API_PREFIX}/ingest/*` router
// (`src/routers/curation/ingest.py`, `_common_models.py` on the
// backend). BA-1..BA-7 (OpenProcessor #36, c5c606f) landed: upload bytes
// are persisted (`source_identifier` alongside `image_path`), `error_kind`
// is a stable code on every failed result, `GET {API_PREFIX}/ingest/config`
// is real and typed, and `GET {API_PREFIX}/ingest/region_drain` serves a
// server-computed `drained` verdict.

export type IngestItemStatus = 'success' | 'duplicate' | 'failed';

// BA-7: stable machine codes served alongside `error`'s free-text message
// on a failed result. Not an exhaustive union on the wire (a future
// failure mode may still surface a message without extending this list),
// so this type stays a documented `string`, not a literal union.
export type IngestErrorKind =
  | 'empty'
  | 'unservable_path'
  | 'unsupported_type'
  | 'too_large'
  | 'decode_failed'
  | 'detector_infer'
  | 'bulk_index';

export interface IngestImageResult {
  status: IngestItemStatus;
  image_id: string;
  image_path: string;
  imohash: string;
  n_crops: number;
  n_regions: number;
  error: string | null;
  /** BA-7. Always present when `status === 'failed'`, null otherwise. */
  error_kind: IngestErrorKind | string | null;
  /**
   * BA-1: the client-supplied identifier for a byte-upload ingest, where
   * `image_path` is now the server-persisted path. Null for a
   * server-path ingest (`image_path` already IS the client-meaningful
   * identifier).
   */
  source_identifier: string | null;
  /** v0.4.0: items that got a vector, items the embedding policy (or an
   *  encoder failure) left without one, and boxes the ingest policy's
   *  detect filter dropped. */
  n_embedded: number;
  n_not_embedded: number;
  /** Of `n_not_embedded`, how many failed in the encoder (the rest were
   *  not selected or deferred by the policy). */
  n_embed_failed: number;
  n_filtered: number;
  /**
   * OpenProcessor d72cc63: set when the image itself ingested but the
   * optional secondary detector failed on it (so it carries only the
   * primary detector's crops). Null otherwise.
   */
  secondary_detector_error?: string | null;
}

export interface BatchIngestSummary {
  successful: number;
  duplicates: number;
  failed: number;
  crops_indexed: number;
  /** v0.4.0: the per-image `n_embedded` / `n_not_embedded` / `n_filtered`
   *  summed over the batch. */
  n_embedded: number;
  n_not_embedded: number;
  /** Of `n_not_embedded`, how many failed in the encoder (the rest were
   *  not selected or deferred by the policy). */
  n_embed_failed: number;
  n_filtered: number;
  /** d72cc63: how many results carry a `secondary_detector_error`. */
  secondary_detector_failures?: number;
}

export interface BatchIngestResponse {
  status: 'success' | 'partial' | 'error';
  summary: BatchIngestSummary;
  results: IngestImageResult[];
}

export interface IngestStatusBucket {
  key: string;
  doc_count: number;
  key_as_string?: string;
}

export interface IngestStatus {
  total: number;
  by_source: IngestStatusBucket[];
  by_day: IngestStatusBucket[];
}

export interface RegionDrain {
  pending_detection: number;
  pending_verification: number;
  total_unfinished: number;
  /**
   * BA-3 (landed, c5c606f): true once `total_unfinished` has read 0 for
   * `IngestConfig.region_drain.stable_polls` consecutive polls of this
   * endpoint — computed server-side, not a client-invented window.
   */
  drained: boolean;
  /** Seconds since the last non-zero reading (0 while still non-zero). */
  stable_for_s: number;
  /** ISO timestamp of this poll. */
  observed_at: string;
  /**
   * V-1 (OpenProcessor d72cc63): the active region profile's Triton
   * dependencies and whether each is ready now; empty with no profile.
   */
  region_dependencies?: RegionDependencyStatus[];
  /**
   * V-1: a served, human-readable line when items are pending AND a
   * dependency is down; null otherwise. Rendered verbatim.
   */
  stall_reason?: string | null;
}

export interface RegionDependencyStatus {
  role: string;
  model: string;
  ready: boolean;
  detail: string;
  unavailable_since: string | null;
}

export interface IngestPathLookupResponse {
  known_paths: Record<string, string>;
}

export interface IngestBatchItem {
  path: string;
  source?: string;
}

export interface IngestBatchRequest {
  items: IngestBatchItem[];
}

export interface IngestUploadRequest {
  files: File[];
  identifiers: string[];
  source: string;
}

/**
 * BA-2 (landed, c5c606f): `GET {API_PREFIX}/ingest/config`'s served
 * shape — real, enforced limits, not client-guessed constants. Field
 * names/nesting match `IngestConfigResponse` in
 * `src/routers/curation/_common_models.py` on the backend exactly.
 */
export interface IngestConfig {
  upload: {
    enabled: boolean;
    max_images_per_request: number;
    max_bytes_per_request: number;
    accepted_extensions: string[];
    persists_bytes: boolean;
  };
  batch: { enabled: boolean; max_items: number; source_roots: string[] };
  region_drain: { poll_interval_s: number; stable_polls: number };
  /** v0.4.0: the deployment detector, `null` when none is reported. */
  detector?: IngestDetectorInfo | null;
  /** v0.4.0: the project ingest policy echo. */
  policy?: IngestPolicy;
}
