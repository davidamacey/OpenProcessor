/**
 * Wire types for bulk ingest of a folder or manifest
 * (`{prefix}/ingest/bulk*`). Pinned key-for-key to the vendored OpenAPI by
 * `contract/bulkIngestContract.test.ts`.
 */

import type { S3Location } from '$lib/types_ingestSources';

export type BulkIngestStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'completed_with_errors'
  | 'failed'
  | 'cancelled'
  | 'interrupted';

export type BulkSourceKind = 'folder' | 'manifest' | 'queue';

/** `GET /ingest/bulk/config`: where a source may live and the run bounds. */
export interface BulkIngestConfig {
  source_roots: string[];
  default_extensions: string[];
  default_batch_size: number;
  default_concurrency: number;
  max_batch_size: number;
  max_concurrency: number;
}

/** `POST /ingest/bulk`: exactly one of `folder`, `manifest`, `s3` and `source_id`. */
export interface BulkIngestRequest {
  folder?: string | null;
  manifest?: string | null;
  s3?: S3Location | null;
  source_id?: string | null;
  source?: string;
  extensions?: string[] | null;
  batch_size?: number;
  concurrency?: number;
}

export interface BulkSource {
  kind: BulkSourceKind;
  /** A folder or manifest path; for a `queue` source the id of the registered source. */
  path: string;
  extensions: string[];
  /** What to show for a registered source (its name and location). */
  label?: string | null;
}

export interface BulkOptions {
  source_tag: string;
  batch_size: number;
  concurrency: number;
}

export interface BulkIngestProgress {
  discovered: number;
  discovery_complete: boolean;
  skipped: number;
  ingested: number;
  duplicates: number;
  failed: number;
  processed: number;
  crops_indexed: number;
  n_embedded: number;
  n_not_embedded: number;
  n_embed_failed: number;
  in_flight: number;
  images_per_s?: number | null;
  images_per_s_recent?: number | null;
  eta_s?: number | null;
  elapsed_s?: number | null;
}

export interface BulkIngestRun {
  project: string;
  run_id: string;
  status: BulkIngestStatus;
  resumed?: boolean;
  source: BulkSource;
  options: BulkOptions;
  progress: BulkIngestProgress;
  stage_s?: Record<string, number>;
  attempts?: number;
  /** What the run allows now (served): cancel while live, resume once stopped with work left. */
  can_cancel: boolean;
  can_resume: boolean;
  /** A cancel was asked for and the run has not stopped yet. */
  cancel_requested?: boolean;
  error?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  /** Seconds to wait before the next read; null once the run is over. */
  poll_after_s?: number | null;
}

export interface BulkIngestList {
  items: BulkIngestRun[];
  total: number;
  page: number;
  page_size: number;
}

export interface BulkIngestFailure {
  path: string;
  error: string;
  error_kind: string;
}

export interface BulkIngestFailurePage {
  items: BulkIngestFailure[];
  total: number;
  offset: number;
  limit: number;
}
