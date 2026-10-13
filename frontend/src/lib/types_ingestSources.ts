/**
 * Wire types for registered ingest sources (`{prefix}/ingest/sources*`): a
 * folder or S3 bucket/prefix with options and triggers, and the status the
 * backend serves for it. Pinned key-for-key to the vendored OpenAPI by
 * `contract/ingestSourcesContract.test.ts`.
 */

export type SourceBackend = 's3' | 'folder';

/** Why a source cannot be run now; served, never derived here. */
export type BlockedReason =
  's3_not_configured' | 's3_not_allowed' | 'invalid_source' | 'project_run_live';

export interface S3Location {
  bucket: string;
  prefix?: string;
}

export interface WatchTrigger {
  enabled?: boolean;
  interval_s?: number;
}

export interface ScheduleTrigger {
  enabled?: boolean;
  /** Five fields, UTC: minute hour day-of-month month day-of-week. */
  cron?: string;
}

export interface Triggers {
  watch?: WatchTrigger;
  schedule?: ScheduleTrigger;
  webhook_enabled?: boolean;
}

/** `POST /ingest/sources`: `s3` for an S3 source, `folder` for a folder source. */
export interface IngestSourceRequest {
  name: string;
  backend: SourceBackend;
  s3?: S3Location | null;
  folder?: string | null;
  extensions?: string[] | null;
  source?: string;
  batch_size?: number;
  concurrency?: number;
  triggers?: Triggers;
}

/** `PATCH /ingest/sources/{id}`: options and triggers; the location never changes. */
export interface IngestSourcePatch {
  name?: string | null;
  extensions?: string[] | null;
  source?: string | null;
  batch_size?: number | null;
  concurrency?: number | null;
  triggers?: Triggers | null;
}

export interface SourceCounts {
  pending: number;
  done: number;
  failed: number;
  dead: number;
}

export interface LastScan {
  finished_at: number;
  pages: number;
  listed: number;
  queued: number;
}

export interface SourceScanStatus {
  /** A pass is under way (or was interrupted and will resume). */
  active: boolean;
  pages: number;
  listed: number;
  queued: number;
  last?: LastScan | null;
}

export interface SourceWatchStatus {
  last_poll_at?: string | null;
  next_poll_at?: string | null;
  error?: string | null;
  backoff_until?: string | null;
}

export interface SourceScheduleStatus {
  last_fire_at?: string | null;
  next_fire_at?: string | null;
}

export interface SourceWebhookStatus {
  /** Path of the events route, relative to the API origin. */
  path: string;
  received: number;
  queued: number;
  duplicates: number;
  ignored: number;
  last_event_at?: string | null;
}

export interface SourceRunSummary {
  run_id: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  ingested: number;
  duplicates: number;
  failed: number;
  error?: string | null;
}

export interface SourceObjectError {
  key: string;
  etag: string;
  state: 'failed' | 'dead';
  attempts: number;
  error: string;
}

export interface IngestSourceStatus {
  project: string;
  source_id: string;
  name: string;
  backend: SourceBackend;
  /** `s3://bucket/prefix` or the folder path. */
  location: string;
  s3?: S3Location | null;
  folder?: string | null;
  extensions: string[];
  source: string;
  batch_size: number;
  concurrency: number;
  triggers: Triggers;
  created_at: string;
  counts: SourceCounts;
  scan: SourceScanStatus;
  watch: SourceWatchStatus;
  schedule: SourceScheduleStatus;
  webhook: SourceWebhookStatus;
  last_run?: SourceRunSummary | null;
  errors?: SourceObjectError[];
  can_run: boolean;
  can_retry_dead: boolean;
  blocked_reason?: BlockedReason | null;
  /** The reason in words, for display. */
  blocked_message?: string | null;
}

export interface IngestSourceList {
  items: IngestSourceStatus[];
  /** Whether S3 sources can be created at all, and the entries they may use. */
  s3_configured: boolean;
  s3_allowed: string[];
  default_batch_size: number;
  default_concurrency: number;
  default_watch_interval_s: number;
  min_watch_interval_s: number;
  /** Seconds before the next read: short while a scan or run is live. */
  poll_after_s: number;
}

export interface RetryDeadResult {
  retried: number;
  source: IngestSourceStatus;
}
