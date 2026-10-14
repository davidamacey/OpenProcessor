/**
 * The ingest-source wire types (`types_ingestSources.ts`) carry exactly the
 * properties the vendored OpenAPI serves, so a field added or renamed on the
 * backend fails here instead of rendering as undefined.
 */
import { describe, expect, it } from 'vitest';
import openapi from '$contracts/openapi/curation.json';
import type {
  IngestSourceList,
  IngestSourcePatch,
  IngestSourceRequest,
  IngestSourceStatus,
  LastScan,
  RetryDeadResult,
  S3Location,
  ScheduleTrigger,
  SourceCounts,
  SourceObjectError,
  SourceRunSummary,
  SourceScanStatus,
  SourceScheduleStatus,
  SourceWatchStatus,
  SourceWebhookStatus,
  Triggers,
  WatchTrigger,
} from '$lib/types_ingestSources';

type Schema = { properties?: Record<string, unknown>; enum?: string[] };
const schemas = (openapi as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

function keysOf(name: string): string[] {
  const schema = schemas[name];
  if (!schema?.properties) throw new Error(`schema not found: ${name}`);
  return Object.keys(schema.properties).sort();
}

function pin<T>(keys: (keyof T)[], schema: string): void {
  expect([...(keys as string[])].sort()).toEqual(keysOf(schema));
}

describe('ingest sources contract', () => {
  it('request and patch types', () => {
    pin<IngestSourceRequest>(
      [
        'name',
        'backend',
        's3',
        'folder',
        'extensions',
        'source',
        'batch_size',
        'concurrency',
        'triggers',
      ],
      'IngestSourceRequest',
    );
    pin<IngestSourcePatch>(
      ['name', 'extensions', 'source', 'batch_size', 'concurrency', 'triggers'],
      'IngestSourcePatch',
    );
    pin<S3Location>(['bucket', 'prefix'], 'S3Location');
    pin<Triggers>(['watch', 'schedule', 'webhook_enabled'], 'TriggersWire');
    pin<WatchTrigger>(['enabled', 'interval_s'], 'WatchTriggerWire');
    pin<ScheduleTrigger>(['enabled', 'cron'], 'ScheduleTriggerWire');
  });

  it('status and list types', () => {
    pin<IngestSourceStatus>(
      [
        'project',
        'source_id',
        'name',
        'backend',
        'location',
        's3',
        'folder',
        'extensions',
        'source',
        'batch_size',
        'concurrency',
        'triggers',
        'created_at',
        'counts',
        'scan',
        'watch',
        'schedule',
        'webhook',
        'last_run',
        'errors',
        'can_run',
        'can_retry_dead',
        'blocked_reason',
        'blocked_message',
      ],
      'IngestSourceStatus',
    );
    pin<IngestSourceList>(
      [
        'items',
        's3_configured',
        's3_allowed',
        'default_batch_size',
        'default_concurrency',
        'default_watch_interval_s',
        'min_watch_interval_s',
        'poll_after_s',
      ],
      'IngestSourceList',
    );
    pin<RetryDeadResult>(['retried', 'source'], 'RetryDeadResult');
  });

  it('the parts of a status', () => {
    pin<SourceCounts>(['pending', 'done', 'failed', 'dead'], 'SourceCounts');
    pin<SourceScanStatus>(
      ['active', 'pages', 'listed', 'queued', 'last'],
      'SourceScanStatus',
    );
    pin<LastScan>(['finished_at', 'pages', 'listed', 'queued'], 'LastScan');
    pin<SourceWatchStatus>(
      ['last_poll_at', 'next_poll_at', 'error', 'backoff_until'],
      'SourceWatchStatus',
    );
    pin<SourceScheduleStatus>(['last_fire_at', 'next_fire_at'], 'SourceScheduleStatus');
    pin<SourceWebhookStatus>(
      ['path', 'received', 'queued', 'duplicates', 'ignored', 'last_event_at'],
      'SourceWebhookStatus',
    );
    pin<SourceRunSummary>(
      [
        'run_id',
        'status',
        'started_at',
        'finished_at',
        'ingested',
        'duplicates',
        'failed',
        'error',
      ],
      'SourceRunSummary',
    );
    pin<SourceObjectError>(
      ['key', 'etag', 'state', 'attempts', 'error'],
      'SourceObjectError',
    );
  });
});
