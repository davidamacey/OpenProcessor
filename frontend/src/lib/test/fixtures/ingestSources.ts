import type { IngestSourceList, IngestSourceStatus } from '$lib/types_ingestSources';

export const SOURCE_ID = 'src_1a2b3c4d5e';

/** A served S3 source with a watcher and a webhook. */
export function sourceFixture(
  over: Partial<IngestSourceStatus> = {},
): IngestSourceStatus {
  return {
    project: 'default',
    source_id: SOURCE_ID,
    name: 'Incoming photos',
    backend: 's3',
    location: 's3://photos/incoming/',
    s3: { bucket: 'photos', prefix: 'incoming/' },
    folder: null,
    extensions: ['.jpg', '.jpeg', '.png'],
    source: 'source',
    batch_size: 16,
    concurrency: 16,
    triggers: {
      watch: { enabled: true, interval_s: 60 },
      schedule: { enabled: true, cron: '0 2 * * *' },
      webhook_enabled: true,
    },
    created_at: '2026-10-12T10:00:00+00:00',
    counts: { pending: 3, done: 120, failed: 1, dead: 2 },
    scan: { active: false, pages: 0, listed: 0, queued: 0, last: null },
    watch: {
      last_poll_at: '2026-10-12T11:00:00+00:00',
      next_poll_at: '2026-10-12T11:01:00+00:00',
      error: null,
      backoff_until: null,
    },
    schedule: {
      last_fire_at: '2026-10-12T02:00:00+00:00',
      next_fire_at: '2026-10-13T02:00:00+00:00',
    },
    webhook: {
      path: `/curation/projects/default/ingest/sources/${SOURCE_ID}/events`,
      received: 40,
      queued: 38,
      duplicates: 1,
      ignored: 1,
      last_event_at: '2026-10-12T10:59:00+00:00',
    },
    last_run: {
      run_id: 'bulk_20261012T110000_1a2b3c4d',
      status: 'completed_with_errors',
      started_at: '2026-10-12T11:00:00+00:00',
      finished_at: '2026-10-12T11:02:00+00:00',
      ingested: 118,
      duplicates: 2,
      failed: 3,
      error: null,
    },
    errors: [
      {
        key: 'incoming/gone.jpg',
        etag: 'e1',
        state: 'dead',
        attempts: 1,
        error: 'NoSuchKey',
      },
    ],
    can_run: true,
    can_retry_dead: true,
    blocked_reason: null,
    blocked_message: null,
    ...over,
  };
}

export function listFixture(
  items: IngestSourceStatus[] = [sourceFixture()],
  over: Partial<IngestSourceList> = {},
): IngestSourceList {
  return {
    items,
    s3_configured: true,
    s3_allowed: ['photos/incoming/'],
    default_batch_size: 16,
    default_concurrency: 16,
    default_watch_interval_s: 60,
    min_watch_interval_s: 10,
    poll_after_s: 10,
    ...over,
  };
}
