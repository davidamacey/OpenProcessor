import type {
  BulkIngestConfig,
  BulkIngestProgress,
  BulkIngestRun,
} from '$lib/types_bulkIngest';

export const RUN_ID = 'bulk_20261011T120000_1a2b3c4d';

export function configFixture(over: Partial<BulkIngestConfig> = {}): BulkIngestConfig {
  return {
    source_roots: ['/data/source'],
    default_extensions: ['.jpg', '.jpeg', '.png'],
    default_batch_size: 16,
    default_concurrency: 8,
    max_batch_size: 64,
    max_concurrency: 32,
    ...over,
  };
}

export function progressFixture(
  over: Partial<BulkIngestProgress> = {},
): BulkIngestProgress {
  return {
    discovered: 0,
    discovery_complete: false,
    skipped: 0,
    ingested: 0,
    duplicates: 0,
    failed: 0,
    processed: 0,
    crops_indexed: 0,
    n_embedded: 0,
    n_not_embedded: 0,
    n_embed_failed: 0,
    in_flight: 0,
    images_per_s: null,
    images_per_s_recent: null,
    eta_s: null,
    elapsed_s: 0,
    ...over,
  };
}

/** A served run; defaults to a live one. */
export function runFixture(over: Partial<BulkIngestRun> = {}): BulkIngestRun {
  const live = (over.status ?? 'running') === 'running' || over.status === 'queued';
  return {
    project: 'default',
    run_id: RUN_ID,
    status: 'running',
    resumed: false,
    source: { kind: 'folder', path: '/data/source/coco', extensions: ['.jpg'] },
    options: { source_tag: 'bulk_ingest', batch_size: 16, concurrency: 8 },
    progress: progressFixture(),
    stage_s: {},
    attempts: 1,
    can_cancel: live,
    can_resume: false,
    cancel_requested: false,
    error: null,
    started_at: '2026-10-11T12:00:00+00:00',
    finished_at: null,
    poll_after_s: live ? 2 : null,
    ...over,
  };
}
