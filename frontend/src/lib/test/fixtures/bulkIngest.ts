import type {
  BulkIngestConfig,
  BulkPreview,
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
    preview: {
      max_files: 200000,
      max_seconds: 60,
      header_sample: 300,
      gpu_decode: true,
      gpu_min_megapixels: 3,
      rates: [
        {
          route: 'cpu_small',
          images_per_s: 29,
          source_section: 'Decode once and the GPU decode (#217)',
          measured_on: 'COCO frames (0.3 MP), CPU decode',
        },
        {
          route: 'cpu_large',
          images_per_s: 27,
          source_section: 'Decode once and the GPU decode (#217)',
          measured_on: 'Commons photos (12 to 24 MP), CPU decode',
        },
        {
          route: 'gpu_large',
          images_per_s: 35,
          source_section: 'Decode once and the GPU decode (#217)',
          measured_on: 'Commons photos (12 to 24 MP), GPU decode',
        },
      ],
      embeddings_per_image: {
        value: 7.4,
        unit: 'embeddings per image',
        source_section: 'v0.5.0 baseline (set A, public COCO, 2,000 images)',
        measured_on: 'COCO, policy all',
      },
      opensearch_bytes_per_image: {
        value: 65298,
        unit: 'bytes per image',
        source_section: 'Storage per image and per vector',
        measured_on: 'COCO, policy all',
      },
      source_document: 'docs/PERFORMANCE.md',
    },
    ...over,
  };
}

export function previewFixture(over: Partial<BulkPreview> = {}): BulkPreview {
  return {
    project: 'default',
    kind: 'folder',
    target: '/data/source/coco',
    extensions: ['.jpg', '.jpeg', '.png'],
    files: 2000,
    total_bytes: 836_000_000,
    by_extension: {
      '.jpg': { files: 1990, bytes: 830_000_000 },
      '.png': { files: 10, bytes: 6_000_000 },
    },
    megapixels: {
      sampled: 300,
      unreadable: 0,
      histogram: { '<1': 300 },
      gate_megapixels: 3,
      share_at_or_above_gate: 0,
    },
    estimate: {
      basis: 'megapixel_sample',
      duration_s_min: 69,
      duration_s_max: 69,
      vectors: 14800,
      opensearch_bytes: 130_596_000,
    },
    partial: false,
    stop_reason: null,
    unreadable_dirs: 0,
    unreadable_samples: [],
    symlinks_skipped: 0,
    missing: 0,
    outside_roots: 0,
    unreadable_files: 0,
    gpu_decode: true,
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
