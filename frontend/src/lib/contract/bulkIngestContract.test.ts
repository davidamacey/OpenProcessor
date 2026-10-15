/**
 * The bulk-ingest wire types (`types_bulkIngest.ts`) carry exactly the
 * properties the vendored OpenAPI serves, so a field added or renamed on the
 * backend fails here instead of rendering as undefined.
 */
import { describe, expect, it } from 'vitest';
import openapi from '$contracts/openapi/curation.json';
import type {
  BulkIngestConfig,
  BulkIngestFailure,
  BulkIngestFailurePage,
  BulkIngestList,
  BulkIngestProgress,
  BulkIngestRequest,
  BulkIngestRun,
  BulkIngestStatus,
  BulkOptions,
  BulkPreview,
  BulkPreviewConfig,
  BulkPreviewRequest,
  PreviewEstimate,
  PreviewExtension,
  PreviewMeasured,
  PreviewMegapixels,
  PreviewRate,
  BulkSource,
} from '$lib/types_bulkIngest';

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

describe('bulk ingest contract', () => {
  it('request, config and source types', () => {
    pin<BulkIngestRequest>(
      [
        'folder',
        'manifest',
        's3',
        'source_id',
        'source',
        'extensions',
        'batch_size',
        'concurrency',
      ],
      'BulkIngestRequest',
    );
    pin<BulkIngestConfig>(
      [
        'source_roots',
        'default_extensions',
        'default_batch_size',
        'default_concurrency',
        'max_batch_size',
        'max_concurrency',
        'preview',
      ],
      'BulkIngestConfig',
    );
    pin<BulkSource>(['kind', 'path', 'extensions', 'label'], 'BulkSourceWire');
    pin<BulkOptions>(['source_tag', 'batch_size', 'concurrency'], 'BulkOptionsWire');
  });

  it('preview request, config and answer types', () => {
    pin<BulkPreviewRequest>(
      ['folder', 'manifest', 's3', 'source_id', 'extensions'],
      'BulkPreviewRequest',
    );
    pin<BulkPreviewConfig>(
      [
        'max_files',
        'max_seconds',
        'header_sample',
        'gpu_decode',
        'gpu_min_megapixels',
        'rates',
        'embeddings_per_image',
        'opensearch_bytes_per_image',
        'source_document',
      ],
      'BulkPreviewConfig',
    );
    pin<PreviewRate>(
      ['route', 'images_per_s', 'source_section', 'measured_on'],
      'PreviewRateWire',
    );
    pin<PreviewMeasured>(
      ['value', 'unit', 'source_section', 'measured_on'],
      'PreviewMeasuredWire',
    );
    pin<BulkPreview>(
      [
        'project',
        'kind',
        'target',
        'extensions',
        'files',
        'total_bytes',
        'by_extension',
        'megapixels',
        'estimate',
        'partial',
        'stop_reason',
        'unreadable_dirs',
        'unreadable_samples',
        'symlinks_skipped',
        'missing',
        'outside_roots',
        'unreadable_files',
        'gpu_decode',
      ],
      'BulkPreview',
    );
    pin<PreviewExtension>(['files', 'bytes'], 'PreviewExtensionWire');
    pin<PreviewMegapixels>(
      ['sampled', 'unreadable', 'histogram', 'gate_megapixels', 'share_at_or_above_gate'],
      'PreviewMegapixelsWire',
    );
    pin<PreviewEstimate>(
      ['basis', 'duration_s_min', 'duration_s_max', 'vectors', 'opensearch_bytes'],
      'PreviewEstimateWire',
    );
  });

  it('run and progress types', () => {
    pin<BulkIngestRun>(
      [
        'project',
        'run_id',
        'status',
        'resumed',
        'source',
        'options',
        'progress',
        'stage_s',
        'attempts',
        'can_cancel',
        'can_resume',
        'cancel_requested',
        'error',
        'started_at',
        'finished_at',
        'poll_after_s',
      ],
      'BulkIngestRun',
    );
    pin<BulkIngestProgress>(
      [
        'discovered',
        'discovery_complete',
        'skipped',
        'ingested',
        'duplicates',
        'failed',
        'processed',
        'crops_indexed',
        'n_embedded',
        'n_not_embedded',
        'n_embed_failed',
        'in_flight',
        'images_per_s',
        'images_per_s_recent',
        'eta_s',
        'elapsed_s',
      ],
      'BulkIngestProgress',
    );
  });

  it('list and failure page types', () => {
    pin<BulkIngestList>(['items', 'total', 'page', 'page_size'], 'BulkIngestList');
    pin<BulkIngestFailurePage>(
      ['items', 'total', 'offset', 'limit'],
      'BulkIngestFailurePage',
    );
    pin<BulkIngestFailure>(['path', 'error', 'error_kind'], 'BulkIngestFailure');
  });

  it('every served status is a BulkIngestStatus', () => {
    const served = ((
      schemas['BulkIngestRun'].properties as Record<string, { enum?: string[] }>
    ).status.enum ?? []) as string[];
    const known = [
      'queued',
      'running',
      'completed',
      'completed_with_errors',
      'failed',
      'cancelled',
      'interrupted',
    ] satisfies BulkIngestStatus[];
    expect([...known].sort()).toEqual([...served].sort());
  });
});
