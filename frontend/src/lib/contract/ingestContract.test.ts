/**
 * Ingest-specific contract checks against the vendored OpenAPI snapshot
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.3), beyond
 * what the generic `endpointCatalog.test.ts` path/method/query-param
 * sweep already covers: the `_PathLookupRequest.image_paths.maxItems`
 * value, the ingest response property sets, the `IngestItemStatus`
 * enum, and the multipart body's field names.
 */
import { describe, expect, it } from 'vitest';
import openapi from '../../../contracts/openprocessor/openapi/curation.json';
import { PATH_LOOKUP_MAX } from '$lib/ingest/ingestConfig';
import type {
  BatchIngestResponse,
  BatchIngestSummary,
  IngestImageResult,
  IngestItemStatus,
  RegionDependencyStatus,
  RegionDrain,
} from '$lib/types';

type Schemas = Record<string, { properties?: Record<string, unknown> }>;
const schemas = (openapi as { components: { schemas: Schemas } }).components.schemas;

function keysOf(schemaName: string): string[] {
  const schema = schemas[schemaName];
  if (!schema?.properties) throw new Error(`schema not found: ${schemaName}`);
  return Object.keys(schema.properties).sort();
}

describe('ingest contract', () => {
  it('PATH_LOOKUP_MAX equals the vendored _PathLookupRequest.image_paths.maxItems', () => {
    const schema = schemas['_PathLookupRequest'] as {
      properties: { image_paths: { maxItems: number } };
    };
    expect(schema.properties.image_paths.maxItems).toBe(PATH_LOOKUP_MAX);
    expect(PATH_LOOKUP_MAX).toBe(10_000);
  });

  it('IngestImageResult (types.ts) has exactly the served IngestImageResponse properties', () => {
    const IMAGE_RESULT_KEYS = [
      'error',
      'error_kind',
      'image_id',
      'image_path',
      'imohash',
      'n_crops',
      'n_embedded',
      'n_filtered',
      'n_not_embedded',
      'n_regions',
      'secondary_detector_error',
      'source_identifier',
      'status',
    ] satisfies (keyof IngestImageResult)[];
    expect([...IMAGE_RESULT_KEYS].sort()).toEqual(keysOf('IngestImageResponse'));
  });

  it('BatchIngestSummary (types.ts) has exactly the served BatchIngestSummaryResponse properties', () => {
    const SUMMARY_KEYS = [
      'successful',
      'duplicates',
      'failed',
      'crops_indexed',
      'n_embedded',
      'n_filtered',
      'n_not_embedded',
      'secondary_detector_failures',
    ] satisfies (keyof BatchIngestSummary)[];
    expect([...SUMMARY_KEYS].sort()).toEqual(keysOf('BatchIngestSummaryResponse'));
  });

  it('BatchIngestResponse (types.ts) has exactly the served BatchIngestResponse properties', () => {
    const RESPONSE_KEYS = [
      'status',
      'summary',
      'results',
    ] satisfies (keyof BatchIngestResponse)[];
    // The generated schema name is mangled by the module path FastAPI
    // resolved the model from (two `BatchIngestResponse`-titled models
    // exist server-side); `src__routers__curation___common_models__` is
    // the real one this endpoint's response references (BA-1/BA-7 split
    // the wire models out of `_common.py` into `_common_models.py` —
    // this schema name moved with them, was `___common__` before).
    expect([...RESPONSE_KEYS].sort()).toEqual(
      keysOf('src__routers__curation___common_models__BatchIngestResponse'),
    );
  });

  it('RegionDependencyStatus (types.ts) has exactly the served properties', () => {
    const KEYS = [
      'role',
      'model',
      'ready',
      'detail',
      'unavailable_since',
    ] satisfies (keyof RegionDependencyStatus)[];
    expect([...KEYS].sort()).toEqual(keysOf('RegionDependencyStatusResponse'));
  });

  it('IngestImageResponse.status enum equals IngestItemStatus', () => {
    const schema = schemas['IngestImageResponse'] as {
      properties: { status: { enum: string[] } };
    };
    const served = [...schema.properties.status.enum].sort();
    const client: IngestItemStatus[] = ['success', 'duplicate', 'failed'];
    expect(served).toEqual([...client].sort());
  });

  it('the upload multipart body field names match the served request schema', () => {
    // P1 projects cutover: the operation moved to the scoped path
    // (`/curation/projects/{project}/ingest/upload`), which changes
    // FastAPI's auto-generated schema name to match.
    const schema =
      schemas[
        'Body_curation_ingest_upload_curation_projects__project__ingest_upload_post'
      ];
    // BA-4 added an optional `run_id` form field (client identifier for
    // an upload run, echoed on `GET /ingest/status?run_id=`) — not sent
    // by `ingestUpload()` today (no run_id UI yet), but it's a real,
    // served field on the request schema this test mirrors.
    expect(Object.keys(schema.properties ?? {}).sort()).toEqual(
      ['image_paths', 'images', 'run_id', 'source'].sort(),
    );
  });

  it('IngestConfig (types.ts) has exactly the served IngestConfigResponse shape', () => {
    const uploadKeys = keysOf('IngestUploadConfig');
    const batchKeys = keysOf('IngestBatchConfig');
    const drainKeys = keysOf('IngestRegionDrainConfig');
    expect(uploadKeys).toEqual(
      [
        'enabled',
        'max_images_per_request',
        'max_bytes_per_request',
        'accepted_extensions',
        'persists_bytes',
      ].sort(),
    );
    expect(batchKeys).toEqual(['enabled', 'max_items', 'source_roots'].sort());
    expect(drainKeys).toEqual(['poll_interval_s', 'stable_polls'].sort());
  });

  it('RegionDrain (types.ts) has exactly the served IngestRegionDrainResponse properties', () => {
    const DRAIN_KEYS = [
      'pending_detection',
      'pending_verification',
      'total_unfinished',
      'drained',
      'stable_for_s',
      'observed_at',
      'region_dependencies',
      'stall_reason',
    ] satisfies (keyof RegionDrain)[];
    expect([...DRAIN_KEYS].sort()).toEqual(keysOf('IngestRegionDrainResponse'));
  });
});
