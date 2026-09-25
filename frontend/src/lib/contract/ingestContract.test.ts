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
      'image_id',
      'image_path',
      'imohash',
      'n_crops',
      'n_regions',
      'status',
    ] satisfies (keyof IngestImageResult)[];
    expect([...IMAGE_RESULT_KEYS].sort()).toEqual(keysOf('IngestImageResponse'));
  });

  it('BatchIngestSummary (types.ts) has exactly the served BatchIngestSummaryResponse properties', () => {
    const SUMMARY_KEYS = [
      'successful',
      'duplicates',
      'failed',
      'mismatches',
      'missed_labels',
      'unmatched_detections',
      'labels_imported',
      'crops_indexed',
    ] satisfies (keyof BatchIngestSummary)[];
    expect([...SUMMARY_KEYS].sort()).toEqual(keysOf('BatchIngestSummaryResponse'));
  });

  it('BatchIngestResponse (types.ts) has exactly the served BatchIngestResponse properties', () => {
    const RESPONSE_KEYS = [
      'status',
      'summary',
      'results',
      'disagreements',
    ] satisfies (keyof BatchIngestResponse)[];
    // The generated schema name is mangled by the module path FastAPI
    // resolved the model from (two `BatchIngestResponse`-titled models
    // exist server-side); `src__routers__curation___common__` is the
    // real one this endpoint's response references.
    expect([...RESPONSE_KEYS].sort()).toEqual(
      keysOf('src__routers__curation___common__BatchIngestResponse'),
    );
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
    const schema = schemas['Body_curation_ingest_upload_curation_ingest_upload_post'];
    expect(Object.keys(schema.properties ?? {}).sort()).toEqual(
      ['image_paths', 'images', 'source'].sort(),
    );
  });
});
