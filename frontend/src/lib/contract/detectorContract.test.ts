/**
 * `types_detector.ts` vs the vendored OpenAPI schemas (v0.4.0). Each key
 * map is compile-time exact against its type and pinned to the schema's
 * property set, so a served rename fails here instead of rendering "-".
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type * as T from '$lib/types_detector';
import {
  CLASS_RESOLUTIONS,
  EMBEDDING_MODES,
  SEED_CONFLICT_REASONS,
  SEED_SKIP_REASONS,
} from '$lib/types_detector';

type Schema = {
  properties?: Record<string, { enum?: string[] }>;
};
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'IngestDetectorInfo',
    keys<keyof T.IngestDetectorInfo>({
      model: true,
      version: true,
      input_size: true,
      assigns_class: true,
      confidence_floor_applies: true,
      n_labels: true,
      labels: true,
    }),
  ],
  [
    'IngestDetectorLabel',
    keys<keyof T.IngestDetectorLabel>({ class_id: true, name: true, slug: true }),
  ],
  [
    'DetectFilter',
    keys<keyof T.DetectFilter>({
      class_resolution: true,
      classes: true,
      exclude_classes: true,
      max_per_image: true,
      min_box_area_frac: true,
      min_confidence: true,
    }),
  ],
  [
    'EmbeddingPolicy',
    keys<keyof T.EmbeddingPolicy>({
      classes: true,
      max_per_image: true,
      min_box_area_frac: true,
      min_confidence: true,
      mode: true,
    }),
  ],
  [
    'DetectorOverride',
    keys<keyof T.DetectorOverride>({
      input_size: true,
      labels_path: true,
      model: true,
      version: true,
    }),
  ],
  [
    'IngestPolicy',
    keys<keyof T.IngestPolicy>({
      detect: true,
      detector: true,
      embedding: true,
      revision: true,
    }),
  ],
  [
    'IngestPolicyBody',
    keys<keyof T.IngestPolicyBody>({ detect: true, detector: true, embedding: true }),
  ],
  [
    'IngestPolicyUpdate',
    keys<keyof T.IngestPolicyUpdate>({
      detect: true,
      detector: true,
      embedding: true,
      expected_revision: true,
    }),
  ],
  [
    'IngestPolicyPutResponse',
    keys<keyof T.IngestPolicyPutResponse>({
      detect: true,
      detector: true,
      embedding: true,
      revision: true,
      unknown_names: true,
    }),
  ],
  [
    'IngestPolicyPreview',
    keys<keyof T.IngestPolicyPreview>({
      total_items: true,
      scanned: true,
      truncated: true,
      would_embed: true,
      would_not_embed: true,
      estimated_vector_mb: true,
      by_class: true,
    }),
  ],
  [
    'PolicyPreviewClass',
    keys<keyof T.PolicyPreviewClass>({
      name: true,
      would_embed: true,
      would_not_embed: true,
    }),
  ],
  [
    'SeedFromDetectorRequest',
    keys<keyof T.SeedFromDetectorRequest>({ dry_run: true, group: true, names: true }),
  ],
  [
    'SeedFromDetectorResponse',
    keys<keyof T.SeedFromDetectorResponse>({
      conflicts: true,
      created: true,
      detector_model: true,
      dry_run: true,
      skipped: true,
    }),
  ],
  [
    'SeededClass',
    keys<keyof T.SeededClass>({ class_id: true, detector_label: true, name: true }),
  ],
  [
    'SeedSkipped',
    keys<keyof T.SeedSkipped>({ detector_label: true, name: true, reason: true }),
  ],
  [
    'SeedConflict',
    keys<keyof T.SeedConflict>({
      class_id_in_detector: true,
      detector_label: true,
      reason: true,
    }),
  ],
  [
    'DetectionsSummary',
    keys<keyof T.DetectionsSummary>({
      by_label: true,
      embedding: true,
      labels_truncated: true,
      suggested_reprocess: true,
      total: true,
    }),
  ],
  [
    'EmbeddingBreakdown',
    keys<keyof T.EmbeddingBreakdown>({
      by_state: true,
      embedded: true,
      not_embedded: true,
    }),
  ],
  [
    'EmbeddingByState',
    keys<keyof T.EmbeddingByState>({
      deferred: true,
      embedded: true,
      failed: true,
      not_selected: true,
      unknown: true,
    }),
  ],
  [
    'LabelSummary',
    keys<keyof T.LabelSummary>({ count: true, embedding: true, name: true }),
  ],
];

describe('detector wire types vs the vendored contract', () => {
  for (const [schema, expected] of CASES) {
    it(`${schema} has exactly the served properties`, () => {
      const served = schemas[schema];
      expect(served, `${schema} missing from the contract`).toBeDefined();
      expect(Object.keys(served!.properties ?? {}).sort()).toEqual(expected);
    });
  }

  it('the enums equal the served enums', () => {
    expect([...CLASS_RESOLUTIONS]).toEqual(
      schemas['DetectFilter']!.properties!['class_resolution']!.enum,
    );
    expect([...EMBEDDING_MODES]).toEqual(
      schemas['EmbeddingPolicy']!.properties!['mode']!.enum,
    );
    expect([...SEED_SKIP_REASONS]).toEqual(
      schemas['SeedSkipped']!.properties!['reason']!.enum,
    );
    expect([...SEED_CONFLICT_REASONS]).toEqual(
      schemas['SeedConflict']!.properties!['reason']!.enum,
    );
  });
});
