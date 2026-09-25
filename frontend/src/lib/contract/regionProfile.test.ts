/**
 * The served region profile (OpenProcessor naming-w2): `ServedRegionProfile`
 * must match the vendored `RegionProfileSummary` schema, and both routes
 * the frontend reads it from must serve it.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type { ServedRegionProfile } from '$lib/types';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

// Compile-time exhaustive: adding a key to ServedRegionProfile without
// listing it here is a type error.
const FRONTEND_KEYS = {
  name: true,
  display_name: true,
  display_name_singular: true,
  region_class_name: true,
  text_reader: true,
} satisfies Record<keyof ServedRegionProfile, true>;

describe('RegionProfileSummary', () => {
  it('ServedRegionProfile has exactly the served keys, all required', () => {
    const summary = schemas.RegionProfileSummary;
    expect(Object.keys(summary.properties ?? {}).sort()).toEqual(
      Object.keys(FRONTEND_KEYS).sort(),
    );
    expect([...(summary.required ?? [])].sort()).toEqual(
      Object.keys(FRONTEND_KEYS).sort(),
    );
  });

  it.each(['HealthResponse', 'RegionVocabularyResponse'])(
    '%s serves region_profile (nullable RegionProfileSummary)',
    (name) => {
      const prop = schemas[name].properties?.region_profile as { anyOf?: unknown[] };
      expect(prop?.anyOf).toEqual([
        { $ref: '#/components/schemas/RegionProfileSummary' },
        { type: 'null' },
      ]);
    },
  );
});
