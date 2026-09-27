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
  reads_text: true,
  text_hint_enabled: true,
  limits: true,
} satisfies Record<keyof ServedRegionProfile, true>;

// W8.9 (feat/w8-multibox-lockstep, docs/design/
// w8-multibox-frontend-plan-2026-09-26.md): `limits` is new in the
// backend's W8 wave and isn't in the vendored pre-W8 OpenAPI snapshot yet.
// Remove this allow-list entry (not widen it) the moment `npm run
// contract:sync` picks up the backend's W8 addition to
// `RegionProfileSummary` — same pattern as endpointCatalog.test.ts's
// PENDING_BACKEND.
const PENDING_BACKEND_W8_KEYS = new Set(['limits']);

describe('RegionProfileSummary', () => {
  it('ServedRegionProfile has exactly the served keys, all required', () => {
    const summary = schemas.RegionProfileSummary;
    const frontendKeys = Object.keys(FRONTEND_KEYS).filter(
      (k) => !PENDING_BACKEND_W8_KEYS.has(k),
    );
    expect(Object.keys(summary.properties ?? {}).sort()).toEqual(frontendKeys.sort());
    expect([...(summary.required ?? [])].sort()).toEqual(frontendKeys.sort());
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
