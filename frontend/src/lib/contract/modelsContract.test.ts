/**
 * `types_models.ts` vs the vendored OpenAPI schemas (OpenProcessor
 * projects P2, §5.5 model sharing). Same technique as
 * `projectsContract.test.ts`: each key map is compile-time exact against
 * its interface and pinned to the schema's property set. Not pinned:
 * `ModelClassMappingSummary` (the per-entry `class_mapping` on the
 * untyped `GET /models/status` response) — see `types_models.ts`.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type * as M from '$lib/types_models';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'ModelSharingRequest',
    keys({ shared: true, expected_revision: true } satisfies Record<
      keyof M.ModelSharingRequest,
      true
    >),
  ],
  [
    'ModelSharingResponse',
    keys({
      name: true,
      project: true,
      shared: true,
      revision: true,
      used_by: true,
    } satisfies Record<keyof M.ModelSharingResponse, true>),
  ],
  [
    'ModelSharingUser',
    keys({ project: true, profile: true } satisfies Record<
      keyof M.ModelSharingUser,
      true
    >),
  ],
  [
    'ModelClassMappingResponse',
    keys({
      model: true,
      model_project: true,
      project: true,
      entries: true,
      unmapped: true,
      not_covered: true,
      labels: true,
    } satisfies Record<keyof M.ModelClassMappingResponse, true>),
  ],
  [
    'ModelClassMappingEntry',
    keys({
      model_id: true,
      model_name: true,
      class_id: true,
      class_name: true,
      match: true,
    } satisfies Record<keyof M.ModelClassMappingEntry, true>),
  ],
  [
    'ModelClassMappingLabels',
    keys({ match: true } satisfies Record<keyof M.ModelClassMappingLabels, true>),
  ],
];

describe('types_models.ts matches the vendored OpenAPI', () => {
  it.each(CASES)('%s', (name, ours) => {
    const schema = schemas[name];
    expect(schema, `${name} missing from the vendored OpenAPI`).toBeDefined();
    expect(ours).toEqual(Object.keys(schema!.properties ?? {}).sort());
  });

  it('the sharing request requires exactly what the server requires', () => {
    const body: M.ModelSharingRequest = { shared: true, expected_revision: 1 };
    expect(body).toBeTruthy();
    expect(schemas.ModelSharingRequest!.required?.sort()).toEqual([
      'expected_revision',
      'shared',
    ]);
  });

  it('the served match kinds are exactly ModelClassMatch', () => {
    const entry = schemas.ModelClassMappingEntry!.properties!.match as {
      enum?: string[];
    };
    const ours: Record<M.ModelClassMatch, true> = {
      exact: true,
      case_insensitive: true,
      none: true,
    };
    expect(Object.keys(ours).sort()).toEqual([...(entry.enum ?? [])].sort());
  });
});
