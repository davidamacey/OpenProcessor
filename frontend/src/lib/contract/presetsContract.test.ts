/**
 * `types_presets.ts` and `api_presets.ts` vs the vendored OpenAPI: each wire type's keys are exact
 * against the schema (`satisfies Record<keyof T, true>` rejects a missing or an extra key), and every
 * route each kind uses exists with the method the wrapper sends.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import { PRESET_KINDS } from '$lib/presets/presetKinds';
import type * as T from '$lib/types_presets';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const doc = spec as unknown as {
  components: { schemas: Record<string, Schema> };
  paths: Record<string, Record<string, unknown>>;
};

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'ResearchConfigDoc',
    keys({
      kind: true,
      name: true,
      revision: true,
      description: true,
      body: true,
      created_at: true,
      updated_at: true,
      updated_by: true,
    } satisfies Record<keyof T.PresetDoc, true>),
  ],
  [
    'ResearchConfigPut',
    keys({
      body: true,
      description: true,
      expected_revision: true,
    } satisfies Record<keyof T.PresetPutRequest, true>),
  ],
  [
    'ResearchActivateRequest',
    keys({
      axis: true,
      revision: true,
      expected_active: true,
    } satisfies Record<keyof T.PresetActivateRequest, true>),
  ],
  [
    'ServingActivateRequest',
    keys({
      revision: true,
      expected_revision: true,
    } satisfies Record<keyof T.ServingActivateRequest, true>),
  ],
  [
    'ResearchActive',
    keys({
      axis: true,
      state: true,
      name: true,
      revision: true,
      activated_at: true,
    } satisfies Record<keyof T.PresetActiveWire, true>),
  ],
  [
    'ResearchConfigSchema',
    keys({
      kind: true,
      axes: true,
      json_schema: true,
    } satisfies Record<keyof T.PresetSchemaWire, true>),
  ],
  [
    'ResearchAxis',
    keys({
      axis: true,
      label: true,
      target: true,
    } satisfies Record<keyof T.PresetAxisWire, true>),
  ],
  [
    'ServingActive',
    keys({
      name: true,
      revision: true,
      setting_revision: true,
      body: true,
    } satisfies Record<keyof T.ServingActiveWire, true>),
  ],
  [
    'ServingActiveState',
    keys({
      active: true,
    } satisfies Record<keyof T.ServingActiveStateWire, true>),
  ],
];

describe('preset wire types match the OpenAPI schemas', () => {
  it.each(CASES)('%s', (schema, expected) => {
    expect(Object.keys(doc.components.schemas[schema]!.properties!).sort()).toEqual(
      expected,
    );
  });

  it('the activation states are the schema enum', () => {
    const states = (
      doc.components.schemas.ResearchActive!.properties!.state as { enum: string[] }
    ).enum;
    const mine: T.PresetActiveWire['state'][] = ['none', 'active', 'off'];
    expect([...states].sort()).toEqual([...mine].sort());
  });
});

describe('every preset route exists with the method the wrapper sends', () => {
  const COLLECTION: Record<T.PresetKindId, string> = {
    serving_profile: '/curation/serving_profiles',
    cluster_profile: '/curation/projects/{project}/cluster_profiles',
    assist_loop_preset: '/curation/projects/{project}/assist_loop_presets',
    near_dup_profile: '/curation/projects/{project}/near_dup_profiles',
  };

  it.each(PRESET_KINDS.map((k) => [k.id] as const))('%s', (id) => {
    const base = COLLECTION[id];
    const has = (path: string, method: string) =>
      expect(Object.keys(doc.paths[path] ?? {}), `${method} ${path}`).toContain(method);
    has(base, 'get');
    has(`${base}/active`, 'get');
    has(`${base}/schema`, 'get');
    has(`${base}/{name}`, 'get');
    has(`${base}/{name}`, 'put');
    has(`${base}/{name}`, 'delete');
    has(`${base}/{name}/activate`, 'post');
  });
});
