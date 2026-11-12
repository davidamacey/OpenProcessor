import { describe, expect, it } from 'vitest';
import { PRESET_SCHEMAS } from '$lib/test/fixtures/presetSchemas';
import { schemaFields } from './presetSchemaFields';

const byPath = (kind: Parameters<typeof schemaFields>[0]) =>
  Object.fromEntries(schemaFields(kind).map((f) => [f.path, f]));

describe('schemaFields', () => {
  it('lists the cluster profile fields with choices, defaults, required and help', () => {
    const f = byPath(PRESET_SCHEMAS.cluster_profile);
    expect(f.target).toMatchObject({
      type: 'choice',
      required: true,
      choices: ['item', 'region_box'],
    });
    expect(f.target!.description).toMatch(/axis/);
    expect(f.backend).toMatchObject({ required: false, default: '"cpu"' });
    expect(f.backend!.choices).toEqual(['cpu', 'gpu_if_validated']);
  });

  it('descends into nested models with a dotted path and their limits', () => {
    const f = byPath(PRESET_SCHEMAS.cluster_profile);
    expect(f['refresh.max_deferral_s']).toMatchObject({
      depth: 1,
      type: 'integer',
      limits: ['at least 0'],
    });
    expect(f['refresh.growth_threshold']!.limits).toEqual(['above 0']);
  });

  it('reads lists of models, maps and nullable fields', () => {
    const f = byPath(PRESET_SCHEMAS.serving_profile);
    expect(f.servers).toMatchObject({ type: 'list of object', required: true });
    expect(f['servers[].numa_node']).toMatchObject({
      type: 'integer or null',
      default: 'null',
    });
    expect(f.load_list!.type).toBe('map of list of string');
    expect(f['servers[].gpu_uuids']!.limits).toEqual(['at least 1 item']);
  });

  it('gives every field a description from the server', () => {
    for (const schema of Object.values(PRESET_SCHEMAS)) {
      expect(schemaFields(schema).filter((f) => f.description === '')).toEqual([]);
    }
  });
});
