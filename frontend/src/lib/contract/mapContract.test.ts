/**
 * `types_map.ts` vs the generated contract: each key map is compile-time exact against its type
 * and pinned to the schema's property set, and the cell collection keeps the GeoJSON shape the map
 * layer reads.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_map';

type Schema = { properties?: Record<string, unknown> };
const doc = spec as unknown as { components: { schemas: Record<string, Schema> } };
const schemas = doc.components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'MapMeta',
    keys<keyof T.MapMeta>({
      available: true,
      location_policy: true,
      n_located: true,
      n_without_location: true,
      bounds: true,
      tile_url: true,
      tile_attribution: true,
      max_zoom: true,
    }),
  ],
  ['GeoBox', keys<keyof T.GeoBox>({ west: true, south: true, east: true, north: true })],
  [
    'MapCellCollection',
    keys<keyof T.MapCellCollection>({
      type: true,
      features: true,
      precision: true,
      truncated: true,
      join_truncated: true,
      n_located_matching: true,
      n_unlocated_matching: true,
    }),
  ],
  [
    'MapCellFeature',
    keys<keyof T.MapCellFeature>({ type: true, geometry: true, properties: true }),
  ],
  [
    'MapCellProperties',
    keys<keyof T.MapCellProperties>({
      count: true,
      tile: true,
      bounds: true,
      image_id: true,
      thumbnail_url: true,
    }),
  ],
];

describe('map wire types equal the contract', () => {
  for (const [name, local] of CASES) {
    it(name, () => {
      expect(local).toEqual(Object.keys(schemas[name]?.properties ?? {}).sort());
    });
  }

  it('the map routes exist and cells are served as GeoJSON', () => {
    const paths = (spec as unknown as { paths: Record<string, Record<string, unknown>> })
      .paths;
    const cells = Object.keys(paths).find((p) =>
      p.endsWith('/projects/{project}/map/cells'),
    );
    expect(cells).toBeDefined();
    const get = paths[cells as string].get as {
      responses: Record<string, { content?: Record<string, unknown> }>;
    };
    expect(Object.keys(get.responses['200'].content ?? {})).toContain(
      'application/geo+json',
    );
    expect(
      Object.keys(paths).filter((p) => /\/projects\/\{project\}\/map(\/|$)/.test(p))
        .length,
    ).toBe(2);
  });
});
