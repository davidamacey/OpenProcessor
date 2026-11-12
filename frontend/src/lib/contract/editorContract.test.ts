/**
 * `types_editor.ts` vs the generated contract: each wire type is compile-time exact and pinned to
 * the schema's property set, so a renamed field fails here before a write misses it.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_editor';

type Schema = { properties?: Record<string, unknown> };
const doc = spec as unknown as { components: { schemas: Record<string, Schema> } };
const schemas = doc.components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'ItemCreateRequest',
    keys<keyof T.ItemCreateBody>({ bbox_norm: true, class_name: true, request_id: true }),
  ],
  [
    'ItemCreated',
    keys<keyof T.ItemCreatedWire<unknown>>({
      item: true,
      created: true,
      image_boxes: true,
      image_counts: true,
      embedding_state: true,
      embedding_message: true,
    }),
  ],
  [
    'GeometryPatchRequest',
    keys<keyof T.GeometryPatchBody>({
      bbox_norm: true,
      expected_geometry_revision: true,
      drop_region_box_ids: true,
    }),
  ],
  [
    'GeometryWritten',
    keys<keyof T.GeometryWrittenWire<unknown>>({
      item: true,
      image_boxes: true,
      embedding_state: true,
      embedding_message: true,
    }),
  ],
  [
    'ImageLocate',
    keys<keyof T.ImageLocate>({
      image_id: true,
      in_view: true,
      index: true,
      total: true,
      cursor_before: true,
      cursor_after: true,
      reason: true,
      reason_message: true,
    }),
  ],
];

describe('editor wire types match the contract', () => {
  it.each(CASES)('%s has exactly the contract properties', (name, mine) => {
    expect(Object.keys(schemas[name].properties ?? {}).sort()).toEqual(mine);
  });
});
