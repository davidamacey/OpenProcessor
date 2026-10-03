/**
 * `types_itemFilter.ts` vs the vendored OpenAPI (OpenProcessor v0.4.0,
 * fce17771). The backend serves no vocabulary for the shared filter's
 * enums, so the const arrays are pinned to the contract enums here; the
 * `ItemFilter` / `ItemSelection` key maps are compile-time exact against
 * their interfaces (`satisfies Record<keyof T, true>`) and pinned to the
 * schema properties; every `ItemFilterQuery` key must be a declared query
 * parameter of `GET /crops`.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import {
  EMBEDDING_STATES,
  ITEM_ORIGINS,
  REVIEW_STATUSES,
  SELECTION_SAMPLES,
} from '$lib/types_itemFilter';
import type * as T from '$lib/types_itemFilter';

type Prop = { enum?: string[]; items?: Prop; anyOf?: Prop[] };
type Schema = { properties?: Record<string, Prop> };
type Param = { name: string; in: string };
type Spec = {
  components: { schemas: Record<string, Schema> };
  paths: Record<string, Record<string, { parameters?: Param[] }>>;
};
const S = spec as unknown as Spec;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

function schema(name: string): Schema {
  const s = S.components.schemas[name];
  if (!s?.properties) throw new Error(`schema not found: ${name}`);
  return s;
}

/** The string enum a property declares, directly, on its array items, or
 *  inside an `anyOf` (nullable) branch. */
function enumOf(p: Prop | undefined): string[] {
  if (!p) throw new Error('property not found');
  if (p.enum) return [...p.enum].sort();
  if (p.items) return enumOf(p.items);
  for (const branch of p.anyOf ?? []) {
    if (branch.enum || branch.items) return enumOf(branch);
  }
  throw new Error('no enum on property');
}

const prop = (name: string, key: string) => schema(name).properties?.[key];

describe('shared item-filter enums equal the contract enums', () => {
  it('EMBEDDING_STATES', () => {
    expect([...EMBEDDING_STATES].sort()).toEqual(
      enumOf(prop('ItemFilter', 'embedding_state')),
    );
    expect([...EMBEDDING_STATES].sort()).toEqual(
      enumOf(prop('ItemDoc', 'embedding_state')),
    );
  });

  it('ITEM_ORIGINS', () => {
    expect([...ITEM_ORIGINS].sort()).toEqual(enumOf(prop('ItemFilter', 'origin')));
  });

  it('REVIEW_STATUSES', () => {
    expect([...REVIEW_STATUSES].sort()).toEqual(
      enumOf(prop('ItemFilter', 'review_status')),
    );
  });

  it('SELECTION_SAMPLES', () => {
    expect([...SELECTION_SAMPLES].sort()).toEqual(
      enumOf(prop('ItemSelection', 'sample')),
    );
    expect([...SELECTION_SAMPLES].sort()).toEqual(
      enumOf(prop('ReprocessTargets', 'sample')),
    );
  });
});

describe('item-filter body schemas', () => {
  it('ItemFilter has exactly the served properties', () => {
    const local = keys({
      class_id: true,
      class_names: true,
      class_source: true,
      classifier_conf_lt: true,
      cluster_id: true,
      conf_max: true,
      conf_min: true,
      dataset_split: true,
      embedding_state: true,
      exclude_class_names: true,
      import_id: true,
      item_text: true,
      label_source: true,
      label_validated: true,
      max_area: true,
      max_rank: true,
      min_area: true,
      min_blur_ratio: true,
      needs_new_class: true,
      on_negative_frame: true,
      open_vocab_set: true,
      origin: true,
      proposed_by_import: true,
      region_gate_skipped: true,
      review_dismissed: true,
      review_status: true,
      source: true,
      source_prompt: true,
    } satisfies Record<keyof T.ItemFilter, true>);
    expect(local).toEqual(Object.keys(schema('ItemFilter').properties ?? {}).sort());
  });

  it('ItemSelection has exactly the served properties', () => {
    const local = keys({
      crop_ids: true,
      filter: true,
      include_excluded: true,
      include_test: true,
      limit: true,
      sample: true,
      seed: true,
    } satisfies Record<keyof T.ItemSelection, true>);
    expect(local).toEqual(Object.keys(schema('ItemSelection').properties ?? {}).sort());
  });
});

describe('ItemFilterQuery', () => {
  it('every key is a declared query parameter of GET /crops', () => {
    const local = keys({
      class_name: true,
      exclude_class_name: true,
      conf_min: true,
      conf_max: true,
      min_area: true,
      max_area: true,
      max_rank: true,
      origin: true,
      embedding_state: true,
      review_status: true,
      open_vocab_set: true,
      source_prompt: true,
    } satisfies Record<keyof T.ItemFilterQuery, true>);
    const path = Object.keys(S.paths).find((p) =>
      /\/projects\/\{project\}\/crops$/.test(p),
    );
    if (!path) throw new Error('GET /crops not found');
    const declared = new Set(
      (S.paths[path].get.parameters ?? [])
        .filter((p) => p.in === 'query')
        .map((p) => p.name),
    );
    expect(local.filter((k) => !declared.has(k))).toEqual([]);
  });
});
