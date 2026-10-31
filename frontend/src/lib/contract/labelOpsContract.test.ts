/**
 * `types_labelOps.ts` vs the generated contract. Each key map is
 * compile-time exact against its type and pinned to the schema's property
 * set, so a served rename fails here instead of rendering "—".
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_labelOps';
import { LABEL_OP_STATUSES, LABEL_OPS_MAX_LIMIT } from '$lib/types_labelOps';

type Schema = { properties?: Record<string, { enum?: string[] }> };
type Param = { name: string; schema: { maximum?: number } };
const doc = spec as unknown as {
  components: { schemas: Record<string, Schema> };
  paths: Record<string, Record<string, { parameters?: Param[] }>>;
};
const schemas = doc.components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'LabelOp',
    keys<keyof T.LabelOp>({
      op_id: true,
      kind: true,
      at: true,
      finished_at: true,
      class_name: true,
      target_cluster_id: true,
      target_kind: true,
      label_source: true,
      target: true,
      selected: true,
      count: true,
      conflicts: true,
      current: true,
      status: true,
      undone: true,
      undoable: true,
      last_undo: true,
    }),
  ],
  [
    'LabelOpList',
    keys<keyof T.LabelOpList>({ items: true, limit: true, retention: true }),
  ],
  [
    'LabelOpUndoPass',
    keys<keyof T.LabelOpUndoPass>({
      at: true,
      restored: true,
      already_undone: true,
      kept_changed: true,
      kept_holdout: true,
      kept_stale_class: true,
      conflicts: true,
    }),
  ],
  ['LabelOpUndoRequest', keys<keyof T.LabelOpUndoRequest>({ dry_run: true })],
  [
    'LabelOpUndoResponse',
    keys<keyof T.LabelOpUndoResponse>({
      op_id: true,
      dry_run: true,
      status: true,
      members_known: true,
      matched: true,
      restored: true,
      already_undone: true,
      kept_changed: true,
      kept_holdout: true,
      kept_stale_class: true,
      conflicts: true,
      conflict_ids: true,
    }),
  ],
];

describe('label operations contract', () => {
  for (const [schema, expected] of CASES) {
    it(`${schema} has exactly the keys the frontend type declares`, () => {
      const served = schemas[schema];
      expect(served, `${schema} missing from the contract`).toBeDefined();
      expect(Object.keys(served!.properties ?? {}).sort()).toEqual(expected);
    });
  }

  it('the status ids are the served enum, in the served order', () => {
    expect(schemas.LabelOp!.properties!.status!.enum).toEqual([...LABEL_OP_STATUSES]);
  });

  it('the list limit maximum is the served one', () => {
    const get = doc.paths['/curation/projects/{project}/label_ops']!.get!;
    const limit = get.parameters!.find((p) => p.name === 'limit')!;
    expect(limit.schema.maximum).toBe(LABEL_OPS_MAX_LIMIT);
  });
});
