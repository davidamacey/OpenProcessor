/**
 * `types_table.ts` vs the generated contract: each key map is compile-time exact against its
 * type and pinned to the schema's property set; the two table routes exist.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_table';

type Schema = { properties?: Record<string, unknown> };
const doc = spec as unknown as {
  components: { schemas: Record<string, Schema> };
  paths: Record<string, unknown>;
};

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'TableColumn',
    keys<keyof T.TableColumn>({
      id: true,
      label: true,
      sortable: true,
      align: true,
      kind: true,
    }),
  ],
  [
    'TableColumns',
    keys<keyof T.TableColumns>({ rows: true, columns: true, default: true }),
  ],
  ['TableCell', keys<keyof T.TableCell>({ text: true, value: true, tone: true })],
  [
    'TableRow',
    keys<keyof T.TableRow>({
      crop_id: true,
      image_id: true,
      thumbnail_url: true,
      cells: true,
    }),
  ],
  [
    'TablePage',
    keys<keyof T.TablePage>({
      rows: true,
      columns: true,
      items: true,
      next_cursor: true,
      total: true,
      limit: true,
      sort_applied: true,
      sort_options: true,
      hidden: true,
      join_truncated: true,
      empty_reason: true,
      empty_message: true,
    }),
  ],
];

describe('table wire types equal the contract', () => {
  for (const [name, local] of CASES) {
    it(name, () => {
      expect(local).toEqual(
        Object.keys(doc.components.schemas[name]?.properties ?? {}).sort(),
      );
    });
  }

  it('the table routes exist', () => {
    for (const suffix of ['/table', '/table/columns']) {
      expect(
        Object.keys(doc.paths).some((p) => p.endsWith(`/projects/{project}${suffix}`)),
      ).toBe(true);
    }
  });
});
