/**
 * Selection writes and `export/yolo item_filter` vs the vendored OpenAPI
 * (OpenProcessor e817e4f4): the four batch write requests carry
 * `selection` and `dry_run`, a dry run answers `{dry_run, selected}`, the
 * real responses carry `updated_ids`, `ExportYoloRequest` has
 * `item_filter`, and every region write serves `vector_refresh`.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type {
  SelectionDryRun,
  SelectionExcludeResult,
  VectorRefresh,
} from '$lib/types_itemFilter';
import type { BulkLabelResult } from '$lib/types';

type Prop = { $ref?: string; anyOf?: Prop[] };
type Schema = { properties?: Record<string, Prop>; required?: string[] };
const S = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;
const props = (n: string) => Object.keys(S[n]?.properties ?? {}).sort();

describe('selection requests', () => {
  for (const n of [
    'CropBatchLabelRequest',
    'CropExcludeRequest',
    'CropUnexcludeRequest',
    'CropMoveRequest',
  ]) {
    it(`${n} takes crop_ids or selection, plus dry_run`, () => {
      const p = props(n);
      expect(p).toContain('selection');
      expect(p).toContain('dry_run');
      expect(p).toContain('crop_ids');
      expect(S[n].required ?? []).not.toContain('crop_ids');
    });
  }

  it('ExportYoloRequest has item_filter', () => {
    expect(props('ExportYoloRequest')).toContain('item_filter');
  });
});

describe('selection responses', () => {
  it('SelectionDryRun is exactly {dry_run, selected}', () => {
    const local = { dry_run: true, selected: true } satisfies Record<
      keyof SelectionDryRun,
      true
    >;
    expect(Object.keys(local).sort()).toEqual(props('SelectionDryRunResponse'));
  });

  it('the relabel / move response carries updated, updated_ids, conflicts', () => {
    const local = { updated: true, updated_ids: true, conflicts: true } satisfies Record<
      keyof BulkLabelResult,
      true
    >;
    expect(Object.keys(local).sort()).toEqual(props('BatchRelabelResponse'));
  });

  it('exclude / unexclude carry updated_ids and errors', () => {
    const local = {
      excluded: true,
      unexcluded: true,
      updated_ids: true,
      errors: true,
    } satisfies Record<keyof SelectionExcludeResult, true>;
    const served = new Set([
      ...props('BatchExcludeResponse'),
      ...props('BatchUnexcludeResponse'),
    ]);
    expect(Object.keys(local).sort()).toEqual([...served].sort());
  });
});

describe('region writes serve vector_refresh', () => {
  it('VectorRefresh is exactly {embedded, pending}', () => {
    const local = { embedded: true, pending: true } satisfies Record<
      keyof VectorRefresh,
      true
    >;
    expect(Object.keys(local).sort()).toEqual(props('VectorRefresh'));
  });

  for (const n of [
    'RegionBatchWriteResponse',
    'RegionItemWriteResponse',
    'RegionBoxPatchResponse',
  ]) {
    it(`${n} requires vector_refresh`, () => {
      expect(props(n)).toContain('vector_refresh');
      expect(S[n].required).toContain('vector_refresh');
    });
  }
});
