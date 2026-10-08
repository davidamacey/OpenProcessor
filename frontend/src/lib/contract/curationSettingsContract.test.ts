/**
 * `CurationSettings` / `ResourceLink` (src/lib/curationSettings.ts) vs the
 * vendored OpenAPI: key maps are compile-time exact against the interfaces
 * and pinned to the schema's property sets and enums, so a served rename
 * fails here instead of rendering a blank Resources menu.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type { CurationSettings, ResourceLink } from '$lib/curationSettings';
import { RESOURCE_KINDS, RESOURCE_STATUSES } from '$lib/curationSettings';

type Schema = {
  properties?: Record<string, { enum?: string[] }>;
  required?: string[];
};
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

describe('curation settings contract', () => {
  it('ResourceLink keys match the served schema', () => {
    const ours = keys({
      id: true,
      label: true,
      url: true,
      kind: true,
      status: true,
      hint: true,
      reachable: true,
    } satisfies Record<keyof ResourceLink, true>);
    expect(Object.keys(schemas.ResourceLink!.properties!).sort()).toEqual(ours);
  });

  it('ResourceLink kind and status enums match the served schema', () => {
    const p = schemas.ResourceLink!.properties!;
    expect([...RESOURCE_KINDS].sort()).toEqual([...p.kind!.enum!].sort());
    expect([...RESOURCE_STATUSES].sort()).toEqual([...p.status!.enum!].sort());
  });

  it('CurationSettingsResponse serves resource_links and no monitoring_links', () => {
    const ours = keys({
      defaults: true,
      updated_at: true,
      updated_by: true,
      resource_links: true,
    } satisfies Record<keyof CurationSettings, true>);
    expect(Object.keys(schemas.CurationSettingsResponse!.properties!).sort()).toEqual(
      ours,
    );
  });
});
