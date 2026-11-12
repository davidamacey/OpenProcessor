/**
 * The Workspace's selection summary and exclude reasons vs the generated contract.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type { ExcludeReasonEntry, SelectionSummary } from '$lib/types_itemFilter';

type Schema = { properties?: Record<string, unknown> };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;
const served = (name: string) => Object.keys(schemas[name]!.properties ?? {}).sort();
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

describe('workspace contract', () => {
  it('SelectionSummary has the declared keys', () => {
    expect(served('SelectionSummary')).toEqual(
      keys<keyof SelectionSummary>({
        detections: true,
        images: true,
        clusters: true,
        locked: true,
        in_holdout: true,
        by_class: true,
        taken_from: true,
        taken_to: true,
        located: true,
        join_truncated: true,
        summary_text: true,
      }),
    );
  });

  it('ExcludeReasonEntry has the declared keys', () => {
    expect(served('ExcludeReasonEntry')).toEqual(
      keys<keyof ExcludeReasonEntry>({ id: true, label: true }),
    );
  });
});
