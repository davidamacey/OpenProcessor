/**
 * `types_clusterList.ts` vs the vendored OpenAPI: the sort ids and
 * validation states equal the contract enums, the facets and flag shapes
 * carry exactly the schema's properties, and every card filter the list
 * sends is a declared query parameter of `GET /clusters`.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import {
  CLUSTER_SORT_IDS,
  VALIDATION_STATES,
  type ClusterFacets,
  type ClusterFlag,
  type ClusterListFilter,
  type ClusterSortOption,
  type ConfidenceBin,
  type FacetCount,
} from '$lib/types_clusterList';

type Prop = { enum?: string[]; anyOf?: Prop[] };
type Spec = {
  components: { schemas: Record<string, { properties: Record<string, Prop> }> };
  paths: Record<
    string,
    Record<string, { parameters?: { name: string; schema: Prop }[] }>
  >;
};
const S = spec as unknown as Spec;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const propsOf = (name: string) =>
  Object.keys(S.components.schemas[name]!.properties).sort();
const LIST = '/curation/projects/{project}/clusters';
const listParams = () => S.paths[LIST]!.get!.parameters!;

describe('cluster list contract', () => {
  it('CLUSTER_SORT_IDS equal the sort enum', () => {
    const sort = listParams().find((p) => p.name === 'sort')!;
    expect([...CLUSTER_SORT_IDS].sort()).toEqual([...sort.schema.enum!].sort());
  });

  it('VALIDATION_STATES equal the card enum and the filter enum', () => {
    const card = S.components.schemas.ClusterCard!.properties.validation_state!;
    expect([...VALIDATION_STATES].sort()).toEqual([...card.enum!].sort());
    const filter = listParams().find((p) => p.name === 'validation_state')!;
    const branch = filter.schema.anyOf!.find((b) => b.enum)!;
    expect([...VALIDATION_STATES].sort()).toEqual([...branch.enum!].sort());
  });

  it('every card filter is a declared query parameter', () => {
    const declared = new Set(listParams().map((p) => p.name));
    const filter = keys<keyof ClusterListFilter>({
      dominant_class: true,
      detector_class: true,
      min_size: true,
      max_size: true,
      validation_state: true,
      flag: true,
      needs_review: true,
    });
    for (const k of filter) expect(declared.has(k), k).toBe(true);
  });

  it('flag, sort option, facet and bin shapes carry the schema properties', () => {
    expect(keys<keyof ClusterFlag>({ id: true, label: true, description: true })).toEqual(
      propsOf('ClusterFlag'),
    );
    expect(keys<keyof ClusterSortOption>({ id: true, label: true })).toEqual(
      propsOf('ClusterSortOption'),
    );
    expect(keys<keyof FacetCount>({ value: true, count: true })).toEqual(
      propsOf('FacetCount'),
    );
    expect(keys<keyof ConfidenceBin>({ min: true, max: true, count: true })).toEqual(
      propsOf('ConfidenceBin'),
    );
    expect(
      keys<keyof ClusterFacets>({
        cluster_id: true,
        total: true,
        detector_classes: true,
        class_sources: true,
        validation: true,
        review_tabs: true,
        detector_confidence: true,
        detector_confidence_missing: true,
      }),
    ).toEqual(propsOf('ClusterFacets'));
  });

  it('the facets route exists', () => {
    expect(
      S.paths['/curation/projects/{project}/clusters/{cluster_id}/facets'],
    ).toBeDefined();
  });
});
