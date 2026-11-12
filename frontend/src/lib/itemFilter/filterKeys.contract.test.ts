/**
 * Every query parameter `GET /crops` takes for the shared filter is either a URL key the filter
 * state writes or named here as not on the bar (a paging or sort option, a route-only scope, or a
 * value another control owns).
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import { URL_KEYS } from './itemFilterState.svelte';

const NOT_ON_BAR = [
  // paging, ordering and lookups
  'page',
  'page_size',
  'limit',
  'sort',
  'order',
  'k',
  'ids',
  'project',
  // the scope or the view owns these
  'image_id',
  'near_dup_group',
  'review_tab',
  'include_test',
  'include_excluded',
  // route-only item scopes with their own controls
  'class_source',
  'classifier_conf_lt',
  'dataset_split',
  'detector_class_name',
  'detector_conf_max',
  'detector_conf_min',
  'import_id',
  'item_text',
  'label_source',
  'label_validated',
  'min_blur_ratio',
  'needs_new_class',
  'on_negative_frame',
  'proposed_by_import',
  'region_text',
  'region_text_match',
  'review_dismissed',
  'same_text_as',
  'source',
];

describe('the /crops filter parameters', () => {
  it('are a URL key or listed as not on the bar', () => {
    const paths = (
      spec as unknown as {
        paths: Record<string, { get?: { parameters?: { name: string }[] } }>;
      }
    ).paths;
    const crops = Object.keys(paths).find((p) => p.endsWith('/projects/{project}/crops'));
    const names = (paths[crops as string].get?.parameters ?? []).map((p) => p.name);
    const unknown = names.filter(
      (n) => !(URL_KEYS as readonly string[]).includes(n) && !NOT_ON_BAR.includes(n),
    );
    expect(unknown).toEqual([]);
  });
});
