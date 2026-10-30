import { describe, expect, it } from 'vitest';
import {
  EMPTY_CLUSTER_FILTER,
  clusterFilterFromUrl,
  clusterFilterToQuery,
  clusterFilterToUrl,
  isClusterFilterActive,
} from './clusterListFilter';
import type { ClusterListFilter } from '$lib/types_clusterList';

const FULL: ClusterListFilter = {
  dominant_class: ['class_a', 'class_b'],
  detector_class: ['widget_x'],
  min_size: 5,
  max_size: 400,
  validation_state: 'partial',
  flag: ['mixed_detector_classes', 'unvalidated'],
  needs_review: true,
};

describe('clusterFilterToQuery', () => {
  it('sends nothing for the empty filter', () => {
    expect(clusterFilterToQuery(EMPTY_CLUSTER_FILTER)).toEqual({});
  });

  it('sends every set field under its served parameter name, unchanged', () => {
    expect(clusterFilterToQuery(FULL)).toEqual({
      dominant_class: ['class_a', 'class_b'],
      detector_class: ['widget_x'],
      min_size: 5,
      max_size: 400,
      validation_state: 'partial',
      flag: ['mixed_detector_classes', 'unvalidated'],
      needs_review: true,
    });
  });

  it('keeps a size bound of zero (a real served value)', () => {
    expect(clusterFilterToQuery({ ...EMPTY_CLUSTER_FILTER, min_size: 0 })).toEqual({
      min_size: 0,
    });
  });
});

describe('URL round trip', () => {
  it('writes and reads the same filter and sort', () => {
    const p = new URLSearchParams('class=7&class_name=x');
    clusterFilterToUrl(p, 'purity_desc', FULL);
    expect(p.get('class')).toBe('7');
    expect(p.getAll('flag')).toEqual(['mixed_detector_classes', 'unvalidated']);
    expect(clusterFilterFromUrl(p)).toEqual({ sort: 'purity_desc', filter: FULL });
  });

  it('removes keys that are now empty and leaves foreign keys alone', () => {
    const p = new URLSearchParams('class=7&min_size=3&flag=unvalidated');
    clusterFilterToUrl(p, null, EMPTY_CLUSTER_FILTER);
    expect(p.toString()).toBe('class=7');
  });

  it('returns null when the URL names no cluster-list key', () => {
    expect(clusterFilterFromUrl(new URLSearchParams('class=7'))).toBeNull();
  });

  it('drops an unknown sort, an unknown validation state and a non-numeric size', () => {
    const p = new URLSearchParams('sort=bogus&validation_state=x&min_size=abc&flag=a');
    expect(clusterFilterFromUrl(p)).toEqual({
      sort: null,
      filter: { ...EMPTY_CLUSTER_FILTER, flag: ['a'] },
    });
  });
});

describe('isClusterFilterActive', () => {
  it('is false for the empty filter and true for any set field', () => {
    expect(isClusterFilterActive(EMPTY_CLUSTER_FILTER)).toBe(false);
    expect(isClusterFilterActive({ ...EMPTY_CLUSTER_FILTER, needs_review: true })).toBe(
      true,
    );
    expect(isClusterFilterActive({ ...EMPTY_CLUSTER_FILTER, min_size: 0 })).toBe(true);
    expect(isClusterFilterActive(FULL)).toBe(true);
  });
});
