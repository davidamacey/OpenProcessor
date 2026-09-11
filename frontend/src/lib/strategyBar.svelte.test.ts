import { describe, expect, it } from 'vitest';
import { createStrategyBar } from './strategyBar.svelte';

describe('createStrategyBar', () => {
  it('starts on the default id with no filters, serializing to an empty object', () => {
    const bar = createStrategyBar();
    expect(bar.sort).toBe('default');
    expect(bar.minMistakenness).toBeNull();
    expect(bar.hideNearDuplicates).toBe(false);
    expect(bar.isDefault).toBe(true);
    expect(bar.toQueryParams()).toEqual({});
  });

  it('honors a custom defaultId', () => {
    const bar = createStrategyBar({ defaultId: 'outliers' });
    expect(bar.sort).toBe('outliers');
    expect(bar.isDefault).toBe(true);
    expect(bar.toQueryParams()).toEqual({});
  });

  it('serializes a non-default sort under the sort key', () => {
    const bar = createStrategyBar();
    bar.sort = 'mistakenness';
    expect(bar.isDefault).toBe(false);
    expect(bar.toQueryParams()).toEqual({ sort: 'mistakenness' });
  });

  it('serializes minMistakenness independently of the sort selection', () => {
    const bar = createStrategyBar();
    bar.minMistakenness = 0.5;
    expect(bar.sort).toBe('default'); // selecting a filter doesn't touch sort
    expect(bar.toQueryParams()).toEqual({ min_mistakenness: 0.5 });

    bar.sort = 'mistakenness';
    expect(bar.minMistakenness).toBe(0.5); // and vice versa
    expect(bar.toQueryParams()).toEqual({ sort: 'mistakenness', min_mistakenness: 0.5 });
  });

  it('serializes hideNearDuplicates as a bare boolean flag, only when true', () => {
    const bar = createStrategyBar();
    expect(bar.toQueryParams()).toEqual({});
    bar.hideNearDuplicates = true;
    expect(bar.toQueryParams()).toEqual({ hide_near_duplicates: true });
    bar.hideNearDuplicates = false;
    expect(bar.toQueryParams()).toEqual({});
  });

  it('combines all three independently-set fields in one query-param object', () => {
    const bar = createStrategyBar();
    bar.sort = 'uncertainty_entropy';
    bar.minMistakenness = 0.25;
    bar.hideNearDuplicates = true;
    expect(bar.toQueryParams()).toEqual({
      sort: 'uncertainty_entropy',
      min_mistakenness: 0.25,
      hide_near_duplicates: true,
    });
  });

  it('reset() restores every field to its default', () => {
    const bar = createStrategyBar({ defaultId: 'outliers' });
    bar.sort = 'mistakenness';
    bar.minMistakenness = 0.9;
    bar.hideNearDuplicates = true;
    expect(bar.isDefault).toBe(false);

    bar.reset();

    expect(bar.sort).toBe('outliers');
    expect(bar.minMistakenness).toBeNull();
    expect(bar.hideNearDuplicates).toBe(false);
    expect(bar.isDefault).toBe(true);
    expect(bar.toQueryParams()).toEqual({});
  });

  it('minMistakenness of exactly 0 still serializes (0 is a real threshold, not "unset")', () => {
    const bar = createStrategyBar();
    bar.minMistakenness = 0;
    expect(bar.toQueryParams()).toEqual({ min_mistakenness: 0 });
  });
});
