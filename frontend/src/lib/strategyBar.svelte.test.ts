import { describe, expect, it } from 'vitest';
import { createStrategyBar } from './strategyBar.svelte';

describe('createStrategyBar', () => {
  it('starts on the default id with no filters, serializing to an empty object', () => {
    const bar = createStrategyBar();
    expect(bar.sort).toBe('default');
    expect(bar.minMistakenness).toBeNull();
    expect(bar.hideNearDuplicates).toBe(false);
    expect(bar.k).toBeNull();
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
    bar.k = 250;
    expect(bar.isDefault).toBe(false);

    bar.reset();

    expect(bar.sort).toBe('outliers');
    expect(bar.minMistakenness).toBeNull();
    expect(bar.hideNearDuplicates).toBe(false);
    expect(bar.k).toBeNull();
    expect(bar.isDefault).toBe(true);
    expect(bar.toQueryParams()).toEqual({});
  });

  it('minMistakenness of exactly 0 still serializes (0 is a real threshold, not "unset")', () => {
    const bar = createStrategyBar();
    bar.minMistakenness = 0;
    expect(bar.toQueryParams()).toEqual({ min_mistakenness: 0 });
  });

  // Phase 4 — 'diverse' mode's k stepper (docs/curation-strategy-plan-2026-09.md
  // §5/§7). k is deliberately NOT part of toQueryParams(): only
  // /clusters/[id] forwards it, and it reads bar.k directly the same way
  // it already reads bar.sort directly (see this file's header comment
  // and src/routes/clusters/[id]/+page.svelte's cropQuery()).
  describe('k (diverse-mode pool-scale selection count)', () => {
    it('defaults to null and is settable independently of sort/filters', () => {
      const bar = createStrategyBar();
      expect(bar.k).toBeNull();
      bar.k = 120;
      expect(bar.k).toBe(120);
      expect(bar.sort).toBe('default'); // setting k doesn't touch sort
      expect(bar.toQueryParams()).toEqual({}); // never serialized here
    });

    it('marks isDefault false once k is set, even with sort/filters untouched', () => {
      const bar = createStrategyBar();
      expect(bar.isDefault).toBe(true);
      bar.k = 60;
      expect(bar.isDefault).toBe(false);
    });

    it('null is a real "unset" value distinct from 0', () => {
      const bar = createStrategyBar();
      bar.k = 0;
      expect(bar.isDefault).toBe(false);
      expect(bar.k).toBe(0);
      bar.k = null;
      expect(bar.isDefault).toBe(true);
    });
  });

  // P2-10: /review's diverse overlay reuses toQueryParams() for its
  // non-diverse filters, so an overlay id selected as `sort` must never
  // leak into the `sort` query param — /curation/review/{tab} 400s on
  // `sort=diverse`, since diverse selection is a wholly separate call
  // (POST /curation/select/diverse), not a review-queue sort.
  describe('overlayIds (P2-10)', () => {
    it('omits sort entirely when the current selection is a registered overlay id', () => {
      const bar = createStrategyBar({ overlayIds: ['diverse'] });
      bar.sort = 'diverse';
      expect(bar.toQueryParams()).toEqual({});
    });

    it('still serializes a real sort id even with overlayIds configured', () => {
      const bar = createStrategyBar({ overlayIds: ['diverse'] });
      bar.sort = 'mistakenness';
      expect(bar.toQueryParams()).toEqual({ sort: 'mistakenness' });
    });

    it('combines with other filters, still omitting only the overlay sort key', () => {
      const bar = createStrategyBar({ overlayIds: ['diverse'] });
      bar.sort = 'diverse';
      bar.minMistakenness = 0.5;
      expect(bar.toQueryParams()).toEqual({ min_mistakenness: 0.5 });
    });

    it('defaults to no overlay ids, matching pre-P2-10 behavior', () => {
      const bar = createStrategyBar();
      bar.sort = 'diverse';
      expect(bar.toQueryParams()).toEqual({ sort: 'diverse' });
    });
  });
});
