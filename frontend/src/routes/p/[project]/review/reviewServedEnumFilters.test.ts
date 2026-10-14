/**
 * The generic served-enum review filter bar (`ReviewFilterSpec`, e.g.
 * Regions' `region_status`). Static source scan (no @testing-library/svelte
 * in this repo; see reviewFilterConsistency.test.ts's header comment for the
 * established precedent).
 */
import { describe, expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

const src = reviewSource('lib/review/reviewFilterController.svelte.ts');
const barSrc = reviewSource('lib/components/review/ReviewFilterBar.svelte');
const tabsSrc = reviewSource('lib/review/reviewTabsController.svelte.ts');
const loaderSrc = reviewSource('lib/review/reviewQueueLoader.svelte.ts');

describe('generic served-enum filter bar (no tab/param-specific code)', () => {
  it('renders one ServedFilterField (drawn by the spec kind) per activeFilterSpecs entry, keyed by spec.param', () => {
    expect(barSrc).toMatch(/\{#each activeFilterSpecs as spec \(spec\.param\)\}/);
    expect(barSrc).toMatch(/<ServedFilterField[\s\S]*?onchange=\{setEnumFilter\}/);
  });

  it("activeFilterSpecs is the active tab's served specs minus the params the page draws itself", () => {
    expect(src).toMatch(
      /const servedSpecs = \$derived\(\s*reviewTabsVocabularyStore\.filterSpecsFor\(activeTabEndpointId\),?\s*\);/,
    );
    expect(src).toMatch(/const activeFilterSpecs = \$derived\(\s*servedSpecs\.filter\(/);
    expect(src).toMatch(/!SELF_DRAWN_PARAMS\.has\(s\.param\)/);
  });

  it('_filter() forwards only params the active tab declares in filter_specs', () => {
    const filterFnStart = src.indexOf('function _filter()');
    const filterFnBody = src.slice(filterFnStart, src.indexOf('\n  }\n', filterFnStart));
    expect(filterFnBody).toMatch(/Object\.assign\(f, activeEnumParams\);/);
    // enumFilterValues is seeded from every URL param, so iterating it
    // directly would forward unrelated/stale params to the backend.
    expect(filterFnBody).not.toMatch(/enumFilterValues/);
    expect(src).toMatch(
      /for \(const spec of activeFilterSpecs\) \{\s*const value = enumFilterValues\[spec\.param\];\s*if \(value && value\.length > 0\) out\[spec\.param\] = value;\s*\}/,
    );
  });

  it('setEnumFilter persists the value in the URL (like preset)', () => {
    const fnStart = src.indexOf('function setEnumFilter(');
    expect(fnStart).toBeGreaterThan(-1);
    const fnBody = src.slice(fnStart, src.indexOf('\n  }\n', fnStart));
    // A list value is persisted as repeated keys.
    expect(fnBody).toMatch(/url\.searchParams\.append\(param, v\)/);
    expect(fnBody).toMatch(/url\.searchParams\.delete\(param\)/);
    expect(fnBody).toMatch(
      /replaceState\(resolve\(projectHref\(`\/review\$\{url\.search\}`\)\), \{\}\)/,
    );
  });

  it('enumFilterValues resets (state + URL) on every tab-click, not just on mount', () => {
    // The tab switch (tabs controller) hands its URL to the filter
    // controller's resetForTab, which clears the enum values in both.
    const idx = tabsSrc.indexOf("url.searchParams.set('tab', t.urlId);");
    expect(idx).toBeGreaterThan(-1);
    const nearby = tabsSrc.slice(idx, idx + 500);
    expect(nearby).toMatch(/opts\.resetTabFilters\(url\);/);
    const routeSrc = reviewSource('routes/p/[project]/review/+page.svelte');
    expect(routeSrc).toMatch(/resetTabFilters: \(url\) => filters\.resetForTab\(url\)/);
    const resetStart = src.indexOf('function resetForTab(url: URL)');
    expect(resetStart).toBeGreaterThan(-1);
    const reset = src.slice(resetStart, src.indexOf('\n  }\n', resetStart));
    expect(reset).toMatch(/for \(const param of Object\.keys\(enumFilterValues\)\)/);
    expect(reset).toMatch(/url\.searchParams\.delete\(param\);/);
    expect(reset).toMatch(/enumFilterValues = \{\};/);
  });

  it('the debounced filter effect keys on activeEnumParams, so a select change or a late-loading spec refetches', () => {
    // Keying on the raw enumFilterValues missed the case where a
    // URL-seeded ?region_status= was set before /review/tabs loaded: the
    // spec arriving changes what _filter() sends but not enumFilterValues.
    // e2e test_region_status_from_the_url_reaches_the_queue_request
    // covers the behavior; this pins the wiring.
    const effectIdx = loaderSrc.indexOf('let lastFilterKey: string | null = null;');
    expect(effectIdx).toBeGreaterThan(-1);
    const effectBody = loaderSrc.slice(effectIdx, effectIdx + 1500);
    expect(effectBody).toMatch(/void activeEnumParams;/);
    expect(effectBody).toMatch(
      /activeEnumParams,\s*activeUrlFilters,\s*itemFilterQuery,\s*\]\);/,
    );
    expect(effectBody).not.toMatch(/void enumFilterValues;/);
  });
});
