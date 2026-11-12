/**
 * The queue-only filters (the region queue's status and text search, "include the test holdout",
 * ...) are the active queue's served specs minus the ones the shared item filter owns. They live
 * in the URL as plain parameters and reach the source with the Workspace filter. Static source
 * scan, as for the other Review wiring.
 */
import { describe, expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

const tabsSrc = reviewSource('lib/review/reviewTabsController.svelte.ts');
const loaderSrc = reviewSource('lib/review/reviewQueueLoader.svelte.ts');
const filtersSrc = reviewSource('lib/components/review/QueueFilters.svelte');

describe('queue-only served filters (no queue- or param-specific code)', () => {
  it("are the active queue's served specs the shared filter bar does not draw", () => {
    expect(tabsSrc).toMatch(
      /\.filterSpecsFor\(activeTabEndpointId\)\s*\.filter\(\(s\) => !ITEM_FILTER_PARAMS\.has\(s\.param\)\)/,
    );
  });

  it('only a param the queue serves is read from the URL', () => {
    expect(tabsSrc).toMatch(
      /for \(const spec of queueFilterSpecs\) \{\s*const all = ws\.searchParams\.getAll\(spec\.param\);/,
    );
  });

  it('are drawn one ServedFilterField per served spec, keyed by spec.param', () => {
    expect(filtersSrc).toMatch(
      /\{#each tabs\.queueFilterSpecs as spec \(spec\.param\)\}/,
    );
    expect(filtersSrc).toMatch(/<ServedFilterField/);
  });

  it('draw no capture-date markup of their own', () => {
    expect(filtersSrc).not.toMatch(/taken_(from|to)|type="date"/);
  });

  it('are persisted as plain URL parameters, repeated for a list value', () => {
    expect(filtersSrc).toMatch(/params\.delete\(param\);/);
    expect(filtersSrc).toMatch(/params\.append\(param, v\)/);
    expect(filtersSrc).toMatch(/ws\.patchUrl\(/);
  });

  it('reach the source with the Workspace filter, and a queue is never also a review_tab', () => {
    expect(loaderSrc).toMatch(
      /\.\.\.ws\.itemQuery\(withoutOpenVocab\),\s*\.\.\.tabs\.queueFilterValues,/,
    );
    expect(loaderSrc).toMatch(/if \(queue\) delete filter\.review_tab;/);
  });

  it('a change of queue, filter or sort reloads page 1 once per distinct source', () => {
    expect(loaderSrc).toMatch(
      /const sourceKey = \$derived\(JSON\.stringify\(sourceQuery\)\);/,
    );
    expect(loaderSrc).toMatch(/if \(key === lastKey\) return;/);
  });
});
