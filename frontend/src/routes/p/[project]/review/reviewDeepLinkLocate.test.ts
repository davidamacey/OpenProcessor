/**
 * The `/review?crop_id=` deep link used to page 1..loc.page sequentially via
 * `queue.loadMore()` in a while loop (103 requests and 5.7s to reach rank
 * 3000) and rendered item #1 with its keybindings live for the ~4s that
 * took, so a keypress during that window acted on the wrong crop. It now
 * asks the backend where the crop sits (`locateInReviewQueue`), fetches only
 * the page that starts at the located item (`queue.loadAt(loc.cursor_before, loc.index)`, see
 * src/lib/pager.svelte.ts),
 * and gates the tab-action keybinding effect (plus the rendered item) on an
 * `awaitingDeepLink` state so nothing is live or shown until the target
 * lands.
 *
 * Static source scan (no `@testing-library/svelte` mount harness for a page
 * this size): which pager call fires, and which state gates the keybinding
 * effect.
 */
import { describe, expect, it } from 'vitest';
import { reviewPageSource, reviewSource } from '$lib/testing/reviewPageSources';

const loaderSrc = reviewSource('lib/review/reviewQueueLoader.svelte.ts');
const tabsSrc = reviewSource('lib/review/reviewTabsController.svelte.ts');
const keyboardSrc = reviewSource('lib/review/reviewKeyboard.svelte.ts');
const pageSrc = reviewSource('routes/p/[project]/review/+page.svelte');

function fn(name: string): string {
  const m = loaderSrc.match(new RegExp(`async function ${name}\\([\\s\\S]*?\\n {2}\\}`));
  expect(m, `function ${name} not found`).not.toBeNull();
  return m![0];
}

describe('/review?crop_id= deep link calls locateInReviewQueue, not a paging scan', () => {
  it('jumpToPendingCrop calls locateInReviewQueue with the effective tab and current filters', () => {
    const fn = loaderSrc.match(/async function jumpToPendingCrop\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(
      /await locateInReviewQueue\(\s*endpointForTab\(effectiveTab\),\s*cropId,\s*_filter\(\),\s*\)/,
    );
    // Fetches only the page that starts at the located item. The original
    // while-loop paged 1..loc.page one request per page (103 requests / 5.7s
    // at rank 3000, with item #1's keybindings live the whole time); that
    // paging loop is the bug this guards against.
    expect(fn).toMatch(/await queue\.loadAt\(loc\.cursor_before, loc\.index\)/);
    expect(fn).not.toMatch(/await queue\.loadMore\(\)/);
  });

  it('never references the deleted DEEP_LINK_MAX_ITEMS paging-scan constant', () => {
    expect(reviewPageSource()).not.toMatch(/DEEP_LINK_MAX_ITEMS/);
  });
});

describe('jumpToPendingCrop fetches only the located page', () => {
  const body = fn('jumpToPendingCrop');

  it('calls queue.loadAt(loc.cursor_before, loc.index), not a loadMore() paging loop', () => {
    expect(body).toMatch(/queue\.loadAt\(loc\.cursor_before, loc\.index\)/);
    expect(body).not.toMatch(/await queue\.loadMore\(\)/);
  });

  it('clears awaitingDeepLink on every exit path (finally)', () => {
    const finallyBlock = body.match(/finally\s*\{[\s\S]*?\}\s*$/)?.[0];
    expect(finallyBlock).not.toBeUndefined();
    expect(finallyBlock).toMatch(/awaitingDeepLink = false;/);
  });

  it('clears awaitingDeepLink on the diverse/search early-return too', () => {
    const earlyReturn = body.match(
      /if \(diverseMode \|\| searchModeActive\) \{[\s\S]*?\n {4}\}/,
    )?.[0];
    expect(earlyReturn).not.toBeUndefined();
    expect(earlyReturn).toMatch(/awaitingDeepLink = false;/);
  });
});

describe('awaitingDeepLink gates keybindings and rendering', () => {
  it('seeds true only when a crop_id deep link is present', () => {
    expect(tabsSrc).toMatch(
      /let awaitingDeepLink = \$state<boolean>\(deepLink\.cropId != null\);/,
    );
  });

  it('the tab-action keybinding $effect returns before registering anything while awaiting', () => {
    const idx = keyboardSrc.indexOf('$effect(() => {\n    // DQ-M7');
    expect(
      idx,
      'keybinding effect with the DQ-M7 guard comment not found',
    ).toBeGreaterThan(-1);
    const slice = keyboardSrc.slice(idx, idx + 400);
    expect(slice).toMatch(/if \(awaitingDeepLink\) return;/);
  });

  it('the body renders a locating state instead of the (possibly page-1) current item while awaiting', () => {
    expect(pageSrc).toMatch(/\{:else if awaitingDeepLink\}/);
    const idx = pageSrc.indexOf('{:else if awaitingDeepLink}');
    const slice = pageSrc.slice(idx, idx + 350);
    expect(slice).toMatch(/Locating crop/);
  });

  it('switching tabs manually also clears awaitingDeepLink, not just pendingCropId', () => {
    const idx = tabsSrc.indexOf('tab = t.id;\n    pendingCropId = null;');
    expect(idx).toBeGreaterThan(-1);
    const slice = tabsSrc.slice(idx, idx + 120);
    expect(slice).toMatch(/awaitingDeepLink = false;/);
    // ...and that is the handler every tab button runs.
    expect(reviewSource('lib/components/review/ReviewTabsBar.svelte')).toMatch(
      /onclick=\{\(\) => selectTab\(t\)\}/,
    );
  });
});
