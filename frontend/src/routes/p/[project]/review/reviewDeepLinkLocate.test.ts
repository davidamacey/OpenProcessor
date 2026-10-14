/**
 * The `/review?crop_id=` deep link used to page 1..loc.page sequentially via
 * `queue.loadMore()` in a while loop (103 requests and 5.7s to reach rank
 * 3000) and rendered item #1 with its keybindings live for the ~4s that
 * took, so a keypress during that window acted on the wrong crop. It now
 * asks the backend where the crop sits (`locateInReviewQueue`), fetches only
 * the located page (`queue.loadPage(loc.page)`, see src/lib/pager.svelte.ts),
 * and gates the tab-action keybinding effect (plus the rendered item) on an
 * `awaitingDeepLink` state so nothing is live or shown until the target
 * lands.
 *
 * Static source scan (no `@testing-library/svelte` mount harness for a page
 * this size): which pager call fires, and which state gates the keybinding
 * effect.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

function fn(name: string): string {
  const m = src.match(new RegExp(`async function ${name}\\([\\s\\S]*?\\n {2}\\}`));
  expect(m, `function ${name} not found`).not.toBeNull();
  return m![0];
}

describe('/review?crop_id= deep link calls locateInReviewQueue, not a paging scan', () => {
  it('jumpToPendingCrop calls locateInReviewQueue with the effective tab, page size and current filters', () => {
    const fn = src.match(/async function jumpToPendingCrop\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(
      /await locateInReviewQueue\(\s*endpointForTab\(effectiveTab\),\s*cropId,\s*pageSize,\s*_filter\(\),\s*\)/,
    );
    // Fetches only the located page via queue.loadPage(loc.page) — the
    // original while-loop here paged 1..loc.page one request per page
    // (103 requests / 5.7s at rank 3000/page 101, with item #1's
    // keybindings live the whole time), which was a regression this test
    // used to pin as the intended behavior. That paging loop is the bug this guards against.
    expect(fn).toMatch(/await queue\.loadPage\(loc\.page\)/);
    expect(fn).not.toMatch(/while \(queue\.loadedPages < loc\.page/);
  });

  it('never references the deleted DEEP_LINK_MAX_ITEMS paging-scan constant', () => {
    expect(src).not.toMatch(/DEEP_LINK_MAX_ITEMS/);
  });
});

describe('jumpToPendingCrop fetches only the located page', () => {
  const body = fn('jumpToPendingCrop');

  it('calls queue.loadPage(loc.page), not a loadMore() paging loop', () => {
    expect(body).toMatch(/queue\.loadPage\(loc\.page\)/);
    expect(body).not.toMatch(/while\s*\(\s*queue\.loadedPages\s*<\s*loc\.page/);
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
    expect(src).toMatch(
      /let awaitingDeepLink = \$state<boolean>\(deepLink\.cropId != null\);/,
    );
  });

  it('the tab-action keybinding $effect returns before registering anything while awaiting', () => {
    const idx = src.indexOf('$effect(() => {\n    // DQ-M7');
    expect(
      idx,
      'keybinding effect with the DQ-M7 guard comment not found',
    ).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 400);
    expect(slice).toMatch(/if \(awaitingDeepLink\) return;/);
  });

  it('the body renders a locating state instead of the (possibly page-1) current item while awaiting', () => {
    expect(src).toMatch(/\{:else if awaitingDeepLink\}/);
    const idx = src.indexOf('{:else if awaitingDeepLink}');
    const slice = src.slice(idx, idx + 350);
    expect(slice).toMatch(/Locating crop/);
  });

  it('switching tabs manually also clears awaitingDeepLink, not just pendingCropId', () => {
    const idx = src.indexOf('tab = t.id;\n            pendingCropId = null;');
    expect(idx).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 120);
    expect(slice).toMatch(/awaitingDeepLink = false;/);
  });
});
