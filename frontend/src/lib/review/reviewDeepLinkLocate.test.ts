/**
 * The focused item of a Review link (`focus=item:<id>`) is found with one locate request, not a
 * paging scan (the old scan paged 1..N one request per page: 103 requests and 5.7s to reach rank
 * 3000, with item #1's keys live the whole time, so a keypress could act on the wrong crop). The
 * source asks the server where the item sits, the loader fetches only the page that starts there,
 * and an `awaitingDeepLink` state gates the keys and the rendered item until the target lands.
 *
 * Static source scan (no mount harness for a view this size): which pager call fires, and which
 * state gates the keybinding effect.
 */
import { describe, expect, it } from 'vitest';
import { reviewPageSource, reviewSource } from '$lib/testing/reviewPageSources';

const loaderSrc = reviewSource('lib/review/reviewQueueLoader.svelte.ts');
const tabsSrc = reviewSource('lib/review/reviewTabsController.svelte.ts');
const keyboardSrc = reviewSource('lib/review/reviewKeyboard.svelte.ts');
const viewSrc = reviewSource('lib/workspace/views/ReviewView.svelte');

const jump =
  loaderSrc.match(/async function jumpToPendingCrop\([\s\S]*?\n {2}\}/)?.[0] ?? '';

describe('a Review link locates the focused item, not a paging scan', () => {
  it('jumpToPendingCrop asks the source where the item sits, once', () => {
    expect(jump).toMatch(/await source\.locate\(cropId\)/);
  });

  it('fetches only the page that starts at the located item', () => {
    expect(jump).toMatch(/await queue\.loadAt\(loc\.cursorBefore, loc\.index\)/);
    expect(jump).not.toMatch(/await queue\.loadMore\(\)/);
  });

  it('never references the deleted DEEP_LINK_MAX_ITEMS paging-scan constant', () => {
    expect(reviewPageSource()).not.toMatch(/DEEP_LINK_MAX_ITEMS/);
  });

  it('clears awaitingDeepLink on every exit path (finally)', () => {
    const finallyBlock = jump.match(/finally\s*\{[\s\S]*?\}\s*$/)?.[0];
    expect(finallyBlock).toMatch(/awaitingDeepLink = false;/);
  });
});

describe('awaitingDeepLink gates keybindings and rendering', () => {
  it('seeds true only when the URL focuses an item', () => {
    expect(tabsSrc).toMatch(
      /let awaitingDeepLink = \$state<boolean>\(pendingCropId != null\);/,
    );
  });

  it('the queue-action keybinding $effect returns before registering anything while awaiting', () => {
    const idx = keyboardSrc.indexOf('$effect(() => {\n    // DQ-M7');
    expect(
      idx,
      'keybinding effect with the DQ-M7 guard comment not found',
    ).toBeGreaterThan(-1);
    expect(keyboardSrc.slice(idx, idx + 400)).toMatch(/if \(awaitingDeepLink\) return;/);
  });

  it('the body renders a locating state instead of the (possibly page-1) current item', () => {
    const idx = viewSrc.indexOf('{:else if tabs.awaitingDeepLink}');
    expect(idx).toBeGreaterThan(-1);
    expect(viewSrc.slice(idx, idx + 500)).toMatch(/Locating crop/);
  });

  it('the focus is not written back to the URL while the item is still being located', () => {
    expect(viewSrc).toMatch(/if \(tabs\.awaitingDeepLink \|\| !current\) return;/);
  });
});
