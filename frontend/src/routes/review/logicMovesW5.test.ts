/**
 * W5 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — review
 * queue: the locate-based deep link, "Accept model's class" via
 * probe_pred_class_id, and the removal of the client proposed_class_*
 * fill-ins.
 *
 * This repo has no component-mount harness (see
 * slotReviewCharacterization.test.ts's doc comment); pinned via the same
 * static source-scan convention.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('W5: /review?crop_id= deep link calls locateInReviewQueue, not a paging scan', () => {
  it('jumpToPendingCrop calls locateInReviewQueue with the effective tab, page size and current filters', () => {
    const fn = src.match(/async function jumpToPendingCrop\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(
      /await locateInReviewQueue\(\s*endpointForTab\(effectiveTab\),\s*cropId,\s*pageSize,\s*_filter\(\),\s*\)/,
    );
    // DQ-M7 (2026-09-24 data-quality pass, see dqM7Locate.test.ts):
    // fetches only the located page via queue.loadPage(loc.page) — the
    // original while-loop here paged 1..loc.page one request per page
    // (103 requests / 5.7s at rank 3000/page 101, with item #1's
    // keybindings live the whole time), which was a regression this test
    // used to pin as the intended behavior. It's the bug DQ-M7 fixes.
    expect(fn).toMatch(/await queue\.loadPage\(loc\.page\)/);
    expect(fn).not.toMatch(/while \(queue\.loadedPages < loc\.page/);
  });

  it('never references the deleted DEEP_LINK_MAX_ITEMS paging-scan constant', () => {
    expect(src).not.toMatch(/DEEP_LINK_MAX_ITEMS/);
  });
});

describe('W5: "Accept model\'s class" assigns probe_pred_class_id directly', () => {
  it('acceptModelClass assigns current.probe_pred_class_id with no name lookup', () => {
    const fn = src.match(/async function acceptModelClass\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/await assign\(current\.probe_pred_class_id\)/);
  });

  it('the accept button only renders when probe_pred_class_id differs from the current class', () => {
    // The function doc comment above also says "Accept model's class" —
    // skip past it to the actual <button> label in the template.
    const idx = src.indexOf("Accept model's class", src.indexOf('</script>'));
    expect(idx).toBeGreaterThan(-1);
    const preceding = src.slice(0, idx);
    const lastIf = [...preceding.matchAll(/\{#if [^}]*\}/g)].pop();
    expect(lastIf?.[0]).toMatch(
      /current\.probe_pred_class_id != null && current\.probe_pred_class_id !== current\.class_id/,
    );
  });
});

describe('W5: no client-side proposed_class_* fill-ins remain', () => {
  it('never sets a literal proposed_class_id/_name (diverse hydration, undo-restore, search results all read the served value)', () => {
    expect(src).not.toMatch(/proposed_class_id:\s*null/);
    expect(src).not.toMatch(/proposed_class_name:\s*null/);
    expect(src).not.toMatch(/proposed_class_id:\s*crop\.class_id/);
  });
});
