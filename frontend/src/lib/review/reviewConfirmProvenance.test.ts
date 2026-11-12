/**
 * Confirming an untouched box must not rewrite detector provenance. Static
 * source scan (no component-mount harness; see
 * slotReviewCharacterization.test.ts's doc comment).
 */
import { describe, expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';

const src = reviewSource('lib/review/reviewSlotActions.svelte.ts');

describe("confirming an untouched box doesn't rewrite provenance", () => {
  // The multi-box confirm sends an untouched stored box as `{box_id}` alone
  // (multiBox.test.ts pins buildRegionsPutBoxes), so the server keeps its
  // geometry, detector and score verbatim. The only confirm left in the
  // slot actions is the box-less slot's, a status-only write.
  const confirmFn = src.match(/async function confirmSlot\([\s\S]*?\n {2}\}/)?.[0];

  it('confirmSlot (a slot with no box list) is a status-only PATCH region_meta via patchSlotMeta', () => {
    expect(confirmFn).not.toBeUndefined();
    expect(confirmFn).toMatch(
      /await patchSlotMeta\(activeSlot, item\.id, \{ status: confirmStatus \}\);/,
    );
    expect(confirmFn).not.toMatch(/setSlotBox/);
  });

  it('confirmStatus is read from the server-served regionStatusesStore, not a client constant', () => {
    expect(confirmFn).toMatch(/regionStatusesStore\.confirmStatus \?\?/);
  });

  it('the multi-box confirm routes through the controller, never a page-local write', () => {
    const fn = src.match(/async function confirmMultiBoxSlot\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/multiBox\.confirmAndSave\(item\.id\)/);
    expect(fn).not.toMatch(/putRegionBoxes|patchSlotMeta/);
  });
});
