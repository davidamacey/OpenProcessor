/**
 * /review's slot-panel writes render the server's own returned item instead
 * of a hand-computed post-write state, box edits send frame: 'parent' with no
 * client-side projection, and status vocabulary/actions are driven by the
 * served `GET {API_PREFIX}/regions/statuses` vocabulary with a profile
 * fallback. Static source scan (no component-mount harness; see
 * slotReviewCharacterization.test.ts's doc comment).
 */
import { describe, expect, it } from 'vitest';
import { reviewPageSource, reviewSource } from '$lib/testing/reviewPageSources';

const pageSrc = reviewPageSource();
const editSrc = reviewSource('lib/review/reviewSlotEditController.svelte.ts');
const panelSrc = reviewSource('lib/components/review/ReviewSlotPanel.svelte');
const actionsSrc = reviewSource('lib/review/reviewSlotActions.svelte.ts');

describe('shape-warning badge and client shape-gate are gone', () => {
  it('never imports shapeGate or renders a shapeWarning badge', () => {
    expect(pageSrc).not.toMatch(/shapeGate/);
    expect(pageSrc).not.toMatch(/shapeWarning/);
    expect(pageSrc).not.toMatch(/describeEnvelope/);
  });
});

describe('box edits never project client-side, and the removed single-box write is gone', () => {
  it('never imports projectFromParent or setSlotBox (both deleted)', () => {
    expect(pageSrc).not.toMatch(/projectFromParent/);
    expect(pageSrc).not.toMatch(/setSlotBox/);
    expect(pageSrc).not.toMatch(/saveBboxAndExit/);
  });
});

describe('slot writes render the server-returned item, not a computed patch', () => {
  it('saveSlotMeta spreads res.item onto the queue item', () => {
    const fn = editSrc.match(
      /async function saveSlotMeta\([\s\S]*?\n {2}async function commitSlotText/,
    )?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/\.\.\.res\.item/);
    expect(fn).not.toMatch(/applyOptimistic/);
  });

  it('commitSlotStatus goes through saveSlotMeta, which renders the returned item (the server clears the box list itself)', () => {
    const statusFn = editSrc.match(
      /async function commitSlotStatus\([\s\S]*?\n {2}\}/,
    )?.[0];
    expect(statusFn).toMatch(/await saveSlotMeta\(\{ status: editedSlotStatus \}\)/);
    expect(statusFn).not.toMatch(/setSlotBox|editedSlotBox/);
  });
});

describe('status vocabulary/actions are server-driven with a profile fallback', () => {
  it('humanWritableStates/statusWantsRejectionReason are called with regionStatusesStore.list', () => {
    expect(editSrc).toMatch(
      /humanWritableStates\(activeSlot, regionStatusesStore\.list\)/,
    );
    expect(panelSrc).toMatch(
      /statusWantsRejectionReason\(activeSlot, editedSlotStatus, regionStatusesStore\.list\)/,
    );
  });

  it('markFalsePositive prefers the served false_positive_status', () => {
    const fn = actionsSrc.match(
      /async function markFalsePositive\([\s\S]*?\n {2}\}/,
    )?.[0];
    expect(fn).toMatch(/regionStatusesStore\.falsePositiveStatus \?\?/);
  });
});
