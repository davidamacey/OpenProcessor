/**
 * W2 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — /review's
 * slot-panel writes render the server's own returned item instead of a
 * hand-computed post-write state, box edits send frame: 'parent' with no
 * client-side projection, and status vocabulary/actions are driven by
 * the served `GET {API_PREFIX}/regions/statuses` vocabulary with a
 * profile fallback.
 *
 * This repo has no component-mount harness (see slotReviewCharacterization
 * .test.ts's doc comment); pinned via the same static source-scan
 * convention.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('W2: shape-warning badge and client shape-gate are gone', () => {
  it('never imports shapeGate or renders a shapeWarning badge', () => {
    expect(src).not.toMatch(/shapeGate/);
    expect(src).not.toMatch(/shapeWarning/);
    expect(src).not.toMatch(/describeEnvelope/);
  });
});

describe('W2: box edits send frame: "parent", no client-side projectFromParent on write', () => {
  it('never imports projectFromParent (deleted from the write path)', () => {
    expect(src).not.toMatch(/projectFromParent/);
  });

  it("saveBboxAndExit and confirmSlot call setSlotBox with frame 'parent'", () => {
    const saveFn = src.match(/async function saveBboxAndExit\([\s\S]*?\n {2}\}/)?.[0];
    const confirmFn = src.match(/async function confirmSlot\([\s\S]*?\n {2}\}/)?.[0];
    expect(saveFn).toMatch(/setSlotBox\(activeSlot, id, tuple, 'parent'\)/);
    expect(confirmFn).toMatch(/setSlotBox\(activeSlot, item\.id, tuple, 'parent'\)/);
  });
});

describe('W2: slot writes render the server-returned item, not a computed patch', () => {
  it('saveSlotMeta spreads res.item onto the queue item', () => {
    const fn = src.match(
      /async function saveSlotMeta\([\s\S]*?\n {2}async function commitSlotText/,
    )?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/\.\.\.res\.item/);
    expect(fn).not.toMatch(/applyOptimistic/);
  });

  it('saveBboxAndExit and the clears-box branch of commitSlotStatus spread the returned item', () => {
    const saveFn = src.match(/async function saveBboxAndExit\([\s\S]*?\n {2}\}/)?.[0];
    expect(saveFn).toMatch(
      /queue\.items\[idx\] = \{ \.\.\.queue\.items\[idx\], \.\.\.item \} as ReviewItem/,
    );
    const statusFn = src.match(/async function commitSlotStatus\([\s\S]*?\n {2}\}/)?.[0];
    expect(statusFn).toMatch(
      /queue\.items\[idx\] = \{ \.\.\.queue\.items\[idx\], \.\.\.item \} as ReviewItem/,
    );
  });
});

describe('W2: status vocabulary/actions are server-driven with a profile fallback', () => {
  it('humanWritableStates/statusClearsBox/statusWantsRejectionReason are called with regionStatusesStore.list', () => {
    expect(src).toMatch(/humanWritableStates\(activeSlot, regionStatusesStore\.list\)/);
    expect(src).toMatch(
      /statusClearsBox\(activeSlot, editedSlotStatus, regionStatusesStore\.list\)/,
    );
    expect(src).toMatch(
      /statusWantsRejectionReason\(activeSlot, editedSlotStatus, regionStatusesStore\.list\)/,
    );
  });

  it('markFalsePositive prefers the served false_positive_status', () => {
    const fn = src.match(/async function markFalsePositive\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/regionStatusesStore\.falsePositiveStatus \?\?/);
  });
});
