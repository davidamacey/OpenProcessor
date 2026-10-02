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

describe('W2: box edits never project client-side, and the removed single-box write is gone', () => {
  it('never imports projectFromParent or setSlotBox (both deleted)', () => {
    expect(src).not.toMatch(/projectFromParent/);
    expect(src).not.toMatch(/setSlotBox/);
    expect(src).not.toMatch(/saveBboxAndExit/);
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

  it('commitSlotStatus goes through saveSlotMeta, which renders the returned item (the server clears the box list itself)', () => {
    const statusFn = src.match(/async function commitSlotStatus\([\s\S]*?\n {2}\}/)?.[0];
    expect(statusFn).toMatch(/await saveSlotMeta\(\{ status: editedSlotStatus \}\)/);
    expect(statusFn).not.toMatch(/setSlotBox|editedSlotBox/);
  });
});

describe('W2: status vocabulary/actions are server-driven with a profile fallback', () => {
  it('humanWritableStates/statusWantsRejectionReason are called with regionStatusesStore.list', () => {
    expect(src).toMatch(/humanWritableStates\(activeSlot, regionStatusesStore\.list\)/);
    expect(src).toMatch(
      /statusWantsRejectionReason\(activeSlot, editedSlotStatus, regionStatusesStore\.list\)/,
    );
  });

  it('markFalsePositive prefers the served false_positive_status', () => {
    const fn = src.match(/async function markFalsePositive\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/regionStatusesStore\.falsePositiveStatus \?\?/);
  });
});
