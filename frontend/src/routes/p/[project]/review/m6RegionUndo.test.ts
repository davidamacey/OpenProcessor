/**
 * Regression tests for M6's frontend half (docs/design/interactive-pass-2026-09-24.md
 * §6 FRONTEND item 5): once the backend ships region undo (b654da5,
 * `POST {API_PREFIX}/crops/{id}/region/undo`), the regions/slot review tab
 * must record an undo entry on every confirm/reject/FP/box-edit write so Z
 * reverses it server-side — and the misleading "← to go back" copy (which
 * implied Back itself undid the write) is renamed to "step back", since
 * Back only re-queues the crop locally.
 *
 * Same static source-scan convention as interactivePassFixes.test.ts — no
 * mount harness for a page this size; the behavior under test is "which
 * write path calls undoStore.recordRegionWrites", not renderable markup.
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

describe('M6: confirmSlot/rejectSlot/markFalsePositive record a region undo entry', () => {
  it('confirmSlot calls undoStore.recordRegionWrites on a successful write, before the success toast', () => {
    const body = fn('confirmSlot');
    const recordIdx = body.indexOf('undoStore.recordRegionWrites([item.id])');
    const toastIdx = body.indexOf('toastStore.success(');
    expect(recordIdx).toBeGreaterThan(-1);
    expect(recordIdx).toBeLessThan(toastIdx);
  });

  it('rejectSlot calls undoStore.recordRegionWrites on a successful write', () => {
    expect(fn('rejectSlot')).toMatch(/undoStore\.recordRegionWrites\(\[item\.id\]\)/);
  });

  it('markFalsePositive calls undoStore.recordRegionWrites on a successful write', () => {
    expect(fn('markFalsePositive')).toMatch(
      /undoStore\.recordRegionWrites\(\[item\.id\]\)/,
    );
  });

  // The multi-box writes (confirm / per-box accept-reject / box edits) record
  // inside multiBoxRegionController.svelte.ts — multiBoxRegionController.test.ts
  // asserts every successful write records and a failed one does not.

  it('none of the three record on the failure path (only inside the try, before catch)', () => {
    for (const name of ['confirmSlot', 'rejectSlot', 'markFalsePositive']) {
      const body = fn(name);
      const catchIdx = body.indexOf('} catch');
      const recordIdx = body.indexOf('undoStore.recordRegionWrites');
      expect(recordIdx, `${name} never calls recordRegionWrites`).toBeGreaterThan(-1);
      expect(recordIdx, `${name} records after its catch block`).toBeLessThan(catchIdx);
      // ...and only after the write itself resolved: an entry recorded before
      // the awaited write would survive a failed write and Z would replay it.
      const writeIdx = body.indexOf('await patchSlotMeta(');
      expect(writeIdx, `${name} no longer awaits patchSlotMeta`).toBeGreaterThan(-1);
      expect(recordIdx, `${name} records before its write resolves`).toBeGreaterThan(
        writeIdx,
      );
    }
  });
});

describe('M6: Z is wired globally, so it reaches region-undo entries pushed from the slot tab too', () => {
  it("registers 'z' unconditionally (outside the activeSlot-only branch), not a second slot-only binding", () => {
    // K1: registered by keymap action id (`review.undo`, default z).
    expect(src).toMatch(/reg\('review\.undo', undoLast\);/);
  });
});

describe('M6: "← back"/"to go back" copy renamed to "step back" — Back only re-queues locally, Z undoes the write', () => {
  it('no remaining "← back" / "to go back" strings that implied Back itself undoes the server write', () => {
    expect(src).not.toMatch(/← to go back/);
    expect(src).not.toMatch(/Nothing to go back to/);
  });

  it('the step-back toast and button both say "step back", not just "back"', () => {
    expect(src).toMatch(/Nothing to step back to\./);
    expect(src).toMatch(/Step back/);
  });

  it('the three write-success toasts point at Z for undo and at step-back (←) separately', () => {
    // K1: the keys are the keymap's glyphs, not literals — `undoHint()`
    // prints review.undo (Z) and review.region.back (←) separately.
    expect(src).toMatch(
      /const undoHint = \(\) =>\s*`Press \$\{kg\('review\.undo'\)\} to undo, step back with \$\{kg\('review\.region\.back'\)\}\.`;/,
    );
    for (const name of ['confirmSlot', 'rejectSlot', 'markFalsePositive']) {
      const body = fn(name);
      expect(body).toMatch(/\$\{undoHint\(\)\}`/);
    }
  });
});
