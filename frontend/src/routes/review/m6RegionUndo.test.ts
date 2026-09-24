/**
 * Regression tests for M6's frontend half (docs/design/interactive-pass-2026-09-24.md
 * §6 FRONTEND item 5): once the backend ships region undo (07cc061,
 * `POST {API_PREFIX}/crops/{id}/region/undo`), the plates/slot review tab
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
  expect(m, `function ${name} not found`).toBeDefined();
  return m![0];
}

describe('M6: confirmSlot/rejectSlot/markFalsePositive/saveBboxAndExit record a region undo entry', () => {
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

  it('saveBboxAndExit calls undoStore.recordRegionWrites on a successful write', () => {
    expect(fn('saveBboxAndExit')).toMatch(/undoStore\.recordRegionWrites\(\[id\]\)/);
  });

  it('none of the four record on the failure path (only inside the try, before catch)', () => {
    for (const name of [
      'confirmSlot',
      'rejectSlot',
      'markFalsePositive',
      'saveBboxAndExit',
    ]) {
      const body = fn(name);
      const catchIdx = body.indexOf('} catch');
      const recordIdx = body.indexOf('undoStore.recordRegionWrites');
      expect(recordIdx, `${name} never calls recordRegionWrites`).toBeGreaterThan(-1);
      expect(recordIdx, `${name} records after its catch block`).toBeLessThan(catchIdx);
    }
  });
});

describe('M6: Z is wired globally, so it reaches region-undo entries pushed from the slot tab too', () => {
  it("registers 'z' unconditionally (outside the activeSlot-only branch), not a second slot-only binding", () => {
    expect(src).toMatch(/reg\('z', undoLast, 'Undo last'\);/);
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
    for (const name of ['confirmSlot', 'rejectSlot', 'markFalsePositive']) {
      const body = fn(name);
      expect(body).toMatch(/Press Z to undo, step back with ←\./);
    }
  });
});
