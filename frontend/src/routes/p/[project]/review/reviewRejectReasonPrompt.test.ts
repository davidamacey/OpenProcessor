/**
 * Rejecting a region/slot asks for a reason in an in-app modal, not a native
 * `window.prompt()`. A native dialog is OS-level chrome, outside the page's
 * DOM/CDP-rendered surface, so it never appears in a page screenshot and an
 * automated driver that doesn't arm a native-dialog handler gets it silently
 * auto-dismissed. The modal (promptForRejectionReason/rejectReasonPromptOpen)
 * is real DOM, screenshot-visible and keyboard-driveable like every other
 * modal on this page. The reject also asks up front when the served status
 * wants a reason, and the undo-stack footer no longer says "confirmed" after
 * a reject or false positive.
 *
 * Static source scan (no mount harness for a page this size).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('the reject-reason prompt is real DOM, not a native window.prompt()', () => {
  it('there is no window.prompt() call anywhere in the page (only doc comments may mention the retired call)', () => {
    const codeOnly = src.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\/.*$/gm, '');
    expect(codeOnly).not.toMatch(/window\.prompt\(/);
  });

  it('promptForRejectionReason opens the in-app modal and returns a Promise the caller awaits', () => {
    const fn = src.match(/function promptForRejectionReason\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/rejectReasonPromptOpen = true;/);
    expect(fn).toMatch(/return new Promise/);
  });

  it('the modal template renders when rejectReasonPromptOpen is true, with submit/cancel wired to resolve the pending promise', () => {
    const idx = src.indexOf('{#if rejectReasonPromptOpen}');
    expect(idx).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 2200);
    expect(slice).toMatch(/onclick=\{submitRejectReasonPrompt\}/);
    expect(slice).toMatch(/onclick=\{cancelRejectReasonPrompt\}/);
    expect(slice).toMatch(/bind:value=\{rejectReasonPromptValue\}/);
  });

  it('cancelling resolves null (reject still proceeds with no reason, matching the old Cancel behavior) rather than throwing', () => {
    const fn = src.match(/function cancelRejectReasonPrompt\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/rejectReasonPromptResolve\?\.\(null\)/);
  });
});

describe('the slot undo-stack footer no longer says "confirmed" for a reject/FP', () => {
  it('renders "actioned", not a hardcoded "confirmed"', () => {
    const idx = src.indexOf('{slotUndoStack.length} actioned in this session');
    expect(idx).toBeGreaterThan(-1);
    expect(src).not.toMatch(/\{slotUndoStack\.length\} confirmed in this session/);
  });
});

describe('reject asks for a reason up front when the served status wants one', () => {
  const fn = src.match(/async function rejectSlot\(\)[\s\S]*?\n {2}\}/)?.[0];

  it('checks statusWantsRejectionReason against the served reject status before writing anything', () => {
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(
      /statusWantsRejectionReason\(activeSlot, rejectStatus, regionStatusesStore\.list\)/,
    );
  });

  it('prompts before the optimistic queue removal / the write, not after', () => {
    expect(fn).not.toBeUndefined();
    // DQ-m6 (2026-09-24 data-quality pass): window.prompt() replaced with
    // an in-app modal (promptForRejectionReason) — a native dialog
    // blocks the JS thread and renders outside the page's DOM/CDP
    // surface, which is why a screenshot/automation pass saw no prompt
    // at all. The modal's own coverage is the first describe
    // block in this file.
    const promptIdx = fn!.indexOf('await promptForRejectionReason()');
    const removeIdx = fn!.indexOf('_removeFromQueue(item)');
    const writeIdx = fn!.indexOf('await patchSlotMeta(');
    expect(promptIdx).toBeGreaterThan(-1);
    expect(promptIdx).toBeLessThan(removeIdx);
    expect(promptIdx).toBeLessThan(writeIdx);
    expect(fn).not.toMatch(/window\.prompt\(/);
  });

  it('sends the reject status and the reason in ONE region write, so one Z undoes it', () => {
    expect(fn).toMatch(
      /await patchSlotMeta\(activeSlot, item\.id, \{\s*status: rejectStatus,\s*\.\.\.\(reason \? \{ rejectionReason: reason \} : \{\}\),\s*\}\);/,
    );
    // Exactly one region write is recorded for the undo stack.
    expect(fn?.match(/recordRegionWrites\(/g)?.length).toBe(1);
    expect(fn).not.toMatch(
      /patchSlotMeta\(activeSlot, item\.id, \{ rejectionReason: reason \}\)/,
    );
  });
});
