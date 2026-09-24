/**
 * DQ-m6 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 7):
 * two findings against the plates/slot review tab.
 *
 *  - "D rejects with no reason prompt" — rejectSlot() DID call
 *    window.prompt() (added by phase-A's m5), but the audit's
 *    screenshot/automation pass saw nothing: a native `window.prompt()`
 *    dialog is OS-level chrome, outside the page's DOM/CDP-rendered
 *    surface, so it never appears in a page screenshot, and an automated
 *    driver that doesn't specifically arm a native-dialog handler gets it
 *    silently auto-dismissed. Replaced with an in-app modal
 *    (promptForRejectionReason/rejectReasonPromptOpen) — real DOM,
 *    screenshot-visible, keyboard-driveable the same way every other
 *    modal on this page is.
 *  - "the footer reads '1 confirmed in this session'" after a reject —
 *    the undo-stack counter's wording was hardcoded to "confirmed"
 *    regardless of which action (confirm/reject/FP) actually populated
 *    the stack.
 *
 * Same static source-scan convention as the other review/+page.svelte
 * regression tests — no mount harness for a page this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-m6: the reject-reason prompt is real DOM, not a native window.prompt()', () => {
  it('there is no window.prompt() call anywhere in the page (only doc comments may mention the retired call)', () => {
    const codeOnly = src.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\/.*$/gm, '');
    expect(codeOnly).not.toMatch(/window\.prompt\(/);
  });

  it('promptForRejectionReason opens the in-app modal and returns a Promise the caller awaits', () => {
    const fn = src.match(/function promptForRejectionReason\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
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
    expect(fn).toBeDefined();
    expect(fn).toMatch(/rejectReasonPromptResolve\?\.\(null\)/);
  });
});

describe('DQ-m6: the slot undo-stack footer no longer says "confirmed" for a reject/FP', () => {
  it('renders "actioned", not a hardcoded "confirmed"', () => {
    const idx = src.indexOf('{slotUndoStack.length} actioned in this session');
    expect(idx).toBeGreaterThan(-1);
    expect(src).not.toMatch(/\{slotUndoStack\.length\} confirmed in this session/);
  });
});
