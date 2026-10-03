/**
 * Regression tests for the FRONTEND findings fixed by the 2026-09-24
 * interactive-pass follow-up (docs/design/interactive-pass-2026-09-24.md
 * §6 FRONTEND): M1 (Dismissed panel 400), M2/M12 (Details collapses the
 * crop image / region unreadable at 1280×720), and B2's frontend half
 * (confirm-with-unchanged-box must not rewrite detector provenance).
 *
 * Same static source-scan convention as logicMovesW2.test.ts /
 * slotReviewCharacterization.test.ts — this repo has no
 * `@testing-library/svelte` mount harness for a page this size, and the
 * behavior under test (layout classes, which write path a branch takes)
 * is either non-renderable under jsdom+mount or is exactly the kind of
 * "did the source really change" pin a mount test can't beat.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('M1: Dismissed panel sends a sort the backend accepts and reports failure', () => {
  it('toggleDismissedPanel requests updated_at:desc, not the 400-ing "recent"', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/sort:\s*'updated_at:desc'/);
    expect(fn).not.toMatch(/sort:\s*'recent'/);
  });

  it('a failed load sets dismissedError instead of silently emptying the list', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/dismissedError = apiErrorText\(e\);/);
    expect(fn).toMatch(/dismissedItems = \[\];/);
  });

  it('the panel renders an error state before falling back to the empty state', () => {
    const markup = src.match(/\{#if dismissedLoading\}[\s\S]*?\{\/if\}/)?.[0];
    expect(markup).not.toBeUndefined();
    expect(markup).toMatch(/\{:else if dismissedError\}/);
    // The error branch must come before the empty-state branch so a
    // failed request never renders "No dismissed crops."
    expect(markup!.indexOf('dismissedError')).toBeLessThan(
      markup!.indexOf('No dismissed crops.'),
    );
  });
});

describe('M2/M12: the crop/region image never collapses to 0px when Details opens', () => {
  it('the image wrapper is shrink-0 with a floor min-height, not flex-1', () => {
    // DQ-M5 (2026-09-24 data-quality pass) added a `max-h-[46%]` ceiling
    // alongside the floor — the wrapper is still shrink-0 with min-height,
    // just no longer unbounded above.
    // F8 D4 (2026-09-25): a fixed floor height below lg (the body scrolls
    // there), the floor + ceiling from lg up; overflow-hidden so a region
    // canvas can't paint over the first metadata row.
    expect(src).toMatch(
      /flex h-\[\d+px\] shrink-0 items-center justify-center overflow-hidden bg-zinc-950 lg:h-auto lg:max-h-\[\d+%\] lg:min-h-\[\d+px\]/,
    );
  });

  it('the metadata/Details region below the image scrolls independently', () => {
    // From lg up it scrolls in its own region; below lg the whole review
    // body scrolls instead (F8 D4: the inner pane was ~79px at 800px).
    expect(src).toMatch(/class="mt-3 pr-1 lg:min-h-0 lg:flex-1 lg:overflow-y-auto"/);
  });
});

describe("B2 (frontend half): confirming an untouched box doesn't rewrite provenance", () => {
  // The multi-box confirm sends an untouched stored box as `{box_id}` alone
  // (multiBox.test.ts pins buildRegionsPutBoxes), so the server keeps its
  // geometry, detector and score verbatim. The only confirm left in the
  // page is the box-less slot's, a status-only write.
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

describe('m1 (2026-09-24 interactive pass): a name-only proposal is styled as a hint, not a confirmable proposal', () => {
  it('the Proposed row branches on proposed_class_id == null before using the yellow confirmable style', () => {
    expect(src).toMatch(
      /\{#if current\.proposed_class_name && current\.proposed_class_id == null\}/,
    );
  });

  it('the name-only branch uses a neutral color, not text-yellow-200', () => {
    const block = src.match(
      /\{#if current\.proposed_class_name && current\.proposed_class_id == null\}[\s\S]*?\{:else\}/,
    )?.[0];
    expect(block).not.toBeUndefined();
    expect(block).not.toMatch(/text-yellow-200/);
    expect(block).toMatch(/hint only/);
  });
});

describe('m5 (2026-09-24 interactive pass): reject asks for a reason up front when the served status wants one', () => {
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
    // at all. See dqM6RejectReasonPrompt.test.ts for the modal's own
    // coverage.
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

describe('p9 (2026-09-24 interactive pass): the /review crop thumbnail upscales to fill its panel, at any viewport', () => {
  it('the plain-<img> crop branch uses h-full w-full, not max-h-full max-w-full (which never upscales)', () => {
    const block = src.match(/\{:else\}\s*\n\s*<!-- p9[\s\S]*?<img[\s\S]*?\/>/)?.[0];
    expect(block).not.toBeUndefined();
    expect(block).toMatch(/class="h-full w-full object-contain"/);
    expect(block).not.toMatch(/class="max-h-full max-w-full object-contain"/);
  });
});
