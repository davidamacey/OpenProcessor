/**
 * Regression tests for the FRONTEND findings fixed by the 2026-09-24
 * interactive-pass follow-up (docs/design/interactive-pass-2026-09-24.md
 * §6 FRONTEND): M1 (Dismissed panel 400), M2/M12 (Details collapses the
 * crop image / plate unreadable at 1280×720), and B2's frontend half
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
    expect(fn).toBeDefined();
    expect(fn).toMatch(/sort:\s*'updated_at:desc'/);
    expect(fn).not.toMatch(/sort:\s*'recent'/);
  });

  it('a failed load sets dismissedError instead of silently emptying the list', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/dismissedError = \(e as Error\)\.message;/);
    expect(fn).toMatch(/dismissedItems = \[\];/);
  });

  it('the panel renders an error state before falling back to the empty state', () => {
    const markup = src.match(/\{#if dismissedLoading\}[\s\S]*?\{\/if\}/)?.[0];
    expect(markup).toBeDefined();
    expect(markup).toMatch(/\{:else if dismissedError\}/);
    // The error branch must come before the empty-state branch so a
    // failed request never renders "No dismissed crops."
    expect(markup!.indexOf('dismissedError')).toBeLessThan(
      markup!.indexOf('No dismissed crops.'),
    );
  });
});

describe('M2/M12: the crop/plate image never collapses to 0px when Details opens', () => {
  it('the image wrapper is shrink-0 with a floor min-height, not flex-1', () => {
    expect(src).toMatch(
      /flex min-h-\[\d+px\] shrink-0 items-center justify-center bg-zinc-950/,
    );
  });

  it('the metadata/Details region below the image scrolls independently', () => {
    expect(src).toMatch(/class="mt-3 min-h-0 flex-1 overflow-y-auto pr-1"/);
  });
});

describe("B2 (frontend half): confirming an unchanged plate box doesn't rewrite provenance", () => {
  const confirmFn = src.match(/async function confirmSlot\([\s\S]*?\n {2}\}/)?.[0];

  it('confirmSlot is defined', () => {
    expect(confirmFn).toBeDefined();
  });

  it('compares the edited box against the seeded (served) box', () => {
    expect(confirmFn).toMatch(
      /const boxUnchanged = _boxesEqual\(editedSlotBox, seededSlotBox\)/,
    );
  });

  it('an unchanged box with a served confirm_status sends PATCH region_meta via patchSlotMeta, not PUT region', () => {
    expect(confirmFn).toMatch(
      /boxUnchanged &&\s*confirmStatus &&\s*activeSlot\.capabilities\.lifecycle\?\.statusField/,
    );
    expect(confirmFn).toMatch(
      /await patchSlotMeta\(activeSlot, item\.id, \{ status: confirmStatus \}\);/,
    );
  });

  it('a changed box still falls through to the PUT region (frame: parent) write', () => {
    expect(confirmFn).toMatch(
      /await setSlotBox\(activeSlot, item\.id, tuple, 'parent'\);/,
    );
  });

  it('confirmStatus is read from the server-served regionStatusesStore, not a client constant', () => {
    expect(confirmFn).toMatch(
      /const confirmStatus = regionStatusesStore\.confirmStatus;/,
    );
  });
});

describe('_boxesEqual value equality (used by the B2 fix)', () => {
  it('is defined as a field-by-field comparison, not reference equality alone', () => {
    const fn = src.match(/function _boxesEqual\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(
      /a\.cx === b\.cx && a\.cy === b\.cy && a\.w === b\.w && a\.h === b\.h/,
    );
  });
});

describe('seededSlotBox is captured every time the box is (re)seeded from the server', () => {
  it('_seedSlotFromCurrent sets seededSlotBox alongside editedSlotBox', () => {
    const fn = src.match(/function _seedSlotFromCurrent\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/seededSlotBox = editedSlotBox;/);
  });
});
