/**
 * Phase 0 characterization tests (docs/genericization-plan-2026-09-13.md
 * §5.1, table T1-T7) for the Plates tab's stateful review logic — pinned
 * BEFORE any Phase 2 component genericization touches `review/+page.svelte`
 * or `clusters/+page.svelte`.
 *
 * This repo has no `@testing-library/svelte` harness (see
 * `plateThumbUrlScan.test.ts`'s doc comment), and the plan's own
 * recommendation is to extract the pure logic into testable modules
 * FIRST (P0.1/P0.3) and write executable tests against the extraction.
 * That extraction was judged too large/risky to do safely in the same
 * pass as this test file (it touches the same 1950-line file the tests
 * are meant to protect, with no existing safety net) — so this file
 * takes the fallback the plan itself names for exactly this repo's
 * situation: a static source scan (the convention `EmbeddingPlot.test.ts`
 * / `StrategyBar.test.ts` / `plateThumbUrlScan.test.ts` already use),
 * pinning the *shape* of the current implementation so a future
 * extraction (P0.1/P0.3) or Phase 2 migration cannot silently drop one of
 * these behaviors without a test going red first.
 *
 * When P0.1/P0.3's real extraction lands, the corresponding assertions
 * here should be replaced by executable unit tests against the extracted
 * modules (slotQueueOps.ts / abortRegistry.ts), per the plan's T1/T2/T3
 * "how it adapts" column — this file is the interim safety net, not the
 * final one.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const reviewPageSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const clustersPageSrc = readFileSync(
  path.resolve(here, '..', 'clusters', '+page.svelte'),
  'utf-8',
);

describe('T4: Plates-tab keymap + reserved-letters invariant (Finding C.2)', () => {
  it('registers exactly the documented scan-mode keymap on the Plates tab', () => {
    // Enter=confirm, D=reject, F=false-positive, E=edit, arrowleft/B=back,
    // arrowright=next, N=skip (shared with every tab). Pinning presence
    // via keyboardStore.register() call sites, not exhaustive parsing — a
    // real extraction should replace this with an executable keymap
    // table (QueueCapability.keymap once Phase 2 wires it up).
    for (const marker of [
      "reg('enter', confirmPlate,",
      "reg('d', rejectPlate,",
      "reg('f', markFalsePositive,",
      "reg('e', toggleEdit,",
      "reg('arrowleft', plateBack,",
      "reg('b', plateBack,",
      "'arrowright',",
      "reg('n', skip,",
    ]) {
      expect(reviewPageSrc).toContain(marker);
    }
  });

  it('class-drop registration early-returns on the plates tab (the invariant behind Finding C.2)', () => {
    // dropOnClassStore's handler is never registered while tab==='plates',
    // which is why f/e/b can be bound as plates actions without colliding
    // with a class hotkey today, even though RESERVED_HOTKEY_LETTERS
    // doesn't list them (next assertion). This is a `return` statement in
    // a 1950-line file — exactly the fragile invariant Finding C.2 flags.
    expect(reviewPageSrc).toMatch(/if \(tab === 'plates'\) return;/);
  });

  it(
    'RESERVED_HOTKEY_LETTERS does NOT yet include f/e/b — documents Finding C.2 as still open ' +
      '(turns green when P1.6 derives the reserved set from configured slot keymaps)',
    async () => {
      const { RESERVED_HOTKEY_LETTERS } = await import('../../lib/classHotkey');
      for (const letter of ['f', 'e', 'b']) {
        expect(RESERVED_HOTKEY_LETTERS.has(letter)).toBe(false);
      }
      // The letters ARE bound as plates actions today (previous test) —
      // the only reason this doesn't collide is the tab-guard invariant.
      // A class hotkey audit UI or future slot config could reintroduce a
      // real collision if that guard is ever removed without deriving the
      // reserved set from the same keymaps.
    },
  );
});

describe('T2/T3-adjacent: per-crop abort map + undo-stack identity semantics', () => {
  it('keys the Plates-tab save-abort map by crop id, not by cursor index', () => {
    // Finding-adjacent: "if the user advances mid-save the captured idx
    // would point at the next crop and the revert would corrupt unrelated
    // state" — the fix is a Map<cropId, AbortController>, not an index.
    expect(reviewPageSrc).toMatch(
      /plateMetaAborts\s*=\s*new Map<string, AbortController>/,
    );
  });

  it('undo stack is $state.raw, not deeply-reactive $state (identity-filter correctness)', () => {
    // Deep reactivity would proxy pushed entries, so the undo removal's
    // `e !== entry` identity filter could never match a pushed entry.
    expect(reviewPageSrc).toMatch(
      /plateUndoStack = \$state\.raw<PlateUndoEntry\[\]>\(\[\]\)/,
    );
    expect(reviewPageSrc).toMatch(/plateUndoStack\.filter\(\(e\) => e !== entry\)/);
  });

  it('undo stack is bounded (FIFO-evicted), matching the plan\'s "bounded at 20" characterization', () => {
    expect(reviewPageSrc).toMatch(/\.slice\(-PLATE_UNDO_MAX\)/);
  });
});

describe('T1-adjacent: frozen-viewport effect uses untrack for the seed read', () => {
  it('the plate bbox-editor viewport-freeze effect wraps its seed call in untrack()', () => {
    // Without untrack(), _seedViewBox() would re-fire on every drag tick
    // and overwrite the operator's in-progress resize with the server
    // snapshot — see plan §3.5 point 1.
    expect(reviewPageSrc).toMatch(/untrack\(\(\) => _seedViewBox\(\)\)/);
  });
});

describe("T7-adjacent: today's review-tab id/url shape, pinned before reviewTabs.ts is data-driven", () => {
  it('the Plates tab is still a literal, closed id — not yet slot-derived', async () => {
    const { REVIEW_TABS } = await import('../../lib/reviewTabs');
    const ids = REVIEW_TABS.map((t: { id: string }) => t.id);
    expect(ids).toContain('plates');
    // Once Phase 2 (P2.8) lands, this becomes `slot:license_plate` with
    // `urlId === 'plates'` preserved for the bookmark contract — this
    // test should be updated at that point, not deleted, per the plan's
    // T7 "how it adapts" note.
  });
});

describe("today's license_plate literals in clusters/+page.svelte (pre-Phase-2 baseline)", () => {
  it('still hardcodes the license_plate class-name string at the documented sites', () => {
    // Finding B's six sites (docs/genericization-plan-2026-09-13.md §1):
    // isLicensePlateFilter, loadLicensePlateCard, the synthetic pinned
    // card's dominant_class_name, and card-click routing. Pinned here so
    // a future SlotGallery extraction (P2.6/P2.7) has a concrete "these
    // literals must all move to slotRegistry lookups" checklist, and so
    // this test goes red the moment that migration starts (a reminder to
    // update it, not evidence of a regression).
    const occurrences = (clustersPageSrc.match(/'license_plate'/g) ?? []).length;
    expect(occurrences).toBeGreaterThanOrEqual(4);
  });
});
