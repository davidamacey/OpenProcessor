/**
 * Phase 0 characterization tests (docs/genericization-plan-2026-09-13.md
 * §5.1, table T1-T7) for the Plates tab's stateful review logic — pinned
 * BEFORE any Phase 2 component genericization touches `review/+page.svelte`
 * or `clusters/+page.svelte`.
 *
 * This repo has no `@testing-library/svelte` harness (see
 * `plateThumbUrlScan.test.ts`'s doc comment). The plan's P0.1/P0.3 call
 * for extracting the pure undo-stack/abort-map logic into testable
 * modules FIRST — that extraction has landed
 * (`src/lib/review/slotQueueOps.ts`, `src/lib/review/abortRegistry.ts`,
 * each with its own executable unit-test suite), and `review/+page.svelte`
 * now delegates to them. The remaining pieces here (the keymap, the
 * class-drop tab guard, the frozen-viewport `untrack()` seed, and today's
 * `REVIEW_TABS`/`license_plate` baseline) are still genuinely inline
 * Svelte state/effects with no extraction seam, so they stay pinned via
 * the static source-scan convention this repo already uses
 * (`EmbeddingPlot.test.ts` / `StrategyBar.test.ts` /
 * `plateThumbUrlScan.test.ts`) until Phase 2's `reviewTabs.ts`
 * data-driving and keymap parameterization give them one.
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

describe('T1/T2 (real extraction, P0.1/P0.3): per-crop abort map + undo-stack delegate to extracted modules', () => {
  // These behaviors are no longer inline closures — they were extracted
  // to src/lib/review/abortRegistry.ts and slotQueueOps.ts (P0.1/P0.3),
  // each with its own executable unit test suite (abortRegistry.test.ts,
  // slotQueueOps.test.ts) that supersedes the source-scan style below for
  // the actual behavior. What's pinned here is that the page still wires
  // to them the way the plan requires (keyed by crop id, not cursor;
  // $state.raw for identity-correct removal).
  it('keys the Plates-tab save-abort registry by crop id, not by cursor index', () => {
    expect(reviewPageSrc).toMatch(/plateMetaAborts = new AbortRegistry\(\)/);
    expect(reviewPageSrc).toMatch(/plateMetaAborts\.start\(id\)/);
    expect(reviewPageSrc).toMatch(/plateMetaAborts\.finish\(id, ac\)/);
  });

  it('undo stack is $state.raw and delegates push/remove/pop to slotQueueOps', () => {
    // Deep reactivity would proxy pushed entries, so removeUndo's
    // identity-based filter could never match a pushed entry.
    expect(reviewPageSrc).toMatch(
      /plateUndoStack = \$state\.raw<PlateUndoEntry\[\]>\(\[\]\)/,
    );
    expect(reviewPageSrc).toMatch(/pushUndo\(plateUndoStack, entry, PLATE_UNDO_MAX\)/);
    expect(reviewPageSrc).toMatch(/removeUndo\(plateUndoStack, entry\)/);
    expect(reviewPageSrc).toMatch(/popUndo\(plateUndoStack\)/);
  });

  it('plateBack() re-insertion delegates to reinsertAt (clamped splice)', () => {
    expect(reviewPageSrc).toMatch(/reinsertAt\(queue\.items, last\.insertAt, fresh\)/);
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
