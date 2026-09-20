/**
 * Phase 0 characterization tests (docs/genericization-plan-2026-09-13.md
 * §5.1, table T1-T7) for the Plates tab's stateful review logic — pinned
 * BEFORE any Phase 2 component genericization touches `review/+page.svelte`
 * or `clusters/+page.svelte`.
 *
 * This repo has no `@testing-library/svelte` harness (see
 * `plateThumbUrlScan.test.ts`'s doc comment). The plan's P0.1/P0.3 call
 * for extracting the pure logic into testable modules FIRST, and that
 * has now landed for every piece that had a real extraction seam:
 * `src/lib/review/slotQueueOps.ts` + `abortRegistry.ts` (undo stack +
 * per-crop abort map), `slotKeymap.ts` (the scan/edit-mode keymap
 * table, generalized off `licensePlateSlot` by P2.8c), `slotTabGuard.ts`
 * (the class-drop suppression predicate
 * behind Finding C.2), and `viewBox.ts` (the frozen-viewport
 * padding/squaring/clamping math) — each with its own executable
 * unit-test suite that supersedes the corresponding assertions below as
 * the real behavior pin. `review/+page.svelte` delegates to all of
 * them. What's left genuinely inline (the `untrack()` wrapping itself —
 * a Svelte-reactivity concern, not extractable math — and today's
 * `REVIEW_TABS`/`license_plate` baseline in `clusters/+page.svelte`)
 * stays pinned via the static source-scan convention this repo already
 * uses (`EmbeddingPlot.test.ts` / `StrategyBar.test.ts` /
 * `plateThumbUrlScan.test.ts`) until Phase 2's `reviewTabs.ts`
 * data-driving and the SlotGallery extraction give them one.
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
    //
    // UPDATE (P2.8c): the extraction is now slot-generic —
    // src/lib/review/slotKeymap.ts's buildSlotKeymap(spec, editMode, …),
    // its own slotKeymap.test.ts asserts the exact combo set/wiring per
    // mode for both licensePlateSlot (byte-identical to the old
    // buildPlateKeymap) and a second slot. What's pinned here now is
    // just that the page delegates to it rather than re-inlining the
    // table.
    expect(reviewPageSrc).toMatch(/buildSlotKeymap\(activeSlot, editMode, \{/);
  });

  it('class-drop registration early-returns on the plates tab (the invariant behind Finding C.2)', () => {
    // dropOnClassStore's handler is never registered while tab==='plates',
    // which is why f/e/b can be bound as plates actions without colliding
    // with a class hotkey today, even though RESERVED_HOTKEY_LETTERS
    // doesn't list them (next assertion).
    //
    // UPDATE: the invariant is now the extracted isSlotSuppressedTab()
    // predicate (src/lib/review/slotTabGuard.ts, its own
    // slotTabGuard.test.ts), not a bare `if (tab === 'plates') return;`
    // in a 1950-line file.
    expect(reviewPageSrc).toMatch(/if \(isSlotSuppressedTab\(tab\)\) return;/);
  });

  it(
    'RESERVED_HOTKEY_LETTERS (the base constant) still does NOT include f/e/b — ' +
      'reservedHotkeyLetters() is the derived superset that closes Finding C.2 (P2.8c)',
    async () => {
      const { RESERVED_HOTKEY_LETTERS, reservedHotkeyLetters } =
        await import('../../lib/classHotkey');
      for (const letter of ['f', 'e', 'b']) {
        expect(RESERVED_HOTKEY_LETTERS.has(letter)).toBe(false);
      }
      // The base constant deliberately stays narrow — action keys that
      // are never bound to a keyboardStore combo don't belong in it.
      // reservedHotkeyLetters(), reading every registered queue-capable
      // slot's QueueCapability.keymap, is what actually prevents a new
      // class hotkey from colliding with a slot's letters.
      const derived = reservedHotkeyLetters();
      for (const letter of ['f', 'e', 'b', 'd']) {
        expect(derived.has(letter)).toBe(true);
      }
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

describe('T7 (adapted, P2.8b): the Plates tab id is slot-derived, urlId keeps the bookmark contract', () => {
  it('the Plates tab is slot:license_plate internally, with urlId "plates" preserved', async () => {
    const { REVIEW_TABS, tabFromUrlId } = await import('../../lib/reviewTabs');
    const ids = REVIEW_TABS.map((t: { id: string }) => t.id);
    expect(ids).toContain('slot:license_plate');
    expect(ids).not.toContain('plates');
    const plates = REVIEW_TABS.find((t: { id: string }) => t.id === 'slot:license_plate');
    expect(plates?.urlId).toBe('plates');
    expect(tabFromUrlId('plates')).toBe('slot:license_plate');
  });
});

describe('P2.10: clusters/+page.svelte routes via registeredSlots, not a license_plate literal', () => {
  it('no longer hardcodes the license_plate class-name string as a routing condition', () => {
    // Finding B's four routing sites (docs/genericization-plan-2026-09-13.md
    // §1: isLicensePlateFilter, loadLicensePlateCard, the synthetic pinned
    // card's dominant_class_name, and card-click routing) now all resolve
    // via slotForClassName()/registeredSlots (P2.10) instead of comparing
    // against the literal string 'license_plate'. This test intentionally
    // flipped red the moment that migration landed -- see the prior
    // baseline version of this test (before this commit) for the pinned
    // "must still be >= 4" pre-migration checklist it replaces.
    expect(clustersPageSrc).toMatch(/slotForClassName\(/);
    const occurrences = (clustersPageSrc.match(/'license_plate'/g) ?? []).length;
    expect(occurrences).toBe(0);
  });
});
