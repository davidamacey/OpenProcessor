/**
 * Phase 0 characterization tests (docs/genericization-plan-2026-09-13.md
 * §5.1, table T1-T7) for the slot tab's stateful review logic — pinned
 * BEFORE any Phase 2 component genericization touches `review/+page.svelte`
 * or `clusters/+page.svelte`.
 *
 * This repo has no `@testing-library/svelte` harness (see
 * `regionThumbUrlScan.test.ts`'s doc comment). The plan's P0.1/P0.3 call
 * for extracting the pure logic into testable modules FIRST, and that
 * has now landed for every piece that had a real extraction seam:
 * `src/lib/review/slotQueueOps.ts` + `abortRegistry.ts` (undo stack +
 * per-crop abort map), `slotKeymap.ts` (the scan/edit-mode keymap
 * table, slot-generic since P2.8c), `slotTabGuard.ts`
 * (the class-drop suppression predicate
 * behind Finding C.2), and `viewBox.ts` (the frozen-viewport
 * padding/squaring/clamping math) — each with its own executable
 * unit-test suite that supersedes the corresponding assertions below as
 * the real behavior pin. `review/+page.svelte` delegates to all of
 * them. What's left genuinely inline (the `untrack()` wrapping itself —
 * a Svelte-reactivity concern, not extractable math — and the
 * `REVIEW_TABS`/slot-routing baseline in `clusters/+page.svelte`)
 * stays pinned via the static source-scan convention this repo already
 * uses (`EmbeddingPlot.test.ts` / `StrategyBar.test.ts` /
 * `regionThumbUrlScan.test.ts`) until Phase 2's `reviewTabs.ts`
 * data-driving and the SlotGallery extraction give them one.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { registeredSlots } from '../../lib/annotations/registeredSlots';
import { reservedHotkeyLetters } from '../../lib/classHotkey';
import { REVIEW_TABS, tabFromUrlId } from '../../lib/reviewTabs';
import { classesStore } from '../../lib/stores/classes.svelte';

const here = path.dirname(fileURLToPath(import.meta.url));
const reviewPageSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const clustersPageSrc = readFileSync(
  path.resolve(here, '..', 'clusters', '+page.svelte'),
  'utf-8',
);
const queueSlots = registeredSlots.filter((s) => s.capabilities.queue);

describe('T4: slot-tab keymap + reserved-letters invariant (Finding C.2)', () => {
  it('registers exactly the documented scan-mode keymap on a slot tab', () => {
    // Enter=confirm, D=reject, F=false-positive, E=edit, arrowleft/B=back,
    // arrowright=next, N=skip (shared with every tab). Pinning presence
    // via keyboardStore.register() call sites, not exhaustive parsing — a
    // real extraction should replace this with an executable keymap
    // table (QueueCapability.keymap once Phase 2 wires it up).
    //
    // UPDATE (P2.8c): the extraction is now slot-generic —
    // src/lib/review/slotKeymap.ts's buildSlotKeymap(spec, editMode, …),
    // its own slotKeymap.test.ts asserts the exact combo set/wiring per
    // mode for a region slot and a second slot. What's pinned here now is
    // just that the page delegates to it rather than re-inlining the
    // table.
    expect(reviewPageSrc).toMatch(/buildSlotKeymap\(activeSlot, editMode, \{/);
  });

  it('class-drop registration early-returns on a slot tab (the invariant behind Finding C.2)', () => {
    // dropOnClassStore's handler is never registered on a slot tab, which
    // is why a slot's own letters can be bound as slot actions without
    // colliding with a class hotkey. The invariant is the extracted
    // isSlotSuppressedTab() predicate (src/lib/review/slotTabGuard.ts,
    // its own slotTabGuard.test.ts).
    expect(reviewPageSrc).toMatch(/if \(isSlotSuppressedTab\(tab\)\) return;/);
  });

  it(
    "reservedHotkeyLetters() includes every registered queue slot's single-letter " +
      'keymap combos even with no served reserved_hotkeys (W4, 2026-09-24) — the ' +
      'registry union is what closes Finding C.2, independent of the server response',
    async () => {
      // RESERVED_HOTKEY_LETTERS (the hand-maintained base constant) is gone —
      // GET {API_PREFIX}/classes's own `reserved_hotkeys` is the base now
      // (classesStore.reservedHotkeys). Simulate the pre-fetch/offline state
      // (empty) to prove the registry union alone still protects a class
      // hotkey from colliding with a slot's own keymap.
      const slotLetters = queueSlots.flatMap((s) =>
        Object.values(s.capabilities.queue!.keymap)
          .flat()
          .filter((c): c is string => typeof c === 'string' && c.length === 1),
      );
      const prev = classesStore.reservedHotkeys;
      classesStore.reservedHotkeys = [];
      try {
        const derived = reservedHotkeyLetters();
        for (const letter of slotLetters) {
          expect(derived.has(letter)).toBe(true);
        }
      } finally {
        classesStore.reservedHotkeys = prev;
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
  it('keys the slot-tab save-abort registry by crop id, not by cursor index', () => {
    expect(reviewPageSrc).toMatch(/slotMetaAborts = new AbortRegistry\(\)/);
    expect(reviewPageSrc).toMatch(/slotMetaAborts\.start\(id\)/);
    expect(reviewPageSrc).toMatch(/slotMetaAborts\.finish\(id, ac\)/);
  });

  it('undo stack is $state.raw and delegates push/remove/pop to slotQueueOps', () => {
    // Deep reactivity would proxy pushed entries, so removeUndo's
    // identity-based filter could never match a pushed entry.
    expect(reviewPageSrc).toMatch(
      /slotUndoStack = \$state\.raw<SlotUndoEntry\[\]>\(\[\]\)/,
    );
    expect(reviewPageSrc).toMatch(/pushUndo\(slotUndoStack, entry, SLOT_UNDO_MAX\)/);
    expect(reviewPageSrc).toMatch(/removeUndo\(slotUndoStack, entry\)/);
    expect(reviewPageSrc).toMatch(/popUndo\(slotUndoStack\)/);
  });

  it('slotBack() re-insertion delegates to reinsertAt (clamped splice)', () => {
    expect(reviewPageSrc).toMatch(/reinsertAt\(queue\.items, last\.insertAt, fresh\)/);
  });
});

describe('T1-adjacent: frozen-viewport effect uses untrack for the seed read', () => {
  it('the slot bbox-editor viewport-freeze effect wraps its seed call in untrack()', () => {
    // Without untrack(), _seedViewBox() would re-fire on every drag tick
    // and overwrite the operator's in-progress resize with the server
    // snapshot — see plan §3.5 point 1.
    expect(reviewPageSrc).toMatch(/untrack\(\(\) => _seedViewBox\(\)\)/);
  });
});

describe('T7 (adapted, P2.8b): a slot tab id is slot-derived, urlId keeps the bookmark contract', () => {
  it('each queue slot tab is slot:<key> internally, with its own urlId preserved', async () => {
    const ids = REVIEW_TABS.map((t: { id: string }) => t.id);
    for (const slot of queueSlots) {
      const urlId = slot.capabilities.queue!.urlId;
      expect(ids).toContain(`slot:${slot.key}`);
      expect(ids).not.toContain(urlId);
      const tab = REVIEW_TABS.find((t: { id: string }) => t.id === `slot:${slot.key}`);
      expect(tab?.urlId).toBe(urlId);
      expect(tabFromUrlId(urlId)).toBe(`slot:${slot.key}`);
    }
  });
});

describe('P2.10: clusters/+page.svelte routes via registeredSlots, not a class-name literal', () => {
  it("never hardcodes a registered slot's class-name string as a routing condition", () => {
    // Finding B's four routing sites (docs/genericization-plan-2026-09-13.md
    // §1: isSlotFilter, loadSlotInventoryCards, the synthetic pinned
    // card's dominant_class_name, and card-click routing) all resolve via
    // slotForClassName()/registeredSlots instead of comparing against a
    // literal class name.
    expect(clustersPageSrc).toMatch(/slotForClassName\(/);
    for (const slot of registeredSlots) {
      const className = slot.bind.className;
      if (!className) continue;
      expect(clustersPageSrc.split(`'${className}'`).length - 1).toBe(0);
    }
  });
});
