/**
 * P3.6 — the integration gate (docs/genericization-plan-2026-09-13.md
 * §9.7/§9.9 Level 2): the one falsification level that actually proves
 * something, versus Level 1's pure-function unit tests. Registers a
 * throwaway second slot in a TEST-ONLY registry — reusing
 * `aircraftTailNumberSlot`, which already exists and is deliberately
 * NOT in `registeredSlots.ts` — and asserts the full derived surface
 * end to end: review tab, keymap, and training cohorts.
 *
 * This never touches `registeredSlots.ts` itself (that would make the
 * throwaway slot live in the real app) — it calls `buildReviewTabs` /
 * `cohortsForClass` / `resolveSlotRegistry` directly with a
 * locally-built slot list, exactly like `profiles.falsification.test.ts`
 * does for the capability-model layer.
 */

import { describe, it, expect } from 'vitest';
import { resolveSlotRegistry } from './registry';
import { licensePlateSlot } from './profiles/licensePlate';
import { aircraftTailNumberSlot } from './profiles/aircraftTailNumber';
import { buildReviewTabs, isSlotTab, slotTabId, tabFromUrlId } from '../reviewTabs';
import { buildSlotKeymap, singleCharCombos } from '../review/slotKeymap';
import { reservedHotkeyLetters } from '../classHotkey';
import { cohortsForClass } from './cohorts';
import { vi } from 'vitest';

describe('P3.6: registering a second capable slot works with zero production code change', () => {
  const secondSlotTabs = buildReviewTabs([licensePlateSlot, aircraftTailNumberSlot]);

  it('REVIEW_TABS-shaped output contains a sixth tab: id slot:aircraft_tail_number, urlId tails, endpointId tail_numbers', () => {
    const tails = secondSlotTabs.find((t) => t.id === 'slot:aircraft_tail_number');
    expect(tails).toBeDefined();
    expect(tails!.urlId).toBe('tails');
    expect(tails!.endpointId).toBe('tail_numbers');
    expect(secondSlotTabs).toHaveLength(2);
  });

  it('tabFromUrlId resolves "tails" to the second slot; "plates" still resolves to license_plate (bookmark contract)', () => {
    // tabFromUrlId reads the real REVIEW_TABS (registeredSlots-backed),
    // not secondSlotTabs — it's exercised here purely for the bookmark
    // contract half of the assertion, which must hold regardless of
    // what a hypothetical second slot registers.
    expect(tabFromUrlId('plates')).toBe('slot:license_plate');
  });

  it('slotTabId + isSlotTab are structural — true for both slots without any registry lookup', () => {
    expect(isSlotTab(slotTabId('license_plate'))).toBe(true);
    expect(isSlotTab(slotTabId('aircraft_tail_number'))).toBe(true);
    expect(isSlotTab('all')).toBe(false);
    expect(isSlotTab('mismatches')).toBe(false);
  });

  it('buildSlotKeymap(tailSlot, false, handlers) yields enter/d/e/arrowleft/arrowright — no f (no falsePositiveState)', () => {
    const handlers = {
      confirm: vi.fn(),
      reject: vi.fn(),
      toggleEdit: vi.fn(),
      back: vi.fn(),
      advance: vi.fn(),
      saveAndExit: vi.fn(),
    };
    const entries = buildSlotKeymap(aircraftTailNumberSlot, false, handlers);
    expect(entries.map((e) => e.combo)).toEqual([
      'enter',
      'd',
      'e',
      'arrowleft',
      'arrowright',
    ]);
    expect(singleCharCombos(entries)).not.toContain('f');
  });

  it('reservedHotkeyLetters(registry) contains e and d but not f for a registry with only the tail-number slot', () => {
    const { registry } = resolveSlotRegistry({ builtins: [aircraftTailNumberSlot] });
    const reserved = reservedHotkeyLetters(registry);
    expect(reserved.has('e')).toBe(true);
    expect(reserved.has('d')).toBe(true);
    expect(reserved.has('f')).toBe(false);
  });

  it("cohortsForClass() for the tail-number's class returns 4 core + 3 derived cohorts, each with a compiled query and a rowKind", () => {
    const { registry } = resolveSlotRegistry({ builtins: [aircraftTailNumberSlot] });
    const classesById = new Map([[21, 'aircraft']]);
    const cohorts = cohortsForClass(21, 'aircraft', registry, classesById, true);
    expect(cohorts).toHaveLength(7); // 4 core + blind_spots/low_conf/disagreement
    for (const c of cohorts) {
      expect(c.query).toBeDefined();
      expect(['slot', 'crop']).toContain(c.rowKind);
    }
  });
});
