/**
 * `buildSlotKeymap` entries name their keymap action ids (configurable-
 * keyboard-shortcuts plan §5.3, step K1). The slot-independent keys
 * (confirm/next, save/cancel) follow the keymap; the slot's own verbs
 * follow its declared `queue.keymap`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildSlotKeymap } from './slotKeymap';
import { keymapStore } from '$stores/keymap.svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';

const handlers = {
  confirm: vi.fn(),
  reject: vi.fn(),
  markFalsePositive: vi.fn(),
  toggleEdit: vi.fn(),
  back: vi.fn(),
  advance: vi.fn(),
  saveAndExit: vi.fn(),
};

function withKeys(overrides: Record<string, string[]>): KeymapDocument {
  return {
    ...FALLBACK_KEYMAP,
    actions: FALLBACK_KEYMAP.actions.map((a) =>
      a.id in overrides ? { ...a, keys: overrides[a.id] } : a,
    ),
  };
}

afterEach(() => keymapStore.resetToFallback());

describe('buildSlotKeymap action ids', () => {
  it('scan mode maps each combo to its review.region action', () => {
    const entries = buildSlotKeymap(widgetTagSlot, false, handlers);
    expect(entries.map((e) => [e.combo, e.actionId])).toEqual([
      ['enter', 'review.region.confirm'],
      ['d', 'review.region.reject'],
      ['f', 'review.region.false_positive'],
      ['e', 'review.region.edit_box'],
      ['arrowleft', 'review.region.back'],
      ['b', 'review.region.back'],
      ['arrowright', 'review.region.next'],
    ]);
  });

  it('edit mode maps to box_edit.save / box_edit.cancel', () => {
    const entries = buildSlotKeymap(widgetTagSlot, true, handlers);
    expect(entries.map((e) => [e.combo, e.actionId])).toEqual([
      ['enter', 'box_edit.save'],
      ['escape', 'box_edit.cancel'],
    ]);
  });

  it("descriptions are the keymap's labels with the slot's noun", () => {
    const entries = buildSlotKeymap(aircraftTailNumberSlot, false, handlers);
    const noun = aircraftTailNumberSlot.label.singular;
    expect(entries.find((e) => e.actionId === 'review.region.confirm')?.description).toBe(
      `Confirm ${noun} & advance`,
    );
  });

  it('extra keys on a slot-independent action come from the keymap', () => {
    keymapStore.setDocument(
      withKeys({ 'review.region.next': ['arrowright', 'j'] }),
      'served',
    );
    const entries = buildSlotKeymap(widgetTagSlot, false, handlers);
    expect(
      entries.filter((e) => e.actionId === 'review.region.next').map((e) => e.combo),
    ).toEqual(['arrowright', 'j']);
  });

  it("a slot's declared verb keys are its own, not the keymap's", () => {
    keymapStore.setDocument(withKeys({ 'review.region.reject': ['k'] }), 'served');
    const entries = buildSlotKeymap(aircraftTailNumberSlot, false, handlers);
    expect(
      entries.filter((e) => e.actionId === 'review.region.reject').map((e) => e.combo),
    ).toEqual(['d']);
  });
});
