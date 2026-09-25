import { describe, it, expect, vi } from 'vitest';
import { buildSlotKeymap, rejectKeyGlyph, singleCharCombos } from './slotKeymap';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import type { SlotSpec } from '../annotations/types';

const slotHandlers = {
  confirm: vi.fn(),
  reject: vi.fn(),
  markFalsePositive: vi.fn(),
  toggleEdit: vi.fn(),
  back: vi.fn(),
  advance: vi.fn(),
  saveAndExit: vi.fn(),
};

describe('buildSlotKeymap — widgetTagSlot (the standard region keymap)', () => {
  it('scan mode registers exactly enter/d/f/e/arrowleft/b/arrowright', () => {
    const entries = buildSlotKeymap(widgetTagSlot, false, slotHandlers);
    expect(entries.map((e) => e.combo)).toEqual([
      'enter',
      'd',
      'f',
      'e',
      'arrowleft',
      'b',
      'arrowright',
    ]);
  });

  it('scan mode wires each combo to the matching handler', () => {
    const entries = buildSlotKeymap(widgetTagSlot, false, slotHandlers);
    expect(entries.find((e) => e.combo === 'enter')?.fn).toBe(slotHandlers.confirm);
    expect(entries.find((e) => e.combo === 'd')?.fn).toBe(slotHandlers.reject);
    expect(entries.find((e) => e.combo === 'f')?.fn).toBe(slotHandlers.markFalsePositive);
    expect(entries.find((e) => e.combo === 'e')?.fn).toBe(slotHandlers.toggleEdit);
    expect(entries.find((e) => e.combo === 'arrowleft')?.fn).toBe(slotHandlers.back);
    expect(entries.find((e) => e.combo === 'b')?.fn).toBe(slotHandlers.back);
    expect(entries.find((e) => e.combo === 'arrowright')?.fn).toBe(slotHandlers.advance);
  });

  it('edit mode registers only enter/escape', () => {
    const entries = buildSlotKeymap(widgetTagSlot, true, slotHandlers);
    expect(entries.map((e) => e.combo)).toEqual(['enter', 'escape']);
    expect(entries.find((e) => e.combo === 'enter')?.fn).toBe(slotHandlers.saveAndExit);
    expect(entries.find((e) => e.combo === 'escape')?.fn).toBe(slotHandlers.toggleEdit);
  });

  it('singleCharCombos keeps single letters, drops multi-char combos', () => {
    const entries = buildSlotKeymap(widgetTagSlot, false, slotHandlers);
    expect(singleCharCombos(entries).sort()).toEqual(['b', 'd', 'e', 'f']);
  });

  it('edit mode has no single-char combos (enter/escape are both multi-char)', () => {
    const entries = buildSlotKeymap(widgetTagSlot, true, slotHandlers);
    expect(singleCharCombos(entries)).toEqual([]);
  });
});

describe('buildSlotKeymap — aircraftTailNumberSlot (second-slot case, no falsePositiveState)', () => {
  const tailHandlers = {
    confirm: vi.fn(),
    reject: vi.fn(),
    // No markFalsePositive handler supplied — mirrors the slot having no
    // falsePositiveState / no markFalsePositive combo in its keymap.
    toggleEdit: vi.fn(),
    back: vi.fn(),
    advance: vi.fn(),
    saveAndExit: vi.fn(),
  };

  it('scan mode registers enter/d/e/arrowleft/arrowright — no f', () => {
    const entries = buildSlotKeymap(aircraftTailNumberSlot, false, tailHandlers);
    expect(entries.map((e) => e.combo)).toEqual([
      'enter',
      'd',
      'e',
      'arrowleft',
      'arrowright',
    ]);
    expect(entries.some((e) => e.combo === 'f')).toBe(false);
  });

  it('single-char combos never include f for a slot with no falsePositiveState', () => {
    const entries = buildSlotKeymap(aircraftTailNumberSlot, false, tailHandlers);
    expect(singleCharCombos(entries).sort()).toEqual(['d', 'e']);
  });
});

describe('rejectKeyGlyph', () => {
  // review/+page.svelte's hint strip used to hardcode "D" for every
  // slot's reject action instead of reading the active slot's own
  // `queue.keymap.reject` — the same lookup `buildSlotKeymap` already
  // used correctly for real key dispatch. This fixture's only purpose
  // is proving the hint glyph tracks a keymap that binds reject
  // somewhere other than 'd'.
  const rBoundSlot: SlotSpec = {
    ...widgetTagSlot,
    capabilities: {
      ...widgetTagSlot.capabilities,
      queue: {
        ...widgetTagSlot.capabilities.queue!,
        keymap: { ...widgetTagSlot.capabilities.queue!.keymap, reject: ['r'] },
      },
    },
  };

  it('is behavior-neutral for widgetTagSlot — still "D"', () => {
    expect(rejectKeyGlyph(widgetTagSlot)).toBe('D');
  });

  it('is behavior-neutral for aircraftTailNumberSlot — still "D"', () => {
    expect(rejectKeyGlyph(aircraftTailNumberSlot)).toBe('D');
  });

  it('tracks a slot whose keymap binds reject to a different key', () => {
    expect(rejectKeyGlyph(rBoundSlot)).toBe('R');
  });
});
