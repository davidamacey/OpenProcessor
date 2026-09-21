import { describe, it, expect, vi } from 'vitest';
import { buildSlotKeymap, rejectKeyGlyph, singleCharCombos } from './slotKeymap';
import { licensePlateSlot } from '../annotations/profiles/licensePlate';
import { aircraftTailNumberSlot } from '../annotations/profiles/aircraftTailNumber';
import type { SlotSpec } from '../annotations/types';

const plateHandlers = {
  confirm: vi.fn(),
  reject: vi.fn(),
  markFalsePositive: vi.fn(),
  toggleEdit: vi.fn(),
  back: vi.fn(),
  advance: vi.fn(),
  saveAndExit: vi.fn(),
};

describe('buildSlotKeymap — licensePlateSlot (no-regression proof for the old buildPlateKeymap)', () => {
  it('scan mode registers exactly enter/d/f/e/arrowleft/b/arrowright', () => {
    const entries = buildSlotKeymap(licensePlateSlot, false, plateHandlers);
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
    const entries = buildSlotKeymap(licensePlateSlot, false, plateHandlers);
    expect(entries.find((e) => e.combo === 'enter')?.fn).toBe(plateHandlers.confirm);
    expect(entries.find((e) => e.combo === 'd')?.fn).toBe(plateHandlers.reject);
    expect(entries.find((e) => e.combo === 'f')?.fn).toBe(
      plateHandlers.markFalsePositive,
    );
    expect(entries.find((e) => e.combo === 'e')?.fn).toBe(plateHandlers.toggleEdit);
    expect(entries.find((e) => e.combo === 'arrowleft')?.fn).toBe(plateHandlers.back);
    expect(entries.find((e) => e.combo === 'b')?.fn).toBe(plateHandlers.back);
    expect(entries.find((e) => e.combo === 'arrowright')?.fn).toBe(plateHandlers.advance);
  });

  it('edit mode registers only enter/escape', () => {
    const entries = buildSlotKeymap(licensePlateSlot, true, plateHandlers);
    expect(entries.map((e) => e.combo)).toEqual(['enter', 'escape']);
    expect(entries.find((e) => e.combo === 'enter')?.fn).toBe(plateHandlers.saveAndExit);
    expect(entries.find((e) => e.combo === 'escape')?.fn).toBe(plateHandlers.toggleEdit);
  });

  it('singleCharCombos keeps single letters, drops multi-char combos', () => {
    const entries = buildSlotKeymap(licensePlateSlot, false, plateHandlers);
    expect(singleCharCombos(entries).sort()).toEqual(['b', 'd', 'e', 'f']);
  });

  it('edit mode has no single-char combos (enter/escape are both multi-char)', () => {
    const entries = buildSlotKeymap(licensePlateSlot, true, plateHandlers);
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
    ...licensePlateSlot,
    capabilities: {
      ...licensePlateSlot.capabilities,
      queue: {
        ...licensePlateSlot.capabilities.queue!,
        keymap: { ...licensePlateSlot.capabilities.queue!.keymap, reject: ['r'] },
      },
    },
  };

  it('is behavior-neutral for licensePlateSlot — still "D"', () => {
    expect(rejectKeyGlyph(licensePlateSlot)).toBe('D');
  });

  it('is behavior-neutral for aircraftTailNumberSlot — still "D"', () => {
    expect(rejectKeyGlyph(aircraftTailNumberSlot)).toBe('D');
  });

  it('tracks a slot whose keymap binds reject to a different key', () => {
    expect(rejectKeyGlyph(rBoundSlot)).toBe('R');
  });
});
