import { describe, it, expect, vi } from 'vitest';
import { buildPlateKeymap, singleCharCombos } from './plateKeymap';

const handlers = {
  confirmPlate: vi.fn(),
  rejectPlate: vi.fn(),
  markFalsePositive: vi.fn(),
  toggleEdit: vi.fn(),
  plateBack: vi.fn(),
  advance: vi.fn(),
  saveBboxAndExit: vi.fn(),
};

describe('buildPlateKeymap', () => {
  it('scan mode registers exactly enter/d/f/e/arrowleft/b/arrowright', () => {
    const entries = buildPlateKeymap(false, handlers);
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
    const entries = buildPlateKeymap(false, handlers);
    expect(entries.find((e) => e.combo === 'enter')?.fn).toBe(handlers.confirmPlate);
    expect(entries.find((e) => e.combo === 'd')?.fn).toBe(handlers.rejectPlate);
    expect(entries.find((e) => e.combo === 'f')?.fn).toBe(handlers.markFalsePositive);
    expect(entries.find((e) => e.combo === 'e')?.fn).toBe(handlers.toggleEdit);
    expect(entries.find((e) => e.combo === 'arrowleft')?.fn).toBe(handlers.plateBack);
    expect(entries.find((e) => e.combo === 'b')?.fn).toBe(handlers.plateBack);
    expect(entries.find((e) => e.combo === 'arrowright')?.fn).toBe(handlers.advance);
  });

  it('edit mode registers only enter/escape', () => {
    const entries = buildPlateKeymap(true, handlers);
    expect(entries.map((e) => e.combo)).toEqual(['enter', 'escape']);
    expect(entries.find((e) => e.combo === 'enter')?.fn).toBe(handlers.saveBboxAndExit);
    expect(entries.find((e) => e.combo === 'escape')?.fn).toBe(handlers.toggleEdit);
  });
});

describe('singleCharCombos', () => {
  it('keeps single letters, drops multi-char combos', () => {
    const entries = buildPlateKeymap(false, handlers);
    expect(singleCharCombos(entries).sort()).toEqual(['b', 'd', 'e', 'f']);
  });

  it('edit mode has no single-char combos (enter/escape are both multi-char)', () => {
    const entries = buildPlateKeymap(true, handlers);
    expect(singleCharCombos(entries)).toEqual([]);
  });
});
