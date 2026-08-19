import { describe, expect, it } from 'vitest';
import { createSelection } from './selection.svelte';

const ORDER = ['a', 'b', 'c', 'd', 'e'];

const plain = undefined;
const ctrl = { ctrlKey: true } as MouseEvent;
const shift = { shiftKey: true } as MouseEvent;

describe('createSelection', () => {
  it("replace mode: a plain click selects exactly one card", () => {
    const sel = createSelection({ plainClick: 'replace' });
    sel.click('a', plain, ORDER);
    sel.click('c', plain, ORDER);
    expect([...sel.ids]).toEqual(['c']);
    expect(sel.anchorId).toBe('c');
  });

  it('toggle mode: plain clicks accumulate and un-toggle', () => {
    const sel = createSelection({ plainClick: 'toggle' });
    sel.click('a', plain, ORDER);
    sel.click('c', plain, ORDER);
    expect([...sel.ids].sort()).toEqual(['a', 'c']);
    sel.click('a', plain, ORDER);
    expect([...sel.ids]).toEqual(['c']);
  });

  it('ctrl/cmd-click toggles a single card in both modes', () => {
    for (const mode of ['replace', 'toggle'] as const) {
      const sel = createSelection({ plainClick: mode });
      sel.click('a', plain, ORDER);
      sel.click('d', ctrl, ORDER);
      expect([...sel.ids].sort()).toEqual(['a', 'd']);
      sel.click('d', ctrl, ORDER);
      expect([...sel.ids]).toEqual(['a']);
      expect(sel.anchorId).toBe('d');
    }
  });

  it('shift-click selects the inclusive range from the anchor', () => {
    const sel = createSelection({ plainClick: 'replace' });
    sel.click('b', plain, ORDER);
    sel.click('d', shift, ORDER);
    expect([...sel.ids].sort()).toEqual(['b', 'c', 'd']);
    // Anchor stays put so a second shift-click re-ranges from it.
    expect(sel.anchorId).toBe('b');
  });

  it('shift-click works backwards and unions with the existing selection', () => {
    const sel = createSelection({ plainClick: 'replace' });
    sel.click('e', ctrl, ORDER);
    sel.click('c', ctrl, ORDER);
    sel.click('a', shift, ORDER); // anchor is 'c', range a..c
    expect([...sel.ids].sort()).toEqual(['a', 'b', 'c', 'e']);
  });

  it('falls back to the plain branch when the anchor left the list', () => {
    const sel = createSelection({ plainClick: 'replace' });
    sel.click('a', plain, ORDER);
    sel.click('c', shift, ['b', 'c', 'd']); // 'a' no longer displayed
    expect([...sel.ids]).toEqual(['c']);
  });

  it('selectAll and clear', () => {
    const sel = createSelection({ plainClick: 'replace' });
    sel.selectAll(ORDER);
    expect(sel.size).toBe(5);
    expect(sel.has('d')).toBe(true);
    sel.clear();
    expect(sel.size).toBe(0);
    expect(sel.anchorId).toBeNull();
  });

  it('ids can be replaced wholesale (arrow-key navigation)', () => {
    const sel = createSelection({ plainClick: 'replace' });
    sel.ids = new Set(['b']);
    sel.anchorId = 'b';
    expect([...sel.ids]).toEqual(['b']);
    expect(sel.anchorId).toBe('b');
  });
});
