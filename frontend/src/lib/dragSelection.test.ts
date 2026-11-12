import { describe, expect, it } from 'vitest';
import { createSelection } from '$lib/selection.svelte';
import { captureDragSelection } from './dragSelection';

describe('captureDragSelection', () => {
  it('drags the whole selection when a selected card is grabbed', () => {
    const sel = createSelection();
    sel.ids = new Set(['a', 'b']);
    expect(captureDragSelection('a', [], sel)).toEqual(['a', 'b']);
    expect([...sel.ids]).toEqual(['a', 'b']);
  });

  it('replaces the selection with an unselected card that is grabbed', () => {
    const sel = createSelection();
    sel.ids = new Set(['a', 'b']);
    expect(captureDragSelection('z', [], sel)).toEqual(['z']);
    expect([...sel.ids]).toEqual(['z']);
  });

  it('a lone selected card drags just itself', () => {
    const sel = createSelection();
    sel.ids = new Set(['a']);
    expect(captureDragSelection('a', [], sel)).toEqual(['a']);
  });

  it('changes nothing for a repeated consider or a missing id', () => {
    const sel = createSelection();
    sel.ids = new Set(['a', 'b']);
    expect(captureDragSelection('a', ['a', 'b'], sel)).toBeNull();
    expect(captureDragSelection(undefined, [], sel)).toBeNull();
    expect([...sel.ids]).toEqual(['a', 'b']);
  });
});
