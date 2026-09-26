/**
 * The shared `box_edit` key handler (configurable-keyboard-shortcuts plan
 * §1.1 row 24): one resolver for `/review`'s edit mode and the
 * `SlotBboxEditor` modal, matching the key-only switch it replaced.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { boxEditActionFor, runBoxEditKey } from './boxEditKeys';
import { keymapStore } from '$stores/keymap.svelte';
import { FALLBACK_KEYMAP } from './keymapFallback';

const key = (k: string, init: KeyboardEventInit = {}) =>
  new KeyboardEvent('keydown', { key: k, ...init });

function handlers() {
  return {
    save: vi.fn(),
    cancel: vi.fn(),
    nudge: vi.fn(),
    nudgeRightEdge: vi.fn(),
    deleteBox: vi.fn(),
  };
}

afterEach(() => keymapStore.resetToFallback());

describe('boxEditActionFor', () => {
  it('resolves every default box-edit key', () => {
    expect(boxEditActionFor(key('Enter'))).toBe('box_edit.save');
    expect(boxEditActionFor(key('Escape'))).toBe('box_edit.cancel');
    expect(boxEditActionFor(key('Backspace'))).toBe('box_edit.delete_box');
    expect(boxEditActionFor(key('ArrowUp'))).toBe('box_edit.nudge_up');
    expect(boxEditActionFor(key('['))).toBe('box_edit.shrink_right');
    expect(boxEditActionFor(key(']'))).toBe('box_edit.grow_right');
  });

  it('ignores held modifiers, like the e.key switch it replaced', () => {
    expect(boxEditActionFor(key('ArrowLeft', { shiftKey: true }))).toBe(
      'box_edit.nudge_left',
    );
    expect(boxEditActionFor(key('Backspace', { ctrlKey: true }))).toBe(
      'box_edit.delete_box',
    );
  });

  it('resolves the W8 next-box key now that this branch enables it', () => {
    expect(boxEditActionFor(key('Tab'))).toBe('box_edit.next_box');
  });
});

describe('runBoxEditKey', () => {
  it('runs nudges with the given step and reports the key consumed', () => {
    const h = handlers();
    expect(runBoxEditKey(key('ArrowDown'), h, 3)).toBe(true);
    expect(h.nudge).toHaveBeenCalledWith(0, 3);
    expect(runBoxEditKey(key('['), h, 3)).toBe(true);
    expect(h.nudgeRightEdge).toHaveBeenCalledWith(-3);
  });

  it('leaves save/cancel unconsumed when the caller supplies no handler', () => {
    const { save: _s, cancel: _c, ...partial } = handlers();
    expect(runBoxEditKey(key('Enter'), partial, 1)).toBe(false);
    expect(runBoxEditKey(key('Escape'), partial, 1)).toBe(false);
  });

  it('follows a rebind', () => {
    keymapStore.setDocument(
      {
        ...FALLBACK_KEYMAP,
        actions: FALLBACK_KEYMAP.actions.map((a) =>
          a.id === 'box_edit.delete_box' ? { ...a, keys: ['q'] } : a,
        ),
      },
      'served',
    );
    const h = handlers();
    expect(runBoxEditKey(key('Backspace'), h, 1)).toBe(false);
    expect(runBoxEditKey(key('q'), h, 1)).toBe(true);
    expect(h.deleteBox).toHaveBeenCalledTimes(1);
  });
});
