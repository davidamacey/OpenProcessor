/**
 * `keyboardStore.registerAction` (configurable-keyboard-shortcuts plan
 * §5.2, step K1): a registration stores the action id, and dispatch
 * resolves its keys through `keymapStore` at keypress time.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { keyboardStore } from './keyboard.svelte';
import { keymapStore } from './keymap.svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';

const cleanups: Array<() => void> = [];

function withKeys(overrides: Record<string, string[]>): KeymapDocument {
  return {
    ...FALLBACK_KEYMAP,
    actions: FALLBACK_KEYMAP.actions.map((a) =>
      a.id in overrides ? { ...a, keys: overrides[a.id] } : a,
    ),
  };
}

function press(init: KeyboardEventInit & { key: string }): void {
  window.dispatchEvent(
    new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }),
  );
}

afterEach(() => {
  while (cleanups.length) cleanups.pop()!();
  keymapStore.resetToFallback();
  keyboardStore.setScope('global');
  keyboardStore.closeOverlay();
});

describe('keyboardStore.registerAction', () => {
  it("fires on the action's default key", () => {
    const discard = vi.fn();
    cleanups.push(
      keyboardStore.registerAction('review.queue.discard', discard, 'review'),
    );
    keyboardStore.setScope('review');
    press({ key: 'd' });
    expect(discard).toHaveBeenCalledTimes(1);
  });

  it('a rebind takes effect with no re-registration, and the old key stops firing', () => {
    const discard = vi.fn();
    cleanups.push(
      keyboardStore.registerAction('review.queue.discard', discard, 'review'),
    );
    keyboardStore.setScope('review');
    keymapStore.setDocument(withKeys({ 'review.queue.discard': ['x'] }), 'served');

    press({ key: 'd' });
    expect(discard).not.toHaveBeenCalled();
    press({ key: 'x' });
    expect(discard).toHaveBeenCalledTimes(1);
  });

  it('an action bound to no key never fires', () => {
    const discard = vi.fn();
    cleanups.push(
      keyboardStore.registerAction('review.queue.discard', discard, 'review'),
    );
    keyboardStore.setScope('review');
    keymapStore.setDocument(withKeys({ 'review.queue.discard': [] }), 'served');
    press({ key: 'd' });
    expect(discard).not.toHaveBeenCalled();
  });

  it('pinned keys (a slot-declared keymap) win over the keymap', () => {
    const reject = vi.fn();
    cleanups.push(
      keyboardStore.registerAction('review.region.reject', reject, 'review', {
        keys: ['r'],
      }),
    );
    keyboardStore.setScope('review');
    press({ key: 'd' });
    expect(reject).not.toHaveBeenCalled();
    press({ key: 'r' });
    expect(reject).toHaveBeenCalledTimes(1);
  });

  it('registers the W8 per-box accept action now that this branch enables it', () => {
    const accept = vi.fn();
    cleanups.push(
      keyboardStore.registerAction('review.region.accept_box', accept, 'review'),
    );
    keyboardStore.setScope('review');
    press({ key: 'y' });
    expect(accept).toHaveBeenCalledTimes(1);
    expect(
      keyboardStore.shortcutsForCurrentScope().some((s) => s.keys.includes('y')),
    ).toBe(true);
  });

  // K2 fix (plan §5, item 3): a multi-key action used to print one row
  // per key ("Step back" showed twice, once for ← and once for B) —
  // it's one row per ACTION, listing every combo it owns.
  it('lists the keymap label once, with every combo, for a multi-key action', () => {
    cleanups.push(
      keyboardStore.registerAction('review.region.back', vi.fn(), 'review', {
        labelVars: { region: 'widget tag' },
      }),
    );
    keyboardStore.setScope('review');
    const rows = keyboardStore.shortcutsForCurrentScope();
    expect(rows).toEqual([
      {
        keys: ['arrowleft', 'b'],
        scope: 'review',
        description: 'Step back to last confirmed widget tag',
      },
    ]);
  });
});

describe('overlay toggle and close resolve through the keymap', () => {
  it('toggles on every default spelling, including the physical backtick key', () => {
    for (const init of [
      { key: '`' },
      { key: '~', shiftKey: true },
      { key: '`', shiftKey: true },
      { key: 'Dead', code: 'Backquote' },
    ]) {
      keyboardStore.closeOverlay();
      press(init);
      expect(keyboardStore.overlayOpen, JSON.stringify(init)).toBe(true);
    }
    press({ key: 'Escape' });
    expect(keyboardStore.overlayOpen).toBe(false);
  });

  it('a rebound toggle opens on the new key and drops the backtick aliases', () => {
    keymapStore.setDocument(
      withKeys({ 'global.shortcuts_overlay': ['shift+/'] }),
      'served',
    );
    press({ key: '`', code: 'Backquote' });
    expect(keyboardStore.overlayOpen).toBe(false);
    press({ key: '/', shiftKey: true });
    expect(keyboardStore.overlayOpen).toBe(true);
  });
});
