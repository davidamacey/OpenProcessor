/**
 * Registry tests for the global keyboard store.
 *
 * The store is a module singleton with a lazily-installed window listener,
 * so tests register against the real instance and unregister in afterEach.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { keyboardStore } from './keyboard.svelte';

const cleanups: Array<() => void> = [];

function reg(
  combo: string,
  handler: (e: KeyboardEvent) => void,
  scope = 'global',
): void {
  cleanups.push(keyboardStore.register(combo, handler, scope));
}

function press(
  init: KeyboardEventInit & { key: string },
  target: EventTarget = window,
): void {
  target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, ...init }));
}

afterEach(() => {
  while (cleanups.length) cleanups.pop()!();
  keyboardStore.setScope('global');
  keyboardStore.closeOverlay();
});

describe('keyboardStore', () => {
  it("routes Shift+N to a 'shift+n' registration, unshadowed by a plain 'n'", () => {
    const skip = vi.fn();
    const flag = vi.fn();
    // Registration order mirrors the cluster page: the plain-letter skip
    // binding is registered before the shifted flag binding.
    reg('n', skip);
    reg('shift+n', flag);

    press({ key: 'N', shiftKey: true });

    expect(flag).toHaveBeenCalledTimes(1);
    expect(skip).not.toHaveBeenCalled();
  });

  it("never fires a bare uppercase 'N' registration (register() lowercases)", () => {
    // Pinned regression for bug 1.2: reg('N', ...) collapses to combo 'n',
    // which the earlier plain-'n' binding wins, and Shift+N normalizes to
    // 'shift+n' and matches nothing. Bindings must be spelled 'shift+n'.
    const skip = vi.fn();
    const flag = vi.fn();
    reg('n', skip);
    reg('N', flag);

    press({ key: 'N', shiftKey: true });
    expect(flag).not.toHaveBeenCalled();
    expect(skip).not.toHaveBeenCalled();

    press({ key: 'n' });
    expect(skip).toHaveBeenCalledTimes(1);
    expect(flag).not.toHaveBeenCalled();
  });

  it('dispatches to the first matching registration only', () => {
    const first = vi.fn();
    const second = vi.fn();
    reg('g', first);
    reg('g', second);

    press({ key: 'g' });

    expect(first).toHaveBeenCalledTimes(1);
    expect(second).not.toHaveBeenCalled();
  });

  it('skips registrations whose scope is neither global nor current', () => {
    const clusterOnly = vi.fn();
    reg('m', clusterOnly, 'cluster');

    keyboardStore.setScope('review');
    press({ key: 'm' });
    expect(clusterOnly).not.toHaveBeenCalled();

    keyboardStore.setScope('cluster');
    press({ key: 'm' });
    expect(clusterOnly).toHaveBeenCalledTimes(1);
  });

  it('suppresses handlers while a text input is the event target', () => {
    const handler = vi.fn();
    reg('d', handler);

    const input = document.createElement('input');
    document.body.appendChild(input);
    try {
      press({ key: 'd' }, input);
      expect(handler).not.toHaveBeenCalled();
    } finally {
      input.remove();
    }

    press({ key: 'd' });
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('normalizes modifier combos in ctrl+meta+alt+shift order', () => {
    const handler = vi.fn();
    reg('ctrl+shift+enter', handler);

    press({ key: 'Enter', ctrlKey: true, shiftKey: true });

    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('unregistering removes the handler', () => {
    const handler = vi.fn();
    const off = keyboardStore.register('q', handler);
    off();

    press({ key: 'q' });

    expect(handler).not.toHaveBeenCalled();
  });
});
