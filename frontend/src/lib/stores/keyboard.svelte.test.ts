/**
 * Registry tests for the global keyboard store.
 *
 * The store is a module singleton with a lazily-installed window listener,
 * so tests register against the real instance and unregister in afterEach.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { keyboardStore } from './keyboard.svelte';
import { RESERVED_HOTKEY_LETTERS } from '$lib/classHotkey';

const cleanups: Array<() => void> = [];

function reg(combo: string, handler: (e: KeyboardEvent) => void, scope = 'global'): void {
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

  // Phase 7 (audit remediation plan, P1-4): /review opens a fuzzy-search
  // class picker on '/'. Two guards for that addition, per the plan's test
  // list.
  it('registering the review class-picker "/" combo does not install a second window keydown listener', () => {
    // keyboardStore lazily installs exactly one 'keydown' listener the first
    // time anything registers, and #install() no-ops on every call after
    // that (#listenerInstalled). A combobox that rolled its own
    // window.addEventListener instead of going through keyboardStore.register
    // would break this invariant — the guard this test pins.
    const spy = vi.spyOn(window, 'addEventListener');
    const keydownCallsBefore = spy.mock.calls.filter((c) => c[0] === 'keydown').length;
    reg('/', vi.fn(), 'review');
    reg('some-other-review-combo', vi.fn(), 'review');
    const keydownCallsAfter = spy.mock.calls.filter((c) => c[0] === 'keydown').length;
    expect(keydownCallsAfter).toBe(keydownCallsBefore);
    spy.mockRestore();
  });

  it('"/" dispatches only to the registered review-scope handler', () => {
    const handler = vi.fn();
    reg('/', handler, 'review');
    keyboardStore.setScope('review');
    press({ key: '/' });
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('reserves "/" as a class hotkey so it can never collide with the class-picker open key', () => {
    // Mirrors the existing g/n/d/z/x/u/a/m guard: a class bound to '/'
    // would fire both the layout's per-class assign listener AND the
    // class-picker's open handler on the same keypress (neither listener's
    // preventDefault stops the other).
    expect(RESERVED_HOTKEY_LETTERS.has('/')).toBe(true);
  });
});
