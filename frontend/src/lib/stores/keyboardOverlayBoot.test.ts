/**
 * m11 (2026-09-24 interactive pass): on /clusters, /classes and /dashboard
 * — pages that register no shortcuts of their own — the overlay's window
 * keydown listener was never installed, because `#install()` only ran
 * from `register()`. Also covers Shift+` (~), which normalize()s to
 * "shift+~" and previously matched nothing.
 *
 * Needs a *fresh* module instance (the listener-installed flag is
 * per-instance), so this imports dynamically with vi.resetModules()
 * rather than sharing keyboard.svelte.test.ts's already-registered
 * singleton.
 */
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

function press(init: KeyboardEventInit & { key: string }): void {
  window.dispatchEvent(
    new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }),
  );
}

// Compile the store once up front: under a loaded full suite the first
// fresh import can exceed the test timeout, and the timed-out import then
// finishes in the background and leaves the overlay open for the next test.
beforeAll(async () => {
  await import('./keyboard.svelte');
});

beforeEach(() => {
  vi.resetModules();
});

describe('keyboardStore overlay listener installs at boot (m11)', () => {
  it('toggles the overlay on a bare backtick with zero prior register() calls', async () => {
    const { keyboardStore } = await import('./keyboard.svelte');
    expect(keyboardStore.overlayOpen).toBe(false);
    press({ key: '`', code: 'Backquote' });
    expect(keyboardStore.overlayOpen).toBe(true);
  });

  it('toggles the overlay on Shift+` (~) too', async () => {
    const { keyboardStore } = await import('./keyboard.svelte');
    expect(keyboardStore.overlayOpen).toBe(false);
    press({ key: '~', shiftKey: true, code: 'Backquote' });
    expect(keyboardStore.overlayOpen).toBe(true);
  });
});
