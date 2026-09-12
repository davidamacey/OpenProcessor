import { afterEach, describe, expect, it, vi } from 'vitest';
import { toastStore } from './toast.svelte';

/**
 * Regression test for the real root cause behind "drag-and-drop a crop to
 * relabel it, it flickers back to the old cluster, but a page refresh
 * proves the backend move actually succeeded" (2026-09-12 live report).
 *
 * `crypto.randomUUID` only exists in secure contexts (HTTPS or localhost)
 * per spec. This app is routinely accessed over plain HTTP via a LAN IP,
 * where Safari has no `crypto.randomUUID` at all -- calling it throws.
 * Every optimistic-mutation call site (drag-drop label, discard, ignore,
 * move...) calls `toastStore.success(...)` as its LAST statement inside
 * the same `try` block that did the optimistic UI update, so a throwing
 * toast call was caught by that block's own `catch` and reverted a
 * successful operation's local state -- the backend was right, the UI
 * was wrong, and only a hard refresh (a fresh fetch) showed the truth.
 */
describe('toastStore.push (crypto.randomUUID secure-context fallback)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    toastStore.toasts = [];
  });

  it('does not throw when crypto.randomUUID is undefined (Safari over plain HTTP via LAN IP)', () => {
    vi.stubGlobal('crypto', {});
    expect(() => toastStore.success('Labeled 2 → toyotacar.')).not.toThrow();
    expect(toastStore.toasts.at(-1)?.text).toBe('Labeled 2 → toyotacar.');
  });

  it('still uses crypto.randomUUID when it is available', () => {
    vi.stubGlobal('crypto', { randomUUID: () => 'fixed-uuid' });
    const id = toastStore.success('ok');
    expect(id).toBe('fixed-uuid');
  });

  it('produces unique, non-empty ids across pushes without crypto.randomUUID', () => {
    vi.stubGlobal('crypto', {});
    const id1 = toastStore.info('a');
    const id2 = toastStore.info('b');
    expect(id1).toBeTruthy();
    expect(id2).toBeTruthy();
    expect(id1).not.toBe(id2);
  });
});
