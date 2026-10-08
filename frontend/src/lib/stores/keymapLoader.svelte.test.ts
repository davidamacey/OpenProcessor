/**
 * `loadKeymap()` / `keymapAvailability` (K2, docs/design/
 * configurable-keyboard-shortcuts-plan-2026-09-26.md §5.1). Stubs
 * `fetch` directly, same pattern as `strategies.svelte.test.ts` — both
 * `keymapStore` and `keymapAvailability` are module-scope singletons, so
 * mocking `$lib/api` via `vi.doMock` + dynamic import doesn't reliably
 * rebind the
 * already-loaded `keymap.svelte.ts` closure in this suite's module
 * graph; stubbing the actual network boundary avoids that trap.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { keymapAvailability, keymapStore, loadKeymap } from './keymap.svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  keymapAvailability.reset();
  keymapStore.resetToFallback();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  keymapAvailability.reset();
  keymapStore.resetToFallback();
});

describe('loadKeymap / keymapAvailability', () => {
  it('sets available = false and stays on the fallback on a 404', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ detail: 'nf' }, 404)),
    );
    await loadKeymap();
    expect(keymapAvailability.available).toBe(false);
    expect(keymapStore.source).toBe('fallback');
  });

  it('sets available = false on a 501', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ detail: 'ni' }, 501)),
    );
    await loadKeymap();
    expect(keymapAvailability.available).toBe(false);
  });

  it('adopts a served document and sets available = true', async () => {
    const served: KeymapDocument = {
      ...FALLBACK_KEYMAP,
      revision: 7,
      is_default: false,
      reserved_hotkeys: ['x', 'y'],
      actions: FALLBACK_KEYMAP.actions.map((a) =>
        a.id === 'review.queue.discard' ? { ...a, keys: ['x'], default: ['x'] } : a,
      ),
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(served)));
    await loadKeymap();
    expect(keymapAvailability.available).toBe(true);
    expect(keymapStore.source).toBe('served');
    expect(keymapStore.revision).toBe(7);
    expect(keymapStore.isDefault).toBe(false);
    expect(keymapStore.keysFor('review.queue.discard')).toEqual(['x']);
    expect(keymapStore.reserved).toEqual(['x', 'y']);
  });

  // A served id this build doesn't know is ignored — no handler for it
  // (plan §5.1, "unknown ids"). A build-known id the server omits falls
  // back to the fallback default (covered by keymap.svelte.test.ts
  // already for K1's document-merge logic; this just proves the loader
  // hands the served document to that same merge path unmodified).
  it('a build-known id the served doc omits falls back to its default', async () => {
    const served: KeymapDocument = {
      ...FALLBACK_KEYMAP,
      revision: 1,
      actions: FALLBACK_KEYMAP.actions.filter((a) => a.id !== 'cluster.undo'),
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(served)));
    await loadKeymap();
    expect(keymapStore.keysFor('cluster.undo')).toEqual(['z']);
  });

  // The fail-open rule every other availability probe in this codebase
  // follows: a transient failure must not be indistinguishable from "the
  // route doesn't exist" (that would hide a working route on a blip).
  it('leaves availability at its prior value on a network error, stays on the fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await loadKeymap();
    expect(keymapAvailability.available).not.toBe(false);
    expect(keymapStore.source).toBe('fallback');
  }, 10000);
});
