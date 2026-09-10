/**
 * StrategiesStore is Phase-0 read-only infrastructure: it fetches
 * `/curation/methods` once and exposes the result. Nothing wires it into the UI
 * yet, and per plan §5 ("no new global keybindings, no behavior change to
 * any existing route") it must never become a UI concern itself — in
 * particular it must not register any window/document event listener,
 * unlike e.g. healthStore's visibilitychange listener.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { strategiesStore } from './strategies.svelte';
import { FALLBACK_METHODS } from '$lib/strategies';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  strategiesStore.reset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  strategiesStore.reset();
});

describe('strategiesStore', () => {
  it('registers zero window/document event listeners while loading', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          cluster_methods: [
            {
              id: 'ivf',
              label: 'FAISS IVF-512 (production)',
              status: 'stable',
              default: true,
            },
          ],
          review_sorts: [
            { id: 'default', label: 'Recent first', status: 'stable', default: true },
          ],
          overlays: [],
          scores: [],
        }),
      ),
    );
    const windowSpy = vi.spyOn(window, 'addEventListener');
    const docSpy = vi.spyOn(document, 'addEventListener');

    await strategiesStore.init();

    expect(windowSpy).not.toHaveBeenCalled();
    expect(docSpy).not.toHaveBeenCalled();
  });

  it('starts with the fallback list before init() resolves', () => {
    expect(strategiesStore.methods).toEqual(FALLBACK_METHODS);
    expect(strategiesStore.loaded).toBe(false);
  });

  it('loads the real response from /curation/methods on success', async () => {
    const serverBody = {
      cluster_methods: [
        {
          id: 'ivf',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
      ],
      review_sorts: [
        { id: 'default', label: 'Recent first', status: 'stable', default: true },
        { id: 'uncertainty', label: 'Uncertainty margin', status: 'experimental' },
      ],
      overlays: [],
      scores: [],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(serverBody)));

    await strategiesStore.init();

    expect(strategiesStore.loaded).toBe(true);
    expect(strategiesStore.loading).toBe(false);
    expect(strategiesStore.error).toBeNull();
    expect(strategiesStore.methods.review_sorts).toHaveLength(2);
    expect(strategiesStore.defaultClusterMethodId).toBe('ivf');
  });

  it('falls back to FALLBACK_METHODS (never throws) when /curation/methods 404s', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
        ),
    );

    await expect(strategiesStore.init()).resolves.toBeUndefined();

    expect(strategiesStore.methods).toEqual(FALLBACK_METHODS);
    expect(strategiesStore.loaded).toBe(true);
  });

  it('only fetches once across repeated init() calls (idempotent load)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ cluster_methods: [], review_sorts: [], overlays: [], scores: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await strategiesStore.init();
    await strategiesStore.init();
    await strategiesStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reset() clears back to the fallback and allows a fresh load', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          cluster_methods: [],
          review_sorts: [],
          overlays: [],
          scores: [],
        }),
      ),
    );
    await strategiesStore.init();
    expect(strategiesStore.loaded).toBe(true);

    strategiesStore.reset();
    expect(strategiesStore.loaded).toBe(false);
    expect(strategiesStore.methods).toEqual(FALLBACK_METHODS);
  });
});
