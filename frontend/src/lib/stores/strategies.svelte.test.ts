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
import { FALLBACK_METHODS, isScopedAssistAvailable } from '$lib/strategies';

// Same METHODS_TODAY / METHODS_WITH_ASSIST_AXES fixtures as
// strategies.test.ts (this plan §6) — inlined rather than cross-imported
// from a sibling *.test.ts file, which isn't a pattern this repo uses.
const METHODS_TODAY = {
  strategies: [
    {
      id: 'ivf',
      axis: 'cluster',
      label: 'FAISS IVF-512 (production)',
      status: 'stable',
      default: true,
    },
    {
      id: 'default',
      axis: 'sort',
      label: 'Recent first',
      status: 'stable',
      default: true,
    },
  ],
  flags: {},
};

const METHODS_WITH_ASSIST_AXES = {
  strategies: [
    ...METHODS_TODAY.strategies,
    {
      id: 'grounding_v2',
      axis: 'detection_profile',
      label: 'Grounding detector v2',
      status: 'stable',
      default: true,
    },
    {
      id: 'warehouse_v1',
      axis: 'prompt_pack',
      label: 'Warehouse vocabulary',
      status: 'stable',
      default: true,
    },
  ],
  flags: {},
};

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
          strategies: [
            {
              id: 'ivf',
              axis: 'cluster',
              label: 'FAISS IVF-512 (production)',
              status: 'stable',
              default: true,
            },
            {
              id: 'default',
              axis: 'sort',
              label: 'Recent first',
              status: 'stable',
              default: true,
            },
          ],
          flags: {},
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
      strategies: [
        {
          id: 'ivf',
          axis: 'cluster',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
        {
          id: 'default',
          axis: 'sort',
          label: 'Recent first',
          status: 'stable',
          default: true,
        },
        {
          id: 'uncertainty',
          axis: 'sort',
          label: 'Uncertainty margin',
          status: 'experimental',
        },
      ],
      flags: {},
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
      .mockResolvedValue(jsonResponse({ strategies: [], flags: {} }));
    vi.stubGlobal('fetch', fetchMock);

    await strategiesStore.init();
    await strategiesStore.init();
    await strategiesStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reset() clears back to the fallback and allows a fresh load', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ strategies: [], flags: {} })),
    );
    await strategiesStore.init();
    expect(strategiesStore.loaded).toBe(true);

    strategiesStore.reset();
    expect(strategiesStore.loaded).toBe(false);
    expect(strategiesStore.methods).toEqual(FALLBACK_METHODS);
  });

  it('loads the assist axes and flips isScopedAssistAvailable true once /curation/methods advertises them', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(METHODS_WITH_ASSIST_AXES)),
    );

    await strategiesStore.init();

    expect(strategiesStore.methods.detection_profiles.map((p) => p.id)).toEqual([
      'grounding_v2',
    ]);
    expect(strategiesStore.methods.prompt_packs.map((p) => p.id)).toEqual([
      'warehouse_v1',
    ]);
    expect(isScopedAssistAvailable(strategiesStore.methods)).toBe(true);
  });

  it("keeps isScopedAssistAvailable false against today's real /curation/methods shape (no assist axes)", async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(METHODS_TODAY)));

    await strategiesStore.init();

    expect(strategiesStore.methods.detection_profiles).toEqual([]);
    expect(strategiesStore.methods.prompt_packs).toEqual([]);
    expect(isScopedAssistAvailable(strategiesStore.methods)).toBe(false);
  });
});
