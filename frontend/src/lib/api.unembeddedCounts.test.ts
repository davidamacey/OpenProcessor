/**
 * v0.4.0 response passthrough: `GET /crops` serves `n_unembedded` and
 * `GET /search/text` serves `unembedded_in_scope`; both are copied
 * verbatim onto the `PaginatedResponse`, and a missing one reads `null`
 * (never a client 0).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCrops, searchCrops } from '$lib/api';

function serve(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    ),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('unembedded counts', () => {
  it('getCrops copies the served n_unembedded', async () => {
    serve({ crops: [], total: 0, page: 1, page_size: 50, n_unembedded: 7 });
    expect((await getCrops()).n_unembedded).toBe(7);
  });

  it('getCrops reads a missing n_unembedded as null', async () => {
    serve({ crops: [], total: 0, page: 1, page_size: 50 });
    expect((await getCrops()).n_unembedded).toBeNull();
  });

  it('searchCrops copies the served unembedded_in_scope, 0 included', async () => {
    serve({ items: [], total: 0, page: 1, page_size: 30, unembedded_in_scope: 0 });
    expect((await searchCrops('widget')).unembedded_in_scope).toBe(0);
    serve({ items: [], total: 0, page: 1, page_size: 30, unembedded_in_scope: 12 });
    expect((await searchCrops('widget')).unembedded_in_scope).toBe(12);
  });

  it('searchCrops reads a missing unembedded_in_scope as null', async () => {
    serve({ items: [], total: 0, page: 1, page_size: 30 });
    expect((await searchCrops('widget')).unembedded_in_scope).toBeNull();
  });
});
