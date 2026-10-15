/**
 * `GET /search/text` can serve one crop id more than once (a crop matched
 * through several of its boxes). The keyed result grids throw
 * `each_key_duplicate` on a repeat, so the repeat is dropped where the
 * response is mapped: the first (best-ranked) hit wins.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { searchCrops } from '$lib/api_cropRead';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('searchCrops duplicate crop ids', () => {
  it('keeps the first hit per id and its score', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            total: 3,
            page: 1,
            page_size: 30,
            items: [
              { crop_id: 'a', semantic_score: 0.9 },
              { crop_id: 'b', semantic_score: 0.8 },
              { crop_id: 'a', semantic_score: 0.4 },
            ],
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    );
    const res = await searchCrops('widget');
    expect(res.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(res.items[0].similarity_score).toBe(0.9);
  });
});
