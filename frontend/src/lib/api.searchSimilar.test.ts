/**
 * `GET /search/similar` ("More like this"): the stored vector of one crop is the query, the
 * page is the `/search/text` page.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { searchSimilarCrops } from '$lib/api_cropRead';

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubFetch(body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('searchSimilarCrops', () => {
  it('calls /search/similar with the crop id, paging and the passed filters', async () => {
    const fetchMock = stubFetch({ total: 0, page: 2, page_size: 50, items: [] });
    await searchSimilarCrops('crop-7', 2, 50, { cluster_id: 3 });
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain(`${API_PREFIX}/search/similar?`);
    const params = new URL(url, 'http://x').searchParams;
    expect(params.get('crop_id')).toBe('crop-7');
    expect(params.get('page')).toBe('2');
    expect(params.get('page_size')).toBe('50');
    expect(params.get('cluster_id')).toBe('3');
  });

  it('maps semantic_score, dedupes by crop id and copies unembedded_in_scope', async () => {
    stubFetch({
      total: 3,
      page: 1,
      page_size: 30,
      unembedded_in_scope: 4,
      items: [
        { crop_id: 'a', semantic_score: 0.9 },
        { crop_id: 'b', semantic_score: 0.8 },
        { crop_id: 'a', semantic_score: 0.4 },
      ],
    });
    const res = await searchSimilarCrops('src');
    expect(res.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(res.items[0].similarity_score).toBe(0.9);
    expect(res.unembedded_in_scope).toBe(4);
  });
});
