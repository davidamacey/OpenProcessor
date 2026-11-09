/**
 * The search routes take `space` and repeated `fuse`, echo both, and name the
 * space each score was measured in (`score_space`).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { searchCrops } from '$lib/api_cropRead';
import { getEmbeddingSpaces } from '$lib/api_embeddingSpaces';

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

describe('search space and fuse', () => {
  it('sends space once and each fuse key as its own query param', async () => {
    const fetchMock = stubFetch({ total: 0, page: 1, page_size: 30, items: [] });
    await searchCrops('dog', 1, 30, { space: 'pe_core_v1', fuse: ['b', 'c'] });
    const params = new URL(String(fetchMock.mock.calls[0][0]), 'http://x').searchParams;
    expect(params.get('space')).toBe('pe_core_v1');
    expect(params.getAll('fuse')).toEqual(['b', 'c']);
  });

  it('sends neither when unset', async () => {
    const fetchMock = stubFetch({ total: 0, page: 1, page_size: 30, items: [] });
    await searchCrops('dog', 1, 30, { space: null, fuse: null });
    const params = new URL(String(fetchMock.mock.calls[0][0]), 'http://x').searchParams;
    expect(params.has('space')).toBe(false);
    expect(params.has('fuse')).toBe(false);
  });

  it('carries the echoed space, fuse and each score_space onto the page', async () => {
    stubFetch({
      total: 1,
      page: 1,
      page_size: 30,
      space: 'a',
      fuse: ['b'],
      items: [{ crop_id: 'x', semantic_score: 0.5, score_space: 'b' }],
    });
    const res = await searchCrops('dog');
    expect(res.space).toBe('a');
    expect(res.fuse).toEqual(['b']);
    expect(res.items[0].score_space).toBe('b');
  });
});

describe('getEmbeddingSpaces', () => {
  it('lists the spaces for one target', async () => {
    const fetchMock = stubFetch({ spaces: [{ space_key: 'a' }] });
    const spaces = await getEmbeddingSpaces('item');
    expect(String(fetchMock.mock.calls[0][0])).toContain('/embedding/spaces?target=item');
    expect(spaces).toEqual([{ space_key: 'a' }]);
  });
});
