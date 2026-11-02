/**
 * `POST /search/image` (query by picture) and the `target=box` fields of the search pages.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { searchImageCrops, searchSimilarCrops } from '$lib/api_cropRead';

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

describe('searchImageCrops', () => {
  it('posts the picture and box as multipart and the filters as the query', async () => {
    const fetchMock = stubFetch({ total: 0, page: 1, page_size: 200, items: [] });
    const file = new File([new Uint8Array([1, 2, 3])], 'q.png', { type: 'image/png' });
    await searchImageCrops({ file, box: [0.1, 0.2, 0.5, 0.9] }, 1, 200, {
      target: 'box',
      cluster_id: 3,
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain(`${API_PREFIX}/search/image?`);
    const params = new URL(String(url), 'http://x').searchParams;
    expect(params.get('target')).toBe('box');
    expect(params.get('cluster_id')).toBe('3');
    expect(init.method).toBe('POST');
    const fd = init.body as FormData;
    expect(fd).toBeInstanceOf(FormData);
    expect((fd.get('image') as File).name).toBe('q.png');
    expect(fd.get('box')).toBe('0.1,0.2,0.5,0.9');
    expect(fd.has('image_id')).toBe(false);
  });

  it('sends a project image by id, with no upload', async () => {
    const fetchMock = stubFetch({ total: 0, page: 1, page_size: 30, items: [] });
    await searchImageCrops({ imageId: 'img-1' });
    const fd = fetchMock.mock.calls[0][1].body as FormData;
    expect(fd.get('image_id')).toBe('img-1');
    expect(fd.has('image')).toBe(false);
    expect(fd.has('box')).toBe(false);
  });

  it('maps the matched box of a box-target page', async () => {
    stubFetch({
      total: 2,
      page: 1,
      page_size: 30,
      items: [
        {
          crop_id: 'a',
          semantic_score: 0.9,
          matched_box_id: 'b2',
          matched_box_stale: true,
        },
        { crop_id: 'b', semantic_score: 0.8 },
      ],
    });
    const res = await searchImageCrops({ imageId: 'x' });
    expect(res.items[0].matched_box_id).toBe('b2');
    expect(res.items[0].matched_box_stale).toBe(true);
    expect(res.items[1].matched_box_id).toBeNull();
  });
});

describe('searchSimilarCrops with a box', () => {
  it('threads box_id and target through the filter', async () => {
    const fetchMock = stubFetch({ total: 0, page: 1, page_size: 30, items: [] });
    await searchSimilarCrops('crop-7', 1, 30, { box_id: 'b1', target: 'box' });
    const params = new URL(String(fetchMock.mock.calls[0][0]), 'http://x').searchParams;
    expect(params.get('box_id')).toBe('b1');
    expect(params.get('target')).toBe('box');
  });
});
