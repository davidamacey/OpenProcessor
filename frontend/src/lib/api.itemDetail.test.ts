/**
 * W7/W8 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — item
 * detail: label history, source image + siblings, and the
 * include_excluded / item_text crop-filter params.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, API_PREFIX, getCropHistory, getCropContext, getCrops } from './api';
import { makeItem } from './test/makeItem';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getCropHistory', () => {
  it('hits GET /crops/{id}/history and passes the entries through verbatim', async () => {
    const entries = [
      { writer: 'human:label_crop', at: '2026-09-24T00:00:00Z', class_name: 'sedan' },
      { writer: 'vlm_pipeline', at: '2026-09-23T00:00:00Z', class_name: null },
    ];
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ crop_id: 'c1', entries }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCropHistory('c1');

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/history`);
    expect(res.crop_id).toBe('c1');
    expect(res.entries).toEqual(entries);
  });
});

describe('getCropContext', () => {
  it('hits GET /crops/{id}/context and maps every sibling through mapRawCrop', async () => {
    const image = {
      image_id: 'img-1',
      image_path: '/nas/img-1.jpg',
      width: 640,
      height: 427,
      source: 'tag_holdout_sample',
      indexed_at: '2026-09-24T00:00:00Z',
    };
    const raw = makeItem({ crop_id: 'sibling-1' });
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ image, items: [raw] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCropContext('c1');

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/context`);
    expect(res.image).toEqual(image);
    // Sibling went through mapRawCrop, not a raw pass-through — the wire
    // shape's `class_source` isn't a Crop field name collision, but
    // `source` (renamed from confidence-adjacent mapping) proves the
    // mapping actually ran rather than just spreading the raw object.
    expect(res.items[0].id).toBe('sibling-1');
    expect(res.items[0].source).toBe(raw.source);
  });
});

describe('getCrops: include_excluded / item_text params', () => {
  it('forwards include_excluded and item_text on the query string', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 20, crops: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getCrops({ cluster_id: -2, include_excluded: true, item_text: 'ABC' });

    const [url] = fetchMock.mock.calls[0];
    const parsed = new URL(url as string, 'http://x');
    expect(parsed.searchParams.get('cluster_id')).toBe('-2');
    expect(parsed.searchParams.get('include_excluded')).toBe('true');
    expect(parsed.searchParams.get('item_text')).toBe('ABC');
  });

  it('surfaces a 400 item_text rejection as an ApiError with the server detail', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ detail: 'item_text must contain a letter or digit' }, 400),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getCrops({ item_text: '   ' })).rejects.toMatchObject({
      status: 400,
      detail: 'item_text must contain a letter or digit',
    });
    // ApiError is the concrete type callers narrow on (routes/clusters
    // catches this instanceof check to show an inline hint vs a toast).
    await expect(getCrops({ item_text: '   ' })).rejects.toBeInstanceOf(ApiError);
  });
});
