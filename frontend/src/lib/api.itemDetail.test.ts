/**
 * Item
 * detail: label history, source image + siblings, and the
 * include_excluded / item_text crop-filter params.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { getCropHistory, getImage, getCrops } from '$lib/api_cropRead';
import { imageDetail } from '$lib/test/fixtures/imageDetail';
import { ApiError } from '$lib/api_errors';
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
      { writer: 'human:label_crop', at: '2026-09-24T00:00:00Z', class_name: 'widget_a' },
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

describe('getImage', () => {
  it('hits GET /images/{id} and maps every item through mapRawCrop', async () => {
    const detail = imageDetail({ image: { image_id: 'img-1', width: 640, height: 427 } });
    const raw = makeItem({ crop_id: 'sibling-1' });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ ...detail, items: [raw] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getImage('img-1');

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/images/img-1`);
    expect(res.image).toEqual(detail.image);
    expect(res.counts).toEqual(detail.counts);
    expect(res.boxes).toEqual([]);
    // The item went through mapRawCrop, not a raw pass-through: `source`
    // is renamed from the wire shape, so the mapping demonstrably ran.
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

    await getCrops({
      cluster_uid: ['uid-1'],
      include_excluded: true,
      item_text: 'ABC',
    });

    const [url] = fetchMock.mock.calls[0];
    const parsed = new URL(url as string, 'http://x');
    expect(parsed.searchParams.get('cluster_uid')).toBe('uid-1');
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
