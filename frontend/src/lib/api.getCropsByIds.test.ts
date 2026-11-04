import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCropsByIds } from '$lib/api_cropRead';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

const row = (id: string) => ({
  crop_id: id,
  image_path: `/img/${id}.jpg`,
  bbox_norm: [0, 0, 1, 1],
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getCropsByIds', () => {
  it('reads a page in one request and keeps the served order', async () => {
    const ids = Array.from({ length: 30 }, (_, i) => `c${i}`);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(ok({ total: 30, page: 1, page_size: 30, crops: ids.map(row) }));
    vi.stubGlobal('fetch', fetchMock);

    const crops = await getCropsByIds(ids);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain('/crops?ids=');
    expect(decodeURIComponent(url.split('ids=')[1])).toBe(ids.join(','));
    expect(crops.map((c) => c.id)).toEqual(ids);
  });

  it('drops an id the server no longer has without failing the page', async () => {
    const ids = Array.from({ length: 30 }, (_, i) => `c${i}`);
    const served = ids.filter((id) => id !== 'c7');
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        ok({ total: 29, page: 1, page_size: 30, crops: served.map(row) }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const crops = await getCropsByIds(ids);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(crops).toHaveLength(29);
    expect(crops.map((c) => c.id)).not.toContain('c7');
  });

  it('makes no request for an empty page', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    expect(await getCropsByIds([])).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
