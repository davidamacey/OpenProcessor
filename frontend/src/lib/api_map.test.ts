import { beforeEach, describe, expect, it, vi } from 'vitest';
import { setScopedPrefix } from '$lib/api';
import { getMapCells, getMapMeta, getMapTileImages } from '$lib/api_map';

const fetchMock = vi.fn();

function ok(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
  setScopedPrefix('/curation/projects/default');
});

describe('map wrappers', () => {
  it('reads the meta from the project scope', async () => {
    fetchMock.mockResolvedValue(ok({ available: false }));
    await getMapMeta();
    expect(fetchMock.mock.calls[0][0]).toBe('/curation/projects/default/map');
  });

  it('sends zoom, box and the shared filter to /map/cells', async () => {
    fetchMock.mockResolvedValue(ok({ type: 'FeatureCollection', features: [] }));
    await getMapCells(
      { zoom: 6, bbox: '7,46,9,48' },
      { class_name: ['class_a', 'class_b'], taken_from: '2024-03-05' },
    );
    const url = new URL(fetchMock.mock.calls[0][0], 'http://x');
    expect(url.pathname).toBe('/curation/projects/default/map/cells');
    expect(url.searchParams.getAll('class_name')).toEqual(['class_a', 'class_b']);
    expect(url.searchParams.get('taken_from')).toBe('2024-03-05');
    expect(url.searchParams.get('zoom')).toBe('6');
    expect(url.searchParams.get('bbox')).toBe('7,46,9,48');
  });

  it('parses the geo+json the server answers /map/cells with as an object, not text', async () => {
    const body = { type: 'FeatureCollection', features: [] };
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/geo+json' },
      }),
    );
    expect(await getMapCells({ zoom: 6, bbox: '7,46,9,48' }, {})).toEqual(body);
  });

  it('asks for the images of one cell', async () => {
    fetchMock.mockResolvedValue(
      ok({ tile: '9/1/1', count: 0, image_ids: [], truncated: false }),
    );
    await getMapTileImages('9/1/1', {});
    const url = new URL(fetchMock.mock.calls[0][0], 'http://x');
    expect(url.pathname).toBe('/curation/projects/default/map/images');
    expect(url.searchParams.get('tile')).toBe('9/1/1');
  });
});
