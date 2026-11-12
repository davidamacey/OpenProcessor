/** The editor's wire clients: URL, verb, body and the item mapping, through a stubbed `fetch`. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { patchItemGeometry, undoItemGeometry } from '$lib/api_cropWrites';
import { createImageItem, locateImage } from '$lib/api_images';
import { makeItem } from './test/makeItem';

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

afterEach(() => vi.unstubAllGlobals());

describe('createImageItem', () => {
  it('POSTs the box, class name and request id and maps the item', async () => {
    const f = stubFetch({
      item: makeItem({ crop_id: 'new-1', geometry_revision: 1 }),
      created: true,
      image_boxes: [],
      image_counts: { items: 1, labeled: 1, validated: 1, unlabeled: 0, in_holdout: 0 },
      embedding_state: 'embedded',
      embedding_message: null,
    });
    const body = {
      bbox_norm: [0.1, 0.1, 0.5, 0.5] as [number, number, number, number],
      class_name: 'car',
      request_id: '6f1c3b6e-0000-4000-8000-000000000001',
    };
    const res = await createImageItem('img-1', body);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/images/img-1/items`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual(body);
    expect(res.item.id).toBe('new-1');
    expect(res.item.geometry_revision).toBe(1);
  });
});

describe('patchItemGeometry and undoItemGeometry', () => {
  it('PATCHes the box with the expected revision and maps the served item', async () => {
    const f = stubFetch({
      item: makeItem({
        geometry_revision: 5,
        geometry_locked: true,
        lock_label: 'Placed',
      }),
      image_boxes: [],
      embedding_state: 'embedded',
      embedding_message: null,
    });
    const res = await patchItemGeometry('c1', {
      bbox_norm: [0.2, 0.2, 0.6, 0.6],
      expected_geometry_revision: 4,
      drop_region_box_ids: ['b1'],
    });
    const [url, init] = f.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/geometry`);
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body)).toEqual({
      bbox_norm: [0.2, 0.2, 0.6, 0.6],
      expected_geometry_revision: 4,
      drop_region_box_ids: ['b1'],
    });
    expect(res.item.geometry_revision).toBe(5);
    expect(res.item.geometry_locked).toBe(true);
    expect(res.item.lock_label).toBe('Placed');
  });

  it('POSTs the revision the editor holds to the undo route', async () => {
    const f = stubFetch({
      item: makeItem({ geometry_revision: 6 }),
      image_boxes: [],
      embedding_state: 'embedded',
      embedding_message: null,
    });
    const res = await undoItemGeometry('c1', 5);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/geometry/undo`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ expected_geometry_revision: 5 });
    expect(res.item.geometry_revision).toBe(6);
  });
});

describe('locateImage', () => {
  it('sends the gallery filter and the target image id', async () => {
    const f = stubFetch({
      image_id: 'img-9',
      in_view: true,
      index: 41,
      total: 100,
      cursor_before: 'cb',
      cursor_after: 'ca',
      reason: null,
      reason_message: null,
    });
    const res = await locateImage(
      { class_name: ['car'], sort: 'taken_at', label_state: 'has_unlabeled' },
      'img-9',
    );
    const [url] = f.mock.calls[0];
    const u = new URL(url, 'http://x');
    expect(u.pathname).toBe(`${API_PREFIX}/images/locate`);
    expect(u.searchParams.getAll('class_name')).toEqual(['car']);
    expect(u.searchParams.get('target_image_id')).toBe('img-9');
    expect(u.searchParams.get('sort')).toBe('taken_at');
    expect(res.cursor_before).toBe('cb');
  });
});
