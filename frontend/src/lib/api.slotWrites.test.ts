/**
 * patchSlotMeta / the per-box region writes — the generic write surface
 * /review's inline panel uses. For the region slot, patchSlotMeta's body
 * is exactly the item-level `region_status` / `region_rejection_reason`
 * keys (its reading is per box, so there is no `region_text` key); the box
 * list, a box's state and its text go through the four per-box routes.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  patchSlotMeta,
  batchRegionStatus,
  putRegionBoxes,
  putBatchRegions,
  patchRegionBox,
  postBatchBoxState,
  regionConflictDetail,
  ApiError,
  API_PREFIX,
} from './api';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';

function okResponse(body: unknown = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

/** Minimal RawCrop the live backend's `{..., item}` wrapper carries. */
function rawItem(id: string): Record<string, unknown> {
  return { crop_id: id, image_path: `/img/${id}.jpg`, bbox_norm: [0, 0, 1, 1] };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('patchSlotMeta', () => {
  it('widgetTagSlot: body carries only the item-level region_* meta wire keys (no region_text)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ crop_id: 'c1', updated_fields: [], item: rawItem('c1') }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await patchSlotMeta(widgetTagSlot, 'c1', {
      status: 'detected',
      text: 'TAG-001',
      rejectionReason: null,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/region_meta`);
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body)).toEqual({
      region_status: 'detected',
      region_rejection_reason: null,
    });
  });

  it('a scalar slot with an item-level text field still writes it through its own wire name', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ crop_id: 'c1', updated_fields: [], item: rawItem('c1') }),
      );
    vi.stubGlobal('fetch', fetchMock);
    await patchSlotMeta(aircraftTailNumberSlot, 'c1', { text: 'N123AB' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/tail_meta`);
    expect(JSON.parse(init.body)).toEqual({ tail_number: 'N123AB' });
  });

  it('returns the mapped item alongside crop_id/updated_fields', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        crop_id: 'c1',
        updated_fields: ['region_status'],
        item: rawItem('c1'),
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await patchSlotMeta(widgetTagSlot, 'c1', { status: 'detected' });
    expect(res.crop_id).toBe('c1');
    expect(res.updated_fields).toEqual(['region_status']);
    expect(res.item.id).toBe('c1');
  });

  it('omits keys whose value is undefined (leaves them untouched, per the old contract)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ crop_id: 'c1', updated_fields: [], item: rawItem('c1') }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await patchSlotMeta(widgetTagSlot, 'c1', { status: 'detected' });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ region_status: 'detected' });
  });

  it('a slot with no matching capability produces an empty body rather than throwing', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ crop_id: 'c1', updated_fields: [], item: rawItem('c1') }),
      );
    vi.stubGlobal('fetch', fetchMock);
    const noLifecycleSpec = {
      ...widgetTagSlot,
      capabilities: { text: widgetTagSlot.capabilities.text },
    };
    await patchSlotMeta(noLifecycleSpec, 'c1', { status: 'detected' });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({});
  });
});

describe('putRegionBoxes (PUT /crops/{id}/regions)', () => {
  it('sends the box list, frame "parent", the human label source and the revision', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') }));
    vi.stubGlobal('fetch', fetchMock);
    const { crop: item } = await putRegionBoxes(
      'c1',
      [{ box_id: 'b1' }, { box_id: null, bbox_norm: [0.1, 0.1, 0.2, 0.2] }],
      { regionStatus: 'detected', expectedRegionRevision: 3 },
    );
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/regions`);
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({
      boxes: [{ box_id: 'b1' }, { box_id: null, bbox_norm: [0.1, 0.1, 0.2, 0.2] }],
      frame: 'parent',
      region_label_source: 'human',
      region_status: 'detected',
      expected_region_revision: 3,
    });
    expect(item.id).toBe('c1');
  });

  it('omits region_status and the revision when not given', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') }));
    vi.stubGlobal('fetch', fetchMock);
    await putRegionBoxes('c1', []);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      boxes: [],
      frame: 'parent',
      region_label_source: 'human',
    });
  });
});

describe('vector_refresh on the region writes', () => {
  const refresh = { embedded: 1, pending: 2 };

  it('putRegionBoxes and patchRegionBox return the served vector_refresh', async () => {
    const fetchMock = vi.fn(async () =>
      okResponse({ item: rawItem('c1'), vector_refresh: refresh }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const put = await putRegionBoxes('c1', []);
    expect(put.vectorRefresh).toEqual(refresh);
    const patch = await patchRegionBox('c1', 'b1', { state: 'accepted' });
    expect(patch.vectorRefresh).toEqual(refresh);
    expect(patch.crop.id).toBe('c1');
  });

  it('a response without vector_refresh reads null, never a guessed zero', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') })),
    );
    expect((await putRegionBoxes('c1', [])).vectorRefresh).toBeNull();
  });

  it('putBatchRegions and postBatchBoxState return it too', async () => {
    const fetchMock = vi.fn(async () =>
      okResponse({
        updated: 1,
        invalid: [],
        conflicts: [],
        items: [],
        vector_refresh: refresh,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    expect((await putBatchRegions(['a'], [])).vectorRefresh).toEqual(refresh);
    expect(
      (await postBatchBoxState([{ cropId: 'a', boxId: 'b' }], 'accepted')).vectorRefresh,
    ).toEqual(refresh);
  });
});

describe('putBatchRegions (PUT /crops/batch_regions)', () => {
  it('sends only the keys the batch route declares (no frame: the route forbids extras)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    vi.stubGlobal('fetch', fetchMock);
    await putBatchRegions(['a', 'b'], [], { regionStatus: 'no_region_visible' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/batch_regions`);
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({
      crop_ids: ['a', 'b'],
      boxes: [],
      region_status: 'no_region_visible',
    });
  });
});

describe('patchRegionBox (PATCH /crops/{id}/regions/{box_id})', () => {
  it('addresses the box, and writes only the keys given', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') }));
    vi.stubGlobal('fetch', fetchMock);
    await patchRegionBox('c/1', 'b 2', { state: 'accepted', expectedRegionRevision: 4 });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c%2F1/regions/b%202`);
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body)).toEqual({
      state: 'accepted',
      expected_region_revision: 4,
    });
  });

  it('text null clears the reading; an absent text key leaves it alone', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      okResponse({ item: rawItem('c1') }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await patchRegionBox('c1', 'b1', { text: null });
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({ text: null });
    await patchRegionBox('c1', 'b1', { state: 'rejected' });
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({
      state: 'rejected',
    });
  });
});

describe('postBatchBoxState (POST /regions/batch_box_state)', () => {
  it('sends crop/box targets and one state', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ updated: 1, invalid: [], conflicts: [], items: [rawItem('a')] }),
      );
    vi.stubGlobal('fetch', fetchMock);
    const res = await postBatchBoxState([{ cropId: 'a', boxId: 'b1' }], 'false_positive');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/regions/batch_box_state`);
    expect(JSON.parse(init.body)).toEqual({
      targets: [{ crop_id: 'a', box_id: 'b1' }],
      state: 'false_positive',
      region_label_source: 'human',
    });
    expect(res.updated).toBe(1);
    expect(res.items[0].id).toBe('a');
  });

  it('refuses an empty target list before any request', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    await expect(postBatchBoxState([], 'accepted')).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('regionConflictDetail (409 region_conflict)', () => {
  const conflictBody = {
    detail: {
      error: 'region_conflict',
      current_region_revision: 9,
      current_box_ids: ['b1'],
      item: rawItem('c1'),
    },
  };

  it('extracts the current revision, box ids and the mapped item', () => {
    const d = regionConflictDetail(new ApiError(409, '/x', conflictBody));
    expect(d?.currentRegionRevision).toBe(9);
    expect(d?.currentBoxIds).toEqual(['b1']);
    expect(d?.item.id).toBe('c1');
  });

  it('is null for any other error', () => {
    expect(regionConflictDetail(new ApiError(409, '/x', { detail: 'nope' }))).toBeNull();
    expect(regionConflictDetail(new ApiError(422, '/x', conflictBody))).toBeNull();
    expect(regionConflictDetail(new Error('x'))).toBeNull();
  });
});

describe('batchRegionStatus', () => {
  it("writes the slot's own lifecycle wire fields (B3 region_* body)", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ updated: 2, conflicts: [], invalid: [], items: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await batchRegionStatus(widgetTagSlot, ['a', 'b'], 'detected', {
      verified: true,
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/regions/batch_status`);
    expect(JSON.parse(init.body)).toEqual({
      crop_ids: ['a', 'b'],
      region_status: 'detected',
      region_verified: true,
      region_label_source: 'human',
    });
  });

  it('p5 (2026-09-24 interactive pass): omits region_verified entirely when the caller never passed verified, instead of sending an ignored null', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ updated: 1, conflicts: [], invalid: [], items: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);

    // The region gallery's bulk reject/false-positive calls never pass
    // `verified` at all (SlotGallery.svelte / slotGalleryController).
    await batchRegionStatus(widgetTagSlot, ['a'], 'no_region_visible');

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      crop_ids: ['a'],
      region_status: 'no_region_visible',
      region_label_source: 'human',
    });
    expect('region_verified' in body).toBe(false);
  });
});
