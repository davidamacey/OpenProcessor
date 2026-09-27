/**
 * setSlotBox / patchSlotMeta (C6, docs/design/slot-generic-crop-mapping-
 * plan-2026-09-21.md §6.3) — the generic write surface /review's inline
 * panel uses. For a region slot, patchSlotMeta's request body is exactly
 * the region_text / region_status / region_rejection_reason wire keys.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { setSlotBox, patchSlotMeta, batchRegionStatus, API_PREFIX } from './api';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

// W8 (owner decision 2026-09-26, no backward compatibility): the served
// region slot dropped its scalar bboxField/setBox/clearBox entirely — box
// writes now go through putRegionBoxes/patchRegionBox (multiBox.test.ts,
// multiBoxRegionController.test.ts cover those). setSlotBox itself is
// still a real, generic write path for a tier-2 single-box slot (e.g. a
// deployment profile that isn't multi-box-capable), so its own tests use
// a minimal synthetic single-box spec rather than the region fixture.
const singleBoxSlot = {
  ...widgetTagSlot,
  capabilities: {
    ...widgetTagSlot.capabilities,
    subBox: {
      bboxField: 'region_bbox_norm',
      storedFrame: 'source' as const,
      ring: { confirmed: '', proposed: '', rejected: '' },
      editor: { thumbSize: 512, viewPadding: 2.5, nudgeStep: 1 / 512 },
    },
  },
  endpoints: {
    ...widgetTagSlot.endpoints,
    setBox: (id: string) => `/crops/${id}/region`,
    clearBox: (id: string) => `/crops/${id}/region`,
  },
};

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
  it('widgetTagSlot: body carries exactly the region_* meta wire keys', async () => {
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
      region_text: 'TAG-001',
      region_rejection_reason: null,
    });
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

    await patchSlotMeta(widgetTagSlot, 'c1', { text: 'XYZ' });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ region_text: 'XYZ' });
  });

  it('a slot with no matching capability produces an empty body rather than throwing', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ crop_id: 'c1', updated_fields: [], item: rawItem('c1') }),
      );
    vi.stubGlobal('fetch', fetchMock);
    const textOnlySpec = {
      ...widgetTagSlot,
      capabilities: { text: widgetTagSlot.capabilities.text },
    };
    await patchSlotMeta(textOnlySpec, 'c1', { status: 'detected' });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({});
  });
});

describe('setSlotBox', () => {
  it('PUTs the bbox to the slot-declared endpoint, defaulting frame to source', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') }));
    vi.stubGlobal('fetch', fetchMock);

    const item = await setSlotBox(singleBoxSlot, 'c1', [0.1, 0.1, 0.2, 0.2]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/region`);
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({
      region_bbox_norm: [0.1, 0.1, 0.2, 0.2],
      frame: 'source',
    });
    // Unwraps {..., item} into the mapped Crop rather than returning the
    // wrapper — the write path renders this, not the raw response.
    expect(item.id).toBe('c1');
  });

  it('passes frame: "parent" through when the caller drew in the parent-crop frame', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') }));
    vi.stubGlobal('fetch', fetchMock);

    await setSlotBox(singleBoxSlot, 'c1', [0.1, 0.1, 0.2, 0.2], 'parent');

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      region_bbox_norm: [0.1, 0.1, 0.2, 0.2],
      frame: 'parent',
    });
  });

  it('clearing (null) falls back to setBox when no distinct clearBox is declared', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ item: rawItem('c1') }));
    vi.stubGlobal('fetch', fetchMock);

    await setSlotBox(singleBoxSlot, 'c1', null);

    const [url, init] = fetchMock.mock.calls[0];
    // singleBoxSlot's clearBox and setBox are the same URL today.
    expect(url).toBe(`${API_PREFIX}/crops/c1/region`);
    expect(JSON.parse(init.body)).toEqual({ region_bbox_norm: null, frame: 'source' });
  });

  it('rejects when the slot declares no setBox endpoint', async () => {
    const noEndpointSpec = { ...widgetTagSlot, endpoints: {} };
    await expect(setSlotBox(noEndpointSpec, 'c1', null)).rejects.toThrow(/no setBox/);
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
