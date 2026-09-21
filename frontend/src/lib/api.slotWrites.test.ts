/**
 * setSlotBox / patchSlotMeta (C6, docs/design/slot-generic-crop-mapping-
 * plan-2026-09-21.md §6.3) — the generic write surface /review's inline
 * panel now uses instead of the deleted setCropPlate-only PlateMetaPatch
 * union. The equivalence proof: for licensePlateSlot, patchSlotMeta's
 * request body is byte-identical to what the old PlateMetaPatch produced
 * (plate_text / plate_status / plate_rejection_reason).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { setSlotBox, patchSlotMeta, API_PREFIX } from './api';
import { licensePlateSlot } from './annotations/profiles/licensePlate';

function okResponse(body: unknown = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('patchSlotMeta', () => {
  it('licensePlateSlot: body is byte-identical to the old PlateMetaPatch shape', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(okResponse({ crop_id: 'c1', updated_fields: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await patchSlotMeta(licensePlateSlot, 'c1', {
      status: 'detected',
      text: 'ABC123',
      rejectionReason: null,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/region_meta`);
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body)).toEqual({
      plate_status: 'detected',
      plate_text: 'ABC123',
      plate_rejection_reason: null,
    });
  });

  it('omits keys whose value is undefined (leaves them untouched, per the old contract)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(okResponse({ crop_id: 'c1', updated_fields: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await patchSlotMeta(licensePlateSlot, 'c1', { text: 'XYZ' });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ plate_text: 'XYZ' });
  });

  it('a slot with no matching capability produces an empty body rather than throwing', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(okResponse({ crop_id: 'c1', updated_fields: [] }));
    vi.stubGlobal('fetch', fetchMock);
    const textOnlySpec = {
      ...licensePlateSlot,
      capabilities: { text: licensePlateSlot.capabilities.text },
    };
    await patchSlotMeta(textOnlySpec, 'c1', { status: 'detected' });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({});
  });
});

describe('setSlotBox', () => {
  it('PUTs the bbox to the slot-declared endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ id: 'c1' }));
    vi.stubGlobal('fetch', fetchMock);

    await setSlotBox(licensePlateSlot, 'c1', [0.1, 0.1, 0.2, 0.2]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/region`);
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({ bbox_norm: [0.1, 0.1, 0.2, 0.2] });
  });

  it('clearing (null) falls back to setBox when no distinct clearBox is declared', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ id: 'c1' }));
    vi.stubGlobal('fetch', fetchMock);

    await setSlotBox(licensePlateSlot, 'c1', null);

    const [url, init] = fetchMock.mock.calls[0];
    // licensePlateSlot's clearBox and setBox are the same URL today.
    expect(url).toBe(`${API_PREFIX}/crops/c1/region`);
    expect(JSON.parse(init.body)).toEqual({ bbox_norm: null });
  });

  it('rejects when the slot declares no setBox endpoint', async () => {
    const noEndpointSpec = { ...licensePlateSlot, endpoints: {} };
    await expect(setSlotBox(noEndpointSpec, 'c1', null)).rejects.toThrow(/no setBox/);
  });
});
