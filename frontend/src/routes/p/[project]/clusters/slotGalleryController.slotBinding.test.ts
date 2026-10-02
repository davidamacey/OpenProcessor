/**
 * The gallery controller is bound to the slot it is created with
 * (domain-neutral audit H1-H6). It used to import one specific profile, so
 * a second slot's class filter showed that profile's data: every browse
 * went to its `browsePath`, bulk status writes went through its spec, and
 * saved boxes were read from its slot key.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { SlotData, SlotSpec } from '$lib/annotations/types';
import type { Crop } from '$lib/types';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegions: vi.fn(), batchRegionStatus: vi.fn() };
});
import { batchRegionStatus, getRegions } from '$lib/api';
import { createSlotGalleryController } from './slotGalleryController.svelte';

const gadgetMarkSlot: SlotSpec = {
  ...widgetTagSlot,
  key: 'gadget_mark',
  bind: { className: 'gadget_mark' },
  capabilities: {
    ...widgetTagSlot.capabilities,
    queue: { ...widgetTagSlot.capabilities.queue!, browsePath: '/gadget_marks' },
  },
};

afterEach(() => {
  vi.mocked(getRegions).mockReset();
  vi.mocked(batchRegionStatus).mockReset();
});

describe('createSlotGalleryController(slot)', () => {
  it("browses the given slot's own browsePath", async () => {
    vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
    const gallery = createSlotGalleryController(gadgetMarkSlot);
    await gallery.loadFirst();

    expect(getRegions).toHaveBeenCalledTimes(1);
    expect(vi.mocked(getRegions).mock.calls[0][0]).toBe('/gadget_marks');
  });

  it('writes bulk status through the given slot', async () => {
    vi.mocked(batchRegionStatus).mockResolvedValue({
      updated: 1,
      conflicts: [],
      invalid: [],
      items: [],
    } as never);
    const gallery = createSlotGalleryController(gadgetMarkSlot);
    await gallery.applyStatus(['w1'], 'detected');

    expect(batchRegionStatus).toHaveBeenCalledTimes(1);
    expect(vi.mocked(batchRegionStatus).mock.calls[0][0]).toBe(gadgetMarkSlot);
  });

  it("patches a saved box from the given slot's key on the returned item", () => {
    const gallery = createSlotGalleryController(gadgetMarkSlot);
    const data: SlotData = {
      key: gadgetMarkSlot.key,
      subBoxes: [makeSlotBox({ boxId: 'b1', rawXyxy: [0.1, 0.2, 0.3, 0.4] })],
      lifecycle: {
        status: 'detected',
        state: null,
        verified: true,
        validated: true,
        autoConfirmed: null,
        rejectionReason: null,
      },
    };
    const item = { id: 'w1', slots: { [gadgetMarkSlot.key]: data } } as unknown as Crop;
    gallery.pager.items = [{ crop_id: 'w1', region_status: null } as never];
    gallery.editCrop = item;
    gallery.saveBox(item);

    expect(gallery.pager.items[0].region_status).toBe('detected');
    expect(
      gallery.pager.items[0].slots?.[gadgetMarkSlot.key]?.subBoxes?.[0].rawXyxy,
    ).toEqual([0.1, 0.2, 0.3, 0.4]);
  });
});
