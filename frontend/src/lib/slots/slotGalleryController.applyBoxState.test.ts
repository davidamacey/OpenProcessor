/**
 * Region-cluster bulk triage: "Triage from a cluster
 * goes through batch_box_state, never the item-level batch_status, which
 * would flip every sibling box." `applyBoxState` is the controller's
 * per-box triage path, distinct from `applyStatus` (item-level
 * region_status, used outside a selected cluster).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api_regionBoxes', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_regionBoxes')>('$lib/api_regionBoxes');
  return { ...actual, postBatchBoxState: vi.fn() };
});
vi.mock('$lib/api_regions', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_regions')>('$lib/api_regions');
  return { ...actual, getRegions: vi.fn() };
});
import { postBatchBoxState } from '$lib/api_regionBoxes';
import { getRegions } from '$lib/api_regions';
import { createSlotGalleryController } from './slotGalleryController.svelte';
import { undoStore } from '$stores/undo.svelte';

afterEach(() => {
  vi.mocked(getRegions).mockReset();
  vi.mocked(postBatchBoxState).mockReset();
});

describe('createSlotGalleryController: applyBoxState (per-box cluster triage)', () => {
  it('targets {cropId, boxId} pairs from the served region_box_id on each row, never the item id alone', async () => {
    vi.mocked(postBatchBoxState).mockResolvedValue({
      updated: 2,
      invalid: [],
      conflicts: [],
      items: [{ id: 'w1' }, { id: 'w2' }],
    } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.pager.items = [
      { crop_id: 'w1', region_box_id: 'b1', row_key: 'w1#b1' } as never,
      { crop_id: 'w2', region_box_id: 'b2', row_key: 'w2#b2' } as never,
    ];

    await gallery.applyBoxState(['w1#b1', 'w2#b2'], 'accepted');

    expect(postBatchBoxState).toHaveBeenCalledTimes(1);
    expect(vi.mocked(postBatchBoxState).mock.calls[0][0]).toEqual([
      { cropId: 'w1', boxId: 'b1' },
      { cropId: 'w2', boxId: 'b2' },
    ]);
    expect(vi.mocked(postBatchBoxState).mock.calls[0][1]).toBe('accepted');
  });

  it('skips a row with no served region_box_id rather than falling back to an item-level write', async () => {
    vi.mocked(postBatchBoxState).mockResolvedValue({
      updated: 1,
      invalid: [],
      conflicts: [],
      items: [{ id: 'w1' }],
    } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.pager.items = [
      { crop_id: 'w1', region_box_id: 'b1', row_key: 'w1#b1' } as never,
      { crop_id: 'w2', region_box_id: null, row_key: 'w2#item' } as never,
    ];

    await gallery.applyBoxState(['w1#b1', 'w2#item'], 'rejected');

    expect(vi.mocked(postBatchBoxState).mock.calls[0][0]).toEqual([
      { cropId: 'w1', boxId: 'b1' },
    ]);
  });

  it('never calls postBatchBoxState when no selected row has a box id (pre-W8 backend)', async () => {
    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.pager.items = [
      { crop_id: 'w1', region_box_id: null, row_key: 'w1#item' } as never,
    ];

    await gallery.applyBoxState(['w1#item'], 'false_positive');

    expect(postBatchBoxState).not.toHaveBeenCalled();
  });

  it("records the server's own returned item ids for undo, not the request's crop ids", async () => {
    vi.mocked(postBatchBoxState).mockResolvedValue({
      updated: 1,
      invalid: [],
      conflicts: [],
      items: [{ id: 'w1' }],
    } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.pager.items = [
      { crop_id: 'w1', region_box_id: 'b1', row_key: 'w1#b1' } as never,
    ];
    const spy = vi.spyOn(undoStore, 'recordRegionWrites');

    await gallery.applyBoxState(['w1#b1'], 'accepted');

    expect(spy).toHaveBeenCalledWith(['w1']);
    spy.mockRestore();
  });

  it('selecting one box row of a multi-box item targets only that box, never its siblings', async () => {
    vi.mocked(postBatchBoxState).mockResolvedValue({
      updated: 1,
      invalid: [],
      conflicts: [],
      items: [{ id: 'w1' }],
    } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.pager.items = [
      { crop_id: 'w1', region_box_id: 'b1', row_key: 'w1#b1' } as never,
      { crop_id: 'w1', region_box_id: 'b2', row_key: 'w1#b2' } as never,
    ];

    gallery.toggleSelect(gallery.pager.items[1]!);
    expect(gallery.sel.has('w1#b1')).toBe(false);
    await gallery.applyBoxState([...gallery.sel.ids], 'accepted');

    expect(vi.mocked(postBatchBoxState).mock.calls[0][0]).toEqual([
      { cropId: 'w1', boxId: 'b2' },
    ]);
  });
});
