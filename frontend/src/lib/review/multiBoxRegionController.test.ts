import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    putRegionBoxes: vi.fn(),
    patchRegionBox: vi.fn(),
  };
});

import { putRegionBoxes, patchRegionBox, ApiError } from '$lib/api';
import { createMultiBoxRegionController } from './multiBoxRegionController.svelte';
import {
  WIDGET_TAG_PROFILE,
  widgetTagServedSlot,
  widgetTagSlot,
} from '$lib/test/fixtures/regionSlot';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';
import type { Crop } from '$lib/types';
import type { VectorRefresh } from '$lib/types_itemFilter';
import type { SlotData } from '$lib/annotations/types';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';

function cropWithBoxes(
  id: string,
  subBoxes: SlotData['subBoxes'],
  revision: number | null = null,
): Crop {
  return {
    id,
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 1,
    class_name: 'widget_tag',
    class_source: null,
    label_source: 'model',
    label_validated: false,
    class_validated: false,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    test_holdout: false,
    updated_at: '',
    slots: {
      [widgetTagSlot.key]: {
        key: widgetTagSlot.key,
        subBoxes,
        boxSet: {
          count: null,
          rejectedCount: null,
          maxScore: null,
          setComplete: null,
          revision,
        },
      },
    },
  } as unknown as Crop;
}

function written(crop: Crop, vectorRefresh: VectorRefresh | null = null) {
  return { crop, vectorRefresh };
}

const boxA = makeSlotBox({
  boxId: 'b1',
  rawXyxy: [0.1, 0.1, 0.2, 0.2],
  parent: { cx: 0.15, cy: 0.15, w: 0.1, h: 0.1 },
});
const boxB = makeSlotBox({
  boxId: 'b2',
  state: 'rejected',
  parent: { cx: 0.6, cy: 0.6, w: 0.1, h: 0.1 },
});

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

describe('createMultiBoxRegionController', () => {
  it('seeds boxes from the crop and selects the first one', () => {
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    expect(c.boxes).toHaveLength(2);
    expect(c.selectedIndex).toBe(0);
  });

  it('Tab cycles the selection', () => {
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    c.next();
    expect(c.selectedIndex).toBe(1);
    c.next();
    expect(c.selectedIndex).toBe(0);
  });

  it('addBox appends and selects the new box, marking the set dirty', () => {
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA]));
    c.addBox({ cx: 0.8, cy: 0.8, w: 0.1, h: 0.1 });
    expect(c.boxes).toHaveLength(2);
    expect(c.selectedIndex).toBe(1);
    expect(c.dirty).toBe(true);
  });

  it('deleteSelected removes the box and moves selection', () => {
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    c.deleteSelected();
    expect(c.boxes.map((b) => b.boxId)).toEqual(['b2']);
    expect(c.dirty).toBe(true);
  });

  it('acceptSelected PATCHes the selected box and reseeds from the server item', async () => {
    const returned = cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }, boxB]);
    vi.mocked(patchRegionBox).mockResolvedValue(written(returned));
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    await c.acceptSelected('c1');
    expect(patchRegionBox).toHaveBeenCalledWith('c1', 'b1', {
      state: 'accepted',
      expectedRegionRevision: undefined,
    });
    expect(c.boxes[0].state).toBe('accepted');
  });

  it('rejectSelected on a not-yet-saved local box flips state without a PATCH', async () => {
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA]));
    c.addBox({ cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 });
    await c.rejectSelected('c1');
    expect(patchRegionBox).not.toHaveBeenCalled();
    expect(c.boxes[1].state).toBe('rejected');
  });

  it('confirmAndSave sends only proposed->accepted and untouched siblings, with region_status', async () => {
    const returned = cropWithBoxes('c1', [
      { ...boxA, state: 'accepted' },
      boxB, // still rejected — a whole-set confirm never overrides a per-box decision
    ]);
    vi.mocked(putRegionBoxes).mockResolvedValue(written(returned));
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    const result = await c.confirmAndSave('c1');
    expect(result.ok).toBe(true);
    expect(putRegionBoxes).toHaveBeenCalledWith(
      'c1',
      [{ box_id: 'b1', state: 'accepted' }, { box_id: 'b2' }],
      { regionStatus: 'detected', expectedRegionRevision: undefined },
    );
  });

  it('confirmAndSave sends a moved box AND a new box in the same write', async () => {
    const returned = cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }]);
    vi.mocked(putRegionBoxes).mockResolvedValue(written(returned));
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA]));
    c.moveSelected({ cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 });
    c.addBox({ cx: 0.9, cy: 0.9, w: 0.05, h: 0.05 });
    await c.confirmAndSave('c1');
    const [, body] = vi.mocked(putRegionBoxes).mock.calls[0];
    expect(body).toHaveLength(2);
    expect(body[0]).toMatchObject({ box_id: 'b1', state: 'accepted' });
    expect((body[0] as { bbox_norm: unknown }).bbox_norm).toBeDefined();
    expect(body[1]).toMatchObject({ box_id: null });
  });

  it('records a region undo entry on every successful write', async () => {
    const returned = cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }]);
    vi.mocked(putRegionBoxes).mockResolvedValue(written(returned));
    const spy = vi.spyOn(undoStore, 'recordRegionWrites');
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA]));
    await c.confirmAndSave('c1');
    expect(spy).toHaveBeenCalledWith(['c1']);
  });

  it('confirmAndSave surfaces a server rejection (e.g. no_accepted_box) without throwing', async () => {
    vi.mocked(putRegionBoxes).mockRejectedValue(new Error('no_accepted_box'));
    const toastSpy = vi.spyOn(toastStore, 'error');
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxB]));
    const result = await c.confirmAndSave('c1');
    expect(result.ok).toBe(false);
    expect(toastSpy).toHaveBeenCalled();
  });

  describe('saveEdits (SlotBboxEditor modal — no confirm semantics)', () => {
    it('sends the geometry diff with no region_status key, unlike confirmAndSave', async () => {
      const returned = cropWithBoxes('c1', [{ ...boxA, state: 'proposed' }]);
      vi.mocked(putRegionBoxes).mockResolvedValue(written(returned));
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      c.moveSelected({ cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 });
      await c.saveEdits('c1');
      expect(putRegionBoxes).toHaveBeenCalledTimes(1);
      const [cropId, body, opts] = vi.mocked(putRegionBoxes).mock.calls[0];
      expect(cropId).toBe('c1');
      expect(body[0]).toMatchObject({ box_id: 'b1' });
      // The key differentiator from confirmAndSave: no region_status.
      expect(opts?.regionStatus).toBeUndefined();
    });

    it('never promotes a proposed box to accepted (no confirm semantics)', async () => {
      const returned = cropWithBoxes('c1', [{ ...boxA, state: 'proposed' }]);
      vi.mocked(putRegionBoxes).mockResolvedValue(written(returned));
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      c.moveSelected({ cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 });
      await c.saveEdits('c1');
      const [, body] = vi.mocked(putRegionBoxes).mock.calls[0];
      expect(body[0]).not.toHaveProperty('state');
    });

    it('records a region undo entry and reports the new item on success', async () => {
      const returned = cropWithBoxes('c1', [{ ...boxA, state: 'proposed' }]);
      vi.mocked(putRegionBoxes).mockResolvedValue(written(returned));
      const spy = vi.spyOn(undoStore, 'recordRegionWrites');
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      c.moveSelected({ cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 });
      const result = await c.saveEdits('c1');
      expect(spy).toHaveBeenCalledWith(['c1']);
      expect(result.ok).toBe(true);
      expect(result.item).toBe(returned);
    });

    it('surfaces a server failure without throwing', async () => {
      vi.mocked(putRegionBoxes).mockRejectedValue(new Error('region_conflict'));
      const toastSpy = vi.spyOn(toastStore, 'error');
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      const result = await c.saveEdits('c1');
      expect(result.ok).toBe(false);
      expect(toastSpy).toHaveBeenCalled();
    });
  });

  describe('region revision (optimistic concurrency)', () => {
    it('echoes the served region_revision as expected_region_revision on every write kind', async () => {
      vi.mocked(patchRegionBox).mockResolvedValue(
        written(cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }], 5)),
      );
      vi.mocked(putRegionBoxes).mockResolvedValue(
        written(cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }], 6)),
      );
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA], 4));
      expect(c.revision).toBe(4);
      await c.acceptSelected('c1');
      expect(vi.mocked(patchRegionBox).mock.calls[0][2]).toMatchObject({
        expectedRegionRevision: 4,
      });
      // The write's returned revision is what the next write echoes.
      expect(c.revision).toBe(5);
      await c.confirmAndSave('c1');
      expect(vi.mocked(putRegionBoxes).mock.calls[0][2]).toMatchObject({
        expectedRegionRevision: 5,
      });
      expect(c.revision).toBe(6);
      await c.saveEdits('c1');
      expect(vi.mocked(putRegionBoxes).mock.calls[1][2]).toMatchObject({
        expectedRegionRevision: 6,
      });
    });

    it('adopts the server item on a stale revision instead of a generic error', async () => {
      // The conflict body carries a raw wire item; mapRawCrop maps its
      // slots through the registry, so the served region slot is installed.
      installServedRegionProfile(WIDGET_TAG_PROFILE);
      const wireBox = (id: string, state: string, at: number) => ({
        box_id: id,
        state,
        bbox_norm: [at, at, at + 0.1, at + 0.1],
        bbox_in_parent: [at, at, at + 0.1, at + 0.1],
      });
      vi.mocked(putRegionBoxes).mockRejectedValue(
        new ApiError(409, '/regions', {
          detail: {
            error: 'region_conflict',
            current_region_revision: 9,
            current_box_ids: ['b1', 'b2'],
            item: {
              crop_id: 'c1',
              image_path: '/img.jpg',
              bbox_norm: [0, 0, 1, 1],
              region_revision: 9,
              region_boxes: [
                wireBox('b1', 'accepted', 0.1),
                wireBox('b2', 'rejected', 0.5),
              ],
            },
          },
        }),
      );
      const warn = vi.spyOn(toastStore, 'warn');
      const err = vi.spyOn(toastStore, 'error');
      const seen: Crop[] = [];
      const c = createMultiBoxRegionController(() => widgetTagServedSlot, {
        onitem: (crop) => seen.push(crop),
      });
      c.seedFrom(cropWithBoxes('c1', [boxA], 4));
      const result = await c.confirmAndSave('c1');
      resetDeploymentSlots();
      expect(result.ok).toBe(false);
      expect(result.item?.id).toBe('c1');
      expect(warn).toHaveBeenCalled();
      expect(err).not.toHaveBeenCalled();
      expect(c.boxes.map((b) => b.boxId)).toEqual(['b1', 'b2']);
      expect(c.revision).toBe(9);
      expect(seen).toHaveLength(1);
    });

    it('a non-conflict 409 is still a plain error', async () => {
      vi.mocked(patchRegionBox).mockRejectedValue(
        new ApiError(409, '/regions', { detail: 'nope' }),
      );
      const err = vi.spyOn(toastStore, 'error');
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA], 4));
      await c.acceptSelected('c1');
      expect(err).toHaveBeenCalled();
    });

    it('reports every returned item through onitem', async () => {
      const returned = cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }], 5);
      vi.mocked(patchRegionBox).mockResolvedValue(written(returned));
      const onitem = vi.fn();
      const c = createMultiBoxRegionController(() => widgetTagSlot, { onitem });
      c.seedFrom(cropWithBoxes('c1', [boxA], 4));
      await c.acceptSelected('c1');
      expect(onitem).toHaveBeenCalledWith(returned);
    });
  });

  describe('setSelectedText (a reading is per box)', () => {
    it('PATCHes the selected stored box with the text and the revision', async () => {
      vi.mocked(patchRegionBox).mockResolvedValue(
        written(cropWithBoxes('c1', [{ ...boxA, text: 'TAG-002' }], 5)),
      );
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA, boxB], 4));
      c.select(1);
      await c.setSelectedText('c1', 'TAG-002');
      expect(patchRegionBox).toHaveBeenCalledWith('c1', 'b2', {
        text: 'TAG-002',
        expectedRegionRevision: 4,
      });
    });

    it('clears a reading with null (not undefined)', async () => {
      vi.mocked(patchRegionBox).mockResolvedValue(
        written(cropWithBoxes('c1', [boxA], 5)),
      );
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA], 4));
      await c.setSelectedText('c1', null);
      expect(vi.mocked(patchRegionBox).mock.calls[0][2]).toHaveProperty('text', null);
    });

    it('refuses a not-yet-saved local box with a warning and no request', async () => {
      const warn = vi.spyOn(toastStore, 'warn');
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      c.addBox({ cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 });
      await c.setSelectedText('c1', 'X');
      expect(patchRegionBox).not.toHaveBeenCalled();
      expect(warn).toHaveBeenCalled();
    });
  });

  describe('maxBoxes', () => {
    it('reads the served subBox.maxBoxesPerWrite, or null when the slot declares none', () => {
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      expect(c.maxBoxes).toBe(500);

      const none = {
        ...widgetTagSlot,
        capabilities: {
          ...widgetTagSlot.capabilities,
          subBox: { ...widgetTagSlot.capabilities.subBox!, maxBoxesPerWrite: undefined },
        },
      };
      expect(createMultiBoxRegionController(() => none).maxBoxes).toBeNull();

      const limited = {
        ...widgetTagSlot,
        capabilities: {
          ...widgetTagSlot.capabilities,
          subBox: { ...widgetTagSlot.capabilities.subBox!, maxBoxesPerWrite: 5 },
        },
      };
      const c2 = createMultiBoxRegionController(() => limited);
      expect(c2.maxBoxes).toBe(5);
    });

    it('returns null when there is no active slot', () => {
      const c = createMultiBoxRegionController(() => null);
      expect(c.maxBoxes).toBeNull();
    });
  });
});

describe('vector_refresh (boxes without a vector)', () => {
  it('keeps the last served value from a write and exposes it', async () => {
    const returned = cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }, boxB]);
    vi.mocked(patchRegionBox).mockResolvedValue(
      written(returned, { embedded: 1, pending: 2 }),
    );
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    expect(c.vectorRefresh).toBeNull();
    await c.acceptSelected('c1');
    expect(c.vectorRefresh).toEqual({ embedded: 1, pending: 2 });
  });

  it('a later write replaces it, and a null (not served) clears it', async () => {
    const returned = cropWithBoxes('c1', [boxA, boxB]);
    vi.mocked(putRegionBoxes).mockResolvedValueOnce(
      written(returned, { embedded: 0, pending: 3 }),
    );
    vi.mocked(putRegionBoxes).mockResolvedValueOnce(written(returned, null));
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    await c.saveEdits('c1');
    expect(c.vectorRefresh?.pending).toBe(3);
    await c.saveEdits('c1');
    expect(c.vectorRefresh).toBeNull();
  });

  it('survives a re-seed of the same crop (the page re-seeds after its own write)', async () => {
    const returned = cropWithBoxes('c1', [boxA, boxB]);
    vi.mocked(putRegionBoxes).mockResolvedValue(
      written(returned, { embedded: 0, pending: 1 }),
    );
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    await c.saveEdits('c1');
    c.seedFrom(returned);
    expect(c.vectorRefresh).toEqual({ embedded: 0, pending: 1 });
  });

  it('is cleared when the next item is seeded', async () => {
    const returned = cropWithBoxes('c1', [boxA, boxB]);
    vi.mocked(putRegionBoxes).mockResolvedValue(
      written(returned, { embedded: 0, pending: 1 }),
    );
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    await c.saveEdits('c1');
    expect(c.vectorRefresh).not.toBeNull();
    c.seedFrom(cropWithBoxes('c2', [boxA]));
    expect(c.vectorRefresh).toBeNull();
  });
});
