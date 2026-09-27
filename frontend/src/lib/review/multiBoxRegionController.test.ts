import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    putRegionBoxes: vi.fn(),
    patchRegionBox: vi.fn(),
  };
});

import { putRegionBoxes, patchRegionBox } from '$lib/api';
import { createMultiBoxRegionController } from './multiBoxRegionController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';
import type { Crop } from '$lib/types';
import type { SlotData } from '$lib/annotations/types';

function cropWithBoxes(id: string, subBoxes: SlotData['subBoxes']): Crop {
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
    slots: { [widgetTagSlot.key]: { key: widgetTagSlot.key, subBoxes } },
  } as unknown as Crop;
}

const boxA: NonNullable<SlotData['subBoxes']>[number] = {
  boxId: 'b1',
  state: 'proposed',
  rawXyxy: [0.1, 0.1, 0.2, 0.2],
  parent: { cx: 0.15, cy: 0.15, w: 0.1, h: 0.1 },
  score: 0.9,
  detector: 'tag_detector_v1',
  detectorVersion: '1',
  source: 'detector',
  bboxCorrect: null,
  confidence: null,
  rejectionReason: null,
  text: null,
  clusterId: null,
  thumbnailUrl: null,
};
const boxB: NonNullable<SlotData['subBoxes']>[number] = {
  ...boxA,
  boxId: 'b2',
  state: 'rejected',
  parent: { cx: 0.6, cy: 0.6, w: 0.1, h: 0.1 },
};

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
    vi.mocked(patchRegionBox).mockResolvedValue(returned);
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    await c.acceptSelected('c1');
    expect(patchRegionBox).toHaveBeenCalledWith('c1', 'b1', { state: 'accepted' });
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
    vi.mocked(putRegionBoxes).mockResolvedValue(returned);
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(cropWithBoxes('c1', [boxA, boxB]));
    const result = await c.confirmAndSave('c1');
    expect(result.ok).toBe(true);
    expect(putRegionBoxes).toHaveBeenCalledWith(
      'c1',
      [{ box_id: 'b1', state: 'accepted' }, { box_id: 'b2' }],
      { regionStatus: 'detected' },
    );
  });

  it('confirmAndSave sends a moved box AND a new box in the same write', async () => {
    const returned = cropWithBoxes('c1', [{ ...boxA, state: 'accepted' }]);
    vi.mocked(putRegionBoxes).mockResolvedValue(returned);
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
    vi.mocked(putRegionBoxes).mockResolvedValue(returned);
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
      vi.mocked(putRegionBoxes).mockResolvedValue(returned);
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      c.moveSelected({ cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 });
      await c.saveEdits('c1');
      expect(putRegionBoxes).toHaveBeenCalledTimes(1);
      const [cropId, body, opts] = vi.mocked(putRegionBoxes).mock.calls[0];
      expect(cropId).toBe('c1');
      expect(body[0]).toMatchObject({ box_id: 'b1' });
      // The key differentiator from confirmAndSave: no region_status.
      expect(opts).toBeUndefined();
    });

    it('never promotes a proposed box to accepted (no confirm semantics)', async () => {
      const returned = cropWithBoxes('c1', [{ ...boxA, state: 'proposed' }]);
      vi.mocked(putRegionBoxes).mockResolvedValue(returned);
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      c.seedFrom(cropWithBoxes('c1', [boxA]));
      c.moveSelected({ cx: 0.3, cy: 0.3, w: 0.1, h: 0.1 });
      await c.saveEdits('c1');
      const [, body] = vi.mocked(putRegionBoxes).mock.calls[0];
      expect(body[0]).not.toHaveProperty('state');
    });

    it('records a region undo entry and reports the new item on success', async () => {
      const returned = cropWithBoxes('c1', [{ ...boxA, state: 'proposed' }]);
      vi.mocked(putRegionBoxes).mockResolvedValue(returned);
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

  describe('maxBoxes', () => {
    it('reads the served subBox.maxBoxesPerWrite, or null when absent (pre-W8.8 backend)', () => {
      const c = createMultiBoxRegionController(() => widgetTagSlot);
      expect(c.maxBoxes).toBeNull();

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
