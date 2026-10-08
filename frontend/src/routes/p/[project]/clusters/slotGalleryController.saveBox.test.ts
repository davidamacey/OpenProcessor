/**
 * C8 (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §7.1):
 * `saveBox` used to re-PUT the box a second time even though
 * `SlotBboxEditor` had already saved it (`PUT /crops/{id}/regions`) — a redundant
 * double-write on every region-gallery bbox save. It is now a pure local
 * patch off the server's own returned item: no network call, no `fetch`
 * stub needed, which is itself part of the proof (a lingering second
 * write would require one), and no client-side
 * confirmed-vs-rejected derivation — it renders `item.slots.widget_tag`
 * verbatim.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createSlotGalleryController } from './slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { Crop } from '$lib/types';
import type { SlotData } from '$lib/annotations/types';
import type { RegionBrowseItem } from '$lib/api';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';

function fakeCropWithSlot(id: string, slot: SlotData): Crop {
  return {
    id,
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 3,
    class_name: 'widget_tag',
    class_source: null,
    label_source: 'human',
    label_validated: true,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    test_holdout: false,
    updated_at: '',
    slots: { [widgetTagSlot.key]: slot },
  } as Crop;
}

function fakeRegionItem(id: string): RegionBrowseItem {
  return {
    crop_id: id,
    id,
    image_path: '/img.jpg',
    bbox_norm: [0.3, 0.3, 0.7, 0.7],
    region_status: 'pending_verification',
    region_verified: false,
    region_validated: null,
    region_detector_chain: null,
    region_detected_at: null,
    region_verifier: null,
    region_verifier_version: null,
    region_verified_at: null,
    region_rejection_reason: null,
    region_visible: null,
    class_id: 3,
    class_name: 'widget_tag',
    cluster_id: null,
    updated_at: '',
  } as RegionBrowseItem;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('saveBox — no redundant write', () => {
  it('patches the local pager item from the returned item and never calls fetch', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.editCrop = fakeCropWithSlot('c1', {
      key: widgetTagSlot.key,
      subBoxes: [
        makeSlotBox({
          boxId: 'b1',
          state: 'accepted',
          rawXyxy: [0.4, 0.45, 0.6, 0.55],
          thumbnailUrl: '/curation/crops/c1/region_thumbnail?box_id=b1',
        }),
      ],
      boxSet: {
        count: 1,
        rejectedCount: 0,
        maxScore: 0.9,
        setComplete: true,
        revision: 8,
      },
      lifecycle: {
        status: 'detected',
        state: null,
        verified: true,
        validated: true,
        autoConfirmed: null,
        rejectionReason: null,
      },
    });
    gallery.pager.items = [fakeRegionItem('c1')];

    gallery.saveBox(gallery.editCrop);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(gallery.editCrop).toBeNull();
    const patched = gallery.pager.items.find((p) => p.crop_id === 'c1');
    expect(patched?.region_status).toBe('detected');
    expect(patched?.region_verified).toBe(true);
    // The card re-renders from the returned boxes and revision (which is
    // also what re-crops its thumbnail).
    const data = patched?.slots?.[widgetTagSlot.key];
    expect(data?.subBoxes?.[0].rawXyxy).toEqual([0.4, 0.45, 0.6, 0.55]);
    expect(data?.boxSet?.revision).toBe(8);
  });

  it('a cleared item (no boxes) patches to the served status and empties the box list', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const gallery = createSlotGalleryController(widgetTagSlot);
    gallery.editCrop = fakeCropWithSlot('c2', {
      key: widgetTagSlot.key,
      subBoxes: [],
      boxSet: {
        count: 0,
        rejectedCount: 0,
        maxScore: null,
        setComplete: null,
        revision: 3,
      },
      lifecycle: {
        status: 'no_region_visible',
        state: null,
        verified: null,
        validated: null,
        autoConfirmed: null,
        rejectionReason: null,
      },
    });
    const before = fakeRegionItem('c2');
    before.slots = {
      [widgetTagSlot.key]: { key: widgetTagSlot.key, subBoxes: [makeSlotBox()] },
    };
    gallery.pager.items = [before];

    gallery.saveBox(gallery.editCrop);

    expect(fetchMock).not.toHaveBeenCalled();
    const patched = gallery.pager.items.find((p) => p.crop_id === 'c2');
    expect(patched?.region_status).toBe('no_region_visible');
    expect(patched?.slots?.[widgetTagSlot.key]?.subBoxes).toEqual([]);
  });
});
