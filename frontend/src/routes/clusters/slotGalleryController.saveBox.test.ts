/**
 * C8 (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §7.1):
 * `saveBox` used to re-PUT the box via `setCropPlate` even though
 * `SlotBboxEditor` had already saved it via `setSlotBox` — a redundant
 * double-write on every plate-gallery bbox save. It is now a pure local
 * patch off the server's own returned item: no network call, no `fetch`
 * stub needed, which is itself part of the proof (a lingering
 * `setCropPlate` call would require one), and no client-side
 * confirmed-vs-rejected derivation — it renders `item.slots.license_plate`
 * verbatim.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createSlotGalleryController } from './slotGalleryController.svelte';
import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';
import type { Crop } from '$lib/types';
import type { SlotData } from '$lib/annotations/types';
import type { RegionBrowseItem } from '$lib/api';

function fakeCropWithSlot(id: string, slot: SlotData): Crop {
  return {
    id,
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 3,
    class_name: 'license_plate',
    class_source: null,
    label_source: 'human',
    label_validated: true,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    test_holdout: false,
    updated_at: '',
    slots: { [licensePlateSlot.key]: slot },
  } as Crop;
}

function fakeRegionItem(id: string): RegionBrowseItem {
  return {
    crop_id: id,
    id,
    image_path: '/img.jpg',
    bbox_norm: [0.3, 0.3, 0.7, 0.7],
    region_bbox_norm: null,
    region_score: null,
    region_status: 'pending_verification',
    region_verified: false,
    region_validated: null,
    region_detector: null,
    region_detector_version: null,
    region_detector_chain: null,
    region_bbox_frame: null,
    region_detected_at: null,
    region_verifier: null,
    region_verifier_version: null,
    region_verified_at: null,
    region_rejection_reason: null,
    region_visible: null,
    region_text: null,
    region_text_source: null,
    region_text_confidence: null,
    class_id: 3,
    class_name: 'license_plate',
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

    const gallery = createSlotGalleryController();
    gallery.editCrop = fakeCropWithSlot('c1', {
      key: licensePlateSlot.key,
      subBox: {
        rawXyxy: [0.4, 0.45, 0.6, 0.55],
        frame: 'source',
        parent: { cx: 0.5, cy: 0.5, w: 0.2, h: 0.1 },
        score: null,
        visible: true,
        candidate: null,
      },
      lifecycle: {
        status: 'detected',
        state: null,
        verified: true,
        validated: true,
        autoConfirmed: null,
        rejectionReason: null,
        boxCorrect: null,
      },
    });
    gallery.pager.items = [fakeRegionItem('c1')];

    gallery.saveBox(gallery.editCrop);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(gallery.editCrop).toBeNull();
    const patched = gallery.pager.items.find((p) => p.crop_id === 'c1');
    expect(patched?.region_status).toBe('detected');
    expect(patched?.region_verified).toBe(true);
    expect(patched?.region_bbox_norm).toEqual([0.4, 0.45, 0.6, 0.55]);
  });

  it('a cleared item (no subBox.rawXyxy) patches to the rejectState and clears the bbox', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const gallery = createSlotGalleryController();
    gallery.editCrop = fakeCropWithSlot('c2', {
      key: licensePlateSlot.key,
      subBox: {
        rawXyxy: null,
        frame: 'source',
        parent: null,
        score: null,
        visible: null,
        candidate: null,
      },
      lifecycle: {
        status: 'no_region_visible',
        state: null,
        verified: null,
        validated: null,
        autoConfirmed: null,
        rejectionReason: null,
        boxCorrect: null,
      },
    });
    gallery.pager.items = [fakeRegionItem('c2')];

    gallery.saveBox(gallery.editCrop);

    expect(fetchMock).not.toHaveBeenCalled();
    const patched = gallery.pager.items.find((p) => p.crop_id === 'c2');
    expect(patched?.region_status).toBe('no_region_visible');
    expect(patched?.region_bbox_norm).toBeNull();
  });
});
