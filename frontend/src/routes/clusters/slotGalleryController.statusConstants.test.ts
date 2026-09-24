/**
 * C4b (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §7.3):
 * slotGalleryController.svelte.ts and SlotGallery.svelte used to
 * hardcode 'false_positive' | 'no_plate_visible' | 'detected' as raw
 * lifecycle-state literals — a state literal that silently means
 * nothing for another slot, and (per the plan) the highest-risk class
 * of straggler in this codebase.
 *
 * m9 (2026-09-24 interactive pass): PLATE_CONFIRM_STATE / etc. were
 * `export const`s derived from licensePlateSlot alone — the plate
 * gallery's bulk-status buttons ignored the served
 * `GET {API_PREFIX}/regions/statuses` vocabulary the review tab already
 * reads. They're now functions that prefer `regionStatusesStore`'s
 * loaded value and fall back to the profile literal only when the
 * store hasn't loaded (or the endpoint 404s) — same degrade contract
 * `regionStatuses.svelte.ts`'s own doc comment describes.
 */
import { afterEach, describe, expect, it } from 'vitest';
import {
  PLATE_CONFIRM_STATE,
  PLATE_REJECT_STATE,
  PLATE_FALSE_POSITIVE_STATE,
} from './slotGalleryController.svelte';
import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

afterEach(() => {
  regionStatusesStore.confirmStatus = null;
  regionStatusesStore.rejectStatus = null;
  regionStatusesStore.falsePositiveStatus = null;
});

describe('slotGalleryController status constants', () => {
  it('fall back to licensePlateSlot when the served vocabulary has not loaded', () => {
    const lifecycle = licensePlateSlot.capabilities.lifecycle!;
    expect(PLATE_CONFIRM_STATE()).toBe(lifecycle.confirmState);
    expect(PLATE_REJECT_STATE()).toBe(lifecycle.rejectState);
    expect(PLATE_FALSE_POSITIVE_STATE()).toBe(lifecycle.falsePositiveState);
  });

  it('prefer the served regionStatusesStore values once loaded, over the static profile', () => {
    regionStatusesStore.confirmStatus = 'served_confirm';
    regionStatusesStore.rejectStatus = 'served_reject';
    regionStatusesStore.falsePositiveStatus = 'served_fp';

    expect(PLATE_CONFIRM_STATE()).toBe('served_confirm');
    expect(PLATE_REJECT_STATE()).toBe('served_reject');
    expect(PLATE_FALSE_POSITIVE_STATE()).toBe('served_fp');
  });
});
