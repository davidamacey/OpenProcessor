/**
 * The gallery's bulk-status buttons read confirm / reject / false-positive
 * states from the served `GET {API_PREFIX}/regions/statuses` vocabulary
 * (`regionStatusesStore`) first, and fall back to the controller's own
 * slot lifecycle only when the store hasn't loaded (or the endpoint 404s),
 * the same degrade contract `regionStatuses.svelte.ts` documents. The
 * states come from the slot the controller was created with, never from
 * a specific profile.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { createSlotGalleryController } from './slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { SlotSpec } from '$lib/annotations/types';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';

afterEach(() => {
  regionStatusesStore.confirmStatus = null;
  regionStatusesStore.rejectStatus = null;
  regionStatusesStore.falsePositiveStatus = null;
});

describe('slotGalleryController lifecycle states', () => {
  it("fall back to the controller's own slot when the served vocabulary has not loaded", () => {
    const slot: SlotSpec = {
      ...widgetTagSlot,
      capabilities: {
        ...widgetTagSlot.capabilities,
        lifecycle: {
          ...widgetTagSlot.capabilities.lifecycle!,
          confirmState: 'slot_confirm',
          rejectState: 'slot_reject',
          falsePositiveState: 'slot_fp',
        },
      },
    };
    const gallery = createSlotGalleryController(slot);
    expect(gallery.slot).toBe(slot);
    expect(gallery.confirmState()).toBe('slot_confirm');
    expect(gallery.rejectState()).toBe('slot_reject');
    expect(gallery.falsePositiveState()).toBe('slot_fp');
  });

  it('prefer the served regionStatusesStore values once loaded, over the slot', () => {
    regionStatusesStore.confirmStatus = 'served_confirm';
    regionStatusesStore.rejectStatus = 'served_reject';
    regionStatusesStore.falsePositiveStatus = 'served_fp';

    const gallery = createSlotGalleryController(widgetTagSlot);
    expect(gallery.confirmState()).toBe('served_confirm');
    expect(gallery.rejectState()).toBe('served_reject');
    expect(gallery.falsePositiveState()).toBe('served_fp');
  });

  it('has no false-positive state for a slot that declares none', () => {
    const slot: SlotSpec = {
      ...widgetTagSlot,
      capabilities: {
        ...widgetTagSlot.capabilities,
        lifecycle: {
          ...widgetTagSlot.capabilities.lifecycle!,
          falsePositiveState: undefined,
        },
      },
    };
    expect(createSlotGalleryController(slot).falsePositiveState()).toBeUndefined();
  });
});
