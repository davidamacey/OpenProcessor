/**
 * W8 (docs/design/w8-multibox-frontend-plan-2026-09-26.md; supersedes the
 * pre-W8 dq-region "candidate" round trip this file originally covered —
 * a rejected box is just a `SlotBox` with `state: 'rejected'` now, no
 * separate candidate concept). `POST {API_PREFIX}/crops/{id}/region/undo`
 * (`undoCropRegion`) returns the post-undo item, mapped through
 * `mapRawCrop` -> `mapCropSlots` -> `readSlot` like any other crop — a
 * Z-undo that restores an earlier `region_boxes` list re-renders it from
 * the server's own response, with no client-side reconstruction. This is
 * the contract `review/+page.svelte`'s `undoLast()` relies on: whatever
 * this function returns is what `queue.items[idx]` gets replaced with
 * directly. The backend-confirmed W8 undo contract (owner resolution,
 * 2026-09-26): one call restores the whole prior `region_boxes` list plus
 * `region_status`/`region_revision` in one step.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { undoCropRegion, API_PREFIX } from './api';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from './annotations/registeredSlots';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  installDeploymentSlots([widgetTagSlot]);
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetDeploymentSlots();
});

describe('undoCropRegion — W8 multi-box round trip', () => {
  it('maps the restored region_boxes list, item status, and text choice onto Crop.slots', async () => {
    const raw = {
      crop_id: 'c1',
      image_path: '/nas/img.jpg',
      bbox_norm: [0, 0, 0.4, 0.2],
      region_boxes: [
        {
          box_id: 'b1',
          state: 'rejected',
          bbox_norm: [0.1, 0.02, 0.3, 0.06],
          bbox_in_parent: [0.1, 0.02, 0.3, 0.06],
          score: 0.55,
          detector: 'tag_detector_v1',
          detector_version: null,
          source: null,
          bbox_correct: null,
          confidence: null,
          rejection_reason: 'sanity_reject:aspect_ratio',
          text: null,
          cluster_id: null,
          thumbnail_url: null,
        },
      ],
      region_status: 'verify_rejected',
      region_validated: false,
      region_auto_confirmed: false,
      region_text_choice: 'no_valid_reading',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await undoCropRegion('c1');

    const [url, init] = (vi.mocked(fetch) as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(String(url)).toBe(`${API_PREFIX}/crops/c1/region/undo`);
    expect((init as RequestInit).method).toBe('POST');

    const slot = crop.slots?.[widgetTagSlot.key];
    expect(slot?.subBoxes).toHaveLength(1);
    expect(slot?.subBoxes?.[0].state).toBe('rejected');
    expect(slot?.subBoxes?.[0].rawXyxy).toEqual([0.1, 0.02, 0.3, 0.06]);
    expect(slot?.subBoxes?.[0].score).toBeCloseTo(0.55);
    expect(slot?.subBoxes?.[0].rejectionReason).toBe('sanity_reject:aspect_ratio');
    expect(slot?.lifecycle?.status).toBe('verify_rejected');
    expect(slot?.lifecycle?.validated).toBe(false);
    expect(slot?.lifecycle?.autoConfirmed).toBe(false);
    expect(slot?.text?.choice).toBe('no_valid_reading');
  });
});
