/**
 * dq-region (2026-09-24): `POST {API_PREFIX}/crops/{id}/region/undo`
 * (`undoCropRegion`) returns the post-undo item, mapped through
 * `mapRawCrop` -> `mapCropSlots` -> `readSlot` like any other crop — so
 * a Z-undo that reverts a confirm back to `verify_rejected` re-renders
 * the candidate box/rejection-reason from the server's own response,
 * with no client-side reconstruction. This is the "undo restores
 * candidate fields in the UI" contract review/+page.svelte's
 * undoLast()/slotBack() rely on: whatever this function returns is what
 * `queue.items[idx]` gets replaced with directly.
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

describe('undoCropRegion — dq-region candidate round trip', () => {
  it('maps the restored candidate box, rejection reason, and text choice onto Crop.slots', async () => {
    const raw = {
      crop_id: 'c1',
      image_path: '/nas/img.jpg',
      bbox_norm: [0, 0, 0.4, 0.2],
      region_bbox_norm: null,
      region_status: 'verify_rejected',
      region_rejection_reason: 'sanity_reject:aspect_ratio',
      region_candidate_bbox_norm: [0.1, 0.02, 0.3, 0.06],
      region_candidate_score: 0.55,
      region_candidate_detector: 'tag_detector_v1',
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
    expect(slot?.subBox?.rawXyxy).toBeNull();
    expect(slot?.subBox?.candidate?.rawXyxy).toEqual([0.1, 0.02, 0.3, 0.06]);
    expect(slot?.subBox?.candidate?.score).toBeCloseTo(0.55);
    expect(slot?.lifecycle?.status).toBe('verify_rejected');
    expect(slot?.lifecycle?.rejectionReason).toBe('sanity_reject:aspect_ratio');
    expect(slot?.lifecycle?.validated).toBe(false);
    expect(slot?.lifecycle?.autoConfirmed).toBe(false);
    expect(slot?.text?.choice).toBe('no_valid_reading');
  });
});
