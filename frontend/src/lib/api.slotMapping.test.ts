/**
 * mapRawCrop's slots mapping (docs/design/slot-generic-crop-mapping-
 * plan-2026-09-21.md §4/§10).
 *
 * Originally (C2) this asserted FALSIFIABLE EQUIVALENCE between the
 * hand-copied plate_* fields and readSlot's independently-computed
 * `slots.license_plate` — proving the adapter reproduced the hand-copy
 * before anything depended on it. C9 deleted that hand-copy entirely
 * (Crop no longer has plate_* fields at all — `slots` is the only
 * path), so there is nothing left to compare against. These assertions
 * now pin `slots.license_plate`'s values directly against the raw wire
 * payload instead.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCrop } from './api';
import { mapCropSlots } from './annotations/cropSlots';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from './annotations/registeredSlots';
import { aircraftTailNumberSlot } from './annotations/profiles/aircraftTailNumber';
import type { XYXY } from './annotations/types';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  resetDeploymentSlots();
});

describe('mapRawCrop slots mapping', () => {
  it('maps every plate_* wire field into slots.license_plate for a fully-populated row', async () => {
    const raw = {
      crop_id: 'c1',
      image_path: '/img/1.jpg',
      bbox_norm: [0, 0, 0.4, 0.2],
      region_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      region_bbox_frame: 'source',
      region_score: 0.91,
      region_visible: true,
      region_status: 'detected',
      region_verified: true,
      region_detector: 'lpr_nanov11_640',
      region_detector_version: '1.0',
      region_detector_chain: ['lpr_nanov11_640:hit'],
      region_detected_at: '2026-09-01T00:00:00Z',
      region_verifier: 'gemma-4-e4b',
      region_verifier_version: '4',
      region_verified_at: '2026-09-02T00:00:00Z',
      region_rejection_reason: null,
      region_text: 'ABC123',
      region_text_raw: 'abc123',
      region_text_source: 'gemma',
      region_text_confidence: 0.8,
      region_text_engine_version: '1',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const out = await getCrop('c1');
    const slot = out.slots?.license_plate;
    expect(slot).toBeDefined();

    expect(slot!.subBox?.rawXyxy).toEqual(raw.region_bbox_norm);
    expect(slot!.subBox?.score).toBe(0.91);
    expect(slot!.lifecycle?.status).toBe('detected');
    expect(slot!.lifecycle?.verified).toBe(true);
    expect(slot!.text?.value).toBe('ABC123');
    expect(slot!.text?.raw).toBe('abc123');
    expect(slot!.text?.source).toBe('gemma');
    expect(slot!.text?.confidence).toBe(0.8);
    expect(slot!.provenance?.detector).toBe('lpr_nanov11_640');
    expect(slot!.provenance?.detectorVersion).toBe('1.0');
    expect(slot!.provenance?.chain).toEqual(['lpr_nanov11_640:hit']);
    expect(slot!.provenance?.verifier).toBe('gemma-4-e4b');
    expect(slot!.provenance?.verifierVersion).toBe('4');
    expect(slot!.provenance?.verifiedAt).toBe('2026-09-02T00:00:00Z');
    expect(slot!.provenance?.detectedAt).toBe('2026-09-01T00:00:00Z');
  });

  it('is absent when region_bbox_norm is missing and there is no other evidence', async () => {
    const raw = { crop_id: 'c2', image_path: '/img/2.jpg', bbox_norm: [0, 0, 0.4, 0.2] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c2');
    expect(out.slots?.license_plate).toBeUndefined();
  });

  it('does not crash when the parent bbox is degenerate (NaN-safe projection)', async () => {
    const raw = {
      crop_id: 'c3',
      image_path: '/img/3.jpg',
      bbox_norm: [0, 0, 0, 0],
      region_bbox_norm: [0.1, 0.08, 0.3, 0.12],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c3');
    expect(out.slots?.license_plate?.subBox?.rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(out.slots?.license_plate?.subBox?.parent).toBeNull();
  });

  it('a crop with no plate_* keys at all yields slots === {} (absence, not a block of nulls)', async () => {
    const raw = { crop_id: 'c5', image_path: '/img/5.jpg', bbox_norm: [0, 0, 1, 1] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c5');
    expect(out.slots).toEqual({});
  });

  it('a second registered slot maps under its own key with zero production-code change', async () => {
    installDeploymentSlots([aircraftTailNumberSlot]);
    const raw = { crop_id: 'c6', image_path: '/img/6.jpg', bbox_norm: [0, 0, 1, 1] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c6');
    expect(out.slots?.license_plate).toBeUndefined();
    expect(out.slots?.aircraft_tail_number).toBeUndefined();

    const raw2 = {
      crop_id: 'c7',
      image_path: '/img/7.jpg',
      bbox_norm: [0, 0, 1, 1],
      tail_status: 'detected',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw2)));
    const out2 = await getCrop('c7');
    expect(out2.slots?.aircraft_tail_number?.lifecycle?.status).toBe('detected');
  });

  it('mapCropSlots reads slotRegistry at CALL time, not module-scope destructure', () => {
    const parent: XYXY = [0, 0, 1, 1];
    const before = mapCropSlots({ tail_status: 'detected' }, parent);
    expect(before.aircraft_tail_number).toBeUndefined();

    installDeploymentSlots([aircraftTailNumberSlot]);
    const after = mapCropSlots({ tail_status: 'detected' }, parent);
    expect(after.aircraft_tail_number?.lifecycle?.status).toBe('detected');
  });
});

// B3 item keys the crop grid's accept-suggestion flow and meta panel read.
// Before this mapping existed the suggestion chip and G/Shift+Enter could
// never fire: nothing populated the fields.
describe('mapRawCrop VLM fields', () => {
  it('maps the VLM proposal and categorical confidence', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          crop_id: 'v1',
          image_path: '/x.jpg',
          bbox_norm: [0, 0, 1, 1],
          label_source: 'vlm',
          vlm_confidence: 'medium',
          vlm_proposed_class_id: 12,
          vlm_proposed_class_name: 'forklift',
        }),
      ),
    );
    const out = await getCrop('v1');
    expect(out.vlm_suggested_class_id).toBe(12);
    expect(out.vlm_suggested_class_name).toBe('forklift');
    expect(out.vlm_confidence).toBe('medium');
    expect(out.label_source).toBe('vlm');
  });

  it("defaults an empty label_source to 'unknown', not an invented writer", async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          crop_id: 'v2',
          image_path: '/x.jpg',
          bbox_norm: [],
          label_source: '',
        }),
      ),
    );
    const out = await getCrop('v2');
    expect(out.label_source).toBe('unknown');
    expect(out.vlm_suggested_class_id).toBeNull();
  });
});
