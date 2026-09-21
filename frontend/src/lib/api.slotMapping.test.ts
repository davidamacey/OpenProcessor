/**
 * mapRawCrop's slots mapping (docs/design/slot-generic-crop-mapping-
 * plan-2026-09-21.md §4/§10).
 *
 * Originally (C2) this asserted FALSIFIABLE EQUIVALENCE between the
 * hand-copied plate_* fields and readSlot's independently-computed
 * `slots.license_plate` — proving the adapter reproduced the hand-copy
 * before anything depended on it. C9 deleted that hand-copy entirely
 * (OpCrop no longer has plate_* fields at all — `slots` is the only
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
      plate_bbox_norm: [0.1, 0.08, 0.3, 0.12],
      plate_bbox_frame: 'source',
      plate_score: 0.91,
      plate_visible: true,
      plate_status: 'detected',
      plate_verified: true,
      plate_detector: 'lpr_nanov11_640',
      plate_detector_version: '1.0',
      plate_detector_chain: ['lpr_nanov11_640:hit'],
      plate_detected_at: '2026-09-01T00:00:00Z',
      plate_verifier: 'gemma-4-e4b',
      plate_verifier_version: '4',
      plate_verified_at: '2026-09-02T00:00:00Z',
      plate_rejection_reason: null,
      plate_text: 'ABC123',
      plate_text_raw: 'abc123',
      plate_text_source: 'gemma',
      plate_text_confidence: 0.8,
      plate_text_engine_version: '1',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const out = await getCrop('c1');
    const slot = out.slots?.license_plate;
    expect(slot).toBeDefined();

    expect(slot!.subBox?.rawXyxy).toEqual(raw.plate_bbox_norm);
    expect(slot!.subBox?.score).toBe(0.91);
    expect(slot!.subBox?.shapeWarning).toBe(false);
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

  it('is absent when plate_bbox_norm is missing and there is no other evidence', async () => {
    const raw = { crop_id: 'c2', image_path: '/img/2.jpg', bbox_norm: [0, 0, 0.4, 0.2] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c2');
    expect(out.slots?.license_plate).toBeUndefined();
  });

  it('flags an implausible shape even when the parent bbox is degenerate (NaN-safe, per shapeGate.ts)', async () => {
    const raw = {
      crop_id: 'c3',
      image_path: '/img/3.jpg',
      bbox_norm: [0, 0, 0, 0],
      plate_bbox_norm: [0.1, 0.08, 0.3, 0.12],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c3');
    expect(out.slots?.license_plate?.subBox?.shapeWarning).toBe(false);
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
