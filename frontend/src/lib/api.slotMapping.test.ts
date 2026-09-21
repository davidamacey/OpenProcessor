/**
 * Falsifiable dual-write equivalence proof (Wave 0, C2 of
 * docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §4/§10).
 *
 * `mapRawCrop` keeps its existing hand-copied `plate_*` fields
 * UNCHANGED and additionally computes `slots` via `mapCropSlots` off
 * the SAME raw payload, independently. This test asserts the two paths
 * agree field-for-field — if `readSlot`'s mapping diverges from the
 * hand-copy in any way, this test fails. It intentionally does NOT use
 * a shared helper between the two sides.
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

describe('mapRawCrop <-> mapCropSlots equivalence (falsifiable)', () => {
  it('agrees on every plate_* field for a fully-populated row', async () => {
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
    expect(slot!.subBox?.score).toBe(out.plate_score);
    expect(slot!.subBox?.shapeWarning).toBe(out.plate_shape_warning);
    expect(slot!.lifecycle?.status).toBe(out.plate_status);
    expect(slot!.lifecycle?.verified).toBe(out.plate_verified);
    expect(slot!.text?.value).toBe(out.plate_text);
    expect(slot!.text?.raw).toBe(out.plate_text_raw);
    expect(slot!.text?.source).toBe(out.plate_text_source);
    expect(slot!.text?.confidence).toBe(out.plate_text_confidence);
    expect(slot!.provenance?.detector).toBe(out.plate_detector);
    expect(slot!.provenance?.detectorVersion).toBe(out.plate_detector_version);
    expect(slot!.provenance?.chain).toEqual(out.plate_detector_chain);
    expect(slot!.provenance?.verifier).toBe(out.plate_verifier);
    expect(slot!.provenance?.verifierVersion).toBe(out.plate_verifier_version);
    expect(slot!.provenance?.verifiedAt).toBe(out.plate_verified_at);
    expect(slot!.provenance?.detectedAt).toBe(out.plate_detected_at);
  });

  it('agrees on shapeWarning when plate_bbox_norm is missing', async () => {
    const raw = { crop_id: 'c2', image_path: '/img/2.jpg', bbox_norm: [0, 0, 0.4, 0.2] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c2');
    expect(out.slots?.license_plate).toBeUndefined();
    expect(out.plate_shape_warning).toBe(false);
  });

  it('agrees on shapeWarning when the parent bbox is degenerate (NaN-safe)', async () => {
    const raw = {
      crop_id: 'c3',
      image_path: '/img/3.jpg',
      bbox_norm: [0, 0, 0, 0],
      plate_bbox_norm: [0.1, 0.08, 0.3, 0.12],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c3');
    expect(out.slots?.license_plate?.subBox?.shapeWarning).toBe(out.plate_shape_warning);
  });

  it('agrees when plate_bbox_norm has a malformed length (both sides reject it)', async () => {
    const raw = {
      crop_id: 'c4',
      image_path: '/img/4.jpg',
      bbox_norm: [0, 0, 0.4, 0.2],
      plate_bbox_norm: [0.1, 0.08, 0.3],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c4');
    expect(out.slots?.license_plate).toBeUndefined();
    expect(out.plate_bbox_norm).toBeNull();
    expect(out.plate_shape_warning).toBe(false);
  });

  // Non-finite (NaN) coordinates cannot be exercised through this file's
  // fetch-mock harness — JSON.stringify(NaN) serializes to `null` before
  // it ever reaches the parsed response, so a real NaN never survives the
  // round trip. That case (both sides resolve to `true`, per §4.4's
  // traced table) is exercised directly against readSlot/evaluateShapeGate
  // in readSlot.test.ts and shapeGate.test.ts instead.

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
    // No tail evidence on the row -> absent, but the key space must
    // still include both slots once installed.
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
