/**
 * mapRawCrop's slots mapping (docs/design/slot-generic-crop-mapping-
 * plan-2026-09-21.md §4/§10).
 *
 * `slots` is the only path from the raw `region_*` wire fields to a
 * crop's slot data. These assertions pin a registered region slot's
 * `slots.<key>` values directly against the raw wire payload.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getCrop } from './api';
import { mapCropSlots } from './annotations/cropSlots';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from './annotations/registeredSlots';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { XYXY } from './annotations/types';

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

describe('mapRawCrop slots mapping', () => {
  it('maps every region_* wire field (W8 region_boxes list + item-level fields) into slots.widget_tag', async () => {
    const raw = {
      crop_id: 'c1',
      image_path: '/img/1.jpg',
      bbox_norm: [0, 0, 0.4, 0.2],
      region_boxes: [
        {
          box_id: 'b1',
          state: 'accepted',
          bbox_norm: [0.1, 0.08, 0.3, 0.12],
          bbox_in_parent: [0.1, 0.08, 0.3, 0.12],
          score: 0.91,
          detector: null,
          detector_version: null,
          source: null,
          bbox_correct: null,
          confidence: null,
          rejection_reason: null,
          text: null,
          cluster_id: null,
          thumbnail_url: null,
        },
      ],
      region_status: 'detected',
      region_verified: true,
      region_detector: 'tag_detector_v1',
      region_detector_version: '1.0',
      region_detector_chain: ['tag_detector_v1:hit'],
      region_detected_at: '2026-09-01T00:00:00Z',
      region_verifier: 'gemma-4-e4b',
      region_verifier_version: '4',
      region_verified_at: '2026-09-02T00:00:00Z',
      region_rejection_reason: null,
      region_text: 'TAG-001',
      region_text_raw: 'tag-001',
      region_text_source: 'gemma',
      region_text_confidence: 0.8,
      region_text_engine_version: '1',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const out = await getCrop('c1');
    const slot = out.slots?.widget_tag;
    expect(slot).toBeDefined();

    expect(slot!.subBoxes).toHaveLength(1);
    expect(slot!.subBoxes![0].rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    expect(slot!.subBoxes![0].score).toBe(0.91);
    expect(slot!.lifecycle?.status).toBe('detected');
    expect(slot!.lifecycle?.verified).toBe(true);
    // Item-level text/provenance fields are unaffected by the W8 box list
    // (region_text* / region_detector* stay item-level per the spec).
    expect(slot!.text?.value).toBe('TAG-001');
    expect(slot!.text?.raw).toBe('tag-001');
    expect(slot!.text?.source).toBe('gemma');
    expect(slot!.text?.confidence).toBe(0.8);
    expect(slot!.provenance?.detector).toBe('tag_detector_v1');
    expect(slot!.provenance?.detectorVersion).toBe('1.0');
    expect(slot!.provenance?.chain).toEqual(['tag_detector_v1:hit']);
    expect(slot!.provenance?.verifier).toBe('gemma-4-e4b');
    expect(slot!.provenance?.verifierVersion).toBe('4');
    expect(slot!.provenance?.verifiedAt).toBe('2026-09-02T00:00:00Z');
    expect(slot!.provenance?.detectedAt).toBe('2026-09-01T00:00:00Z');
  });

  it('is absent when region_boxes is missing/empty and there is no other evidence', async () => {
    const raw = { crop_id: 'c2', image_path: '/img/2.jpg', bbox_norm: [0, 0, 0.4, 0.2] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c2');
    expect(out.slots?.widget_tag).toBeUndefined();
  });

  it('does not crash on a degenerate item (NaN-safe — no parent-frame box served)', async () => {
    const raw = {
      crop_id: 'c3',
      image_path: '/img/3.jpg',
      bbox_norm: [0, 0, 0, 0],
      region_boxes: [
        {
          box_id: 'b1',
          state: 'accepted',
          bbox_norm: [0.1, 0.08, 0.3, 0.12],
          bbox_in_parent: null,
          score: null,
          detector: null,
          detector_version: null,
          source: null,
          bbox_correct: null,
          confidence: null,
          rejection_reason: null,
          text: null,
          cluster_id: null,
          thumbnail_url: null,
        },
      ],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));
    const out = await getCrop('c3');
    const box = out.slots?.widget_tag?.subBoxes?.[0];
    expect(box?.rawXyxy).toEqual([0.1, 0.08, 0.3, 0.12]);
    // No bbox_in_parent served -> not drawable in the crop view.
    expect(box?.parent).toBeNull();
  });

  it('a crop with no region_* keys at all yields slots === {} (absence, not a block of nulls)', async () => {
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
    expect(out.slots?.widget_tag).toBeUndefined();
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
