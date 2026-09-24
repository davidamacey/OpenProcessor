/**
 * Full `mapRawCrop` field-mapping test (docs/design/test-audit-2026-09-24.md
 * §2.2/P0-3). Every field `RawCrop` declares (`makeItem`, `src/lib/test/
 * makeItem.ts`) is asserted through to its mapped `Crop` field with the
 * fixture's exact, distinct value — a dropped field, a field mapped to a
 * hardcoded default, or a field mis-mapped to the wrong wire key all fail
 * this test. This closes three mutations the audit found surviving the
 * previous suite: `cluster_subid` silently mapped to `null`,
 * `test_holdout` silently mapped to `false`, and `confidence` dropped
 * instead of surfacing as `label_confidence`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCrop } from './api';
import { makeItem } from './test/makeItem';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('mapRawCrop full field mapping', () => {
  it('maps every RawCrop field onto Crop with the fixture value, not a default', async () => {
    const raw = makeItem();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.id).toBe(raw.crop_id);
    expect(crop.source_image_path).toBe(raw.image_path);
    // bbox_norm: RawCrop is [x1,y1,x2,y2]; Crop is {cx,cy,w,h}.
    expect(crop.bbox_norm.cx).toBeCloseTo(0.35, 10);
    expect(crop.bbox_norm.cy).toBeCloseTo(0.5, 10);
    expect(crop.bbox_norm.w).toBeCloseTo(0.5, 10);
    expect(crop.bbox_norm.h).toBeCloseTo(0.6, 10);
    expect(crop.class_id).toBe(raw.class_id);
    expect(crop.class_name).toBe(raw.class_name);
    expect(crop.class_source).toBe(raw.class_source);
    expect(crop.label_source).toBe(raw.label_source);
    expect(crop.label_validated).toBe(true);
    expect(crop.class_validated).toBe(true);
    // confidence -> label_confidence (the field is renamed on the wire).
    expect(crop.label_confidence).toBe(raw.confidence);
    expect(crop.cluster_id).toBe(raw.cluster_id);
    // similarity_to_centroid = max(0, 1 - cluster_distance)
    expect(crop.similarity_to_centroid).toBeCloseTo(1 - raw.cluster_distance!, 10);
    expect(crop.cluster_subid).toBe(raw.cluster_subid);
    expect(crop.class_detector).toBe(raw.class_detector);
    expect(crop.class_detector_version).toBe(raw.class_detector_version);
    expect(crop.class_labeled_at).toBe(raw.class_labeled_at);
    expect(crop.class_labeler).toBe(raw.class_labeler);
    expect(crop.test_holdout).toBe(true);
    expect(crop.crop_rank_in_image).toBe(raw.crop_rank_in_image);
    expect(crop.crop_area_norm).toBe(raw.crop_area_norm);
    expect(crop.blur_lap_ratio).toBe(raw.blur_lap_ratio);
    expect(crop.classifier_raw_confidence).toBe(raw.classifier_raw_confidence);
    expect(crop.proposal_name).toBe(raw.proposal_name);
    expect(crop.vlm_confidence).toBe(raw.vlm_confidence);
    // vlm_proposed_* -> vlm_suggested_* (the field is renamed on the wire).
    expect(crop.vlm_suggested_class_id).toBe(raw.vlm_proposed_class_id);
    expect(crop.vlm_suggested_class_name).toBe(raw.vlm_proposed_class_name);
    expect(crop.mistakenness_score).toBe(raw.mistakenness_score);
    expect(crop.mistakenness_method).toBe(raw.mistakenness_method);
    expect(crop.mistakenness_version).toBe(raw.mistakenness_version);
    expect(crop.mistakenness_scored_at).toBe(raw.mistakenness_scored_at);
    expect(crop.updated_at).toBe(raw.updated_at);

    // image_id and thumbnail_url are declared on RawCrop but intentionally
    // not carried onto Crop by mapRawCrop today — documented here so a
    // future intentional wiring doesn't get flagged as a regression, and an
    // accidental one shows up as a diff in this test instead of nowhere.
    expect((crop as unknown as Record<string, unknown>).image_id).toBeUndefined();
    expect((crop as unknown as Record<string, unknown>).thumbnail_url).toBeUndefined();
  });

  it('maps falsy-but-meaningful booleans/numbers correctly (regression guard for ?? vs || bugs)', async () => {
    const raw = makeItem({
      label_validated: false,
      class_validated: false,
      test_holdout: false,
      confidence: 0,
      cluster_distance: 0,
      crop_rank_in_image: 0,
      mistakenness_score: 0,
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.label_validated).toBe(false);
    expect(crop.class_validated).toBe(false);
    expect(crop.test_holdout).toBe(false);
    expect(crop.label_confidence).toBe(0);
    expect(crop.similarity_to_centroid).toBe(1);
    expect(crop.crop_rank_in_image).toBe(0);
    expect(crop.mistakenness_score).toBe(0);
  });

  it('leaves similarity_to_centroid null when cluster_distance is null, rather than computing 1 - null', async () => {
    const raw = makeItem({ cluster_distance: null });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.similarity_to_centroid).toBeNull();
  });

  it("falls back updated_at to '' when the wire omits it, rather than the string cast leaking a non-string", async () => {
    // makeItem() always sets updated_at; build the raw item without it to
    // exercise mapRawCrop's `typeof ... === 'string'` guard's false branch.
    const { updated_at: _omit, ...raw } = makeItem();
    void _omit;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.updated_at).toBe('');
  });
});
