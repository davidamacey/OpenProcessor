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
import { getCrop, mapRawCrop } from './api';
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
    // similarity_to_centroid is served verbatim as cluster_similarity — no
    // client 1-cluster_distance computation (2026-09-24 logic-moves W6).
    expect(crop.similarity_to_centroid).toBe(raw.cluster_similarity);
    expect(crop.cluster_is_core).toBe(raw.cluster_is_core);
    expect(crop.cluster_subid).toBe(raw.cluster_subid);
    expect(crop.class_detector).toBe(raw.class_detector);
    expect(crop.class_detector_version).toBe(raw.class_detector_version);
    expect(crop.class_labeled_at).toBe(raw.class_labeled_at);
    expect(crop.class_labeler).toBe(raw.class_labeler);
    // source (2026-09-24 logic-moves item 14/G3 — replaces the dead
    // hdd_source field) and proposed_class_id/_name (item 11 — served on
    // every crop-shaped item, not just review-queue rows).
    expect(crop.source).toBe(raw.source);
    expect(crop.proposed_class_id).toBe(raw.proposed_class_id);
    expect(crop.proposed_class_name).toBe(raw.proposed_class_name);
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
    // F8 D1 (51b05d7): the probe's opinion is carried through.
    expect(crop.probe_disagreement).toBe(raw.probe_disagreement);
    expect(crop.probe_in_scope).toBe(raw.probe_in_scope);
    expect(crop.probe_model_version).toBe(raw.probe_model_version);
    // OpenProcessor main 8990ede: server-computed actionability.
    expect(crop.probe_actionable).toBe(raw.probe_actionable);
    expect(crop.updated_at).toBe(raw.updated_at);
    // 2026-09-24 logic-moves W7: source (replaces the dead hdd_source),
    // exclude/ignore provenance, and item-text OCR lines.
    expect(crop.source).toBe(raw.source);
    expect(crop.class_excluded).toBe(true);
    expect(crop.excluded_reason).toBe(raw.excluded_reason);
    expect(crop.excluded_at).toBe(raw.excluded_at);
    expect(crop.item_text_lines).toEqual(raw.item_text_lines);

    // image_id is carried (it targets an image Reprocess, W10); an empty
    // served one (a legacy item) reads undefined. thumbnail_url is declared
    // on RawCrop but intentionally not carried onto Crop by mapRawCrop
    // today — documented here so a future intentional wiring doesn't get
    // flagged as a regression, and an accidental one shows up as a diff
    // in this test instead of nowhere.
    expect(crop.image_id).toBe(raw.image_id);
    expect(mapRawCrop({ ...raw, image_id: '' }).image_id).toBeUndefined();
    expect((crop as unknown as Record<string, unknown>).thumbnail_url).toBeUndefined();
  });

  it('maps falsy-but-meaningful booleans/numbers correctly (regression guard for ?? vs || bugs)', async () => {
    const raw = makeItem({
      label_validated: false,
      class_validated: false,
      test_holdout: false,
      confidence: 0,
      cluster_similarity: 0,
      cluster_is_core: false,
      crop_rank_in_image: 0,
      mistakenness_score: 0,
      class_excluded: false,
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.label_validated).toBe(false);
    expect(crop.class_validated).toBe(false);
    expect(crop.test_holdout).toBe(false);
    expect(crop.label_confidence).toBe(0);
    expect(crop.similarity_to_centroid).toBe(0);
    expect(crop.cluster_is_core).toBe(false);
    expect(crop.crop_rank_in_image).toBe(0);
    expect(crop.mistakenness_score).toBe(0);
    expect(crop.class_excluded).toBe(false);
  });

  it('parses item_text_lines tolerantly, dropping malformed entries instead of throwing', async () => {
    const raw = makeItem({
      item_text_lines: [
        { text: 'OK', confidence: 0.5, box_norm: [0, 0, 1, 1], rel_height: 0.2 },
        'not-an-object',
        null,
        { confidence: 'nope' },
      ] as never,
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.item_text_lines).toEqual([
      { text: 'OK', confidence: 0.5, box_norm: [0, 0, 1, 1], rel_height: 0.2 },
      { text: null, confidence: null, box_norm: null, rel_height: null },
    ]);
  });

  it('defaults item_text_lines to [] when the wire omits it entirely', async () => {
    const { item_text_lines: _omit, ...raw } = makeItem();
    void _omit;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.item_text_lines).toEqual([]);
  });

  it('leaves similarity_to_centroid/cluster_is_core null when the backend has not computed them, rather than deriving them from cluster_distance', async () => {
    const raw = makeItem({ cluster_similarity: null, cluster_is_core: null });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(raw)));

    const crop = await getCrop(raw.crop_id);

    expect(crop.similarity_to_centroid).toBeNull();
    expect(crop.cluster_is_core).toBeNull();
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

  describe('W9 / W10 / P4 item fields', () => {
    it('maps the VLM provenance fields verbatim', () => {
      const raw = makeItem();
      const crop = mapRawCrop(raw);
      expect(crop.vlm_endpoint).toBe('vlm_widget@3');
      expect(crop.vlm_model).toBe('widget-vl-7b');
      expect(crop.vlm_prompt_pack).toBe('widget_pack');
    });

    it('maps the W10 lock and import fields verbatim', () => {
      const crop = mapRawCrop(makeItem());
      expect(crop.label_locked).toBe(true);
      expect(crop.import_ids).toEqual(['imp-1', 'imp-2']);
      expect(crop.dataset_split).toBe('val');
      expect(crop.imported_at).toBe('2026-05-06T07:08:09Z');
      expect(crop.proposed_by_import).toBe('imp-2');
      expect(crop.on_negative_frame).toBe(true);
      expect(crop.import_standalone_region).toBe(true);
      expect(crop.proposal_chain).toEqual(['import:imp-2', 'vlm:widget_pack']);
    });

    it('maps the P4 combine origin fields verbatim', () => {
      const crop = mapRawCrop(makeItem());
      expect(crop.origin_project).toBe('widgets_a');
      expect(crop.origin_item_id).toBe('item-origin-9');
      expect(crop.origin_image_id).toBe('image-origin-9');
      expect(crop.origin_split).toBe('train');
      expect(crop.combine_conflict).toBe(true);
      expect(crop.combine_conflict_origins).toEqual(['widgets_a', 'widgets_b']);
      expect(crop.combine_merged_origins).toEqual(['widgets_c']);
    });

    it('defaults a missing key to null / false / [] like the served serializer', () => {
      const {
        vlm_endpoint: _a,
        label_locked: _b,
        import_ids: _c,
        origin_project: _d,
        combine_conflict: _e,
        combine_merged_origins: _f,
        ...raw
      } = makeItem();
      void [_a, _b, _c, _d, _e, _f];
      const crop = mapRawCrop(raw);
      expect(crop.vlm_endpoint).toBeNull();
      expect(crop.label_locked).toBe(false);
      expect(crop.import_ids).toEqual([]);
      expect(crop.origin_project).toBeNull();
      expect(crop.combine_conflict).toBe(false);
      expect(crop.combine_merged_origins).toEqual([]);
    });
  });
});
