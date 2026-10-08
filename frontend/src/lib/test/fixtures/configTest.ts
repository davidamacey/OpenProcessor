/**
 * W5 test-on-crop fixtures (`POST /prompt_packs/test`,
 * `POST /region_profiles/test`), shaped after the vendored contract
 * (PackTestResponse, RegionTestResponse) in the neutral widget/tag domain.
 */
import type {
  PackTestResponse,
  RegionTestCandidate,
  RegionTestLeg,
  RegionTestResponse,
} from '$lib/types_configTest';
import { cleanReport } from './promptPacks';

export function packTestResponseFixture(): PackTestResponse {
  return {
    call: 'classify',
    pack: { name: null, revision: null, draft: true },
    vlm: {
      name: 'env',
      revision: null,
      draft: false,
      endpoint: 'env@abc123',
      model: 'example/vision-model',
    },
    prompt: { system: 'You classify widgets.', user_text: 'Pick one of: widget, gadget' },
    raw_reply: '[{"img": 1, "class": "widget", "confidence": 0.91}]',
    reasoning: null,
    latency_ms: 812.4,
    parse_ok: true,
    parse_error: null,
    validation: cleanReport(),
    results: [
      {
        crop_id: 'c_123',
        box_id: null,
        parsed: { class_name: 'widget', confidence: 0.91 },
        skipped: null,
        preview_item: {
          crop_id: 'c_123',
          image_id: 'img_1',
          bbox_norm: [0.1, 0.1, 0.5, 0.5],
          proposed_class_name: 'widget',
        },
      },
    ],
  };
}

export function candidateFixture(
  over: Partial<RegionTestCandidate> = {},
): RegionTestCandidate {
  return {
    bbox_correct: null,
    bbox_norm: [0.2, 0.2, 0.6, 0.5],
    bbox_in_parent: [0.1, 0.1, 0.9, 0.9],
    box_id: null,
    candidate_index: 0,
    cluster_distance: null,
    cluster_id: null,
    cluster_subid: null,
    confidence: null,
    detected_at: null,
    detector: 'tag_detector_v1',
    detector_version: null,
    drop_reason: null,
    locked: false,
    mask_iou: null,
    mask_polygon: null,
    mask_polygon_in_parent: null,
    rejection_reason: null,
    score: 0.91,
    selected: true,
    source: null,
    state: 'detected',
    text: null,
    text_choice: null,
    text_confidence: null,
    text_disagreement: null,
    text_engine_version: null,
    text_ocr: null,
    text_raw: null,
    text_source: null,
    text_vlm: null,
    text_vlm_invalid: null,
    thumbnail_url: null,
    ...over,
  };
}

export function legsFixture(): RegionTestLeg[] {
  return [
    {
      leg: 'detector',
      status: 'ok',
      reason: null,
      elapsed_ms: 41,
      candidates: [
        candidateFixture(),
        candidateFixture({
          candidate_index: 1,
          selected: false,
          drop_reason: 'below_min_score',
          score: 0.12,
          bbox_norm: [0.7, 0.1, 0.9, 0.3],
          bbox_in_parent: [0.6, 0.6, 0.8, 0.8],
        }),
      ],
    },
    {
      leg: 'segmenter',
      status: 'ok',
      reason: null,
      elapsed_ms: 220,
      candidates: [
        candidateFixture({
          candidate_index: 0,
          detector: 'tag_segmenter_v1',
          mask_iou: 0.83,
          mask_polygon: [
            [0.2, 0.2],
            [0.6, 0.2],
            [0.4, 0.5],
          ],
          mask_polygon_in_parent: [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.5, 0.9],
          ],
        }),
      ],
    },
  ];
}

export function regionTestResponseFixture(
  over: Partial<RegionTestResponse> = {},
): Omit<RegionTestResponse, 'preview'> {
  return {
    crop_id: 'c_123',
    item_eligible: true,
    testable: true,
    legs: legsFixture(),
    preview_basis: 'selection_accepted',
    preview_item: {
      crop_id: 'c_123',
      image_id: 'img_1',
      bbox_norm: [0.1, 0.1, 0.5, 0.5],
    },
    profile: { name: 'widget_tag', revision: 2, draft: false },
    validation: cleanReport(),
    verify: null,
    ...over,
  };
}
