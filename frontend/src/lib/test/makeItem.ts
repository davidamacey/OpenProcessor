/**
 * Shared raw-item fixture builder for `{API_PREFIX}/crops` payloads
 * (docs/design/test-audit-2026-09-24.md, recommendation 4 / P0-3).
 *
 * Every field carries a distinct, non-default value (distinct strings/
 * numbers, `true` where the mapped default is `false`, non-null where the
 * mapped default is `null`) so that dropping a field, or mapping it to a
 * hardcoded default instead of the wire value, is detectable by a plain
 * `toEqual`/`toBe` against the fixture — not just "the test looked fine".
 *
 * `RAW_CROP_KEYS` (`src/lib/api.ts`) is a `satisfies`-checked exhaustive
 * key list, so this file breaking (missing a key TypeScript now requires)
 * is the signal that `RawCrop` grew a field this fixture doesn't cover yet.
 */
import { RAW_CROP_KEYS, type RawCrop } from '../api';

// `Required<RawCrop>` (not `RawCrop`) so TypeScript itself rejects an
// omitted optional field — the compile-time half of "every field has a
// distinct value"; RAW_CROP_KEYS below is the runtime half.
const DEFAULT_ITEM: Required<RawCrop> = {
  crop_id: 'crop-fixture-001',
  image_id: 'image-fixture-777',
  image_path: '/fixtures/img-001.jpg',
  bbox_norm: [0.1, 0.2, 0.6, 0.8],
  class_id: 42,
  class_name: 'sedan',
  class_source: 'human',
  confidence: 0.73,
  class_confidence: 0.92,
  class_confidence_source: 'vlm',
  vlm_raw_class: 'sedan-ish',
  vlm_class_attempted_at: '2026-09-24T01:02:03Z',
  vlm_class_empty_reason: 'no_match',
  cluster_id: 17,
  cluster_distance: 0.33,
  cluster_nearest_id: 23,
  cluster_similarity: 0.81,
  cluster_is_core: true,
  cluster_subid: '47a',
  label_validated: true,
  class_validated: true,
  label_source: 'human_review',
  class_detector: 'v6_model',
  class_detector_version: '6.2.1',
  class_labeled_at: '2026-01-02T03:04:05Z',
  class_labeler: 'user@example.com',
  source: 'tag_holdout_sample',
  test_holdout: true,
  crop_rank_in_image: 2,
  crop_area_norm: 0.19,
  blur_lap_ratio: 4.5,
  classifier_raw_confidence: 0.61,
  proposal_name: 'car',
  vlm_confidence: 'high',
  vlm_proposed_class_id: 99,
  vlm_proposed_class_name: 'pickup_truck',
  proposed_class_id: 101,
  proposed_class_name: 'suv',
  mistakenness_score: 0.27,
  mistakenness_method: 'entropy',
  mistakenness_version: 'v3',
  mistakenness_scored_at: '2026-02-03T04:05:06Z',
  thumbnail_url: '/thumb/crop-fixture-001.jpg',
  updated_at: '2026-03-04T05:06:07Z',
  class_excluded: true,
  excluded_reason: 'blurry',
  excluded_at: '2026-04-05T06:07:08Z',
  item_text_lines: [
    { text: 'STOP', confidence: 0.88, box_norm: [0.1, 0.1, 0.3, 0.2], rel_height: 0.1 },
  ],
};

// Runtime cross-check that DEFAULT_ITEM's own keys match RAW_CROP_KEYS
// exactly, in case the two literals drift even though both individually
// typecheck (e.g. a stray extra key `Required<RawCrop>` wouldn't catch).
const defaultItemKeys = new Set(Object.keys(DEFAULT_ITEM));
const rawCropKeySet = new Set<string>(RAW_CROP_KEYS);
if (
  defaultItemKeys.size !== rawCropKeySet.size ||
  [...defaultItemKeys].some((k) => !rawCropKeySet.has(k))
) {
  throw new Error(
    'makeItem.ts DEFAULT_ITEM keys have drifted from RAW_CROP_KEYS in src/lib/api.ts',
  );
}

/**
 * Builds a raw `{API_PREFIX}/crops` item carrying every `RawCrop` field with
 * a distinct value, so a fetch-mocked test can assert the full mapping.
 * `overrides` may set/omit specific fields, e.g. a slot-specific test that
 * layers `region_*` fields on top (those are not part of `RawCrop` — pass
 * them as extra properties via a wider type at the call site).
 */
export function makeItem(overrides: Partial<RawCrop> = {}): RawCrop {
  return { ...DEFAULT_ITEM, ...overrides };
}
