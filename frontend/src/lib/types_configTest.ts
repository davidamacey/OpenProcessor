/**
 * Wire types for OpenProcessor W5 test-on-crop: `POST /prompt_packs/test`
 * and `POST /region_profiles/test` (any_domain_plan.md §5, §7.5-7.7).
 * Neither route writes anything. Pinned key-for-key to the vendored
 * OpenAPI (OpenProcessor f582aa05) by `contract/configTestContract.test.ts`.
 *
 * Owner: Track C. `preview_item` is the item as the write would leave it;
 * the wrappers in `api_configTest.ts` also map it through `mapRawCrop`
 * into `preview`.
 */
import type { Crop } from './types';
import type { ValidationReport } from './types_config';
import type { PromptPackBody } from './types_packs';
import type { RegionProfileBody } from './types_profiles';

/** The call ids `PackTestRequest.call` enumerates. */
export type PackTestCall =
  'combined' | 'classify' | 'open_classify' | 'region_verify' | 'region_visible';

/** Where the VLM answer comes from, shared by both test requests. Unset
 *  keys take the server's default (the active endpoint). */
export interface TestVlmSelection {
  vlm_name?: string | null;
  vlm_revision?: number | null;
  acknowledge_external?: boolean;
}

/** `POST /prompt_packs/test`. Exactly one pack source: `draft`, or
 *  `pack_name` (+ `pack_revision`); none is the active pack. */
export interface PackTestRequest extends TestVlmSelection {
  pack_name?: string | null;
  pack_revision?: number | null;
  draft?: PromptPackBody | null;
  call: PackTestCall;
  crop_ids?: string[];
  use_region_box?: 'current' | 'none';
  class_names?: string[] | null;
  profile_name?: string | null;
}

/** `PackTestPackRef`. */
export interface PackTestPackRef {
  draft: boolean;
  name: string | null;
  revision: number | null;
}

/** `PackTestVlmRef`: the endpoint that answered (`name@revision`). */
export interface PackTestVlmRef {
  draft: boolean;
  endpoint: string;
  model: string;
  name: string | null;
  revision: number | null;
}

/** `PackTestPrompt`. */
export interface PackTestPrompt {
  system: string | null;
  user_text: string | null;
}

/** `PackTestCropResult`. `preview` is `preview_item` mapped through
 *  `mapRawCrop`. */
export interface PackTestCropResult<P = unknown> {
  crop_id: string;
  box_id?: string | null;
  parsed?: Record<string, unknown> | boolean | null;
  preview_item?: Record<string, unknown> | null;
  /** The served reason this crop was not sent to the VLM. */
  skipped?: string | null;
  preview?: P | null;
}

/** `PackTestResponse`. */
export interface PackTestResponse<P = unknown> {
  call: PackTestCall;
  latency_ms: number;
  pack: PackTestPackRef;
  parse_error: string | null;
  parse_ok: boolean;
  prompt: PackTestPrompt;
  raw_reply: string | null;
  reasoning: string | null;
  results: PackTestCropResult<P>[];
  validation: ValidationReport;
  vlm: PackTestVlmRef;
}

/** `POST /region_profiles/test`: one crop. Profile source: at most one of
 *  `profile_name` (+ `profile_revision`) or `draft`; none is the active
 *  profile. */
export interface RegionTestRequest extends TestVlmSelection {
  crop_id: string;
  draft?: RegionProfileBody | null;
  profile_name?: string | null;
  profile_revision?: number | null;
  prompt_pack_draft?: PromptPackBody | null;
  prompt_pack_name?: string | null;
  prompt_pack_revision?: number | null;
  segmenter_text_prompt?: string | null;
  verify?: boolean;
}

/** `RegionTestProfileRef`. */
export interface RegionTestProfileRef {
  draft: boolean;
  name: string | null;
  revision: number | null;
}

/** `RegionTestCandidate`: one raw candidate of a leg, a complete
 *  region-box wire element (`box_id` is null: ids are assigned on write)
 *  plus what the selection did with it. */
export interface RegionTestCandidate {
  bbox_correct: boolean | null;
  /** Normalised in the source-image frame. */
  bbox_norm: number[];
  /** Normalised in the parent crop's frame, server-projected. */
  bbox_in_parent: number[] | null;
  box_id: string | null;
  candidate_index: number;
  cluster_distance: number | null;
  cluster_id: number | null;
  cluster_subid: string | null;
  confidence: string | null;
  detected_at: string | null;
  detector: string | null;
  detector_version: string | null;
  drop_reason: 'below_min_score' | 'nms' | 'over_max' | null;
  locked: boolean;
  mask_iou: number | null;
  mask_polygon: number[][] | null;
  mask_polygon_in_parent: number[][] | null;
  rejection_reason: string | null;
  score: number | null;
  selected: boolean;
  source: string | null;
  state: string;
  text: string | null;
  text_choice: string | null;
  text_confidence: number | null;
  text_disagreement: boolean | null;
  text_engine_version: string | null;
  text_ocr: string | null;
  text_raw: string | null;
  text_source: string | null;
  text_vlm: string | null;
  text_vlm_invalid: string | null;
  thumbnail_url: string | null;
}

/** `RegionTestLeg`. */
export interface RegionTestLeg {
  leg: 'detector' | 'segmenter';
  status: 'ok' | 'skipped' | 'error';
  reason?: string | null;
  elapsed_ms?: number | null;
  candidates?: RegionTestCandidate[];
}

/** `RegionTestVerify`. */
export interface RegionTestVerify {
  latency_ms: number;
  pack: PackTestPackRef;
  parse_error: string | null;
  parse_ok: boolean;
  prompt: PackTestPrompt;
  raw_reply: string | null;
  reasoning: string | null;
  vlm: PackTestVlmRef;
}

/** `RegionTestResponse`; `preview` is `preview_item` mapped. */
export interface RegionTestResponse {
  crop_id: string;
  item_eligible: boolean;
  legs: RegionTestLeg[];
  preview_basis: 'selection_accepted' | 'vlm_verdicts';
  preview_item: Record<string, unknown>;
  preview: Crop;
  profile: RegionTestProfileRef;
  validation: ValidationReport;
  verify?: RegionTestVerify | null;
}
