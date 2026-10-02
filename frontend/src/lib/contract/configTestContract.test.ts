/**
 * `types_configTest.ts` vs the vendored OpenAPI (OpenProcessor f582aa05,
 * W5). Each key map is compile-time exact against its interface and is
 * pinned to the schema's property set. Two things are deliberately not
 * pinned: `vlm_draft` on both requests (a cross-page draft, not offered
 * by this UI) and the client-added `preview` (the mapped `preview_item`).
 * Both requests are `additionalProperties: false`, so the bodies the
 * controllers build are checked against the declared keys.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import { createPackTest } from '$lib/packs/packTestController.svelte';
import { createProfileTest } from '$lib/profiles/profileTestController.svelte';
import type * as T from '$lib/types_configTest';

type Schema = { properties?: Record<string, unknown>; additionalProperties?: boolean };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const declared = (name: string): string[] => {
  const s = schemas[name];
  if (!s?.properties) throw new Error(`schema not found: ${name}`);
  return Object.keys(s.properties).sort();
};

/** Request keys the contract declares but this UI never sends. */
const UNSENT = ['vlm_draft'];

const PACK_REQUEST = keys({
  pack_name: true,
  pack_revision: true,
  draft: true,
  call: true,
  crop_ids: true,
  use_region_box: true,
  class_names: true,
  profile_name: true,
  vlm_name: true,
  vlm_revision: true,
  acknowledge_external: true,
} satisfies Record<keyof T.PackTestRequest, true>);

const REGION_REQUEST = keys({
  crop_id: true,
  draft: true,
  profile_name: true,
  profile_revision: true,
  prompt_pack_draft: true,
  prompt_pack_name: true,
  prompt_pack_revision: true,
  segmenter_text_prompt: true,
  verify: true,
  vlm_name: true,
  vlm_revision: true,
  acknowledge_external: true,
} satisfies Record<keyof T.RegionTestRequest, true>);

const CASES: [string, string[]][] = [
  [
    'PackTestPackRef',
    keys({ draft: true, name: true, revision: true } satisfies Record<
      keyof T.PackTestPackRef,
      true
    >),
  ],
  [
    'PackTestVlmRef',
    keys({
      draft: true,
      endpoint: true,
      model: true,
      name: true,
      revision: true,
    } satisfies Record<keyof T.PackTestVlmRef, true>),
  ],
  [
    'PackTestPrompt',
    keys({ system: true, user_text: true } satisfies Record<
      keyof T.PackTestPrompt,
      true
    >),
  ],
  [
    'PackTestCropResult',
    // `preview` is the client-added mapped preview_item.
    keys({
      crop_id: true,
      box_id: true,
      parsed: true,
      preview_item: true,
      skipped: true,
    } satisfies Record<Exclude<keyof T.PackTestCropResult, 'preview'>, true>),
  ],
  [
    'PackTestResponse',
    keys({
      call: true,
      latency_ms: true,
      pack: true,
      parse_error: true,
      parse_ok: true,
      prompt: true,
      raw_reply: true,
      reasoning: true,
      results: true,
      validation: true,
      vlm: true,
    } satisfies Record<keyof T.PackTestResponse, true>),
  ],
  [
    'RegionTestProfileRef',
    keys({ draft: true, name: true, revision: true } satisfies Record<
      keyof T.RegionTestProfileRef,
      true
    >),
  ],
  [
    'RegionTestCandidate',
    keys({
      bbox_correct: true,
      bbox_norm: true,
      bbox_in_parent: true,
      box_id: true,
      candidate_index: true,
      cluster_distance: true,
      cluster_id: true,
      cluster_subid: true,
      confidence: true,
      detected_at: true,
      detector: true,
      detector_version: true,
      drop_reason: true,
      locked: true,
      mask_iou: true,
      mask_polygon: true,
      mask_polygon_in_parent: true,
      rejection_reason: true,
      score: true,
      selected: true,
      source: true,
      state: true,
      text: true,
      text_choice: true,
      text_confidence: true,
      text_disagreement: true,
      text_engine_version: true,
      text_ocr: true,
      text_raw: true,
      text_source: true,
      text_vlm: true,
      text_vlm_invalid: true,
      thumbnail_url: true,
    } satisfies Record<keyof T.RegionTestCandidate, true>),
  ],
  [
    'RegionTestLeg',
    keys({
      leg: true,
      status: true,
      reason: true,
      elapsed_ms: true,
      candidates: true,
    } satisfies Record<keyof T.RegionTestLeg, true>),
  ],
  [
    'RegionTestVerify',
    keys({
      latency_ms: true,
      pack: true,
      parse_error: true,
      parse_ok: true,
      prompt: true,
      raw_reply: true,
      reasoning: true,
      vlm: true,
    } satisfies Record<keyof T.RegionTestVerify, true>),
  ],
  [
    'RegionTestResponse',
    // `preview` is the client-added mapped preview_item.
    keys({
      crop_id: true,
      item_eligible: true,
      legs: true,
      preview_basis: true,
      preview_item: true,
      profile: true,
      validation: true,
      verify: true,
    } satisfies Record<Exclude<keyof T.RegionTestResponse, 'preview'>, true>),
  ],
];

describe('types_configTest.ts keys match the vendored OpenAPI schemas', () => {
  it('loaded a non-trivial case list (guards a vacuous pass)', () => {
    expect(CASES.length).toBeGreaterThan(8);
  });
  for (const [schema, tsKeys] of CASES) {
    it(schema, () => {
      expect(tsKeys).toEqual(declared(schema));
    });
  }
  it('PackTestRequest (minus the unsent vlm_draft)', () => {
    expect([...PACK_REQUEST, ...UNSENT].sort()).toEqual(declared('PackTestRequest'));
  });
  it('RegionTestRequest (minus the unsent vlm_draft)', () => {
    expect([...REGION_REQUEST, ...UNSENT].sort()).toEqual(declared('RegionTestRequest'));
  });
});

describe('both test requests are strict, and the controllers send only declared keys', () => {
  it('the schemas are additionalProperties: false', () => {
    expect(schemas.PackTestRequest!.additionalProperties).toBe(false);
    expect(schemas.RegionTestRequest!.additionalProperties).toBe(false);
  });

  it('pack test: every option set, draft and saved', () => {
    const t = createPackTest(async () => {
      throw new Error('not called');
    });
    t.call = 'combined';
    t.cropIdsText = 'c_1 c_2';
    t.useRegionBox = 'current';
    t.vlmSelection = {
      vlm_name: 'remote_a',
      vlm_revision: null,
      acknowledge_external: true,
    };
    const ctx = { name: 'widget_tag', revision: 2, draft: { a: 'b' } };
    const allowed = declared('PackTestRequest');
    for (const source of ['draft', 'saved'] as const) {
      t.source = source;
      for (const k of Object.keys(t.request(ctx))) expect(allowed, k).toContain(k);
    }
  });

  it('profile test: every option set, draft and saved', () => {
    const t = createProfileTest(async () => {
      throw new Error('not called');
    });
    t.cropId = 'c_1';
    t.segmenterPrompt = 'a tag';
    t.verify = true;
    t.vlmSelection = {
      vlm_name: 'remote_a',
      vlm_revision: 1,
      acknowledge_external: true,
    };
    const ctx = { name: 'widget_tag', revision: 2, draft: { a: 'b' } };
    const allowed = declared('RegionTestRequest');
    for (const source of ['draft', 'saved'] as const) {
      t.source = source;
      for (const k of Object.keys(t.request(ctx))) expect(allowed, k).toContain(k);
    }
  });

  it('the call and leg/status/preview_basis enums the types use are the contract ones', () => {
    const enums = (schema: string, prop: string): string[] =>
      [...((schemas[schema]!.properties![prop] as { enum: string[] }).enum ?? [])].sort();
    const calls: T.PackTestCall[] = [
      'combined',
      'classify',
      'open_classify',
      'region_verify',
      'region_visible',
    ];
    expect([...calls].sort()).toEqual(enums('PackTestRequest', 'call'));
    const legs: T.RegionTestLeg['leg'][] = ['detector', 'segmenter'];
    expect([...legs].sort()).toEqual(enums('RegionTestLeg', 'leg'));
    const status: T.RegionTestLeg['status'][] = ['ok', 'skipped', 'error'];
    expect([...status].sort()).toEqual(enums('RegionTestLeg', 'status'));
    const basis: T.RegionTestResponse['preview_basis'][] = [
      'selection_accepted',
      'vlm_verdicts',
    ];
    expect([...basis].sort()).toEqual(enums('RegionTestResponse', 'preview_basis'));
  });
});
