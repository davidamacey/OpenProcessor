/**
 * `types_openVocab.ts` vs the vendored OpenAPI schemas (OpenProcessor
 * fce17771). Each key map is compile-time exact against its type
 * (`satisfies Record<keyof T, true>` rejects a missing or an extra key) and
 * pinned to the schema's property set, so a served rename fails here
 * instead of rendering "—". The required-key maps pin which served keys
 * the types may not make optional. The `scope` / `type` unions are pinned
 * to the schema enums.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_openVocab';

type Schema = {
  properties?: Record<string, { enum?: string[]; items?: { enum?: string[] } }>;
  required?: string[];
};
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
type ReqKeys<X> = { [K in keyof X]-?: object extends Pick<X, K> ? never : K }[keyof X];
const required = <X>(o: Record<ReqKeys<X>, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'OpenVocabBody-Input',
    keys({
      display_name: true,
      run_on_ingest: true,
      image_max_side: true,
      dedup_iou: true,
      max_enabled_targets: true,
      targets: true,
      gating: true,
    } satisfies Record<keyof T.OpenVocabBody, true>),
  ],
  [
    'OpenVocabTargetBody',
    keys({
      prompt: true,
      class_name: true,
      enabled: true,
      mask: true,
      min_score: true,
      min_area_frac: true,
      max_area_frac: true,
      max_instances: true,
      parent_classes: true,
    } satisfies Record<keyof T.OpenVocabTargetBody, true>),
  ],
  [
    'OpenVocabGatingBody',
    keys({ tier2_vlm_precheck: true, tier3_hit_rate: true } satisfies Record<
      keyof T.OpenVocabGatingBody,
      true
    >),
  ],
  [
    'OpenVocabHitRateBody',
    keys({
      enabled: true,
      miss_threshold: true,
      sample_floor: true,
      window: true,
    } satisfies Record<keyof T.OpenVocabHitRateBody, true>),
  ],
  [
    'OpenVocabDoc',
    // The shared `ConfigDocBase` also types `updated_by`; this schema does
    // not serve it, so it is left out of the pinned set (see the type's doc).
    keys({
      name: true,
      source: true,
      read_only: true,
      revision: true,
      etag: true,
      description: true,
      body: true,
      created_at: true,
      updated_at: true,
      cloned_from: true,
      active: true,
      active_revision: true,
      validation: true,
    } satisfies Record<Exclude<keyof T.OpenVocabDoc, 'updated_by'>, true>),
  ],
  [
    'OpenVocabSummary',
    keys({
      name: true,
      source: true,
      read_only: true,
      revision: true,
      etag: true,
      display_name: true,
      n_targets: true,
      n_enabled_targets: true,
      run_on_ingest: true,
      active: true,
      active_revision: true,
      updated_at: true,
    } satisfies Record<keyof T.OpenVocabSummary, true>),
  ],
  [
    'OpenVocabTemplateSummary',
    keys({
      name: true,
      path: true,
      n_targets: true,
      display_name: true,
      read_only: true,
      source: true,
    } satisfies Record<keyof T.OpenVocabTemplateSummary, true>),
  ],
  [
    'OpenVocabList',
    keys({
      sets: true,
      templates: true,
      active: true,
      config_revision: true,
      stale: true,
      segmenter: true,
    } satisfies Record<keyof T.OpenVocabList, true>),
  ],
  [
    'OpenVocabFieldSchema',
    keys({
      scope: true,
      field: true,
      label: true,
      type: true,
      default: true,
      advanced: true,
      help: true,
      min: true,
      max: true,
    } satisfies Record<keyof T.OpenVocabFieldSchema, true>),
  ],
  [
    'OpenVocabSchema',
    keys({
      fields: true,
      max_enabled_targets_ceiling: true,
      vocabulary: true,
    } satisfies Record<keyof T.OpenVocabSchema, true>),
  ],
  [
    'OpenVocabVocabulary',
    keys({ statuses: true, drop_reasons: true, gate_reasons: true } satisfies Record<
      keyof T.OpenVocabVocabulary,
      true
    >),
  ],
  [
    'VocabularyOption',
    keys({ value: true, label: true } satisfies Record<keyof T.VocabularyOption, true>),
  ],
  [
    'SegmenterAvailability',
    keys({ configured: true, reachable: true } satisfies Record<
      keyof T.SegmenterAvailability,
      true
    >),
  ],
  [
    'OpenVocabRevisionSummary',
    keys({
      revision: true,
      saved_at: true,
      cloned_from: true,
      description: true,
    } satisfies Record<keyof T.OpenVocabRevisionSummary, true>),
  ],
  [
    'OpenVocabRevisionsResponse',
    keys({ name: true, revisions: true } satisfies Record<
      keyof T.OpenVocabRevisionsResponse,
      true
    >),
  ],
  [
    'OpenVocabCreateRequest',
    keys({ name: true, body: true, description: true } satisfies Record<
      keyof T.OpenVocabCreateRequest,
      true
    >),
  ],
  [
    'OpenVocabSaveRequest',
    keys({ expected_revision: true, body: true, description: true } satisfies Record<
      keyof T.OpenVocabSaveRequest,
      true
    >),
  ],
  [
    'OpenVocabCloneRequest',
    keys({
      new_name: true,
      revision: true,
      source: true,
      description: true,
      from_project: true,
    } satisfies Record<keyof T.OpenVocabCloneRequest, true>),
  ],
  [
    'OpenVocabActivateRequest',
    keys({ revision: true, expected_active: true, force: true } satisfies Record<
      keyof T.OpenVocabActivateRequest,
      true
    >),
  ],
  [
    'OpenVocabRollbackRequest',
    keys({ expected_active: true } satisfies Record<
      keyof T.OpenVocabRollbackRequest,
      true
    >),
  ],
  [
    'OpenVocabDeactivateRequest',
    keys({ expected_active: true } satisfies Record<
      keyof T.OpenVocabDeactivateRequest,
      true
    >),
  ],
  [
    'OpenVocabValidateRequest',
    keys({ name: true, body: true } satisfies Record<
      keyof T.OpenVocabValidateRequest,
      true
    >),
  ],
  [
    'OpenVocabActivateResponse',
    keys({
      axis: true,
      active: true,
      source: true,
      activated_at: true,
      previous: true,
      config_revision: true,
      stale: true,
      applied: true,
      validation: true,
    } satisfies Record<keyof T.OpenVocabActivateResponse, true>),
  ],
  [
    'OpenVocabTestRequest',
    keys({
      target: true,
      image_id: true,
      image_base64: true,
      image_max_side: true,
      dedup_iou: true,
      gating: true,
    } satisfies Record<keyof T.OpenVocabTestRequest, true>),
  ],
  [
    'OpenVocabTestHit',
    keys({
      bbox_norm: true,
      score: true,
      selected: true,
      drop_reason: true,
      mask_polygon: true,
    } satisfies Record<keyof T.OpenVocabTestHit, true>),
  ],
  [
    'OpenVocabTestGate',
    keys({ run: true, tier: true, reason: true } satisfies Record<
      keyof T.OpenVocabTestGate,
      true
    >),
  ],
  [
    'OpenVocabTestImage',
    keys({ width: true, height: true } satisfies Record<
      keyof T.OpenVocabTestImage,
      true
    >),
  ],
  [
    'OpenVocabTestResponse',
    keys({
      image: true,
      prompt: true,
      class_name: true,
      gate: true,
      hits: true,
      elapsed_ms: true,
      validation: true,
    } satisfies Record<keyof T.OpenVocabTestResponse, true>),
  ],
];

describe('open-vocabulary types vs the vendored contract', () => {
  it.each(CASES)('%s has exactly the served properties', (name, declared) => {
    const s = schemas[name];
    expect(s, `schema ${name}`).toBeDefined();
    expect(declared).toEqual(Object.keys(s!.properties ?? {}).sort());
  });

  it('required served keys are non-optional on the types that carry them', () => {
    const pins: [string, string[]][] = [
      [
        'OpenVocabSummary',
        required<T.OpenVocabSummary>({
          name: true,
          source: true,
          read_only: true,
          revision: true,
          etag: true,
          display_name: true,
          n_targets: true,
          n_enabled_targets: true,
          run_on_ingest: true,
          active: true,
        }),
      ],
      [
        'OpenVocabTestResponse',
        required<T.OpenVocabTestResponse>({
          image: true,
          prompt: true,
          class_name: true,
          gate: true,
          hits: true,
          elapsed_ms: true,
          validation: true,
        }),
      ],
      [
        'OpenVocabTestHit',
        required<T.OpenVocabTestHit>({ bbox_norm: true, score: true, selected: true }),
      ],
      [
        'OpenVocabSaveRequest',
        required<T.OpenVocabSaveRequest>({ expected_revision: true, body: true }),
      ],
      [
        'OpenVocabCreateRequest',
        required<T.OpenVocabCreateRequest>({ name: true, body: true }),
      ],
      ['OpenVocabCloneRequest', ['new_name']],
      [
        'OpenVocabList',
        required<T.OpenVocabList>({
          sets: true,
          templates: true,
          active: true,
          config_revision: true,
          segmenter: true,
        }),
      ],
      [
        'OpenVocabSchema',
        required<T.OpenVocabSchema>({
          fields: true,
          max_enabled_targets_ceiling: true,
          vocabulary: true,
        }),
      ],
      [
        'OpenVocabVocabulary',
        required<T.OpenVocabVocabulary>({
          statuses: true,
          drop_reasons: true,
          gate_reasons: true,
        }),
      ],
      ['VocabularyOption', required<T.VocabularyOption>({ value: true, label: true })],
      [
        'SegmenterAvailability',
        required<T.SegmenterAvailability>({ configured: true, reachable: true }),
      ],
    ];
    for (const [name, declared] of pins) {
      expect(declared, name).toEqual([...(schemas[name]!.required ?? [])].sort());
    }
  });

  it('the field scope and type unions are the served enums', () => {
    const props = schemas.OpenVocabFieldSchema!.properties!;
    const scopes: T.OpenVocabFieldScope[] = ['set', 'target', 'gating', 'tier3_hit_rate'];
    const types: T.OpenVocabFieldType[] = [
      'string',
      'int',
      'float',
      'bool',
      'string_list',
    ];
    expect([...scopes].sort()).toEqual([...props.scope!.enum!].sort());
    expect([...types].sort()).toEqual([...props.type!.enum!].sort());
  });

  it('the closed value sets are the served enums', () => {
    const dropReasons: T.OpenVocabDropReason[] = [
      'below_min_score',
      'too_small',
      'too_large',
      'nms',
      'over_max',
      'cross_target_nms',
      'agree_existing',
      'skipped_locked',
    ];
    const gateReasons: T.OpenVocabGateReason[] = [
      'disabled',
      'no_parent_class',
      'vlm_no',
      'hit_rate',
    ];
    const statuses: T.OpenVocabStatus[] = ['pending', 'done', 'skipped_gate', 'failed'];
    const anyOfEnum = (p: unknown): string[] =>
      (p as { anyOf: { enum?: string[] }[] }).anyOf.find((a) => a.enum)!.enum!;
    expect([...dropReasons].sort()).toEqual(
      anyOfEnum(schemas.OpenVocabTestHit!.properties!.drop_reason).sort(),
    );
    expect([...gateReasons].sort()).toEqual(
      anyOfEnum(schemas.OpenVocabTestGate!.properties!.reason).sort(),
    );
    const statusEnum = (
      schemas.ReprocessFilter!.properties!.open_vocab_status as {
        items: { enum: string[] };
      }
    ).items.enum;
    expect([...statuses].sort()).toEqual([...statusEnum].sort());
  });
});
