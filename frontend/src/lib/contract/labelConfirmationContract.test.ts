/**
 * `types_labelConfirmation.ts` vs the generated contract (#119). Each key
 * map is compile-time exact against its type and pinned to the schema's
 * property set, so a served rename fails here instead of rendering "—".
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import itemWire from '$contracts/json/item_wire.json';
import type * as T from '$lib/types_labelConfirmation';
import { VLM_SCOPES } from '$lib/types_labelConfirmation';
import { RAW_CROP_KEYS } from '$lib/api';
import { CORE_REVIEW_TABS } from '$lib/reviewTabs';

type Schema = {
  properties?: Record<string, { enum?: string[] }>;
  required?: string[];
};
const doc = spec as unknown as {
  components: { schemas: Record<string, Schema> };
  paths: Record<string, Record<string, { description?: string }>>;
};
const schemas = doc.components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'VlmPolicy',
    keys<keyof T.VlmPolicy>({
      scope: true,
      conf_max: true,
      per_cluster: true,
      max_crops_per_day: true,
      sample_frac: true,
      revision: true,
    }),
  ],
  [
    'VlmPolicyUpdate',
    keys<keyof T.VlmPolicyUpdate>({
      scope: true,
      conf_max: true,
      per_cluster: true,
      max_crops_per_day: true,
      sample_frac: true,
      expected_revision: true,
    }),
  ],
  [
    'AuditStartRequest',
    keys<keyof T.AuditStartRequest>({ min_per_class: true, sample_size: true }),
  ],
  [
    'AuditStratum',
    keys<keyof T.AuditStratum>({
      detector_class: true,
      available: true,
      sampled: true,
      short_of_floor: true,
    }),
  ],
  [
    'AuditStartResponse',
    keys<keyof T.AuditStartResponse>({
      batch_id: true,
      min_per_class: true,
      requested: true,
      sampled: true,
      strata: true,
    }),
  ],
  [
    'AuditClassStat',
    keys<keyof T.AuditClassStat>({
      name: true,
      n: true,
      correct: true,
      precision: true,
      ci_low: true,
      ci_high: true,
      insufficient_sample: true,
    }),
  ],
  [
    'AuditReport',
    keys<keyof T.AuditReport>({
      audited: true,
      pending: true,
      min_per_class: true,
      detector: true,
      vlm: true,
      confusion: true,
      outcomes: true,
    }),
  ],
];

describe('label-confirmation types vs the generated contract', () => {
  for (const [schema, expected] of CASES) {
    it(`${schema} has exactly the keys the frontend type declares`, () => {
      const served = schemas[schema];
      expect(served, `${schema} missing from the contract`).toBeDefined();
      expect(Object.keys(served!.properties ?? {}).sort()).toEqual(expected);
    });
  }

  it('the scope ids are the served enum, in the served order', () => {
    expect(schemas.VlmPolicy!.properties!.scope!.enum).toEqual([...VLM_SCOPES]);
    expect(schemas.VlmPolicyUpdate!.properties!.scope!.enum).toEqual([...VLM_SCOPES]);
  });

  it('every served required AuditReport and AuditClassStat key is non-optional in the type', () => {
    // A compile-time half: these literals must satisfy the types.
    const stat: T.AuditClassStat = {
      name: 'widget_a',
      n: 1,
      correct: 1,
      precision: 1,
      ci_low: 0.2,
      ci_high: 1,
      insufficient_sample: true,
    };
    const report: T.AuditReport = {
      audited: 1,
      pending: 0,
      min_per_class: 30,
      detector: [stat],
      vlm: [],
      confusion: { widget_a: { widget_a: 1 } },
      outcomes: { agree: 1 },
    };
    expect(Object.keys(report).sort()).toEqual(
      [...(schemas.AuditReport!.required ?? [])].sort(),
    );
    expect(Object.keys(stat).sort()).toEqual(
      [...(schemas.AuditClassStat!.required ?? [])].sort(),
    );
  });
});

describe('the item wire carries the detector class (#119)', () => {
  it('serves the three detector keys, and RawCrop declares them', () => {
    for (const k of ['detector_class_name', 'detector_class_id', 'detector_confidence']) {
      expect(itemWire.item_keys).toContain(k);
      expect(RAW_CROP_KEYS).toContain(k);
    }
  });
});

describe('the detector_disagreements review tab (#119)', () => {
  it('is a tab id the review route documents, and the frontend knows it', () => {
    const route = doc.paths['/curation/projects/{project}/review/{tab}'];
    const text = JSON.stringify(route);
    expect(text).toContain('detector_disagreements');
    expect(CORE_REVIEW_TABS.map((t) => t.endpointId)).toContain('detector_disagreements');
  });
});
