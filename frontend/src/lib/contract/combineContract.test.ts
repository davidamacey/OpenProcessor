/**
 * `types_combine.ts` vs the vendored OpenAPI (OpenProcessor P4). Each key
 * map is compile-time exact against its interface and pinned to the
 * schema's property set, so a served rename fails here. The two request
 * bodies are `additionalProperties: false` (an unknown key is a 422), so
 * they are also driven through the real wrappers and the JSON they send is
 * checked against the declared keys.
 *
 * Not pinned: the contract leaves `CombinePreview.sources/target/dedup`
 * and `CombineJobResponse.report/next_steps` untyped (bare objects), so
 * `CombinePreviewSource` and friends are pinned by the fixtures in the
 * controller tests instead (plan question P4-6).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import { previewCombine, startCombine } from '$lib/api_combine';
import type * as T from '$lib/types_combine';

type Schema = {
  properties?: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
};
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const REQUEST = {
  target: true,
  sources: true,
  class_mapping: true,
  target_classes: true,
  dedup: true,
  dedup_iou: true,
  holdout: true,
  settings_from: true,
} satisfies Record<keyof T.CombineRequest, true>;

const CASES: [string, string[]][] = [
  ['CombineRequest', keys(REQUEST)],
  [
    'CombineStartRequest',
    keys({ ...REQUEST, expected_preview_sha: true } satisfies Record<
      keyof T.CombineStartRequest,
      true
    >),
  ],
  [
    'CombineSource',
    keys({ project: true, include: true } satisfies Record<keyof T.CombineSource, true>),
  ],
  [
    'CombineInclude',
    keys({ label_states: true } satisfies Record<keyof T.CombineInclude, true>),
  ],
  [
    'CombineTarget',
    keys({ slug: true, display_name: true, description: true } satisfies Record<
      keyof T.CombineTarget,
      true
    >),
  ],
  [
    'CombineIssue',
    keys({
      code: true,
      id: true,
      severity: true,
      project: true,
      message: true,
      detail: true,
    } satisfies Record<keyof T.CombineIssue, true>),
  ],
  [
    'CombineStartResponse',
    keys({ job_id: true, target: true } satisfies Record<
      keyof T.CombineStartResponse,
      true
    >),
  ],
  [
    'CombinePreview',
    keys({
      ok: true,
      errors: true,
      warnings: true,
      preview_sha: true,
      suggested_mapping: true,
      sources: true,
      target: true,
      dedup: true,
      bytes: true,
    } satisfies Record<keyof T.CombinePreview, true>),
  ],
  [
    'CombineJobResponse',
    keys({
      job_id: true,
      status: true,
      phase: true,
      done: true,
      total: true,
      started_at: true,
      finished_at: true,
      sources: true,
      target: true,
      error: true,
      report: true,
      next_steps: true,
    } satisfies Record<keyof T.CombineJobResponse, true>),
  ],
  [
    'LabeledChoice',
    keys({ value: true, label: true, description: true } satisfies Record<
      keyof T.CombineMappingActionChoice,
      true
    >),
  ],
];

describe('types_combine.ts matches the vendored contract', () => {
  for (const [name, expected] of CASES) {
    it(`${name} keys`, () => {
      const s = schemas[name];
      expect(s, `${name} missing from the vendored OpenAPI`).toBeDefined();
      expect(Object.keys(s!.properties ?? {}).sort()).toEqual(expected);
    });
  }

  it('the enums the form offers are the contract enums', () => {
    const prop = (schema: string, key: string) =>
      (schemas[schema]!.properties as Record<string, { enum?: string[] }>)[key]!.enum;
    expect(prop('CombineRequest', 'dedup')).toEqual([
      'content_hash',
      'none',
    ] satisfies T.CombineDedupMode[]);
    expect(prop('CombineRequest', 'holdout')).toEqual([
      'preserve_union',
      'recompute',
      'none',
    ] satisfies T.CombineHoldoutMode[]);
    expect(prop('CombineInclude', 'label_states')).toEqual([
      'all',
      'validated_only',
    ] satisfies T.CombineLabelStates[]);
  });
});

describe('strict request bodies send only declared keys', () => {
  afterEach(() => vi.unstubAllGlobals());

  function capture(): () => Record<string, unknown> {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({}), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    return () => JSON.parse(String((fetchMock.mock.calls[0]![1] as RequestInit).body));
  }

  const FULL: T.CombineRequest = {
    target: { slug: 'merged', display_name: 'Merged', description: 'd' },
    sources: [
      { project: 'a', include: { label_states: 'validated_only' } },
      { project: 'b' },
    ],
    class_mapping: {
      a: [
        { dataset_class: 'widget', action: 'create', new_class_name: 'widget' },
        { dataset_class: 'tag', action: 'skip' },
      ],
    },
    target_classes: ['widget'],
    dedup: 'none',
    dedup_iou: 0.5,
    holdout: 'recompute',
    settings_from: 'a',
  };

  function expectDeclared(body: Record<string, unknown>, schema: string): void {
    const s = schemas[schema]!;
    expect(s.additionalProperties).toBe(false);
    const allowed = Object.keys(s.properties ?? {});
    for (const k of Object.keys(body)) expect(allowed).toContain(k);
  }

  it('POST /projects/combine/preview (CombineRequest and its nested bodies)', async () => {
    const body = capture();
    await previewCombine(FULL);
    const sent = body();
    expectDeclared(sent, 'CombineRequest');
    expectDeclared(sent.target as Record<string, unknown>, 'CombineTarget');
    for (const src of sent.sources as Record<string, unknown>[]) {
      expectDeclared(src, 'CombineSource');
      if (src.include)
        expectDeclared(src.include as Record<string, unknown>, 'CombineInclude');
    }
    for (const row of (sent.class_mapping as Record<string, Record<string, unknown>[]>)
      .a!)
      expectDeclared(row, 'ClassMappingEntry');
  });

  it('POST /projects/combine (CombineStartRequest requires expected_preview_sha)', async () => {
    const body = capture();
    await startCombine({ ...FULL, expected_preview_sha: 'sha' });
    const sent = body();
    expectDeclared(sent, 'CombineStartRequest');
    expect(schemas.CombineStartRequest!.required).toContain('expected_preview_sha');
    expect(sent.expected_preview_sha).toBe('sha');
  });
});
