/**
 * `types_models.ts` vs the vendored OpenAPI schemas (OpenProcessor
 * projects P2, §5.5 model sharing). Same technique as
 * `projectsContract.test.ts`: each key map is compile-time exact against
 * its interface and pinned to the schema's property set. Not pinned:
 * `ModelClassMappingSummary` (the per-entry `class_mapping` on the
 * untyped `GET /models/status` response) — see `types_models.ts`.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as M from '$lib/types_models';
import type { ConfigErrorDetail } from '$lib/types_config';

type Schema = { properties?: Record<string, unknown>; required?: string[] };
const schemas = (spec as { components: { schemas: Record<string, Schema> } }).components
  .schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'ModelSharingRequest',
    keys({ shared: true, expected_revision: true } satisfies Record<
      keyof M.ModelSharingRequest,
      true
    >),
  ],
  [
    'ModelSharingResponse',
    keys({
      name: true,
      project: true,
      shared: true,
      revision: true,
      used_by: true,
    } satisfies Record<keyof M.ModelSharingResponse, true>),
  ],
  [
    'ModelSharingUser',
    keys({ project: true, profile: true } satisfies Record<
      keyof M.ModelSharingUser,
      true
    >),
  ],
  [
    'ModelInUseDetail',
    keys({ error: true, message: true, projects: true, used_by: true } satisfies Record<
      keyof M.ModelInUseDetail,
      true
    >),
  ],
  [
    'ModelRevisionConflictDetail',
    keys({ error: true, message: true, current_revision: true } satisfies Record<
      keyof M.ModelRevisionConflictDetail,
      true
    >),
  ],
  [
    'ModelSharingUnavailableDetail',
    keys({ error: true, message: true } satisfies Record<
      keyof M.ModelSharingUnavailableDetail,
      true
    >),
  ],
  [
    'ModelClassMappingResponse',
    keys({
      model: true,
      model_project: true,
      project: true,
      entries: true,
      unmapped: true,
      not_covered: true,
      labels: true,
    } satisfies Record<keyof M.ModelClassMappingResponse, true>),
  ],
  [
    'ModelClassMappingEntry',
    keys({
      model_id: true,
      model_name: true,
      class_id: true,
      class_name: true,
      match: true,
    } satisfies Record<keyof M.ModelClassMappingEntry, true>),
  ],
  [
    'ModelClassMappingLabels',
    keys({ match: true } satisfies Record<keyof M.ModelClassMappingLabels, true>),
  ],
];

describe('types_models.ts matches the vendored OpenAPI', () => {
  it.each(CASES)('%s', (name, ours) => {
    const schema = schemas[name];
    expect(schema, `${name} missing from the vendored OpenAPI`).toBeDefined();
    expect(ours).toEqual(Object.keys(schema!.properties ?? {}).sort());
  });

  it('the sharing request requires exactly what the server requires', () => {
    const body: M.ModelSharingRequest = { shared: true, expected_revision: 1 };
    expect(body).toBeTruthy();
    expect(schemas.ModelSharingRequest!.required?.sort()).toEqual([
      'expected_revision',
      'shared',
    ]);
  });

  it('the sharing write publishes a typed 409 and 503 whose detail codes are ours', () => {
    const op = (
      spec as unknown as {
        paths: Record<string, Record<string, { responses: Record<string, unknown> }>>;
      }
    ).paths['/curation/projects/{project}/models/{model_name}/sharing']!.put!;
    const ref = (code: string) =>
      JSON.stringify(op.responses[code]).match(/schemas\/(\w+)/)?.[1];
    expect(ref('409')).toBe('ModelSharingConflictResponse');
    expect(ref('503')).toBe('ModelSharingUnavailableResponse');

    type Const = { const?: string };
    const code = (name: string) => (schemas[name]!.properties!.error as Const).const;
    const ours: Record<
      string,
      M.ModelSharingConflictDetail['error'] | 'config_store_unavailable'
    > = {
      ModelInUseDetail: 'in_use',
      ModelRevisionConflictDetail: 'revision_conflict',
      ModelSharingUnavailableDetail: 'config_store_unavailable',
    };
    for (const [name, c] of Object.entries(ours)) expect(code(name)).toBe(c);

    const conflict = schemas.ModelSharingConflictResponse!.properties!.detail as {
      anyOf: { $ref: string }[];
    };
    expect(conflict.anyOf.map((a) => a.$ref.split('/').pop()).sort()).toEqual([
      'ModelInUseDetail',
      'ModelRevisionConflictDetail',
    ]);
    // `in_use` names each project with its profile, not just the slug.
    expect(schemas.ModelInUseDetail!.required).toContain('used_by');
  });

  it('the served match kinds are exactly ModelClassMatch', () => {
    const entry = schemas.ModelClassMappingEntry!.properties!.match as {
      enum?: string[];
    };
    const ours: Record<M.ModelClassMatch, true> = {
      exact: true,
      case_insensitive: true,
      none: true,
    };
    expect(Object.keys(ours).sort()).toEqual([...(entry.enum ?? [])].sort());
  });

  it('ConfigErrorDetail carries owner_project and the project_owned_model code', () => {
    const k: keyof ConfigErrorDetail = 'owner_project';
    const props = schemas.ConfigErrorDetail!.properties!;
    expect(Object.keys(props)).toContain(k);
    expect((props.error as { enum: string[] }).enum).toContain('project_owned_model');
  });
});
