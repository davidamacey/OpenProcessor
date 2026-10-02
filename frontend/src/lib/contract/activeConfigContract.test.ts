/**
 * The typed `ActiveConfigResponse` (prompt packs, region profiles, rollback,
 * deactivate) and `RegionProfileSchema` (the profile editor's field schema)
 * vs the vendored OpenAPI (OpenProcessor f14f4ddc). Key maps are
 * compile-time exact against our types; vocabularies are pinned to the
 * served enums so a backend rename fails here, not as a blank control.
 */
import { describe, expect, it } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type {
  ActivateResponse,
  ActiveConfigResponse,
  ActiveSource,
  AppliedRuntime,
  ConfigAxis,
} from '$lib/types_config';
import type {
  ActivationImpact,
  ChoicesFrom,
  ProfileActivateResponse,
  AppliesWhen,
  Choice,
  ProfileFieldType,
  ProfileSchemaField,
  ProfileSchemaGroup,
  RegionProfileSchema,
} from '$lib/types_profiles';

type Prop = { enum?: string[]; anyOf?: { enum?: string[] }[] };
type Schema = { properties?: Record<string, Prop> };
type Spec = {
  components: { schemas: Record<string, Schema> };
  paths: Record<
    string,
    Record<string, { responses: Record<string, { content?: Record<string, unknown> }> }>
  >;
};
const doc = spec as unknown as Spec;
const schemas = doc.components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const declared = (name: string) => {
  const s = schemas[name];
  if (!s?.properties) throw new Error(`schema not found: ${name}`);
  return Object.keys(s.properties).sort();
};
const enumOf = (schema: string, prop: string): string[] => {
  const p = schemas[schema]!.properties![prop]!;
  return [...(p.enum ?? p.anyOf?.find((a) => a.enum)?.enum ?? [])].sort();
};

// Literal-union members of our open-ended types (the `string & {}` escape
// hatch is stripped by listing the known ids explicitly, compile-checked).
const FIELD_TYPES = [
  'string',
  'int',
  'float',
  'bool',
  'enum',
  'string_list',
  'int_list',
  'float_pair',
  'rgb',
] as const satisfies readonly ProfileFieldType[];
const CHOICES_FROM = [
  'detectors',
  'segmenters',
  'ocr_pipeline_models',
  'ocr_det_models',
  'ocr_rec_models',
  'registry_classes',
  'text_reader_modes',
  'vlm_catalog',
  'secret_refs',
] as const satisfies readonly ChoicesFrom[];
const APPLIES_WHEN = [
  'detector',
  'segmenter',
  'reads_text',
  'text_hint',
] as const satisfies readonly AppliesWhen[];
const SOURCES = ['stored', 'env', 'off'] as const satisfies readonly ActiveSource[];
const AXES = [
  'prompt_pack',
  'detection_profile',
  'vlm',
] as const satisfies readonly ConfigAxis[];

describe('ActiveConfigResponse', () => {
  it('keys match the schema', () => {
    expect(
      keys<keyof ActiveConfigResponse>({
        axis: true,
        active: true,
        source: true,
        activated_at: true,
        previous: true,
        config_revision: true,
        stale: true,
        applied: true,
      }),
    ).toEqual(declared('ActiveConfigResponse'));
  });

  it('AppliedRuntime keys match (host/applied_at are real served fields)', () => {
    expect(
      keys<keyof AppliedRuntime>({
        process: true,
        host: true,
        applied_config_revision: true,
        profile: true,
        pack: true,
        vlm: true,
        applied_at: true,
        lagging: true,
      }),
    ).toEqual(declared('AppliedRuntime'));
  });

  it('AppliedRuntime.vlm is required and nullable: null means the worker never reported a VLM axis', () => {
    const s = schemas['AppliedRuntime'] as Schema & { required: string[] };
    expect(s.required).toEqual(expect.arrayContaining(['profile', 'pack', 'vlm']));
    const vlm = s.properties!.vlm as { anyOf?: { type?: string; $ref?: string }[] };
    expect(vlm.anyOf?.some((a) => a.type === 'null')).toBe(true);
    expect(vlm.anyOf?.some((a) => a.$ref?.endsWith('/ActiveRef'))).toBe(true);
    // Ours: profile and pack are always objects, vlm alone may be null.
    const row: AppliedRuntime = {
      process: 'p',
      host: 'h',
      applied_config_revision: 1,
      profile: { name: null, revision: null },
      pack: { name: null, revision: null },
      vlm: null,
      applied_at: null,
      lagging: false,
    };
    expect(row.vlm).toBeNull();
  });

  it('source and axis are exactly the served enums', () => {
    expect([...SOURCES].sort()).toEqual(enumOf('ActiveConfigResponse', 'source'));
    expect([...AXES].sort()).toEqual(enumOf('ActiveConfigResponse', 'axis'));
  });

  it.each([
    '/prompt_packs/active',
    '/prompt_packs/active/rollback',
    '/region_profiles/active',
    '/region_profiles/active/rollback',
    '/region_profiles/deactivate',
  ])('%s publishes ActiveConfigResponse', (suffix) => {
    const entry = doc.paths[`/curation/projects/{project}${suffix}`];
    expect(entry).toBeDefined();
    const op = Object.values(entry!)[0]!;
    const body = JSON.stringify(op.responses['200']!.content);
    expect(body).toContain('#/components/schemas/ActiveConfigResponse');
  });
});

const activateSchemaRef = (suffix: string): string => {
  const entry = doc.paths[`/curation/projects/{project}${suffix}`];
  expect(entry).toBeDefined();
  const op = entry!['post']!;
  const content = op.responses['200']!.content as {
    'application/json': { schema: { $ref: string } };
  };
  return content['application/json'].schema.$ref;
};

describe('activate responses', () => {
  it('prompt_packs activate publishes ActivateResponse', () => {
    expect(activateSchemaRef('/prompt_packs/{name}/activate')).toBe(
      '#/components/schemas/ActivateResponse',
    );
  });

  it('region_profiles activate publishes RegionProfileActivateResponse (ActivateResponse + impact)', () => {
    expect(activateSchemaRef('/region_profiles/{name}/activate')).toBe(
      '#/components/schemas/RegionProfileActivateResponse',
    );
  });

  it('ActivateResponse keys match the schema', () => {
    expect(
      keys<keyof ActivateResponse>({
        axis: true,
        active: true,
        source: true,
        activated_at: true,
        previous: true,
        config_revision: true,
        stale: true,
        applied: true,
        validation: true,
      }),
    ).toEqual(declared('ActivateResponse'));
  });

  it('ProfileActivateResponse keys match the schema', () => {
    expect(
      keys<keyof ProfileActivateResponse>({
        axis: true,
        active: true,
        source: true,
        activated_at: true,
        previous: true,
        config_revision: true,
        stale: true,
        applied: true,
        validation: true,
        impact: true,
      }),
    ).toEqual(declared('RegionProfileActivateResponse'));
  });

  it('ActivationImpact keys match the schema', () => {
    expect(
      keys<keyof ActivationImpact>({
        items_total: true,
        by_profile: true,
        validated_items: true,
        unseeded_items: true,
        pending_items: true,
        pending_not_matching: true,
        stale_items: true,
        suggested_reprocess: true,
      }),
    ).toEqual(declared('ActivationImpact'));
  });

  it('every activate response requires validation (and impact for profiles)', () => {
    const req = (n: string) =>
      [...((schemas[n] as { required?: string[] }).required ?? [])].sort();
    expect(req('ActivateResponse')).toContain('validation');
    expect(req('RegionProfileActivateResponse')).toEqual(
      expect.arrayContaining(['validation', 'impact']),
    );
  });
});

describe('RegionProfileSchema', () => {
  it('is served as a typed schema on GET /region_profiles/schema', () => {
    const op = doc.paths['/curation/projects/{project}/region_profiles/schema']!.get!;
    expect(JSON.stringify(op.responses['200']!.content)).toContain(
      '#/components/schemas/RegionProfileSchema',
    );
  });

  it('top-level, field and group keys match', () => {
    expect(keys<keyof RegionProfileSchema>({ fields: true, groups: true })).toEqual(
      declared('RegionProfileSchema'),
    );
    expect(
      keys<keyof ProfileSchemaField>({
        field: true,
        label: true,
        group: true,
        type: true,
        default: true,
        min: true,
        max: true,
        enum: true,
        advanced: true,
        applies_when: true,
        choices_from: true,
        empty_choice: true,
        help: true,
      }),
    ).toEqual(declared('RegionProfileFieldSchema'));
    expect(keys<keyof ProfileSchemaGroup>({ id: true, label: true })).toEqual(
      declared('RegionProfileGroup'),
    );
    expect(keys<keyof Choice>({ id: true, label: true })).toEqual(
      declared('RegionProfileChoice'),
    );
  });

  it('field type, applies_when and choices_from vocabularies cover the served enums', () => {
    expect([...FIELD_TYPES].sort()).toEqual(enumOf('RegionProfileFieldSchema', 'type'));
    expect([...APPLIES_WHEN].sort()).toEqual(
      enumOf('RegionProfileFieldSchema', 'applies_when'),
    );
    const served = enumOf('RegionProfileFieldSchema', 'choices_from');
    expect(served.length).toBeGreaterThan(0);
    // Ours also names the VLM editor's lists (same renderer), so a superset.
    for (const c of served) expect(CHOICES_FROM as readonly string[]).toContain(c);
  });
});

describe('PUT /settings for the activation-backed axes', () => {
  it('the body is {defaults} and a null clears an axis (prompt_pack/detection_profile/vlm)', () => {
    expect(declared('CurationSettingsUpdateRequest')).toEqual(['defaults']);
    const defaults = schemas['CurationSettingsUpdateRequest']!.properties![
      'defaults'
    ] as {
      additionalProperties: { anyOf: { type: string }[] };
    };
    expect(defaults.additionalProperties.anyOf.map((a) => a.type).sort()).toEqual([
      'null',
      'string',
    ]);
  });

  it('GET /settings serves an open string map (activation names, `off`)', () => {
    const defaults = schemas['CurationSettingsResponse']!.properties!['defaults'] as {
      additionalProperties: { type: string };
    };
    expect(defaults.additionalProperties.type).toBe('string');
  });
});
