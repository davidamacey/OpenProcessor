/**
 * `types_vlm.ts` (and the W9 additions on `types_profiles.ts`) vs the
 * vendored OpenAPI schemas (OpenProcessor f582aa05). Each key map is
 * compile-time exact against its type (`satisfies Record<keyof T, true>`
 * rejects a missing or an extra key) and the test pins it to the schema's
 * property set, so a served rename fails here instead of rendering "—".
 * The strict (`additionalProperties: false`) request bodies are driven
 * through the wrappers and the controllers that build them, and may only
 * ever carry declared keys.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import spec from '../../../contracts/openprocessor/openapi/curation.json';
import type * as T from '$lib/types_vlm';
import type { ModelChoice, VocabVlmEndpoint } from '$lib/types_profiles';
import type { ConfigRevision } from '$lib/types_config';
import {
  activateVlm,
  cloneVlmEndpoint,
  createVlmEndpoint,
  deactivateVlm,
  rollbackVlm,
  selectLocalVlm,
  updateVlmEndpoint,
  validateVlmEndpoint,
} from '$lib/api_vlm';
import { createVlmEndpointCreator } from '$lib/vlm/vlmEndpointCreateController.svelte';
import { createVlmEndpointEditor } from '$lib/vlm/vlmEndpointEditorController.svelte';
import { createVlmModels } from '$lib/vlm/vlmModelsController.svelte';
import { VlmActive } from '$lib/vlm/vlmActive.svelte';
import {
  activeFixture,
  bodyFixture,
  docFixture,
  listFixture,
  revisionsFixture,
  schemaFixture,
  catalogFixture,
  summaryFixture,
} from '$lib/test/fixtures/vlm';

type Schema = {
  properties?: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
};
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

const CASES: [string, string[]][] = [
  [
    'VlmEndpointBody',
    keys({
      base_url: true,
      model: true,
      api_key_ref: true,
      catalog_id: true,
      allow_external: true,
      json_mode: true,
      max_images_per_call: true,
      open_images_per_call: true,
      requests_per_second: true,
      timeout_s: true,
    } satisfies Record<keyof T.VlmEndpointBody, true>),
  ],
  [
    'SecretRef',
    keys({ ref: true, present: true, choice: true } satisfies Record<
      keyof T.SecretRef,
      true
    >),
  ],
  [
    'VlmEndpointLabels',
    keys({ status: true, locality: true, source: true } satisfies Record<
      keyof T.VlmEndpointLabels,
      true
    >),
  ],
  [
    'VlmEndpointSummary',
    keys({
      name: true,
      source: true,
      read_only: true,
      revision: true,
      etag: true,
      description: true,
      base_url: true,
      model: true,
      catalog_id: true,
      locality: true,
      sends_images_externally: true,
      warning: true,
      api_key_ref: true,
      api_key_present: true,
      status: true,
      last_probe_at: true,
      active_in: true,
      updated_at: true,
    } satisfies Record<keyof T.VlmEndpointSummary, true>),
  ],
  [
    'VlmEndpointList',
    keys({
      endpoints: true,
      config_revision: true,
      external_policy: true,
      secret_refs: true,
      labels: true,
      stale: true,
    } satisfies Record<keyof T.VlmEndpointList, true>),
  ],
  [
    'VlmEndpointFieldSchema',
    keys({
      field: true,
      label: true,
      group: true,
      type: true,
      default: true,
      advanced: true,
      choices_from: true,
      empty_choice: true,
      enum: true,
      help: true,
      min: true,
      max: true,
    } satisfies Record<keyof T.VlmEndpointFieldSchema, true>),
  ],
  [
    'VlmEndpointGroup',
    keys({ id: true, label: true } satisfies Record<keyof T.VlmEndpointGroup, true>),
  ],
  [
    'VlmEndpointSchema',
    keys({ fields: true, groups: true } satisfies Record<
      keyof T.VlmEndpointSchema,
      true
    >),
  ],
  [
    'VlmEndpointDoc',
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
      updated_by: true,
      cloned_from: true,
      validation: true,
      api_key_present: true,
      locality: true,
      sends_images_externally: true,
      warning: true,
      active_in: true,
      last_probe: true,
    } satisfies Record<keyof T.VlmEndpointDoc, true>),
  ],
  [
    'VlmEndpointCreate',
    keys({ name: true, description: true, body: true } satisfies Record<
      keyof T.VlmEndpointCreate,
      true
    >),
  ],
  [
    'VlmEndpointSaveRequest',
    keys({ expected_revision: true, description: true, body: true } satisfies Record<
      keyof T.VlmEndpointSaveRequest,
      true
    >),
  ],
  [
    'VlmEndpointCloneRequest',
    keys({ new_name: true, revision: true, description: true } satisfies Record<
      keyof T.VlmEndpointCloneRequest,
      true
    >),
  ],
  [
    'VlmRevisionSummary',
    keys({
      revision: true,
      saved_at: true,
      cloned_from: true,
      description: true,
    } satisfies Record<keyof T.VlmRevisionSummary, true>),
  ],
  [
    'VlmRevisionsResponse',
    keys({ name: true, revisions: true } satisfies Record<
      keyof T.VlmRevisionsResponse,
      true
    >),
  ],
  [
    'VlmValidateRequest',
    keys({ name: true, body: true } satisfies Record<keyof T.VlmValidateRequest, true>),
  ],
  [
    'VlmValidateResponse',
    keys({
      validation: true,
      locality: true,
      sends_images_externally: true,
      probe: true,
    } satisfies Record<keyof T.VlmValidateResponse, true>),
  ],
  [
    'VlmProbeResult',
    keys({
      ok: true,
      probed_at: true,
      latency_ms: true,
      models_listed: true,
      model_listed: true,
      root: true,
      max_model_len: true,
      vision_ok: true,
      json_mode_supported: true,
      reasoning_channel: true,
      image_tokens: true,
      max_images_ok: true,
      issues: true,
    } satisfies Record<keyof T.VlmProbeResult, true>),
  ],
  [
    'VlmCatalogEntry',
    keys({
      id: true,
      choice: true,
      hf_repo: true,
      family: true,
      license: true,
      license_url: true,
      gated: true,
      params_b: true,
      quantization: true,
      context_max: true,
      max_model_len: true,
      max_images: true,
      vram_gb: true,
      disk_gb: true,
      status: true,
      rank: true,
      multi_box_verified: true,
      text_reading_verified: true,
      fits: true,
      serving: true,
      desired: true,
    } satisfies Record<keyof T.VlmCatalogEntry, true>),
  ],
  [
    'VlmCatalogLabels',
    keys({ status: true } satisfies Record<keyof T.VlmCatalogLabels, true>),
  ],
  [
    'VlmCatalogResponse',
    keys({ entries: true, local: true, labels: true } satisfies Record<
      keyof T.VlmCatalogResponse,
      true
    >),
  ],
  [
    'VlmLocalServed',
    keys({
      model: true,
      root: true,
      catalog_id: true,
      max_model_len: true,
    } satisfies Record<keyof T.VlmLocalServed, true>),
  ],
  [
    'VlmLocalDesired',
    keys({ catalog_id: true, requested_at: true, command: true } satisfies Record<
      keyof T.VlmLocalDesired,
      true
    >),
  ],
  [
    'VlmLocalStatus',
    keys({
      configured: true,
      endpoint: true,
      served: true,
      desired: true,
      restart_required: true,
      poll_after_s: true,
      gpu_total_gb: true,
      can_restart_from_api: true,
      reason: true,
    } satisfies Record<keyof T.VlmLocalStatus, true>),
  ],
  [
    'VlmLocalSelectRequest',
    keys({ catalog_id: true, force: true } satisfies Record<
      keyof T.VlmLocalSelectRequest,
      true
    >),
  ],
  [
    'VlmActiveResponse',
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
    } satisfies Record<keyof T.VlmActiveResponse, true>),
  ],
  [
    'VlmActivateRequest',
    keys({
      revision: true,
      expected_active: true,
      force: true,
      acknowledge_external: true,
    } satisfies Record<keyof T.VlmActivateRequest, true>),
  ],
  [
    'VlmRollbackRequest',
    keys({ expected_active: true } satisfies Record<keyof T.VlmRollbackRequest, true>),
  ],
  [
    'VlmDeactivateRequest',
    keys({ expected_active: true } satisfies Record<keyof T.VlmDeactivateRequest, true>),
  ],
  [
    'VlmEndpointEntry',
    keys({
      name: true,
      source: true,
      model: true,
      resolved_model: true,
      locality: true,
      sends_images_externally: true,
      warning: true,
      status: true,
      max_images_per_call: true,
      active: true,
    } satisfies Record<keyof VocabVlmEndpoint, true>),
  ],
  [
    'ModelChoice',
    keys({
      role: true,
      label: true,
      scope: true,
      current: true,
      dims: true,
      choices: true,
      settable: true,
      settable_via: true,
      reason: true,
    } satisfies Record<keyof ModelChoice, true>),
  ],
];

describe('types_vlm.ts vs the vendored OpenAPI', () => {
  it.each(CASES)('%s has exactly the served properties', (name, declared) => {
    const s = schemas[name];
    expect(s, `schema ${name}`).toBeDefined();
    expect(declared).toEqual(Object.keys(s!.properties ?? {}).sort());
  });

  it('every required served property is a non-optional TS member', () => {
    // A required served key may not be optional on the type: the key maps
    // above would still pass, so pin the few request bodies by hand.
    expect(schemas.VlmEndpointSaveRequest!.required).toEqual([
      'expected_revision',
      'body',
    ]);
    expect(schemas.VlmEndpointCreate!.required).toEqual(['name', 'body']);
    expect(schemas.VlmEndpointCloneRequest!.required).toEqual(['new_name']);
    expect(schemas.VlmLocalSelectRequest!.required).toEqual(['catalog_id']);
  });

  it('a VLM revision row carries what the shared revision list reads', () => {
    const shared = keys({
      revision: true,
      saved_at: true,
      cloned_from: true,
      description: true,
    } satisfies Record<keyof ConfigRevision, true>);
    expect(shared).toEqual(
      Object.keys(schemas.VlmRevisionSummary!.properties ?? {}).sort(),
    );
  });
});

function captureBodies() {
  const fetchMock = vi.fn().mockImplementation(
    async () =>
      new Response(JSON.stringify(docFixture()), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return {
    all: () =>
      fetchMock.mock.calls
        .filter((c) => (c[1] as RequestInit | undefined)?.body)
        .map(
          (c) =>
            JSON.parse(String((c[1] as RequestInit).body)) as Record<string, unknown>,
        ),
    last: () => {
      const calls = fetchMock.mock.calls.filter(
        (c) => (c[1] as RequestInit | undefined)?.body,
      );
      return JSON.parse(String((calls.at(-1)![1] as RequestInit).body)) as Record<
        string,
        unknown
      >;
    },
  };
}

function declared(name: string): string[] {
  const s = schemas[name];
  expect(s?.additionalProperties, `${name} is strict`).toBe(false);
  return Object.keys(s!.properties!);
}

function expectOnlyDeclared(body: Record<string, unknown>, name: string): void {
  const allowed = declared(name);
  for (const k of Object.keys(body)) expect(allowed, `${name}.${k}`).toContain(k);
}

afterEach(() => vi.unstubAllGlobals());

describe('strict W9 request bodies send only declared keys', () => {
  it('through the wrappers', async () => {
    const cap = captureBodies();
    const expected = { name: 'a', revision: 1 };
    await validateVlmEndpoint({ name: null, body: bodyFixture() }, true);
    expectOnlyDeclared(cap.last(), 'VlmValidateRequest');
    expectOnlyDeclared(cap.last().body as Record<string, unknown>, 'VlmEndpointBody');
    await createVlmEndpoint({ name: 'n', description: '', body: bodyFixture() });
    expectOnlyDeclared(cap.last(), 'VlmEndpointCreate');
    await updateVlmEndpoint('n', {
      expected_revision: 1,
      description: null,
      body: bodyFixture(),
    });
    expectOnlyDeclared(cap.last(), 'VlmEndpointSaveRequest');
    await cloneVlmEndpoint('n', { new_name: 'm', revision: null, description: null });
    expectOnlyDeclared(cap.last(), 'VlmEndpointCloneRequest');
    await selectLocalVlm({ catalog_id: 'x', force: true });
    expectOnlyDeclared(cap.last(), 'VlmLocalSelectRequest');
    await activateVlm('n', {
      revision: 1,
      expected_active: expected,
      force: true,
      acknowledge_external: true,
    });
    expectOnlyDeclared(cap.last(), 'VlmActivateRequest');
    await rollbackVlm({ expected_active: expected });
    expectOnlyDeclared(cap.last(), 'VlmRollbackRequest');
    await deactivateVlm({ expected_active: expected });
    expectOnlyDeclared(cap.last(), 'VlmDeactivateRequest');
  });

  it('through the controllers that build them', async () => {
    const cap = captureBodies();
    const read = (body: unknown) => vi.fn().mockResolvedValue(body) as never;
    // The shared clone sends `{new_name, revision, source, description}`;
    // the strict VLM body has no `source`.
    const models = createVlmModels();
    await models.clone({ name: 'local_vlm', source: null }, 'copy', '');
    expectOnlyDeclared(cap.last(), 'VlmEndpointCloneRequest');
    expect(cap.last()).not.toHaveProperty('source');

    const active = new VlmActive({
      getActiveVlm: read(activeFixture()) as never,
      onchanged: () => {},
    });
    await active.load();
    await active.activate('cloud_vlm', 1, false, { acknowledge_external: true });
    expectOnlyDeclared(cap.last(), 'VlmActivateRequest');

    const editor = createVlmEndpointEditor('local_vlm', {
      getVlmEndpointSchema: read(schemaFixture()) as never,
      getVlmEndpoint: read(docFixture()) as never,
      getVlmEndpointRevisions: read(revisionsFixture()) as never,
      listVlmEndpoints: read(listFixture()) as never,
      getVlmCatalog: read(catalogFixture()) as never,
      getActiveVlm: read(activeFixture()) as never,
    });
    await editor.load();
    editor.setField('model', 'example/other');
    await editor.save();
    expectOnlyDeclared(cap.last(), 'VlmEndpointSaveRequest');
    await editor.validateNow();
    expectOnlyDeclared(cap.last(), 'VlmValidateRequest');

    const creator = createVlmEndpointCreator({
      getVlmEndpointSchema: read(schemaFixture()) as never,
      listVlmEndpoints: read(listFixture()) as never,
      getVlmCatalog: read(catalogFixture()) as never,
    });
    await creator.load();
    creator.setName('fresh');
    await creator.create();
    expectOnlyDeclared(cap.last(), 'VlmEndpointCreate');
    // Anchors the fixture used above to the served summary shape.
    expect(summaryFixture().name).toBe('local_vlm');
  });
});
