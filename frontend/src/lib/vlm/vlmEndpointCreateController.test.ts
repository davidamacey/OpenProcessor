/**
 * The create page's controller: the body is seeded from each served schema
 * row's `default`; live validation posts `{name: <typed name>, body}`
 * (null while the name is empty) so the server's own name issues surface;
 * Test connection probes the draft; Create posts `{name, description,
 * body}` and returns the new doc; a 409 `name_conflict` / 422
 * `validation_failed` show the served message and report.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { VALIDATE_DEBOUNCE_MS } from '$lib/config/configEditor.svelte';
import {
  catalogFixture,
  cleanReport,
  docFixture,
  listFixture,
  probeFixture,
  schemaFixture,
} from '$lib/test/fixtures/vlm';
import {
  createVlmEndpointCreator,
  type VlmCreateDeps,
} from './vlmEndpointCreateController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

afterEach(() => vi.useRealTimers());

function setup(over: Partial<VlmCreateDeps> = {}) {
  const base = {
    getVlmEndpointSchema: vi.fn().mockResolvedValue(schemaFixture()),
    listVlmEndpoints: vi.fn().mockResolvedValue(listFixture()),
    getVlmCatalog: vi.fn().mockResolvedValue(catalogFixture()),
    validateVlmEndpoint: vi.fn().mockResolvedValue({
      validation: cleanReport(),
      locality: 'unknown',
      sends_images_externally: false,
    }),
    createVlmEndpoint: vi
      .fn()
      .mockResolvedValue(docFixture({ name: 'fresh', revision: 1 })),
  };
  const deps = { ...base, ...over } as typeof base;
  return { c: createVlmEndpointCreator(deps as unknown as Partial<VlmCreateDeps>), deps };
}

describe('VlmEndpointCreator', () => {
  it('seeds the body from the served defaults and loads the pickers', async () => {
    const { c } = setup();
    await c.load();
    expect(c.draftBody).toMatchObject({
      base_url: '',
      model: '',
      api_key_ref: null,
      allow_external: false,
      json_mode: 'auto',
      max_images_per_call: 8,
      timeout_s: 240,
    });
    expect(c.list?.secret_refs).toHaveLength(2);
    expect(c.catalog?.entries).toHaveLength(2);
  });

  it('a failed schema read is the load error; a failed list read only drops that picker', async () => {
    const bad = setup({
      getVlmEndpointSchema: vi
        .fn()
        .mockRejectedValue(
          refusal(503, { error: 'config_store_unavailable', message: 'Down.' }),
        ),
    });
    await bad.c.load();
    expect(bad.c.loadError).toBe('Down.');
    expect(bad.c.schema).toBeNull();

    const ok = setup({ listVlmEndpoints: vi.fn().mockRejectedValue(new Error('x')) });
    await ok.c.load();
    expect(ok.c.loadError).toBeNull();
    expect(ok.c.list).toBeNull();
  });

  it('live validation posts the typed name (null while empty), once per burst', async () => {
    vi.useFakeTimers();
    const { c, deps } = setup();
    await c.load();
    c.setField('base_url', 'http://x');
    await vi.advanceTimersByTimeAsync(VALIDATE_DEBOUNCE_MS);
    expect(deps.validateVlmEndpoint.mock.calls[0]![0].name).toBeNull();
    c.setName(' taken ');
    c.setName(' taken_vlm ');
    await vi.advanceTimersByTimeAsync(VALIDATE_DEBOUNCE_MS);
    expect(deps.validateVlmEndpoint).toHaveBeenCalledTimes(2);
    const [req, probe] = deps.validateVlmEndpoint.mock.calls[1]!;
    expect(req.name).toBe('taken_vlm');
    expect(req.body.base_url).toBe('http://x');
    expect(probe).toBe(false);
  });

  it('shows the served name issue from validation', async () => {
    vi.useFakeTimers();
    const issue = {
      code: 'name_conflict',
      severity: 'error' as const,
      field: 'name',
      message: 'That name is taken.',
      detail: {},
      bypassable: false,
    };
    const { c } = setup({
      validateVlmEndpoint: vi.fn().mockResolvedValue({
        validation: { ok: false, errors: [issue], warnings: [], force_allowed: false },
        locality: null,
        sends_images_externally: false,
      }),
    });
    await c.load();
    c.setName('local_vlm');
    await vi.advanceTimersByTimeAsync(VALIDATE_DEBOUNCE_MS);
    expect(c.report?.errors[0]?.message).toBe('That name is taken.');
  });

  it('test connection posts the draft with probe=true and keeps the served probe', async () => {
    const { c, deps } = setup({
      validateVlmEndpoint: vi.fn().mockResolvedValue({
        validation: cleanReport(),
        locality: 'private',
        sends_images_externally: false,
        probe: probeFixture(),
      }),
    });
    await c.load();
    c.setName('fresh');
    await c.testConnection();
    const [req, probe] = deps.validateVlmEndpoint.mock.calls.at(-1)!;
    expect(req.name).toBe('fresh');
    expect(probe).toBe(true);
    expect(c.checks.probe?.ok).toBe(true);
  });

  it('Create posts {name, description, body} and returns the new doc', async () => {
    const { c, deps } = setup();
    await c.load();
    c.setName(' fresh ');
    c.setDescription(' new one ');
    c.setField('model', 'example/new');
    const doc = await c.create();
    expect(doc?.name).toBe('fresh');
    expect(deps.createVlmEndpoint).toHaveBeenCalledWith({
      name: 'fresh',
      description: 'new one',
      body: { ...c.draftBody, model: 'example/new' },
    });
  });

  it('Create is unavailable without a name and sends nothing', async () => {
    const { c, deps } = setup();
    await c.load();
    expect(c.canCreate).toBe(false);
    expect(await c.create()).toBeNull();
    expect(deps.createVlmEndpoint).not.toHaveBeenCalled();
  });

  it('409 name_conflict and 422 validation_failed show the served message and report', async () => {
    const report = {
      ok: false,
      errors: [
        {
          code: 'vlm_base_url_invalid',
          severity: 'error' as const,
          field: 'base_url',
          message: 'Not a URL.',
          detail: {},
          bypassable: false,
        },
      ],
      warnings: [],
      force_allowed: false,
    };
    const { c } = setup({
      createVlmEndpoint: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(409, { error: 'name_conflict', message: 'Taken.' }),
        )
        .mockRejectedValueOnce(
          refusal(422, { error: 'validation_failed', message: 'Invalid.', report }),
        ),
    });
    await c.load();
    c.setName('fresh');
    expect(await c.create()).toBeNull();
    expect(c.createError).toBe('Taken.');
    expect(c.createReport).toBeNull();
    expect(await c.create()).toBeNull();
    expect(c.createError).toBe('Invalid.');
    expect(c.createReport?.errors[0]?.code).toBe('vlm_base_url_invalid');
  });
});
