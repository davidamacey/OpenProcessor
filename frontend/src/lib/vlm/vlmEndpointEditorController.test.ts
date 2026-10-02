/**
 * One VLM endpoint's editor: schema, doc, active ref, the registry list and
 * the catalog load together (a failed list/catalog read is shown, not
 * fatal); the validate adapter returns the served report and keeps the
 * response's facts, always with `name: null`; "Test connection" posts the
 * draft with `probe=true` and shows the served probe (a `probe_busy` 429 is
 * the served message); "Probe saved" re-reads the doc without touching the
 * draft; a cleared "none" pick is the served empty_choice id; Save and its
 * conflict; the wake-ups.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { VALIDATE_DEBOUNCE_MS } from '$lib/config/configEditor.svelte';
import {
  activeFixture,
  bodyFixture,
  catalogFixture,
  cleanReport,
  docFixture,
  listFixture,
  probeFixture,
  revisionsFixture,
  schemaFixture,
} from '$lib/test/fixtures/vlm';
import type { VlmEvent } from './vlmEvents';
import {
  createVlmEndpointEditor,
  type VlmEditorDeps,
} from './vlmEndpointEditorController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

afterEach(() => vi.useRealTimers());

function setup(over: Partial<VlmEditorDeps> = {}) {
  let emit: (e: VlmEvent) => void = () => {};
  const base = {
    getVlmEndpointSchema: vi.fn().mockResolvedValue(schemaFixture()),
    getVlmEndpoint: vi.fn().mockResolvedValue(docFixture()),
    getVlmEndpointRevisions: vi.fn().mockResolvedValue(revisionsFixture()),
    getVlmEndpointRevision: vi.fn().mockResolvedValue(docFixture({ revision: 1 })),
    updateVlmEndpoint: vi
      .fn()
      .mockImplementation(
        async (_n: string, b: { body: ReturnType<typeof bodyFixture> }) =>
          docFixture({ revision: 4, body: b.body }),
      ),
    validateVlmEndpoint: vi.fn().mockResolvedValue({
      validation: cleanReport(),
      locality: 'private',
      sends_images_externally: false,
    }),
    probeVlmEndpoint: vi.fn().mockResolvedValue(probeFixture()),
    listVlmEndpoints: vi.fn().mockResolvedValue(listFixture()),
    getVlmCatalog: vi.fn().mockResolvedValue(catalogFixture()),
    getActiveVlm: vi.fn().mockResolvedValue(activeFixture()),
    activateVlm: vi.fn().mockResolvedValue(activeFixture()),
    rollbackVlm: vi.fn(),
    deactivateVlm: vi.fn(),
    onchanged: vi.fn(),
    subscribe: vi.fn((cb: (e: VlmEvent) => void) => {
      emit = cb;
      return { close: vi.fn() };
    }),
  };
  const deps = { ...base, ...over } as typeof base;
  const ed = createVlmEndpointEditor(
    'local_vlm',
    deps as unknown as Partial<VlmEditorDeps>,
  );
  return { ed, deps, emit: (e: VlmEvent) => emit(e) };
}

describe('VlmEndpointEditor load', () => {
  it('loads the schema, doc, active ref, list, catalog and revisions', async () => {
    const { ed, deps } = setup();
    await ed.load();
    expect(deps.getVlmEndpoint).toHaveBeenCalledWith('local_vlm');
    expect(ed.schema?.groups[0]?.id).toBe('connection');
    expect(ed.list?.secret_refs).toHaveLength(2);
    expect(ed.catalog?.entries).toHaveLength(2);
    expect(ed.revisions).toHaveLength(3);
    expect(ed.active.active?.active).toEqual({ name: 'local_vlm', revision: 3 });
    expect(ed.dirty).toBe(false);
  });

  it('a failed list or catalog read is shown, not fatal', async () => {
    const { ed } = setup({
      listVlmEndpoints: vi
        .fn()
        .mockRejectedValue(
          refusal(503, { error: 'config_store_unavailable', message: 'Down.' }),
        ),
    });
    await ed.load();
    expect(ed.doc).not.toBeNull();
    expect(ed.loadError).toBeNull();
    expect(ed.list).toBeNull();
    expect(ed.catalog).not.toBeNull();
    expect(ed.extrasError).toBe('Down.');
  });

  it('an unknown endpoint is the served 404 message', async () => {
    const { ed } = setup({
      getVlmEndpoint: vi
        .fn()
        .mockRejectedValue(
          refusal(404, { error: 'not_found', message: 'No such endpoint.' }),
        ),
    });
    await ed.load();
    expect(ed.doc).toBeNull();
    expect(ed.loadError).toBe('No such endpoint.');
  });

  it('an env endpoint is read-only with no revisions', async () => {
    const { ed, deps } = setup({
      getVlmEndpoint: vi.fn().mockResolvedValue(
        docFixture({
          name: 'env_default',
          source: 'env',
          read_only: true,
          revision: null,
        }),
      ),
    });
    await ed.load();
    expect(ed.editable).toBe(false);
    expect(ed.revisions).toBeNull();
    expect(deps.getVlmEndpointRevisions).not.toHaveBeenCalled();
  });
});

describe('validation and test connection', () => {
  it('live validation posts {name: null, body} without probe once per burst and keeps the facts', async () => {
    vi.useFakeTimers();
    const { ed, deps } = setup({
      validateVlmEndpoint: vi.fn().mockResolvedValue({
        validation: { ...cleanReport(), warnings: [] },
        locality: 'external',
        sends_images_externally: true,
      }),
    });
    await ed.load();
    ed.setField('model', 'example/a');
    ed.setField('model', 'example/b');
    await vi.advanceTimersByTimeAsync(VALIDATE_DEBOUNCE_MS);
    expect(deps.validateVlmEndpoint).toHaveBeenCalledTimes(1);
    const [req, probe] = deps.validateVlmEndpoint.mock.calls[0]!;
    expect(req).toEqual({ name: null, body: { ...bodyFixture(), model: 'example/b' } });
    expect(probe).toBe(false);
    expect(ed.report?.ok).toBe(true);
    expect(ed.checks.facts).toEqual({
      locality: 'external',
      sends_images_externally: true,
    });
  });

  it('test connection sends probe=true with name null and shows the served probe and report', async () => {
    const issue = {
      code: 'vlm_model_not_listed',
      severity: 'warning' as const,
      field: 'model',
      message: 'The endpoint does not list this model.',
      detail: {},
      bypassable: true,
    };
    const { ed, deps } = setup({
      validateVlmEndpoint: vi.fn().mockResolvedValue({
        validation: { ok: true, errors: [], warnings: [issue], force_allowed: false },
        locality: 'private',
        sends_images_externally: false,
        probe: probeFixture({ ok: false, model_listed: false }),
      }),
    });
    await ed.load();
    await ed.testConnection();
    const [req, probe] = deps.validateVlmEndpoint.mock.calls.at(-1)!;
    expect(req).toEqual({ name: null, body: bodyFixture() });
    expect(probe).toBe(true);
    expect(ed.checks.probe?.ok).toBe(false);
    expect(ed.report?.warnings[0]?.code).toBe('vlm_model_not_listed');
    expect(ed.checks.testing).toBe(false);
  });

  it('a 429 probe_busy is the served message and leaves the report alone', async () => {
    const { ed } = setup({
      validateVlmEndpoint: vi
        .fn()
        .mockRejectedValue(
          refusal(429, { error: 'probe_busy', message: 'A probe is running.' }),
        ),
    });
    await ed.load();
    await ed.testConnection();
    expect(ed.checks.testError).toBe('A probe is running.');
    expect(ed.checks.probe).toBeNull();
    expect(ed.report).toEqual(docFixture().validation);
  });

  it('an edit clears a shown probe (it was for a different draft)', async () => {
    const { ed } = setup({
      validateVlmEndpoint: vi.fn().mockResolvedValue({
        validation: cleanReport(),
        locality: 'private',
        sends_images_externally: false,
        probe: probeFixture(),
      }),
    });
    await ed.load();
    await ed.testConnection();
    expect(ed.checks.probe).not.toBeNull();
    ed.setField('model', 'example/other');
    expect(ed.checks.probe).toBeNull();
  });

  it('a cleared "none" pick stores the served empty_choice id (null), not an empty string', async () => {
    const { ed } = setup();
    await ed.load();
    ed.setField('api_key_ref', 'CLOUD_VLM_KEY');
    expect(ed.draftBody.api_key_ref).toBe('CLOUD_VLM_KEY');
    ed.setField('api_key_ref', '');
    expect(ed.draftBody.api_key_ref).toBeNull();
  });
});

describe('probe saved', () => {
  it('probes by name, then re-reads the doc and leaves the draft alone', async () => {
    const { ed, deps } = setup();
    await ed.load();
    ed.setField('model', 'example/edited');
    deps.getVlmEndpoint.mockResolvedValue(docFixture({ last_probe: probeFixture() }));
    await ed.probeSaved();
    expect(deps.probeVlmEndpoint).toHaveBeenCalledWith('local_vlm');
    expect(ed.probeResult?.ok).toBe(true);
    expect(ed.doc?.last_probe?.ok).toBe(true);
    expect(ed.draftBody.model).toBe('example/edited');
    expect(ed.probing).toBe(false);
  });

  it('a 429 probe_busy is the served message and nothing is re-read', async () => {
    const { ed, deps } = setup({
      probeVlmEndpoint: vi
        .fn()
        .mockRejectedValue(
          refusal(429, { error: 'probe_busy', message: 'A probe is running.' }),
        ),
    });
    await ed.load();
    const reads = deps.getVlmEndpoint.mock.calls.length;
    await ed.probeSaved();
    expect(ed.probeError).toBe('A probe is running.');
    expect(deps.getVlmEndpoint.mock.calls.length).toBe(reads);
  });
});

describe('save and wake-ups', () => {
  it('Save sends expected_revision and the draft; a 409 revision_conflict is offered back', async () => {
    const { ed, deps } = setup({
      updateVlmEndpoint: vi.fn().mockRejectedValue(
        refusal(409, {
          error: 'revision_conflict',
          message: 'Someone saved first.',
          current_revision: 5,
        }),
      ),
    });
    await ed.load();
    ed.setField('model', 'example/x');
    expect(await ed.save()).toBe(false);
    expect(deps.updateVlmEndpoint).toHaveBeenCalledWith('local_vlm', {
      expected_revision: 3,
      description: 'The GPU box',
      body: { ...bodyFixture(), model: 'example/x' },
    });
    expect(ed.conflict).toEqual({ message: 'Someone saved first.', currentRevision: 5 });
  });

  it('a registry wake-up with no endpoint name re-reads this doc when the draft is clean', async () => {
    const { ed, deps, emit } = setup();
    await ed.load();
    ed.start();
    await vi.waitFor(() => expect(deps.subscribe).toHaveBeenCalled());
    const reads = deps.getVlmEndpoint.mock.calls.length;
    deps.getVlmEndpoint.mockResolvedValue(docFixture({ revision: 4 }));
    emit({ type: 'vlm.changed', axis: 'registry' } as never);
    await vi.waitFor(() =>
      expect(deps.getVlmEndpoint.mock.calls.length).toBeGreaterThan(reads),
    );
    await vi.waitFor(() => expect(ed.doc?.revision).toBe(4));
    ed.stop();
  });

  it('a dirty draft gets a notice, not a silent overwrite', async () => {
    const { ed, deps, emit } = setup();
    ed.start();
    await vi.waitFor(() => expect(deps.getVlmEndpointRevisions).toHaveBeenCalledTimes(1));
    ed.setField('model', 'example/mine');
    emit({ type: 'vlm.changed', axis: 'registry', name: 'local_vlm' } as never);
    await vi.waitFor(() => expect(ed.remoteChanged).toBe(true));
    expect(ed.draftBody.model).toBe('example/mine');
    expect(deps.getVlmEndpoint).toHaveBeenCalledTimes(1);
    ed.stop();
  });

  it('local_vlm events and other axes are ignored', async () => {
    const { ed, deps, emit } = setup();
    await ed.load();
    ed.start();
    const actives = deps.getActiveVlm.mock.calls.length;
    emit({ type: 'vlm.changed', axis: 'local_vlm' } as never);
    emit({ type: 'config.changed', axis: 'prompt_pack' } as never);
    await Promise.resolve();
    expect(deps.getActiveVlm.mock.calls.length).toBe(actives);
    ed.stop();
  });
});
