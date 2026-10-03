/**
 * The `/settings/models` controller: load; activation pins the served
 * revision and `expected_active`, sends `acknowledge_external` only when
 * the operator checked it and records the served ack refusal; rollback and
 * turn-off bodies; the health re-poll and `/methods` reset only after a
 * successful write; delete's served `in_use` projects; probe; the local
 * model (restart banner, poll until `poll_after_s` is null, `force` only
 * on the retry after `vlm_catalog_does_not_fit`); and the SSE wake-ups
 * (`vlm.changed` registry vs local_vlm, `config.changed axis=vlm` only).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import {
  activeFixture,
  catalogFixture,
  listFixture,
  localFixture,
  probeFixture,
  summaryFixture,
} from '$lib/test/fixtures/vlm';
import { healthStore } from '$stores/health.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import type { VlmEvent } from './vlmEvents';
import { createVlmModels, type VlmModelsDeps } from './vlmModelsController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

beforeEach(() => {
  vi.spyOn(healthStore, 'poll').mockResolvedValue(undefined as never);
  vi.spyOn(strategiesStore, 'reset').mockImplementation(() => {});
});
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function setup(over: Partial<VlmModelsDeps> = {}) {
  let emit: (e: VlmEvent) => void = () => {};
  const close = vi.fn();
  const base = {
    listVlmEndpoints: vi.fn().mockResolvedValue(listFixture()),
    getVlmCatalog: vi.fn().mockResolvedValue(catalogFixture()),
    getLocalVlm: vi.fn().mockResolvedValue(localFixture()),
    selectLocalVlm: vi.fn().mockResolvedValue(localFixture()),
    clearLocalVlmSelection: vi.fn().mockResolvedValue(localFixture()),
    cloneVlmEndpoint: vi.fn().mockResolvedValue(summaryFixture()),
    deleteVlmEndpoint: vi.fn().mockResolvedValue(undefined),
    probeVlmEndpoint: vi.fn().mockResolvedValue(probeFixture()),
    getConfigVocabulary: vi.fn().mockResolvedValue({ model_choices: [], labels: {} }),
    getActiveVlm: vi.fn().mockResolvedValue(activeFixture()),
    activateVlm: vi.fn().mockResolvedValue(activeFixture()),
    rollbackVlm: vi.fn().mockResolvedValue(activeFixture()),
    deactivateVlm: vi
      .fn()
      .mockResolvedValue(activeFixture({ active: { name: null, revision: null } })),
    subscribe: vi.fn((cb: (e: VlmEvent) => void) => {
      emit = cb;
      return { close };
    }),
  };
  const deps = { ...base, ...over } as typeof base;
  // `onchanged` is left to its default so the health / methods hook is real.
  const models = createVlmModels(deps as unknown as Partial<VlmModelsDeps>);
  return { models, deps, close, emit: (e: VlmEvent) => emit(e) };
}

describe('VlmModels load', () => {
  it('reads the registry, the catalog (with the local status) and the active ref', async () => {
    const { models, deps } = setup();
    await models.load();
    expect(deps.listVlmEndpoints).toHaveBeenCalledTimes(1);
    expect(deps.getVlmCatalog).toHaveBeenCalledTimes(1);
    expect(models.list?.endpoints.map((e) => e.name)).toEqual([
      'local_vlm',
      'cloud_vlm',
      'env_default',
    ]);
    expect(models.catalog?.entries).toHaveLength(2);
    expect(models.local?.configured).toBe(true);
    expect(models.active.active?.active).toEqual({ name: 'local_vlm', revision: 3 });
  });

  it('shows a served load error instead of an empty list', async () => {
    const { models } = setup({
      listVlmEndpoints: vi
        .fn()
        .mockRejectedValue(
          refusal(503, { error: 'config_store_unavailable', message: 'Down.' }),
        ),
    });
    await models.load();
    expect(models.list).toBeNull();
    expect(models.loadError).toBe('Down.');
  });

  it('reads the model choices and their scope labels from the vocabulary', async () => {
    const choice = {
      role: 'vlm',
      label: 'VLM',
      scope: 'per_run',
      current: 'local_vlm',
      choices: [],
      settable: true,
    };
    const { models, deps } = setup({
      getConfigVocabulary: vi.fn().mockResolvedValue({
        model_choices: [choice],
        labels: { scope: { per_run: 'Per run' } },
      }),
    });
    await models.loadModelChoices();
    expect(deps.getConfigVocabulary).toHaveBeenCalledWith(false);
    expect(models.modelChoices).toEqual([choice]);
    expect(models.modelChoiceLabels).toEqual({ per_run: 'Per run' });
  });
});

describe('activation', () => {
  it('pins the served revision and expected_active; the ack key is absent unless checked', async () => {
    const { models, deps } = setup();
    await models.load();
    expect(await models.active.activate('cloud_vlm', 1, false)).toBe(true);
    expect(deps.activateVlm).toHaveBeenCalledWith('cloud_vlm', {
      revision: 1,
      expected_active: { name: 'local_vlm', revision: 3 },
      force: false,
    });
    await models.active.activate('cloud_vlm', 1, false, { acknowledge_external: true });
    expect(deps.activateVlm.mock.calls[1]![1]).toEqual({
      revision: 1,
      expected_active: { name: 'local_vlm', revision: 3 },
      force: false,
      acknowledge_external: true,
    });
  });

  it('an env endpoint has no revision: null is sent as served', async () => {
    const { models, deps } = setup();
    await models.load();
    await models.active.activate('env_default', null, false);
    expect(deps.activateVlm.mock.calls[0]![1]).toMatchObject({ revision: null });
  });

  it('records the served 422 ack refusal, then resends with the acknowledgement', async () => {
    const { models, deps } = setup({
      activateVlm: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(422, {
            error: 'vlm_external_not_acknowledged',
            message: 'cloud_vlm sends crops outside this deployment.',
            endpoint: 'cloud_vlm',
            activate_via: 'settings/models',
          }),
        )
        .mockResolvedValueOnce(activeFixture()),
    });
    await models.load();
    expect(await models.active.activate('cloud_vlm', 1, false)).toBe(false);
    expect(models.active.actionError).toBe(
      'cloud_vlm sends crops outside this deployment.',
    );
    expect(models.active.errorDetail?.error).toBe('vlm_external_not_acknowledged');
    expect(
      await models.active.activate('cloud_vlm', 1, false, { acknowledge_external: true }),
    ).toBe(true);
    expect(models.active.errorDetail).toBeNull();
    expect(deps.activateVlm.mock.calls[1]![1]).toMatchObject({
      acknowledge_external: true,
    });
  });

  it('force is sent only when the caller passes it (after a served force_allowed)', async () => {
    const { models, deps } = setup({
      activateVlm: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(422, {
            error: 'validation_failed',
            message: 'Not ready.',
            report: {
              ok: false,
              errors: [],
              warnings: [],
              force_allowed: true,
            },
          }),
        )
        .mockResolvedValueOnce(activeFixture()),
    });
    await models.load();
    await models.active.activate('cloud_vlm', 1, false);
    expect(models.active.activateReport?.force_allowed).toBe(true);
    await models.active.activate('cloud_vlm', 1, true);
    expect(
      deps.activateVlm.mock.calls.map((c) => (c[1] as { force: boolean }).force),
    ).toEqual([false, true]);
  });

  it('re-polls /health and drops /methods only after a successful write', async () => {
    const { models } = setup({
      activateVlm: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(409, { error: 'active_conflict', message: 'Changed.' }),
        )
        .mockResolvedValueOnce(activeFixture()),
    });
    await models.load();
    await models.active.activate('cloud_vlm', 1, false);
    expect(healthStore.poll).not.toHaveBeenCalled();
    expect(strategiesStore.reset).not.toHaveBeenCalled();
    await models.active.activate('cloud_vlm', 1, false);
    expect(healthStore.poll).toHaveBeenCalledTimes(1);
    expect(strategiesStore.reset).toHaveBeenCalledTimes(1);
  });

  it('an active_conflict re-reads the active ref', async () => {
    const { models, deps } = setup({
      activateVlm: vi
        .fn()
        .mockRejectedValue(
          refusal(409, { error: 'active_conflict', message: 'Changed.' }),
        ),
    });
    await models.load();
    const reads = deps.getActiveVlm.mock.calls.length;
    await models.active.activate('cloud_vlm', 1, false);
    expect(deps.getActiveVlm.mock.calls.length).toBe(reads + 1);
  });

  it('rollback sends expected_active and reloads; turn off sends expected_active', async () => {
    const { models, deps } = setup();
    await models.load();
    expect(await models.rollback()).toBe(true);
    expect(deps.rollbackVlm).toHaveBeenCalledWith({
      expected_active: { name: 'local_vlm', revision: 3 },
    });
    expect(deps.listVlmEndpoints.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(await models.deactivate()).toBe(true);
    expect(deps.deactivateVlm).toHaveBeenCalledWith({
      expected_active: { name: 'local_vlm', revision: 3 },
    });
    expect(models.active.active?.active.name).toBeNull();
  });
});

describe('clone / delete / probe', () => {
  it('delete sends the served revision; stored rows only reach the wrapper', async () => {
    const { models, deps } = setup();
    await models.load();
    expect(await models.remove({ name: 'local_vlm', revision: 3 })).toBe(true);
    expect(deps.deleteVlmEndpoint).toHaveBeenCalledWith('local_vlm', 3);
    // An env row has no revision: nothing is sent.
    expect(await models.remove({ name: 'env_default', revision: null })).toBe(false);
    expect(deps.deleteVlmEndpoint).toHaveBeenCalledTimes(1);
  });

  it('a 409 in_use keeps the served message and the served project slugs', async () => {
    const { models } = setup({
      deleteVlmEndpoint: vi.fn().mockRejectedValue(
        refusal(409, {
          error: 'in_use',
          message: 'local_vlm is active in other projects.',
          projects: ['alpha', 'beta'],
        }),
      ),
    });
    await models.load();
    expect(await models.remove({ name: 'local_vlm', revision: 3 })).toBe(false);
    expect(models.deleteError).toBe('local_vlm is active in other projects.');
    expect(models.deleteProjects).toEqual(['alpha', 'beta']);
  });

  it('a 403 read_only is shown verbatim', async () => {
    const { models } = setup({
      deleteVlmEndpoint: vi
        .fn()
        .mockRejectedValue(
          refusal(403, { error: 'read_only', message: 'Set by the deployment.' }),
        ),
    });
    await models.load();
    await models.remove({ name: 'local_vlm', revision: 3 });
    expect(models.deleteError).toBe('Set by the deployment.');
    expect(models.deleteProjects).toEqual([]);
  });

  it('clone sends only the keys the strict body declares', async () => {
    const { models, deps } = setup();
    await models.load();
    await models.clone({ name: 'local_vlm', source: null }, ' copy ', ' note ');
    expect(deps.cloneVlmEndpoint).toHaveBeenCalledWith('local_vlm', {
      new_name: 'copy',
      revision: null,
      description: 'note',
    });
  });

  it('probe keeps the served result and re-reads the list', async () => {
    const { models, deps } = setup();
    await models.load();
    const reads = deps.listVlmEndpoints.mock.calls.length;
    await models.probe('local_vlm');
    expect(models.probes.local_vlm?.ok).toBe(true);
    expect(deps.listVlmEndpoints.mock.calls.length).toBe(reads + 1);
    expect(models.probing).toBeNull();
  });

  it('a 429 probe_busy is the served message', async () => {
    const { models } = setup({
      probeVlmEndpoint: vi
        .fn()
        .mockRejectedValue(
          refusal(429, { error: 'probe_busy', message: 'A probe is running.' }),
        ),
    });
    await models.load();
    await models.probe('cloud_vlm');
    expect(models.probeErrors.cloud_vlm).toBe('A probe is running.');
    expect(models.probes.cloud_vlm).toBeUndefined();
  });
});

describe('local model', () => {
  const pending = localFixture({
    restart_required: true,
    poll_after_s: 5,
    reason: 'A restart is needed to serve vision-30b.',
    desired: {
      catalog_id: 'vision-30b',
      requested_at: '2026-10-01T10:00:00Z',
      command: 'docker compose up -d vlm',
    },
  });

  it('select posts the catalog id without force and shows the served restart state', async () => {
    const { models, deps } = setup({
      selectLocalVlm: vi.fn().mockResolvedValue(pending),
      getVlmCatalog: vi
        .fn()
        .mockResolvedValueOnce(catalogFixture())
        .mockResolvedValue(catalogFixture({ local: pending })),
    });
    await models.load();
    expect(await models.selectLocal('vision-30b', false)).toBe(true);
    expect(deps.selectLocalVlm).toHaveBeenCalledWith({ catalog_id: 'vision-30b' });
    expect(models.local?.restart_required).toBe(true);
    expect(models.local?.desired?.command).toBe('docker compose up -d vlm');
  });

  it('polls GET /vlm/local every poll_after_s and stops when it is null', async () => {
    vi.useFakeTimers();
    const getLocalVlm = vi
      .fn()
      .mockResolvedValueOnce(pending)
      .mockResolvedValueOnce(localFixture({ poll_after_s: null }));
    const { models, deps } = setup({
      // The first read is pending; the read after the poll saw it serve.
      getVlmCatalog: vi
        .fn()
        .mockResolvedValueOnce(catalogFixture({ local: pending }))
        .mockResolvedValue(catalogFixture()),
      getLocalVlm,
    });
    await models.loadCatalog();
    expect(getLocalVlm).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(4999);
    expect(getLocalVlm).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getLocalVlm).toHaveBeenCalledTimes(1);
    // Still pending: the next read is another poll_after_s away.
    await vi.advanceTimersByTimeAsync(5000);
    expect(getLocalVlm).toHaveBeenCalledTimes(2);
    // poll_after_s is now null: no further reads, however long we wait.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(getLocalVlm).toHaveBeenCalledTimes(2);
    expect(models.local?.poll_after_s).toBeNull();
    expect(deps.getVlmCatalog.mock.calls.length).toBeGreaterThanOrEqual(2);
    models.stop();
  });

  it('never claims serving: the served flags are all there is', async () => {
    const { models } = setup({ selectLocalVlm: vi.fn().mockResolvedValue(pending) });
    await models.load();
    await models.selectLocal('vision-30b', false);
    expect(models.catalog?.entries.find((e) => e.id === 'vision-30b')?.serving).toBe(
      false,
    );
  });

  it('does-not-fit keeps the served message and code; force is sent only on the retry', async () => {
    const { models, deps } = setup({
      selectLocalVlm: vi
        .fn()
        .mockRejectedValueOnce(
          refusal(422, {
            error: 'vlm_catalog_does_not_fit',
            message: 'vision-30b needs 80 GB; this GPU has 48.',
          }),
        )
        .mockResolvedValueOnce(pending),
    });
    await models.load();
    expect(await models.selectLocal('vision-30b', false)).toBe(false);
    expect(models.localError).toBe('vision-30b needs 80 GB; this GPU has 48.');
    expect(models.localErrorCode).toBe('vlm_catalog_does_not_fit');
    expect(await models.selectLocal('vision-30b', true)).toBe(true);
    expect(deps.selectLocalVlm.mock.calls.map((c) => c[0])).toEqual([
      { catalog_id: 'vision-30b' },
      { catalog_id: 'vision-30b', force: true },
    ]);
    expect(models.localError).toBeNull();
  });

  it('409 no_local_vlm and 422 unknown_catalog_id are the served messages', async () => {
    for (const [status, code, message] of [
      [409, 'no_local_vlm', 'No local model server is configured.'],
      [422, 'unknown_catalog_id', 'Unknown catalog id.'],
    ] as const) {
      const { models } = setup({
        selectLocalVlm: vi
          .fn()
          .mockRejectedValue(refusal(status, { error: code, message })),
      });
      await models.load();
      await models.selectLocal('x', false);
      expect(models.localError).toBe(message);
      expect(models.localErrorCode).toBe(code);
    }
  });

  it('cancel clears the request and shows the served status', async () => {
    const { models, deps } = setup();
    await models.load();
    expect(await models.clearLocal()).toBe(true);
    expect(deps.clearLocalVlmSelection).toHaveBeenCalledTimes(1);
  });
});

describe('wake-ups', () => {
  it('vlm.changed registry re-reads the list only', async () => {
    const { models, deps, emit } = setup();
    models.start();
    await vi.waitFor(() => expect(deps.listVlmEndpoints).toHaveBeenCalled());
    const [lists, catalogs, actives] = [
      deps.listVlmEndpoints.mock.calls.length,
      deps.getVlmCatalog.mock.calls.length,
      deps.getActiveVlm.mock.calls.length,
    ];
    emit({
      type: 'vlm.changed',
      axis: 'registry',
      topic: 'project',
      project: null,
    } as never);
    await vi.waitFor(() =>
      expect(deps.listVlmEndpoints.mock.calls.length).toBe(lists + 1),
    );
    expect(deps.getVlmCatalog.mock.calls.length).toBe(catalogs);
    expect(deps.getActiveVlm.mock.calls.length).toBe(actives);
    models.stop();
  });

  it('vlm.changed local_vlm re-reads the catalog / local status only', async () => {
    const { models, deps, emit } = setup();
    models.start();
    await vi.waitFor(() => expect(deps.getVlmCatalog).toHaveBeenCalled());
    const [lists, catalogs, actives] = [
      deps.listVlmEndpoints.mock.calls.length,
      deps.getVlmCatalog.mock.calls.length,
      deps.getActiveVlm.mock.calls.length,
    ];
    emit({ type: 'vlm.changed', axis: 'local_vlm' } as never);
    await vi.waitFor(() =>
      expect(deps.getVlmCatalog.mock.calls.length).toBe(catalogs + 1),
    );
    expect(deps.listVlmEndpoints.mock.calls.length).toBe(lists);
    expect(deps.getActiveVlm.mock.calls.length).toBe(actives);
    models.stop();
  });

  it('config.changed axis=vlm re-reads the active ref only; other axes are ignored', async () => {
    const { models, deps, emit } = setup();
    models.start();
    await vi.waitFor(() => expect(deps.getActiveVlm).toHaveBeenCalled());
    const [lists, catalogs, actives] = [
      deps.listVlmEndpoints.mock.calls.length,
      deps.getVlmCatalog.mock.calls.length,
      deps.getActiveVlm.mock.calls.length,
    ];
    emit({ type: 'config.changed', axis: 'prompt_pack' } as never);
    emit({ type: 'config.changed', axis: 'vlm' } as never);
    await vi.waitFor(() => expect(deps.getActiveVlm.mock.calls.length).toBe(actives + 1));
    expect(deps.listVlmEndpoints.mock.calls.length).toBe(lists);
    expect(deps.getVlmCatalog.mock.calls.length).toBe(catalogs);
    models.stop();
  });

  it('stop closes the subscription', () => {
    const { models, close } = setup();
    models.start();
    models.stop();
    expect(close).toHaveBeenCalledTimes(1);
  });
});
