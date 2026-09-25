import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { createBakeoffController, type BakeoffApi } from './bakeoffController.svelte';
import {
  ACCEPTED,
  COMPARISON,
  DS_CURRENT,
  EVAL_DATASETS,
  MATRIX,
  status,
  trainedModel,
} from '$lib/test/fixtures/bakeoff';

function fakeApi(over: Partial<BakeoffApi> = {}): BakeoffApi {
  return {
    profiles: vi.fn(async () => ({
      profiles: [],
      count: 0,
      default_profile: 'generic',
      default_error: null,
    })),
    baselines: vi.fn(async () => ({ baselines: [], count: 0 })),
    datasets: vi.fn(async () => ({
      datasets: EVAL_DATASETS,
      count: EVAL_DATASETS.length,
    })),
    trainedModels: vi.fn(async (p: { datasetId?: string } = {}) => ({
      models: [
        trainedModel(
          p.datasetId
            ? {
                for_dataset: {
                  dataset_id: p.datasetId,
                  same_export: true,
                  same_frozen_test: null,
                  n_classes_mapped: 2,
                  train_test_overlap: null,
                },
              }
            : {},
        ),
      ],
      count: 1,
    })),
    run: vi.fn(async () => ACCEPTED),
    status: vi.fn(async () => status({ state: 'done', progress: { done: 2, total: 2 } })),
    runs: vi.fn(async () => ({ runs: [] })),
    results: vi.fn(async () => COMPARISON),
    matrix: vi.fn(async () => MATRIX),
    ...over,
  } as BakeoffApi;
}

const controllers: ReturnType<typeof createBakeoffController>[] = [];
function make(api: BakeoffApi) {
  const c = createBakeoffController(api, { pollMs: 60_000 });
  controllers.push(c);
  return c;
}
afterEach(() => {
  for (const c of controllers.splice(0)) c.destroy();
});

describe('bakeoffController', () => {
  it('preselects the served default profile and the current export, with its facts', async () => {
    const api = fakeApi();
    const c = make(api);
    await c.init();
    expect(c.state.profile).toBe('generic');
    expect(api.baselines).toHaveBeenCalledWith('generic');
    expect(c.state.selectedDatasets).toEqual([DS_CURRENT]);
    expect(api.trainedModels).toHaveBeenCalledWith({ datasetId: DS_CURRENT, limit: 100 });
    expect(c.state.facts[DS_CURRENT]['run-a']?.n_classes_mapped).toBe(2);
  });

  it('posts the built body, then polls to done and loads matrix + results', async () => {
    const api = fakeApi();
    const c = make(api);
    await c.init();
    c.setRunSelected('run-a', true);
    expect(await c.submit()).toBe(true);
    expect(api.run).toHaveBeenCalledWith({
      datasets: [{ id: DS_CURRENT }],
      models: [{ source: 'run', run_id: 'run-a' }],
      profile: 'generic',
    });
    expect(c.state.activeJob).toBe('job-1');
    await vi.waitFor(() => expect(c.state.comparison).not.toBeNull());
    expect(api.status).toHaveBeenCalledWith('job-1');
    expect(c.state.activeStatus?.state).toBe('done');
    expect(api.matrix).toHaveBeenCalledWith('job-1');
    expect(api.results).toHaveBeenCalledWith('job-1', DS_CURRENT);
    expect(c.state.matrix).toEqual(MATRIX);
  });

  it('keeps polling while queued/running and stops at a terminal state', async () => {
    const states = ['queued', 'running', 'error'] as const;
    let i = 0;
    const api = fakeApi({
      status: vi.fn(async () =>
        status({ state: states[Math.min(i++, 2)], error: 'boom' }),
      ),
    });
    const c = make(api);
    c.state.activeJob = 'job-1';
    await c.pollOnce();
    expect(c.state.activeStatus?.state).toBe('queued');
    await c.pollOnce();
    expect(api.runs).not.toHaveBeenCalled();
    await c.pollOnce();
    expect(c.state.activeStatus?.state).toBe('error');
    // Terminal: the run list refreshes once; an errored job has no matrix.
    expect(api.runs).toHaveBeenCalledTimes(1);
    expect(api.matrix).not.toHaveBeenCalled();
  });

  it('shows the served detail verbatim when the run is rejected', async () => {
    const detail = 'single_cls run over 2 classes cannot be scored per class';
    const api = fakeApi({
      run: vi.fn(async () => {
        throw new ApiError(422, '/curation/bakeoff/run', { detail });
      }),
    });
    const c = make(api);
    await c.init();
    c.setRunSelected('run-a', true);
    expect(await c.submit()).toBe(false);
    expect(c.state.runError).toBe(detail);
    expect(c.state.activeJob).toBeNull();
  });

  it('marks a 409 result as legacy rather than an error', async () => {
    const api = fakeApi({
      matrix: vi.fn(async () => {
        throw new ApiError(409, '/m', { detail: 'unsupported schema' });
      }),
      results: vi.fn(async () => {
        throw new ApiError(409, '/r', { detail: 'unsupported schema' });
      }),
    });
    const c = make(api);
    await c.viewRun('old-job');
    expect(c.state.comparisonLegacy).toBe(true);
    expect(c.state.comparisonError).toBeNull();
    expect(c.state.matrixError).toBeNull();
  });

  it('adopts a queued run found in the run list', async () => {
    const api = fakeApi({
      runs: vi.fn(async () => ({
        runs: [
          {
            job_id: 'live',
            state: 'queued' as const,
            profile: null,
            datasets: [],
            models: [],
            started_at: null,
            finished_at: null,
          },
        ],
      })),
      status: vi.fn(async () => status({ job_id: 'live', state: 'queued' })),
    });
    const c = make(api);
    await c.refreshRuns();
    expect(c.state.activeJob).toBe('live');
    await vi.waitFor(() => expect(api.status).toHaveBeenCalledWith('live'));
  });
});
