/**
 * v2 bake-off wrappers (OpenProcessor #34 §7): URL, query params and the
 * run body reach the wire exactly as served routes expect.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  bakeoffBaselineModels,
  bakeoffEvalDatasets,
  bakeoffMatrix,
  bakeoffProfiles,
  bakeoffResults,
  bakeoffRun,
  bakeoffRuns,
  bakeoffStatus,
  bakeoffTrainedModels,
} from './api';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

function mockFetch(body: unknown = {}) {
  const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(ok(body)));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}
const urlOf = (m: ReturnType<typeof mockFetch>, i = 0) => m.mock.calls[i]![0] as string;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('bakeoff v2 wrappers', () => {
  it('lists profiles', async () => {
    const f = mockFetch({ profiles: [], count: 0, default_profile: null });
    await bakeoffProfiles();
    expect(urlOf(f)).toBe(`${API_PREFIX}/bakeoff/profiles`);
  });

  it('scopes baselines to a profile, and omits it when unset', async () => {
    const f = mockFetch({ baselines: [], count: 0 });
    await bakeoffBaselineModels('generic');
    await bakeoffBaselineModels(undefined);
    expect(urlOf(f, 0)).toBe(`${API_PREFIX}/bakeoff/baseline_models?profile=generic`);
    expect(urlOf(f, 1)).toBe(`${API_PREFIX}/bakeoff/baseline_models`);
  });

  it('lists eval datasets, optionally by source', async () => {
    const f = mockFetch({ datasets: [], count: 0 });
    await bakeoffEvalDatasets();
    await bakeoffEvalDatasets('external');
    expect(urlOf(f, 0)).toBe(`${API_PREFIX}/bakeoff/eval_datasets`);
    expect(urlOf(f, 1)).toBe(`${API_PREFIX}/bakeoff/eval_datasets?source=external`);
  });

  it('lists trained models with dataset_id and limit', async () => {
    const f = mockFetch({ models: [], count: 0 });
    await bakeoffTrainedModels({ datasetId: 'export:20260924T233203Z', limit: 100 });
    await bakeoffTrainedModels();
    expect(urlOf(f, 0)).toBe(
      `${API_PREFIX}/bakeoff/trained_models?dataset_id=export%3A20260924T233203Z&limit=100`,
    );
    expect(urlOf(f, 1)).toBe(`${API_PREFIX}/bakeoff/trained_models`);
  });

  it('posts the run body verbatim', async () => {
    const f = mockFetch({ status: 'enqueued', job_id: 'j' });
    const body = {
      datasets: [{ id: 'export:a' }],
      models: [
        { source: 'run' as const, run_id: 'r1' },
        { source: 'baseline' as const, name: 'b1' },
      ],
      profile: 'generic',
    };
    await bakeoffRun(body);
    expect(urlOf(f)).toBe(`${API_PREFIX}/bakeoff/run`);
    const init = f.mock.calls[0]![1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual(body);
  });

  it('encodes job ids and passes dataset_id to results', async () => {
    const f = mockFetch({});
    await bakeoffStatus('a/b');
    await bakeoffRuns();
    await bakeoffMatrix('j1');
    await bakeoffResults('j1', 'external:curated/set');
    await bakeoffResults('j1');
    expect(urlOf(f, 0)).toBe(`${API_PREFIX}/bakeoff/status/a%2Fb`);
    expect(urlOf(f, 1)).toBe(`${API_PREFIX}/bakeoff/runs`);
    expect(urlOf(f, 2)).toBe(`${API_PREFIX}/bakeoff/matrix/j1`);
    expect(urlOf(f, 3)).toBe(
      `${API_PREFIX}/bakeoff/results/j1?dataset_id=external%3Acurated%2Fset`,
    );
    expect(urlOf(f, 4)).toBe(`${API_PREFIX}/bakeoff/results/j1`);
  });
});
