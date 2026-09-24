/**
 * BakeoffProfile wiring (OpenProcessor B1): the chosen profile must reach
 * the wire on both the baseline lookup and the run request, and an unset
 * profile must be omitted entirely so the evaluator's deployment default
 * applies.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, bakeoffBaselineModels, bakeoffProfiles, bakeoffRun } from './api';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('bakeoff profile wiring', () => {
  it('lists profiles from /bakeoff/profiles', async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok({ profiles: [], count: 0 }));
    vi.stubGlobal('fetch', fetchMock);
    await bakeoffProfiles();
    expect(fetchMock.mock.calls[0]![0]).toBe(`${API_PREFIX}/bakeoff/profiles`);
  });

  it('scopes baseline models to the chosen profile', async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok({ baselines: [], count: 0 }));
    vi.stubGlobal('fetch', fetchMock);
    await bakeoffBaselineModels('license_plate');
    expect(fetchMock.mock.calls[0]![0]).toBe(
      `${API_PREFIX}/bakeoff/baseline_models?profile=license_plate`,
    );
  });

  it('omits profile from the baseline lookup when unset', async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok({ baselines: [], count: 0 }));
    vi.stubGlobal('fetch', fetchMock);
    await bakeoffBaselineModels(undefined);
    expect(fetchMock.mock.calls[0]![0]).toBe(`${API_PREFIX}/bakeoff/baseline_models`);
  });

  it('sends the profile in the run body, and omits it when unset', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(() =>
        Promise.resolve(ok({ status: 'enqueued', job_id: 'j', out_dir: '/o' })),
      );
    vi.stubGlobal('fetch', fetchMock);
    const models = [{ backend: 'ultralytics' as const, name: 'm' }];

    await bakeoffRun({ datasets: [{ path: '/d' }], models, profile: 'license_plate' });
    await bakeoffRun({ datasets: [{ path: '/d' }], models, profile: undefined });

    const first = JSON.parse(fetchMock.mock.calls[0]![1].body as string);
    const second = JSON.parse(fetchMock.mock.calls[1]![1].body as string);
    expect(first.profile).toBe('license_plate');
    expect('profile' in second).toBe(false);
  });
});
