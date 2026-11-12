/**
 * Research preset wrappers: each route's method, URL (project-scoped or deployment-wide), name
 * encoding, query and body, through the real `apiFetch` (fetch stubbed).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import {
  activatePreset,
  activateServingProfile,
  deletePreset,
  getActivePreset,
  getActiveServingProfile,
  getPreset,
  listPresets,
  presetRefusalText,
  putPreset,
} from '$lib/api_presets';

const PROJECT = `${API_PREFIX}/projects/alpha`;

afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
});

function respond(status: number, body: unknown = {}) {
  const fetchMock = vi.fn().mockResolvedValue(
    status === 204
      ? new Response(null, { status })
      : new Response(JSON.stringify(body), {
          status,
          headers: { 'content-type': 'application/json' },
        }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return () => {
    const [url, init] = fetchMock.mock.calls[0]! as [string, RequestInit];
    return {
      url: String(url),
      method: init.method ?? 'GET',
      body: init.body ? JSON.parse(String(init.body)) : undefined,
    };
  };
}

describe('preset wrappers', () => {
  it('a project kind is read under the project, the serving profile under the deployment', async () => {
    setScopedPrefix(PROJECT);
    for (const [kind, url] of [
      ['cluster_profile', `${PROJECT}/cluster_profiles`],
      ['assist_loop_preset', `${PROJECT}/assist_loop_presets`],
      ['near_dup_profile', `${PROJECT}/near_dup_profiles`],
      ['serving_profile', `${API_PREFIX}/serving_profiles`],
    ] as const) {
      const call = respond(200, { items: [] });
      await listPresets(kind);
      expect(call()).toMatchObject({ url, method: 'GET' });
    }
  });

  it('get encodes the name and can ask for an older revision', async () => {
    setScopedPrefix(PROJECT);
    const call = respond(200);
    await getPreset('near_dup_profile', 'a b', 2);
    expect(call().url).toBe(`${PROJECT}/near_dup_profiles/a%20b?revision=2`);
  });

  it('put sends expected_revision (null creates) and delete the revision as a query', async () => {
    setScopedPrefix(PROJECT);
    const put = respond(200);
    await putPreset('cluster_profile', 'wide', {
      body: { target: 'item' },
      description: 'd',
      expected_revision: null,
    });
    expect(put()).toEqual({
      url: `${PROJECT}/cluster_profiles/wide`,
      method: 'PUT',
      body: { body: { target: 'item' }, description: 'd', expected_revision: null },
    });
    const del = respond(204);
    await deletePreset('serving_profile', 'two gpu', 3);
    expect(del()).toMatchObject({
      url: `${API_PREFIX}/serving_profiles/two%20gpu?expected_revision=3`,
      method: 'DELETE',
    });
  });

  it('a project kind is activated on an axis against what was read', async () => {
    setScopedPrefix(PROJECT);
    const call = respond(200, { axis: 'clustering_item', state: 'active' });
    await activatePreset('cluster_profile', 'wide', {
      axis: 'clustering_item',
      revision: 2,
      expected_active: { name: 'old', revision: 1 },
    });
    expect(call()).toEqual({
      url: `${PROJECT}/cluster_profiles/wide/activate`,
      method: 'POST',
      body: {
        axis: 'clustering_item',
        revision: 2,
        expected_active: { name: 'old', revision: 1 },
      },
    });
  });

  it('active reads ask for the axis and keep the served state', async () => {
    setScopedPrefix(PROJECT);
    const call = respond(200, {
      axis: 'clustering_region_box',
      state: 'off',
      name: null,
      revision: null,
    });
    const a = await getActivePreset('cluster_profile', 'clustering_region_box');
    expect(call().url).toBe(
      `${PROJECT}/cluster_profiles/active?axis=clustering_region_box`,
    );
    expect(a).toMatchObject({ axis: 'clustering_region_box', state: 'off', name: null });
  });

  it('the serving profile is activated against the setting revision; {active: null} is none active', async () => {
    const call = respond(200, { name: 'p', revision: 1, setting_revision: 4, body: {} });
    await activateServingProfile('p', { revision: 1, expected_revision: 3 });
    expect(call()).toEqual({
      url: `${API_PREFIX}/serving_profiles/p/activate`,
      method: 'POST',
      body: { revision: 1, expected_revision: 3 },
    });

    respond(200, { active: { name: 'p', revision: 1, setting_revision: 4, body: {} } });
    expect(await getActiveServingProfile()).toMatchObject({
      state: 'active',
      name: 'p',
      setting_revision: 4,
    });

    respond(200, { active: null });
    expect(await getActiveServingProfile()).toMatchObject({
      state: 'none',
      name: null,
      setting_revision: null,
    });
  });

  it('another failure of the active read is not "none active"', async () => {
    respond(409, { detail: { error: 'x', message: 'no' } });
    await expect(getActiveServingProfile()).rejects.toBeInstanceOf(ApiError);
  });

  it('a refusal reads as the served message followed by its reasons', async () => {
    respond(422, {
      detail: {
        error: 'validation_failed',
        message: 'invalid config body: 1 error(s)',
        reasons: ['k: Extra inputs are not permitted'],
      },
    });
    const err = await putPreset('near_dup_profile', 'x', {
      body: {},
      description: '',
      expected_revision: null,
    }).catch((e: unknown) => e);
    expect(presetRefusalText(err)).toBe(
      'invalid config body: 1 error(s): k: Extra inputs are not permitted',
    );
  });
});
