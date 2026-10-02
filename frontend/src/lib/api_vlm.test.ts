/**
 * W9 wrappers: each route's method, GLOBAL vs project-scoped URL, name
 * encoding, query and body, through the real `apiFetch` (fetch stubbed).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import {
  activateVlm,
  clearLocalVlmSelection,
  cloneVlmEndpoint,
  createVlmEndpoint,
  deactivateVlm,
  deleteVlmEndpoint,
  getActiveVlm,
  getLocalVlm,
  getVlmCatalog,
  getVlmEndpoint,
  getVlmEndpointRevision,
  getVlmEndpointRevisions,
  getVlmEndpointSchema,
  listVlmEndpoints,
  probeVlmEndpoint,
  rollbackVlm,
  selectLocalVlm,
  updateVlmEndpoint,
  validateVlmEndpoint,
} from '$lib/api_vlm';
import { bodyFixture, docFixture } from '$lib/test/fixtures/vlm';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => vi.unstubAllGlobals());

function capture(body: unknown = {}, status = 200) {
  const fetchMock = vi
    .fn()
    .mockResolvedValue(
      status === 204 ? new Response(null, { status }) : json(body, status),
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

const GLOBAL = `${API_PREFIX}/vlm`;

describe('global registry routes (never project-scoped)', () => {
  it('list / schema / catalog / local are plain GETs', async () => {
    for (const [fn, path] of [
      [() => listVlmEndpoints(), '/endpoints'],
      [() => getVlmEndpointSchema(), '/endpoints/schema'],
      [() => getVlmCatalog(), '/catalog'],
      [() => getLocalVlm(), '/local'],
    ] as const) {
      const call = capture();
      await fn();
      expect(call()).toMatchObject({ url: `${GLOBAL}${path}`, method: 'GET' });
    }
  });

  it('endpoint reads encode the name', async () => {
    let call = capture(docFixture());
    await getVlmEndpoint('a b/c');
    expect(call().url).toBe(`${GLOBAL}/endpoints/a%20b%2Fc`);
    call = capture({ name: 'x', revisions: [] });
    await getVlmEndpointRevisions('a b');
    expect(call().url).toBe(`${GLOBAL}/endpoints/a%20b/revisions`);
    call = capture(docFixture());
    await getVlmEndpointRevision('a b', 4);
    expect(call().url).toBe(`${GLOBAL}/endpoints/a%20b/revisions/4`);
  });

  it('validate posts the body with the probe flag on the query', async () => {
    let call = capture({});
    await validateVlmEndpoint({ name: null, body: bodyFixture() });
    expect(call()).toMatchObject({
      url: `${GLOBAL}/endpoints/validate?probe=false`,
      method: 'POST',
      body: { name: null, body: bodyFixture() },
    });
    call = capture({});
    await validateVlmEndpoint({ name: 'x', body: bodyFixture() }, true);
    expect(call().url).toBe(`${GLOBAL}/endpoints/validate?probe=true`);
  });

  it('create, update, clone, probe and delete', async () => {
    let call = capture(docFixture(), 201);
    await createVlmEndpoint({ name: 'n', description: 'd', body: bodyFixture() });
    expect(call()).toMatchObject({
      url: `${GLOBAL}/endpoints`,
      method: 'POST',
      body: { name: 'n', description: 'd', body: bodyFixture() },
    });
    call = capture(docFixture());
    await updateVlmEndpoint('a b', {
      expected_revision: 3,
      description: null,
      body: bodyFixture(),
    });
    expect(call()).toMatchObject({
      url: `${GLOBAL}/endpoints/a%20b`,
      method: 'PUT',
      body: { expected_revision: 3, description: null, body: bodyFixture() },
    });
    call = capture(docFixture(), 201);
    await cloneVlmEndpoint('a b', { new_name: 'c', revision: null, description: null });
    expect(call()).toMatchObject({
      url: `${GLOBAL}/endpoints/a%20b/clone`,
      method: 'POST',
      body: { new_name: 'c', revision: null, description: null },
    });
    call = capture({ ok: true, probed_at: 'x' });
    await probeVlmEndpoint('a b');
    expect(call()).toMatchObject({
      url: `${GLOBAL}/endpoints/a%20b/probe`,
      method: 'POST',
    });
    call = capture(null, 204);
    await deleteVlmEndpoint('a b', 3);
    expect(call()).toMatchObject({
      url: `${GLOBAL}/endpoints/a%20b?expected_revision=3`,
      method: 'DELETE',
    });
  });

  it('local select posts {catalog_id, force?}; clear is a DELETE', async () => {
    let call = capture({}, 202);
    await selectLocalVlm({ catalog_id: 'vision-7b', force: true });
    expect(call()).toMatchObject({
      url: `${GLOBAL}/local/select`,
      method: 'POST',
      body: { catalog_id: 'vision-7b', force: true },
    });
    call = capture({});
    await clearLocalVlmSelection();
    expect(call()).toMatchObject({ url: `${GLOBAL}/local/select`, method: 'DELETE' });
  });
});

describe('project-scoped activation routes', () => {
  // `setScopedPrefix` defaults to API_PREFIX in tests, so a scoped URL is
  // `${API_PREFIX}${path}`; what matters is the path under it.
  it('active read, activate, rollback, deactivate', async () => {
    let call = capture({});
    await getActiveVlm();
    expect(call()).toMatchObject({
      url: `${API_PREFIX}/vlm/endpoints/active`,
      method: 'GET',
    });
    call = capture({});
    const expected = { name: 'a', revision: 1 };
    await activateVlm('a b', {
      revision: 3,
      expected_active: expected,
      force: false,
      acknowledge_external: true,
    });
    expect(call()).toMatchObject({
      url: `${API_PREFIX}/vlm/endpoints/a%20b/activate`,
      method: 'POST',
      body: {
        revision: 3,
        expected_active: expected,
        force: false,
        acknowledge_external: true,
      },
    });
    call = capture({});
    await rollbackVlm({ expected_active: expected });
    expect(call()).toMatchObject({
      url: `${API_PREFIX}/vlm/endpoints/active/rollback`,
      method: 'POST',
      body: { expected_active: expected },
    });
    call = capture({});
    await deactivateVlm({ expected_active: expected });
    expect(call()).toMatchObject({
      url: `${API_PREFIX}/vlm/endpoints/deactivate`,
      method: 'POST',
      body: { expected_active: expected },
    });
  });
});
