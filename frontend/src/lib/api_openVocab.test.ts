/**
 * Open-vocabulary wrappers: each route's method, project-scoped URL, name
 * encoding, query and body, through the real `apiFetch` (fetch stubbed).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import {
  activateOpenVocab,
  cloneOpenVocab,
  createOpenVocab,
  deactivateOpenVocab,
  deleteOpenVocab,
  getActiveOpenVocab,
  getOpenVocab,
  getOpenVocabRevision,
  getOpenVocabRevisions,
  getOpenVocabSchema,
  listOpenVocab,
  rollbackOpenVocab,
  testOpenVocab,
  updateOpenVocab,
  validateOpenVocab,
} from '$lib/api_openVocab';

const PROJECT = `${API_PREFIX}/projects/alpha`;

afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
});

function capture(status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(
    status === 204
      ? new Response(null, { status })
      : new Response('{}', {
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

const ACTIVE = { name: 'tags', revision: 2 };

describe('open-vocabulary wrappers', () => {
  it('list always asks for the templates', async () => {
    setScopedPrefix(PROJECT);
    const call = capture();
    await listOpenVocab();
    expect(call()).toMatchObject({
      url: `${PROJECT}/open_vocab?include_templates=true`,
      method: 'GET',
    });
  });

  it('schema, active and revisions are plain scoped GETs; names are encoded', async () => {
    setScopedPrefix(PROJECT);
    for (const [fn, path] of [
      [() => getOpenVocabSchema(), '/open_vocab/schema'],
      [() => getActiveOpenVocab(), '/open_vocab/active'],
      [() => getOpenVocab('a b'), '/open_vocab/a%20b'],
      [() => getOpenVocabRevisions('a b'), '/open_vocab/a%20b/revisions'],
      [() => getOpenVocabRevision('a b', 3), '/open_vocab/a%20b/revisions/3'],
    ] as const) {
      const call = capture();
      await fn();
      expect(call()).toMatchObject({ url: `${PROJECT}${path}`, method: 'GET' });
    }
  });

  it('create posts name, body and description', async () => {
    setScopedPrefix(PROJECT);
    const call = capture(201);
    await createOpenVocab({ name: 'tags', body: {} });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab`,
      method: 'POST',
      body: { name: 'tags', body: {} },
    });
  });

  it('save is a PUT carrying expected_revision', async () => {
    setScopedPrefix(PROJECT);
    const call = capture();
    await updateOpenVocab('tags', {
      expected_revision: 4,
      description: null,
      body: { display_name: 'Tags' },
    });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab/tags`,
      method: 'PUT',
      body: { expected_revision: 4, description: null, body: { display_name: 'Tags' } },
    });
  });

  it('delete sends expected_revision as a query param', async () => {
    setScopedPrefix(PROJECT);
    const call = capture(204);
    await deleteOpenVocab('tags', 5);
    expect(call()).toMatchObject({
      url: `${PROJECT}/open_vocab/tags?expected_revision=5`,
      method: 'DELETE',
    });
  });

  it('validate sends for_activation only when true', async () => {
    setScopedPrefix(PROJECT);
    let call = capture();
    await validateOpenVocab({ name: null, body: {} });
    expect(call()).toMatchObject({
      url: `${PROJECT}/open_vocab/validate`,
      body: { name: null, body: {} },
    });
    call = capture();
    await validateOpenVocab({ name: null, body: {} }, true);
    expect(call().url).toBe(`${PROJECT}/open_vocab/validate?for_activation=true`);
  });

  it('test posts the request as given', async () => {
    setScopedPrefix(PROJECT);
    const call = capture();
    await testOpenVocab({ image_id: 'img1', target: { prompt: 'blue widget' } });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab/test`,
      method: 'POST',
      body: { image_id: 'img1', target: { prompt: 'blue widget' } },
    });
  });

  it('activate, rollback and deactivate carry the OCC ref', async () => {
    setScopedPrefix(PROJECT);
    let call = capture();
    await activateOpenVocab('tags', {
      revision: 2,
      expected_active: ACTIVE,
      force: false,
    });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab/tags/activate`,
      method: 'POST',
      body: { revision: 2, expected_active: ACTIVE, force: false },
    });
    call = capture();
    await rollbackOpenVocab({ expected_active: ACTIVE });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab/active/rollback`,
      method: 'POST',
      body: { expected_active: ACTIVE },
    });
    call = capture();
    await deactivateOpenVocab({ expected_active: { name: null, revision: null } });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab/deactivate`,
      method: 'POST',
      body: { expected_active: { name: null, revision: null } },
    });
  });

  it('clone posts to the source name', async () => {
    setScopedPrefix(PROJECT);
    const call = capture(201);
    await cloneOpenVocab('tpl', {
      new_name: 'mine',
      revision: null,
      source: 'template',
      description: null,
    });
    expect(call()).toEqual({
      url: `${PROJECT}/open_vocab/tpl/clone`,
      method: 'POST',
      body: { new_name: 'mine', revision: null, source: 'template', description: null },
    });
  });
});
