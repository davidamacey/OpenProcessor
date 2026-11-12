/** Embedding-space writes: method, URL, target query, body and the idempotency header, through the real `apiFetch`. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import {
  buildEmbeddingSpace,
  cancelEmbeddingReindexRun,
  getEmbeddingModels,
  getEmbeddingReindexRun,
  promoteEmbeddingSpace,
  registerEmbeddingSpace,
  retireEmbeddingSpace,
} from '$lib/api_embeddingSpaces';

const PROJECT = `${API_PREFIX}/projects/alpha`;

afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
});

function capture(body: unknown = {}) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status: 200,
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
      headers: init.headers as Record<string, string>,
    };
  };
}

describe('embedding space wrappers', () => {
  it('the encoder catalog is deployment-wide', async () => {
    setScopedPrefix(PROJECT);
    const call = capture({ models: [{ model_key: 'm' }] });
    expect(await getEmbeddingModels()).toEqual([{ model_key: 'm' }]);
    expect(call()).toMatchObject({
      url: `${API_PREFIX}/embedding/models`,
      method: 'GET',
    });
  });

  it('register posts the request with its Idempotency-Key', async () => {
    setScopedPrefix(PROJECT);
    const call = capture();
    await registerEmbeddingSpace(
      { model_key: 'mobileclip2_s2', model_version: 1, target: 'item', dim: null },
      'try-12345678',
    );
    const c = call();
    expect(c).toMatchObject({
      url: `${PROJECT}/embedding/spaces`,
      method: 'POST',
      body: { model_key: 'mobileclip2_s2', model_version: 1, target: 'item', dim: null },
    });
    expect(c.headers['Idempotency-Key']).toBe('try-12345678');
  });

  it('build, promote and retire name the target in the query and encode the key', async () => {
    setScopedPrefix(PROJECT);
    const build = capture();
    await buildEmbeddingSpace('a b', 'box', { mode: 'backfill', batch_size: 64 });
    expect(build()).toMatchObject({
      url: `${PROJECT}/embedding/spaces/a%20b/build?target=box`,
      method: 'POST',
      body: { mode: 'backfill', batch_size: 64 },
    });
    const promote = capture();
    await promoteEmbeddingSpace('k', 'item', { force: true });
    expect(promote()).toMatchObject({
      url: `${PROJECT}/embedding/spaces/k/promote?target=item`,
      body: { force: true },
    });
    const retire = capture();
    await retireEmbeddingSpace('k', 'frame');
    expect(retire()).toMatchObject({
      url: `${PROJECT}/embedding/spaces/k/retire?target=frame`,
      method: 'POST',
    });
  });

  it('a run is read and cancelled by id; abandon is sent only when true', async () => {
    setScopedPrefix(PROJECT);
    const read = capture();
    await getEmbeddingReindexRun('r1');
    expect(read().url).toBe(`${PROJECT}/embedding/reindex-runs/r1`);
    const keep = capture();
    await cancelEmbeddingReindexRun('r1', false);
    expect(keep()).toMatchObject({
      url: `${PROJECT}/embedding/reindex-runs/r1/cancel`,
      method: 'POST',
    });
    const abandon = capture();
    await cancelEmbeddingReindexRun('r1', true);
    expect(abandon().url).toBe(
      `${PROJECT}/embedding/reindex-runs/r1/cancel?abandon=true`,
    );
  });
});
