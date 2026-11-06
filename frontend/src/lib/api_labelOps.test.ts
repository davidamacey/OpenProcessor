/** Label-operation wrappers: method, project-scoped URL, query and body through the real `apiFetch`. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import { getLabelOps, undoLabelOp } from '$lib/api_labelOps';

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
    };
  };
}

describe('label ops wrappers', () => {
  it('lists with the limit as a query parameter', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ items: [], limit: 20, retention: 200, next_after: null });
    await getLabelOps({ limit: 20 });
    expect(last()).toMatchObject({ url: `${PROJECT}/label_ops?limit=20`, method: 'GET' });
  });

  it('pages with the previous next_after as after', async () => {
    setScopedPrefix(PROJECT);
    const last = capture({ items: [], limit: 20, retention: 200, next_after: null });
    await getLabelOps({ limit: 20, after: '01a1b00f-8d37-77d2-a545-50e87fe15f52' });
    expect(last()).toMatchObject({
      url: `${PROJECT}/label_ops?limit=20&after=01a1b00f-8d37-77d2-a545-50e87fe15f52`,
      method: 'GET',
    });
  });

  it('sends dry_run explicitly, true then false', async () => {
    setScopedPrefix(PROJECT);
    let last = capture();
    await undoLabelOp('op 1', true);
    expect(last()).toEqual({
      url: `${PROJECT}/label_ops/op%201/undo`,
      method: 'POST',
      body: { dry_run: true },
    });
    last = capture();
    await undoLabelOp('op 1', false);
    expect(last().body).toEqual({ dry_run: false });
  });
});
