/**
 * Tests for the embedding-projection wrappers.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { ApiError, apiErrorText } from '$lib/api_errors';
import { getVizProjection, rebuildVizProjection } from '$lib/api_viz';

/** getVizProjection(): only the served 404 `projection_not_built` is "not built"; every other failure rejects. */
describe('getVizProjection', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses a well-formed built response, dropping malformed points', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        points: [
          { crop_id: 'a', x: 1.5, y: -2.3, cluster_id: 17, class_name: 'widget_a' },
          { crop_id: 'b', x: 0, y: 0, cluster_id: null, class_name: null },
          // Malformed entries — must be dropped, not crash the whole parse.
          { crop_id: '', x: 1, y: 1 },
          { x: 1, y: 1 },
          { crop_id: 'c', x: 'nope', y: 1 },
          null,
          'garbage',
        ],
        projection_version: '1',
        fitted_at: '2026-09-10T00:00:00Z',
        stale: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection({ max_points: 100 });

    expect(res.built).toBe(true);
    expect(res.points).toEqual([
      {
        crop_id: 'a',
        x: 1.5,
        y: -2.3,
        cluster_id: 17,
        class_name: 'widget_a',
        class_source: null,
      },
      {
        crop_id: 'b',
        x: 0,
        y: 0,
        cluster_id: null,
        class_name: null,
        class_source: null,
      },
    ]);
    expect(res.total).toBe(2);
    expect(res.projection_version).toBe('1');
    expect(res.fitted_at).toBe('2026-09-10T00:00:00Z');
    expect(res.stale).toBe(false);

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/viz/projection`);
    expect(url).toContain('max_points=100');
  });

  it('forwards cluster_id/class_id and omits unset params', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        points: [],
        projection_version: '1',
        fitted_at: null,
        stale: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await getVizProjection({ cluster_id: 42 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('cluster_id=42');
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('max_points');
  });

  it("reports built:false only for the served 404 'projection_not_built'", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse(
          { detail: { error: 'projection_not_built', message: 'Build one first.' } },
          { status: 404 },
        ),
      );
    vi.stubGlobal('fetch', fetchMock);
    const res = await getVizProjection();
    expect(res.built).toBe(false);
    expect(res.points).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reports stale:true when the server flags a partial-coverage projection', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        points: [],
        projection_version: '2',
        fitted_at: '2026-09-01T00:00:00Z',
        stale: true,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    expect((await getVizProjection()).stale).toBe(true);
  });

  it('rejects a 404 that is not the projection_not_built code (not a false "not built")', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);
    await expect(getVizProjection()).rejects.toMatchObject({ status: 404 });
  });

  it('rejects a 503 projection_unavailable with the served message', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(async () =>
        jsonResponse(
          { detail: { error: 'projection_unavailable', message: 'Index is down.' } },
          { status: 503 },
        ),
      );
    vi.stubGlobal('fetch', fetchMock);
    const err = await getVizProjection().catch((e: unknown) => e);
    expect(apiErrorText(err)).toBe('Index is down.');
  }, 10_000);

  it('rejects on a network failure after the retry budget', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);
    await expect(getVizProjection()).rejects.toThrow('fetch failed');
    expect(fetchMock).toHaveBeenCalledTimes(4);
  }, 10_000);

  it('rejects a malformed 200 body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not json', {
        status: 200,
        headers: { 'content-type': 'text/plain' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await expect(getVizProjection()).rejects.toThrow();
  });

  it('propagates a caller-initiated abort instead of swallowing it into the fallback', async () => {
    const ctrl = new AbortController();
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        });
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const p = getVizProjection(undefined, ctrl.signal);
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });
});

/**
 * rebuildVizProjection() — the real `embedding_viz._JobState` is flat
 * (confirmed 2026-09-10), not the nested `{running, result: {...}}`
 * shape an earlier version of this file guessed: `status` is a string
 * enum, timestamps are unix-epoch numbers (0 when unset), and
 * `n_written`/`projection_version` are top-level fields.
 */
describe('rebuildVizProjection', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs to {API_PREFIX}/viz/projection/rebuild and returns the job snapshot', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        job_id: 'viz-rebuild-1',
        status: 'running',
        scope: 'all',
        cluster_id: null,
        n_pool: 1200,
        n_written: 0,
        started_at: 1757462400,
        finished_at: 0,
        error: null,
        projection_version: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await rebuildVizProjection();

    expect(res.status).toBe('running');
    expect(res.job_id).toBe('viz-rebuild-1');
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/viz/projection/rebuild`);
    expect(init.method).toBe('POST');
  });

  it('propagates a 5xx failure rather than swallowing it (unlike getVizProjection)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'busy' }), { status: 503 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(rebuildVizProjection()).rejects.toBeInstanceOf(ApiError);
  }, 10_000);
});
