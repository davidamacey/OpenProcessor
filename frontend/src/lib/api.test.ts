/**
 * Tests for ApiError's message composition.
 *
 * Every UI callsite renders `(e as Error).message`, so the server's reason
 * has to be baked into the message or the operator never sees it.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  apiBase,
  ApiError,
  getCluster,
  getClassRegistryUrl,
  getDataYamlUrl,
  getManifestUrl,
  getMethods,
  getReviewQueue,
  getVizProjection,
  rebuildVizProjection,
} from './api';
import { FALLBACK_METHODS } from './strategies';

const URL = 'http://localhost:4603/op/crops/batch_label';

describe('ApiError', () => {
  it("appends a FastAPI 'detail' string to the message", () => {
    const e = new ApiError(422, URL, { detail: 'plate bbox outside crop envelope' });
    expect(e.detail).toBe('plate bbox outside crop envelope');
    expect(e.message).toBe(`API 422 ${URL} — plate bbox outside crop envelope`);
  });

  it("falls back to a 'message' property", () => {
    const e = new ApiError(400, URL, { message: 'hotkey already bound' });
    expect(e.message).toContain('hotkey already bound');
  });

  it('accepts a plain-text body', () => {
    const e = new ApiError(502, URL, 'upstream unavailable');
    expect(e.detail).toBe('upstream unavailable');
  });

  it('truncates a long detail to 200 chars', () => {
    const e = new ApiError(422, URL, { detail: 'x'.repeat(500) });
    expect(e.detail).toHaveLength(200);
    expect(e.detail?.endsWith('…')).toBe(true);
  });

  it('leaves the message unadorned when there is no usable detail', () => {
    expect(new ApiError(500, URL, null).message).toBe(`API 500 ${URL}`);
    expect(new ApiError(500, URL, { detail: '   ' }).message).toBe(`API 500 ${URL}`);
    expect(new ApiError(500, URL, { detail: [{ loc: ['body'] }] }).message).toBe(
      `API 500 ${URL}`,
    );
  });

  it('honors an explicit message override', () => {
    const e = new ApiError(404, URL, { detail: 'nope' }, 'custom');
    expect(e.message).toBe('custom');
    expect(e.detail).toBe('nope');
  });
});

/**
 * getMethods() must never throw — /curation/methods is optional capability
 * discovery (plan §5.3). A 404 or any other failure resolves to the
 * hardcoded FALLBACK_METHODS instead of rejecting, so a backend that
 * hasn't shipped the endpoint yet can't break app boot.
 */
describe('getMethods', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the real parsed response on success', async () => {
    // Real /curation/methods wire shape (confirmed 2026-09-10 against
    // strategy_registry.py's get_registry()): a flat `strategies` array,
    // each entry carrying an `axis` field — not four separate top-level
    // arrays.
    const serverBody = {
      strategies: [
        {
          id: 'ivf',
          axis: 'cluster',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
        { id: 'default', axis: 'sort', label: 'Recent first', status: 'stable', default: true },
        { id: 'uncertainty', axis: 'sort', label: 'Uncertainty margin', status: 'experimental' },
      ],
      flags: {},
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(serverBody));
    vi.stubGlobal('fetch', fetchMock);

    const result = await getMethods();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.cluster_methods).toEqual([
      { id: 'ivf', label: 'FAISS IVF-512 (production)', status: 'stable', default: true },
    ]);
    expect(result.review_sorts).toEqual([
      { id: 'default', label: 'Recent first', status: 'stable', default: true },
      { id: 'uncertainty', label: 'Uncertainty margin', status: 'experimental', default: undefined },
    ]);
    // Real backend response, not the hardcoded fallback.
    expect(result).not.toEqual(FALLBACK_METHODS);
  });

  it('resolves to FALLBACK_METHODS on a 404, without throwing or retrying', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getMethods()).resolves.toEqual(FALLBACK_METHODS);
    // No retry on 4xx — matches apiFetch's documented "don't retry on 4xx" rule.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('resolves to FALLBACK_METHODS on a network failure, after the normal 5xx/network retry budget', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    await expect(getMethods()).resolves.toEqual(FALLBACK_METHODS);
    // apiFetch's retry loop: 1 initial + 3 retries = 4 attempts.
    expect(fetchMock).toHaveBeenCalledTimes(4);
  }, 10_000);

  it('resolves to FALLBACK_METHODS (not throws) on a malformed 200 body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not json', {
        status: 200,
        headers: { 'content-type': 'text/plain' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    // A plain-text 200 body parses to a string, which parseKbMethodsResponse
    // treats as "not a usable object" and degrades to empty lists — not a
    // crash, and distinct from the true-failure fallback path.
    const result = await getMethods();
    expect(result).toEqual({
      cluster_methods: [],
      review_sorts: [],
      overlays: [],
      scores: [],
    });
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

    const p = getMethods(ctrl.signal);
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });
});

/**
 * Phase 3 (curation-strategy plan §5): getReviewQueue's `filter` argument
 * already accepts arbitrary keys, so StrategyBar's `sort` /
 * `min_mistakenness` / `hide_near_duplicates` need no new plumbing in
 * getReviewQueue itself — just qs()'s existing null-dropping behavior and
 * a pass-through of the new `sort_fallback_reason` + mistakenness fields
 * on the response.
 */
describe('getReviewQueue', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards sort/min_mistakenness/hide_near_duplicates from an arbitrary filter object', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getReviewQueue('all', 1, 30, {
      sort: 'mistakenness',
      min_mistakenness: 0.5,
      hide_near_duplicates: true,
    });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('sort=mistakenness');
    expect(url).toContain('min_mistakenness=0.5');
    expect(url).toContain('hide_near_duplicates=true');
  });

  it('omits strategy params entirely when the filter object is empty (qs() drops nothing extra)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getReviewQueue('all', 1, 30, {});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('sort=');
    expect(url).not.toContain('min_mistakenness');
    expect(url).not.toContain('hide_near_duplicates');
  });

  it('surfaces sort_fallback_reason from the raw response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 0,
        page: 1,
        page_size: 30,
        items: [],
        sort_fallback_reason: 'mistakenness not backfilled for this pool',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, { sort: 'mistakenness' });
    expect(res.sort_fallback_reason).toBe('mistakenness not backfilled for this pool');
  });

  it('defaults sort_fallback_reason to null when the server omits it', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.sort_fallback_reason).toBeNull();
  });

  it('maps mistakenness_score/method/version through onto each item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            mistakenness_score: 0.87,
            mistakenness_method: 'mistakenness',
            mistakenness_version: 'v1',
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.items[0]?.mistakenness_score).toBe(0.87);
    expect(res.items[0]?.mistakenness_method).toBe('mistakenness');
    expect(res.items[0]?.mistakenness_version).toBe('v1');
  });

  it('leaves mistakenness fields null when the server omits them (un-backfilled pool)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [{ crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1] }],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.items[0]?.mistakenness_score).toBeNull();
  });
});

/**
 * getCluster's `order` param is forwarded verbatim to `/curation/crops?order=`
 * (broadened from a fixed `'outliers'` literal so a future
 * `/curation/methods`-reported order id doesn't require touching this
 * signature — see the comment on `order` in api.ts). An id the backend
 * doesn't recognize should be harmless: qs() still sends it, and callers
 * are responsible for only offering ids `/curation/methods` actually reports.
 */
describe('getCluster order param', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards an arbitrary order id to /curation/crops without special-casing it client-side', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith('/curation/crops')) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.resolve(jsonResponse({ items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);

    await getCluster(42, 1, 60, undefined, { order: 'mistakenness' });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith('/curation/crops'));
    expect(cropsUrl).toContain('order=mistakenness');
  });

  it('omits order entirely when null (unchanged default behavior)', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith('/curation/crops')) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.resolve(jsonResponse({ items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);

    await getCluster(42, 1, 60, undefined, { order: null });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith('/curation/crops'));
    expect(cropsUrl).not.toContain('order');
  });
});

/**
 * `k` (curation-strategy plan Phase 4 — "how many diverse crops?", forwarded
 * alongside `order=diverse`). Same forward-verbatim contract as `order`:
 * getCluster doesn't validate the id/count pair, it just plumbs whatever the
 * caller (gated by /curation/methods, see strategies.test.ts's
 * isDiverseOverlayAvailable coverage) decided to send.
 */
describe('getCluster k param', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubCrops(body: unknown) {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith('/curation/crops')) return Promise.resolve(jsonResponse(body));
      return Promise.resolve(jsonResponse({ items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  it('forwards k to /curation/crops when set alongside order=diverse', async () => {
    const fetchMock = stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    await getCluster(42, 1, 60, undefined, { order: 'diverse', k: 120 });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith('/curation/crops'));
    expect(cropsUrl).toContain('order=diverse');
    expect(cropsUrl).toContain('k=120');
  });

  it('omits k entirely when null/undefined (qs() drops it, no ?k= at all)', async () => {
    const fetchMock = stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    await getCluster(42, 1, 60, undefined, { order: null, k: null });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith('/curation/crops'));
    expect(cropsUrl).not.toContain('k=');

    fetchMock.mockClear();
    await getCluster(42, 1, 60, undefined, {});
    const cropsUrl2 = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith('/curation/crops'));
    expect(cropsUrl2).not.toContain('k=');
  });

  it('surfaces order_method/order_version/n_pool when the server sends them', async () => {
    stubCrops({
      total: 500,
      page: 1,
      page_size: 60,
      crops: [],
      method: 'kcenter_greedy',
      version: '1',
      n_pool: 4832,
    });

    const res = await getCluster(42, 1, 60, undefined, { order: 'diverse', k: 60 });

    expect(res.crops.order_method).toBe('kcenter_greedy');
    expect(res.crops.order_version).toBe('1');
    expect(res.crops.n_pool).toBe(4832);
  });

  it('defaults order_method/order_version/n_pool to null when the server omits them', async () => {
    stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    const res = await getCluster(42, 1, 60, undefined, { order: null });

    expect(res.crops.order_method).toBeNull();
    expect(res.crops.order_version).toBeNull();
    expect(res.crops.n_pool).toBeNull();
  });
});

/**
 * getVizProjection() — curation-strategy plan Phase 5
 * (docs/curation-strategy-plan-2026-09.md §2.7/§5.6). Never rejects
 * (same contract as getMethods): `/curation/viz/projection` may not exist yet
 * (the openprocessor Phase 5 branch lands independently) and the UMAP
 * purity gate may mean the capability never ships at all — a fetch
 * failure here must degrade `EmbeddingPlot` to its pending/empty state,
 * never crash the page it replaced the grid on.
 *
 * The real wire shape (confirmed 2026-09-10 against
 * `embedding_viz.get_cached_projection`) is `{status: 'not_built'}` when
 * nothing has been fit, or `{points, projection_version, fitted_at,
 * stale}` otherwise — there is no `built`/`built_at`/`version` on the
 * wire; those were an earlier, wrong guess. `built` is this file's own
 * derived convenience field.
 */
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
          { crop_id: 'a', x: 1.5, y: -2.3, cluster_id: 17, class_name: 'sedan' },
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
        class_name: 'sedan',
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
    expect(url).toContain('/curation/viz/projection');
    expect(url).toContain('max_points=100');
  });

  it('forwards cluster_id/class_id and omits unset params', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ points: [], projection_version: '1', fitted_at: null, stale: false }));
    vi.stubGlobal('fetch', fetchMock);

    await getVizProjection({ cluster_id: 42 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('cluster_id=42');
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('max_points');
  });

  it("reports built:false when the server says status: 'not_built'", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: 'not_built' }));
    vi.stubGlobal('fetch', fetchMock);
    expect((await getVizProjection()).built).toBe(false);
  });

  it('reports stale:true when the server flags a partial-coverage projection', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ points: [], projection_version: '2', fitted_at: '2026-09-01T00:00:00Z', stale: true }),
    );
    vi.stubGlobal('fetch', fetchMock);

    expect((await getVizProjection()).stale).toBe(true);
  });

  it('resolves to the empty/pending fallback on a 404, without throwing or retrying', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection();
    expect(res).toEqual({
      points: [],
      total: 0,
      built: false,
      fitted_at: null,
      projection_version: null,
      stale: false,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('resolves to the empty/pending fallback on a network failure, after the retry budget', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection();
    expect(res.built).toBe(false);
    expect(res.points).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(4);
  }, 10_000);

  it('resolves to the empty/pending fallback (not throws) on a malformed 200 body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not json', {
        status: 200,
        headers: { 'content-type': 'text/plain' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection();
    expect(res.points).toEqual([]);
    expect(res.built).toBe(false);
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

  it('POSTs to /curation/viz/projection/rebuild and returns the job snapshot', async () => {
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
    expect(url).toContain('/curation/viz/projection/rebuild');
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

/**
 * Regression test for the /export download-button filename mismatch:
 * getDataYamlUrl() used to request `data_v7.yaml`, which the export service
 * never writes (the real on-disk file is `data.yaml`), so the download would
 * 404 against the real `/curation/export/registry/{artifact}` backend contract.
 * Pin all three exact URLs so this can't silently regress.
 */
describe('registry download URL builders', () => {
  it('getClassRegistryUrl() points at the real class_registry.json filename', () => {
    expect(getClassRegistryUrl()).toBe(`${apiBase}/curation/export/registry/class_registry.json`);
  });

  it('getDataYamlUrl() points at the real data.yaml filename (not data_v7.yaml)', () => {
    expect(getDataYamlUrl()).toBe(`${apiBase}/curation/export/registry/data.yaml`);
  });

  it('getManifestUrl() points at the real manifest.json filename', () => {
    expect(getManifestUrl()).toBe(`${apiBase}/curation/export/registry/manifest.json`);
  });
});
