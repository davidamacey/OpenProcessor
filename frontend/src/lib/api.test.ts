/**
 * Tests for ApiError's message composition.
 *
 * Every UI callsite renders `(e as Error).message`, so the server's reason
 * has to be baked into the message or the operator never sees it.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  apiBase,
  API_PREFIX,
  ApiError,
  cancelSelect,
  getCluster,
  getClassRegistryUrl,
  getCurationSettings,
  getDataYamlUrl,
  getManifestUrl,
  getMethods,
  getReviewQueue,
  getSelectStatus,
  getVizProjection,
  normalizeApiPrefix,
  putCurationDefaults,
  rebuildVizProjection,
  searchCrops,
  selectDiverse,
  startAutoLabel,
} from './api';
import { FALLBACK_METHODS } from './strategies';

const URL = `http://localhost:4603${API_PREFIX}/crops/batch_label`;

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
        {
          id: 'default',
          axis: 'sort',
          label: 'Recent first',
          status: 'stable',
          default: true,
        },
        {
          id: 'uncertainty',
          axis: 'sort',
          label: 'Uncertainty margin',
          status: 'experimental',
        },
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
      {
        id: 'uncertainty',
        label: 'Uncertainty margin',
        status: 'experimental',
        default: undefined,
      },
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
      dataset_exports: [],
      detection_profiles: [],
      prompt_packs: [],
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
 * searchCrops (P2-14 semantic text search) mirrors getReviewQueue's
 * response-shape handling — same qs()-forwarding, same mapRawCrop
 * normalization — but points at `GET /curation/search/text` and adds the
 * per-item similarity score instead of the review-queue's reason/
 * proposed-class fields.
 */
describe('searchCrops', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends q/page/page_size and hits GET /curation/search/text', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await searchCrops('red sedan', 1, 30);

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/search/text`);
    expect(url).toContain('q=red+sedan');
    expect(url).toContain('page=1');
    expect(url).toContain('page_size=30');
  });

  it('forwards extra filter params (e.g. cluster_id, review tab filters) unchanged', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await searchCrops('blue truck', 1, 30, { cluster_id: 42, max_rank: 1 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('cluster_id=42');
    expect(url).toContain('max_rank=1');
  });

  it('maps similarity_score through onto each item', async () => {
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
            similarity_score: 0.91,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red sedan', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0.91);
    expect(res.items[0]?.id).toBe('c1');
  });

  it('maps the live backend field `semantic_score` through onto each item (openprocessor _hydrate_item)', async () => {
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
            semantic_score: 0.73,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red sedan', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0.73);
  });

  it('prefers similarity_score over semantic_score over score when more than one is present', async () => {
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
            similarity_score: 0.91,
            semantic_score: 0.73,
            score: 0.5,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const res1 = await searchCrops('red sedan', 1, 30);
    expect(res1.items[0]?.similarity_score).toBe(0.91);

    const fetchMock2 = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            semantic_score: 0.73,
            score: 0.5,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock2);
    const res2 = await searchCrops('red sedan', 1, 30);
    expect(res2.items[0]?.similarity_score).toBe(0.73);
  });

  it('falls back to a bare score field if the server sends that instead', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          { crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1], score: 0.5 },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red sedan', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0.5);
  });

  it('defaults similarity_score to 0 when the server omits both fields', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [{ crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1] }],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red sedan', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0);
  });

  it('returns an empty result set on a zero-hit response (empty-state input)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('zzzznonexistentqueryzzzz', 1, 30);
    expect(res.items).toEqual([]);
    expect(res.total).toBe(0);
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
      if (url.startsWith(`${API_PREFIX}/crops`)) {
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
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).toContain('order=mistakenness');
  });

  it('omits order entirely when null (unchanged default behavior)', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
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
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
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
      if (url.startsWith(`${API_PREFIX}/crops`))
        return Promise.resolve(jsonResponse(body));
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
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).toContain('order=diverse');
    expect(cropsUrl).toContain('k=120');
  });

  it('omits k entirely when null/undefined (qs() drops it, no ?k= at all)', async () => {
    const fetchMock = stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    await getCluster(42, 1, 60, undefined, { order: null, k: null });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).not.toContain('k=');

    fetchMock.mockClear();
    await getCluster(42, 1, 60, undefined, {});
    const cropsUrl2 = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
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

  it("reports built:false when the server says status: 'not_built'", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: 'not_built' }));
    vi.stubGlobal('fetch', fetchMock);
    expect((await getVizProjection()).built).toBe(false);
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

/**
 * Regression test for the /export download-button filename mismatch:
 * getDataYamlUrl() used to request `data_v7.yaml`, which the export service
 * never writes (the real on-disk file is `data.yaml`), so the download would
 * 404 against the real `/curation/export/registry/{artifact}` backend contract.
 * Pin all three exact URLs so this can't silently regress.
 */
describe('registry download URL builders', () => {
  it('getClassRegistryUrl() points at the real class_registry.json filename', () => {
    expect(getClassRegistryUrl()).toBe(
      `${apiBase}${API_PREFIX}/export/registry/class_registry.json`,
    );
  });

  it('getDataYamlUrl() points at the real data.yaml filename (not data_v7.yaml)', () => {
    expect(getDataYamlUrl()).toBe(`${apiBase}${API_PREFIX}/export/registry/data.yaml`);
  });

  it('getManifestUrl() points at the real manifest.json filename', () => {
    expect(getManifestUrl()).toBe(
      `${apiBase}${API_PREFIX}/export/registry/manifest.json`,
    );
  });
});

/**
 * P2-10: `/review`'s diverse overlay (`POST /curation/select/diverse`). The
 * endpoint answers 200 (small pool, `crop_ids` ready now), 202 (large
 * pool — job enqueued, poll `getSelectStatus`), 400 (feature disabled),
 * or 409 (singleton job already running elsewhere) — the last two are
 * expected states the UI treats as routine, not toast-worthy failures, so
 * `selectDiverse` must resolve a typed result for them rather than throw.
 */
describe('selectDiverse', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns a "ready" result for a 200 response with crop_ids', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          crop_ids: ['a', 'b'],
          method: 'k_center_greedy',
          version: 'v1',
          n_pool: 2,
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 100);
    expect(res).toEqual({
      kind: 'ready',
      selection: {
        crop_ids: ['a', 'b'],
        method: 'k_center_greedy',
        version: 'v1',
        n_pool: 2,
      },
    });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/select/diverse`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      scope: { review_tab: 'all' },
      k: 100,
    });
  });

  it('returns a "job" result for a 202 response with job_id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-123', status: 'running' }), {
        status: 202,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 500);
    expect(res).toEqual({ kind: 'job', job_id: 'job-123' });
  });

  it('returns "disabled" (not a throw) on a 400', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'diverse selection disabled' }), {
        status: 400,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 100);
    expect(res).toEqual({ kind: 'disabled' });
  });

  it('returns "already_running" (not a throw) on a 409', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'job already running' }), { status: 409 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 100);
    expect(res).toEqual({ kind: 'already_running' });
  });

  it('still throws on an unrelated 4xx/5xx', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'boom' }), { status: 422 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(selectDiverse({ review_tab: 'all' }, 100)).rejects.toBeInstanceOf(
      ApiError,
    );
  });

  it('forwards seed_crop_id only when given', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ crop_ids: [], method: 'm', version: 'v', n_pool: 0 }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    await selectDiverse({ cluster_id: 5 }, 10, 'seed-crop-1');
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      scope: { cluster_id: 5 },
      k: 10,
      seed_crop_id: 'seed-crop-1',
    });
  });
});

describe('getSelectStatus / cancelSelect', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses a running-job status payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-1', status: 'running' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const st = await getSelectStatus();
    expect(st.status).toBe('running');
    expect(st.job_id).toBe('job-1');
  });

  it('parses a completed-job status payload, result nested (real selection/job.py shape)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'completed',
          result: {
            crop_ids: ['x', 'y'],
            method: 'kcenter_greedy',
            version: 'v1',
            n_pool: 2,
          },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const st = await getSelectStatus();
    expect(st.status).toBe('completed');
    expect(st.result).toEqual({
      crop_ids: ['x', 'y'],
      method: 'kcenter_greedy',
      version: 'v1',
      n_pool: 2,
    });
  });

  it('leaves result null for a failed/cancelled job', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'failed',
          error: 'selection job heartbeat stale (34.6s ago)',
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const st = await getSelectStatus();
    expect(st.status).toBe('failed');
    expect(st.result).toBeNull();
    expect(st.error).toBe('selection job heartbeat stale (34.6s ago)');
  });

  it('cancelSelect POSTs to /curation/select/cancel', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await cancelSelect();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/select/cancel`);
    expect(init.method).toBe('POST');
  });
});

/**
 * Scoped VLM-assisted labeling (2026-09-20 contract — confirmed live
 * 2026-09-21 against a real OpenProcessor backend via the actual UI,
 * see docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md
 * §1.3/§5.3). `class_id`/`detection_profile`/`prompt_pack` are optional
 * query params on the existing start call; `qs()` drops null/undefined so
 * an unscoped call must be byte-identical to the pre-scope request.
 */
describe('startAutoLabel', () => {
  const jobResponse = () =>
    new Response(
      JSON.stringify({
        job_id: 'job-1',
        status: 'idle',
        stage: '',
        processed: 0,
        total: 0,
        started_at: 0,
        finished_at: 0,
        error: null,
        result: {},
        args: {},
        eta_seconds: null,
        elapsed_seconds: 0,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends no scoping params when the scope is untouched (byte-identical to the pre-scope request)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ train_clusters: true, gemma_concurrency: 16 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('detection_profile');
    expect(url).not.toContain('prompt_pack');
  });

  it('forwards class_id when the run is scoped to one class', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ class_id: 7 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('class_id=7');
  });

  it('drops an explicitly null class_id rather than sending class_id=null', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ class_id: null });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('null');
  });

  it('forwards detection_profile and prompt_pack when selected', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({
      detection_profile: 'grounding_v2',
      prompt_pack: 'warehouse_v1',
    });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('detection_profile=grounding_v2');
    expect(url).toContain('prompt_pack=warehouse_v1');
  });

  it('composes the path from API_PREFIX', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/pipeline/auto_label/start`);
  });
});

describe('API_PREFIX', () => {
  // T-E2: the default matches OpenProcessor's OP_API_PREFIX default.
  it('defaults to /curation when PUBLIC_API_PREFIX is unset', () => {
    expect(API_PREFIX).toBe('/curation');
  });

  it('treats empty, whitespace and an unsubstituted placeholder as unset', () => {
    expect(normalizeApiPrefix('')).toBe('/curation');
    expect(normalizeApiPrefix('   ')).toBe('/curation');
    expect(normalizeApiPrefix('__API_PREFIX__')).toBe('/curation');
  });

  it('normalizes a configured prefix to a leading slash and no trailing slash', () => {
    expect(normalizeApiPrefix('/curation')).toBe('/curation');
    expect(normalizeApiPrefix('curation')).toBe('/curation');
    expect(normalizeApiPrefix('/curation/')).toBe('/curation');
    expect(normalizeApiPrefix('/curation///')).toBe('/curation');
  });
});

/**
 * `getCurationSettings`/`putCurationDefaults` — docs/design/
 * curation-settings-ui-plan-2026-09-21.md §4.1. Deliberately asymmetric
 * vs. `getMethods`: THIS endpoint throws on failure rather than
 * degrading to a fallback, since a settings page must tell a 404
 * ("backend predates the feature") apart from every other outcome.
 */
describe('getCurationSettings / putCurationDefaults', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('getCurationSettings composes ${API_PREFIX}/settings and parses the body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: { cluster: 'ivf' },
        updated_at: '2026-09-20T23:04:39+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await getCurationSettings();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const calledUrl = (fetchMock.mock.calls[0]?.[0] as string) ?? '';
    expect(calledUrl).toContain(`${API_PREFIX}/settings`);
    expect(result).toEqual({
      defaults: { cluster: 'ivf' },
      updated_at: '2026-09-20T23:04:39+00:00',
      updated_by: null,
    });
  });

  it('putCurationDefaults sends PUT with the exact partial body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: { sort: 'uncertainty_entropy' },
        updated_at: '2026-09-21T00:00:00+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await putCurationDefaults({ sort: 'uncertainty_entropy' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain(`${API_PREFIX}/settings`);
    expect(calledInit.method).toBe('PUT');
    expect(calledInit.body).toBe(
      JSON.stringify({ defaults: { sort: 'uncertainty_entropy' } }),
    );
    expect(result.defaults).toEqual({ sort: 'uncertainty_entropy' });
  });

  it('putCurationDefaults accepts null to clear an axis, sending the literal null', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: {},
        updated_at: '2026-09-21T00:00:00+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await putCurationDefaults({ sort: null });

    const [, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledInit.body).toBe(JSON.stringify({ defaults: { sort: null } }));
    expect(result.defaults).toEqual({});
  });

  it('getCurationSettings THROWS on 404 (unlike getMethods, which falls back)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getCurationSettings()).rejects.toBeInstanceOf(ApiError);
    await expect(getCurationSettings()).rejects.toMatchObject({ status: 404 });

    // Contrast: getMethods() on the identical 404 response resolves rather
    // than rejecting. Same fetch mock, different endpoint contract.
    await expect(getMethods()).resolves.toEqual(FALLBACK_METHODS);
  });
});
