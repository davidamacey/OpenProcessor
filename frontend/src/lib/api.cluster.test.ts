/**
 * Tests for `getCluster` (one cluster's crops page).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { getCluster } from '$lib/api_clusters';

/**
 * getCluster's `order` param is forwarded verbatim to `{API_PREFIX}/crops?order=`
 * (broadened from a fixed `'outliers'` literal so a future
 * `{API_PREFIX}/methods`-reported order id doesn't require touching this
 * signature — see the comment on `order` in api.ts). An id the backend
 * doesn't recognize should be harmless: qs() still sends it, and callers
 * are responsible for only offering ids `{API_PREFIX}/methods` actually reports.
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

  it('forwards an arbitrary order id to {API_PREFIX}/crops without special-casing it client-side', async () => {
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

  it('hands back the served n_unembedded and suggested_reprocess of an ordered view', async () => {
    const request = {
      targets: { filter: { embedding_state: ['deferred'] } },
      scopes: ['embed'],
      dry_run: true,
    };
    const fetchMock = vi.fn().mockImplementation((url: string) =>
      Promise.resolve(
        url.startsWith(`${API_PREFIX}/crops`)
          ? jsonResponse({
              total: 0,
              page: 1,
              page_size: 60,
              crops: [],
              n_unembedded: 9,
              suggested_reprocess: request,
            })
          : jsonResponse({ items: [] }),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCluster(42, 1, 60, undefined, { order: 'outliers' });
    expect(res.crops.n_unembedded).toBe(9);
    expect(res.crops.suggested_reprocess).toEqual(request);

    const plain = await getCluster(42, 1, 60, undefined, {});
    fetchMock.mockImplementation((url: string) =>
      Promise.resolve(
        url.startsWith(`${API_PREFIX}/crops`)
          ? jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] })
          : jsonResponse({ items: [] }),
      ),
    );
    const none = await getCluster(42, 1, 60, undefined, {});
    expect(none.crops.n_unembedded).toBeNull();
    expect(none.crops.suggested_reprocess).toBeNull();
    expect(plain).toBeDefined();
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
 * caller (gated by {API_PREFIX}/methods, see strategies.test.ts's
 * isDiverseOverlayAvailable coverage) decided to send.
 */
describe('getCluster card lookup failure', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stub(clustersResponse: () => Response) {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockImplementation((url: string) =>
          Promise.resolve(
            url.startsWith(`${API_PREFIX}/crops`)
              ? new Response(
                  JSON.stringify({ total: 3, page: 1, page_size: 60, crops: [] }),
                  { status: 200, headers: { 'content-type': 'application/json' } },
                )
              : clustersResponse(),
          ),
        ),
    );
  }

  it('reports why the cluster card could not be read instead of hiding the failure', async () => {
    stub(
      () =>
        new Response(JSON.stringify({ detail: 'aggregation unavailable' }), {
          status: 400,
          headers: { 'content-type': 'application/json' },
        }),
    );
    const res = await getCluster(42);
    expect(res.cardError).toBe('aggregation unavailable');
    expect(res.crops.total).toBe(3);
  });

  it('control: a served card reports no error', async () => {
    stub(
      () =>
        new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
    );
    expect((await getCluster(42)).cardError).toBeNull();
  });
});

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

  it('forwards k to {API_PREFIX}/crops when set alongside order=diverse', async () => {
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

describe('getCluster cluster_subid param', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  const crops = (fetchMock: ReturnType<typeof vi.fn>) =>
    fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`)) ?? '';

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stub() {
    const fetchMock = vi
      .fn()
      .mockImplementation((url: string) =>
        Promise.resolve(
          url.startsWith(`${API_PREFIX}/crops`)
            ? jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] })
            : jsonResponse({ items: [] }),
        ),
      );
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  it('asks the server for one sub-cluster instead of filtering the loaded page', async () => {
    const fetchMock = stub();
    await getCluster(42, 1, 60, undefined, { clusterSubid: '42b' });
    const url = crops(fetchMock);
    expect(url).toContain('cluster_id=42');
    expect(url).toContain('cluster_subid=42b');
  });

  it('omits cluster_subid for the whole cluster', async () => {
    const fetchMock = stub();
    await getCluster(42, 1, 60, undefined, { clusterSubid: null });
    expect(crops(fetchMock)).not.toContain('cluster_subid');
  });
});
