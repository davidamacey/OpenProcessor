/**
 * Tests for `getClusters` / `getCluster` (the cluster card list and one card):
 * the card is read from the served `ClusterCard` by `cluster_uid`, with no field
 * the server stopped serving (kind, dominant class id, purity, promotable,
 * sub-clusters) and no client recomputation.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { getCluster, getClusters } from '$lib/api_clusters';

const UID_A = '0b9f9a38-0000-4000-8000-00000000000a';
const UID_B = '0b9f9a38-0000-4000-8000-00000000000b';

const jsonResponse = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

function card(over: Record<string, unknown> = {}) {
  return {
    cluster_uid: UID_A,
    display_no: 3,
    size: 37,
    validated_count: 30,
    labelled_count: 37,
    dominant_class_name: 'widget_b',
    dominant_count: 37,
    label_agreement: 1.0,
    labelled_share: 1.0,
    is_unlabeled: false,
    updated_at: null,
    representatives: [{ crop_id: 'c1', cluster_score: 0.9, class_name: 'widget_b' }],
    ...over,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getClusters card mapping', () => {
  it('keys the card by cluster_uid and reads the served fields verbatim', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [
            card(),
            card({
              cluster_uid: UID_B,
              dominant_class_name: null,
              is_unlabeled: true,
              label_agreement: null,
              labelled_share: 0,
              representatives: [],
            }),
          ],
          total: 2,
        }),
      ),
    );

    const res = await getClusters();

    expect(res.items[0]).toMatchObject({
      uid: UID_A,
      size: 37,
      dominant_class_name: 'widget_b',
      representative_crop_ids: ['c1'],
    });
    expect(res.items[1]).toMatchObject({ uid: UID_B, is_unlabeled: true });
    // None of the retired fields is invented client-side.
    for (const gone of [
      'id',
      'cluster_kind',
      'dominant_class_id',
      'purity',
      'purity_tier',
      'promotable',
      'n_subclusters',
      'subcluster_ids',
      'core_similarity_min',
    ]) {
      expect(res.items[0]).not.toHaveProperty(gone);
    }
    expect(res).not.toHaveProperty('purity_thresholds');
  });

  it('carries the served display_no, null when no partition numbers the cluster', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ items: [card(), card({ display_no: null })], total: 2 }),
        ),
    );
    const res = await getClusters();
    expect(res.items.map((c) => c.display_no)).toEqual([3, null]);
  });

  it('dominant_pct is the served label_agreement', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [
            card({ label_agreement: 0.82, dominant_count: 31, labelled_count: 37 }),
          ],
          total: 1,
        }),
      ),
    );
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({
      dominant_pct: 0.82,
      label_agreement: 0.82,
      dominant_count: 31,
      labelled_count: 37,
    });
  });

  it('maps the quality fields and leaves the optional ones empty when omitted', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [
            card({
              detector_top_class: 'widget_a',
              detector_top_share: 0.62,
              detector_conf_mean: 0.81,
              validation_state: 'partial',
              review_counts: { all: 7, outliers: 3 },
              needs_review_count: 7,
              flags: ['low_label_agreement'],
            }),
            card({ cluster_uid: UID_B }),
          ],
          total: 2,
        }),
      ),
    );
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({
      detector_top_class: 'widget_a',
      detector_top_share: 0.62,
      detector_conf_mean: 0.81,
      validation_state: 'partial',
      review_counts: { all: 7, outliers: 3 },
      needs_review_count: 7,
      flags: ['low_label_agreement'],
    });
    expect(res.items[1]).toMatchObject({
      detector_top_class: null,
      review_counts: {},
      needs_review_count: 0,
      flags: [],
    });
  });
});

describe('getCluster', () => {
  function stub(clustersResponse: () => Response) {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockImplementation((url: string) =>
          Promise.resolve(
            url.startsWith(`${API_PREFIX}/crops`)
              ? jsonResponse({ total: 3, page: 1, page_size: 60, crops: [] })
              : clustersResponse(),
          ),
        ),
    );
  }

  it('reads the served card of the uid from its own route, with no list call', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation((url: string) =>
        Promise.resolve(
          url.startsWith(`${API_PREFIX}/crops`)
            ? jsonResponse({ total: 3, page: 1, page_size: 60, crops: [] })
            : jsonResponse(card()),
        ),
      );
    vi.stubGlobal('fetch', fetchMock);
    const res = await getCluster(UID_A);
    expect(res.cluster).toMatchObject({ uid: UID_A, dominant_class_name: 'widget_b' });
    expect(res.cardError).toBeNull();
    const urls = fetchMock.mock.calls.map((c) => c[0] as string);
    expect(urls.some((u) => u.includes(`/clusters/${UID_A}`))).toBe(true);
    expect(urls.some((u) => u.includes('max_clusters'))).toBe(false);
  });

  it('has no card, and names the served reason, when the card route refuses', async () => {
    stub(
      () =>
        new Response(JSON.stringify({ detail: 'down' }), {
          status: 404,
          headers: { 'content-type': 'application/json' },
        }),
    );
    const res = await getCluster(UID_A);
    expect(res.cluster).toBeNull();
    expect(res.crops.total).toBe(3);
    expect(res.cardError).toContain('down');
  });
});

describe('getClusters D-4 representatives windowing', () => {
  it('sends representatives_offset/representatives_limit as offset/limit', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        representatives_offset: 24,
        representatives_limit: 24,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await getClusters({ representatives_offset: 24, representatives_limit: 24 });

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).toContain('offset=24');
    expect(calledUrl).toContain('limit=24');
  });

  it('echoes the response window back, not the requested one', async () => {
    // The backend can serve a smaller window than requested (e.g. near the
    // end of the list) — the caller must advance by what actually came back.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [],
          total: 30,
          representatives_offset: 24,
          representatives_limit: 6,
        }),
      ),
    );

    const res = await getClusters({
      representatives_offset: 24,
      representatives_limit: 24,
    });

    expect(res.representatives_offset).toBe(24);
    expect(res.representatives_limit).toBe(6);
  });

  it('omits offset/limit query params when no window is requested', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    vi.stubGlobal('fetch', fetchMock);

    await getClusters({});

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).not.toContain('offset=');
    expect(calledUrl).not.toContain('limit=');
  });
});
