/**
 * Tests for `getClusters` (the cluster card list).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { getCluster, getClusters } from '$lib/api_clusters';

/**
 * `getClusters`/`getCluster` map the served `purity_tier`, `promotable`
 * and `core_similarity_min` — no client 0.8
 * "pure" threshold, no recomputation.
 */
describe('getClusters purity_tier/promotable/core_similarity_min', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps purity_tier, promotable and core_similarity_min from the response verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            cluster_id: 67,
            cluster_kind: 'class',
            size: 37,
            validated_count: 30,
            labelled_count: 37,
            dominant_class_id: 67,
            dominant_class_name: 'widget_b',
            dominant_count: 37,
            purity: 1.0,
            purity_tier: 'pure',
            promotable: true,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
          {
            cluster_id: 10000,
            cluster_kind: 'candidate',
            size: 108,
            validated_count: 0,
            labelled_count: 3,
            dominant_class_id: null,
            dominant_class_name: null,
            dominant_count: 1,
            purity: 0.33,
            purity_tier: 'noisy',
            promotable: false,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
        ],
        total: 2,
        total_class_clusters: 1,
        total_candidate_clusters: 1,
        cluster_id_offset: 10000,
        purity_thresholds: {
          pure_min: 0.85,
          mixed_min: 0.6,
          promote_min_members: 4,
          promote_min_labelled_share: 0.5,
        },
        core_similarity_min: 0.75,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters();

    expect(res.items[0]).toMatchObject({
      id: 67,
      purity_tier: 'pure',
      promotable: true,
      core_similarity_min: 0.75,
    });
    expect(res.items[1]).toMatchObject({
      id: 10000,
      purity_tier: 'noisy',
      promotable: false,
      core_similarity_min: 0.75,
    });
    // m7 (2026-09-24 interactive pass): the /clusters legend used to
    // hardcode "≥80% / ≥60%" instead of reading this.
    expect(res.purity_thresholds).toMatchObject({ pure_min: 0.85, mixed_min: 0.6 });
  });

  it('C1: dominant_pct is the served label_purity, never the geometry purity', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [
            {
              cluster_id: 64,
              cluster_kind: 'class',
              size: 616,
              validated_count: 0,
              labelled_count: 616,
              dominant_class_id: 64,
              dominant_class_name: 'class_b',
              dominant_count: 616,
              purity: 0.03,
              purity_n: 616,
              purity_basis: 'nearest_centroid',
              purity_tier: 'noisy',
              label_purity: 1.0,
              labelled_share: 1.0,
              promotable: false,
              is_unlabeled: false,
              n_subclusters: 0,
              updated_at: null,
              representatives: [],
            },
          ],
          total: 1,
          total_class_clusters: 1,
          total_candidate_clusters: 0,
          cluster_id_offset: 10000,
        }),
      ),
    );
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({
      dominant_pct: 1.0,
      purity: 0.03,
      dominant_count: 616,
      labelled_count: 616,
    });
  });

  it('#61: label_agreement is read for label quality (dominant_pct); purity stays geometry-only', async () => {
    const card = (extra: Record<string, unknown>) => ({
      cluster_id: 64,
      cluster_kind: 'class',
      size: 10,
      validated_count: 0,
      labelled_count: 10,
      dominant_class_id: 64,
      dominant_class_name: 'class_b',
      dominant_count: 9,
      purity: 0.4,
      purity_n: 10,
      purity_basis: 'nearest_centroid',
      purity_tier: 'noisy',
      labelled_share: 1.0,
      promotable: false,
      is_unlabeled: false,
      n_subclusters: 0,
      updated_at: null,
      representatives: [],
      ...extra,
    });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [
            card({ label_agreement: 0.9, label_purity: 0.9 }),
            card({ cluster_id: 65, label_agreement: null, label_purity: null }),
            card({ cluster_id: 66, label_purity: 0.7 }),
          ],
          total: 3,
          total_class_clusters: 3,
          total_candidate_clusters: 0,
          cluster_id_offset: 10000,
        }),
      ),
    );
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({
      label_agreement: 0.9,
      dominant_pct: 0.9,
      purity: 0.4,
      purity_tier: 'noisy',
    });
    expect(res.items[1]).toMatchObject({ label_agreement: null, dominant_pct: null });
    // An older backend that only serves label_purity still renders.
    expect(res.items[2]).toMatchObject({ label_agreement: 0.7, dominant_pct: 0.7 });
  });

  it('getClusters reports purity_thresholds as null when the server omits it, never a stale hardcoded value', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        total_class_clusters: 0,
        total_candidate_clusters: 0,
        cluster_id_offset: 0,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const res = await getClusters();
    expect(res.purity_thresholds ?? null).toBeNull();
  });

  it('getCluster carries purity_tier/promotable/core_similarity_min through for the single-cluster lookup', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.resolve(
        jsonResponse({
          items: [
            {
              cluster_id: 67,
              cluster_kind: 'class',
              size: 37,
              validated_count: 30,
              labelled_count: 37,
              dominant_class_id: 67,
              dominant_class_name: 'widget_b',
              dominant_count: 37,
              purity: 1.0,
              purity_tier: 'pure',
              promotable: true,
              is_unlabeled: false,
              n_subclusters: 0,
              updated_at: null,
              representatives: [],
            },
          ],
          total: 1,
          total_class_clusters: 1,
          total_candidate_clusters: 0,
          cluster_id_offset: 10000,
          core_similarity_min: 0.75,
        }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCluster(67, 1, 60);

    expect(res.cluster).toMatchObject({
      id: 67,
      cluster_kind: 'class',
      purity_tier: 'pure',
      promotable: true,
      core_similarity_min: 0.75,
    });
  });

  it('getCluster falls back to a null-identity stub (purity_tier null, promotable false) when the cluster-card lookup fails', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.reject(new Error('network blip'));
    });
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCluster(999, 1, 60);

    expect(res.cluster.purity_tier).toBeNull();
    expect(res.cluster.promotable).toBe(false);
    expect(res.cluster.core_similarity_min).toBeNull();
  });
});

/**
 * DQ-M2 fix (dq-queues cutover, 2026-09-24): `purity` is now
 * nearest-centroid geometry purity (`purity_basis: 'nearest_centroid'`,
 * `purity_n` members measured), independent of `label_purity`
 * (the old label-based number, always 1.0 for a class cluster) and
 * `labelled_share`. `getClusters` must map all four verbatim.
 */
describe('getClusters purity_n/purity_basis/label_purity/labelled_share (DQ-M2)', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps purity_n/purity_basis/label_purity/labelled_share verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            cluster_id: 43,
            cluster_kind: 'class',
            size: 24,
            validated_count: 0,
            labelled_count: 24,
            dominant_class_id: 43,
            dominant_class_name: 'motardbike',
            dominant_count: 24,
            purity: 0.29,
            purity_n: 24,
            purity_basis: 'nearest_centroid',
            purity_tier: 'noisy',
            label_purity: 1.0,
            labelled_share: 1.0,
            promotable: false,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
        ],
        total: 1,
        total_class_clusters: 1,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters();

    expect(res.items[0]).toMatchObject({
      id: 43,
      purity: 0.29,
      purity_n: 24,
      purity_basis: 'nearest_centroid',
      label_purity: 1.0,
      labelled_share: 1.0,
    });
    // The whole point of DQ-M2: purity and label_purity must be able to
    // disagree (a class cluster is no longer 1.0-purity by construction).
    expect(res.items[0].purity).not.toBe(res.items[0].label_purity);
  });

  it('leaves the new fields null when the server omits them (older backend)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            cluster_id: 67,
            cluster_kind: 'class',
            size: 10,
            validated_count: 0,
            labelled_count: 10,
            dominant_class_id: 67,
            dominant_class_name: 'widget_b',
            dominant_count: 10,
            purity: 1.0,
            purity_tier: 'pure',
            promotable: false,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
        ],
        total: 1,
        total_class_clusters: 1,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters();
    expect(res.items[0].purity_n ?? null).toBeNull();
    expect(res.items[0].purity_basis ?? null).toBeNull();
    expect(res.items[0].label_purity ?? null).toBeNull();
    expect(res.items[0].labelled_share ?? null).toBeNull();
  });
});

/**
 * D-4 (docs/design/curation_query_performance_audit.md): `/clusters`
 * representatives are paged by offset/limit independent of the card list
 * itself. `getClusters` forwards `representatives_offset`/
 * `representatives_limit` as `offset`/`limit` query params and echoes the
 * response's window back so a caller can advance past exactly what it got.
 */
describe('getClusters D-4 representatives windowing', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends representatives_offset/representatives_limit as offset/limit', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        total_class_clusters: 0,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
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

  it('echoes the response window back on ClustersResponse, not the requested one', async () => {
    // The backend can serve a smaller window than requested (e.g. near the
    // end of the list) — the caller must advance by what actually came
    // back, not by what it asked for, or it'll skip/repeat cards.
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 30,
        total_class_clusters: 30,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
        representatives_offset: 24,
        representatives_limit: 6,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters({
      representatives_offset: 24,
      representatives_limit: 24,
    });

    expect(res.representatives_offset).toBe(24);
    expect(res.representatives_limit).toBe(6);
  });

  it('omits offset/limit query params when no window is requested', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        total_class_clusters: 0,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await getClusters({});

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).not.toContain('offset=');
    expect(calledUrl).not.toContain('limit=');
  });
});
