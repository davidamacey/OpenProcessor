/**
 * Tests for the diverse-selection wrappers.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import { cancelSelect, getSelectStatus, selectDiverse } from '$lib/api_select';

/**
 * P2-10: `/review`'s diverse overlay (`POST {API_PREFIX}/select/diverse`). The
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

  it('cancelSelect POSTs to {API_PREFIX}/select/cancel', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await cancelSelect();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/select/cancel`);
    expect(init.method).toBe('POST');
  });
});
