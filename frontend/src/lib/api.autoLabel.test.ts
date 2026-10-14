/**
 * Tests for the auto-label job wrappers.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import {
  getAutoLabelJobStatus,
  pollAutoLabelJob,
  runVlmOnCluster,
  startAutoLabel,
} from '$lib/api_autoLabel';
import { ApiError } from '$lib/api_errors';

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

    await startAutoLabel({ train_clusters: true, vlm_concurrency: 16 });

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

  it('forwards prompt_pack when selected', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ prompt_pack: 'warehouse_v1' });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('prompt_pack=warehouse_v1');
  });

  it('composes the path from API_PREFIX', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/pipeline/auto_label/start`);
  });

  // G5: run_vlm was never sent at all before this change, so a scoped
  // run silently skipped the VLM stage it claimed to scope.
  it('forwards run_vlm=true when set', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ class_id: 7, run_vlm: true });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('run_vlm=true');
  });

  it('forwards vlm; acknowledge_external only when true (never =false); neither unset', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ vlm: 'cloud_vlm', acknowledge_external: true });
    await startAutoLabel({ vlm: 'cloud_vlm', acknowledge_external: false });
    await startAutoLabel({ vlm: null });

    const urls = fetchMock.mock.calls.map((c) => c[0] as string);
    expect(urls[0]).toContain('vlm=cloud_vlm');
    expect(urls[0]).toContain('acknowledge_external=true');
    expect(urls[1]).toContain('vlm=cloud_vlm');
    expect(urls[1]).not.toContain('acknowledge_external');
    expect(urls[2]).not.toContain('vlm');
    expect(urls[2]).not.toContain('acknowledge_external');
  });

  it('omits run_vlm entirely when unset (byte-identical to the pre-G5 request)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ train_clusters: true, vlm_concurrency: 16 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('run_vlm');
  });
});

/**
 * `runVlmOnCluster` — a single
 * `POST {API_PREFIX}/vlm/label_cluster/{id}[?prompt_pack=]`, replacing the
 * old fetch-200-crops-then-chunk-of-64 loop against
 * `{API_PREFIX}/vlm/label_batch`. No client-side crop selection or
 * chunking remains.
 */
describe('runVlmOnCluster', () => {
  const jobResponse = (overrides: Record<string, unknown> = {}) =>
    new Response(
      JSON.stringify({
        job_id: 'job-vlm-1',
        status: 'running',
        stage: 'vlm',
        processed: 0,
        total: 12,
        started_at: 0,
        finished_at: 0,
        error: null,
        result: {},
        args: {},
        eta_seconds: null,
        elapsed_seconds: 0,
        ...overrides,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs {API_PREFIX}/vlm/label_cluster/{id} with no crop-fetch round trip', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    const job = await runVlmOnCluster(42);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/vlm/label_cluster/42`);
    expect(init.method).toBe('POST');
    expect(job.job_id).toBe('job-vlm-1');
  });

  it('forwards prompt_pack when provided', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await runVlmOnCluster(42, { promptPack: 'generic_item_v1' });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('prompt_pack=generic_item_v1');
  });

  it('omits prompt_pack entirely when null/undefined', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await runVlmOnCluster(42, { promptPack: null });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('prompt_pack');
  });

  it('forwards vlm and acknowledge_external=true; sends neither unset, never acknowledge_external=false', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await runVlmOnCluster(42, { vlm: 'cloud_vlm', acknowledgeExternal: true });
    await runVlmOnCluster(42, { vlm: 'cloud_vlm', acknowledgeExternal: false });
    await runVlmOnCluster(42, {});

    const urls = fetchMock.mock.calls.map((c) => c[0] as string);
    expect(urls[0]).toBe(
      `${API_PREFIX}/vlm/label_cluster/42?vlm=cloud_vlm&acknowledge_external=true`,
    );
    expect(urls[1]).toBe(`${API_PREFIX}/vlm/label_cluster/42?vlm=cloud_vlm`);
    expect(urls[2]).toBe(`${API_PREFIX}/vlm/label_cluster/42`);
  });
});

describe('getAutoLabelJobStatus', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the job state on 200', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: 'job-1',
          status: 'running',
          stage: 'vlm',
          processed: 1,
          total: 2,
          started_at: 0,
          finished_at: null,
          error: null,
          args: {},
          eta_seconds: null,
          elapsed_seconds: 0,
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await getAutoLabelJobStatus('job-1');

    expect(job?.job_id).toBe('job-1');
    expect(fetchMock.mock.calls[0][0]).toContain('/pipeline/auto_label/status/job-1');
  });

  it('returns null on 404 (unknown job id)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'not found' }), {
        status: 404,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await getAutoLabelJobStatus('job-unknown');

    expect(job).toBeNull();
  });

  it('rethrows a non-404 error', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'boom' }), {
        status: 500,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getAutoLabelJobStatus('job-1')).rejects.toThrow(ApiError);
  });
});

describe('pollAutoLabelJob', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const statusResponse = (body: Record<string, unknown>) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  it('polls {API_PREFIX}/pipeline/auto_label/status until the job leaves running, calling onUpdate each time', async () => {
    const bodies = [
      { status: 'running', stage: 'vlm', processed: 1, total: 3 },
      { status: 'running', stage: 'vlm', processed: 2, total: 3 },
      { status: 'completed', stage: 'finalize', processed: 3, total: 3, result: {} },
    ];
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        statusResponse({
          job_id: 'job-vlm-1',
          started_at: 0,
          finished_at: 0,
          error: null,
          args: {},
          eta_seconds: null,
          elapsed_seconds: 0,
          ...bodies.shift(),
        }),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const updates: string[] = [];
    const final = await pollAutoLabelJob((j) => updates.push(j.status), undefined, 0);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(updates).toEqual(['running', 'running', 'completed']);
    expect(final.status).toBe('completed');
  });

  it('returns immediately (one fetch) when the first poll is already terminal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      statusResponse({
        job_id: 'job-vlm-1',
        status: 'failed',
        stage: 'vlm',
        processed: 0,
        total: 0,
        started_at: 0,
        finished_at: 0,
        error: 'boom',
        result: {},
        args: {},
        eta_seconds: null,
        elapsed_seconds: 0,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const final = await pollAutoLabelJob(() => {}, undefined, 0);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(final.status).toBe('failed');
    expect(final.error).toBe('boom');
  });

  /**
   * M7 (docs/design/interactive-pass-2026-09-24.md): with `expectedJobId`,
   * polls `GET {API_PREFIX}/pipeline/auto_label/status/{job_id}` — the
   * per-job endpoint the backend now serves (b654da5) — not the
   * "current/most recent job" `.../status` slot, and no longer needs to
   * skip a mismatched `job_id` client-side.
   */
  it('with expectedJobId, polls the per-job status/{job_id} endpoint directly', async () => {
    const bodies = [
      { status: 'running', stage: 'vlm', processed: 1, total: 3 },
      { status: 'completed', stage: 'finalize', processed: 3, total: 3, result: {} },
    ];
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      expect(url).toContain('/pipeline/auto_label/status/job-vlm-2');
      return Promise.resolve(
        statusResponse({
          job_id: 'job-vlm-2',
          started_at: 0,
          finished_at: 0,
          error: null,
          args: {},
          eta_seconds: null,
          elapsed_seconds: 0,
          ...bodies.shift(),
        }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const final = await pollAutoLabelJob(() => {}, undefined, 0, 'job-vlm-2');

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(final.status).toBe('completed');
  });

  it('with expectedJobId, treats a 404 as still-waiting and keeps polling until maxWaitMs', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'not found' }), {
        status: 404,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      pollAutoLabelJob(() => {}, undefined, 0, 'job-never-appears', 5),
    ).rejects.toThrow(/Timed out waiting for job job-never-appears/);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
  });
});
