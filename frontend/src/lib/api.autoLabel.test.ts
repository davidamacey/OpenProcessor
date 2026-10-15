/**
 * Tests for the auto-label job wrappers.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { startAutoLabel } from '$lib/api_autoLabel';

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
