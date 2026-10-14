/**
 * Tests for the HTTP core and error shapes: ApiError message composition,
 * apiErrorText, API_PREFIX, the download-URL builders and apiFetch retries.
 *
 * Every UI callsite renders `(e as Error).message`, so the server's reason
 * has to be baked into the message or the operator never sees it.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiBase, API_PREFIX, apiFetch, normalizeApiPrefix } from '$lib/api';
import { ApiError, apiErrorText } from '$lib/api_errors';
import { getClassRegistryUrl, getDataYamlUrl, getManifestUrl } from '$lib/api_export';

const URL = `http://localhost:4603${API_PREFIX}/crops/batch_label`;

describe('ApiError', () => {
  it("appends a FastAPI 'detail' string to the message", () => {
    const e = new ApiError(422, URL, { detail: 'region bbox outside crop envelope' });
    expect(e.detail).toBe('region bbox outside crop envelope');
    expect(e.message).toBe(`API 422 ${URL} — region bbox outside crop envelope`);
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

  it('shows the served message sentence of a structured detail, not its error code', () => {
    const e = new ApiError(409, URL, {
      detail: { error: 'revision_conflict', message: 'Someone else saved revision 4.' },
    });
    expect(e.detail).toBe('Someone else saved revision 4.');
    expect(e.message).toBe(`API 409 ${URL} — Someone else saved revision 4.`);
  });

  it('control: a structured detail with no message still shows its code', () => {
    const e = new ApiError(422, URL, { detail: { error: 'region_text_disabled' } });
    expect(e.detail).toBe('region_text_disabled');
  });

  it('honors an explicit message override', () => {
    const e = new ApiError(404, URL, { detail: 'nope' }, 'custom');
    expect(e.message).toBe('custom');
    expect(e.detail).toBe('nope');
  });
});

describe('apiErrorText', () => {
  it('prefers the structured message, then the detail, then the error message', () => {
    expect(
      apiErrorText(
        new ApiError(409, URL, { detail: { error: 'x_code', message: 'The sentence.' } }),
      ),
    ).toBe('The sentence.');
    expect(apiErrorText(new ApiError(503, URL, { detail: 'segmenter down' }))).toBe(
      'segmenter down',
    );
    expect(
      apiErrorText(
        new ApiError(422, URL, {
          detail: [{ loc: ['body', 'name'], msg: 'Field required' }],
        }),
      ),
    ).toBe('name: Field required');
    expect(apiErrorText(new Error('network down'))).toBe('network down');
    expect(apiErrorText(new ApiError(500, URL, null))).toBe(`API 500 ${URL}`);
  });
});

/**
 * Regression test for the /export download-button filename mismatch:
 * getDataYamlUrl() used to request `data_v7.yaml`, which the export service
 * never writes (the real on-disk file is `data.yaml`), so the download would
 * 404 against the real `{API_PREFIX}/export/registry/{artifact}` backend contract.
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
 * OpenProcessor 3cd4ca87: a 503 (e.g. Triton unreachable) sends
 * `Retry-After: 5` — apiFetch honours it in place of that attempt's fixed
 * backoff delay, still within the existing 3-retry budget, and clamps an
 * oversized value rather than extending the wait unboundedly.
 */
describe('apiFetch 503 Retry-After', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('waits the served Retry-After (seconds) before retrying, then succeeds', async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            detail: 'Inference backend (Triton) is temporarily unavailable',
          }),
          {
            status: 503,
            headers: { 'Retry-After': '3' },
          },
        ),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    const p = apiFetch<{ ok: boolean }>('/x');
    // Flush the first attempt's fetch resolution before asserting timers.
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // A hair under the served 3s: no retry fired yet.
    await vi.advanceTimersByTimeAsync(2900);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Past the served 3s: the retry fires.
    await vi.advanceTimersByTimeAsync(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await expect(p).resolves.toEqual({ ok: true });
  });

  it('clamps an oversized Retry-After to the bounded max, never stalling past it', async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'unavailable' }), {
          status: 503,
          headers: { 'Retry-After': '600' },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    const p = apiFetch<{ ok: boolean }>('/x');
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Clamped to 5s (MAX_RETRY_AFTER_MS), nowhere near the served 600s.
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await expect(p).resolves.toEqual({ ok: true });
  });

  it('fails with the served detail after the retry budget is exhausted', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Response(
          JSON.stringify({
            detail: 'Inference backend (Triton) is temporarily unavailable',
          }),
          { status: 503, headers: { 'Retry-After': '1' } },
        ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const p = apiFetch<unknown>('/x');
    p.catch(() => {});
    // 1 initial + 3 retries at the served 1s each = well within 10s.
    for (let i = 0; i < 4; i++) {
      await vi.advanceTimersByTimeAsync(1000);
    }
    await expect(p).rejects.toMatchObject({
      status: 503,
      detail: 'Inference backend (Triton) is temporarily unavailable',
    });
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });
});
