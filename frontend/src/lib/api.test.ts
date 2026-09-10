/**
 * Tests for ApiError's message composition.
 *
 * Every UI callsite renders `(e as Error).message`, so the server's reason
 * has to be baked into the message or the operator never sees it.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, getMethods } from './api';
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
    const serverBody = {
      cluster_methods: [
        {
          id: 'ivf',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
      ],
      review_sorts: [
        { id: 'default', label: 'Recent first', status: 'stable', default: true },
        { id: 'uncertainty', label: 'Uncertainty margin', status: 'experimental' },
      ],
      overlays: [],
      scores: [],
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(serverBody));
    vi.stubGlobal('fetch', fetchMock);

    const result = await getMethods();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.cluster_methods).toEqual(serverBody.cluster_methods);
    expect(result.review_sorts).toEqual(serverBody.review_sorts.map((s) => ({ ...s })));
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
