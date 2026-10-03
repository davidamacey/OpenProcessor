import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiFetch } from './api';

// A write the server may already have executed must never be re-sent by the
// transport: only GET/HEAD are retried.
describe('apiFetch retries reads only', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const fail504 = () =>
    new Response(JSON.stringify({ detail: 'gateway timeout' }), {
      status: 504,
    });

  for (const method of ['POST', 'PUT', 'PATCH', 'DELETE']) {
    it(`${method} that gets a 504 is sent exactly once`, async () => {
      const fetchMock = vi.fn().mockImplementation(async () => fail504());
      vi.stubGlobal('fetch', fetchMock);
      await expect(apiFetch('/x', { method, body: '{}' })).rejects.toMatchObject({
        status: 504,
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it(`${method} that loses its response (network error) is sent exactly once`, async () => {
      const fetchMock = vi.fn().mockRejectedValue(new TypeError('network'));
      vi.stubGlobal('fetch', fetchMock);
      await expect(apiFetch('/x', { method, body: '{}' })).rejects.toThrow();
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  }

  it('control: a GET that gets a 504 is still retried', async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(async () => fail504())
      .mockImplementationOnce(
        async () =>
          new Response('{"ok":true}', {
            status: 200,
            headers: { 'content-type': 'application/json' },
          }),
      );
    vi.stubGlobal('fetch', fetchMock);
    const p = apiFetch<{ ok: boolean }>('/x');
    await vi.advanceTimersByTimeAsync(5000);
    await expect(p).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
