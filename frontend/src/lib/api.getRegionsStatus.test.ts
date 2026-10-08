/**
 * dq-region (2026-09-24): `getRegions` forwards `RegionsQuery.status` to
 * `GET {API_PREFIX}/regions?status=` (the backend 400s on an unknown value;
 * see contracts/openprocessor's `/regions` operation). Covers the wire
 * param only — the status vocabulary's own contract is
 * regionStatus.test.ts.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getRegions, API_PREFIX } from './api';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getRegions — status filter param', () => {
  it('sends status= when RegionsQuery.status is set', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 60, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getRegions('/regions', { status: 'verify_rejected' });

    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toBe(`${API_PREFIX}/regions?status=verify_rejected`);
  });

  it('omits status= entirely when unset', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 60, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getRegions('/regions', {});

    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).not.toContain('status=');
  });
});
