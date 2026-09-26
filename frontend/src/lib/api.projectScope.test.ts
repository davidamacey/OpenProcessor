/**
 * Groundwork for multi-project support (`docs/design/
 * any-domain-rev3-and-projects-contract-review-2026-09-26.md` §7): every
 * scoped backend call builds its URL through `scoped()`, a one-function
 * choke point over a module-level holder. These tests prove the holder
 * actually drives request URLs — not just that `scoped()` returns the
 * right string in isolation — by mocking `fetch` and reading the URL a
 * representative sample of `api.ts` wrappers actually requested.
 *
 * `setScopedPrefix`/`scoped` are a shared module singleton, so every test
 * restores the default prefix afterwards — a leaked override here would
 * silently break every other test file that asserts against `API_PREFIX`
 * literals.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  activeProjectKey,
  API_PREFIX,
  getMethods,
  getStats,
  globalApi,
  scoped,
  setScopedPrefix,
} from './api';

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

describe('scoped()/globalApi() (multi-project groundwork)', () => {
  afterEach(() => {
    setScopedPrefix(API_PREFIX);
  });

  it('defaults to API_PREFIX, so every URL is unchanged today', () => {
    expect(scoped()).toBe(API_PREFIX);
    expect(activeProjectKey()).toBe(API_PREFIX);
  });

  it('globalApi() always returns API_PREFIX, independent of the scoped holder', () => {
    setScopedPrefix(`${API_PREFIX}/projects/acme`);
    expect(globalApi()).toBe(API_PREFIX);
  });

  it('setScopedPrefix() changes scoped() and activeProjectKey() together', () => {
    const acme = `${API_PREFIX}/projects/acme`;
    setScopedPrefix(acme);
    expect(scoped()).toBe(acme);
    expect(activeProjectKey()).toBe(acme);
  });

  it('getStats() builds both its request URLs from the current scoped prefix', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/stats/dataset')) {
        return Promise.resolve(
          jsonResponse({ total_crops: 0, validated: 0, test_holdout: 0 }),
        );
      }
      return Promise.resolve(jsonResponse({ classes: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);

    await getStats();
    const urlsAtDefault = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urlsAtDefault).toHaveLength(2);
    expect(urlsAtDefault.every((u) => u.includes(`${API_PREFIX}/stats/`))).toBe(true);

    fetchMock.mockClear();
    const acme = `${API_PREFIX}/projects/acme`;
    setScopedPrefix(acme);
    await getStats();
    const urlsScoped = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urlsScoped).toHaveLength(2);
    expect(urlsScoped.every((u) => u.includes(`${acme}/stats/`))).toBe(true);
    expect(urlsScoped.every((u) => !u.includes(`${API_PREFIX}/stats/`))).toBe(true);
  });

  it('getMethods() builds its request URL from the current scoped prefix', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ cluster_methods: [], review_sorts: [], overlays: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await getMethods();
    expect(String(fetchMock.mock.calls[0]![0])).toBe(`${API_PREFIX}/methods`);

    fetchMock.mockClear();
    const widgetco = `${API_PREFIX}/projects/widgetco`;
    setScopedPrefix(widgetco);
    await getMethods();
    expect(String(fetchMock.mock.calls[0]![0])).toBe(`${widgetco}/methods`);
  });
});
