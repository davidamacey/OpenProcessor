/**
 * Regression coverage for the cross-origin plate-thumbnail 404 bug
 * (2026-09-12 report): with `PUBLIC_TRITON_API_URL` set to a remote
 * host, several call sites built plate-thumbnail `<img src>` values as
 * bare relative region-thumbnail paths instead of
 * going through `apiBase`, so the browser resolved them against the
 * frontend's OWN origin instead of the configured remote openprocessor.
 *
 * `apiBase` (and therefore `getThumbUrl`/`getRegionThumbUrl`/
 * `resolveApiUrl`) is computed once at module load from
 * `import.meta.env.PUBLIC_TRITON_API_URL`, so exercising the
 * non-empty-base case requires stubbing the env var, resetting the
 * module registry, and re-importing fresh — `vi.stubEnv` +
 * `vi.resetModules()` + a dynamic `import('./api')` per test.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

const REMOTE_BASE = 'http://remote-host:4603';

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

async function loadApiWithRemoteBase() {
  vi.stubEnv('PUBLIC_TRITON_API_URL', REMOTE_BASE);
  vi.resetModules();
  return import('./api');
}

describe('getRegionThumbUrl', () => {
  it('prefixes the configured remote apiBase', async () => {
    const { getRegionThumbUrl, API_PREFIX } = await loadApiWithRemoteBase();
    expect(getRegionThumbUrl('abc')).toBe(
      `${REMOTE_BASE}${API_PREFIX}/crops/abc/region_thumbnail?size=160`,
    );
  });

  it('accepts a custom size', async () => {
    const { getRegionThumbUrl, API_PREFIX } = await loadApiWithRemoteBase();
    expect(getRegionThumbUrl('abc', 320)).toBe(
      `${REMOTE_BASE}${API_PREFIX}/crops/abc/region_thumbnail?size=320`,
    );
  });

  it('assembles a cache-busting `v` param when a cacheBustKey is passed', async () => {
    const { getRegionThumbUrl, API_PREFIX } = await loadApiWithRemoteBase();
    expect(getRegionThumbUrl('abc', 160, 12345)).toBe(
      `${REMOTE_BASE}${API_PREFIX}/crops/abc/region_thumbnail?size=160&v=12345`,
    );
  });

  it('omits the `v` param entirely when no cacheBustKey is given', async () => {
    const { getRegionThumbUrl } = await loadApiWithRemoteBase();
    expect(getRegionThumbUrl('abc')).not.toMatch(/v=/);
  });

  it('encodes the crop id', async () => {
    const { getRegionThumbUrl } = await loadApiWithRemoteBase();
    expect(getRegionThumbUrl('a/b')).toContain('a%2Fb');
  });
});

describe('resolveApiUrl', () => {
  it('prefixes a bare relative {API_PREFIX}/... path with the configured remote apiBase', async () => {
    const { resolveApiUrl, API_PREFIX } = await loadApiWithRemoteBase();
    expect(resolveApiUrl(`${API_PREFIX}/crops/abc/region_thumbnail?size=160`)).toBe(
      `${REMOTE_BASE}${API_PREFIX}/crops/abc/region_thumbnail?size=160`,
    );
  });

  it('is idempotent — a no-op on an already-absolute URL, never double-prefixes', async () => {
    const { resolveApiUrl, API_PREFIX } = await loadApiWithRemoteBase();
    const once = resolveApiUrl(`${API_PREFIX}/crops/abc/region_thumbnail`);
    const twice = resolveApiUrl(once);
    expect(twice).toBe(once);
    expect(twice.match(new RegExp(REMOTE_BASE, 'g'))).toHaveLength(1);
  });

  it('leaves a foreign absolute URL (e.g. http already present) untouched', async () => {
    const { resolveApiUrl } = await loadApiWithRemoteBase();
    expect(resolveApiUrl('http://other-host/x')).toBe('http://other-host/x');
  });

  it('is a plain no-op prefix (empty apiBase) when PUBLIC_TRITON_API_URL is unset', async () => {
    vi.stubEnv('PUBLIC_TRITON_API_URL', '');
    vi.resetModules();
    const { resolveApiUrl, API_PREFIX } = await import('./api');
    expect(resolveApiUrl(`${API_PREFIX}/crops/abc/region_thumbnail`)).toBe(
      `${API_PREFIX}/crops/abc/region_thumbnail`,
    );
  });
});
