/**
 * bakeoffAvailability — provisional `/bakeoff` capability probe
 * (docs/design/bakeoff-train-genericization-plan-2026-09-21.md §6 commit
 * 1). Mocks `bakeoffRuns` directly (rather than `fetch`, as
 * `strategies.svelte.test.ts` does) since the assertions here are about
 * how the store maps failure modes to `available`, not about the wire
 * format.
 *
 * The store is a module-scope singleton with an idempotent `init()`
 * (same shape as `strategiesStore`), so each test below gets its own
 * fresh module instance via `vi.resetModules()` + a per-test
 * `vi.doMock('$lib/api', ...)` + dynamic `import()`.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ApiError as ApiErrorType } from '$lib/api';

/**
 * `vi.resetModules()` between tests means a fresh `$lib/api` module
 * instance loads each time — so an `ApiError` constructed from a
 * top-level (pre-reset) import would fail `instanceof` against the
 * store's own (post-reset) `ApiError` reference. This resolves both the
 * thrown error and the mock from the exact same module instance to avoid
 * that trap.
 */
async function loadStoreThrowing(makeError: (ApiError: typeof ApiErrorType) => unknown) {
  vi.resetModules();
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  const err = makeError(actual.ApiError);
  vi.doMock('$lib/api', () => ({
    ...actual,
    bakeoffRuns: vi.fn(async () => {
      throw err;
    }),
  }));
  const { bakeoffAvailability } = await import('./bakeoffAvailability.svelte');
  return bakeoffAvailability;
}

async function loadStoreResolving(runs: { runs: unknown[] }) {
  vi.resetModules();
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  vi.doMock('$lib/api', () => ({
    ...actual,
    bakeoffRuns: vi.fn(async () => runs),
  }));
  const { bakeoffAvailability } = await import('./bakeoffAvailability.svelte');
  return bakeoffAvailability;
}

// Each test re-imports a fresh singleton after resetModules(). Compile api.ts
// and the store once up front so the first test doesn't pay the cold
// transform against its own timeout when the full suite is loaded.
beforeAll(async () => {
  await vi.importActual('$lib/api');
  await import('./bakeoffAvailability.svelte');
});

beforeEach(() => {
  vi.resetModules();
});

afterEach(() => {
  vi.doUnmock('$lib/api');
  vi.restoreAllMocks();
});

describe('bakeoffAvailability', () => {
  it('starts as null (optimistic — not yet determined)', async () => {
    const store = await loadStoreResolving({ runs: [] });
    expect(store.available).toBeNull();
  });

  it('sets available = true when bakeoffRuns() resolves', async () => {
    const store = await loadStoreResolving({ runs: [] });
    await store.init();
    expect(store.available).toBe(true);
  });

  it('sets available = false on a 404 (router not mounted)', async () => {
    const store = await loadStoreThrowing(
      (ApiError) => new ApiError(404, '/curation/bakeoff/runs', null),
    );
    await store.init();
    expect(store.available).toBe(false);
  });

  it('sets available = false on a 501 (not implemented)', async () => {
    const store = await loadStoreThrowing(
      (ApiError) => new ApiError(501, '/curation/bakeoff/runs', null),
    );
    await store.init();
    expect(store.available).toBe(false);
  });

  // The fail-open case this whole gate exists for: a transient network
  // failure or 5xx must NOT be indistinguishable from "route doesn't
  // exist" — that would hide a working route because of a blip.
  it('leaves available at its optimistic value (not false) on a network error', async () => {
    const store = await loadStoreThrowing(() => new TypeError('Failed to fetch'));
    await store.init();
    expect(store.available).not.toBe(false);
  });

  it('leaves available at its optimistic value (not false) on a 500', async () => {
    const store = await loadStoreThrowing(
      (ApiError) => new ApiError(500, '/curation/bakeoff/runs', null),
    );
    await store.init();
    expect(store.available).not.toBe(false);
  });

  it('is idempotent — a second init() does not re-fetch', async () => {
    vi.resetModules();
    const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
    const impl = vi.fn(async () => ({ runs: [] }));
    vi.doMock('$lib/api', () => ({ ...actual, bakeoffRuns: impl }));
    const { bakeoffAvailability: store } = await import('./bakeoffAvailability.svelte');

    await store.init();
    await store.init();
    await store.init();

    expect(impl).toHaveBeenCalledTimes(1);
  });
});
