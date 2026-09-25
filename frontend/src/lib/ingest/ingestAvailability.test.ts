/**
 * ingestAvailability — mirrors `bakeoffAvailability.test.ts`'s structure
 * exactly (see that file's header for why `vi.resetModules()` +
 * per-test `vi.doMock` + dynamic `import()` is needed here).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ApiError as ApiErrorType } from '$lib/api';

async function loadStoreThrowing(makeError: (ApiError: typeof ApiErrorType) => unknown) {
  vi.resetModules();
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  const err = makeError(actual.ApiError);
  vi.doMock('$lib/api', () => ({
    ...actual,
    getIngestStatus: vi.fn(async () => {
      throw err;
    }),
  }));
  const { ingestAvailability } = await import('./ingestAvailability.svelte');
  return ingestAvailability;
}

async function loadStoreResolving(status: { total: number; by_source: []; by_day: [] }) {
  vi.resetModules();
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  vi.doMock('$lib/api', () => ({
    ...actual,
    getIngestStatus: vi.fn(async () => status),
  }));
  const { ingestAvailability } = await import('./ingestAvailability.svelte');
  return ingestAvailability;
}

beforeEach(() => {
  vi.resetModules();
});

afterEach(() => {
  vi.doUnmock('$lib/api');
  vi.restoreAllMocks();
});

describe('ingestAvailability', () => {
  it('starts as null (optimistic — not yet determined)', async () => {
    const store = await loadStoreResolving({ total: 0, by_source: [], by_day: [] });
    expect(store.available).toBeNull();
  });

  it('sets available = true when getIngestStatus() resolves', async () => {
    const store = await loadStoreResolving({ total: 0, by_source: [], by_day: [] });
    await store.init();
    expect(store.available).toBe(true);
  });

  it('sets available = false on a 404 (router not mounted)', async () => {
    const store = await loadStoreThrowing(
      (ApiError) => new ApiError(404, '/curation/ingest/status', null),
    );
    await store.init();
    expect(store.available).toBe(false);
  });

  it('sets available = false on a 501 (not implemented)', async () => {
    const store = await loadStoreThrowing(
      (ApiError) => new ApiError(501, '/curation/ingest/status', null),
    );
    await store.init();
    expect(store.available).toBe(false);
  });

  it('leaves available at its optimistic value (not false) on a network error', async () => {
    const store = await loadStoreThrowing(() => new TypeError('Failed to fetch'));
    await store.init();
    expect(store.available).not.toBe(false);
  });

  it('leaves available at its optimistic value (not false) on a 500', async () => {
    const store = await loadStoreThrowing(
      (ApiError) => new ApiError(500, '/curation/ingest/status', null),
    );
    await store.init();
    expect(store.available).not.toBe(false);
  });

  it('is idempotent — a second init() does not re-fetch', async () => {
    vi.resetModules();
    const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
    const impl = vi.fn(async () => ({ total: 0, by_source: [], by_day: [] }));
    vi.doMock('$lib/api', () => ({ ...actual, getIngestStatus: impl }));
    const { ingestAvailability: store } = await import('./ingestAvailability.svelte');

    await store.init();
    await store.init();
    await store.init();

    expect(impl).toHaveBeenCalledTimes(1);
  });
});
