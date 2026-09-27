/**
 * m26 (2026-09-24 interactive pass): verified live that the root
 * layout's mount/release/remount sequence during app boot (an SPA-boot
 * quirk — not itself the bug this fixes) called `healthStore.acquire()`
 * / `classesStore.acquire()` repeatedly, each time with the ref count
 * back at 0 — i.e. each acquire's matching release had already fired.
 * `#release()` used to tear down (abort the in-flight fetch, clear the
 * interval) synchronously, so every one of those cycles aborted a
 * request and started a new one — 3 `/health` and 3 `/classes` requests
 * per page load, 2 and 1 of them respectively aborted.
 *
 * Both stores now defer the actual teardown by one macrotask; a
 * re-acquire before that macrotask fires cancels it, coalescing a
 * release-then-immediately-reacquire burst into a no-op instead of an
 * abort+refetch.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { healthStore } from './health.svelte';
import { classesStore } from './classes.svelte';

function ok(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('healthStore.acquire/release coalescing', () => {
  it('does not abort the in-flight poll when release is immediately followed by re-acquire', async () => {
    // poll() now fires two requests per invocation (global health for
    // the chip, scoped health for the region profile) — a fresh
    // Response per call, since a Response's body can only be read once.
    const fetchMock = vi.fn().mockImplementation(() => ok({ status: 'ok' }));
    vi.stubGlobal('fetch', fetchMock);

    const release1 = healthStore.acquire();
    release1();
    const release2 = healthStore.acquire();

    // Let any scheduled (but not yet fired) teardown timer run.
    await new Promise((r) => setTimeout(r, 10));

    // Only the original poll() from the first acquire() should have
    // fired — the release+reacquire burst must not abort it and start
    // a second one.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    release2();
  });

  it('still tears down for a real, lasting release (no re-acquire follows)', async () => {
    const fetchMock = vi.fn().mockImplementation(() => ok({ status: 'ok' }));
    vi.stubGlobal('fetch', fetchMock);

    const release = healthStore.acquire();
    await new Promise((r) => setTimeout(r, 0));
    release();
    await new Promise((r) => setTimeout(r, 10));

    fetchMock.mockClear();
    // Advance past the 15s poll interval — nothing should fire, since
    // the store actually stopped.
    await new Promise((r) => setTimeout(r, 20));
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('classesStore.acquire/release coalescing', () => {
  it('does not abort the in-flight refresh when release is immediately followed by re-acquire', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(ok({ classes: [], thresholds: {}, reserved_hotkeys: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const release1 = classesStore.acquire();
    release1();
    const release2 = classesStore.acquire();

    await new Promise((r) => setTimeout(r, 10));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    release2();
  });
});
