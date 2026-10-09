/**
 * regionStatusesStore — loads GET {API_PREFIX}/regions/statuses once,
 * and degrades to an empty list (never a throw) on failure so
 * slotPanel.ts's served/fallback branches always have a defined input.
 * The store is a module singleton, so every test resets it first.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { regionStatusesStore } from './regionStatuses.svelte';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function resetStore(): void {
  // No public reset — poke the private fields via a fresh module state
  // isn't possible for a singleton, so tests instead assert relative to
  // whatever init() produces each time by stubbing fetch before every call.
  regionStatusesStore.list = [];
  regionStatusesStore.confirmStatus = null;
  regionStatusesStore.rejectStatus = null;
  regionStatusesStore.falsePositiveStatus = null;
  regionStatusesStore.loaded = false;
}

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('regionStatusesStore.init', () => {
  it('populates list + confirm/reject/false_positive status from the response', async () => {
    const payload = {
      statuses: [
        {
          value: 'detected',
          label: 'detected',
          role: 'positive',
          terminal: true,
          human_writable: true,
          clears_box: false,
          wants_reason: false,
        },
      ],
      confirm_status: 'detected',
      reject_status: 'no_region_visible',
      false_positive_status: 'false_positive',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, payload)));

    await regionStatusesStore.init();

    expect(regionStatusesStore.list).toEqual(payload.statuses);
    expect(regionStatusesStore.confirmStatus).toBe('detected');
    expect(regionStatusesStore.rejectStatus).toBe('no_region_visible');
    expect(regionStatusesStore.falsePositiveStatus).toBe('false_positive');
    expect(regionStatusesStore.loaded).toBe(true);
  });

  it('degrades to an empty list (not a throw) when the endpoint fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(500, { detail: 'boom' })),
    );

    await expect(regionStatusesStore.init()).resolves.toBeUndefined();

    expect(regionStatusesStore.list).toEqual([]);
    expect(regionStatusesStore.confirmStatus).toBeNull();
    expect(regionStatusesStore.loaded).toBe(true);
  });

  it('a second call is a no-op once loaded (fetch called exactly once)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, {
        statuses: [],
        confirm_status: 'detected',
        reject_status: 'no_region_visible',
        false_positive_status: 'false_positive',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await regionStatusesStore.init();
    await regionStatusesStore.init();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
