/**
 * The W3 "not yet deployed" gate: a 404/501 on `/prompt_packs` hides every
 * pack surface; a 200 marks it available; any other failure keeps
 * availability unknown and records the error.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { resetForProjectChange } from '$lib/projectChange';
import { listFixture } from '$lib/test/fixtures/promptPacks';
import { packsAvailability } from './packsAvailability.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => packsAvailability.reset());
afterEach(() => {
  vi.unstubAllGlobals();
  packsAvailability.reset();
});

describe('packsAvailability', () => {
  it('404 -> absent, and the probe is not repeated', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ detail: 'Not Found' }, 404));
    vi.stubGlobal('fetch', fetchMock);
    await packsAvailability.init();
    await packsAvailability.init();
    expect(packsAvailability.available).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]![0])).toBe(`${API_PREFIX}/prompt_packs`);
  });

  it('501 -> absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({}, 501)));
    await packsAvailability.init();
    expect(packsAvailability.available).toBe(false);
  });

  it('200 -> available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(listFixture())));
    await packsAvailability.init();
    expect(packsAvailability.available).toBe(true);
    expect(packsAvailability.error).toBeNull();
  });

  it('another failure keeps availability unknown, shows the detail, and Retry re-probes', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ detail: 'bad request' }, 400));
    vi.stubGlobal('fetch', fetchMock);
    await packsAvailability.init();
    expect(packsAvailability.available).toBeNull();
    expect(packsAvailability.error).toBe('bad request');
    fetchMock.mockResolvedValue(json(listFixture()));
    await packsAvailability.retry();
    expect(packsAvailability.available).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('a project switch resets it, so the next project is probed again', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => json(listFixture()));
    vi.stubGlobal('fetch', fetchMock);
    const probes = () =>
      fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/prompt_packs')).length;
    await packsAvailability.init();
    expect(packsAvailability.available).toBe(true);
    resetForProjectChange();
    expect(packsAvailability.available).toBeNull();
    await packsAvailability.init();
    expect(probes()).toBe(2);
  });
});
