/**
 * The W9 "not yet deployed" gate: a 404/501 on the GLOBAL `/vlm/endpoints`
 * hides every registry surface (and that one request is the only one that
 * fires); a 200 marks it available and keeps the served status labels for
 * `/models`; any other failure keeps availability unknown; unlike the
 * project-scoped gates a project switch does NOT re-probe (the registry is
 * deployment-wide).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { resetForProjectChange } from '$lib/projectChange';
import { listFixture } from '$lib/test/fixtures/vlm';
import { vlmAvailability, vlmStatusLabels } from './vlmAvailability.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  vlmAvailability.reset();
  vlmStatusLabels.status = null;
});
afterEach(() => {
  vi.unstubAllGlobals();
  vlmAvailability.reset();
});

describe('vlmAvailability', () => {
  it('probes the global list once; 404 -> absent', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ detail: 'Not Found' }, 404));
    vi.stubGlobal('fetch', fetchMock);
    await vlmAvailability.init();
    await vlmAvailability.init();
    expect(vlmAvailability.available).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]![0])).toBe(`${API_PREFIX}/vlm/endpoints`);
  });

  it('501 -> absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({}, 501)));
    await vlmAvailability.init();
    expect(vlmAvailability.available).toBe(false);
  });

  it('200 -> available, keeping the served status labels', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(listFixture())));
    await vlmAvailability.init();
    expect(vlmAvailability.available).toBe(true);
    expect(vlmStatusLabels.status?.unprobed).toBe('Not probed yet');
  });

  it('a 500 keeps it unknown with the detail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ detail: 'boom' }, 400)));
    await vlmAvailability.init();
    expect(vlmAvailability.available).toBeNull();
    expect(vlmAvailability.error).toBe('boom');
  });

  it('one probe across a project switch', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => json(listFixture()));
    vi.stubGlobal('fetch', fetchMock);
    await vlmAvailability.init();
    resetForProjectChange();
    expect(vlmAvailability.available).toBe(true);
    await vlmAvailability.init();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
