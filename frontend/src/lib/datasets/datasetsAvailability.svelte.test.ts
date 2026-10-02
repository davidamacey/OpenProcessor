/**
 * The W10 "not yet deployed" gate: a 404/501 on `/datasets/formats`
 * hides every W10 surface; a 200 caches the served vocabulary; any other
 * failure keeps availability unknown and records the error.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { resetForProjectChange } from '$lib/projectChange';
import { formatsFixture } from '$lib/test/fixtures/datasetImport';
import { datasetsAvailability } from './datasetsAvailability.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => datasetsAvailability.reset());
afterEach(() => {
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

describe('datasetsAvailability', () => {
  it('404 -> absent, and the probe is not repeated', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ detail: 'Not Found' }, 404));
    vi.stubGlobal('fetch', fetchMock);
    await datasetsAvailability.init();
    await datasetsAvailability.init();
    expect(datasetsAvailability.available).toBe(false);
    expect(datasetsAvailability.formats).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]![0])).toBe(`${API_PREFIX}/datasets/formats`);
  });

  it('501 -> absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({}, 501)));
    await datasetsAvailability.init();
    expect(datasetsAvailability.available).toBe(false);
  });

  it('200 -> available, the served vocabulary cached and its labels used', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(formatsFixture())));
    await datasetsAvailability.init();
    expect(datasetsAvailability.available).toBe(true);
    expect(datasetsAvailability.formats?.mapping_actions[0]).toEqual({
      value: 'map',
      label: 'Map to class',
      description: '',
    });
    expect(datasetsAvailability.statusLabel('paused_backpressure')).toBe(
      'Waiting for the region worker',
    );
    expect(datasetsAvailability.statusLabel('unknown_status')).toBe('unknown_status');
  });

  it('another failure keeps availability unknown and shows the served detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(json({ detail: 'bad request' }, 400)),
    );
    await datasetsAvailability.init();
    expect(datasetsAvailability.available).toBeNull();
    expect(datasetsAvailability.error).toBe('bad request');
  });

  it('a project switch resets it, so the next project is probed again', async () => {
    // Other stores' reset hooks refetch too; count only the probe.
    const fetchMock = vi.fn().mockImplementation(async () => json(formatsFixture()));
    vi.stubGlobal('fetch', fetchMock);
    const probes = () =>
      fetchMock.mock.calls.filter(([u]) => String(u).endsWith('/datasets/formats'))
        .length;
    await datasetsAvailability.init();
    expect(datasetsAvailability.available).toBe(true);
    resetForProjectChange();
    expect(datasetsAvailability.available).toBeNull();
    expect(datasetsAvailability.formats).toBeNull();
    await datasetsAvailability.init();
    expect(probes()).toBe(2);
  });
});
