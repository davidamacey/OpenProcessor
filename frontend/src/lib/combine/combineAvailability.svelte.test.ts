/**
 * The P4 "not yet deployed" gate: a 404 with the structured
 * `combine_not_found` detail means the router is mounted (available); a
 * plain 404 / 501 means it is not (absent); anything else leaves
 * availability unknown with the error. It is global: a project switch
 * does not re-probe.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { resetForProjectChange } from '$lib/projectChange';
import { combineAvailability, COMBINE_PROBE_JOB_ID } from './combineAvailability.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => combineAvailability.reset());
afterEach(() => {
  vi.unstubAllGlobals();
  combineAvailability.reset();
});

describe('combineAvailability', () => {
  it('a structured combine_not_found 404 -> available, probed once with the sentinel id', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        json({ detail: { error: 'combine_not_found', message: 'no combine job' } }, 404),
      );
    vi.stubGlobal('fetch', fetchMock);
    await combineAvailability.init();
    await combineAvailability.init();
    expect(combineAvailability.available).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]![0])).toBe(
      `${API_PREFIX}/projects/combine/${COMBINE_PROBE_JOB_ID}`,
    );
  });

  it('a plain 404 -> absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ detail: 'Not Found' }, 404)));
    await combineAvailability.init();
    expect(combineAvailability.available).toBe(false);
  });

  it('a 404 whose detail names another error is absent too', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          json({ detail: { error: 'project_not_found', message: 'x' } }, 404),
        ),
    );
    await combineAvailability.init();
    expect(combineAvailability.available).toBe(false);
  });

  it('501 -> absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({}, 501)));
    await combineAvailability.init();
    expect(combineAvailability.available).toBe(false);
  });

  it('a 400 keeps availability unknown and records the served detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(json({ detail: 'bad request' }, 400)),
    );
    await combineAvailability.init();
    expect(combineAvailability.available).toBeNull();
    expect(combineAvailability.error).toBe('bad request');
  });

  it('a 200 for the sentinel id is available too (the router answered)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(json({ job_id: 'x', status: 'queued' })),
    );
    await combineAvailability.init();
    expect(combineAvailability.available).toBe(true);
  });

  it('is global: a project switch does not probe again', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        json({ detail: { error: 'combine_not_found', message: '' } }, 404),
      );
    vi.stubGlobal('fetch', fetchMock);
    await combineAvailability.init();
    const before = fetchMock.mock.calls.length;
    resetForProjectChange();
    await combineAvailability.init();
    expect(combineAvailability.available).toBe(true);
    expect(fetchMock.mock.calls.length).toBe(before);
  });
});
