/**
 * The open-vocabulary gate: a 404/501 on `GET /open_vocab` hides every
 * surface, a 200 shows them, any other failure keeps availability unknown
 * with the error and a retry, and a project switch re-probes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, setScopedPrefix } from '$lib/api';
import { resetForProjectChange } from '$lib/projectChange';
import { listFixture } from './fixtures';
import { openVocabAvailability } from './openVocabAvailability.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  setScopedPrefix(`${API_PREFIX}/projects/alpha`);
  openVocabAvailability.reset();
});
afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
  openVocabAvailability.reset();
});

describe('openVocabAvailability', () => {
  it('404 -> absent, probed once, on the scoped list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ detail: 'Not Found' }, 404));
    vi.stubGlobal('fetch', fetchMock);
    await openVocabAvailability.init();
    await openVocabAvailability.init();
    expect(openVocabAvailability.available).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]![0])).toBe(
      `${API_PREFIX}/projects/alpha/open_vocab?include_templates=true`,
    );
  });

  it('501 -> absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({}, 501)));
    await openVocabAvailability.init();
    expect(openVocabAvailability.available).toBe(false);
  });

  it('200 -> available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(listFixture())));
    await openVocabAvailability.init();
    expect(openVocabAvailability.available).toBe(true);
  });

  it('a 500 stays unknown with the error, and retry re-probes', async () => {
    let healthy = false;
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockImplementation(async () =>
          healthy ? json(listFixture()) : json({ detail: 'boom' }, 500),
        ),
    );
    await openVocabAvailability.init();
    expect(openVocabAvailability.available).toBeNull();
    expect(openVocabAvailability.error).toBe('boom');
    healthy = true;
    await openVocabAvailability.retry();
    expect(openVocabAvailability.available).toBe(true);
  });

  it('resets on a project switch', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(listFixture())));
    await openVocabAvailability.init();
    expect(openVocabAvailability.available).toBe(true);
    resetForProjectChange();
    expect(openVocabAvailability.available).toBeNull();
  });
});
