/**
 * The W4 "not yet deployed" gate: a 404/501 on `/region_profiles` hides
 * every profile surface; a 200 marks it available; any other failure
 * keeps availability unknown; a project switch re-probes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { resetForProjectChange } from '$lib/projectChange';
import { profileListFixture } from '$lib/test/fixtures/regionProfiles';
import { profilesAvailability } from './profilesAvailability.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => profilesAvailability.reset());
afterEach(() => {
  vi.unstubAllGlobals();
  profilesAvailability.reset();
});

describe('profilesAvailability', () => {
  it('probes the scoped list once; 404 -> absent', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ detail: 'Not Found' }, 404));
    vi.stubGlobal('fetch', fetchMock);
    await profilesAvailability.init();
    await profilesAvailability.init();
    expect(profilesAvailability.available).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]![0])).toBe(
      `${API_PREFIX}/region_profiles?include_templates=true`,
    );
  });

  it('501 -> absent; 200 -> available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({}, 501)));
    await profilesAvailability.init();
    expect(profilesAvailability.available).toBe(false);

    profilesAvailability.reset();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(profileListFixture())));
    await profilesAvailability.init();
    expect(profilesAvailability.available).toBe(true);
  });

  it('another failure keeps it unknown with the detail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ detail: 'nope' }, 400)));
    await profilesAvailability.init();
    expect(profilesAvailability.available).toBeNull();
    expect(profilesAvailability.error).toBe('nope');
  });

  it('a project switch resets it', async () => {
    const fetchMock = vi.fn().mockImplementation(async () => json(profileListFixture()));
    vi.stubGlobal('fetch', fetchMock);
    await profilesAvailability.init();
    resetForProjectChange();
    expect(profilesAvailability.available).toBeNull();
    await profilesAvailability.init();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
