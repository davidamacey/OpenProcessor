import { beforeEach, describe, expect, it, vi } from 'vitest';

const getMapMeta = vi.fn();
vi.mock('$lib/api_map', () => ({ getMapMeta: (...a: unknown[]) => getMapMeta(...a) }));

import { mapAvailabilityStore } from './mapAvailability.svelte';

beforeEach(() => {
  getMapMeta.mockReset();
});

describe('mapAvailabilityStore', () => {
  it('shows the link only when the server says the project has a map', async () => {
    getMapMeta.mockResolvedValue({ available: true });
    await mapAvailabilityStore.load('cars');
    expect(mapAvailabilityStore.availableFor('cars')).toBe(true);
    expect(mapAvailabilityStore.availableFor('trucks')).toBe(false);
    expect(mapAvailabilityStore.availableFor(null)).toBe(false);
  });

  it('hides it under the drop policy, where the server serves available: false', async () => {
    getMapMeta.mockResolvedValue({ available: false, location_policy: 'drop' });
    await mapAvailabilityStore.load('cars');
    expect(mapAvailabilityStore.availableFor('cars')).toBe(false);
  });

  it('forgets a stale yes when a later read fails, rather than guess', async () => {
    getMapMeta.mockResolvedValue({ available: true });
    await mapAvailabilityStore.load('cars');
    getMapMeta.mockImplementation(() => Promise.reject(new Error('down')));
    await mapAvailabilityStore.load('cars');
    expect(mapAvailabilityStore.availableFor('cars')).toBe(false);
  });

  it('the newest read wins when two overlap', async () => {
    let first: (v: unknown) => void = () => {};
    getMapMeta
      .mockImplementationOnce(() => new Promise((r) => (first = r)))
      .mockResolvedValueOnce({ available: false });
    const slow = mapAvailabilityStore.load('rv');
    await mapAvailabilityStore.load('rv');
    first({ available: true });
    await slow;
    expect(mapAvailabilityStore.availableFor('rv')).toBe(false);
  });
});
