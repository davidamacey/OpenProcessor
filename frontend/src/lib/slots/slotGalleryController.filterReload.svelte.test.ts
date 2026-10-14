/**
 * The /clusters page re-queries the region gallery when a filter changes.
 * Typing in the text filter used to fire a browse request AND a 500-cluster
 * aggregation per keystroke, although the clusters depend only on the rank
 * gate. Discrete filters reload at once; typed ones (text, min score)
 * debounce; the clusters reload only for the rank gate or a cleared bucket.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegions: vi.fn(), getRegionClusters: vi.fn() };
});
import { getRegionClusters, getRegions } from '$lib/api';
import { createSlotGalleryController } from './slotGalleryController.svelte';

let cleanup: (() => void) | undefined;

function wire() {
  const gallery = createSlotGalleryController(widgetTagSlot);
  cleanup = $effect.root(() => {
    $effect(() => {
      gallery.reloadOnFilterChange();
    });
  });
  flushSync();
  return gallery;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(getRegions).mockResolvedValue({ items: [], total: 0 } as never);
  vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
});

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
  vi.useRealTimers();
  vi.mocked(getRegions).mockReset();
  vi.mocked(getRegionClusters).mockReset();
});

describe('region gallery filter reload', () => {
  it('loads the first page and the clusters once on mount', async () => {
    wire();
    await vi.advanceTimersByTimeAsync(1000);
    expect(getRegions).toHaveBeenCalledTimes(1);
    expect(getRegionClusters).toHaveBeenCalledTimes(1);
  });

  it('typing five characters issues one browse request and no cluster request', async () => {
    const gallery = wire();
    await vi.advanceTimersByTimeAsync(1000);
    vi.mocked(getRegions).mockClear();
    vi.mocked(getRegionClusters).mockClear();
    for (const text of ['T', 'TA', 'TAG', 'TAG-', 'TAG-0']) {
      gallery.textQuery = text;
      flushSync();
      await vi.advanceTimersByTimeAsync(50);
    }
    expect(getRegions).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1000);
    expect(getRegions).toHaveBeenCalledTimes(1);
    expect(vi.mocked(getRegions).mock.calls[0]![1]).toMatchObject({ text: 'TAG-0' });
    expect(getRegionClusters).not.toHaveBeenCalled();
  });

  it('control: a discrete filter (status) reloads the browse at once and not the clusters', async () => {
    const gallery = wire();
    await vi.advanceTimersByTimeAsync(1000);
    vi.mocked(getRegions).mockClear();
    vi.mocked(getRegionClusters).mockClear();
    gallery.statusFilter = 'detected';
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(getRegions).toHaveBeenCalledTimes(1);
    expect(getRegionClusters).not.toHaveBeenCalled();
  });

  it('control: the rank gate reloads both', async () => {
    const gallery = wire();
    await vi.advanceTimersByTimeAsync(1000);
    vi.mocked(getRegions).mockClear();
    vi.mocked(getRegionClusters).mockClear();
    gallery.maxRank = 2;
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(getRegions).toHaveBeenCalledTimes(1);
    expect(getRegionClusters).toHaveBeenCalledTimes(1);
  });

  it('control: leaving a selected bucket refreshes the cluster grid', async () => {
    const gallery = wire();
    await vi.advanceTimersByTimeAsync(1000);
    gallery.openCluster(7);
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    vi.mocked(getRegionClusters).mockClear();
    gallery.backToClusters();
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(getRegionClusters).toHaveBeenCalledTimes(1);
  });
});
