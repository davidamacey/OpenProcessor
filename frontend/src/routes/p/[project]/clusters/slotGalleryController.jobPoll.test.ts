/**
 * The region-gallery "Cluster" and "Build FP centroids" actions kick off a
 * multi-minute background job and poll its status. The poll must stop when
 * the controller is disposed (leaving /clusters), and one transient status
 * failure must not end it with a "failed" toast while the job still runs.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    clusterRegions: vi.fn(),
    getRegionClusterStatus: vi.fn(),
    buildRegionFpCentroids: vi.fn(),
    getRegionFpCentroidStatus: vi.fn(),
    getRegionClusters: vi.fn(),
  };
});
import {
  buildRegionFpCentroids,
  clusterRegions,
  getRegionClusterStatus,
  getRegionClusters,
  getRegionFpCentroidStatus,
} from '$lib/api';
import { toastStore } from '$stores/toast.svelte';
import { createSlotGalleryController } from './slotGalleryController.svelte';

const running = {
  running: true,
  started_at: null,
  finished_at: null,
  result: null,
  error: null,
};
const done = {
  running: false,
  started_at: null,
  finished_at: null,
  result: {
    status: 'ok',
    n_regions: 4,
    n_clusters: 2,
    auto_fp: { status: 'ok', n_moved: 1 },
  },
  error: null,
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(clusterRegions).mockResolvedValue(running as never);
  vi.mocked(buildRegionFpCentroids).mockResolvedValue(running as never);
  vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [] } as never);
});

afterEach(() => {
  vi.useRealTimers();
  vi.mocked(getRegionClusterStatus).mockReset();
  vi.mocked(getRegionFpCentroidStatus).mockReset();
  vi.restoreAllMocks();
});

describe('region-gallery clustering poll', () => {
  it('stops polling once the controller is disposed', async () => {
    vi.mocked(getRegionClusterStatus).mockResolvedValue(running as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(10_000);
    const before = vi.mocked(getRegionClusterStatus).mock.calls.length;
    expect(before).toBeGreaterThanOrEqual(2);
    gallery.dispose();
    await vi.advanceTimersByTimeAsync(30_000);
    await run;
    expect(vi.mocked(getRegionClusterStatus).mock.calls.length).toBeLessThanOrEqual(
      before + 1,
    );
    expect(gallery.clusterBusy).toBe(false);
  });

  it('stops the FP-centroid poll once disposed', async () => {
    vi.mocked(getRegionFpCentroidStatus).mockResolvedValue(running as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runBuildFpCentroids();
    await vi.advanceTimersByTimeAsync(7000);
    gallery.dispose();
    const before = vi.mocked(getRegionFpCentroidStatus).mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);
    await run;
    expect(vi.mocked(getRegionFpCentroidStatus).mock.calls.length).toBeLessThanOrEqual(
      before + 1,
    );
  });

  it('a single failed status read does not end the poll; the job is still followed to completion', async () => {
    const errors = vi.spyOn(toastStore, 'error');
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(getRegionClusterStatus)
      .mockResolvedValueOnce(running as never)
      .mockRejectedValueOnce(new Error('503 transient'))
      .mockResolvedValue(done as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(20_000);
    await run;
    expect(errors).not.toHaveBeenCalled();
    expect(success).toHaveBeenCalled();
  });

  it('control: a completed job toasts once and reloads the clusters', async () => {
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(getRegionClusterStatus).mockResolvedValue(done as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(4000);
    await run;
    expect(success).toHaveBeenCalledTimes(1);
    expect(getRegionClusters).toHaveBeenCalled();
  });

  it('a status read that keeps failing is reported as a failure', async () => {
    const errors = vi.spyOn(toastStore, 'error');
    vi.mocked(getRegionClusterStatus).mockRejectedValue(new Error('503 down'));
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(30_000);
    await run;
    expect(errors).toHaveBeenCalledTimes(1);
  });
});
