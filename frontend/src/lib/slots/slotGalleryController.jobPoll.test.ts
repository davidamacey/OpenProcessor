/**
 * The region-gallery "Cluster" and "Build FP centroids" actions queue a
 * background job (202 + job id) and follow `GET /jobs/{id}`. The follow must
 * stop when the controller is disposed (leaving /clusters), and one transient
 * read failure must not end it with a "failed" toast while the job still runs.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { JobView } from '$lib/types_jobs';

vi.mock('$lib/api_regions', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_regions')>('$lib/api_regions');
  return { ...actual, buildRegionFpCentroids: vi.fn(), getRegionClusters: vi.fn() };
});
vi.mock('$lib/api_clusters', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_clusters')>('$lib/api_clusters');
  return { ...actual, startClusterRun: vi.fn() };
});
vi.mock('$lib/api_jobs', () => ({ getJob: vi.fn() }));
import { startClusterRun } from '$lib/api_clusters';
import { getJob } from '$lib/api_jobs';
import { buildRegionFpCentroids, getRegionClusters } from '$lib/api_regions';
import { toastStore } from '$stores/toast.svelte';
import { createSlotGalleryController } from './slotGalleryController.svelte';

function job(status: JobView['status'], result: JobView['result'] = null): JobView {
  return {
    id: 'j1',
    kind: 'cluster_run',
    status,
    result,
    error_message: null,
  } as JobView;
}
const running = job('running');
const done = job('completed', {
  status: 'success',
  n_items: 4,
  n_clusters: 2,
  activated: true,
});

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(startClusterRun).mockResolvedValue({ job_id: 'j1' });
  vi.mocked(buildRegionFpCentroids).mockResolvedValue({
    job_id: 'j1',
    kind: 'region_fp_centroids',
  });
  vi.mocked(getRegionClusters).mockResolvedValue({ clusters: [], count: 0 });
});

afterEach(() => {
  vi.useRealTimers();
  vi.mocked(getJob).mockReset();
  vi.restoreAllMocks();
});

describe('region-gallery clustering job', () => {
  it('queues a region_box run and follows the job it answered with', async () => {
    vi.mocked(getJob).mockResolvedValue(done);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(2000);
    await run;
    expect(startClusterRun).toHaveBeenCalledWith('region_box');
    expect(getJob).toHaveBeenCalledWith('j1', expect.anything());
  });

  it('stops following once the controller is disposed', async () => {
    vi.mocked(getJob).mockResolvedValue(running);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(10_000);
    const before = vi.mocked(getJob).mock.calls.length;
    expect(before).toBeGreaterThanOrEqual(2);
    gallery.dispose();
    await vi.advanceTimersByTimeAsync(30_000);
    await run;
    expect(vi.mocked(getJob).mock.calls.length).toBeLessThanOrEqual(before + 1);
    expect(gallery.clusterBusy).toBe(false);
  });

  it('stops the FP-centroid follow once disposed', async () => {
    vi.mocked(getJob).mockResolvedValue(running);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runBuildFpCentroids();
    await vi.advanceTimersByTimeAsync(7000);
    gallery.dispose();
    const before = vi.mocked(getJob).mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);
    await run;
    expect(vi.mocked(getJob).mock.calls.length).toBeLessThanOrEqual(before + 1);
  });

  it('a single failed read does not end the follow; the job is still followed to completion', async () => {
    const errors = vi.spyOn(toastStore, 'error');
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(getJob)
      .mockResolvedValueOnce(running)
      .mockRejectedValueOnce(new Error('503 transient'))
      .mockResolvedValue(done);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(20_000);
    await run;
    expect(errors).not.toHaveBeenCalled();
    expect(success).toHaveBeenCalled();
  });

  it('control: a completed job toasts once, says the partition is serving, and reloads the clusters', async () => {
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(getJob).mockResolvedValue(done);
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(4000);
    await run;
    expect(success).toHaveBeenCalledTimes(1);
    expect(String(success.mock.calls[0]![0])).not.toContain('not serving');
    expect(getRegionClusters).toHaveBeenCalled();
  });

  it('says a finished run that was not activated is not serving yet', async () => {
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(getJob).mockResolvedValue(
      job('completed', { n_items: 4, n_clusters: 2, activated: false }),
    );
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(4000);
    await run;
    expect(String(success.mock.calls[0]![0])).toContain('not serving yet');
  });

  it('a failed job is reported as a failure, not a success', async () => {
    const errors = vi.spyOn(toastStore, 'error');
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(getJob).mockResolvedValue(job('failed'));
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(4000);
    await run;
    expect(errors).toHaveBeenCalledTimes(1);
    expect(success).not.toHaveBeenCalled();
  });

  it('a read that keeps failing is reported as a failure', async () => {
    const errors = vi.spyOn(toastStore, 'error');
    vi.mocked(getJob).mockRejectedValue(new Error('503 down'));
    const gallery = createSlotGalleryController(widgetTagSlot);
    const run = gallery.runClustering();
    await vi.advanceTimersByTimeAsync(30_000);
    await run;
    expect(errors).toHaveBeenCalledTimes(1);
  });
});
