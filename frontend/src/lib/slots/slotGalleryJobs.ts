/**
 * Background jobs of the slot-gallery controller: a region clustering run and
 * the FP centroid build. Both are queued server jobs (202 + job id) followed to
 * the end with `followJob`; the follow stops silently once the controller is
 * disposed.
 */

import { startClusterRun } from '$lib/api_clusters';
import { apiErrorText } from '$lib/api_errors';
import { buildRegionFpCentroids } from '$lib/api_regions';
import { followJob } from '$lib/jobs/followJob';
import { toastStore } from '$stores/toast.svelte';
import type { SlotSpec } from '$lib/annotations/types';
import { JOB_DONE_STATUSES } from '$lib/types_jobs';

/** The controller state the jobs read and set. */
export interface SlotGalleryJobState {
  readonly slot: SlotSpec;
  readonly disposed: boolean;
  clusterBusy: boolean;
  loadClusters(): Promise<void>;
}

export function createSlotGalleryJobs(s: SlotGalleryJobState) {
  const { slot } = s;

  /** Follow `jobId`; null once the controller is disposed. */
  async function follow(jobId: string) {
    const ctl = new AbortController();
    const timer = setInterval(() => {
      if (s.disposed) ctl.abort();
    }, 1000);
    try {
      return await followJob(jobId, { signal: ctl.signal });
    } catch (e) {
      if (s.disposed) return null;
      throw e;
    } finally {
      clearInterval(timer);
    }
  }

  async function runBuildFpCentroids(): Promise<void> {
    if (s.clusterBusy) return;
    s.clusterBusy = true;
    try {
      const started = await buildRegionFpCentroids();
      toastStore.info('Building FP centroids… sub-typing the false-positive boxes.');
      const job = await follow(started.job_id);
      if (job === null) return;
      if (!JOB_DONE_STATUSES.includes(job.status)) {
        toastStore.error(
          `Build FP centroids ${job.status.replace('_', ' ')}${job.error_message ? `: ${job.error_message}` : '.'}`,
        );
        return;
      }
      toastStore.success('FP centroids built.');
      await s.loadClusters();
    } catch (e) {
      toastStore.error(`Build FP centroids failed: ${apiErrorText(e)}`);
    } finally {
      s.clusterBusy = false;
    }
  }

  async function runClustering(): Promise<void> {
    if (s.clusterBusy) return;
    s.clusterBusy = true;
    try {
      // A run fits the serving profile over the region boxes in the background
      // (minutes for a large pool): queue it and follow the job.
      const started = await startClusterRun('region_box');
      toastStore.info(`Clustering ${slot.label.plural}…`);
      const job = await follow(started.job_id);
      if (job === null) return;
      if (!JOB_DONE_STATUSES.includes(job.status)) {
        toastStore.error(
          `Cluster ${slot.label.plural} ${job.status.replace('_', ' ')}${job.error_message ? `: ${job.error_message}` : '.'}`,
        );
        return;
      }
      const r = job.result ?? {};
      const note =
        r.activated === true
          ? ''
          : ' The new partition is ready but not serving yet; activate it to see it here.';
      toastStore.success(
        `Clustered ${String(r.n_items ?? 0)} ${slot.label.plural} into ${String(r.n_clusters ?? 0)} clusters.${note}`,
      );
      await s.loadClusters();
    } catch (e) {
      toastStore.error(`Cluster ${slot.label.plural} failed: ${apiErrorText(e)}`);
    } finally {
      s.clusterBusy = false;
    }
  }

  return { runBuildFpCentroids, runClustering };
}
