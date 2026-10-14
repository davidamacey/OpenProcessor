/**
 * Background jobs of the slot-gallery controller: region clustering, FP
 * centroid build and sub-cluster refine. The first two are server jobs
 * polled to completion; a poll stops silently once the controller is
 * disposed.
 */

import {
  buildRegionFpCentroids,
  clusterRegions,
  getRegionClusterStatus,
  getRegionFpCentroidStatus,
  refineRegionCluster,
  apiErrorText,
} from '$lib/api';
import { toastStore } from '$stores/toast.svelte';
import type { SlotSpec } from '$lib/annotations/types';

/** The controller state the jobs read and set. */
export interface SlotGalleryJobState {
  readonly slot: SlotSpec;
  readonly disposed: boolean;
  readonly maxRank: number | null;
  readonly selectedCluster: number | null;
  clusterBusy: boolean;
  refineMsg: string | null;
  subTab: string | null;
  loadClusters(): Promise<void>;
  loadFirst(): Promise<void>;
}

export function createSlotGalleryJobs(s: SlotGalleryJobState) {
  const { slot } = s;

  const JOB_POLL_MS = 3000;
  const MAX_CONSECUTIVE_POLL_ERRORS = 3;

  /** Poll `read` until the job stops running. Resolves `null` when disposed.
   *  A failed read is transient (apiFetch already retried a 5xx) and only
   *  ends the poll after several in a row. */
  async function pollUntilDone<T extends { running: boolean }>(
    read: () => Promise<T>,
  ): Promise<T | null> {
    let failures = 0;
    while (true) {
      await new Promise((r) => setTimeout(r, JOB_POLL_MS));
      if (s.disposed) return null;
      try {
        const job = await read();
        if (s.disposed) return null;
        failures = 0;
        if (!job.running) return job;
      } catch (e) {
        if (s.disposed) return null;
        if (++failures >= MAX_CONSECUTIVE_POLL_ERRORS) throw e;
      }
    }
  }

  async function runBuildFpCentroids(): Promise<void> {
    if (s.clusterBusy) return;
    s.clusterBusy = true;
    try {
      await buildRegionFpCentroids();
      toastStore.info('Building FP centroids… sub-typing the false-positive bucket.');
      const job = await pollUntilDone(() => getRegionFpCentroidStatus());
      if (job === null) return;
      if (job.error) {
        toastStore.error(`Build FP centroids failed: ${job.error}`);
      } else if (job.result) {
        toastStore.success(
          `FP centroids built: ${job.result.n_members} members → ${job.result.k} sub-types.`,
        );
        await s.loadClusters();
      }
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
      // One-click pipeline (rebuild FP centroids → auto-pull tight FPs →
      // re-partition good regions) is a multi-minute background job, so we kick
      // it off and poll for completion instead of holding one request open.
      await clusterRegions(s.maxRank ?? undefined);
      toastStore.info(
        `Clustering ${slot.label.plural}… rebuilding FP centroids, pulling FPs, re-bucketing.`,
      );
      const job = await pollUntilDone(() => getRegionClusterStatus());
      if (job === null) return;
      if (job.error) {
        toastStore.error(`Cluster ${slot.label.plural} failed: ${job.error}`);
      } else if (job.result) {
        const r = job.result;
        const moved = r.auto_fp?.n_moved ?? 0;
        if (r.status === 'skipped_repartition_ttl') {
          toastStore.success(
            `Auto-moved ${moved} crop(s) to false positives. Good-${slot.label.singular} re-partition skipped to preserve a refine from the last few minutes — re-run shortly to include it.`,
          );
        } else {
          toastStore.success(
            `Clustered ${r.n_regions ?? 0} ${slot.label.plural} into ${r.n_clusters ?? 0} buckets; auto-moved ${moved} to false positives.`,
          );
        }
        await s.loadClusters();
      }
    } catch (e) {
      toastStore.error(`Cluster ${slot.label.plural} failed: ${apiErrorText(e)}`);
    } finally {
      s.clusterBusy = false;
    }
  }

  async function runRefineCluster(): Promise<void> {
    if (s.selectedCluster == null || s.clusterBusy) return;
    s.clusterBusy = true;
    s.refineMsg = `Refining bucket #${s.selectedCluster}…`;
    try {
      const res = await refineRegionCluster(s.selectedCluster);
      const n = res.n_subclusters ?? 0;
      if (n > 0) {
        s.refineMsg = `Split into ${n} sub-clusters — grouped below.`;
        toastStore.success(`Refine produced ${n} sub-clusters.`);
      } else {
        // Backend skipped it (too small / too large). Surface why.
        const reason = (res as { reason?: string }).reason ?? 'no sub-clusters found';
        s.refineMsg = `Not refined: ${reason}.`;
        toastStore.info(`Bucket not refined: ${reason}.`);
      }
      // Drop back to the "all" tab so the freshly grouped view shows, then
      // reload (server returns regions ordered by sub-cluster).
      s.subTab = null;
      await s.loadFirst();
    } catch (e) {
      s.refineMsg = `Refine failed: ${apiErrorText(e)}`;
      toastStore.error(`Refine failed: ${apiErrorText(e)}`);
    } finally {
      s.clusterBusy = false;
    }
  }

  return { runBuildFpCentroids, runClustering, runRefineCluster };
}
