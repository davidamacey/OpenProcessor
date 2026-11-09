/*
 * Controller for the embedding-projection rebuild job: request, poll,
 * cancel, and adopt a rebuild already running (another tab, or before a
 * reload). Call during component initialisation (it registers a $effect).
 * `onRebuilt` runs after a poll sees the job complete.
 */
import { apiErrorText } from '$lib/api_errors';
import {
  cancelVizProjection,
  getVizProjectionStatus,
  rebuildVizProjection,
  type VizProjectionJob,
} from '$lib/api_viz';
import { classifyRebuildPoll } from '$lib/embeddingPlot';
import { jobPhase } from '$lib/jobs/jobPhase';
import { toastStore } from '$stores/toast.svelte';

const JOB_POLL_MS = 3000;

export function createVizRebuildJob(onRebuilt: () => Promise<void>) {
  let job = $state<VizProjectionJob | null>(null);
  let rebuilding = $state<boolean>(false);
  let jobPoll: ReturnType<typeof setInterval> | null = null;
  // `destroyed` stops an adopt read that resolves after unmount from
  // starting an orphan poll; `polling` stops a slow read from overlapping
  // the next tick.
  let destroyed = false;
  let polling = false;

  function stopJobPoll(): void {
    if (jobPoll) clearInterval(jobPoll);
    jobPoll = null;
  }

  async function pollJob(): Promise<void> {
    if (polling) return;
    polling = true;
    let st: VizProjectionJob;
    try {
      st = await getVizProjectionStatus();
    } catch {
      return; // transient — keep polling
    } finally {
      polling = false;
    }
    if (destroyed || jobPoll === null) return;
    const outcome = classifyRebuildPoll(st.status, job !== null);
    if (outcome === 'running') {
      job = st;
      return;
    }
    job = null;
    stopJobPoll();
    if (outcome === 'completed') {
      toastStore.success(`Embedding projection rebuilt (${st.n_written} points).`);
      await onRebuilt();
    } else if (outcome === 'failed') {
      toastStore.error(`Projection rebuild failed: ${st.error ?? 'unknown error'}`);
    }
  }

  function startJobPoll(): void {
    stopJobPoll();
    if (destroyed) return;
    jobPoll = setInterval(() => void pollJob(), JOB_POLL_MS);
  }

  $effect(() => {
    destroyed = false;
    void getVizProjectionStatus()
      .then((st) => {
        if (jobPhase(st.status) === 'active') {
          job = st;
          startJobPoll();
        }
      })
      .catch(() => {});
    return () => {
      destroyed = true;
      stopJobPoll();
    };
  });

  async function trigger(): Promise<void> {
    if (rebuilding || job) return;
    rebuilding = true;
    try {
      job = await rebuildVizProjection();
      startJobPoll();
    } catch (e) {
      toastStore.error(`Rebuild failed: ${apiErrorText(e)}`);
    } finally {
      rebuilding = false;
    }
  }

  async function cancel(): Promise<void> {
    try {
      await cancelVizProjection();
      toastStore.info('Projection rebuild cancelled.');
    } catch (e) {
      toastStore.error(`Cancel failed: ${apiErrorText(e)}`);
    } finally {
      job = null;
      stopJobPoll();
    }
  }

  return {
    get job() {
      return job;
    },
    get rebuilding() {
      return rebuilding;
    },
    trigger,
    cancel,
  };
}
