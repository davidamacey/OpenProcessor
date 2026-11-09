/**
 * The serving cluster partition of a target, a clustering run and the activation
 * of a ready partition (`GET /clusters/partitions`, `POST /clusters/runs`, `PUT
 * /clusters/activation`).
 *
 * Thin: the server fits, scores, decides whether a run is safe to serve (its job
 * result says `activated`) and refuses a stale activation (409
 * `partition_conflict`). This sequences the calls, follows the queued jobs and
 * prints the server's words.
 */
import {
  activateClusterPartition,
  getClusterPartitions,
  startClusterRun,
} from '$lib/api_clusters';
import { apiErrorText } from '$lib/api_errors';
import { followJob } from '$lib/jobs/followJob';
import type { ClusterPartitionView, ClusterTarget } from '$lib/types_clusterList';
import { JOB_DONE_STATUSES, type JobView } from '$lib/types_jobs';

export interface ClusterRunDeps {
  list: typeof getClusterPartitions;
  start: typeof startClusterRun;
  activate: typeof activateClusterPartition;
  follow: (jobId: string, opts?: { onUpdate?: (j: JobView) => void }) => Promise<JobView>;
}

const DEFAULT_DEPS: ClusterRunDeps = {
  list: getClusterPartitions,
  start: startClusterRun,
  activate: activateClusterPartition,
  follow: (jobId, opts) => followJob(jobId, opts),
};

export class ClusterRunController {
  partitions = $state<ClusterPartitionView[]>([]);
  loadError = $state<string | null>(null);
  /** What the server says the running job is doing (`queued`, `running`, ...). */
  jobStatus = $state<string | null>(null);
  busy = $state(false);
  message = $state<string | null>(null);
  actionError = $state<string | null>(null);

  #deps: ClusterRunDeps;
  readonly target: ClusterTarget;

  constructor(target: ClusterTarget = 'item', deps: ClusterRunDeps = DEFAULT_DEPS) {
    this.target = target;
    this.#deps = deps;
  }

  serving = $derived(this.partitions.find((p) => p.serving) ?? null);

  /** A fitted partition the server has not put into service (the newest one). */
  pending = $derived(
    this.partitions.find((p) => p.status === 'ready' && !p.serving) ?? null,
  );

  async load(): Promise<void> {
    try {
      this.partitions = (
        await this.#deps.list({ target: this.target, limit: 10 })
      ).partitions;
      this.loadError = null;
    } catch (e) {
      this.loadError = apiErrorText(e);
    }
  }

  async #run(
    work: () => Promise<{ job_id: string }>,
    what: string,
  ): Promise<JobView | null> {
    this.busy = true;
    this.actionError = null;
    this.message = null;
    try {
      const started = await work();
      const job = await this.#deps.follow(started.job_id, {
        onUpdate: (j) => (this.jobStatus = j.status),
      });
      if (!JOB_DONE_STATUSES.includes(job.status)) {
        this.actionError = `${what} ${job.status.replace('_', ' ')}${job.error_message ? `: ${job.error_message}` : '.'}`;
        return null;
      }
      return job;
    } catch (e) {
      this.actionError = apiErrorText(e);
      return null;
    } finally {
      this.busy = false;
      this.jobStatus = null;
      await this.load();
    }
  }

  async cluster(): Promise<void> {
    const job = await this.#run(
      () => this.#deps.start(this.target),
      'The clustering run',
    );
    if (!job) return;
    const r = job.result ?? {};
    this.message =
      r.activated === true
        ? `Clustered ${String(r.n_items ?? 0)} items into ${String(r.n_clusters ?? 0)} clusters; the new partition is serving.`
        : `Clustered ${String(r.n_items ?? 0)} items into ${String(r.n_clusters ?? 0)} clusters. The new partition is ready but not serving; review it, then activate it.`;
  }

  async activate(partition: ClusterPartitionView): Promise<void> {
    const job = await this.#run(
      () =>
        this.#deps.activate({
          partition_id: partition.id,
          target: this.target,
          level: partition.level,
          expected_serving_partition_id: this.serving?.id ?? null,
        }),
      'The activation',
    );
    if (job) this.message = 'The partition is serving.';
  }
}
