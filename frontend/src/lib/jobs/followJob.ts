/**
 * Follow a queued job to its end: poll `GET /jobs/{id}` while the server says
 * it is `queued`, `running` or `cancelling`. The page shows the status the
 * server reports on every tick (`onUpdate`), so a waiting job reads "queued"
 * and a cancelling one "cancelling" without the client deciding either.
 *
 * Resolves with the last `JobView` once the job leaves those statuses
 * (`completed`, `completed_with_errors`, `failed`, `cancelled`, `interrupted`,
 * or `paused`). The caller decides what each means. Rejects when the read
 * itself fails `maxReadFailures` times in a row (one 5xx is retried by
 * `apiFetch` already) or when `signal` aborts.
 */
import { getJob } from '$lib/api_jobs';
import { JOB_ACTIVE_STATUSES, type JobView } from '$lib/types_jobs';

export interface FollowJobOptions {
  /** First wait between reads, then it grows by half up to `maxIntervalMs`. */
  intervalMs?: number;
  maxIntervalMs?: number;
  maxReadFailures?: number;
  signal?: AbortSignal;
  onUpdate?: (job: JobView) => void;
  /** The read, replaceable in tests. */
  read?: (jobId: string, signal?: AbortSignal) => Promise<JobView>;
}

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export async function followJob(
  jobId: string,
  opts: FollowJobOptions = {},
): Promise<JobView> {
  const read = opts.read ?? getJob;
  const maxIntervalMs = opts.maxIntervalMs ?? 2000;
  const maxFailures = opts.maxReadFailures ?? 3;
  let interval = opts.intervalMs ?? 500;
  let failures = 0;
  while (true) {
    try {
      const job = await read(jobId, opts.signal);
      failures = 0;
      opts.onUpdate?.(job);
      if (!JOB_ACTIVE_STATUSES.includes(job.status)) return job;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || ++failures >= maxFailures) throw e;
    }
    await wait(interval, opts.signal);
    interval = Math.min(maxIntervalMs, Math.round(interval * 1.5));
  }
}
