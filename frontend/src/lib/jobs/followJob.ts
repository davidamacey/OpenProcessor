/**
 * Follow a queued job to its end: poll `GET /jobs/{id}` while the server says
 * it is `queued`, `running` or `cancelling`. The page shows the status the
 * server reports on every tick (`onUpdate`), so a waiting job reads "queued"
 * and a cancelling one "cancelling" without the client deciding either.
 *
 * `pollWhile` is the same loop over any read: a re-index run follows its own status with it.
 *
 * Resolves with the last `JobView` once the job leaves those statuses
 * (`completed`, `completed_with_errors`, `failed`, `cancelled`, `interrupted`,
 * or `paused`). The caller decides what each means. Rejects when the read
 * itself fails `maxReadFailures` times in a row (one 5xx is retried by
 * `apiFetch` already) or when `signal` aborts.
 */
import { getJob } from '$lib/api_jobs';
import { JOB_ACTIVE_STATUSES, type JobView } from '$lib/types_jobs';

export interface PollOptions<T> {
  /** First wait between reads, then it grows by half up to `maxIntervalMs`. */
  intervalMs?: number;
  maxIntervalMs?: number;
  maxReadFailures?: number;
  signal?: AbortSignal;
  onUpdate?: (value: T) => void;
}

export interface FollowJobOptions extends PollOptions<JobView> {
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

/** Reads until `isActive` is false for what was read; rejects on abort or repeated read failures. */
export async function pollWhile<T>(
  read: (signal?: AbortSignal) => Promise<T>,
  isActive: (value: T) => boolean,
  opts: PollOptions<T> = {},
): Promise<T> {
  const maxIntervalMs = opts.maxIntervalMs ?? 2000;
  const maxFailures = opts.maxReadFailures ?? 3;
  let interval = opts.intervalMs ?? 500;
  let failures = 0;
  while (true) {
    try {
      const value = await read(opts.signal);
      failures = 0;
      opts.onUpdate?.(value);
      if (!isActive(value)) return value;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || ++failures >= maxFailures) throw e;
    }
    await wait(interval, opts.signal);
    interval = Math.min(maxIntervalMs, Math.round(interval * 1.5));
  }
}

export function followJob(jobId: string, opts: FollowJobOptions = {}): Promise<JobView> {
  const read = opts.read ?? getJob;
  return pollWhile(
    (signal) => read(jobId, signal),
    (job) => JOB_ACTIVE_STATUSES.includes(job.status),
    opts,
  );
}
