/**
 * What a poller does with the `status` of a "latest job of a kind" snapshot
 * (`/scores/status`, `/viz/projection/status`, `/probe/status`,
 * `/select/status`): the server serves the job runtime's statuses plus
 * `idle` (the project never ran one). `queued`, `paused` and `cancelling`
 * are still in flight, so a poller that only knew `running` would stop on a
 * queued job and report it as done.
 */
import { JOB_STATUSES, type JobStatus } from '$lib/types_jobs';

export type LatestJobStatus = 'idle' | JobStatus;

export const LATEST_JOB_STATUSES: readonly LatestJobStatus[] = ['idle', ...JOB_STATUSES];

export type JobPhase = 'idle' | 'active' | 'completed' | 'failed' | 'cancelled';

export function isLatestJobStatus(v: unknown): v is LatestJobStatus {
  return typeof v === 'string' && (LATEST_JOB_STATUSES as readonly string[]).includes(v);
}

/** `failed`, `interrupted` and any status this build does not know end the poll as failures. */
export function jobPhase(status: string): JobPhase {
  switch (status) {
    case 'idle':
      return 'idle';
    case 'queued':
    case 'running':
    case 'paused':
    case 'cancelling':
      return 'active';
    case 'completed':
    case 'completed_with_errors':
      return 'completed';
    case 'cancelled':
      return 'cancelled';
    default:
      return 'failed';
  }
}
