/**
 * The one line a label operation shows while its job runs, from the served job: the state, how
 * many jobs are queued ahead of it (`queue_ahead`, served by `GET /jobs/{id}` while it waits) and
 * the progress counters the writer reports. Nothing here is counted by the client.
 */
import type { JobView } from '$lib/types_jobs';

function whole(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** `verb` is the action in the progressive ("Labeling", "Moving"). Null before a job exists. */
export function opStatusLine(job: JobView | null, verb: string): string | null {
  if (!job) return null;
  switch (job.status) {
    case 'queued': {
      const ahead = whole(job.queue_ahead);
      if (ahead === null) return 'Queued';
      return ahead === 0 ? 'Queued, next up' : `Queued, ${ahead} ahead`;
    }
    case 'running': {
      const done = whole(job.progress?.done);
      const total = whole(job.progress?.total);
      return done !== null && total !== null
        ? `${verb} ${done.toLocaleString()} of ${total.toLocaleString()}`
        : `${verb}\u2026`;
    }
    case 'cancelling':
      return 'Cancelling';
    case 'paused':
      return 'Paused';
    default:
      return null;
  }
}
