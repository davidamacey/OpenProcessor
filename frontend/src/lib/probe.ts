/**
 * Pure helpers for the `/train` "Run probe predictions" control (#36 item
 * 8). Split out of `ProbeControl.svelte` so the poll-status
 * classification and gating logic are unit-testable without mounting a
 * component, same convention as `scores.ts`'s `classifyScoresPoll` and
 * `embeddingPlot.ts`'s `classifyRebuildPoll`.
 */

import type { TrainJobStatus } from './types_train';

export type ProbePollOutcome = 'running' | 'completed' | 'failed' | 'cancelled';

/** What a poll of `GET {API_PREFIX}/probe/status` should do next, given
 *  only the job's own `status` (`running`/`completed`/`cancelled`/
 *  `failed`/`idle` — `probe_job.py`'s `_JobState`). `idle` (never
 *  started, or the client's own job resolved without an in-between poll
 *  catching it) is treated the same as `completed`: stop polling, no
 *  error to report. */
export function classifyProbePoll(status: string): ProbePollOutcome {
  if (status === 'running') return 'running';
  if (status === 'failed') return 'failed';
  if (status === 'cancelled') return 'cancelled';
  return 'completed';
}

/** Whether the control offers "Run probe predictions" for this train run
 *  at all — the backend 409s a job that isn't `finished` or has no
 *  exported checkpoint, but the control shouldn't even render the button
 *  for a run that can't possibly qualify (still running, failed, no
 *  checkpoint recorded), so an operator isn't invited to click something
 *  that always 409s. */
export function canRunProbe(status: TrainJobStatus): boolean {
  return status.state === 'finished' && !!status.checkpoint_sha256;
}
