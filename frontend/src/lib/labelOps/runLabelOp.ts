/**
 * Run a recorded label operation to its end: the relabel / move routes answer
 * 202 (`op_id`, `job_id`), the write happens in a queued job, and the counts
 * (`updated`, `conflicts`) are on that job's `result`. This follows the job and
 * hands the caller those server counts; nothing is counted here.
 */
import { previewLabelOpUndo, queueLabelOpUndo } from '$lib/api_labelOps';
import { followJob, type FollowJobOptions } from '$lib/jobs/followJob';
import type { LabelOpAccepted, LabelOpUndoResponse } from '$lib/types_labelOps';
import { JOB_DONE_STATUSES, type JobStatus } from '$lib/types_jobs';

export interface LabelOpOutcome {
  /** The recorded operation (undo it by id); null when nothing was queued. */
  opId: string | null;
  updated: number;
  conflicts: number;
  /** False when the job stopped (paused) before it reached every item. */
  complete: boolean;
  status: JobStatus;
}

function count(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** An empty selection is answered with no job; there is nothing to follow. */
export async function runLabelOp(
  accepted: LabelOpAccepted,
  opts: FollowJobOptions = {},
): Promise<LabelOpOutcome> {
  if (accepted.job_id == null) {
    return { opId: null, updated: 0, conflicts: 0, complete: true, status: 'completed' };
  }
  const job = await followJob(accepted.job_id, opts);
  if (!JOB_DONE_STATUSES.includes(job.status) && job.status !== 'paused') {
    throw new Error(
      job.error_message ?? `The label operation ${job.status.replace('_', ' ')}.`,
    );
  }
  const result = job.result ?? {};
  return {
    opId: accepted.op_id ?? (typeof result.op_id === 'string' ? result.op_id : null),
    updated: count(result.updated),
    conflicts: count(result.conflicts),
    complete: job.status !== 'paused' && result.complete !== false,
    status: job.status,
  };
}

/**
 * The undo of a recorded operation, as the label-ops dialog uses it: a dry run
 * answers at once; a real undo is queued, followed to its end, and answers with
 * the job's result (the same `LabelOpUndoResponse` a dry run has).
 */
export async function undoLabelOpToEnd(
  opId: string,
  dryRun: boolean,
  signal?: AbortSignal,
): Promise<LabelOpUndoResponse> {
  if (dryRun) return previewLabelOpUndo(opId, signal);
  const accepted = await queueLabelOpUndo(opId, signal);
  const job = await followJob(accepted.job_id!, { signal });
  if (!JOB_DONE_STATUSES.includes(job.status) || job.result?.cancelled === true) {
    throw new Error(job.error_message ?? 'The undo did not finish.');
  }
  return job.result as unknown as LabelOpUndoResponse;
}
