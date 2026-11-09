// Job runtime: wire types for `JobView` / `JobPage` of the contract
// (`contracts/openapi/curation.json`), pinned key for key by
// `contract/jobsContract.test.ts`. Every route that queues work answers 202
// with a job id; `GET /jobs/{job_id}` is where its state is read.

export const JOB_STATUSES = [
  'queued',
  'running',
  'paused',
  'cancelling',
  'completed',
  'completed_with_errors',
  'failed',
  'cancelled',
  'interrupted',
] as const;
export type JobStatus = (typeof JOB_STATUSES)[number];

/** Statuses a job still moves out of by itself: keep polling in these. */
export const JOB_ACTIVE_STATUSES: readonly JobStatus[] = [
  'queued',
  'running',
  'cancelling',
];

/** Statuses that end in a result: the job finished (with or without errors). */
export const JOB_DONE_STATUSES: readonly JobStatus[] = [
  'completed',
  'completed_with_errors',
];

/** `JobView`: one job of the project. `progress` and `result` are the job
 *  kind's own objects; read the keys the caller's kind documents. */
export interface JobView {
  id: string;
  kind: string;
  status: JobStatus;
  project: string | null;
  scope: string;
  queue: string;
  params?: Record<string, unknown> | null;
  progress: Record<string, unknown>;
  progress_updated_at: string | null;
  result: Record<string, unknown> | null;
  error_code: string | null;
  error_message: string | null;
  pause_reason: string | null;
  pause_requested: boolean;
  cancel_requested: boolean;
  attempt: number;
  max_attempts: number;
  retry_count: number;
  requested_by: string;
  created_at: string;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string;
}

/** `JobPage`. */
export interface JobPage {
  items: JobView[];
  next_cursor?: string | null;
}

/** The 202 body of a route that starts a job (`job_id` is the `JobView.id`). */
export interface JobStarted {
  job_id: string;
}
