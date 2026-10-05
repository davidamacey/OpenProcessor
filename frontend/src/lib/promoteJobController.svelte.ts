/**
 * One background promote's view state (OpenProcessor #87).
 *
 * `POST /train/promote/{job_id}` answers 202 with a `PromoteJobStatus`
 * (200 for an identical promote already running; 409 `promote_in_progress`
 * with the active `promote_id` for a different name). The controller starts
 * or attaches to that job and re-reads
 * `GET /train/promote/{job_id}/jobs/{promote_id}` every served
 * `poll_after_s` until it is null (`done` / `failed`). A synchronous
 * failure (gate 422, name 409, 404) is rethrown from `start()` so the
 * modal keeps rendering the served gate detail as before.
 */
import { ApiError, apiErrorText, getPromoteJob, promoteTrainJob } from '$lib/api';
import type { PromoteJobStatus, PromoteRequest, PromoteResponse } from '$lib/types_train';

export const PROMOTE_DEFAULT_POLL_S = 3;
const MAX_POLL_FAILURES = 5;

/** Active phases, in the order the backend walks them. */
export const PROMOTE_PHASES = [
  'queued',
  'exporting',
  'loading',
  'building',
  'warming',
] as const;

export function isPromoteActive(job: PromoteJobStatus | null | undefined): boolean {
  return !!job && (PROMOTE_PHASES as readonly string[]).includes(job.status);
}

/** The active job's id from a 409 `promote_in_progress`, else null. */
export function promoteInProgressId(e: unknown): string | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const d = (e.body as { detail?: unknown } | null)?.detail;
  if (!d || typeof d !== 'object') return null;
  const r = d as Record<string, unknown>;
  return r.code === 'promote_in_progress' && typeof r.promote_id === 'string'
    ? r.promote_id
    : null;
}

/** A failed job's served `error`, with the HTTP status a synchronous call
 *  would have returned; 409 (name taken) points at the Overwrite option. */
export function promoteFailureText(job: PromoteJobStatus): string {
  const msg = job.error?.trim();
  if (!msg) return 'Promote failed.';
  const status = job.error_status;
  const base = status != null ? `${msg} (${status})` : msg;
  return status === 409 ? `${base}. Tick "Overwrite existing" to replace it.` : base;
}

export interface PromoteJobDeps {
  promote: typeof promoteTrainJob;
  getJob: typeof getPromoteJob;
  onDone?: (result: PromoteResponse | null, job: PromoteJobStatus) => void;
}

export class PromoteJobController {
  job = $state<PromoteJobStatus | null>(null);
  /** A transport / lookup error (not a promote `failed`). */
  error = $state<string | null>(null);

  #deps: PromoteJobDeps;
  #runJobId: string | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #abort: AbortController | null = null;
  #starting = false;
  #failures = 0;
  #gen = 0;

  constructor(deps: Partial<PromoteJobDeps> = {}) {
    this.#deps = { promote: promoteTrainJob, getJob: getPromoteJob, ...deps };
  }

  get active(): boolean {
    return isPromoteActive(this.job);
  }
  get failureText(): string | null {
    return this.job?.status === 'failed' ? promoteFailureText(this.job) : null;
  }

  /** Start (or follow) a promote. Rejects with the synchronous `ApiError`
   *  unless it is a 409 `promote_in_progress`, which attaches instead. */
  async start(jobId: string, body: PromoteRequest): Promise<void> {
    if (this.#starting || this.active) return; // double-click
    this.#starting = true;
    this.error = null;
    const gen = ++this.#gen;
    try {
      const served = await this.#deps.promote(jobId, body);
      if (gen === this.#gen) this.attach(jobId, served, true);
    } catch (e) {
      const activeId = promoteInProgressId(e);
      if (activeId === null) throw e;
      if (gen !== this.#gen) return;
      this.#runJobId = jobId;
      await this.#read(jobId, activeId, gen);
    } finally {
      this.#starting = false;
    }
  }

  /** Follow an already-served job (200 duplicate, or `TrainJobStatus.promote`
   *  after a page reload). A terminal job is shown, not polled; `onDone`
   *  fires for a `done` one only when `fireDone` (a fresh start). */
  attach(jobId: string, served: PromoteJobStatus, fireDone = false): void {
    this.#clearTimer();
    this.#runJobId = jobId;
    this.#failures = 0;
    this.error = null;
    this.job = served;
    this.#afterRead(served, ++this.#gen, fireDone);
  }

  stop(): void {
    this.#gen++;
    this.#clearTimer();
    this.#abort?.abort();
    this.#abort = null;
  }

  reset(): void {
    this.stop();
    this.job = null;
    this.error = null;
    this.#failures = 0;
  }

  #clearTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #afterRead(served: PromoteJobStatus, gen: number, fireDone: boolean): void {
    if (isPromoteActive(served)) {
      const secs = served.poll_after_s ?? PROMOTE_DEFAULT_POLL_S;
      this.#timer = setTimeout(
        () => void this.#read(this.#runJobId ?? served.job_id, served.promote_id, gen),
        Math.max(0, secs) * 1000,
      );
    } else if (fireDone && served.status === 'done') {
      this.#deps.onDone?.(served.result ?? null, served);
    }
  }

  async #read(jobId: string, promoteId: string, gen: number): Promise<void> {
    this.#abort?.abort();
    const abort = (this.#abort = new AbortController());
    try {
      const served = await this.#deps.getJob(jobId, promoteId, abort.signal);
      if (gen !== this.#gen) return;
      this.#failures = 0;
      this.job = served;
      this.#afterRead(served, gen, true);
    } catch (e) {
      if (gen !== this.#gen) return;
      this.#failures++;
      const gone = e instanceof ApiError && e.status === 404;
      if (gone || this.#failures >= MAX_POLL_FAILURES) {
        this.error = gone
          ? apiErrorText(e)
          : `Lost track of the promote (${apiErrorText(e)}). It may still be running; reload this page to check.`;
        return;
      }
      this.#timer = setTimeout(
        () => void this.#read(jobId, promoteId, gen),
        PROMOTE_DEFAULT_POLL_S * 1000,
      );
    }
  }
}
