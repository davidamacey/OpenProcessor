/**
 * One combine job's view state (docs/design/
 * w9-p4-w5-w10-ui-plan-2026-10-01.md §4.3 "Job view").
 *
 * The served job is the only source. The backend serves no
 * `poll_after_s` and no `actions` (plan question P4-4), so the controller
 * re-reads every 2 s while `status` is `queued` or `running` and stops at
 * any other status; a global `combine.progress` event carrying this
 * `job_id` only wakes an immediate re-read. Which actions are offered is
 * a reading of the spec's own 409 rules: Cancel while queued/running,
 * Resume while interrupted/cancelled.
 *
 * A terminal-successful job is not enough to run a next step: the target
 * project can still be `building` (the step 409s `project_building`). While
 * the job is completed the controller also reads the target's served
 * `status` (`GET /projects/{target}`, same interval) until it leaves
 * `building`; `canRunNextStep` is true only once it is `active`.
 */
import { configErrorText, getProject } from '$lib/api';
import {
  cancelCombine,
  getCombineJob,
  isCombineNotFound,
  resumeCombine,
  runCombineNextStep,
} from '$lib/api_combine';
import { subscribeGlobalEvents, type ProjectEvent } from '$lib/sse';
import type { CombineJobResponse, CombineNextStep } from '$lib/types_combine';

export const COMBINE_POLL_MS = 2000;

const ACTIVE = new Set(['queued', 'running']);
const RESUMABLE = new Set(['interrupted', 'cancelled']);
const COMPLETED = new Set(['completed', 'completed_with_errors']);

export interface CombineJobDeps {
  getCombineJob: typeof getCombineJob;
  cancelCombine: typeof cancelCombine;
  resumeCombine: typeof resumeCombine;
  runCombineNextStep: typeof runCombineNextStep;
  getProject: typeof getProject;
  /** The target project's served prefix; `null` = not in the list. */
  projectOf: (slug: string) => { prefix: string } | null;
  subscribe: (onEvent: (e: ProjectEvent) => void) => { close(): void };
  pollMs: number;
}

const DEFAULT_DEPS: CombineJobDeps = {
  getCombineJob,
  cancelCombine,
  resumeCombine,
  runCombineNextStep,
  getProject,
  projectOf: () => null,
  subscribe: (onEvent) => subscribeGlobalEvents({ onEvent }),
  pollMs: COMBINE_POLL_MS,
};

export class CombineJob {
  readonly jobId: string;
  job = $state<CombineJobResponse | null>(null);
  loadError = $state<string | null>(null);
  /** The served 404 `combine_not_found` message. */
  notFound = $state<string | null>(null);
  actionError = $state<string | null>(null);
  busy = $state(false);
  /** The served response of the last next step that ran; lives as long as
   *  this controller (one per job id). */
  lastStep = $state<{ action: string; result: unknown } | null>(null);

  /** The target project's served `status`; `null` until first read. */
  targetStatus = $state<string | null>(null);

  #deps: CombineJobDeps;
  #targetTimer: ReturnType<typeof setTimeout> | null = null;
  #targetSeq = 0;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #sub: { close(): void } | null = null;
  #stopped = false;
  #seq = 0;

  constructor(jobId: string, deps: Partial<CombineJobDeps> = {}) {
    this.jobId = jobId;
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  get status(): string | null {
    return this.job?.status ?? null;
  }
  get active(): boolean {
    return this.status != null && ACTIVE.has(this.status);
  }
  get canCancel(): boolean {
    return this.active;
  }
  get canResume(): boolean {
    return this.status != null && RESUMABLE.has(this.status);
  }
  get completed(): boolean {
    return this.status != null && COMPLETED.has(this.status);
  }
  /** Next steps are runnable only once the target is served `active`. */
  get canRunNextStep(): boolean {
    return this.completed && this.targetStatus === 'active';
  }
  get failed(): boolean {
    return this.status === 'failed';
  }

  start(): void {
    this.#stopped = false;
    void this.load();
    this.#sub = this.#deps.subscribe((e) => this.onEvent(e));
  }

  stop(): void {
    this.#stopped = true;
    this.#clearTimer();
    this.#clearTargetTimer();
    this.#targetSeq++;
    this.#sub?.close();
    this.#sub = null;
  }

  onEvent(e: ProjectEvent): void {
    if (e.type !== 'combine.progress' || e.job_id !== this.jobId) return;
    void this.load();
  }

  #clearTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #clearTargetTimer(): void {
    if (this.#targetTimer) clearTimeout(this.#targetTimer);
    this.#targetTimer = null;
  }

  /** Read the target's served status; keep reading while it is `building`
   *  (or the read failed), stop at any other status. */
  async loadTarget(): Promise<void> {
    const target = this.job?.target;
    if (!target || this.#stopped) return;
    this.#clearTargetTimer();
    const seq = ++this.#targetSeq;
    let again: boolean;
    try {
      const p = await this.#deps.getProject(target);
      if (seq !== this.#targetSeq) return;
      this.targetStatus = p.status;
      again = p.status === 'building';
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || seq !== this.#targetSeq) return;
      again = true;
    }
    if (again && !this.#stopped) {
      this.#targetTimer = setTimeout(() => void this.loadTarget(), this.#deps.pollMs);
    }
  }

  #adopt(job: CombineJobResponse): void {
    this.job = job;
    this.#clearTimer();
    if (
      !this.#stopped &&
      COMPLETED.has(job.status) &&
      job.target &&
      this.targetStatus !== 'active' &&
      !this.#targetTimer
    ) {
      void this.loadTarget();
    }
    if (this.#stopped || !ACTIVE.has(job.status)) return;
    this.#timer = setTimeout(() => void this.load(), this.#deps.pollMs);
  }

  async load(): Promise<void> {
    const seq = ++this.#seq;
    try {
      const job = await this.#deps.getCombineJob(this.jobId);
      if (seq !== this.#seq) return;
      this.loadError = null;
      this.notFound = null;
      this.#adopt(job);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || seq !== this.#seq) return;
      if (isCombineNotFound(e)) {
        this.notFound = configErrorText(e);
        return;
      }
      this.loadError = configErrorText(e);
      // A transient failure keeps following a running job.
      if (!this.#stopped && this.job && ACTIVE.has(this.job.status)) {
        this.#clearTimer();
        this.#timer = setTimeout(() => void this.load(), this.#deps.pollMs);
      }
    }
  }

  async #act(fn: () => Promise<CombineJobResponse | null>): Promise<boolean> {
    if (this.busy) return false;
    this.busy = true;
    this.actionError = null;
    try {
      const job = await fn();
      if (job) this.#adopt(job);
      return true;
    } catch (e) {
      this.actionError = configErrorText(e);
      // A `project_building` refusal means the target is not ready yet;
      // keep the button usable and follow the served status again.
      if (this.completed) void this.loadTarget();
      // The refusal usually means the status moved; show the served one.
      void this.load();
      return false;
    } finally {
      this.busy = false;
    }
  }

  cancel(): Promise<boolean> {
    return this.#act(() => this.#deps.cancelCombine(this.jobId));
  }

  resume(): Promise<boolean> {
    return this.#act(() => this.#deps.resumeCombine(this.jobId));
  }

  /** Run one served next step against the target project's prefix. */
  runNextStep(step: CombineNextStep): Promise<boolean> {
    const target = this.job?.target ?? null;
    const project = target ? this.#deps.projectOf(target) : null;
    if (!project) {
      this.actionError = 'The target project is not in the project list yet.';
      return Promise.resolve(false);
    }
    return this.#act(async () => {
      const result = await this.#deps.runCombineNextStep(project, step);
      this.lastStep = { action: step.action, result };
      return null;
    });
  }
}

export function createCombineJob(
  jobId: string,
  deps: Partial<CombineJobDeps> = {},
): CombineJob {
  return new CombineJob(jobId, deps);
}
