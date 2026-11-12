/**
 * The Embedding spaces page: the project's spaces (all targets), the encoders the deployment can
 * serve, and the writes that register, restart or backfill, promote, retire and cancel. A started
 * build is followed through the served re-index run until its job leaves the active statuses: the
 * ones started here (the 202 names its `run_id`) and, because the served space list names each
 * space's newest run and its status, the ones started anywhere else. Every refusal is the server's:
 * the controller keeps its code and message and decides nothing (a promote that lacks an evaluation
 * is `embedding_evaluation_missing`, and only that code offers `force`).
 */
import {
  buildEmbeddingSpace,
  cancelEmbeddingReindexRun,
  getEmbeddingEvaluations,
  getEmbeddingModels,
  getEmbeddingReindexRun,
  getEmbeddingSpaces,
  promoteEmbeddingSpace,
  registerEmbeddingSpace,
  retireEmbeddingSpace,
} from '$lib/api_embeddingSpaces';
import { apiErrorText } from '$lib/api_errors';
import { configErrorDetail } from '$lib/api_profiles';
import { pollWhile } from '$lib/jobs/followJob';
import { jobPhase } from '$lib/jobs/jobPhase';
import type {
  EmbeddingBuildMode,
  EmbeddingEvaluationList,
  EmbeddingModelEntry,
  EmbeddingReindexRun,
  EmbeddingSpaceRegisterRequest,
  EmbeddingSpaceStartResponse,
  EmbeddingSpaceView,
  EmbeddingTarget,
} from '$lib/types_embeddingSpaces';

export interface EmbeddingSpacesDeps {
  getEmbeddingSpaces: typeof getEmbeddingSpaces;
  getEmbeddingModels: typeof getEmbeddingModels;
  getEmbeddingEvaluations: typeof getEmbeddingEvaluations;
  registerEmbeddingSpace: typeof registerEmbeddingSpace;
  buildEmbeddingSpace: typeof buildEmbeddingSpace;
  promoteEmbeddingSpace: typeof promoteEmbeddingSpace;
  retireEmbeddingSpace: typeof retireEmbeddingSpace;
  getEmbeddingReindexRun: typeof getEmbeddingReindexRun;
  cancelEmbeddingReindexRun: typeof cancelEmbeddingReindexRun;
  /** Wait between two reads of a followed run. */
  pollIntervalMs: number;
}

/** A served refusal: its stable code and message. */
export interface Refusal {
  code: string;
  message: string;
}

/** The writes a space row offers. */
export type SpaceAction = 'build' | 'promote' | 'retire' | 'cancel';

type SpaceId = Pick<EmbeddingSpaceView, 'target' | 'space_key'>;

export const spaceRunKey = (s: SpaceId): string => `${s.target}:${s.space_key}`;

export class EmbeddingSpacesState {
  spaces = $state<EmbeddingSpaceView[] | null>(null);
  models = $state<EmbeddingModelEntry[] | null>(null);
  loadError = $state<string | null>(null);
  modelsError = $state<string | null>(null);
  /** The latest read of each followed re-index run, by `spaceRunKey`. */
  runs = $state<Record<string, EmbeddingReindexRun>>({});
  /** The run being followed for each space, until its job ends. */
  following = $state<Record<string, string>>({});
  /** The served evaluation of each space whose evaluation was opened, by `spaceRunKey`. */
  evaluations = $state<Record<string, EmbeddingEvaluationList>>({});
  evaluationErrors = $state<Record<string, string>>({});
  busy = $state(false);
  /** The refusal of the last write; null after a success. */
  refusal = $state<Refusal | null>(null);

  #deps: Partial<EmbeddingSpacesDeps>;
  #follows: Record<string, AbortController> = {};
  /** Runs this tab followed to their end: a list that still names one as active (its job row lags the run's) must not start the same follow again. */
  #followed: Record<string, true> = {};

  constructor(deps: Partial<EmbeddingSpacesDeps> = {}) {
    this.#deps = deps;
  }

  async load(): Promise<void> {
    await Promise.all([this.loadSpaces(), this.#loadModels()]);
  }

  async loadSpaces(): Promise<void> {
    try {
      this.spaces = await (this.#deps.getEmbeddingSpaces ?? getEmbeddingSpaces)();
      this.loadError = null;
      this.#followServedRuns(this.spaces);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = apiErrorText(e);
    }
  }

  /** Follows every space whose newest run is still active and is not followed yet. */
  #followServedRuns(spaces: EmbeddingSpaceView[]): void {
    for (const s of spaces) {
      const key = spaceRunKey(s);
      if (
        s.run_id &&
        s.run_status &&
        jobPhase(s.run_status) === 'active' &&
        !(key in this.following) &&
        !(s.run_id in this.#followed)
      ) {
        void this.#follow(key, s.run_id);
      }
    }
  }

  async #loadModels(): Promise<void> {
    try {
      this.models = await (this.#deps.getEmbeddingModels ?? getEmbeddingModels)();
      this.modelsError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.modelsError = apiErrorText(e);
    }
  }

  /** The latest read of the run started for a space in this tab, if any. */
  runOf(space: SpaceId): EmbeddingReindexRun | null {
    return this.runs[spaceRunKey(space)] ?? null;
  }

  /** True while a run started here for the space has not ended. */
  isFollowing(space: SpaceId): boolean {
    return spaceRunKey(space) in this.following;
  }

  evaluationOf(space: SpaceId): EmbeddingEvaluationList | null {
    return this.evaluations[spaceRunKey(space)] ?? null;
  }

  evaluationErrorOf(space: SpaceId): string | null {
    return this.evaluationErrors[spaceRunKey(space)] ?? null;
  }

  /** Reads the served evaluation of a space; a failed read is kept as served for the dialog. */
  async loadEvaluation(space: SpaceId): Promise<void> {
    const key = spaceRunKey(space);
    try {
      const read = this.#deps.getEmbeddingEvaluations ?? getEmbeddingEvaluations;
      this.evaluations = {
        ...this.evaluations,
        [key]: await read(space.space_key, space.target),
      };
      const { [key]: _gone, ...rest } = this.evaluationErrors;
      this.evaluationErrors = rest;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.evaluationErrors = { ...this.evaluationErrors, [key]: apiErrorText(e) };
    }
  }

  clearRefusal(): void {
    this.refusal = null;
  }

  register(req: EmbeddingSpaceRegisterRequest, idempotencyKey: string): Promise<boolean> {
    return this.#write(async () =>
      this.#started(
        req.target,
        await (this.#deps.registerEmbeddingSpace ?? registerEmbeddingSpace)(
          req,
          idempotencyKey,
        ),
      ),
    );
  }

  build(
    space: EmbeddingSpaceView,
    mode: EmbeddingBuildMode,
    batchSize: number | null,
  ): Promise<boolean> {
    return this.#write(async () =>
      this.#started(
        space.target,
        await (this.#deps.buildEmbeddingSpace ?? buildEmbeddingSpace)(
          space.space_key,
          space.target,
          { mode, ...(batchSize == null ? {} : { batch_size: batchSize }) },
        ),
      ),
    );
  }

  promote(space: EmbeddingSpaceView, force: boolean): Promise<boolean> {
    return this.#write(async () => {
      await (this.#deps.promoteEmbeddingSpace ?? promoteEmbeddingSpace)(
        space.space_key,
        space.target,
        { force },
      );
      await this.loadSpaces();
    });
  }

  retire(space: EmbeddingSpaceView): Promise<boolean> {
    return this.#write(async () =>
      this.#started(
        space.target,
        await (this.#deps.retireEmbeddingSpace ?? retireEmbeddingSpace)(
          space.space_key,
          space.target,
        ),
      ),
    );
  }

  /** Cancels the run followed for `space`; `abandon` also marks the space failed and clears its partial vectors. */
  async cancel(space: SpaceId, abandon: boolean): Promise<boolean> {
    const runId = this.following[spaceRunKey(space)];
    if (!runId) return false;
    return this.#write(async () => {
      await (this.#deps.cancelEmbeddingReindexRun ?? cancelEmbeddingReindexRun)(
        runId,
        abandon,
      );
      await this.loadSpaces();
    });
  }

  /** Stops following every run (the page is leaving). */
  stop(): void {
    for (const c of Object.values(this.#follows)) c.abort();
    this.#follows = {};
  }

  async #write(call: () => Promise<void>): Promise<boolean> {
    if (this.busy) return false;
    this.busy = true;
    this.refusal = null;
    try {
      await call();
      return true;
    } catch (e) {
      const d = configErrorDetail(e);
      this.refusal = { code: d?.error ?? 'request_failed', message: apiErrorText(e) };
      return false;
    } finally {
      this.busy = false;
    }
  }

  async #started(
    target: EmbeddingTarget,
    res: EmbeddingSpaceStartResponse,
  ): Promise<void> {
    await this.loadSpaces();
    if (res.run_id) void this.#follow(`${target}:${res.space_key}`, res.run_id);
  }

  async #follow(key: string, runId: string): Promise<void> {
    this.#follows[key]?.abort();
    const ctl = new AbortController();
    this.#follows[key] = ctl;
    this.following = { ...this.following, [key]: runId };
    const read = this.#deps.getEmbeddingReindexRun ?? getEmbeddingReindexRun;
    try {
      await pollWhile(
        (signal) => read(runId, signal),
        (run) =>
          run.finished_at == null &&
          (run.job_status == null || jobPhase(run.job_status) === 'active'),
        {
          signal: ctl.signal,
          intervalMs: this.#deps.pollIntervalMs,
          maxIntervalMs: this.#deps.pollIntervalMs ?? 3000,
          onUpdate: (run) => (this.runs = { ...this.runs, [key]: run }),
        },
      );
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
    }
    if (this.#follows[key] === ctl) {
      this.#followed[runId] = true;
      delete this.#follows[key];
      this.following = Object.fromEntries(
        Object.entries(this.following).filter(([k]) => k !== key),
      );
      await this.loadSpaces();
    }
  }
}

export function createEmbeddingSpaces(
  deps: Partial<EmbeddingSpacesDeps> = {},
): EmbeddingSpacesState {
  return new EmbeddingSpacesState(deps);
}
