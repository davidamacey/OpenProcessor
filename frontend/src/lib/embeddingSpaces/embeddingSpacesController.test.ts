import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type {
  EmbeddingEvaluationList,
  EmbeddingReindexRun,
  EmbeddingSpaceView,
} from '$lib/types_embeddingSpaces';
import { createEmbeddingSpaces } from './embeddingSpacesController.svelte';

const space = (over: Partial<EmbeddingSpaceView> = {}): EmbeddingSpaceView => ({
  space_key: 'mobileclip2_s2_v1_512',
  target: 'item',
  model_key: 'mobileclip2_s2',
  model_version: 1,
  dim: 512,
  role: 'candidate',
  status: 'building',
  ready_at: null,
  docs_with_vector: 10,
  run_id: null,
  run_mode: null,
  run_status: null,
  ...over,
});

const run = (over: Partial<EmbeddingReindexRun> = {}): EmbeddingReindexRun => ({
  id: 'run-1',
  space_key: 'mobileclip2_s2_v1_512',
  mode: 'build',
  scope: {},
  docs_total: 100,
  docs_done: 40,
  docs_failed: 0,
  docs_skipped: 0,
  bytes_estimate: null,
  started_at: null,
  finished_at: null,
  job_id: 'job-1',
  job_status: 'running',
  job_progress: {},
  ...over,
});

const evaluation = (
  over: Partial<EmbeddingEvaluationList> = {},
): EmbeddingEvaluationList => ({
  space_key: 'mobileclip2_s2_v1_512',
  target: 'item',
  primary_space_key: 'pe_core_l14_336_v1',
  evaluated_against_primary: true,
  evaluations: [
    {
      id: 'ev-1',
      baseline_space_key: 'pe_core_l14_336_v1',
      candidate_space_key: 'mobileclip2_s2_v1_512',
      protocol: 'label_consistency_v1',
      verdict: 'better',
      metrics: { recall_at_10: { baseline: 0.71, candidate: 0.78 } },
      sample_spec: { seed: 7, n: 2000 },
      counts_for_promote: true,
      job_id: null,
      created_at: '2026-11-12T10:00:00Z',
      created_by: 'api',
    },
  ],
  ...over,
});

const started = {
  space_key: 'mobileclip2_s2_v1_512',
  target: 'item',
  job_id: 'job-1',
  run_id: 'run-1',
  bytes_estimate: null,
};

const refusal = (status: number, error: string, message: string) =>
  new ApiError(status, '/x', { detail: { error, message } });

function setup(over = {}) {
  const deps = {
    getEmbeddingSpaces: vi.fn().mockResolvedValue([space()]),
    getEmbeddingModels: vi.fn().mockResolvedValue([]),
    getEmbeddingEvaluations: vi.fn().mockResolvedValue(evaluation()),
    registerEmbeddingSpace: vi.fn().mockResolvedValue(started),
    buildEmbeddingSpace: vi.fn().mockResolvedValue(started),
    promoteEmbeddingSpace: vi.fn().mockResolvedValue({}),
    retireEmbeddingSpace: vi.fn().mockResolvedValue({ ...started, run_id: null }),
    getEmbeddingReindexRun: vi.fn().mockResolvedValue(run()),
    cancelEmbeddingReindexRun: vi.fn().mockResolvedValue({}),
    pollIntervalMs: 1,
    ...over,
  };
  return { st: createEmbeddingSpaces(deps), deps };
}

describe('EmbeddingSpacesState follows the runs the list names', () => {
  it('follows a build started elsewhere: the run the list names, until its job ends', async () => {
    const running = space({ run_id: 'run-9', run_mode: 'build', run_status: 'running' });
    const { st, deps } = setup({
      getEmbeddingSpaces: vi.fn().mockResolvedValue([running]),
    });
    deps.getEmbeddingReindexRun
      .mockResolvedValueOnce(run({ id: 'run-9', docs_done: 10 }))
      .mockResolvedValue(
        run({ id: 'run-9', docs_done: 100, job_status: 'succeeded', finished_at: 'x' }),
      );
    await st.loadSpaces();
    expect(st.isFollowing(running)).toBe(true);
    await vi.waitFor(() => expect(st.isFollowing(running)).toBe(false));
    expect(deps.getEmbeddingReindexRun.mock.calls[0]![0]).toBe('run-9');
    expect(st.runOf(running)?.docs_done).toBe(100);
  });

  it('does not follow a run that already ended, and offers cancel only for a followed one', async () => {
    const done = space({ run_id: 'run-3', run_mode: 'build', run_status: 'succeeded' });
    const { st, deps } = setup({ getEmbeddingSpaces: vi.fn().mockResolvedValue([done]) });
    await st.loadSpaces();
    expect(deps.getEmbeddingReindexRun).not.toHaveBeenCalled();
    expect(st.isFollowing(done)).toBe(false);
    expect(await st.cancel(done, false)).toBe(false);
  });

  it('follows a queued, paused or cancelling run too, and one run only once', async () => {
    const queued = space({ run_id: 'run-5', run_mode: 'backfill', run_status: 'queued' });
    const { st, deps } = setup({
      getEmbeddingSpaces: vi.fn().mockResolvedValue([queued]),
      pollIntervalMs: 20,
    });
    await st.loadSpaces();
    await st.loadSpaces();
    st.stop();
    await vi.waitFor(() => expect(deps.getEmbeddingReindexRun).toHaveBeenCalled());
    expect(deps.getEmbeddingReindexRun.mock.calls.every((c) => c[0] === 'run-5')).toBe(
      true,
    );
  });
});

describe('EmbeddingSpacesState evaluation', () => {
  it('reads the served evaluation of the space and keeps it as served', async () => {
    const { st, deps } = setup();
    const s = space();
    await st.loadEvaluation(s);
    expect(deps.getEmbeddingEvaluations).toHaveBeenCalledWith(
      'mobileclip2_s2_v1_512',
      'item',
    );
    expect(st.evaluationOf(s)?.evaluations[0]?.verdict).toBe('better');
    expect(st.evaluationOf(s)?.evaluated_against_primary).toBe(true);
    expect(st.evaluationErrorOf(s)).toBeNull();
  });

  it('a refused read is the served message, and a later good read clears it', async () => {
    const read = vi
      .fn()
      .mockRejectedValueOnce(refusal(404, 'embedding_space_not_found', 'no item space x'))
      .mockResolvedValue(
        evaluation({ evaluations: [], evaluated_against_primary: false }),
      );
    const { st } = setup({ getEmbeddingEvaluations: read });
    const s = space();
    await st.loadEvaluation(s);
    expect(st.evaluationErrorOf(s)).toBe('no item space x');
    expect(st.evaluationOf(s)).toBeNull();
    await st.loadEvaluation(s);
    expect(st.evaluationErrorOf(s)).toBeNull();
    expect(st.evaluationOf(s)?.evaluations).toEqual([]);
  });
});

describe('EmbeddingSpacesState', () => {
  it('loads the spaces and the encoders, each with its own error', async () => {
    const { st } = setup({
      getEmbeddingModels: vi.fn().mockRejectedValue(new Error('catalog down')),
    });
    await st.load();
    expect(st.spaces?.map((s) => s.space_key)).toEqual(['mobileclip2_s2_v1_512']);
    expect(st.modelsError).toBe('catalog down');
    expect(st.loadError).toBeNull();
  });

  it('registers with the request and idempotency key, then follows the served run to its end', async () => {
    const { st, deps } = setup();
    deps.getEmbeddingReindexRun
      .mockResolvedValueOnce(run({ docs_done: 40 }))
      .mockResolvedValue(
        run({
          docs_done: 100,
          finished_at: '2026-11-12T10:00:00Z',
          job_status: 'completed',
        }),
      );
    const req = {
      model_key: 'mobileclip2_s2',
      model_version: 1,
      target: 'item' as const,
    };
    expect(await st.register(req, 'try-12345678')).toBe(true);
    expect(deps.registerEmbeddingSpace).toHaveBeenCalledWith(req, 'try-12345678');
    const s = space();
    await vi.waitFor(() => expect(st.isFollowing(s)).toBe(true));
    await vi.waitFor(() => expect(st.runOf(s)?.docs_done).toBe(100));
    await vi.waitFor(() => expect(st.isFollowing(s)).toBe(false));
    expect(st.runOf(s)?.job_status).toBe('completed');
    // listed once after the 202, once after the run ended
    expect(deps.getEmbeddingSpaces.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it('a queued, running, paused or cancelling job keeps being read', async () => {
    const { st, deps } = setup();
    deps.getEmbeddingReindexRun
      .mockResolvedValueOnce(run({ job_status: 'queued' }))
      .mockResolvedValueOnce(run({ job_status: 'cancelling' }))
      .mockResolvedValue(
        run({ job_status: 'cancelled', finished_at: '2026-11-12T10:00:00Z' }),
      );
    await st.build(space(), 'build', null);
    await vi.waitFor(() => expect(st.isFollowing(space())).toBe(false));
    expect(deps.getEmbeddingReindexRun.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it('restart sends the mode, and the batch size only when one was typed', async () => {
    const { st, deps } = setup();
    await st.build(space(), 'backfill', 128);
    expect(deps.buildEmbeddingSpace).toHaveBeenCalledWith(
      'mobileclip2_s2_v1_512',
      'item',
      {
        mode: 'backfill',
        batch_size: 128,
      },
    );
    await vi.waitFor(() => expect(st.isFollowing(space())).toBe(true));
    await st.build(space(), 'build', null);
    expect(deps.buildEmbeddingSpace).toHaveBeenLastCalledWith(
      'mobileclip2_s2_v1_512',
      'item',
      {
        mode: 'build',
      },
    );
    st.stop();
  });

  it('a refused promote keeps the served code and message; force is sent only when asked', async () => {
    const { st, deps } = setup();
    deps.promoteEmbeddingSpace.mockRejectedValueOnce(
      refusal(
        409,
        'embedding_evaluation_missing',
        'no evaluation of a against b; run one or pass force',
      ),
    );
    expect(await st.promote(space({ status: 'ready' }), false)).toBe(false);
    expect(st.refusal).toEqual({
      code: 'embedding_evaluation_missing',
      message: 'no evaluation of a against b; run one or pass force',
    });
    expect(deps.promoteEmbeddingSpace).toHaveBeenCalledWith(
      'mobileclip2_s2_v1_512',
      'item',
      { force: false },
    );
    expect(await st.promote(space({ status: 'ready' }), true)).toBe(true);
    expect(deps.promoteEmbeddingSpace).toHaveBeenLastCalledWith(
      'mobileclip2_s2_v1_512',
      'item',
      { force: true },
    );
    expect(st.refusal).toBeNull();
  });

  it('a space that is not ready is the served refusal', async () => {
    const { st, deps } = setup();
    deps.promoteEmbeddingSpace.mockRejectedValueOnce(
      refusal(409, 'embedding_space_not_ready', 'x is building, not ready'),
    );
    await st.promote(space(), false);
    expect(st.refusal?.code).toBe('embedding_space_not_ready');
  });

  it('retire sends the target; a refusal (the primary) is shown as served', async () => {
    const { st, deps } = setup();
    expect(await st.retire(space({ role: 'optional', status: 'ready' }))).toBe(true);
    expect(deps.retireEmbeddingSpace).toHaveBeenCalledWith(
      'mobileclip2_s2_v1_512',
      'item',
    );
    deps.retireEmbeddingSpace.mockRejectedValueOnce(
      refusal(409, 'embedding_space_is_primary', 'the primary cannot be retired'),
    );
    expect(await st.retire(space({ role: 'primary' }))).toBe(false);
    expect(st.refusal?.message).toBe('the primary cannot be retired');
  });

  it('cancel needs a run started here and sends abandon as chosen', async () => {
    const { st, deps } = setup();
    expect(await st.cancel(space(), false)).toBe(false);
    expect(deps.cancelEmbeddingReindexRun).not.toHaveBeenCalled();
    await st.build(space(), 'build', null);
    await vi.waitFor(() => expect(st.isFollowing(space())).toBe(true));
    expect(await st.cancel(space(), true)).toBe(true);
    expect(deps.cancelEmbeddingReindexRun).toHaveBeenCalledWith('run-1', true);
    st.stop();
  });

  it('stop ends the reads', async () => {
    const { st, deps } = setup();
    await st.build(space(), 'build', null);
    await vi.waitFor(() => expect(deps.getEmbeddingReindexRun).toHaveBeenCalled());
    st.stop();
    const n = deps.getEmbeddingReindexRun.mock.calls.length;
    await new Promise((r) => setTimeout(r, 30));
    expect(deps.getEmbeddingReindexRun.mock.calls.length).toBeLessThanOrEqual(n + 1);
  });
});
