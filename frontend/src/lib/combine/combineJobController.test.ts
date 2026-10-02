/**
 * One combine job's view: polling every 2 s while queued/running and
 * stopping at any other status, SSE wake-ups only for this job, cancel /
 * resume availability by status, served refusals verbatim, and the
 * next-step call against the target project's prefix.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { combineJob } from '$lib/test/fixtures/combine';
import type { ProjectEvent } from '$lib/sse';
import {
  COMBINE_POLL_MS,
  createCombineJob,
  type CombineJobDeps,
} from './combineJobController.svelte';

const ID = 'cmb_20261001T120000_1a2b3c4d';

function setup(responses: ReturnType<typeof combineJob>[] | (() => Promise<never>)) {
  let i = 0;
  const getCombineJob = vi.fn(async () => {
    if (typeof responses === 'function') return responses();
    return responses[Math.min(i++, responses.length - 1)]!;
  });
  const cancelCombine = vi.fn(async () =>
    combineJob({ status: 'cancelled', phase: null }),
  );
  const resumeCombine = vi.fn(async () => combineJob({ status: 'queued' }));
  const runCombineNextStep = vi.fn(async () => ({}));
  const getProject = vi.fn(async () => ({ status: 'active' }));
  let emit: (e: ProjectEvent) => void = () => {};
  const close = vi.fn();
  const subscribe: CombineJobDeps['subscribe'] = (onEvent) => {
    emit = onEvent;
    return { close };
  };
  const job = createCombineJob(ID, {
    getCombineJob: getCombineJob as unknown as CombineJobDeps['getCombineJob'],
    cancelCombine: cancelCombine as unknown as CombineJobDeps['cancelCombine'],
    resumeCombine: resumeCombine as unknown as CombineJobDeps['resumeCombine'],
    runCombineNextStep:
      runCombineNextStep as unknown as CombineJobDeps['runCombineNextStep'],
    getProject: getProject as unknown as CombineJobDeps['getProject'],
    projectOf: (slug) =>
      slug === 'merged' ? { prefix: '/curation/projects/merged' } : null,
    subscribe,
  });
  return {
    job,
    getCombineJob,
    cancelCombine,
    resumeCombine,
    runCombineNextStep,
    getProject,
    close,
    emit: (e: ProjectEvent) => emit(e),
  };
}

const evt = (over: Record<string, unknown>): ProjectEvent => ({
  type: 'combine.progress',
  topic: 'project',
  project: null,
  ...over,
});

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('polling', () => {
  it('re-reads every 2 s while running and stops at a terminal status', async () => {
    const { job, getCombineJob } = setup([
      combineJob({ status: 'running' }),
      combineJob({ status: 'running', done: 12 }),
      combineJob({ status: 'completed', phase: 'done', done: 20 }),
    ]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(getCombineJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS - 1);
    expect(getCombineJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(getCombineJob).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(getCombineJob).toHaveBeenCalledTimes(3);
    expect(job.completed).toBe(true);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 5);
    expect(getCombineJob).toHaveBeenCalledTimes(3);
  });

  it.each(['interrupted', 'cancelled', 'failed', 'completed', 'completed_with_errors'])(
    'does not poll a %s job',
    async (status) => {
      const { job, getCombineJob } = setup([combineJob({ status })]);
      job.start();
      await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 3);
      expect(getCombineJob).toHaveBeenCalledTimes(1);
    },
  );

  it('queued polls too, and stop() cancels the pending poll and closes the stream', async () => {
    const { job, getCombineJob, close } = setup([combineJob({ status: 'queued' })]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    job.stop();
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 3);
    expect(getCombineJob).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalled();
  });

  it('a transient failure keeps following a running job', async () => {
    let n = 0;
    const { job, getCombineJob } = setup(async () => {
      n += 1;
      if (n === 1) return combineJob({ status: 'running' }) as never;
      if (n === 2) throw new ApiError(503, '/u', { detail: 'down for a moment' });
      return combineJob({ status: 'completed' }) as never;
    });
    job.start();
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(job.loadError).toBe('down for a moment');
    expect(job.job?.status).toBe('running');
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(getCombineJob).toHaveBeenCalledTimes(3);
    expect(job.loadError).toBeNull();
    expect(job.completed).toBe(true);
  });

  it('a structured combine_not_found 404 is shown verbatim and not retried', async () => {
    const { job, getCombineJob } = setup(async () => {
      throw new ApiError(404, '/u', {
        detail: { error: 'combine_not_found', message: "no combine job 'x'" },
      });
    });
    job.start();
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 3);
    expect(job.notFound).toBe("no combine job 'x'");
    expect(job.job).toBeNull();
    expect(getCombineJob).toHaveBeenCalledTimes(1);
  });
});

describe('events', () => {
  it('a combine.progress event for this job wakes an immediate re-read', async () => {
    const { job, getCombineJob, emit } = setup([
      combineJob({ status: 'running' }),
      combineJob({ status: 'completed' }),
    ]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    emit(evt({ job_id: ID }));
    await vi.advanceTimersByTimeAsync(0);
    expect(getCombineJob).toHaveBeenCalledTimes(2);
    expect(job.completed).toBe(true);
  });

  it('ignores another job, another event type and a missing job_id', async () => {
    const { job, getCombineJob, emit } = setup([combineJob({ status: 'running' })]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    emit(evt({ job_id: 'cmb_other' }));
    emit(evt({}));
    emit(evt({ type: 'project.created', job_id: ID }));
    await vi.advanceTimersByTimeAsync(0);
    expect(getCombineJob).toHaveBeenCalledTimes(1);
  });
});

describe('actions', () => {
  it.each([
    ['queued', true, false],
    ['running', true, false],
    ['interrupted', false, true],
    ['cancelled', false, true],
    ['completed', false, false],
    ['completed_with_errors', false, false],
    ['failed', false, false],
  ])('%s: cancel=%s resume=%s', async (status, cancel, resume) => {
    const { job } = setup([combineJob({ status })]);
    await job.load();
    expect(job.canCancel).toBe(cancel);
    expect(job.canResume).toBe(resume);
  });

  it('cancel adopts the served job and stops polling', async () => {
    const { job, cancelCombine, getCombineJob } = setup([
      combineJob({ status: 'running' }),
    ]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(await job.cancel()).toBe(true);
    expect(cancelCombine).toHaveBeenCalledWith(ID);
    expect(job.status).toBe('cancelled');
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 3);
    expect(getCombineJob).toHaveBeenCalledTimes(1);
  });

  it('resume adopts the queued job and resumes polling', async () => {
    const { job, resumeCombine, getCombineJob } = setup([
      combineJob({ status: 'interrupted' }),
    ]);
    await job.load();
    await job.resume();
    expect(resumeCombine).toHaveBeenCalledWith(ID);
    expect(job.status).toBe('queued');
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(getCombineJob).toHaveBeenCalledTimes(2);
  });

  it('a refusal shows the served message verbatim and re-reads the job', async () => {
    const { job, cancelCombine, getCombineJob } = setup([
      combineJob({ status: 'running' }),
    ]);
    cancelCombine.mockRejectedValueOnce(
      new ApiError(409, '/u', {
        detail: { error: 'combine_not_resumable', message: 'the job is not running' },
      }),
    );
    await job.load();
    expect(await job.cancel()).toBe(false);
    expect(job.actionError).toBe('the job is not running');
    await vi.advanceTimersByTimeAsync(0);
    expect(getCombineJob).toHaveBeenCalledTimes(2);
  });

  it('runs a served next step against the target project prefix', async () => {
    const step = {
      action: 'recluster',
      method: 'POST',
      path: '/cluster/umap/rebuild',
      reason: 'r',
    };
    const { job, runCombineNextStep } = setup([
      combineJob({ status: 'completed', next_steps: [step] }),
    ]);
    await job.load();
    expect(await job.runNextStep(step)).toBe(true);
    expect(runCombineNextStep).toHaveBeenCalledWith(
      { prefix: '/curation/projects/merged' },
      step,
    );
  });

  it('a next step for a target missing from the list does not call anything', async () => {
    const step = { action: 'recluster', method: 'POST', path: '/cluster/umap/rebuild' };
    const { job, runCombineNextStep } = setup([
      combineJob({ status: 'completed', target: 'elsewhere' }),
    ]);
    await job.load();
    expect(await job.runNextStep(step)).toBe(false);
    expect(runCombineNextStep).not.toHaveBeenCalled();
    expect(job.actionError).toContain('not in the project list');
  });

  it('stores the served result of a next step and keeps it per job', async () => {
    const step = { action: 'recluster', method: 'POST', path: '/cluster/umap/rebuild' };
    const body = { status: 'no_residuals', n_residuals: 0, refit: false };
    const { job, runCombineNextStep } = setup([combineJob({ status: 'completed' })]);
    runCombineNextStep.mockResolvedValueOnce(body);
    await job.load();
    expect(job.lastStep).toBeNull();
    await job.runNextStep(step);
    expect(job.lastStep).toEqual({ action: 'recluster', result: body });
    // A different job id is a different controller: nothing carries over.
    expect(setup([combineJob()]).job.lastStep).toBeNull();
  });

  it('a refused next step stores no result and shows the served detail', async () => {
    const step = { action: 'recluster', method: 'POST', path: '/cluster/umap/rebuild' };
    const { job, runCombineNextStep } = setup([combineJob({ status: 'completed' })]);
    runCombineNextStep.mockRejectedValueOnce(
      new ApiError(409, '/x', { detail: 'rebuild already running' }),
    );
    await job.load();
    expect(await job.runNextStep(step)).toBe(false);
    expect(job.lastStep).toBeNull();
    expect(job.actionError).toBe('rebuild already running');
  });
});

describe('target readiness gate', () => {
  const step = { action: 'recluster', method: 'POST', path: '/cluster/umap/rebuild' };
  const done = () => combineJob({ status: 'completed', phase: 'done' });

  it('stays disabled while the target is building, enables on active, then stops reading', async () => {
    const { job, getProject } = setup([done()]);
    getProject
      .mockResolvedValueOnce({ status: 'building' })
      .mockResolvedValueOnce({ status: 'building' })
      .mockResolvedValueOnce({ status: 'active' });
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(getProject).toHaveBeenCalledWith('merged');
    expect(job.targetStatus).toBe('building');
    expect(job.canRunNextStep).toBe(false);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(job.canRunNextStep).toBe(false);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(job.targetStatus).toBe('active');
    expect(job.canRunNextStep).toBe(true);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 5);
    expect(getProject).toHaveBeenCalledTimes(3);
  });

  it('does not read the target while the job is not completed', async () => {
    const { job, getProject } = setup([combineJob({ status: 'running' })]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(getProject).not.toHaveBeenCalled();
    expect(job.canRunNextStep).toBe(false);
  });

  it('a non-building, non-active status stops the poll and keeps steps disabled', async () => {
    const { job, getProject } = setup([done()]);
    getProject.mockResolvedValue({ status: 'failed' });
    job.start();
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 3);
    expect(getProject).toHaveBeenCalledTimes(1);
    expect(job.targetStatus).toBe('failed');
    expect(job.canRunNextStep).toBe(false);
  });

  it('stop() ends the target poll', async () => {
    const { job, getProject } = setup([done()]);
    getProject.mockResolvedValue({ status: 'building' });
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(getProject).toHaveBeenCalledTimes(1);
    job.stop();
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS * 5);
    expect(getProject).toHaveBeenCalledTimes(1);
  });

  it('a failed target read is retried', async () => {
    const { job, getProject } = setup([done()]);
    getProject.mockRejectedValueOnce(new ApiError(500, '/u', { detail: 'x' }));
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(job.canRunNextStep).toBe(false);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(job.canRunNextStep).toBe(true);
  });

  it('a 409 project_building shows the served message and the retry still works', async () => {
    const { job, runCombineNextStep, getProject } = setup([done()]);
    await job.load();
    await vi.advanceTimersByTimeAsync(0);
    expect(job.canRunNextStep).toBe(true);
    runCombineNextStep.mockRejectedValueOnce(
      new ApiError(409, '/u', {
        detail: {
          error: 'project_building',
          message: "project 'merged' is still being created",
        },
      }),
    );
    getProject.mockResolvedValueOnce({ status: 'building' });
    expect(await job.runNextStep(step)).toBe(false);
    expect(job.actionError).toContain("project 'merged' is still being created");
    await vi.advanceTimersByTimeAsync(0);
    expect(job.canRunNextStep).toBe(false);
    await vi.advanceTimersByTimeAsync(COMBINE_POLL_MS);
    expect(job.canRunNextStep).toBe(true);
    expect(await job.runNextStep(step)).toBe(true);
    expect(job.actionError).toBeNull();
  });
});
