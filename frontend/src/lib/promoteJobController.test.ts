import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import {
  PromoteJobController,
  isPromoteActive,
  promoteFailureText,
  promoteInProgressId,
} from './promoteJobController.svelte';
import type { PromoteJobStatus, PromoteResponse } from '$lib/types_train';

const RESULT: PromoteResponse = {
  job_id: 'run-1',
  triton_name: 'run_1',
  onnx_path: '/m/model.onnx',
  config_path: '/m/config.pbtxt',
  labels_path: '/m/labels.txt',
  triton_loaded: true,
  cold_start_expected_on_first_inference: false,
};

function job(status: PromoteJobStatus['status'], extra: Partial<PromoteJobStatus> = {}) {
  const active = !['done', 'failed'].includes(status);
  return {
    promote_id: 'p1',
    job_id: 'run-1',
    triton_name: 'run_1',
    status,
    poll_after_s: active ? 3 : null,
    ...extra,
  } satisfies PromoteJobStatus;
}

const BODY = { triton_name: 'run_1' };

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('PromoteJobController', () => {
  it('walks the served phases, polling every poll_after_s, and ends on done with the result', async () => {
    const getJob = vi
      .fn()
      .mockResolvedValueOnce(job('exporting', { poll_after_s: 2 }))
      .mockResolvedValueOnce(job('loading'))
      .mockResolvedValueOnce(job('building'))
      .mockResolvedValueOnce(job('warming'))
      .mockResolvedValueOnce(job('done', { result: RESULT }));
    const onDone = vi.fn();
    const c = new PromoteJobController({
      promote: vi.fn().mockResolvedValue(job('queued', { poll_after_s: 2 })),
      getJob,
      onDone,
    });
    await c.start('run-1', BODY);
    expect(c.job?.status).toBe('queued');
    expect(getJob).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1999);
    expect(getJob).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(c.job?.status).toBe('exporting');
    const seen = [c.job?.status];
    for (let i = 0; i < 3; i++) {
      await vi.advanceTimersByTimeAsync(3000);
      seen.push(c.job?.status);
    }
    expect(seen).toEqual(['exporting', 'loading', 'building', 'warming']);
    await vi.advanceTimersByTimeAsync(3000);
    expect(c.job?.status).toBe('done');
    expect(onDone).toHaveBeenCalledWith(
      RESULT,
      expect.objectContaining({ status: 'done' }),
    );
    expect(getJob).toHaveBeenCalledTimes(5);
    expect(getJob).toHaveBeenCalledWith('run-1', 'p1', expect.anything());

    await vi.advanceTimersByTimeAsync(30_000);
    expect(getJob).toHaveBeenCalledTimes(5);
  });

  it('failed: stops polling and exposes error + error_status', async () => {
    const c = new PromoteJobController({
      promote: vi.fn().mockResolvedValue(job('queued')),
      getJob: vi
        .fn()
        .mockResolvedValue(
          job('failed', { error: 'Triton refused the load', error_status: 502 }),
        ),
    });
    await c.start('run-1', BODY);
    await vi.advanceTimersByTimeAsync(3000);
    expect(c.job?.status).toBe('failed');
    expect(c.failureText).toBe('Triton refused the load (502)');
    expect(c.active).toBe(false);
  });

  it('a double-click (200 for the same active job) just follows that job', async () => {
    const promote = vi.fn().mockResolvedValue(job('building'));
    const getJob = vi.fn().mockResolvedValue(job('done', { result: RESULT }));
    const c = new PromoteJobController({ promote, getJob });
    await Promise.all([c.start('run-1', BODY), c.start('run-1', BODY)]);
    expect(promote).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(3000);
    expect(getJob).toHaveBeenCalledTimes(1);
    expect(c.job?.status).toBe('done');
  });

  it('409 promote_in_progress attaches to the served promote_id', async () => {
    const conflict = new ApiError(409, '/x', {
      detail: {
        code: 'promote_in_progress',
        message: 'a promote of run-1 is already running as other_name',
        promote_id: 'p9',
      },
    });
    const getJob = vi
      .fn()
      .mockResolvedValue(job('loading', { promote_id: 'p9', triton_name: 'other_name' }));
    const c = new PromoteJobController({
      promote: vi.fn().mockRejectedValue(conflict),
      getJob,
    });
    await c.start('run-1', BODY);
    expect(c.error).toBeNull();
    expect(getJob).toHaveBeenCalledWith('run-1', 'p9', expect.anything());
    expect(c.job?.triton_name).toBe('other_name');
  });

  it('a synchronous ApiError is surfaced as a rejection for the caller (gate 422 handling)', async () => {
    const gate = new ApiError(422, '/x', { detail: { message: 'gate', failures: [] } });
    const c = new PromoteJobController({ promote: vi.fn().mockRejectedValue(gate) });
    await expect(c.start('run-1', BODY)).rejects.toBe(gate);
    expect(c.job).toBeNull();
  });

  it('attach() resumes an already-served active job (page reload)', async () => {
    const getJob = vi.fn().mockResolvedValue(job('warming'));
    const c = new PromoteJobController({ getJob });
    c.attach('run-1', job('building'));
    expect(c.job?.status).toBe('building');
    await vi.advanceTimersByTimeAsync(3000);
    expect(c.job?.status).toBe('warming');
  });

  it('attach() with a terminal job does not poll', async () => {
    const getJob = vi.fn();
    const onDone = vi.fn();
    const c = new PromoteJobController({ getJob, onDone });
    c.attach('run-1', job('done', { result: RESULT }));
    await vi.advanceTimersByTimeAsync(10_000);
    expect(getJob).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
  });

  it('gives up with a visible error after repeated poll failures, and stop() halts polling', async () => {
    const getJob = vi.fn().mockRejectedValue(new ApiError(500, '/x', null));
    const c = new PromoteJobController({
      promote: vi.fn().mockResolvedValue(job('queued')),
      getJob,
    });
    await c.start('run-1', BODY);
    await vi.advanceTimersByTimeAsync(3000 * 6);
    expect(c.error).toMatch(/lost track/i);
    const calls = getJob.mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(getJob.mock.calls.length).toBe(calls);

    const c2 = new PromoteJobController({
      promote: vi.fn().mockResolvedValue(job('queued')),
      getJob: vi.fn().mockResolvedValue(job('loading')),
    });
    await c2.start('run-1', BODY);
    c2.stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(c2.job?.status).toBe('queued');
  });
});

describe('promote helpers', () => {
  it('isPromoteActive is true only for in-flight phases', () => {
    expect(isPromoteActive(job('queued'))).toBe(true);
    expect(isPromoteActive(job('warming'))).toBe(true);
    expect(isPromoteActive(job('done'))).toBe(false);
    expect(isPromoteActive(job('failed'))).toBe(false);
    expect(isPromoteActive(null)).toBe(false);
  });

  it('promoteInProgressId reads only a 409 promote_in_progress body', () => {
    const body = { detail: { code: 'promote_in_progress', promote_id: 'p9' } };
    expect(promoteInProgressId(new ApiError(409, '/x', body))).toBe('p9');
    expect(promoteInProgressId(new ApiError(422, '/x', body))).toBeNull();
    expect(
      promoteInProgressId(new ApiError(409, '/x', { detail: { code: 'in_use' } })),
    ).toBeNull();
    expect(promoteInProgressId(new Error('x'))).toBeNull();
  });

  it('promoteFailureText maps 409/422 to actionable text', () => {
    expect(
      promoteFailureText(job('failed', { error: 'exists', error_status: 409 })),
    ).toMatch(/Overwrite existing/);
    expect(
      promoteFailureText(job('failed', { error: 'bad gate', error_status: 422 })),
    ).toBe('bad gate (422)');
    expect(promoteFailureText(job('failed', {}))).toBe('Promote failed.');
  });
});

describe('findActivePromote (resume after reload)', () => {
  it('returns the first run whose served status carries an active promote', async () => {
    const { findActivePromote } = await import('$lib/promote');
    const getStatus = vi.fn(async (id?: string) => {
      if (id === 'r2')
        return { job_id: 'r2', promote: job('loading', { job_id: 'r2' }) } as never;
      return { job_id: id ?? '', promote: id === 'r1' ? job('done') : null } as never;
    });
    const hit = await findActivePromote(['r1', 'r2', 'r3'], getStatus);
    expect(hit?.runJobId).toBe('r2');
    expect(hit?.promote.status).toBe('loading');
    expect(await findActivePromote(['r1', 'r3'], getStatus)).toBeNull();
  });

  it('skips a run whose status lookup fails', async () => {
    const { findActivePromote } = await import('$lib/promote');
    const getStatus = vi.fn(async (id?: string) => {
      if (id === 'bad') throw new Error('boom');
      return { job_id: id ?? '', promote: job('warming', { job_id: id ?? '' }) } as never;
    });
    expect((await findActivePromote(['bad', 'ok'], getStatus))?.runJobId).toBe('ok');
  });
});
