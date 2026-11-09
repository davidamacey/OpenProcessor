import { describe, expect, it, vi } from 'vitest';
import { followJob } from '$lib/jobs/followJob';
import type { JobStatus, JobView } from '$lib/types_jobs';

function job(status: JobStatus, over: Partial<JobView> = {}): JobView {
  return {
    id: 'j1',
    kind: 'label_relabel',
    status,
    project: 'p',
    scope: 'project',
    queue: 'default',
    progress: {},
    progress_updated_at: null,
    result: null,
    error_code: null,
    error_message: null,
    pause_reason: null,
    pause_requested: false,
    cancel_requested: false,
    attempt: 1,
    max_attempts: 3,
    retry_count: 0,
    requested_by: 'human',
    created_at: '2026-11-08T00:00:00Z',
    queued_at: null,
    started_at: null,
    finished_at: null,
    updated_at: '2026-11-08T00:00:00Z',
    ...over,
  };
}

const fast = { intervalMs: 1, maxIntervalMs: 2 };

describe('followJob', () => {
  it('shows queued and running as the server reports them, then resolves on completed', async () => {
    const seq: JobStatus[] = ['queued', 'running', 'completed'];
    const read = vi.fn(async () =>
      job(seq.shift()!, { result: { updated: 3, conflicts: 0 } }),
    );
    const seen: JobStatus[] = [];
    const done = await followJob('j1', {
      ...fast,
      read,
      onUpdate: (j) => seen.push(j.status),
    });
    expect(seen).toEqual(['queued', 'running', 'completed']);
    expect(done.result).toEqual({ updated: 3, conflicts: 0 });
    expect(read).toHaveBeenCalledTimes(3);
  });

  it.each([
    'failed',
    'cancelled',
    'interrupted',
    'paused',
    'completed_with_errors',
  ] as const)('returns on %s without polling again', async (status) => {
    const read = vi.fn(async () => job(status));
    expect((await followJob('j1', { ...fast, read })).status).toBe(status);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it('keeps polling through cancelling', async () => {
    const seq: JobStatus[] = ['cancelling', 'cancelled'];
    const read = vi.fn(async () => job(seq.shift()!));
    expect((await followJob('j1', { ...fast, read })).status).toBe('cancelled');
  });

  it('survives a failed read and gives up after the third in a row', async () => {
    let n = 0;
    const flaky = vi.fn(async () => {
      if (++n === 1) throw new Error('blip');
      return job('completed');
    });
    expect((await followJob('j1', { ...fast, read: flaky })).status).toBe('completed');

    const dead = vi.fn(async () => {
      throw new Error('down');
    });
    await expect(followJob('j1', { ...fast, read: dead })).rejects.toThrow('down');
    expect(dead).toHaveBeenCalledTimes(3);
  });

  it('stops when aborted', async () => {
    const ctl = new AbortController();
    const read = vi.fn(async () => job('running'));
    const p = followJob('j1', { ...fast, read, signal: ctl.signal });
    ctl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });
});
