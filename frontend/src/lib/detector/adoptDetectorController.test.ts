import { describe, expect, it, vi } from 'vitest';
import type { AdoptDetectorResponse } from '$lib/types_labelOps';
import { AdoptDetectorController } from './adoptDetectorController.svelte';

const dry: AdoptDetectorResponse = {
  dry_run: true,
  items: 40,
  classes_to_create: ['bus'],
  blocked: [],
  blocked_items: 0,
  remaining: 0,
  op_id: null,
  job_id: null,
};
const real: AdoptDetectorResponse = {
  ...dry,
  dry_run: false,
  remaining: 5,
  op_id: 'op-1',
  job_id: 'job-1',
};

describe('AdoptDetectorController', () => {
  it('checks with a dry run first and writes nothing until confirmed', async () => {
    const adopt = vi.fn(async (dryRun: boolean) => (dryRun ? dry : real));
    const run = vi.fn();
    const ctl = new AdoptDetectorController(adopt, run);
    await ctl.check();
    expect(adopt.mock.calls).toEqual([[true]]);
    expect(ctl.phase).toBe('ready');
    expect(ctl.preview).toEqual(dry);
    expect(run).not.toHaveBeenCalled();
  });

  it('confirm writes, follows the job and keeps the served remaining and op id', async () => {
    const adopt = vi.fn(async (dryRun: boolean) => (dryRun ? dry : real));
    const run = vi.fn(async () => ({
      opId: 'op-1',
      updated: 35,
      conflicts: 0,
      complete: true,
      status: 'completed' as const,
    }));
    const ctl = new AdoptDetectorController(adopt, run);
    await ctl.check();
    await ctl.confirm();
    expect(adopt.mock.calls).toEqual([[true], [false]]);
    expect(run).toHaveBeenCalledWith({ op_id: 'op-1', job_id: 'job-1', status: null });
    expect(ctl.phase).toBe('done');
    expect(ctl.written?.remaining).toBe(5);
    expect(ctl.outcome?.updated).toBe(35);
  });

  it('a failed dry run is a failure the dialog can retry', async () => {
    const boom = new Error('down');
    const ctl = new AdoptDetectorController(
      vi.fn(async () => {
        throw boom;
      }),
      vi.fn(),
    );
    await ctl.check();
    expect(ctl.phase).toBe('failed');
    expect(ctl.failure).toBe(boom);
  });
});
