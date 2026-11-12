import { describe, expect, it, vi } from 'vitest';
import { RefCountedRun } from './refCountedRun';

function make() {
  let running = false;
  const start = vi.fn(() => {
    running = true;
  });
  const stop = vi.fn(() => {
    running = false;
  });
  const run = new RefCountedRun({ isRunning: () => running, start, stop });
  return { run, start, stop };
}

describe('RefCountedRun', () => {
  it('starts on the first acquire only and stops after the last release, one task later', async () => {
    const { run, start, stop } = make();
    const a = run.acquire();
    const b = run.acquire();
    expect(start).toHaveBeenCalledTimes(1);
    expect(run.held).toBe(true);
    a();
    await new Promise((r) => setTimeout(r, 1));
    expect(stop).not.toHaveBeenCalled();
    b();
    expect(stop).not.toHaveBeenCalled();
    await new Promise((r) => setTimeout(r, 1));
    expect(stop).toHaveBeenCalledTimes(1);
    expect(run.held).toBe(false);
  });

  it('a re-acquire in the same tick cancels the pending stop and does not restart', async () => {
    const { run, start, stop } = make();
    run.acquire()();
    run.acquire();
    await new Promise((r) => setTimeout(r, 1));
    expect(stop).not.toHaveBeenCalled();
    expect(start).toHaveBeenCalledTimes(1);
  });

  it('never counts below zero', () => {
    const { run } = make();
    const release = run.acquire();
    release();
    release();
    expect(run.held).toBe(false);
  });
});
