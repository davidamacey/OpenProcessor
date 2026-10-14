/**
 * /train's status poll is disposed with the component: once the owning
 * effect root is torn down (the page unmounts) no interval keeps firing
 * status/run reads, and the in-flight status read is aborted.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getTrainStatus: vi.fn(), getTrainRuns: vi.fn() };
});
import { getTrainRuns, getTrainStatus } from '$lib/api';
import { createTrainRunsController } from './trainRunsController.svelte';

let cleanup: (() => void) | undefined;

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(getTrainStatus).mockResolvedValue(null as never);
  vi.mocked(getTrainRuns).mockResolvedValue({ items: [], total: 0 } as never);
});

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
  vi.useRealTimers();
  vi.mocked(getTrainStatus).mockReset();
  vi.mocked(getTrainRuns).mockReset();
});

describe('train runs controller polling lifecycle', () => {
  it('polls while mounted and stops polling and aborts the read on teardown', async () => {
    cleanup = $effect.root(() => {
      createTrainRunsController();
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(getTrainStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(10_000);
    expect(getTrainStatus).toHaveBeenCalledTimes(2);
    const signal = vi.mocked(getTrainStatus).mock.calls[1]?.[1] as AbortSignal;
    expect(signal.aborted).toBe(false);

    cleanup();
    cleanup = undefined;
    expect(signal.aborted).toBe(true);
    const statusCalls = vi.mocked(getTrainStatus).mock.calls.length;
    const runsCalls = vi.mocked(getTrainRuns).mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(getTrainStatus).toHaveBeenCalledTimes(statusCalls);
    expect(getTrainRuns).toHaveBeenCalledTimes(runsCalls);
  });
});
