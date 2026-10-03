/**
 * The region-stage panel's state: the served read, a pause/resume that is
 * only ever sent after an explicit confirm, the served state adopted from
 * the write, and the served refusal text.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { RegionStageState } from '$lib/types_openVocab';
import { createRegionStage } from './regionStageController.svelte';

const RERUN = {
  targets: { filter: { region_gate_skipped: true } },
  scopes: ['region'],
  dry_run: true,
};

const state = (over: Partial<RegionStageState> = {}): RegionStageState =>
  ({
    project: 'alpha',
    paused: false,
    paused_since: null,
    pipeline_paused: false,
    counts: { pending_detection: 4, pending_verification: 1, gate_skipped: 7 },
    rerun_skipped: RERUN,
    ...over,
  }) as RegionStageState;

function setup(over: Record<string, unknown> = {}) {
  const deps = {
    getRegionStage: vi.fn().mockResolvedValue(state()),
    pauseRegionStage: vi.fn().mockResolvedValue(state({ paused: true })),
    resumeRegionStage: vi.fn().mockResolvedValue(state()),
    ...over,
  };
  return { s: createRegionStage(deps as never), deps };
}

describe('RegionStage', () => {
  it('loads the served state', async () => {
    const { s } = setup();
    await s.load();
    expect(s.state?.counts.gate_skipped).toBe(7);
    expect(s.loadError).toBeNull();
  });

  it('asking to pause writes nothing until it is confirmed', async () => {
    const { s, deps } = setup();
    await s.load();
    s.ask('pause');
    expect(s.confirming).toBe('pause');
    expect(deps.pauseRegionStage).not.toHaveBeenCalled();
    expect(await s.confirm()).toBe(true);
    expect(deps.pauseRegionStage).toHaveBeenCalledTimes(1);
    expect(s.state?.paused).toBe(true);
    expect(s.confirming).toBeNull();
  });

  it('cancelling the confirm leaves the stage alone', async () => {
    const { s, deps } = setup();
    await s.load();
    s.ask('pause');
    s.cancel();
    expect(s.confirming).toBeNull();
    expect(await s.confirm()).toBe(false);
    expect(deps.pauseRegionStage).not.toHaveBeenCalled();
  });

  it('resume goes through the same confirm', async () => {
    const { s, deps } = setup({
      getRegionStage: vi.fn().mockResolvedValue(state({ paused: true })),
    });
    await s.load();
    s.ask('resume');
    await s.confirm();
    expect(deps.resumeRegionStage).toHaveBeenCalledTimes(1);
    expect(s.state?.paused).toBe(false);
  });

  it('shows a refusal as served and keeps the old state', async () => {
    const { s } = setup({
      pauseRegionStage: vi
        .fn()
        .mockRejectedValue(
          new ApiError(409, '/x', { detail: 'no region profile is configured' }),
        ),
    });
    await s.load();
    s.ask('pause');
    expect(await s.confirm()).toBe(false);
    expect(s.actionError).toBe('no region profile is configured');
    expect(s.state?.paused).toBe(false);
  });

  it('shows a structured no_active_profile refusal by its served message', async () => {
    const { s } = setup({
      pauseRegionStage: vi.fn().mockRejectedValue(
        new ApiError(409, '/x', {
          detail: { error: 'no_active_profile', message: 'No region profile is active.' },
        }),
      ),
    });
    await s.load();
    s.ask('pause');
    expect(await s.confirm()).toBe(false);
    expect(s.actionError).toBe('No region profile is active.');
  });

  it('exposes the served rerun request untouched', async () => {
    const { s } = setup();
    await s.load();
    expect(s.rerunRequest).toEqual(RERUN);
  });
});
