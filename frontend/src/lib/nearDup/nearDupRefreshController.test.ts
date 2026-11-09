import { beforeEach, describe, expect, it, vi } from 'vitest';
import { startNearDupRun } from '$lib/api_nearDup';
import { followJob } from '$lib/jobs/followJob';
import { createNearDupRefreshController } from './nearDupRefreshController.svelte';
import type { NearDupStaleLevel } from '$lib/types_nearDup';
import type { JobView } from '$lib/types_jobs';

vi.mock('$lib/api_nearDup', () => ({ startNearDupRun: vi.fn() }));
vi.mock('$lib/jobs/followJob', () => ({ followJob: vi.fn() }));

const stale: NearDupStaleLevel[] = [
  { level: 'image', stale_reasons: ['no_run'], pending_count: 0 },
  { level: 'item', stale_reasons: ['pending'], pending_count: 5 },
];
const job = (status: string, error_message: string | null = null) =>
  ({ status, error_message }) as unknown as JobView;

beforeEach(() => {
  vi.resetAllMocks();
});

describe('createNearDupRefreshController', () => {
  it('starts one run per stale level in the mode its reasons call for and follows each job', async () => {
    vi.mocked(startNearDupRun)
      .mockResolvedValueOnce({ job_id: 'j1', kind: 'near_dup.build' })
      .mockResolvedValueOnce({ job_id: 'j2', kind: 'near_dup.build' });
    vi.mocked(followJob).mockResolvedValue(job('completed'));
    const c = createNearDupRefreshController();
    c.show(stale);
    await c.refresh();
    expect(startNearDupRun).toHaveBeenNthCalledWith(
      1,
      { level: 'image', mode: 'full' },
      expect.anything(),
    );
    expect(startNearDupRun).toHaveBeenNthCalledWith(
      2,
      { level: 'item', mode: 'incremental' },
      expect.anything(),
    );
    expect(vi.mocked(followJob).mock.calls.map((x) => x[0])).toEqual(['j1', 'j2']);
    expect(c.stale).toEqual([]);
    expect(c.message).toContain('up to date');
    expect(c.error).toBeNull();
  });

  it('keeps the stale levels and names the failure when a run does not complete', async () => {
    vi.mocked(startNearDupRun).mockResolvedValue({ job_id: 'j1', kind: 'k' });
    vi.mocked(followJob).mockResolvedValue(job('failed', 'boom'));
    const c = createNearDupRefreshController();
    c.show(stale);
    await c.refresh();
    expect(c.stale).toEqual(stale);
    expect(c.error).toBe('image level run failed: boom');
    expect(c.busy).toBe(false);
  });

  it('does nothing when nothing is stale', async () => {
    const c = createNearDupRefreshController();
    await c.refresh();
    expect(startNearDupRun).not.toHaveBeenCalled();
  });
});
