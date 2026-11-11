import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  getNearDupStatus,
  listNearDupGroups,
  listNearDupRuns,
  startNearDupRun,
} from '$lib/api_nearDup';
import { ApiError } from '$lib/api_errors';
import { followJob } from '$lib/jobs/followJob';
import { createNearDupPageController } from './nearDupPageController.svelte';
import type { JobView } from '$lib/types_jobs';
import type { NearDupGroupSummary } from '$lib/types_nearDup';

vi.mock('$lib/api_nearDup', () => ({
  getNearDupStatus: vi.fn(),
  listNearDupGroups: vi.fn(),
  listNearDupRuns: vi.fn(),
  startNearDupRun: vi.fn(),
}));
vi.mock('$lib/jobs/followJob', () => ({ followJob: vi.fn() }));

const group = (id: string): NearDupGroupSummary => ({
  group_id: id,
  purpose: 'dedup',
  size: 3,
  representative_id: `rep-${id}`,
  min_edge_cosine: 0.99,
  min_cosine_to_rep: 0.98,
  has_holdout: false,
});

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getNearDupStatus).mockResolvedValue({ profile: null, levels: [] });
  vi.mocked(listNearDupRuns).mockResolvedValue({ items: [] });
  vi.mocked(listNearDupGroups).mockResolvedValue({ items: [], next_cursor: null });
});

describe('createNearDupPageController', () => {
  it('loads the status and the run history together', async () => {
    const c = createNearDupPageController();
    await c.loadOverview();
    expect(getNearDupStatus).toHaveBeenCalledTimes(1);
    expect(listNearDupRuns).toHaveBeenCalledWith(undefined, 10, expect.anything());
    expect(c.loadError).toBeNull();
    c.dispose();
  });

  it('shows the served message when the overview read fails', async () => {
    vi.mocked(getNearDupStatus).mockRejectedValue(
      new ApiError(500, '/x', { detail: 'store down' }),
    );
    const c = createNearDupPageController();
    await c.loadOverview();
    expect(c.loadError).toContain('store down');
    c.dispose();
  });

  it('pages groups with the cursor', async () => {
    vi.mocked(listNearDupGroups)
      .mockResolvedValueOnce({ items: [group('a')], next_cursor: 'c1' })
      .mockResolvedValueOnce({ items: [group('b')], next_cursor: null });
    const c = createNearDupPageController();
    await c.showGroups('image', 'dedup');
    expect(c.groups.map((g) => g.group_id)).toEqual(['a']);
    expect(c.hasMore).toBe(true);
    await c.loadGroups(true);
    expect(vi.mocked(listNearDupGroups).mock.calls[1]![0]).toMatchObject({
      cursor: 'c1',
    });
    expect(c.groups.map((g) => g.group_id)).toEqual(['a', 'b']);
    expect(c.hasMore).toBe(false);
    c.dispose();
  });

  it('keeps the served refusal of a run start, such as a 409', async () => {
    vi.mocked(startNearDupRun).mockRejectedValue(
      new ApiError(409, '/x', {
        detail: { error: 'no_image_run', message: 'Build the image level first.' },
      }),
    );
    const c = createNearDupPageController();
    await c.startRun('item', 'full');
    expect(c.runError).toBe('Build the image level first.');
    expect(c.busyLevel).toBeNull();
    expect(followJob).not.toHaveBeenCalled();
    c.dispose();
  });

  it('follows a started run, then reloads the overview and the groups', async () => {
    vi.mocked(startNearDupRun).mockResolvedValue({
      job_id: 'j1',
      kind: 'near_dup.build',
    });
    vi.mocked(followJob).mockResolvedValue({ status: 'completed' } as unknown as JobView);
    const c = createNearDupPageController();
    await c.startRun('image', 'full');
    expect(startNearDupRun).toHaveBeenCalledWith(
      { level: 'image', mode: 'full' },
      expect.anything(),
    );
    expect(c.runMessage).toContain('rebuilt');
    expect(getNearDupStatus).toHaveBeenCalledTimes(1);
    expect(listNearDupGroups).toHaveBeenCalledTimes(1);
    c.dispose();
  });

  it('reports a run that does not complete with its job error', async () => {
    vi.mocked(startNearDupRun).mockResolvedValue({ job_id: 'j1', kind: 'k' });
    vi.mocked(followJob).mockResolvedValue({
      status: 'failed',
      error_message: 'boom',
    } as unknown as JobView);
    const c = createNearDupPageController();
    await c.startRun('image', 'incremental');
    expect(c.runError).toBe('image level run failed: boom');
    c.dispose();
  });
});
