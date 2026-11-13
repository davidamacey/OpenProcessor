import { describe, expect, it, vi } from 'vitest';
import { nextStepsStore } from '$stores/nextSteps.svelte';
import { ApiError } from '$lib/api_errors';
import {
  ClusterRunController,
  type ClusterRunDeps,
} from '$lib/clusters/clusterRunController.svelte';
import type { ClusterPartitionView } from '$lib/types_clusterList';
import type { JobView } from '$lib/types_jobs';

function partition(over: Partial<ClusterPartitionView> = {}): ClusterPartitionView {
  return {
    id: 'p1',
    target: 'item',
    level: 'default',
    method: 'ivf',
    embedding_field: 'embedding',
    status: 'ready',
    serving: false,
    parent_partition_id: null,
    n_clusters: 12,
    n_items: 900,
    n_review: 0,
    n_unassigned: 3,
    metrics: {},
    diff: null,
    created_at: '2026-11-08T10:00:00Z',
    ...over,
  };
}

const job = (status: JobView['status'], result: JobView['result'] = null) =>
  ({ id: 'j1', status, result, error_message: null }) as JobView;

function make(over: Partial<ClusterRunDeps> = {}) {
  const deps: ClusterRunDeps = {
    list: vi.fn(async () => ({ partitions: [partition({ id: 's', serving: true })] })),
    start: vi.fn(async () => ({ job_id: 'j1' })),
    activate: vi.fn(async () => ({
      job_id: 'j2',
      pending_partition_id: 'p1',
      revision: 4,
    })),
    follow: vi.fn(async () =>
      job('completed', { n_items: 900, n_clusters: 12, activated: true }),
    ),
    ...over,
  };
  return { deps, ctl: new ClusterRunController('item', deps) };
}

describe('ClusterRunController', () => {
  it('re-reads the next steps after an activation, so the header chip does not go stale', async () => {
    const refresh = vi.spyOn(nextStepsStore, 'refresh').mockResolvedValue();
    const { ctl } = make();
    await ctl.load();
    await ctl.activate(partition({ id: 'new' }));
    expect(refresh).toHaveBeenCalled();
    refresh.mockRestore();
  });

  it('shows the serving partition and the ready one that is not serving', async () => {
    const { ctl } = make({
      list: vi.fn(async () => ({
        partitions: [partition({ id: 'new' }), partition({ id: 'old', serving: true })],
      })),
    });
    await ctl.load();
    expect(ctl.serving?.id).toBe('old');
    expect(ctl.pending?.id).toBe('new');
  });

  it('queues a run for its target and follows the job; says when the server served it', async () => {
    const { ctl, deps } = make();
    await ctl.cluster();
    expect(deps.start).toHaveBeenCalledWith('item');
    expect(deps.follow).toHaveBeenCalledWith('j1', expect.anything());
    expect(ctl.message).toContain('the new partition is serving');
    expect(ctl.busy).toBe(false);
  });

  it('says a finished run that the server did not activate is not serving', async () => {
    const { ctl } = make({
      follow: vi.fn(async () =>
        job('completed', { n_items: 5, n_clusters: 2, activated: false }),
      ),
    });
    await ctl.cluster();
    expect(ctl.message).toContain('not serving');
  });

  it('shows the server refusal of a second run', async () => {
    const { ctl } = make({
      start: vi.fn(async () => {
        throw new ApiError(409, '/x', {
          detail: { error: 'job_busy', message: 'a clustering run is already active' },
        });
      }),
    });
    await ctl.cluster();
    expect(ctl.actionError).toContain('already active');
  });

  it('activates with the serving partition it saw, and shows a stale 409 verbatim', async () => {
    const { ctl, deps } = make({
      list: vi.fn(async () => ({
        partitions: [partition({ id: 'new' }), partition({ id: 'old', serving: true })],
      })),
      activate: vi.fn(async () => {
        throw new ApiError(409, '/x', {
          detail: { error: 'partition_conflict', message: 'the serving partition moved' },
        });
      }),
    });
    await ctl.load();
    await ctl.activate(ctl.pending!);
    expect(deps.activate).toHaveBeenCalledWith({
      partition_id: 'new',
      target: 'item',
      level: 'default',
      expected_serving_partition_id: 'old',
    });
    expect(ctl.actionError).toContain('serving partition moved');
  });

  it('reports a failed job instead of a success', async () => {
    const { ctl } = make({ follow: vi.fn(async () => job('failed')) });
    await ctl.cluster();
    expect(ctl.actionError).toContain('failed');
    expect(ctl.message).toBeNull();
  });
});
