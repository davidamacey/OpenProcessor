import { describe, expect, it, vi } from 'vitest';
import { createClusterAssist } from '$lib/assistRun/clusterAssist.svelte';
import { assistRun, proposal } from '$lib/test/fixtures/assist';
import type { AssistRunState } from '$lib/types_assist';

function view(over: { subTab?: string | null; selected?: string[] } = {}) {
  return {
    clusterId: 10000,
    subTab: over.subTab ?? null,
    sel: { ids: new Set(over.selected ?? []) },
  };
}

const withProposal = (id: string): AssistRunState =>
  assistRun('completed', { run_id: id, result: { proposal: proposal() } });

describe('createClusterAssist', () => {
  it('opens on the whole cluster when nothing is selected', () => {
    const a = createClusterAssist(view(), { list: vi.fn() });
    a.openAssist();
    expect(a.assist.isOpen).toBe(true);
    expect(a.assist.opener?.title).toBe('Cluster #10000');
    expect(a.assist.opener?.selected).toBeUndefined();
    expect(a.assist.opener?.whole).toEqual({
      kind: 'cluster',
      cluster_id: 10000,
      cluster_subid: null,
    });
  });

  it('offers the selection with the (sub-)cluster as its context, and the sub-cluster as the whole', () => {
    const a = createClusterAssist(view({ subTab: '10000b', selected: ['c1', 'c2'] }), {
      list: vi.fn(),
    });
    a.openAssist();
    expect(a.assist.opener?.title).toBe('Sub-cluster 10000b');
    expect(a.assist.opener?.selected).toEqual({
      kind: 'crops',
      crop_ids: ['c1', 'c2'],
      context: { cluster_id: 10000, cluster_subid: '10000b' },
    });
    expect(a.assist.opener?.whole).toEqual({
      kind: 'cluster',
      cluster_id: 10000,
      cluster_subid: '10000b',
    });
    expect(a.assist.choice).toBe('selected');
  });

  it('shows the newest run that has a proposal for the viewed scope', async () => {
    const list = vi.fn(async () => ({
      runs: [assistRun('running', { run_id: 'a' }), withProposal('b'), withProposal('c')],
    }));
    const a = createClusterAssist(view({ subTab: '10000b' }), { list });
    await a.load();
    expect(list).toHaveBeenCalledWith({ clusterId: 10000, clusterSubid: '10000b' });
    expect(a.latest?.run_id).toBe('b');
  });

  it('keeps no proposal when the list is unreadable, and says why', async () => {
    const a = createClusterAssist(view(), {
      list: vi.fn(async () => {
        throw new Error('boom');
      }),
    });
    await a.load();
    expect(a.latest).toBeNull();
    expect(a.loadError).toBe('boom');
  });
});
