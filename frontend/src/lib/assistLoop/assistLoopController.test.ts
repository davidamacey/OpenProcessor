import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
import { loopDeps, loopNode, loopState } from '$lib/test/fixtures/assistLoop';

let ctl: AssistLoopController;
afterEach(() => ctl?.dispose());

describe('AssistLoopController', () => {
  it('asks the server for a dry run and never starts one on its own', async () => {
    const create = vi.fn(async () => loopState('dry_run'));
    ctl = new AssistLoopController(loopDeps({ create }));
    await ctl.plan();
    expect(create).toHaveBeenCalledTimes(1);
    expect((create.mock.calls[0] as unknown[])[0]).toMatchObject({
      dry_run: true,
      budget: { max_vlm_crops: 2000 },
    });
    expect(ctl.preview?.status).toBe('dry_run');
    expect(ctl.canStart).toBe(true);
  });

  it('shows the served refusal and blocks the start', async () => {
    const create = vi.fn(async () => {
      throw new ApiError(422, '/x', {
        detail: { error: 'no_roots', message: 'no cluster is big enough' },
      });
    });
    ctl = new AssistLoopController(loopDeps({ create }));
    await ctl.plan();
    expect(ctl.planError).toContain('no cluster is big enough');
    expect(ctl.canStart).toBe(false);
  });

  it('refuses a start without a budget', async () => {
    ctl = new AssistLoopController(loopDeps());
    await ctl.plan();
    ctl.maxVlmCrops = null;
    expect(ctl.canStart).toBe(false);
    expect(ctl.request.budget).toBeNull();
  });

  it('starts with dry_run false and follows the loop until it ends', async () => {
    let reads = 0;
    const create = vi.fn(async (body: { dry_run: boolean }) =>
      loopState(body.dry_run ? 'dry_run' : 'running'),
    );
    const get = vi.fn(async () => loopState(++reads < 3 ? 'running' : 'completed'));
    ctl = new AssistLoopController(loopDeps({ create, get }));
    await ctl.plan();
    await ctl.start();
    expect((create.mock.calls.at(-1) as unknown[])[0]).toMatchObject({ dry_run: false });
    expect(ctl.live).toBe(true);
    await vi.waitFor(() => expect(ctl.loop?.status).toBe('completed'));
    expect(get.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it('reports a missing loop service as unsupported, not as an error', async () => {
    const list = vi.fn(async () => {
      throw new ApiError(404, '/x', { detail: 'Not Found' });
    });
    ctl = new AssistLoopController(loopDeps({ list }));
    await ctl.load();
    expect(ctl.supported).toBe(false);
    expect(ctl.loadError).toBeNull();
  });

  it('shows another failure of the list read', async () => {
    const list = vi.fn(async () => {
      throw new ApiError(500, '/x', { detail: 'index down' });
    });
    ctl = new AssistLoopController(loopDeps({ list }));
    await ctl.load();
    expect(ctl.supported).toBe(true);
    expect(ctl.loadError).toContain('index down');
  });

  it('resumes a stopped loop and cancels a live one through the server', async () => {
    const resume = vi.fn(async () => loopState('queued'));
    const cancel = vi.fn(async () => loopState('cancelled'));
    ctl = new AssistLoopController(
      loopDeps({
        list: async () => ({ loops: [loopState('paused', { pause_reason: 'audit' })] }),
        resume,
        cancel,
        get: async () => loopState('queued'),
      }),
    );
    await ctl.load();
    await ctl.resume();
    expect(resume).toHaveBeenCalledWith('loop_abc123');
    expect(ctl.loop?.status).toBe('queued');
    await ctl.cancel();
    expect(cancel).toHaveBeenCalledWith('loop_abc123');
    expect(ctl.loop?.status).toBe('cancelled');
  });

  it('rolls back only after a dry run, with the nodes it was asked about', async () => {
    const rollback = vi.fn(async (_id: string, b: { dry_run: boolean }) => ({
      dry_run: b.dry_run,
      matched: 10,
      restored: b.dry_run ? 9 : 9,
      kept_changed: 1,
      conflicts: 0,
    }));
    ctl = new AssistLoopController(loopDeps({ rollback }));
    await ctl.load();
    await ctl.confirmRollback();
    expect(rollback).not.toHaveBeenCalled();
    await ctl.previewRollback(['r10001']);
    expect(rollback).toHaveBeenLastCalledWith('loop_abc123', {
      node_paths: ['r10001'],
      dry_run: true,
    });
    await ctl.confirmRollback();
    expect(rollback).toHaveBeenLastCalledWith('loop_abc123', {
      node_paths: ['r10001'],
      dry_run: false,
    });
    expect(ctl.rollbackResult?.restored).toBe(9);
    expect(ctl.rollbackPlan).toBeNull();
  });

  it('pages nodes and members with the server cursor', async () => {
    const nodes = vi
      .fn()
      .mockResolvedValueOnce({ nodes: [loopNode()], next_after: 'r10001' })
      .mockResolvedValueOnce({ nodes: [loopNode({ path: 'r10002' })], next_after: null });
    const members = vi.fn(async () => ({ crop_ids: ['a', 'b'], next_after: null }));
    ctl = new AssistLoopController(loopDeps({ nodes, members }));
    await ctl.load();
    expect(ctl.nodesNext).toBe('r10001');
    await ctl.loadNodes(false);
    expect(ctl.nodes.map((n) => n.path)).toEqual(['r10001', 'r10002']);
    expect(nodes.mock.calls[1][1]).toMatchObject({ after: 'r10001' });
    await ctl.toggleNode('r10002');
    expect(ctl.members).toEqual(['a', 'b']);
    await ctl.toggleNode('r10002');
    expect(ctl.openNode).toBeNull();
  });
});
