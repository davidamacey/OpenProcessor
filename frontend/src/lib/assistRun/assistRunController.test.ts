import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import {
  AssistRunController,
  type AssistOpener,
  type AssistRunDeps,
} from '$lib/assistRun/assistRunController.svelte';
import { assistRun, proposal } from '$lib/test/fixtures/assist';
import type { AssistRunRequest, AssistRunState } from '$lib/types_assist';

const CLUSTER_OPENER: AssistOpener = {
  title: 'Sub-cluster 10000b',
  selected: {
    kind: 'crops',
    crop_ids: ['c1', 'c2'],
    context: { cluster_id: 10000, cluster_subid: '10000b' },
  },
  whole: { kind: 'cluster', cluster_id: 10000, cluster_subid: '10000b' },
};

function deps(over: Partial<AssistRunDeps> = {}): AssistRunDeps {
  return {
    create: vi.fn(async (body: AssistRunRequest) =>
      assistRun(body.dry_run === false ? 'queued' : 'dry_run'),
    ),
    get: vi.fn(async () => assistRun('completed')),
    cancel: vi.fn(async () => assistRun('cancelled')),
    debounceMs: 0,
    pollMs: 0,
    ...over,
  };
}

describe('AssistRunController request', () => {
  it('starts on the selection when there is one and puts the context on it', () => {
    const c = new AssistRunController(deps());
    c.open(CLUSTER_OPENER);
    expect(c.choice).toBe('selected');
    expect(c.request.target).toEqual(CLUSTER_OPENER.selected);
    expect(c.request.sample).toEqual({ mode: 'all' });
    expect(c.request.actions).toEqual(['classify']);
    expect(c.request.dry_run).toBe(true);
  });

  it('maps the sample choices onto the wire: all, then a seeded representative sample', () => {
    const c = new AssistRunController(deps());
    c.open(CLUSTER_OPENER);
    c.setChoice('all');
    expect(c.request.target).toEqual(CLUSTER_OPENER.whole);
    expect(c.request.sample).toEqual({ mode: 'all' });
    c.setChoice('representative');
    c.size = 20;
    c.seed = 7;
    expect(c.request.sample).toEqual({
      mode: 'representative',
      size: 20,
      seed: 7,
      stratify_by: 'auto',
    });
    c.size = null;
    expect(c.request.sample).toMatchObject({ mode: 'representative', size: null });
  });

  it('offers a sample choice only for the scopes the opener gave', () => {
    const c = new AssistRunController(deps());
    c.open({ title: 'This crop', selected: { kind: 'crops', crop_ids: ['c1'] } });
    expect(c.choices).toEqual(['selected']);
    c.open({ title: 'Cluster', whole: CLUSTER_OPENER.whole });
    expect(c.choices).toEqual(['all', 'representative']);
    expect(c.choice).toBe('all');
    c.open(CLUSTER_OPENER);
    expect(c.choices).toEqual(['selected', 'all', 'representative']);
  });

  it('offers only the actions the target kind takes, and defaults a box run to verify', () => {
    const c = new AssistRunController(deps());
    c.open({
      title: 'One box',
      selected: { kind: 'boxes', boxes: [{ crop_id: 'c1', box_id: 'b1' }] },
    });
    expect(c.allowedActions).toEqual(['attributes', 'verify']);
    expect(c.request.actions).toEqual(['verify']);
    c.open(CLUSTER_OPENER);
    c.setChoice('all');
    expect(c.allowedActions).toEqual(['classify', 'attributes', 'segment']);
  });

  it('sends the segment spec only with the segment action, and judges new boxes in a later step when verify is also ticked', () => {
    const c = new AssistRunController(deps());
    c.open({ title: 'Crops', selected: { kind: 'crops', crop_ids: ['c1'] } });
    expect(c.request.segment).toBeUndefined();
    c.toggleAction('segment');
    expect(c.request.segment).toEqual({ verify: 'vlm' });
    c.toggleAction('verify');
    expect(c.request.segment).toEqual({ verify: 'none' });
    c.toggleAction('segment');
    expect(c.request.segment).toBeUndefined();
  });

  it('never unticks the last action', () => {
    const c = new AssistRunController(deps());
    c.open(CLUSTER_OPENER);
    c.toggleAction('classify');
    expect(c.request.actions).toEqual(['classify']);
  });

  it('sends the endpoint and pack choice, and the external acknowledgement only when true', () => {
    const c = new AssistRunController(deps());
    c.open(CLUSTER_OPENER);
    expect(c.request.vlm).toBeUndefined();
    expect(c.request.acknowledge_external).toBeUndefined();
    c.vlm = 'lab-vlm';
    c.acknowledgeExternal = true;
    c.promptPack = 'cars';
    expect(c.request).toMatchObject({
      vlm: 'lab-vlm',
      acknowledge_external: true,
      prompt_pack: 'cars',
    });
  });
});

describe('AssistRunController dry run', () => {
  it('posts a dry run and keeps the served counts', async () => {
    const d = deps();
    const c = new AssistRunController(d);
    c.open(CLUSTER_OPENER);
    c.setChoice('representative');
    await c.plan();
    const body = vi.mocked(d.create).mock.calls[0]![0];
    expect(body.dry_run).toBe(true);
    expect(c.preview?.resolved.sampled).toBe(41);
    expect(c.preview?.resolved.estimated_vlm_calls).toBe(6);
    expect(c.planError).toBeNull();
  });

  it('shows the served refusal and its code, and clears the stale plan', async () => {
    const d = deps();
    const c = new AssistRunController(d);
    c.open(CLUSTER_OPENER);
    await c.plan();
    expect(c.preview).not.toBeNull();
    vi.mocked(d.create).mockRejectedValueOnce(
      new ApiError(409, 'u', {
        detail: { error: 'vlm_not_configured', message: 'no VLM is active' },
      }),
    );
    await c.plan();
    expect(c.planError).toBe('no VLM is active');
    expect(c.planErrorCode).toBe('vlm_not_configured');
    expect(c.preview).toBeNull();
    expect(c.canStart).toBe(false);
  });

  it('keeps only the newest plan when answers arrive out of order', async () => {
    let release!: (s: AssistRunState) => void;
    const slow = new Promise<AssistRunState>((r) => (release = r));
    const d = deps();
    vi.mocked(d.create)
      .mockReturnValueOnce(slow)
      .mockResolvedValueOnce(
        assistRun('dry_run', {
          resolved: { ...assistRun().resolved, sampled: 5 },
        }),
      );
    const c = new AssistRunController(d);
    c.open(CLUSTER_OPENER);
    const first = c.plan();
    await c.plan();
    release(assistRun('dry_run', { resolved: { ...assistRun().resolved, sampled: 99 } }));
    await first;
    expect(c.preview?.resolved.sampled).toBe(5);
  });
});

describe('AssistRunController run', () => {
  it('starts with dry_run false, then polls the run it started until it ends', async () => {
    const seen: string[] = [];
    const d = deps({
      get: vi
        .fn()
        .mockResolvedValueOnce(
          assistRun('running', { progress: { processed: 10, total: 41 } }),
        )
        .mockResolvedValueOnce(assistRun('waiting_region'))
        .mockResolvedValueOnce(
          assistRun('completed', { result: { proposal: proposal() } }),
        ),
    });
    const c = new AssistRunController(d);
    c.open(CLUSTER_OPENER);
    await c.plan();
    const done = c.start();
    await done;
    expect(vi.mocked(d.create).mock.calls.at(-1)![0].dry_run).toBe(false);
    expect(vi.mocked(d.get)).toHaveBeenCalledTimes(3);
    expect(vi.mocked(d.get).mock.calls[0]![0]).toBe('asr_1');
    seen.push(c.run!.status);
    expect(seen).toEqual(['completed']);
    expect(c.run?.result?.proposal?.class_name).toBe('widget_a');
    expect(c.running).toBe(false);
  });

  it('shows a refusal at start and stays on the configuration', async () => {
    const d = deps();
    const c = new AssistRunController(d);
    c.open(CLUSTER_OPENER);
    await c.plan();
    vi.mocked(d.create).mockRejectedValueOnce(
      new ApiError(409, 'u', {
        detail: { error: 'assist_busy', message: 'another assist run is live' },
      }),
    );
    await c.start();
    expect(c.run).toBeNull();
    expect(c.startError).toBe('another assist run is live');
  });

  it('cancels the live run', async () => {
    const d = deps({
      get: vi.fn(async () => assistRun('running')),
    });
    const c = new AssistRunController(d);
    c.open(CLUSTER_OPENER);
    await c.plan();
    const done = c.start();
    await vi.waitFor(() => expect(c.run).not.toBeNull());
    await c.cancelRun();
    expect(vi.mocked(d.cancel)).toHaveBeenCalledWith('asr_1');
    await done;
    expect(c.run?.status).toBe('cancelled');
  });
});
