/**
 * `projectPauseStore`: holds exactly the served pause state per slug, reads
 * and writes through each project's own served prefix, and forgets a
 * value whose read failed (no stale chip).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { projectPauseStore } from './projectPause.svelte';
import { testProject } from '$lib/test/fixtures/projects';

const ALPHA = testProject({ slug: 'alpha', prefix: '/curation/projects/alpha-served' });

let calls: { method: string; url: string }[] = [];
let respond: (url: string, method: string) => Response;

function state(over: { paused: boolean; paused_by?: string[]; reason?: string | null }) {
  return json({
    project: 'alpha',
    paused_by: over.paused ? ['project'] : [],
    reason: null,
    ...over,
  });
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  calls = [];
  projectPauseStore.reset();
  respond = () => state({ paused: false });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const method = init.method ?? 'GET';
      calls.push({ method, url: String(url) });
      return respond(String(url), method);
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  projectPauseStore.reset();
});

describe('projectPauseStore', () => {
  it('is undefined until loaded, then the served flag', async () => {
    respond = () => state({ paused: true });
    expect(projectPauseStore.pausedFor('alpha')).toBeUndefined();
    await projectPauseStore.load(ALPHA);
    expect(projectPauseStore.pausedFor('alpha')).toBe(true);
    // The served prefix verbatim, not a path assembled from the slug.
    expect(calls).toEqual([
      { method: 'GET', url: '/curation/projects/alpha-served/pause' },
    ]);
  });

  it('a failed read forgets the previous value', async () => {
    respond = () => state({ paused: true });
    await projectPauseStore.load(ALPHA);
    respond = () => json({ detail: 'nope' }, 404);
    await projectPauseStore.load(ALPHA);
    expect(projectPauseStore.pausedFor('alpha')).toBeUndefined();
  });

  it('pause and resume POST the served prefix and store the served answer', async () => {
    respond = (url) => state({ paused: url.endsWith('/pause') });
    expect(await projectPauseStore.set(ALPHA, true)).toEqual({ ok: true, paused: true });
    expect(projectPauseStore.pausedFor('alpha')).toBe(true);
    expect(await projectPauseStore.set(ALPHA, false)).toEqual({
      ok: true,
      paused: false,
    });
    expect(projectPauseStore.pausedFor('alpha')).toBe(false);
    expect(calls).toEqual([
      { method: 'POST', url: '/curation/projects/alpha-served/pause' },
      { method: 'POST', url: '/curation/projects/alpha-served/resume' },
    ]);
  });

  it('stores what the server answered, not what was asked', async () => {
    // e.g. a server that refuses to flip the flag but answers 200.
    respond = () => state({ paused: false });
    expect(await projectPauseStore.set(ALPHA, true)).toEqual({ ok: true, paused: false });
    expect(projectPauseStore.pausedFor('alpha')).toBe(false);
  });

  it('a refusal returns the served message and leaves the flag alone', async () => {
    respond = () => state({ paused: false });
    await projectPauseStore.load(ALPHA);
    respond = () =>
      json({ detail: { error: 'project_read_only', message: 'served refusal' } }, 409);
    expect(await projectPauseStore.set(ALPHA, true)).toEqual({
      ok: false,
      message: 'served refusal',
    });
    expect(projectPauseStore.pausedFor('alpha')).toBe(false);
  });

  it('a late read never overwrites a newer write', async () => {
    let releaseRead: (r: Response) => void = () => {};
    respond = (url, method) =>
      method === 'GET'
        ? (new Promise<Response>((r) => (releaseRead = r)) as unknown as Response)
        : state({ paused: true });
    const read = projectPauseStore.load(ALPHA);
    await projectPauseStore.set(ALPHA, true);
    releaseRead(state({ paused: false }));
    await read;
    expect(projectPauseStore.pausedFor('alpha')).toBe(true);
  });

  it('keeps the served paused_by and reason (a global GPU-training claim) verbatim', async () => {
    respond = () =>
      state({
        paused: true,
        paused_by: ['gpu_training'],
        reason: "GPU training claim active (cuda_visible_devices='0')",
      });
    await projectPauseStore.load(ALPHA);
    expect(projectPauseStore.stateFor('alpha')).toEqual({
      project: 'alpha',
      paused: true,
      paused_by: ['gpu_training'],
      reason: "GPU training claim active (cuda_visible_devices='0')",
    });
    expect(projectPauseStore.pausedFor('alpha')).toBe(true);
  });

  it('a write stores the served paused_by too', async () => {
    respond = () => state({ paused: true, paused_by: ['project', 'gpu_training'] });
    await projectPauseStore.set(ALPHA, true);
    expect(projectPauseStore.stateFor('alpha')?.paused_by).toEqual([
      'project',
      'gpu_training',
    ]);
  });
});
