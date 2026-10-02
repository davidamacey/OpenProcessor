/**
 * `projectPauseStore`: holds exactly the served `paused` per slug, reads
 * and writes through each project's own served prefix, and forgets a
 * value whose read failed (no stale chip).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { projectPauseStore } from './projectPause.svelte';
import { testProject } from '$lib/test/fixtures/projects';

const ALPHA = testProject({ slug: 'alpha', prefix: '/curation/projects/alpha-served' });

let calls: { method: string; url: string }[] = [];
let respond: (url: string, method: string) => Response;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  calls = [];
  projectPauseStore.reset();
  respond = () => json({ project: 'alpha', paused: false });
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
    respond = () => json({ project: 'alpha', paused: true });
    expect(projectPauseStore.pausedFor('alpha')).toBeUndefined();
    await projectPauseStore.load(ALPHA);
    expect(projectPauseStore.pausedFor('alpha')).toBe(true);
    // The served prefix verbatim, not a path assembled from the slug.
    expect(calls).toEqual([
      { method: 'GET', url: '/curation/projects/alpha-served/pause' },
    ]);
  });

  it('a failed read forgets the previous value', async () => {
    respond = () => json({ project: 'alpha', paused: true });
    await projectPauseStore.load(ALPHA);
    respond = () => json({ detail: 'nope' }, 404);
    await projectPauseStore.load(ALPHA);
    expect(projectPauseStore.pausedFor('alpha')).toBeUndefined();
  });

  it('pause and resume POST the served prefix and store the served answer', async () => {
    respond = (url) => json({ project: 'alpha', paused: url.endsWith('/pause') });
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
    respond = () => json({ project: 'alpha', paused: false });
    expect(await projectPauseStore.set(ALPHA, true)).toEqual({ ok: true, paused: false });
    expect(projectPauseStore.pausedFor('alpha')).toBe(false);
  });

  it('a refusal returns the served message and leaves the flag alone', async () => {
    respond = () => json({ project: 'alpha', paused: false });
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
        : json({ project: 'alpha', paused: true });
    const read = projectPauseStore.load(ALPHA);
    await projectPauseStore.set(ALPHA, true);
    releaseRead(json({ project: 'alpha', paused: false }));
    await read;
    expect(projectPauseStore.pausedFor('alpha')).toBe(true);
  });
});
