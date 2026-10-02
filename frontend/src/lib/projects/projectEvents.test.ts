/**
 * A global `project.paused` / `project.resumed` event (another tab's
 * action, or the GPU claim changing) re-reads the served list and the
 * target project's own `GET {prefix}/pause`; every other `project.*`
 * event only re-reads the list.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { handleProjectEvent } from './projectEvents';
import { subscribeGlobalEvents, type ProjectEvent } from '$lib/sse';
import { projectPauseStore } from '$stores/projectPause.svelte';
import { projectsStore } from '$stores/projects.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const DEFAULT = testProject({ slug: 'default', prefix: API_PREFIX, is_default: true });
const ALPHA = testProject({ slug: 'alpha', prefix: '/curation/projects/alpha-served' });

let urls: string[];
let alphaPaused: boolean;

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  urls = [];
  alphaPaused = true;
  projectPauseStore.reset();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      urls.push(String(url));
      if (String(url).endsWith('/pause')) {
        return json({
          project: 'alpha',
          paused: alphaPaused,
          paused_by: alphaPaused ? ['gpu_training'] : [],
          reason: alphaPaused ? 'GPU training claim active' : null,
        });
      }
      return json(testProjectsResponse([DEFAULT, { ...ALPHA, paused: alphaPaused }]));
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  projectPauseStore.reset();
});

const event = (type: string, target?: string): ProjectEvent => ({
  type,
  topic: 'project',
  project: null,
  target,
});

describe('handleProjectEvent', () => {
  it('project.paused re-reads the list and the target project own pause state', async () => {
    await handleProjectEvent(event('project.paused', 'alpha'));
    expect(urls).toEqual([
      `${API_PREFIX}/projects`,
      '/curation/projects/alpha-served/pause',
    ]);
    expect(projectsStore.list.find((p) => p.slug === 'alpha')?.paused).toBe(true);
    expect(projectPauseStore.stateFor('alpha')?.paused_by).toEqual(['gpu_training']);
  });

  it('project.resumed refreshes the same two reads and clears the paused state', async () => {
    await handleProjectEvent(event('project.paused', 'alpha'));
    alphaPaused = false;
    await handleProjectEvent(event('project.resumed', 'alpha'));
    expect(projectPauseStore.pausedFor('alpha')).toBe(false);
    expect(projectsStore.list.find((p) => p.slug === 'alpha')?.paused).toBe(false);
  });

  it('another project.* event only re-reads the list', async () => {
    await handleProjectEvent(event('project.archived', 'alpha'));
    expect(urls).toEqual([`${API_PREFIX}/projects`]);
  });

  it('a pause event for a slug the list does not carry reads no pause state', async () => {
    await handleProjectEvent(event('project.paused', 'ghost'));
    expect(urls).toEqual([`${API_PREFIX}/projects`]);
  });
});

describe('the global stream subscribes to the pause events', () => {
  it('registers a listener for project.paused and project.resumed (an unlisted type is never dispatched)', () => {
    const types: string[] = [];
    class FakeEventSource {
      onopen: (() => void) | null = null;
      onerror: ((e: unknown) => void) | null = null;
      constructor(public url: string) {}
      addEventListener(type: string): void {
        types.push(type);
      }
      close(): void {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);
    const sub = subscribeGlobalEvents({ onEvent: () => {} });
    sub.close();
    expect(types).toContain('project.paused');
    expect(types).toContain('project.resumed');
    expect(types).toContain('project.created');
  });
});
