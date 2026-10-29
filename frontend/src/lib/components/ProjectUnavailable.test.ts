/**
 * A project that is still `building` opens by itself once the server serves
 * it as selectable: the page re-reads the project list while it waits and
 * re-runs the route load when the project becomes selectable.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

const invalidateAll = vi.fn();
vi.mock('$app/navigation', () => ({ invalidateAll: () => invalidateAll() }));

import ProjectUnavailable from './ProjectUnavailable.svelte';
import { projectsStore } from '$stores/projects.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const BUILDING = testProject({ slug: 'wip', status: 'building', selectable: false });
const READY = testProject({ slug: 'wip', status: 'active', selectable: true });

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

function render(status: string, selectable: boolean): void {
  const project = testProject({ slug: 'wip', status, selectable });
  projectsStore.list = [project];
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProjectUnavailable, {
    target,
    props: { resolution: { kind: 'unavailable', project } },
  });
  flushSync();
}

beforeEach(() => {
  vi.useFakeTimers();
  invalidateAll.mockClear();
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('ProjectUnavailable', () => {
  it('re-runs the route load once a building project is served selectable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(JSON.stringify(testProjectsResponse([READY])), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          }),
      ),
    );
    render(BUILDING.status, false);
    expect(invalidateAll).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(3500);
    flushSync();
    expect(invalidateAll).toHaveBeenCalled();
  });

  it('does not poll for a project that is not building', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    render('failed', false);
    await vi.advanceTimersByTimeAsync(10000);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
