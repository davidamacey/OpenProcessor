/**
 * `projectsStore`: slug resolution against the served list, and what a
 * change of active project does (moves `scoped()`, runs every
 * registered reset hook — the undo stack among them — exactly once).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, scoped } from '$lib/api';
import { onProjectChange } from '$lib/projectChange';
import { projectsStore } from '$stores/projects.svelte';
import { undoStore } from '$stores/undo.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const DEFAULT = testProject({ slug: 'default', prefix: API_PREFIX, is_default: true });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  projectsStore.list = [DEFAULT, testProject({ slug: 'alpha' })];
  projectsStore.defaultSlug = 'default';
});

afterEach(() => {
  vi.unstubAllGlobals();
  projectsStore.select(DEFAULT);
});

describe('resolve', () => {
  it('finds a listed, selectable project without a request', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const r = await projectsStore.resolve('alpha');
    expect(r.kind).toBe('ok');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('reports a served non-selectable project as unavailable, not ok', async () => {
    projectsStore.list = [
      ...projectsStore.list,
      testProject({
        slug: 'wip',
        status: 'building',
        selectable: false,
        writable: false,
      }),
    ];
    const r = await projectsStore.resolve('wip');
    expect(r).toMatchObject({ kind: 'unavailable', project: { slug: 'wip' } });
  });

  it('reads an unlisted slug (e.g. archived) from GET /projects/{slug}', async () => {
    const archived = testProject({ slug: 'old', status: 'archived', writable: false });
    const fetchMock = vi.fn().mockResolvedValue(json({ ...archived, resources: {} }));
    vi.stubGlobal('fetch', fetchMock);
    const r = await projectsStore.resolve('old');
    expect(String(fetchMock.mock.calls[0]![0])).toContain(`${API_PREFIX}/projects/old`);
    expect(r).toMatchObject({ kind: 'ok', project: { slug: 'old', writable: false } });
  });

  it('reports a 404 as not found', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          json({ detail: { error: 'project_not_found', message: 'no project' } }, 404),
        ),
    );
    expect(await projectsStore.resolve('ghost')).toEqual({
      kind: 'not_found',
      slug: 'ghost',
    });
  });
});

describe('select', () => {
  it('moves scoped() to the served prefix', () => {
    const alpha = projectsStore.list[1]!;
    projectsStore.select(alpha);
    expect(scoped()).toBe(alpha.prefix);
    expect(projectsStore.current?.slug).toBe('alpha');
  });

  it('runs the reset hooks once on a change of project, and clears the undo stack', () => {
    const hook = vi.fn();
    const off = onProjectChange(hook);
    undoStore.stack = [{ kind: 'label', crops: [] } as never];
    const gen = projectsStore.generation;

    expect(projectsStore.select(projectsStore.list[1]!)).toBe(true);
    expect(hook).toHaveBeenCalledTimes(1);
    expect(undoStore.stack).toEqual([]);
    expect(projectsStore.generation).toBe(gen + 1);
    off();
  });

  it('runs no reset for a fresh summary of the same project', () => {
    const hook = vi.fn();
    const off = onProjectChange(hook);
    undoStore.stack = [{ kind: 'label', crops: [] } as never];
    expect(projectsStore.select({ ...DEFAULT, revision: 99 })).toBe(false);
    expect(hook).not.toHaveBeenCalled();
    expect(undoStore.stack).toHaveLength(1);
    expect(projectsStore.current?.revision).toBe(99);
    off();
    undoStore.stack = [];
  });

  it('runs no reset for the very first selection', () => {
    const hook = vi.fn();
    const off = onProjectChange(hook);
    projectsStore.current = null;
    projectsStore.select(DEFAULT);
    expect(hook).not.toHaveBeenCalled();
    off();
  });
});

describe('defaultProject', () => {
  it('is the served default_slug when selectable', () => {
    expect(projectsStore.defaultProject?.slug).toBe('default');
  });

  it('falls back to the first selectable project', () => {
    projectsStore.list = [
      { ...DEFAULT, selectable: false, status: 'deleting' },
      testProject({ slug: 'alpha' }),
    ];
    expect(projectsStore.defaultProject?.slug).toBe('alpha');
  });

  it('is null with nothing selectable', () => {
    projectsStore.list = [{ ...DEFAULT, selectable: false }];
    expect(projectsStore.defaultProject).toBeNull();
  });
});

describe('load / adopt / selectable', () => {
  it('load() reads the global list and never touches scoped()', async () => {
    const res = testProjectsResponse([DEFAULT, testProject({ slug: 'beta' })]);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(res)));
    projectsStore.loaded = false;
    const before = scoped();
    await projectsStore.load();
    expect(projectsStore.list.map((p) => p.slug)).toEqual(['default', 'beta']);
    expect(projectsStore.statusLabel('archived')).toBe('Archived');
    expect(scoped()).toBe(before);
  });

  it('adopt() upserts a lifecycle summary', () => {
    projectsStore.adopt(testProject({ slug: 'new' }));
    projectsStore.adopt({ ...projectsStore.list[1]!, display_name: 'Renamed' });
    expect(projectsStore.list.map((p) => p.slug)).toEqual(['default', 'alpha', 'new']);
    expect(projectsStore.list[1]!.display_name).toBe('Renamed');
  });

  it('selectable lists served selectable projects plus the active one', () => {
    const archived = testProject({ slug: 'old', status: 'archived', writable: false });
    projectsStore.list = [
      DEFAULT,
      testProject({ slug: 'wip', selectable: false, status: 'building' }),
    ];
    projectsStore.select(archived);
    expect(projectsStore.selectable.map((p) => p.slug)).toEqual(['default', 'old']);
  });
});
