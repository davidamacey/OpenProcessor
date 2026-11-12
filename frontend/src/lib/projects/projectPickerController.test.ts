import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const calls = vi.hoisted(() => [] as Record<string, unknown>[]);
const pages = vi.hoisted(() => ({ next: [] as unknown[] }));
vi.mock('$lib/api_projects', () => ({
  getProjects: vi.fn(async (_s: unknown, _a: boolean, f: Record<string, unknown>) => {
    calls.push(f);
    return pages.next.shift();
  }),
}));

import { PICKER_PAGE, ProjectPickerController } from './projectPickerController.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const names = (c: ProjectPickerController) => c.projects.map((p) => p.slug);

describe('ProjectPickerController', () => {
  beforeEach(() => {
    calls.length = 0;
    sessionStorage.clear();
  });
  afterEach(() => vi.useRealTimers());

  it('sends selectable and shows the served total', async () => {
    pages.next = [
      testProjectsResponse([testProject({ slug: 'aa' }), testProject({ slug: 'bb' })], {
        total: 60,
        next_cursor: 'c1',
      }),
    ];
    const c = new ProjectPickerController();
    c.open();
    await vi.waitFor(() => expect(c.loading).toBe(false));
    expect(calls[0]).toMatchObject({ selectable: true, limit: PICKER_PAGE });
    expect(c.total).toBe(60);
    expect(c.hasMore).toBe(true);
  });

  it('appends the next page with the served cursor', async () => {
    pages.next = [
      testProjectsResponse([testProject({ slug: 'aa' })], {
        total: 2,
        next_cursor: 'c1',
      }),
      testProjectsResponse([testProject({ slug: 'bb' })], {
        total: 2,
        next_cursor: null,
      }),
    ];
    const c = new ProjectPickerController();
    c.open();
    await vi.waitFor(() => expect(c.loading).toBe(false));
    await c.more();
    expect(calls[1]).toMatchObject({ cursor: 'c1' });
    expect(names(c)).toEqual(['aa', 'bb']);
    expect(c.hasMore).toBe(false);
  });

  it('remembers recent projects for the session only', () => {
    const c = new ProjectPickerController();
    c.remember(testProject({ slug: 'aa', display_name: 'A' }));
    c.remember(testProject({ slug: 'bb', display_name: 'B' }));
    c.remember(testProject({ slug: 'aa', display_name: 'A' }));
    expect(new ProjectPickerController().recents.map((r) => r.slug)).toEqual([
      'aa',
      'bb',
    ]);
  });
});
