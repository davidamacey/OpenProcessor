/**
 * Top-bar switcher: lists the served `selectable` projects only, shows a
 * served status badge for anything not `active`, a "custom keys" badge
 * off the active project's served keymap `is_default`, and picking a
 * project navigates to the same section under the new slug.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

const goto = vi.fn();
vi.mock('$app/navigation', () => ({ goto: (...a: unknown[]) => goto(...a) }));
vi.mock('$app/state', () => ({
  page: { url: new URL('http://x/p/default/review?tab=regions&crop_id=c9') },
}));

const served = vi.hoisted(() => ({ all: [] as unknown[], calls: [] as unknown[] }));
vi.mock('$lib/api_projects', async (orig) => ({
  ...(await orig<typeof import('$lib/api_projects')>()),
  // A stand-in server: filters by `q`, cuts to `limit`, reports `total`, like GET /projects.
  getProjects: vi.fn(
    async (
      _s: unknown,
      _a: boolean,
      f: { q?: string; limit?: number; selectable?: boolean } = {},
    ) => {
      served.calls.push(f);
      const q = (f.q ?? '').toLowerCase();
      const hit = (
        served.all as { slug: string; display_name: string; selectable: boolean }[]
      ).filter(
        (p) =>
          (!f.selectable || p.selectable) &&
          (p.slug.includes(q) || p.display_name.toLowerCase().includes(q)),
      );
      return { projects: hit.slice(0, f.limit ?? hit.length), total: hit.length };
    },
  ),
}));

import ProjectSwitcher from './ProjectSwitcher.svelte';
import { API_PREFIX } from '$lib/api';
import { FALLBACK_KEYMAP } from '$lib/keymapFallback';
import { keymapStore } from '$stores/keymap.svelte';
import { projectPauseStore } from '$stores/projectPause.svelte';
import { projectsStore } from '$stores/projects.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const DEFAULT = testProject({
  slug: 'default',
  prefix: API_PREFIX,
  is_default: true,
  display_name: 'Default',
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProjectSwitcher, { target });
  flushSync();
}

async function open(): Promise<void> {
  target
    .querySelector<HTMLButtonElement>('[data-testid="project-switcher-trigger"]')!
    .click();
  flushSync();
  await vi.waitFor(() => {
    expect(
      target.querySelector('[data-testid="project-switcher-filter"]'),
    ).not.toBeNull();
  });
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

beforeEach(() => {
  goto.mockClear();
  const res = testProjectsResponse([
    DEFAULT,
    testProject({ slug: 'alpha', display_name: 'Alpha' }),
    testProject({
      slug: 'wip',
      display_name: 'Being built',
      status: 'building',
      selectable: false,
    }),
    testProject({
      slug: 'old',
      display_name: 'Old one',
      status: 'archived',
      writable: false,
    }),
  ]);
  projectsStore.list = res.projects;
  served.all = res.projects;
  served.calls = [];
  projectsStore.labels = res.labels!;
  projectsStore.select(DEFAULT);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  keymapStore.resetToFallback();
  projectPauseStore.reset();
  vi.unstubAllGlobals();
  projectsStore.select(DEFAULT);
});

describe('ProjectSwitcher', () => {
  it('shows the active project and lists only served selectable projects', async () => {
    render();
    expect(
      target.querySelector('[data-testid="project-switcher-current"]')!.textContent,
    ).toBe('Default');
    await open();
    const options = [...target.querySelectorAll('[role="option"]')].map((o) =>
      o.getAttribute('data-testid'),
    );
    expect(options).toEqual([
      'project-option-default',
      'project-option-alpha',
      'project-option-old',
    ]);
  });

  it('badges a non-active status with the served label', async () => {
    render();
    await open();
    const old = target.querySelector('[data-testid="project-option-old"]')!;
    expect(old.textContent).toContain('Archived');
    const alpha = target.querySelector('[data-testid="project-option-alpha"]')!;
    expect(alpha.textContent).not.toContain('Active');
  });

  it('shows the active non-active status on the trigger too', () => {
    projectsStore.select(projectsStore.list[3]!);
    render();
    expect(
      target.querySelector('[data-testid="project-switcher-status"]')!.textContent,
    ).toBe('Archived');
  });

  it('navigates to the same section under the new slug, dropping crop ids', async () => {
    render();
    await open();
    target
      .querySelector<HTMLButtonElement>('[data-testid="project-option-alpha"]')!
      .click();
    flushSync();
    expect(goto).toHaveBeenCalledWith('/p/alpha/review?tab=regions');
    expect(target.querySelector('[data-testid="project-switcher-menu"]')).toBeNull();
  });

  it('does not navigate when the active project is picked', async () => {
    render();
    await open();
    target
      .querySelector<HTMLButtonElement>('[data-testid="project-option-default"]')!
      .click();
    expect(goto).not.toHaveBeenCalled();
  });

  it('asks the server for one bounded page and filters there as the operator types', async () => {
    served.all = [
      ...(served.all as object[]),
      ...Array.from({ length: 40 }, (_, i) =>
        testProject({ slug: `cars-${i}`, display_name: `Cars ${i}` }),
      ),
    ];
    render();
    await open();
    expect(served.calls[0]).toEqual({ q: '', limit: 25, selectable: true });
    const list = target.querySelector('[data-testid="project-switcher-list"]')!;
    expect(list.className).toContain('overflow-y-auto');
    expect(list.className).toContain('max-h-');
    expect(list.querySelectorAll('[role="option"]').length).toBe(25);
    expect(
      target.querySelector('[data-testid="project-switcher-more"]')!.textContent,
    ).toContain('of 43');

    const input = target.querySelector<HTMLInputElement>(
      '[data-testid="project-switcher-filter"]',
    )!;
    input.value = 'alp';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await vi.waitFor(() =>
      expect(served.calls.at(-1)).toEqual({ q: 'alp', limit: 25, selectable: true }),
    );
    await Promise.resolve();
    flushSync();
    expect([...list.querySelectorAll('[role="option"]')].map((o) => o.id)).toEqual([
      'project-option-alpha',
    ]);
  });

  it('is a combobox over a listbox: arrows move the active option, Enter picks it', async () => {
    render();
    await open();
    const input = target.querySelector<HTMLInputElement>(
      '[data-testid="project-switcher-filter"]',
    )!;
    expect(input.getAttribute('role')).toBe('combobox');
    expect(input.getAttribute('aria-controls')).toBe('project-switcher-listbox');
    expect(input.getAttribute('aria-activedescendant')).toBe('project-option-default');
    const key = (k: string) => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
      flushSync();
    };
    key('ArrowDown');
    expect(input.getAttribute('aria-activedescendant')).toBe('project-option-alpha');
    expect(
      target.querySelector('#project-option-alpha')!.getAttribute('aria-selected'),
    ).toBe('true');
    key('End');
    expect(input.getAttribute('aria-activedescendant')).toBe('project-option-old');
    key('Home');
    key('ArrowDown');
    key('Enter');
    expect(goto).toHaveBeenCalledWith('/p/alpha/review?tab=regions');
  });

  it('shows "custom keys" only for a served non-default keymap', () => {
    render();
    expect(
      target.querySelector('[data-testid="project-switcher-custom-keys"]'),
    ).toBeNull();
    keymapStore.setDocument({ ...FALLBACK_KEYMAP, is_default: false }, 'served');
    flushSync();
    expect(
      target.querySelector('[data-testid="project-switcher-custom-keys"]'),
    ).not.toBeNull();
  });

  it('links to the project management page', async () => {
    render();
    await open();
    expect(
      target
        .querySelector('[data-testid="project-switcher-manage"]')!
        .getAttribute('href'),
    ).toBe('/projects');
  });

  it('shows a "paused" chip only when the active project\'s served flag is true', async () => {
    let served = true;
    const seen: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        seen.push(String(url));
        return new Response(
          JSON.stringify({
            project: 'default',
            paused: served,
            paused_by: served ? ['project'] : [],
            reason: null,
          }),
          { headers: { 'content-type': 'application/json' } },
        );
      }),
    );
    render();
    const chip = () => target.querySelector('[data-testid="project-switcher-paused"]');
    expect(chip()).toBeNull();
    await projectPauseStore.load(DEFAULT);
    flushSync();
    expect(chip()?.textContent).toBe('paused');
    expect(seen).toEqual([`${DEFAULT.prefix}/pause`]);
    // Another project being paused never marks the active one.
    served = false;
    await projectPauseStore.load(DEFAULT);
    flushSync();
    expect(chip()).toBeNull();
  });

  it("the chip reads the served summary's paused before the pause read lands", () => {
    projectsStore.list = projectsStore.list.map((p) =>
      p.slug === 'default' ? { ...p, paused: true } : p,
    );
    projectsStore.select(projectsStore.list.find((p) => p.slug === 'default')!);
    render();
    expect(
      target.querySelector('[data-testid="project-switcher-paused"]')?.textContent,
    ).toBe('paused');
  });

  it('a global GPU-training pause shows the served paused_by and reason, verbatim', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              project: 'default',
              paused: true,
              paused_by: ['gpu_training'],
              reason: 'GPU training claim active (cuda_visible_devices=0)',
            }),
            { headers: { 'content-type': 'application/json' } },
          ),
      ),
    );
    render();
    await projectPauseStore.load(DEFAULT);
    flushSync();
    const title = target
      .querySelector('[data-testid="project-switcher-paused"]')!
      .getAttribute('title')!;
    expect(title).toContain('Paused by: gpu_training');
    expect(title).toContain('GPU training claim active (cuda_visible_devices=0)');
  });
});
