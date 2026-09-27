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

function open(): void {
  target
    .querySelector<HTMLButtonElement>('[data-testid="project-switcher-trigger"]')!
    .click();
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
  it('shows the active project and lists only served selectable projects', () => {
    render();
    expect(
      target.querySelector('[data-testid="project-switcher-current"]')!.textContent,
    ).toBe('Default');
    open();
    const options = [...target.querySelectorAll('[role="option"]')].map((o) =>
      o.getAttribute('data-testid'),
    );
    expect(options).toEqual([
      'project-option-default',
      'project-option-alpha',
      'project-option-old',
    ]);
  });

  it('badges a non-active status with the served label', () => {
    render();
    open();
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

  it('navigates to the same section under the new slug, dropping crop ids', () => {
    render();
    open();
    target
      .querySelector<HTMLButtonElement>('[data-testid="project-option-alpha"]')!
      .click();
    flushSync();
    expect(goto).toHaveBeenCalledWith('/p/alpha/review?tab=regions');
    expect(target.querySelector('[data-testid="project-switcher-menu"]')).toBeNull();
  });

  it('does not navigate when the active project is picked', () => {
    render();
    open();
    target
      .querySelector<HTMLButtonElement>('[data-testid="project-option-default"]')!
      .click();
    expect(goto).not.toHaveBeenCalled();
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

  it('links to the project management page', () => {
    render();
    open();
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
        return new Response(JSON.stringify({ project: 'default', paused: served }), {
          headers: { 'content-type': 'application/json' },
        });
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
});
