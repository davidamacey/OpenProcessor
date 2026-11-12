import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { mountUi } from '$lib/test/mountUi';
import { curationSettingsStore } from '$stores/curationSettings.svelte';

const state = vi.hoisted(() => ({
  page: { url: new URL('http://x/p/default/dashboard') },
}));
vi.mock('$app/state', () => state);

import TopNav from './TopNav.svelte';

function render(path: string): HTMLElement {
  state.page.url = new URL(`http://x${path}`);
  return mountUi(TopNav, { apiBase: '', appName: 'Cropwright', appBadge: 'CW' });
}
const primary = (el: HTMLElement) => el.querySelector('[data-testid="primary-nav"]')!;

beforeEach(() => {
  curationSettingsStore.loaded = true; // ResourcesMenu's init() is then a no-op: no request
});

describe('TopNav', () => {
  it('is a 48px bar on the nav surface with elevation 2 and a backdrop blur', () => {
    const header = render('/p/default/dashboard').querySelector('header')!;
    expect(header.className).toContain('h-12');
    expect(header.className).toContain('bg-surface-nav');
    expect(header.className).toContain('shadow-e2');
    expect(header.className).toContain('backdrop-blur');
  });

  it('groups the nav: Dashboard, Data and Model menus, Quality, Settings', () => {
    const nav = primary(render('/p/default/dashboard'));
    const links = [...nav.querySelectorAll('a')].map((a) => a.textContent!.trim());
    expect(links).toEqual(['Dashboard', 'Data quality', 'Settings']);
    const menus = [...nav.querySelectorAll('button[aria-haspopup]')].map((b) =>
      b.textContent!.trim(),
    );
    expect(menus).toEqual(['Data', 'Model']);
  });

  it('a menu lists its pages as links to the project paths', () => {
    const nav = primary(render('/p/default/dashboard'));
    (nav.querySelectorAll('button[aria-haspopup]')[0] as HTMLButtonElement).click();
    flushSync();
    const items = [...nav.querySelectorAll('[role="menuitem"]')].map((a) => [
      a.textContent!.trim(),
      a.getAttribute('href'),
    ]);
    expect(items).toEqual([
      ['Ingest', '/p/default/ingest'],
      ['Workspace', '/p/default/workspace'],
      ['Classes', '/p/default/classes'],
      ['Export', '/p/default/export'],
    ]);
  });

  it('marks the current page: its link, or the group and item of a menu page', () => {
    const quality = primary(render('/p/default/quality/audit'));
    expect(quality.querySelector('a[aria-current="page"]')!.textContent).toContain(
      'Data quality',
    );

    const ws = primary(render('/p/default/workspace'));
    expect(ws.querySelector('a[aria-current="page"]')).toBeNull();
    const data = ws.querySelectorAll('button[aria-haspopup]')[0] as HTMLButtonElement;
    expect(data.className).toContain('after:bg-accent');
    data.click();
    flushSync();
    expect(
      ws.querySelector('[role="menuitem"][aria-current="page"]')!.textContent,
    ).toContain('Workspace');
  });

  it('shows the group and section as a breadcrumb, and none where the nav already says it', () => {
    const el = render('/p/default/classes');
    const crumb = el.querySelector('nav[aria-label="Breadcrumb"]')!;
    expect(crumb.textContent!.replace(/\s+/g, ' ').trim()).toBe('Data Classes');
    expect(crumb.querySelector('a[aria-current="page"]')!.getAttribute('href')).toBe(
      '/p/default/classes',
    );
    expect(
      render('/p/default/dashboard').querySelector('nav[aria-label="Breadcrumb"]'),
    ).toBeNull();
  });

  it('the hamburger opens a drawer with every page, grouped, and Escape closes it', () => {
    const el = render('/p/default/models');
    expect(document.querySelector('[data-testid="drawer"]')).toBeNull();
    (el.querySelector('[data-testid="nav-open"]') as HTMLButtonElement).click();
    flushSync();
    const drawer = document.querySelector('[data-testid="drawer"]')!;
    const links = [...drawer.querySelectorAll('nav a')].map((a) => a.textContent!.trim());
    expect(links).toEqual([
      'Dashboard',
      'Ingest',
      'Workspace',
      'Classes',
      'Export',
      'Models',
      'Train',
      'Bake-off',
      'Data quality',
      'Settings',
    ]);
    expect(drawer.querySelector('a[aria-current="page"]')!.textContent).toContain(
      'Models',
    );
    drawer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(document.querySelector('[data-testid="drawer"]')).toBeNull();
  });

  it('names the two navigation landmarks differently', () => {
    const el = render('/p/default/models');
    (el.querySelector('[data-testid="nav-open"]') as HTMLButtonElement).click();
    flushSync();
    const labels = [...document.querySelectorAll('nav[aria-label]')]
      .map((n) => n.getAttribute('aria-label'))
      .filter((l) => l !== 'Breadcrumb');
    expect(labels).toHaveLength(2);
    expect(new Set(labels).size).toBe(2);
  });

  it('closes the drawer when the screen grows to the desktop nav', () => {
    const listeners = new Set<(e: Event) => void>();
    let matches = false;
    vi.stubGlobal('matchMedia', () => ({
      get matches() {
        return matches;
      },
      addEventListener: (_: string, fn: (e: Event) => void) => listeners.add(fn),
      removeEventListener: (_: string, fn: (e: Event) => void) => listeners.delete(fn),
    }));
    try {
      const el = render('/p/default/models');
      (el.querySelector('[data-testid="nav-open"]') as HTMLButtonElement).click();
      flushSync();
      expect(document.querySelector('[data-testid="drawer"]')).not.toBeNull();
      matches = true;
      listeners.forEach((fn) => fn(new Event('change')));
      flushSync();
      expect(document.querySelector('[data-testid="drawer"]')).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
