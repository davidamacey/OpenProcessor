import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { tick } from 'svelte';
import ResourcesMenu from './ResourcesMenu.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { EMPTY_CURATION_SETTINGS } from '$lib/curationSettings';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ResourcesMenu, { target });
  flushSync();
}
const trigger = () =>
  target.querySelector<HTMLButtonElement>('[data-testid="resources-trigger"]')!;
function hrefs(): string[] {
  return [
    ...target.querySelectorAll<HTMLAnchorElement>('[data-testid="resource-link"]'),
  ].map((a) => a.getAttribute('href') ?? '');
}

beforeEach(() => {
  curationSettingsStore.loaded = true; // init() is then a no-op: no request
  curationSettingsStore.settings = EMPTY_CURATION_SETTINGS;
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
});

describe('ResourcesMenu', () => {
  it('is closed until opened and toggles aria-expanded', () => {
    render();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(hrefs()).toEqual([]);
    trigger().click();
    flushSync();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
  });

  it('always lists the same-origin links, with none of the unserved dashboards', () => {
    render();
    trigger().click();
    flushSync();
    expect(hrefs()).toEqual(['/cropwright/', '/docs', '/redoc', '/openapi.json']);
  });

  it('adds only the served monitoring links, as external new-tab anchors', () => {
    curationSettingsStore.settings = {
      ...EMPTY_CURATION_SETTINGS,
      monitoring_links: {
        grafana: 'http://g:3000',
        prometheus: null,
        opensearch_dashboards: 'http://os:5601',
      },
    };
    render();
    trigger().click();
    flushSync();
    expect(hrefs()).toEqual([
      '/cropwright/',
      '/docs',
      '/redoc',
      '/openapi.json',
      'http://g:3000',
      'http://os:5601',
    ]);
    const a = target.querySelector<HTMLAnchorElement>('[data-key="grafana"]')!;
    expect(a.target).toBe('_blank');
    expect(a.rel).toBe('noopener noreferrer');
  });

  it('Escape closes the menu and returns focus to the button', async () => {
    render();
    trigger().click();
    flushSync();
    target
      .querySelector('[data-testid="resources-list"]')!
      .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await tick();
    flushSync();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(trigger());
  });
});
