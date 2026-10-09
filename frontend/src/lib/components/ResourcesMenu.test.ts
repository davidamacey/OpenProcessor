import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { tick } from 'svelte';
import ResourcesMenu from './ResourcesMenu.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { EMPTY_CURATION_SETTINGS, type ResourceLink } from '$lib/curationSettings';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

const L = (o: Partial<ResourceLink>): ResourceLink => ({
  id: 'grafana',
  label: 'Grafana',
  url: 'http://g:3000',
  kind: 'service',
  status: 'configured',
  hint: 'help text',
  reachable: null,
  ...o,
});

function serve(links: ResourceLink[]): void {
  curationSettingsStore.settings = { ...EMPTY_CURATION_SETTINGS, resource_links: links };
}

function render(): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ResourcesMenu, { target });
  flushSync();
}
const trigger = () =>
  target.querySelector<HTMLButtonElement>('[data-testid="resources-trigger"]')!;
function open(): void {
  trigger().click();
  flushSync();
}
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
    open();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
  });

  it('with nothing served lists only Documentation', () => {
    render();
    open();
    expect(hrefs()).toEqual([
      '/OpenProcessor/docs/cropwright/getting-started/introduction',
    ]);
  });

  it('lists exactly the served entries in the served order after Documentation', () => {
    serve([
      L({ id: 'swagger', label: 'Swagger', url: '/docs', kind: 'docs' }),
      L({}),
      L({ id: 'mlflow', label: 'MLflow', url: 'http://m:5000' }),
    ]);
    render();
    open();
    expect(hrefs()).toEqual([
      '/OpenProcessor/docs/cropwright/getting-started/introduction',
      '/docs',
      'http://g:3000',
      'http://m:5000',
    ]);
    for (const a of target.querySelectorAll<HTMLAnchorElement>(
      '[data-testid="resource-link"]',
    )) {
      expect(a.target).toBe('_blank');
      expect(a.rel).toBe('noopener noreferrer');
    }
  });

  it('a not_configured entry is a muted row with the hint as title and no anchor', () => {
    serve([L({ url: null, status: 'not_configured', hint: 'set OP_GRAFANA_URL' })]);
    render();
    open();
    expect(hrefs()).toEqual([
      '/OpenProcessor/docs/cropwright/getting-started/introduction',
    ]);
    const row = target.querySelector<HTMLElement>('[data-testid="resource-muted"]')!;
    expect(row.textContent).toContain('Grafana: not configured');
    expect(row.getAttribute('title')).toBe('set OP_GRAFANA_URL');
    expect(row.querySelector('a')).toBeNull();
  });

  it('reachable false shows a not-running note; null and true show nothing', () => {
    serve([
      L({ id: 'a', label: 'A', reachable: false }),
      L({ id: 'b', label: 'B', reachable: true }),
      L({ id: 'c', label: 'C', reachable: null }),
    ]);
    render();
    open();
    const notes = target.querySelectorAll('[data-testid="resource-not-running"]');
    expect(notes).toHaveLength(1);
    expect(notes[0]!.closest('[data-key]')!.getAttribute('data-key')).toBe('a');
  });

  it('an unsafe served url renders no anchor', () => {
    serve([L({ url: 'javascript:alert(1)' })]);
    render();
    open();
    expect(hrefs()).toEqual([
      '/OpenProcessor/docs/cropwright/getting-started/introduction',
    ]);
    expect(target.innerHTML).not.toContain('javascript:');
  });

  it('Escape closes the menu and returns focus to the button', async () => {
    render();
    open();
    target
      .querySelector('[data-testid="resources-list"]')!
      .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await tick();
    flushSync();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(trigger());
  });
});
