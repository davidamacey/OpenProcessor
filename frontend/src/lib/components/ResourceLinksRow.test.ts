import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ResourceLinksRow from './ResourceLinksRow.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { EMPTY_CURATION_SETTINGS, type ResourceLink } from '$lib/curationSettings';

const L = (o: Partial<ResourceLink>): ResourceLink => ({
  id: 'grafana',
  label: 'Grafana',
  url: 'https://grafana.example/d/x',
  kind: 'service',
  status: 'configured',
  hint: 'help',
  reachable: null,
  ...o,
});

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

beforeEach(() => {
  curationSettingsStore.loaded = true;
  curationSettingsStore.settings = EMPTY_CURATION_SETTINGS;
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

function render(links: ResourceLink[]): HTMLDivElement {
  curationSettingsStore.settings = { ...EMPTY_CURATION_SETTINGS, resource_links: links };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ResourceLinksRow, { target });
  flushSync();
  return target;
}

describe('ResourceLinksRow', () => {
  it('renders the served service entries in order, verbatim, and skips docs entries', () => {
    const el = render([
      L({ id: 'swagger', label: 'Swagger', url: '/docs', kind: 'docs' }),
      L({}),
      L({ id: 'mlflow', label: 'MLflow', url: 'http://m:5000' }),
    ]);
    const hrefs = [...el.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(hrefs).toEqual(['https://grafana.example/d/x', 'http://m:5000']);
  });

  it('shows a not-configured service as a muted row and a stopped one as not running', () => {
    const el = render([
      L({ url: null, status: 'not_configured', hint: 'set it' }),
      L({
        id: 'prometheus',
        label: 'Prometheus',
        url: 'http://p:9090',
        reachable: false,
      }),
    ]);
    expect(el.querySelector('[data-testid="resource-muted"]')?.textContent).toContain(
      'Grafana: not configured',
    );
    expect(el.querySelector('[data-testid="resource-not-running"]')).not.toBeNull();
  });

  it('shows nothing when no service entry is served', () => {
    const el = render([]);
    expect(el.querySelector('[data-testid="resource-row"]')).toBeNull();
    expect(el.textContent).not.toContain('Dashboards');
  });
});
